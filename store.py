"""Restart-stable fingerprints and the on-disk segment store.

Recipe fingerprints (Addendum A.1)
    Long Shot's RAM memory identifies the model by object identity, which no
    restart survives. For disk, the model and sampler are identified by their
    *recipe* instead: starting from Long Shot's own entry in the API prompt,
    every node upstream of model / clip / vae / audio_vae / sampler / sigmas
    is hashed by class and literal inputs (file names, strengths, shifts,
    steps…), plus the slot of each link. Node ids are never hashed, so a
    renumbered graph gives the same digest. A model file's size and modified
    time go in with its name, so a different file saved under the same name
    is a change.

Segment store (Addendum A.2)
    output/longshot/<cache_name>/segments/<fingerprint>.safetensors, holding
    the sampled window's video and audio latents in fp16 with metadata.
    Written to a .tmp file and renamed, so a crash never leaves a file that
    looks valid; stray .tmp files are ignored and removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

import torch

logger = logging.getLogger("MiniMaxH3LongShot")

STORE_FORMAT = 1
LONGSHOT_VERSION = "1.5.1"
MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pt2", ".pth", ".bin", ".gguf", ".pkl")
RECIPE_INPUTS = {"model": ("model", "clip", "vae", "audio_vae"), "sampler": ("sampler", "sigmas")}


# ---------------------------------------------------------------------------
# Recipe fingerprints
# ---------------------------------------------------------------------------

class NoRecipe(Exception):
    """The prompt can't describe this node's inputs (missing or ephemeral nodes)."""


def _hash(obj) -> str:
    return hashlib.blake2b(json.dumps(obj, sort_keys=True, default=repr).encode(),
                           digest_size=16).hexdigest()


def _is_link(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (str, int))
            and isinstance(v[1], int) and not isinstance(v[1], bool))


def default_file_stat(name: str):
    """(size, whole-second mtime) of a model file ComfyUI knows by this name, or None."""
    try:
        import folder_paths
    except ImportError:
        return None
    for folder in list(getattr(folder_paths, "folder_names_and_paths", {})):
        try:
            path = folder_paths.get_full_path(folder, name)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            st = os.stat(path)
            return [st.st_size, int(st.st_mtime)]
    return None


def _literal(value, file_stat):
    if isinstance(value, str) and value.lower().endswith(MODEL_EXTENSIONS):
        return {"file": value.replace("\\", "/"), "stat": file_stat(value)}
    return value


def recipe_components(prompt, unique_id, file_stat=default_file_stat):
    """{"model": digest, "sampler": digest} from the API prompt, or None when
    the prompt doesn't contain this node (called outside the executor)."""
    if not isinstance(prompt, dict) or unique_id is None:
        return None
    me = prompt.get(str(unique_id))
    if not isinstance(me, dict):
        return None
    memo, stats = {}, {}

    def stat(name):
        if name not in stats:
            stats[name] = file_stat(name)
        return stats[name]

    def node_digest(nid, stack):
        nid = str(nid)
        if nid in memo:
            return memo[nid]
        node = prompt.get(nid)
        if not isinstance(node, dict) or nid in stack:
            raise NoRecipe(nid)
        parts = [node.get("class_type")]
        for name in sorted(node.get("inputs") or {}):
            v = node["inputs"][name]
            if _is_link(v):
                parts.append([name, "link", node_digest(v[0], stack | {nid}), v[1]])
            else:
                parts.append([name, "value", _literal(v, stat)])
        memo[nid] = _hash(parts)
        return memo[nid]

    def input_digest(name):
        v = (me.get("inputs") or {}).get(name)
        if v is None:
            return None
        if _is_link(v):
            return [node_digest(v[0], frozenset({str(unique_id)})), v[1]]
        return ["value", _literal(v, stat)]

    try:
        return {key: _hash([[n, input_digest(n)] for n in names])
                for key, names in RECIPE_INPUTS.items()}
    except NoRecipe as missing:
        logger.info("MiniMax H3 Long Shot: node %s isn't in the prompt; saved segments are "
                    "off for this run", missing)
        return None


def linked_inputs(prompt, unique_id):
    """Names of this node's inputs that are wired to another node."""
    me = (prompt or {}).get(str(unique_id)) if isinstance(prompt, dict) else None
    if not isinstance(me, dict):
        return None
    return {k for k, v in (me.get("inputs") or {}).items() if _is_link(v)}


# ---------------------------------------------------------------------------
# Segment store
# ---------------------------------------------------------------------------

def safe_cache_name(name) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "")).strip("._")
    return out[:96] or "default"


def default_root():
    import folder_paths
    return os.path.join(folder_paths.get_output_directory(), "longshot")


class SegmentStore:
    """One project's saved segments: <root>/<cache_name>/segments/<fp>.safetensors."""

    def __init__(self, cache_name="default", root=None):
        self.name = safe_cache_name(cache_name)
        self.folder = os.path.join(root or default_root(), self.name, "segments")
        self._cleaned = False

    def path(self, fp):
        if not re.fullmatch(r"[0-9a-f]{8,64}", fp):
            raise ValueError("not a fingerprint")
        return os.path.join(self.folder, fp + ".safetensors")

    def clean_tmp(self):
        """Remove leftovers of interrupted writes. They never look like segments."""
        if self._cleaned or not os.path.isdir(self.folder):
            return
        self._cleaned = True
        for f in os.listdir(self.folder):
            if f.endswith(".tmp"):
                try:
                    os.remove(os.path.join(self.folder, f))
                except OSError:
                    pass

    def has(self, fp):
        return os.path.isfile(self.path(fp))

    def save(self, fp, video, audio, meta=None):
        from safetensors.torch import save_file
        os.makedirs(self.folder, exist_ok=True)
        final = self.path(fp)
        tmp = final + ".tmp"
        tensors = {"video": video.detach().to("cpu", torch.float16).contiguous(),
                   "audio": audio.detach().to("cpu", torch.float16).contiguous()}
        metadata = {"format": str(STORE_FORMAT), "longshot_version": LONGSHOT_VERSION,
                    "dtype": str(video.dtype).replace("torch.", ""),
                    "audio_dtype": str(audio.dtype).replace("torch.", ""),
                    "video_shape": json.dumps(list(video.shape)),
                    "audio_shape": json.dumps(list(audio.shape)),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        metadata.update({k: str(v) for k, v in (meta or {}).items()})
        try:
            save_file(tensors, tmp, metadata=metadata)
            os.replace(tmp, final)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return final

    def load(self, fp, video_shape=None, audio_shape=None):
        """(video, audio) in the dtype they were rendered in, or None if the
        file is missing, unreadable or not the expected shape."""
        path = self.path(fp)
        if not os.path.isfile(path):
            return None
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
                if meta.get("format") != str(STORE_FORMAT):
                    return None
                video, audio = f.get_tensor("video"), f.get_tensor("audio")
            if video_shape is not None and tuple(video.shape) != tuple(video_shape):
                raise ValueError(f"video {tuple(video.shape)} != {tuple(video_shape)}")
            if audio_shape is not None and tuple(audio.shape) != tuple(audio_shape):
                raise ValueError(f"audio {tuple(audio.shape)} != {tuple(audio_shape)}")
            vdt = getattr(torch, meta.get("dtype", "float32"), torch.float32)
            adt = getattr(torch, meta.get("audio_dtype", "float32"), torch.float32)
            return video.to(vdt), audio.to(adt)
        except Exception as err:
            logger.warning("MiniMax H3 Long Shot: ignoring saved segment %s (%s)",
                           os.path.basename(path), err)
            return None

    def stats(self):
        if not os.path.isdir(self.folder):
            return {"files": 0, "bytes": 0}
        files = [f for f in os.listdir(self.folder) if f.endswith(".safetensors")]
        return {"files": len(files),
                "bytes": sum(os.path.getsize(os.path.join(self.folder, f)) for f in files)}


# ---------------------------------------------------------------------------
# Takes (round 2)
#
# output/longshot/<cache_name>/takes/<shot_id>__<seed>__<fp8>.safetensors
#
# Every finished piece is kept as a take: the full sampled window (video and
# audio, fp16) plus metadata describing how it joins its neighbours. A locked
# Shot loads its take by name instead of sampling, and its neighbours pin to
# it. The metadata is small and read without loading the tensors, so a plan
# can be made from it before any model loads.
# ---------------------------------------------------------------------------

TAKE_FORMAT = 1
_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
TAKE_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}__\d{1,20}__[0-9a-f]{8}\.safetensors")


def safe_shot_id(shot_id) -> str:
    out = _ID_RE.sub("_", str(shot_id or "")).strip("_")
    return out[:64] or "shot"


def take_shot(name) -> str:
    """The (safe) shot id a take file belongs to. Ids may contain '__', so split
    from the right: <id>__<seed>__<fp8>.safetensors."""
    return str(name).rsplit("__", 2)[0]


def slice_hash(video_slice) -> str:
    """Identity of a shared-zone slice: its video tokens as fp16. Audio is
    left out on purpose: the same zone can hold one token more or less of
    audio depending on where it sits on the 40 Hz grid."""
    t = video_slice.detach().to("cpu", torch.float16).contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{tuple(t.shape)}".encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


class TakeStore:
    """One project's takes."""

    def __init__(self, cache_name="default", root=None):
        self.name = safe_cache_name(cache_name)
        self.folder = os.path.join(root or default_root(), self.name, "takes")
        self._meta = {}          # name -> (mtime_ns, size, meta)
        self._cleaned = False

    @staticmethod
    def take_name(shot_id, seed, fp) -> str:
        return f"{safe_shot_id(shot_id)}__{int(seed) if seed is not None else 0}__{fp[:8]}.safetensors"

    def path(self, name):
        if not isinstance(name, str) or not TAKE_NAME_RE.fullmatch(name):
            raise ValueError(f"not a take file name: {name!r}")
        return os.path.join(self.folder, name)

    def exists(self, name):
        try:
            return os.path.isfile(self.path(name))
        except ValueError:
            return False

    def clean_tmp(self):
        if self._cleaned or not os.path.isdir(self.folder):
            return
        self._cleaned = True
        for f in os.listdir(self.folder):
            if f.endswith(".tmp"):
                try:
                    os.remove(os.path.join(self.folder, f))
                except OSError:
                    pass

    def meta(self, name):
        """The take's metadata (dict, with JSON fields decoded), or None."""
        try:
            path = self.path(name)
            st = os.stat(path)
        except (ValueError, OSError):
            return None
        hit = self._meta.get(name)
        if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            return dict(hit[2])
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                raw = f.metadata() or {}
        except Exception as err:
            logger.warning("MiniMax H3 Long Shot: unreadable take %s (%s)", name, err)
            return None
        if raw.get("take_format") != str(TAKE_FORMAT):
            return None
        meta = {}
        for k, v in raw.items():
            if k in ("window_frames", "overlap", "width", "height", "seed", "audio_tokens"):
                try:
                    meta[k] = int(v)
                except ValueError:
                    meta[k] = None
            else:
                meta[k] = v
        meta["name"] = name
        meta["bytes"] = st.st_size
        self._meta[name] = (st.st_mtime_ns, st.st_size, meta)
        return dict(meta)

    def save(self, name, video, audio, meta):
        from safetensors.torch import save_file
        os.makedirs(self.folder, exist_ok=True)
        final = self.path(name)
        tmp = final + ".tmp"
        tensors = {"video": video.detach().to("cpu", torch.float16).contiguous(),
                   "audio": audio.detach().to("cpu", torch.float16).contiguous()}
        md = {"take_format": str(TAKE_FORMAT), "longshot_version": LONGSHOT_VERSION,
              "dtype": str(video.dtype).replace("torch.", ""),
              "audio_dtype": str(audio.dtype).replace("torch.", ""),
              "audio_tokens": str(audio.shape[-1]),
              "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        md.update({k: "" if v is None else str(v) for k, v in (meta or {}).items()})
        try:
            save_file(tensors, tmp, metadata=md)
            os.replace(tmp, final)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        self._meta.pop(name, None)
        return final

    def load(self, name):
        """(video, audio, meta) in the dtypes they were rendered in. Raises
        FileNotFoundError when the take is gone, ValueError when unreadable."""
        path = self.path(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        meta = self.meta(name)
        if meta is None:
            raise ValueError(f"{name} is not a readable take")
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            video, audio = f.get_tensor("video"), f.get_tensor("audio")
        vdt = getattr(torch, meta.get("dtype") or "float32", torch.float32)
        adt = getattr(torch, meta.get("audio_dtype") or "float32", torch.float32)
        return video.to(vdt), audio.to(adt), meta

    def find(self, shot_id, seed, fp):
        """The take saved for exactly this fingerprint, if any."""
        name = self.take_name(shot_id, seed, fp)
        meta = self.meta(name)
        return name if meta is not None and meta.get("fp") == fp else None

    def list(self, shot_id=None):
        """Metadata of every take (or one Shot's), oldest first."""
        if not os.path.isdir(self.folder):
            return []
        want = safe_shot_id(shot_id) if shot_id is not None else None
        out = []
        for f in os.listdir(self.folder):
            if not TAKE_NAME_RE.fullmatch(f) or (want is not None and take_shot(f) != want):
                continue
            m = self.meta(f)
            if m is not None:
                out.append(m)
        out.sort(key=lambda m: (m.get("created") or "", m["name"]))
        return out

    def delete(self, name):
        path = self.path(name)
        self._meta.pop(name, None)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def stats(self):
        if not os.path.isdir(self.folder):
            return {"files": 0, "bytes": 0}
        files = [f for f in os.listdir(self.folder) if TAKE_NAME_RE.fullmatch(f)]
        return {"files": len(files),
                "bytes": sum(os.path.getsize(os.path.join(self.folder, f)) for f in files)}
