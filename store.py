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
LONGSHOT_VERSION = "1.3.0"
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
