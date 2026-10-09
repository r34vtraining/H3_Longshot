"""MiniMax H3 Long Shot — a chain of Shot nodes rendered as one continuous shot,
with optional lip-sync to a song."""

from __future__ import annotations

import inspect
import logging
import math
from collections import OrderedDict

import torch

from . import planner
from . import store as disk

logger = logging.getLogger("MiniMaxH3LongShot")

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STEREO = 2


# ---------------------------------------------------------------------------
# ComfyUI internals, imported lazily so the pack loads (and reports a clear
# error) on builds without native MiniMax H3.
# ---------------------------------------------------------------------------

def _native():
    try:
        import comfy_extras.nodes_minimax_h3 as h3
        import comfy_extras.nodes_custom_sampler as cs
    except ImportError as err:
        raise RuntimeError("This ComfyUI build has no native MiniMax H3 support. Update ComfyUI.") from err
    return h3, cs


def _require_arbitrary_guides():
    from comfy.ldm.minimax.model import PackedLayout
    if "frame_count" in inspect.signature(PackedLayout.__init__).parameters:
        raise RuntimeError(
            "This ComfyUI build predates native MiniMax H3 guides at arbitrary frames. "
            "Update ComfyUI to use MiniMax H3 Long Shot."
        )


def _unwrap(result):
    return result.args if hasattr(result, "args") else tuple(result)


def _device():
    import comfy.model_management
    return comfy.model_management.intermediate_device()


def _set_keyframes(positive, keyframes):
    import node_helpers
    keyframes = sorted(keyframes, key=lambda k: float(k["resolved_frame_index"]))
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})


def _keyframes(positive):
    return [dict(k) for k in positive[0][1].get("minimax_keyframes", [])]


# ---------------------------------------------------------------------------
# One segment's pieces
# ---------------------------------------------------------------------------

def _empty_window(width, height, seg, like=None):
    """Zero AV target. Audio length comes from global boundaries in the plan,
    not from the window alone — that's what keeps a long chain in sync."""
    from comfy.nested_tensor import NestedTensor
    dev = like.device if like is not None else _device()
    dtype = like.dtype if like is not None else torch.float32
    video = torch.zeros([1, VIDEO_CHANNELS, planner.video_tokens(seg.window_frames),
                         height // 16, width // 16], device=dev, dtype=dtype)
    audio = torch.zeros([1, AUDIO_CHANNELS, AUDIO_STEREO, seg.window_audio_tokens],
                        device=dev, dtype=dtype)
    return {"samples": NestedTensor((video, audio))}


# ---------------------------------------------------------------------------
# RefMods, presented the way H3 RefMod Text Encode presents them
# ---------------------------------------------------------------------------

_LABEL_NAMES = {"image": "Picture", "video": "Video", "audio": "Audio"}
REFMOD_FPS = 24.0   # saved RefMods carry no frame rate; H3 runs at 24


def _active_refmods(refmods):
    """(mod, strength) rows with strength > 0, in loader order."""
    active = []
    for mod, strength in refmods or []:
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"RefMod '{getattr(mod, 'name', '?')}' has strength {strength}; "
                             f"it must be between 0 and 1.")
        if strength > 0:
            active.append((mod, strength))
    return active


def _native_label_counts(refs):
    """How many <Picture>/<Video>/<Audio> labels the native references take.
    A video soundtrack only counts when its same-numbered video is connected."""
    videos = refs["ref_videos"]
    soundtracks = [n for n in refs["ref_video_audios"]
                   if "ref_video_" + n.rsplit("_", 1)[-1] in videos]
    return {"image": len(refs["ref_images"]), "video": len(videos),
            "audio": len(soundtracks) + len(refs["ref_audios"])}


def native_labels(refs):
    """The label each live reference input gets, in the native node's order:
    images, then each video (its soundtrack's <Audio> label first), then
    standalone audio."""
    out = [(f"<Picture {i}>", name) for i, name in enumerate(refs["ref_images"], 1)]
    audio = 0
    for v, name in enumerate(refs["ref_videos"], 1):
        track = "ref_video_audio_" + name.rsplit("_", 1)[-1]
        if track in refs["ref_video_audios"]:
            audio += 1
            out.append((f"<Audio {audio}>", track))
        out.append((f"<Video {v}>", name))
    for name in refs["ref_audios"]:
        audio += 1
        out.append((f"<Audio {audio}>", name))
    return out


def refmod_labels(refs, refmods):
    """Each active RefMod's label, numbered after the native references."""
    counts = _native_label_counts(refs)
    labels = []
    for mod, _strength in _active_refmods(refmods):
        kind = mod.kind
        if kind not in _LABEL_NAMES:
            raise ValueError(f"RefMod '{mod.name}' has unknown kind {kind!r}.")
        counts[kind] += 1
        labels.append((f"<{_LABEL_NAMES[kind]} {counts[kind]}>", mod.name, kind))
    return labels


def _refmod_presentation(refmods, vae):
    """Tokenizer items and model blocks for the active RefMods, in matching
    order — the same thing H3 RefMod Text Encode builds. Visual RefMods are
    decoded from the same (strength-weakened) latent the model receives, so
    the text encoder sees exactly what the model attends to."""
    items, blocks = [], []
    for mod, strength in _active_refmods(refmods):
        block = mod.ref_block(strength)
        if block is None:
            continue
        block["refmod"] = True          # lets H3 RefMod Step Curve find it
        kind = block["kind"]
        item = {"type": kind}
        if kind != "audio":
            if vae is None:
                raise ValueError("Visual RefMods need the H3 video VAE connected.")
            pixels = vae.decode(block["latent"])
            if pixels.ndim == 5 and pixels.shape[0] == 1:
                pixels = pixels[0]
            if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.shape[0] < 1:
                raise ValueError(f"RefMod '{mod.name}' decoded to an unexpected shape "
                                 f"{tuple(pixels.shape)}.")
            if kind == "image":
                item["data"] = pixels[:1].cpu().clone()
            else:
                # the text encoder sees video at 2 fps, as the native node presents it
                times = [i / 2 for i in range(math.ceil(pixels.shape[0] * 2 / REFMOD_FPS))]
                idx = [min(round(t * REFMOD_FPS), pixels.shape[0] - 1) for t in times]
                item["data"] = pixels[idx].cpu()
                item["timestamps"] = times
            del pixels
        items.append(item)
        blocks.append(block)
    return items, blocks


class _PresentingClip:
    """The real CLIP, with extra reference items appended to every tokenize.

    The native Reference to Video node presents its own references to the text
    encoder; this adds the RefMods after them in the same call, so the native
    node keeps doing everything else exactly as it does natively."""

    def __init__(self, clip, extra_items):
        self._clip = clip
        self._extra = list(extra_items)

    def tokenize(self, text, *args, minimax_ref_items=None, **kwargs):
        items = list(minimax_ref_items or []) + self._extra
        return self._clip.tokenize(text, *args, minimax_ref_items=items, **kwargs)

    def __getattr__(self, name):
        return getattr(self._clip, name)


class _RefEncoder:
    """Per-segment conditioning through ComfyUI's own H3 nodes.

    References go to the native Reference to Video node exactly as connected,
    so <Picture N> / <Video N> / <Audio N> numbering matches the native node.
    RefMods are presented right after them, taking the next labels.

    The text encoder has to run per segment (each has its own prompt), but the
    VAE encodes of the references don't: they depend only on the references and
    the window length. Those blocks are encoded once per window length and
    reused, which saves re-encoding every reference video on every segment.
    RefMods are decoded for the text encoder once for the whole run."""

    def __init__(self, clip, vae, audio_vae, width, height, refs, ref_image_size, refmods=None):
        self.clip, self.vae, self.audio_vae = clip, vae, audio_vae
        self.width, self.height = width, height
        self.refs = refs
        self.ref_image_size = ref_image_size
        self._blocks = {}   # window_frames -> native minimax_refs blocks
        self._mod_items, self._mod_blocks = (_refmod_presentation(refmods, vae)
                                             if refmods else ([], []))

    @property
    def has_refs(self):
        return any(self.refs.values()) or bool(self._mod_blocks)

    def encode(self, prompt, frames, first_frame=None, last_frame=None):
        h3, _ = _native()
        if not self.has_refs:
            out = h3.MiniMaxH3ImageToVideo.execute(
                self.clip, self.vae, prompt, self.width, self.height, frames,
                first_frame=first_frame, last_frame=last_frame)
            return _unwrap(out)[0]

        import node_helpers
        clip = _PresentingClip(self.clip, self._mod_items) if self._mod_items else self.clip
        cached = self._blocks.get(frames)
        out = h3.MiniMaxH3ReferenceToVideo.execute(
            clip, prompt, self.width, self.height, frames,
            ref_image_size=self.ref_image_size,
            vae=None if cached is not None else self.vae,
            audio_vae=None if cached is not None else self.audio_vae,
            **self.refs)
        positive = _unwrap(out)[0]
        if cached is None:
            cached = self._blocks[frames] = list(positive[0][1].get("minimax_refs", []) or [])
        return node_helpers.conditioning_set_values(
            positive, {"minimax_refs": list(cached) + list(self._mod_blocks)})


def _song_slice(song, seg):
    """This window's audio from the pre-encoded song, at exact 40 Hz boundaries."""
    a0, a1 = seg.window_audio_span
    return song["latent"][..., a0:a1].clone()


def _pin_audio_at_zero(positive, audio_latent):
    """Segment 1 with a song: pin the song to the window's opening. Attaches to
    an existing frame-0 image guide if there is one, as native AddGuide does."""
    kfs = _keyframes(positive)
    for k in kfs:
        if float(k["resolved_frame_index"]) == 0 and k.get("audio_latent") is None:
            k["audio_latent"] = audio_latent
            break
    else:
        kfs.append({"resolved_frame_index": 0, "audio_latent": audio_latent})
    return _set_keyframes(positive, kfs)


def _add_tail_guide(positive, cumulative, seg, song_audio=None):
    """Pin the hidden overlap to the tail of everything generated so far.

    The video tail always comes from the latent itself (no VAE round trip).
    Audio comes from the song when one is connected — the song is the source
    of truth, and a second audio guide on the same frames would compete —
    otherwise from the generated audio tail."""
    video, audio = cumulative["samples"].tensors
    guide = {
        "resolved_frame_index": 0,
        "latent": video[:, :, -seg.overlap_video_tokens:].clone(),
        "audio_latent": (song_audio if song_audio is not None
                         else audio[..., -seg.overlap_audio_tokens:].clone()),
    }
    kfs = _keyframes(positive)
    for k in kfs:
        if 0 <= float(k["resolved_frame_index"]) < seg.overlap_frames:
            raise ValueError(
                f"Segment {seg.index} already has a guide inside its hidden overlap, "
                f"which would fight the continuation.")
    kfs.append(guide)
    return _set_keyframes(positive, kfs)


def _extra_ref_blocks(extra_refs):
    """Reference blocks carried by an external conditioning (e.g. Apply H3 RefMod)."""
    if not extra_refs:
        return []
    blocks = []
    for entry in extra_refs:
        entry_blocks = entry[1].get("minimax_refs", []) or []
        blocks.extend(entry_blocks)
        if entry_blocks and not entry[1].get(_CARRIER_KEY):
            logger.warning(
                "MiniMax H3 Long Shot: extra_refs came from a node other than MiniMax H3 RefMod "
                "Carrier. Every reference block on it is added, including any references "
                "connected to the node that made it — if those are also connected to Long "
                "Shot, they'll be counted twice.")
    return blocks


def _add_ref_blocks(positive, blocks):
    if not blocks:
        return positive
    import node_helpers
    existing = list(positive[0][1].get("minimax_refs", []) or [])
    return node_helpers.conditioning_set_values(positive, {"minimax_refs": existing + list(blocks)})


def _sample_window(model, noise, sampler, sigmas, positive, latent):
    """One generation through ComfyUI's own guider and advanced sampler, so
    progress bars, previews, and interrupts behave natively."""
    _, cs = _native()
    guider = cs.Guider_Basic(model)
    guider.set_conds(positive)
    return _unwrap(cs.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent))[0]


def _is_reseedable(noise):
    """Any noise source that draws from a seed — ComfyUI's RandomNoise or a
    custom-node equivalent. DisableNoise (Noise_EmptyNoise) also carries a
    seed attribute but ignores it; re-seeding it is pointless, so it's left
    alone."""
    if noise is None or type(noise).__name__ == "Noise_EmptyNoise":
        return False
    seed = getattr(noise, "seed", None)
    return isinstance(seed, int) and not isinstance(seed, bool) and callable(
        getattr(noise, "generate_noise", None))


def segment_seeds(noise, seed_mode, shots):
    """Each segment's seed, and whether it came from the Shot's own seed.

    A Shot seed of -1 follows Long Shot: base + N - 1 with 'increment', base
    with 'same'. Noise that doesn't draw from a seed (DisableNoise) keeps its
    own and ignores Shot seeds."""
    base = getattr(noise, "seed", None)
    reseed = _is_reseedable(noise)
    seeds, own = [], []
    for i, shot in enumerate(shots, 1):
        shot_seed = int(shot.get("seed", -1) if shot.get("seed") is not None else -1)
        if reseed and shot_seed >= 0:
            seeds.append(shot_seed)
            own.append(True)
        elif reseed and seed_mode == "increment":
            seeds.append(base + i - 1)
            own.append(False)
        else:
            seeds.append(base)
            own.append(False)
    return seeds, own


def _segment_noise(noise, seed):
    """A copy of the noise source drawing from this segment's seed. Copying
    keeps whatever else a custom noise node carries."""
    if not _is_reseedable(noise) or noise.seed == seed:
        return noise
    import copy
    out = copy.copy(noise)
    out.seed = seed
    return out


# ---------------------------------------------------------------------------
# Segment reuse
#
# Segment N's result depends only on its own inputs and on segments 1..N-1, so
# each finished segment is remembered under a fingerprint of exactly that: its
# own inputs plus the previous segment's fingerprint. Nothing about later
# segments goes in, so adding a Shot to the end keeps every earlier segment.
# ---------------------------------------------------------------------------

_SEGMENT_CACHE = OrderedDict()   # fingerprint -> sampled window (on CPU)
SEGMENT_CACHE_MAX = 64
_LAST_RUN = []                   # per segment: {component: digest} of the last real run

# Checked in this order when explaining why a segment renders.
_REASONS = (("model", "model, CLIP or VAE changed"),
            ("sampler", "sampler, sigmas or size changed"),
            ("prompt", "prompt changed"),
            ("seed", "seed changed"),
            ("length", "length changed"),
            ("references", "references changed"),
            ("song", "song changed"),
            ("frames", "first/last frame changed"))


_FULL_HASH_ELEMENTS = 16 * 1024 * 1024


def _digest(obj, _depth=0):
    """A stable hash of tensors, containers and plain values. Anything else
    hashes by identity, which is what makes a reloaded model a change."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)

    def feed(o, depth):
        if isinstance(o, torch.Tensor):
            t = o.detach()
            h.update(f"T{t.dtype}{tuple(t.shape)}".encode())
            if t.numel() > _FULL_HASH_ELEMENTS:
                # A long reference video can be gigabytes; hash an even spread of
                # it plus its exact sum instead of every byte.
                flat = t.reshape(-1)
                step = flat.numel() // _FULL_HASH_ELEMENTS + 1
                h.update(repr(float(flat.double().sum())).encode())
                t = flat[::step]
            t = t.contiguous().cpu()
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        elif hasattr(o, "tensors") and isinstance(getattr(o, "tensors"), (list, tuple)):
            h.update(b"N")
            for t in o.tensors:
                feed(t, depth)
        elif isinstance(o, dict):
            h.update(b"{")
            for k in sorted(o, key=str):
                h.update(repr(k).encode())
                feed(o[k], depth)
            h.update(b"}")
        elif isinstance(o, (list, tuple)):
            h.update(b"[")
            for x in o:
                feed(x, depth)
            h.update(b"]")
        elif o is None or isinstance(o, (bool, int, float, str, bytes)):
            h.update(repr(o).encode())
        elif callable(o) and hasattr(o, "__qualname__"):
            h.update(f"F{getattr(o, '__module__', '')}.{o.__qualname__}".encode())
        elif depth < 2 and hasattr(o, "__dict__"):
            h.update(f"O{type(o).__qualname__}".encode())
            feed({k: v for k, v in vars(o).items() if not k.startswith("_")}, depth + 1)
        else:
            h.update(f"I{type(o).__qualname__}:{id(o)}".encode())

    feed(obj, _depth)
    return h.hexdigest()


def _identity(obj):
    """Who an object is, not what it holds: a new LoRA or a reloaded model is a
    new object, and patches_uuid changes when a patcher is re-patched."""
    if obj is None:
        return None
    patcher = getattr(obj, "patcher", obj)
    return (type(obj).__qualname__, id(obj), str(getattr(patcher, "patches_uuid", "")))


def _mods_digest(refmods):
    return [(getattr(m, "name", "?"), getattr(m, "kind", "?"), float(st),
             _digest(getattr(m, "latent", None)))
            for m, st in (refmods or [])]


def shared_components(*, model, clip, vae, audio_vae, sampler, sigmas, width, height,
                      refs, ref_image_size, refmods, extra_blocks, first_frame, last_frame,
                      recipe=None):
    """The parts every segment shares, hashed once per run.

    With a recipe (the run came through ComfyUI's executor), the model and
    sampler are identified by how they were built, which survives restarts
    and lets segments be saved to disk. Without one, by object identity."""
    if recipe is not None:
        model_part = _digest(["recipe", recipe["model"]])
        sampler_part = _digest(["recipe", recipe["sampler"], width, height])
    else:
        model_part = _digest([_identity(model), _identity(clip), _identity(vae),
                              _identity(audio_vae)])
        sampler_part = _digest([sampler, sigmas, width, height])
    return {
        "model": model_part,
        "sampler": sampler_part,
        "references": _digest([refs, ref_image_size, _mods_digest(refmods), extra_blocks]),
        "first": _digest(first_frame),
        "last": _digest(last_frame),
    }


def segment_components(sp, seed, noise, shared, song):
    """Everything that decides one segment's result, apart from the segments
    before it."""
    seg = sp.segment
    return {
        "model": shared["model"],
        "sampler": shared["sampler"],
        "prompt": _digest(sp.prompt),
        "seed": _digest([type(noise).__qualname__, seed]),
        "length": _digest([seg.window_start, seg.window_frames, seg.overlap_frames,
                           seg.new_frames]),
        "references": shared["references"],
        "song": _digest(_song_slice(song, seg) if song is not None else None),
        "frames": _digest([shared["first"] if sp.first_frame else None,
                           shared["last"] if sp.last_frame else None]),
    }


def chain_fingerprints(components):
    fps, prev = [], ""
    for comp in components:
        prev = _digest([prev, [comp[k] for k, _ in _REASONS]])
        fps.append(prev)
    return fps


def segment_statuses(components, fingerprints, reuse, store=None):
    """'reused (memory)', 'reused (disk)' or 'will render — <why>' per segment."""
    out = []
    for i, (comp, fp) in enumerate(zip(components, fingerprints)):
        if reuse and fp in _SEGMENT_CACHE:
            out.append("reused (memory)")
            continue
        if reuse and store is not None and store.has(fp):
            out.append("reused (disk)")
            continue
        if not reuse:
            why = "reuse off"
        elif i >= len(_LAST_RUN):
            why = "first run" if not _LAST_RUN else "new segment"
        else:
            changed = [label for key, label in _REASONS if comp[key] != _LAST_RUN[i][key]]
            why = changed[0] if changed else (
                "follows a changed segment" if i else "not in memory")
        out.append(f"will render — {why}")
    return out


def _remember(fp, sampled):
    from comfy.nested_tensor import NestedTensor
    v, a = sampled["samples"].tensors
    _SEGMENT_CACHE[fp] = {"samples": NestedTensor((v.detach().cpu().clone(),
                                                   a.detach().cpu().clone()))}
    _SEGMENT_CACHE.move_to_end(fp)
    while len(_SEGMENT_CACHE) > SEGMENT_CACHE_MAX:
        _SEGMENT_CACHE.popitem(last=False)


def _recall(fp):
    from comfy.nested_tensor import NestedTensor
    hit = _SEGMENT_CACHE.get(fp)
    if hit is None:
        return None
    _SEGMENT_CACHE.move_to_end(fp)
    v, a = hit["samples"].tensors
    return {"samples": NestedTensor((v.clone(), a.clone()))}


def clear_segment_cache():
    _SEGMENT_CACHE.clear()
    _LAST_RUN.clear()
    _LAST_TIMELINE.clear()


def _append(cumulative, sampled, seg):
    from comfy.nested_tensor import NestedTensor
    prev_v, prev_a = cumulative["samples"].tensors
    samp_v, samp_a = sampled["samples"].tensors
    new_v = samp_v[:, :, seg.overlap_video_tokens:].to(prev_v)
    new_a = samp_a[..., seg.overlap_audio_tokens:].to(prev_a)
    if new_v.shape[2] != seg.new_video_tokens or new_a.shape[-1] != seg.new_audio_tokens:
        raise RuntimeError(
            f"Segment {seg.index} returned {new_v.shape[2]} video / {new_a.shape[-1]} audio new "
            f"tokens; expected {seg.new_video_tokens} / {seg.new_audio_tokens}.")
    return {"samples": NestedTensor((torch.cat((prev_v, new_v), 2),
                                     torch.cat((prev_a, new_a), -1)))}


# ---------------------------------------------------------------------------
# RefMod Carrier
# ---------------------------------------------------------------------------

_CARRIER_KEY = "mmh3_refmod_carrier"


class MiniMaxH3RefModCarrier:
    """An empty conditioning for Apply H3 RefMod to attach its references to.

    Apply H3 RefMod needs a conditioning input, but Long Shot builds its own per
    segment and only wants RefMod's reference blocks. Feeding RefMod from this
    node instead of a Reference to Video node means nothing else rides along:
    no prompt encode, and no second copy of the references already connected
    to Long Shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_TOOLTIPS = ("Connect to Apply H3 RefMod's conditioning input, then RefMod's output "
                       "to Long Shot's extra_refs. Not a usable prompt — don't feed it to a "
                       "sampler.",)
    FUNCTION = "carry"
    CATEGORY = "MiniMax H3"

    def carry(self):
        return ([[torch.zeros(1, 1, 1), {_CARRIER_KEY: True}]],)


# ---------------------------------------------------------------------------
# Song Track
# ---------------------------------------------------------------------------

class MiniMaxH3SongTrack:
    """Encode a song once, for lip-synced Long Shot generations.

    Trim the song in your audio loader; this node takes it as-is. Long Shot
    then gives each segment its exact slice of this encoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The song, already trimmed to where the video "
                                    "starts. It only needs to cover the Shots' total length; "
                                    "Long Shot tells you if it falls short."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
            }
        }

    RETURN_TYPES = ("MMH3_SONG",)
    RETURN_NAMES = ("song",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3"

    def load(self, audio, audio_vae):
        h3, _ = _native()
        latent, _t = h3._encode_ref_audio(audio_vae, audio)
        seconds = audio["waveform"].shape[-1] / audio["sample_rate"]
        logger.info("MiniMax H3 Song Track: %.2fs, %d audio tokens", seconds, latent.shape[-1])
        # Lyric ranges in the plan are timed from the start of this clip.
        return ({"latent": latent, "seconds": seconds, "offset": 0.0},)


# ---------------------------------------------------------------------------
# Long Shot
# ---------------------------------------------------------------------------

def _ref_inputs(io):
    """The four growing reference inputs, matching the native Reference to Video node."""
    def grow(kind, name, prefix, tip, max_n):
        return io.Autogrow.Input(name, optional=True, template=io.Autogrow.TemplatePrefix(
            input=kind.Input(prefix.rstrip("_"), tooltip=tip), prefix=prefix, min=0, max=max_n))
    return [
        grow(io.Image, "ref_images", "ref_image_",
             "Reference image — <Picture N>, numbered from 1 in connection order. Applies to "
             "every segment, which keeps a character consistent across the joins.", 9),
        grow(io.Image, "ref_videos", "ref_video_",
             "Reference video frames at 24 fps (2-15s) — <Video N>.", 3),
        grow(io.Audio, "ref_video_audios", "ref_video_audio_",
             "Soundtrack of the same-numbered reference video. Takes the first <Audio N> "
             "labels, before any standalone audio.", 3),
        grow(io.Audio, "ref_audios", "ref_audio_",
             "Standalone reference audio, e.g. a voice timbre. Numbered after the video "
             "soundtracks.", 3),
    ]


# ---------------------------------------------------------------------------
# Front-end hooks: per-segment progress over the websocket and a
# machine-readable plan, for H3 Long Shot Studio and anything else that wants
# them. Both are best-effort and never affect the render.
# ---------------------------------------------------------------------------

PROGRESS_EVENT = "mmh3.longshot"


def _notify(payload):
    """Send one progress event to the client that queued the prompt."""
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        if server is not None:
            server.send_sync(PROGRESS_EVENT, payload, getattr(server, "client_id", None))
    except Exception:   # no server (tests, scripts) or a closed socket
        pass


def plan_rows(plan, seeds, own_seeds, statuses, show_seeds=True):
    """The plan as data: one dict per segment, for front ends that would
    otherwise have to parse the text plan."""
    rows = []
    for s, seed, own, status in zip(plan.segments, seeds, own_seeds, statuses):
        reused = status.startswith("reused")
        rows.append({
            "index": s.index,
            "requested_seconds": s.requested_seconds,
            "seconds": round(planner.seconds(s.new_frames), 4),
            "start": round(planner.seconds(s.visible_start), 4),
            "end": round(planner.seconds(s.visible_end), 4),
            "frames": s.new_frames,
            "start_frame": s.visible_start,
            "window_frames": s.window_frames,
            "seed": seed if show_seeds else None,
            "own_seed": bool(own),
            "status": "reused" if reused else "render",
            "source": status[len("reused ("):-1] if reused else None,
            "reason": None if reused else status.split("— ", 1)[-1],
        })
    return rows


def run_long_shot(*args, **kwargs):
    """The whole Long Shot run. Returns (latent, total_frames, total_seconds, plan)."""
    return _run_long_shot(*args, **kwargs)[0]


STORE_ROOT = None          # None: <ComfyUI output>/longshot (tests point it elsewhere)
def _require_audio_vae(refs, audio_vae):
    """Reference audio is only encoded when a Shot renders, so the audio VAE is
    only required then: a fully locked or reused run never loads it."""
    if (refs.get("ref_audios") or refs.get("ref_video_audios")) and audio_vae is None:
        raise ValueError("Reference audio is connected — connect the H3 audio VAE to audio_vae.")


LAZY_INPUTS = ("model", "clip", "vae", "sigmas", "audio_vae")


def _prepare(model, clip, vae, noise, sampler, sigmas, prompt, width, height,
             overlap_frames, seed_mode, dry_run=False, reuse_segments=True,
             ref_image_size="match", audio_vae=None,
             song=None, first_frame=None, last_frame=None, extra_refs=None, refmods=None,
             ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
             save_to_disk=True, cache_name="default", _recipe=None, _planning_only=False):
    """Everything up to sampling: checks, plan, prompts, seeds, fingerprints and
    what each segment will do. Shared by the run and by the lazy-input check,
    so both see exactly the same fingerprints."""
    from types import SimpleNamespace
    if width % 32 or height % 32:
        raise ValueError("width and height must be multiples of 32")
    bundle = prompt or {}
    shots = list(bundle.get("shots") or [])
    if not shots:
        raise ValueError(
            "No shots. Connect a chain of MiniMax H3 Shot nodes to the Ref Prompt Builder's "
            "'shots' input — each Shot becomes one segment.")

    refs = {
        "ref_images": {k: v for k, v in (ref_images or {}).items() if v is not None},
        "ref_videos": {k: v for k, v in (ref_videos or {}).items() if v is not None},
        "ref_video_audios": {k: v for k, v in (ref_video_audios or {}).items() if v is not None},
        "ref_audios": {k: v for k, v in (ref_audios or {}).items() if v is not None},
    }
    native_refs = any(refs.values())
    labels = refmod_labels(refs, refmods)
    if (native_refs or labels) and (first_frame is not None or last_frame is not None):
        raise ValueError(
            "first_frame / last_frame use H3's image-to-video path and can't be combined "
            "with reference inputs or RefMods. Connect one or the other.")
    for name in refs["ref_video_audios"]:
        if "ref_video_" + name.rsplit("_", 1)[-1] not in refs["ref_videos"]:
            raise ValueError(
                f"{name} has no matching {'ref_video_' + name.rsplit('_', 1)[-1]}. A video "
                f"soundtrack only counts when its video is connected too.")

    if is_timeline(shots):
        return _prepare_timeline(
            shots=shots, bundle=bundle, refs=refs, labels=labels, native_refs=native_refs,
            model=model, clip=clip, vae=vae, audio_vae=audio_vae, noise=noise, sampler=sampler,
            sigmas=sigmas, width=width, height=height, overlap_frames=overlap_frames,
            seed_mode=seed_mode, reuse_segments=reuse_segments, ref_image_size=ref_image_size,
            song=song, first_frame=first_frame, last_frame=last_frame, extra_refs=extra_refs,
            refmods=refmods, save_to_disk=save_to_disk, cache_name=cache_name, recipe=_recipe,
            planning_only=_planning_only)

    plan = planner.plan_from_durations([float(s.get("seconds", 0)) for s in shots],
                                       overlap_frames)
    if song is not None:
        need, have = plan.total_audio_tokens, song["latent"].shape[-1]
        if have < need:
            raise ValueError(
                f"The song covers {have / planner.AUDIO_LATENT_FPS:.2f}s but the shots add up to "
                f"{planner.seconds(plan.total_frames):.2f}s. Shorten the shots or give Song "
                f"Track a longer duration.")

    extra_blocks = _extra_ref_blocks(extra_refs) if not _planning_only else \
        [b for e in (extra_refs or []) for b in (e[1].get("minimax_refs", []) or [])]
    if labels and any(b.get("refmod") for b in extra_blocks) and not _planning_only:
        logger.warning(
            "MiniMax H3 Long Shot: RefMods are connected to both refmods and extra_refs. If "
            "they're the same RefMods, they go in twice — use one or the other.")
    prompts = planner.build_prompts(
        shots, plan, bundle, reference_mode=native_refs or bool(labels) or bool(extra_blocks),
        audio_reuse=song is not None,
        use_first_frame=first_frame is not None, use_last_frame=last_frame is not None)
    seeds, own_seeds = segment_seeds(noise, seed_mode, shots)
    shared = shared_components(
        model=model, clip=clip, vae=vae, audio_vae=audio_vae, sampler=sampler,
        sigmas=sigmas, width=width, height=height, refs=refs, ref_image_size=ref_image_size,
        refmods=refmods, extra_blocks=extra_blocks, first_frame=first_frame,
        last_frame=last_frame, recipe=_recipe)
    components = [segment_components(sp, seeds[i], noise, shared, song)
                  for i, sp in enumerate(prompts)]
    fingerprints = chain_fingerprints(components)
    # Disk needs a restart-stable fingerprint, so only with a recipe.
    seg_store = (disk.SegmentStore(cache_name, STORE_ROOT)
                 if reuse_segments and save_to_disk and _recipe is not None else None)
    statuses = segment_statuses(components, fingerprints, reuse_segments, seg_store)
    return SimpleNamespace(plan=plan, prompts=prompts, seeds=seeds, own_seeds=own_seeds,
                           components=components, fingerprints=fingerprints,
                           statuses=statuses, store=seg_store, refs=refs, labels=labels,
                           extra_blocks=extra_blocks, bundle=bundle, timeline=False)


def _window_shapes(seg, width, height):
    video = (1, VIDEO_CHANNELS, planner.video_tokens(seg.window_frames), height // 16, width // 16)
    audio = (1, AUDIO_CHANNELS, AUDIO_STEREO, seg.window_audio_tokens)
    return video, audio


def _run_long_shot(model, clip, vae, noise, sampler, sigmas, prompt, width, height,
                   overlap_frames, seed_mode, dry_run, reuse_segments=True,
                   ref_image_size="match", audio_vae=None,
                   song=None, first_frame=None, last_frame=None, extra_refs=None, refmods=None,
                   ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
                   save_to_disk=True, cache_name="default", _recipe=None):
    """The run, plus the plan as data: ((latent, total_frames, total_seconds, plan), rows)."""
    p = _prepare(model, clip, vae, noise, sampler, sigmas, prompt, width, height,
                 overlap_frames, seed_mode, dry_run, reuse_segments, ref_image_size, audio_vae,
                 song, first_frame, last_frame, extra_refs, refmods, ref_images, ref_videos,
                 ref_video_audios, ref_audios, save_to_disk, cache_name, _recipe)
    if p.timeline:
        return _run_timeline(p, model=model, clip=clip, vae=vae, audio_vae=audio_vae,
                             noise=noise, sampler=sampler, sigmas=sigmas, width=width,
                             height=height, dry_run=dry_run, reuse_segments=reuse_segments,
                             ref_image_size=ref_image_size, song=song, first_frame=first_frame,
                             last_frame=last_frame, refmods=refmods)
    plan, prompts, seeds, own_seeds = p.plan, p.prompts, p.seeds, p.own_seeds
    components, fingerprints, statuses = p.components, p.fingerprints, p.statuses
    refs, labels, extra_blocks, seg_store = p.refs, p.labels, p.extra_blocks, p.store
    if reuse_segments and save_to_disk and _recipe is None:
        logger.info("MiniMax H3 Long Shot: no workflow recipe for this run (called outside "
                    "ComfyUI's executor), so segments stay in memory only")

    show_seeds = seeds if getattr(noise, "seed", None) is not None else None
    label_text = ""
    live = native_labels(refs)
    if live or labels:
        rows = [f"  {label} = {name}" for label, name in live]
        rows += [f"  {label} = {name} (RefMod)" for label, name, _kind in labels]
        label_text = ("Reference labels — use these in your subject definitions:\n"
                      + "\n".join(rows) + "\n\n")
    plan_args = dict(song_offset=song["offset"] if song is not None else None,
                     statuses=statuses, own_seeds=own_seeds)
    report = label_text + planner.render_plan(plan, prompts, show_seeds, **plan_args)
    if seg_store is not None:
        report += f"\nSaved segments: {seg_store.folder}"
    # Console gets the short version; the full plan, prompts included, is the
    # plan output, for a text preview node.
    logger.info("\n%s%s", label_text, planner.render_plan(
        plan, prompts, show_seeds, include_prompts=False, **plan_args))
    total_s = planner.seconds(plan.total_frames)
    rows = plan_rows(plan, seeds, own_seeds, statuses, show_seeds is not None)

    if dry_run:
        # Block the latent so nothing downstream of it runs — no decode, no saved
        # clip — while plan, total_frames and total_seconds still go through.
        from comfy_execution.graph_utils import ExecutionBlocker
        return (ExecutionBlocker(None), plan.total_frames, total_s, report), rows

    _native()
    _require_arbitrary_guides()
    import comfy.model_management
    from comfy.nested_tensor import NestedTensor

    if extra_blocks:
        logger.info("MiniMax H3 Long Shot: adding %d external reference block(s) to every "
                    "segment", len(extra_blocks))
    if seg_store is not None:
        seg_store.clean_tmp()
    _LAST_RUN[:] = components
    encoder = None       # built on the first segment that actually renders

    cumulative = None
    of = len(prompts)

    def progress(seg, status, source=None):
        event = {"segment": seg.index, "of": of, "status": status,
                 "seed": seeds[seg.index - 1] if show_seeds is not None else None,
                 "seconds": round(planner.seconds(seg.new_frames), 4)}
        if source:
            event["source"] = source
        _notify(event)

    for i, sp in enumerate(prompts):
        seg = sp.segment
        fp = fingerprints[i]
        comfy.model_management.throw_exception_if_processing_interrupted()
        sampled, source = (_recall(fp) if reuse_segments else None), "memory"
        if sampled is None and seg_store is not None:
            vshape, ashape = _window_shapes(seg, width, height)
            loaded = seg_store.load(fp, vshape, ashape)
            if loaded is not None:
                sampled, source = {"samples": NestedTensor(loaded)}, "disk"
                _remember(fp, sampled)
        if sampled is not None:
            logger.info("MiniMax H3 Long Shot: segment %d/%d reused from %s", seg.index,
                        len(prompts), source)
            progress(seg, "reused", source)
            cumulative = (sampled if cumulative is None else _append(cumulative, sampled, seg))
            continue
        if clip is None or vae is None:      # lazy inputs skipped by the plan
            raise RuntimeError(
                f"Segment {seg.index} has to render, but the model wasn't loaded because the "
                f"plan expected to reuse it (a saved segment may have been deleted while the "
                f"run started). Queue again.")
        progress(seg, "rendering")
        logger.info("MiniMax H3 Long Shot: segment %d/%d, %d-frame window, seed %s",
                    seg.index, len(prompts), seg.window_frames, seeds[i])
        if encoder is None:
            _require_audio_vae(refs, audio_vae)
            encoder = _RefEncoder(clip, vae, audio_vae, width, height, refs, ref_image_size,
                                  refmods)

        positive = encoder.encode(sp.prompt, seg.window_frames,
                                  first_frame if sp.first_frame else None,
                                  last_frame if sp.last_frame else None)
        positive = _add_ref_blocks(positive, extra_blocks)
        song_audio = _song_slice(song, seg) if song is not None else None
        if cumulative is None:
            if song_audio is not None:
                positive = _pin_audio_at_zero(positive, song_audio)
        else:
            positive = _add_tail_guide(positive, cumulative, seg, song_audio)

        target = _empty_window(width, height, seg,
                               like=None if cumulative is None
                               else cumulative["samples"].tensors[0])
        sampled = _sample_window(model, _segment_noise(noise, seeds[i]),
                                 sampler, sigmas, positive, target)
        if reuse_segments:
            _remember(fp, sampled)
        if seg_store is not None:
            v_s, a_s = sampled["samples"].tensors
            try:
                seg_store.save(fp, v_s, a_s, {"segment": seg.index, "of": of,
                                              "seconds": round(planner.seconds(seg.new_frames), 4),
                                              "seed": seeds[i]})
            except Exception as err:   # a full disk mustn't cost the render
                logger.warning("MiniMax H3 Long Shot: couldn't save segment %d to disk: %s",
                               seg.index, err)
        cumulative = ({"samples": sampled["samples"]} if cumulative is None
                      else _append(cumulative, sampled, seg))
        progress(seg, "done")

    v, a = cumulative["samples"].tensors
    if v.shape[2] != plan.total_video_tokens or a.shape[-1] != plan.total_audio_tokens:
        raise RuntimeError(
            f"Stitched latent is {v.shape[2]} video / {a.shape[-1]} audio tokens; expected "
            f"{plan.total_video_tokens} / {plan.total_audio_tokens}.")
    if song is not None:
        # The song is the source of truth: output it exactly.
        a = song["latent"][..., :plan.total_audio_tokens].to(a)
        cumulative = {"samples": NestedTensor((v, a))}
    return (cumulative, plan.total_frames, total_s, report), rows


# ---------------------------------------------------------------------------
# Timeline mode (round 2): pieces, shared zones, pins and locked takes
#
# A Shot chain whose entries carry an "id" (MiniMax H3 Timeline Shot, or H3
# Long Shot Studio) is a timeline of pieces instead of a one-way chain:
#
# * Every boundary has a shared zone of overlap_frames, owned by the left
#   piece's tail; the right piece drops its first overlap_frames.
# * A locked piece loads its take (a file in output/longshot/<cache>/takes)
#   instead of sampling.
# * A piece that renders is pinned at its start to its left neighbour's tail
#   (as in round 1) and, when its right neighbour is locked, at its end to a
#   fixed slice: its own old tail when that neighbour was continued from it,
#   otherwise the neighbour's head. After sampling the end slice is written
#   back exactly, so the locked neighbour joins without a seam.
# * A rendered piece's fingerprint is its own inputs plus the identity of its
#   pin sources, so moving pieces around never re-renders an unchanged one.
# ---------------------------------------------------------------------------

_LAST_TIMELINE = {}      # shot id -> {"comps": {...}, "start": ..., "end": ...}

_PIN_REASONS = (("start", "start pin changed"), ("end", "end pin changed"))


def is_timeline(shots):
    return any(isinstance(sh, dict) and sh.get("id") for sh in shots or [])


def piece_components(sp, seed, noise, shared, song):
    """segment_components without the window's position: a piece's result
    doesn't depend on where it sits, only on its pins (and its song slice)."""
    seg = sp.segment
    comps = segment_components(sp, seed, noise, shared, song)
    comps["length"] = _digest([seg.window_frames, seg.overlap_frames, seg.new_frames])
    return comps


def _fit_audio(audio, n):
    """A take's audio placed where the 40 Hz grid gives one token more or less
    than where it was rendered: drop or repeat the last token."""
    have = audio.shape[-1]
    if have == n:
        return audio
    if abs(have - n) > 2:
        raise RuntimeError(f"audio of {have} tokens can't fill a {n}-token slot")
    if have > n:
        return audio[..., :n]
    return torch.cat([audio] + [audio[..., -1:]] * (n - have), -1)


def _shot_label(i):
    return f"Shot {i + 1}"


def _prepare_timeline(*, shots, bundle, refs, labels, native_refs, model, clip, vae, audio_vae,
                      noise, sampler, sigmas, width, height, overlap_frames, seed_mode,
                      reuse_segments, ref_image_size, song, first_frame, last_frame, extra_refs,
                      refmods, save_to_disk, cache_name, recipe, planning_only):
    from types import SimpleNamespace
    takes = disk.TakeStore(cache_name, STORE_ROOT)
    n = len(shots)
    ids = [str(sh.get("id") or f"seg{i + 1}") for i, sh in enumerate(shots)]
    if len(set(ids)) != n:
        raise ValueError("Two Shots in the timeline have the same id.")
    joins = []
    for i, sh in enumerate(shots):
        j = (sh.get("join") or "bridge").lower()
        if j not in ("bridge", "cut"):
            raise ValueError(f"{_shot_label(i)}: join must be 'bridge' or 'cut', not {j!r}.")
        joins.append("cut" if i and j == "cut" else ("bridge" if i else None))

    # Clips: never sampled; they sit in the timeline like a locked take.
    kinds = ["clip" if sh.get("kind") == "clip" else "shot" for sh in shots]
    for i in range(1, n):
        if kinds[i] == "clip" and kinds[i - 1] == "clip":
            joins[i] = "cut"                     # nothing generated between them to bridge
    ovt_ = planner.video_tokens(overlap_frames)
    if first_frame is not None and kinds[0] == "clip":
        raise ValueError("first_frame guides the first generated Shot, but the timeline starts "
                         "with a clip. Disconnect first_frame or start with a Shot.")
    if last_frame is not None and kinds[-1] == "clip":
        raise ValueError("last_frame guides the last generated Shot, but the timeline ends with "
                         "a clip. Disconnect last_frame or end with a Shot.")
    locks = [None] * n
    for i, sh in enumerate(shots):
        if kinds[i] != "clip":
            continue
        c = sh.get("clip")
        if not isinstance(c, dict) or "video" not in c:
            raise ValueError(f"{_shot_label(i)} is a clip with no prepared video (MiniMax H3 Clip).")
        if (c.get("width"), c.get("height")) != (width, height):
            raise ValueError(f"{_shot_label(i)}'s clip was prepared at {c.get('width')}x{c.get('height')}; "
                             f"the film is {width}x{height}. Prepare it at the film's size.")
        if c["frames"] <= overlap_frames:
            raise ValueError(f"{_shot_label(i)}'s clip is {planner.seconds(c['frames']):.2f}s, too short "
                             f"to join with a {overlap_frames}-frame shared zone. Use a longer trim.")
        ident = c.get("digest") or _digest([c["video"], c["audio"]])
        # Identity strings rather than latent hashes: a clip re-encoded after a
        # restart can differ in the last bits, but it's the same clip.
        locks[i] = {"name": f"clip:{ids[i]}", "clip": c, "window_frames": c["frames"],
                    "fp": ident, "seed": None,
                    "head": f"clip:{ident}:head:{overlap_frames}",
                    "tail": f"clip:{ident}:tail:{overlap_frames}",
                    "left_take": None, "left_tail": None}

    # Locked takes: metadata only, so planning never loads tensors.
    for i, sh in enumerate(shots):
        if kinds[i] == "clip":
            continue
        name = sh.get("lock") or None
        if not name:
            continue
        meta = takes.meta(name)
        if meta is None:
            raise ValueError(
                f"{_shot_label(i)}'s approved take is missing ({name}); re-render it or unlock it.")
        bad = [f"{k} {meta.get(k)} (now {v})" for k, v in
               (("width", width), ("height", height), ("overlap", overlap_frames))
               if meta.get(k) != v]
        if bad:
            raise ValueError(f"{_shot_label(i)}'s take was made with " + ", ".join(bad)
                             + ". Re-render it or unlock it.")
        locks[i] = meta

    specs = []
    for i, sh in enumerate(shots):
        keep = sh.get("frames")
        window = locks[i]["window_frames"] if locks[i] else (
            int(keep) if keep and planner.is_valid_frame_count(int(keep)) else None)
        specs.append(planner.PieceSpec(float(sh.get("seconds", 0) or 0), window, bool(locks[i])))
    plan = planner.plan_pieces(specs, overlap_frames)
    if song is not None:
        need, have = plan.total_audio_tokens, song["latent"].shape[-1]
        if have < need:
            raise ValueError(
                f"The song covers {have / planner.AUDIO_LATENT_FPS:.2f}s but the shots add up to "
                f"{planner.seconds(plan.total_frames):.2f}s. Shorten the shots or give Song "
                f"Track a longer duration.")

    extra_blocks = _extra_ref_blocks(extra_refs) if not planning_only else \
        [b for e in (extra_refs or []) for b in (e[1].get("minimax_refs", []) or [])]
    prompts = planner.build_prompts(
        shots, plan, bundle, reference_mode=native_refs or bool(labels) or bool(extra_blocks),
        audio_reuse=song is not None,
        use_first_frame=first_frame is not None, use_last_frame=last_frame is not None)
    seeds, own_seeds = segment_seeds(noise, seed_mode, shots)
    for i, m in enumerate(locks):
        if m is not None and m.get("seed") is not None:
            seeds[i], own_seeds[i] = m["seed"], True
        elif m is not None and m.get("clip") is not None:
            seeds[i], own_seeds[i] = None, False
    shared = shared_components(
        model=model, clip=clip, vae=vae, audio_vae=audio_vae, sampler=sampler,
        sigmas=sigmas, width=width, height=height, refs=refs, ref_image_size=ref_image_size,
        refmods=refmods, extra_blocks=extra_blocks, first_frame=first_frame,
        last_frame=last_frame, recipe=recipe)
    comps = [piece_components(sp, seeds[i], noise, shared, song) for i, sp in enumerate(prompts)]

    # Pins and fingerprints, left to right.
    ov = overlap_frames
    starts, ends, fps, seams = [None] * n, [None] * n, [None] * n, [None] * n
    prevs, reuse_own = [None] * n, [None] * n
    for i in range(n):
        left = i - 1 if i and joins[i] == "bridge" and kinds[i] != "clip" else None
        if left is not None:
            starts[i] = ({"from": left, "kind": "take_tail", "hash": locks[left]["tail"],
                          "take": locks[left]["name"]} if locks[left] else
                         {"from": left, "kind": "render", "fp": fps[left]})
        if locks[i]:
            continue
        r = i + 1
        if r < n and locks[r] and joins[r] == "bridge":
            prev_take = locks[r].get("left_take") or ""
            own = prev_take and disk.take_shot(prev_take) == disk.safe_shot_id(ids[i])
            old = takes.meta(prev_take) if own else None
            if old is not None and old.get("tail"):
                ends[i] = {"to": r, "kind": "old_tail", "take": prev_take, "hash": old["tail"]}
            else:
                ends[i] = {"to": r, "kind": "head", "take": locks[r]["name"],
                           "hash": locks[r]["head"]}
            if plan.segments[i].window_frames - ov < (ov if starts[i] or i else 1):
                raise ValueError(
                    f"{_shot_label(i)} is too short to be pinned at both ends. Make it at least "
                    f"{planner.seconds(2 * ov + planner.GROUP_FRAMES):.1f}s.")
        if starts[i] is None:
            prev = ""
        elif starts[i]["kind"] == "take_tail":
            # The fingerprint the left take was made with, so a Shot keeps
            # its fingerprint whether the one before it is locked or not.
            prev = locks[i - 1].get("fp") or ("tail:" + starts[i]["hash"])
        else:
            prev = starts[i]["fp"]
        prevs[i] = prev
        parts = [prev, [comps[i][k] for k, _ in _REASONS]]
        if ends[i] and ends[i]["kind"] == "old_tail":
            # Nothing about this Shot changed: its current take is exactly what
            # the locked Shot after it was continued from. Keep it.
            same = _digest(parts)
            old_meta = takes.meta(ends[i]["take"])
            if old_meta is not None and old_meta.get("fp") == same:
                fps[i], ends[i], reuse_own[i] = same, None, ends[i]["take"]
                continue
        if ends[i]:
            parts.append(["end", ends[i]["hash"]])
        fps[i] = _digest(parts)

    # Seams: does each locked piece still join the piece before it?
    for i in range(1, n):
        if joins[i] == "cut":
            seams[i] = "cut"
            continue
        if not locks[i]:
            seams[i] = "ok"
            continue
        left = i - 1
        if kinds[i] == "clip":
            # ok when the Shot before it leads into the clip's head
            led = (locks[left].get("end_src") if locks[left] else
                   ends[left]["hash"] if ends[left] else None)
            seams[i] = "ok" if led == locks[i]["head"] else "mismatch"
            continue
        actual = (locks[left]["tail"] if locks[left] else
                  ends[left]["hash"] if ends[left] else None)
        seams[i] = "ok" if actual in (locks[i].get("left_tail"), locks[i].get("head")) else "mismatch"

    # Round-1 saved segments, adopted once when the timeline is laid out the
    # way round 1 laid it out (nothing locked, no end pins, no cuts).
    legacy = [None] * n
    if (recipe is not None and reuse_segments and not any(locks) and not any(ends)
            and all(j != "cut" for j in joins)):
        old_plan = planner.plan_from_durations([float(sh.get("seconds", 0)) for sh in shots], ov)
        if [s.window_frames for s in old_plan.segments] == [s.window_frames for s in plan.segments]:
            old_store = disk.SegmentStore(cache_name, STORE_ROOT)
            old_fps = chain_fingerprints([segment_components(sp, seeds[i], noise, shared, song)
                                          for i, sp in enumerate(prompts)])
            for i, fp in enumerate(old_fps):
                if not old_store.has(fp):
                    break
                legacy[i] = (old_store, fp)

    statuses, take_names = [], [None] * n
    for i in range(n):
        if kinds[i] == "clip":
            statuses.append("clip")
            continue
        if locks[i]:
            statuses.append("locked")
            take_names[i] = locks[i]["name"]
            continue
        name = reuse_own[i] or (takes.find(ids[i], seeds[i], fps[i]) if recipe is not None else None)
        if reuse_own[i] and reuse_segments:
            statuses.append("reused (disk)")
            take_names[i] = name
        elif reuse_segments and fps[i] in _SEGMENT_CACHE:
            statuses.append("reused (memory)")
        elif reuse_segments and name:
            statuses.append("reused (disk)")
            take_names[i] = name
        elif reuse_segments and legacy[i]:
            statuses.append("reused (disk)")
        else:
            last = _LAST_TIMELINE.get(ids[i])
            if not reuse_segments:
                why = "reuse off"
            elif last is None:
                why = "new take"
            else:
                now = {"comps": comps[i], "start": prevs[i], "end": _pin_identity(ends[i])}
                changed = [label for key, label in _REASONS if now["comps"][key] != last["comps"][key]]
                changed += [label for key, label in _PIN_REASONS if now[key] != last[key]]
                why = changed[0] if changed else "not in memory"
            statuses.append(f"will render — {why}")
    return SimpleNamespace(
        timeline=True, plan=plan, prompts=prompts, seeds=seeds, own_seeds=own_seeds,
        components=comps, fingerprints=fps, statuses=statuses, store=takes, refs=refs,
        labels=labels, extra_blocks=extra_blocks, bundle=bundle, ids=ids, joins=joins,
        locks=locks, starts=starts, ends=ends, seams=seams, legacy=legacy, prevs=prevs, kinds=kinds,
        take_names=take_names, save=save_to_disk, recipe=recipe, overlap=ov)


def _pin_identity(pin):
    if pin is None:
        return None
    return pin.get("hash") or pin.get("fp")


def _pin_text(p, i):
    bits = []
    st, en = p.starts[i], p.ends[i]
    if st:
        bits.append(f"start ← {_shot_label(st['from'])} tail")
    if en:
        bits.append("end → its old tail (keeps " + _shot_label(en["to"]) + ")"
                    if en["kind"] == "old_tail" else f"end → {_shot_label(en['to'])} head")
    if p.joins[i] == "cut":
        bits.append("cut")
    if p.seams[i] == "mismatch":
        bits.append("hard cut: continued from a different take")
    return " · ".join(bits)


def timeline_rows(p, show_seeds=True):
    rows = plan_rows(p.plan, p.seeds, p.own_seeds,
                     ["reused (take)" if s in ("locked", "clip") else s for s in p.statuses], show_seeds)
    for i, row in enumerate(rows):
        st, en = p.starts[i], p.ends[i]
        row.update({
            "kind": p.kinds[i],
            "id": p.ids[i],
            "locked": bool(p.locks[i]),
            "take": p.take_names[i],
            "join": p.joins[i],
            "seam": p.seams[i],
            "pins": {
                "start": {"from": st["from"] + 1} if st else None,
                "end": ({"to": en["to"] + 1, "kind": en["kind"], "take": en["take"]}
                        if en else None),
            },
        })
        if p.kinds[i] == "clip":
            row.update(status="clip", source="clip", reason=None, take=None, seed=None)
        elif p.locks[i]:
            row.update(status="locked", source="take", reason=None)
    return rows


def _load_take(p, name, i=None):
    if isinstance(name, str) and name.startswith("clip:"):
        for m in p.locks:
            if m is not None and m.get("name") == name and m.get("clip") is not None:
                c = m["clip"]
                return c["video"], c["audio"], m
        raise ValueError(f"The clip {name[5:]} isn't in this run.")
    try:
        return p.store.load(name)
    except FileNotFoundError:
        who = f"{_shot_label(i)}'s take" if i is not None else "A take this run needs"
        raise ValueError(f"{who} is missing ({name}); it was deleted while the run started. "
                         f"Re-render it or unlock it.") from None


def _run_timeline(p, *, model, clip, vae, audio_vae, noise, sampler, sigmas, width, height,
                  dry_run, reuse_segments, ref_image_size, song, first_frame, last_frame,
                  refmods):
    plan, prompts, seeds = p.plan, p.prompts, p.seeds
    n = len(prompts)
    ov = p.overlap
    ovt = planner.video_tokens(ov)
    show_seeds = seeds if getattr(noise, "seed", None) is not None else None
    label_text = ""
    live = native_labels(p.refs)
    if live or p.labels:
        rows = [f"  {label} = {name}" for label, name in live]
        rows += [f"  {label} = {name} (RefMod)" for label, name, _kind in p.labels]
        label_text = ("Reference labels — use these in your subject definitions:\n"
                      + "\n".join(rows) + "\n\n")
    text_status = []
    for i, s in enumerate(p.statuses):
        extra = _pin_text(p, i)
        text_status.append(s + (f" · {extra}" if extra else ""))
    plan_args = dict(song_offset=song["offset"] if song is not None else None,
                     statuses=text_status, own_seeds=p.own_seeds)
    report = label_text + planner.render_plan(plan, prompts, show_seeds, **plan_args)
    report += f"\nTakes: {p.store.folder}"
    logger.info("\n%s%s", label_text, planner.render_plan(
        plan, prompts, show_seeds, include_prompts=False, **plan_args))
    total_s = planner.seconds(plan.total_frames)

    if dry_run:
        from comfy_execution.graph_utils import ExecutionBlocker
        return (ExecutionBlocker(None), plan.total_frames, total_s, report), \
            timeline_rows(p, show_seeds is not None)

    from comfy.nested_tensor import NestedTensor
    import comfy.model_management
    p.store.clean_tmp()
    checked = False
    encoder = None

    def progress(i, status, source=None):
        seg = plan.segments[i]
        event = {"segment": i + 1, "of": n, "status": status, "id": p.ids[i],
                 "seed": seeds[i] if show_seeds is not None else None,
                 "seconds": round(planner.seconds(seg.new_frames), 4)}
        if source:
            event["source"] = source
        _notify(event)

    placed = [None] * n          # (video, audio) per piece, audio fitted to its slot

    def place(i, video, audio):
        seg = plan.segments[i]
        if video.shape[2] != planner.video_tokens(seg.window_frames):
            raise RuntimeError(f"{_shot_label(i)}'s take has {video.shape[2]} video tokens; "
                               f"its {seg.window_frames}-frame window needs "
                               f"{planner.video_tokens(seg.window_frames)}.")
        placed[i] = (video, _fit_audio(audio, seg.window_audio_tokens))

    def zone_audio(i, at_end):
        """Audio of piece i's head or tail zone as it sits now."""
        seg = plan.segments[i]
        a = placed[i][1]
        if at_end:
            return a[..., -planner.tail_audio_tokens(seg, ov):]
        return a[..., :planner.head_audio_tokens(seg, ov)]

    def left_hash(i):
        st = p.starts[i]
        if st is None:
            return ""
        if p.kinds[st["from"]] == "clip":
            return p.locks[st["from"]]["tail"]
        return disk.slice_hash(placed[st["from"]][0][:, :, -ovt:])

    def song_span(f0, f1):
        return song["latent"][..., planner.audio_at(f0):planner.audio_at(f1)].clone()

    for i in range(n):
        seg = plan.segments[i]
        comfy.model_management.throw_exception_if_processing_interrupted()
        status = p.statuses[i]
        if status in ("locked", "clip"):
            v, a, _meta = _load_take(p, p.locks[i]["name"], i)
            place(i, v, a)
            progress(i, "reused", "clip" if status == "clip" else "take")
            continue
        fp = p.fingerprints[i]
        got = None
        if status == "reused (memory)":
            got, source = _recall(fp), "memory"
        if got is None and status.startswith("reused") and p.take_names[i]:
            try:
                v, a, _meta = p.store.load(p.take_names[i])
                got, source = {"samples": NestedTensor((v, a))}, "disk"
            except (OSError, ValueError):
                got = None
        if got is None and status.startswith("reused") and p.legacy[i]:
            old_store, old_fp = p.legacy[i]
            loaded = old_store.load(old_fp, *_window_shapes(seg, width, height))
            if loaded is not None:
                got, source = {"samples": NestedTensor(loaded)}, "disk"
        if got is not None:
            v, a = got["samples"].tensors
            place(i, v, a)
            if reuse_segments:
                _remember(fp, got)
            _keep_take(p, i, v, a, left_hash(i))
            progress(i, "reused", source)
            continue

        # --- render -----------------------------------------------------------
        if clip is None or vae is None:
            raise RuntimeError(
                f"{_shot_label(i)} has to render, but the model wasn't loaded because the plan "
                f"expected to reuse it (a take may have been deleted while the run started). "
                f"Queue again.")
        if not checked:
            _native()
            _require_arbitrary_guides()
            checked = True
        progress(i, "rendering")
        logger.info("MiniMax H3 Long Shot: %s, %d-frame window, seed %s", _shot_label(i),
                    seg.window_frames, seeds[i])
        if encoder is None:
            _require_audio_vae(p.refs, audio_vae)
            encoder = _RefEncoder(clip, vae, audio_vae, width, height, p.refs, ref_image_size,
                                  refmods)
        sp = prompts[i]
        positive = encoder.encode(sp.prompt, seg.window_frames,
                                  first_frame if sp.first_frame else None,
                                  last_frame if sp.last_frame else None)
        positive = _add_ref_blocks(positive, p.extra_blocks)
        st, en = p.starts[i], p.ends[i]
        kfs = _keyframes(positive)
        if st is not None:
            lv = placed[st["from"]][0]
            for k in kfs:
                if 0 <= float(k["resolved_frame_index"]) < ov:
                    raise ValueError(f"{_shot_label(i)} already has a guide inside its shared "
                                     f"zone, which would fight the continuation.")
            kfs.append({"resolved_frame_index": 0,
                        "latent": lv[:, :, -ovt:].clone(),
                        "audio_latent": (song_span(seg.window_start, seg.window_end)
                                         if song is not None else zone_audio(st["from"], True).clone())})
        elif song is not None:
            positive = _pin_audio_at_zero(positive, song_span(seg.window_start, seg.window_end))
            kfs = _keyframes(positive)
        end_v = end_a = None
        if en is not None:
            # The zone's audio is exactly what the neighbour drops (or showed),
            # counted by global position.
            n_tail = planner.tail_audio_tokens(seg, ov)
            if en["kind"] == "old_tail":
                ov_v, ov_a, _ov_meta = _load_take(p, en["take"])
                end_v = ov_v[:, :, -ovt:]
                end_a = ov_a[..., -n_tail:]
            else:
                rv, ra, _r_meta = _load_take(p, en["take"])
                end_v = rv[:, :, :ovt]
                end_a = ra[..., :n_tail]
            end_a = _fit_audio(end_a, n_tail)
            ref_dtype = placed[st["from"]][0] if st is not None else None
            end_v = end_v.to(ref_dtype) if ref_dtype is not None else end_v.float()
            end_a = end_a.to(end_v.dtype)
            end_guide = {"resolved_frame_index": seg.window_frames - ov, "latent": end_v.clone()}
            if song is None:
                end_guide["audio_latent"] = end_a.clone()
            kfs.append(end_guide)
        if kfs:
            positive = _set_keyframes(positive, kfs)
        like = placed[st["from"]][0] if st is not None else None
        target = _empty_window(width, height, seg, like=like)
        sampled = _sample_window(model, _segment_noise(noise, seeds[i]),
                                 sampler, sigmas, positive, target)
        v, a = sampled["samples"].tensors
        if end_v is not None:
            # write the end slice back exactly, so the locked neighbour is untouched
            v = v.clone()
            a = a.clone()
            v[:, :, -ovt:] = end_v.to(v)
            a[..., -end_a.shape[-1]:] = end_a.to(a)
            sampled = {"samples": NestedTensor((v, a))}
        _LAST_TIMELINE[p.ids[i]] = {"comps": p.components[i], "start": p.prevs[i],
                                    "end": _pin_identity(en)}
        place(i, v, a)
        if reuse_segments:
            _remember(fp, sampled)
        _keep_take(p, i, v, a, left_hash(i))
        progress(i, "done")

    cumulative = {"samples": NestedTensor(placed[0])}
    for i in range(1, n):
        cumulative = _append(cumulative, {"samples": NestedTensor(placed[i])}, plan.segments[i])
    v, a = cumulative["samples"].tensors
    if v.shape[2] != plan.total_video_tokens or a.shape[-1] != plan.total_audio_tokens:
        raise RuntimeError(
            f"Stitched latent is {v.shape[2]} video / {a.shape[-1]} audio tokens; expected "
            f"{plan.total_video_tokens} / {plan.total_audio_tokens}.")
    if song is not None:
        a = song["latent"][..., :plan.total_audio_tokens].to(a)
        cumulative = {"samples": NestedTensor((v, a))}
    spans = []
    for i, seg in enumerate(plan.segments):
        if p.kinds[i] == "clip":
            spans.append({"start": seg.visible_start, "frames": seg.new_frames,
                          "offset": seg.overlap_frames, "pixels": p.locks[i]["clip"]["pixels"]})
    if spans:
        cumulative["mmh3_clips"] = spans
    return (cumulative, plan.total_frames, total_s, report), timeline_rows(p, show_seeds is not None)


def _keep_take(p, i, video, audio, left_tail=""):
    """Save piece i as a take (once), so the Studio can lock it later."""
    if not p.save or p.locks[i]:
        return
    ov = p.overlap
    ovt = planner.video_tokens(ov)
    seg = p.plan.segments[i]
    name = disk.TakeStore.take_name(p.ids[i], p.seeds[i], p.fingerprints[i])
    p.take_names[i] = name
    if p.store.exists(name):
        meta = p.store.meta(name)
        if meta is not None and meta.get("fp") == p.fingerprints[i]:
            return
    st, en = p.starts[i], p.ends[i]
    left_take = ""
    if st is not None:
        left_take = p.take_names[st["from"]] or ""
    meta = {
        "shot_id": p.ids[i], "seed": p.seeds[i], "fp": p.fingerprints[i],
        "window_frames": seg.window_frames, "overlap": ov,
        "width": video.shape[4] * 16, "height": video.shape[3] * 16,
        "head": disk.slice_hash(video[:, :, :ovt]), "tail": disk.slice_hash(video[:, :, -ovt:]),
        "head_audio": planner.head_audio_tokens(seg, ov),
        "tail_audio": planner.tail_audio_tokens(seg, ov),
        "left_take": left_take, "left_tail": left_tail,
        "end_src": en["hash"] if en else "",
        "seconds": round(planner.seconds(seg.new_frames), 4),
    }
    try:
        p.store.save(name, video, audio, meta)
    except Exception as err:   # a full disk mustn't cost the render
        logger.warning("MiniMax H3 Long Shot: couldn't save %s's take: %s", _shot_label(i), err)
        p.take_names[i] = None



def _plain(value):
    """Lazy checks may hand dynamic inputs over as (value, key) pairs; unwrap them."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return value[0]
    return value


def lazy_requests(kwargs, prompt, unique_id):
    """Which lazy inputs Long Shot needs. Model, CLIP, VAEs and sigmas are only
    worth loading when a segment will actually sample: a dry run, or a run
    where every segment comes from memory or disk, needs none of them."""
    linked = disk.linked_inputs(prompt, unique_id)
    if linked is None:      # no prompt: request the required ones, as before
        return [n for n in ("model", "clip", "vae", "sigmas") if kwargs.get(n) is None]
    wanted = [n for n in LAZY_INPUTS if n in linked and kwargs.get(n) is None]
    if not wanted or kwargs.get("dry_run"):
        return []
    recipe = disk.recipe_components(prompt, unique_id)
    if recipe is None or not kwargs.get("reuse_segments", True):
        return wanted
    args = {k: _plain(v) for k, v in kwargs.items()}
    try:
        p = _prepare(**args, _recipe=recipe, _planning_only=True)
    except Exception:       # let the run itself report the problem
        return wanted
    return [] if all(s.startswith("reused") or s.startswith("locked") or s == "clip"
                     for s in p.statuses) else wanted


def _schema_extras(io):
    """Schema flags newer ComfyUI builds understand. has_intermediate_output
    makes ComfyUI resend the plan UI when the whole run comes from its cache,
    so front ends reading /history still get the plan. Older builds don't know
    the flag, so it's only passed where it exists."""
    fields = getattr(io.Schema, "__dataclass_fields__", {})
    return {"has_intermediate_output": True} if "has_intermediate_output" in fields else {}


def _make_long_shot_node():
    from comfy_api.latest import io

    class MiniMaxH3LongShot(io.ComfyNode):
        """Render a chain of Shot nodes as one continuous shot, one generation per
        Shot, stitched in latent space for a single seamless decode."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MiniMaxH3LongShot",
                display_name="MiniMax H3 Long Shot",
                category="MiniMax H3",
                description="One continuous shot from a chain of Shot nodes: one H3 generation "
                            "per Shot, stitched in latent space so a single decode is seamless.",
                inputs=[
                    io.Model.Input("model", lazy=True,
                                   tooltip="Only loaded when a segment has to render: a dry run "
                                   "or a fully reused run never loads the model."),
                    io.Clip.Input("clip", lazy=True),
                    io.Vae.Input("vae", lazy=True, tooltip="H3 video VAE."),
                    io.Noise.Input("noise"),
                    io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas", lazy=True),
                    io.Custom("MMH3_LONGSHOT").Input(
                        "prompt", tooltip="The long_shot output of MiniMax H3 Ref Prompt Builder "
                        "r2v, with a Shot chain wired into its 'shots' input. Each Shot is one "
                        "segment: its text is that segment's prompt, its seconds its length. "
                        "cut_verb is ignored — this is one continuous shot."),
                    io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                    io.Int.Input("height", default=768, min=32, max=4096, step=32),
                    io.Int.Input("overlap_frames", default=22, min=5, max=107, step=17,
                                 tooltip="Hidden motion context shared across each join: 5, 22, "
                                 "39, 56... More carries motion better but re-samples more."),
                    io.Combo.Input("seed_mode", options=["increment", "same"],
                                   default="increment",
                                   tooltip="'increment' gives each segment seed+1. Only affects "
                                   "RandomNoise."),
                    io.Boolean.Input("dry_run", default=False,
                                     tooltip="Output the plan and every segment's prompt without "
                                     "sampling. The latent is blocked, so everything wired after "
                                     "it (decoders, video save) is skipped."),
                    io.Boolean.Input("reuse_segments", default=True,
                                     tooltip="Remember finished segments and reuse any that "
                                     "would come out identical, so only Shots you changed — "
                                     "and the ones after them — render again. Kept in memory "
                                     "until ComfyUI restarts. The plan shows which segments "
                                     "will be reused."),
                    io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                                   tooltip="'max' holds identity better but is slower — paid on "
                                   "every segment."),
                    # New widgets go last, so saved workflows keep their values.
                    io.Boolean.Input("save_to_disk", default=True, optional=True,
                                     tooltip="Also save each finished segment to "
                                     "output/longshot/<cache_name>/segments, so a crash or "
                                     "restart doesn't cost finished Shots. Needs reuse_segments."),
                    io.String.Input("cache_name", default="default", optional=True,
                                    tooltip="Folder name for this project's saved segments "
                                    "(letters, digits, . _ -). H3 Long Shot Studio sets it to "
                                    "the project."),
                    io.Vae.Input("audio_vae", optional=True, lazy=True,
                                 tooltip="H3 audio VAE. Needed only when reference audio or video "
                                 "soundtracks are connected."),
                    io.Custom("MMH3_SONG").Input(
                        "song", optional=True, tooltip="From MiniMax H3 Song Track. Each segment "
                        "gets its exact slice of the song pinned to its timeline for lip sync."),
                    io.Image.Input("first_frame", optional=True,
                                   tooltip="Opening frame for segment 1. Can't be combined with "
                                   "reference inputs."),
                    io.Image.Input("last_frame", optional=True,
                                   tooltip="Closing frame for the final segment. Can't be "
                                   "combined with reference inputs."),
                    io.Custom("H3_REF_MODS").Input(
                        "refmods", optional=True, tooltip="From Load H3 RefMods. Each RefMod is "
                        "shown to the text encoder under the next free label after your "
                        "reference inputs — the plan lists them — so your subject definitions "
                        "can point at it. Don't also apply the same RefMods through extra_refs."),
                    io.Conditioning.Input(
                        "extra_refs", optional=True, tooltip="Reference blocks only, e.g. from "
                        "MiniMax H3 RefMod Carrier → Apply H3 RefMod, for Apply's retention and "
                        "curve controls. These get no label in the prompt; use refmods for "
                        "RefMods your prompt refers to."),
                ] + _ref_inputs(io),
                outputs=[
                    io.Latent.Output(display_name="latent"),
                    io.Int.Output(display_name="total_frames"),
                    io.Float.Output(display_name="total_seconds"),
                    io.String.Output(display_name="plan"),
                ],
                hidden=[io.Hidden.prompt, io.Hidden.unique_id],
                **_schema_extras(io),
            )

        @classmethod
        def _hidden(cls):
            h = getattr(cls, "hidden", None)
            return getattr(h, "prompt", None), getattr(h, "unique_id", None)

        @classmethod
        def check_lazy_status(cls, **kwargs):
            return lazy_requests(kwargs, *cls._hidden())

        @classmethod
        def execute(cls, **kwargs) -> io.NodeOutput:
            recipe = disk.recipe_components(*cls._hidden())
            outputs, rows = _run_long_shot(**kwargs, _recipe=recipe)
            # The plan also goes out as a UI output, so /history carries it
            # (dry run included) for front ends like H3 Long Shot Studio.
            return io.NodeOutput(*outputs, ui={"text": [outputs[3]], "plan_json": rows})

    return MiniMaxH3LongShot


try:
    MiniMaxH3LongShot = _make_long_shot_node()
except ImportError:   # ComfyUI too old for the V3 node API
    MiniMaxH3LongShot = None


class MiniMaxH3TimelineShot:
    """A Shot for Long Shot's timeline mode: the same as MiniMax H3 Shot, plus
    an id, an optional locked take, and how it joins the piece before it.

    Chain these into the Ref Prompt Builder's 'shots' input like Shot nodes.
    With ids on the chain, Long Shot treats it as a timeline: locked Shots load
    their take, and a Shot that renders is pinned to the locked Shots on both
    sides, so you can re-roll a Shot in the middle and keep the ones after it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_id": ("STRING", {"default": "", "tooltip": "Stable id for this Shot. Its "
                            "takes are saved as <id>__<seed>__<fingerprint>.safetensors."}),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 60.0, "step": 0.01}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "shot_seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                              "tooltip": "-1 follows Long Shot's seed. A Shot that has rendered "
                              "should keep the seed it used, so reordering never changes it."}),
                "lock": ("STRING", {"default": "", "tooltip": "A take file name from "
                         "output/longshot/<cache_name>/takes. Locked Shots load the take instead "
                         "of sampling. Empty: render (or reuse) as usual."}),
                "join": (["bridge", "cut"], {"default": "bridge", "tooltip": "How this Shot "
                         "joins the one before it. 'bridge' continues from it; 'cut' starts "
                         "fresh (an intentional hard cut)."}),
                "frames": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Keep this "
                           "exact window length (17k + 5 frames), e.g. when re-rendering in place. "
                           "0: from seconds."}),
            },
            "optional": {
                "shots": ("MMH3_SHOTS", {"tooltip": "Chain from the previous Shot node. Leave "
                          "empty on the first one."}),
            },
        }

    RETURN_TYPES = ("MMH3_SHOTS",)
    RETURN_NAMES = ("shots",)
    FUNCTION = "add"
    CATEGORY = "MiniMax H3"

    def add(self, shot_id, seconds, text, shot_seed=-1, lock="", join="bridge", frames=0,
            shots=None):
        chain = list(shots) if shots else []
        if not str(shot_id).strip():
            logger.warning("MiniMax H3 Timeline Shot %d has no shot_id; it is filed as shot%d, "
                           "which changes if Shots are inserted before it. Give it an id.",
                           len(chain) + 1, len(chain) + 1)
        if frames and not planner.is_valid_frame_count(int(frames)):
            logger.warning("MiniMax H3 Timeline Shot %d: frames=%d is not 17k + 5; using seconds.",
                           len(chain) + 1, frames)
        chain.append({
            "kind": "shot",
            "id": str(shot_id).strip() or f"shot{len(chain) + 1}",
            "seconds": float(seconds),
            "text": str(text or "").strip(),   # as MiniMax H3 Shot does: same prompt, same fingerprint
            "cut_verb": "the camera cuts to",
            "seed": int(shot_seed),
            "lock": str(lock or "").strip() or None,
            "join": join,
            "frames": int(frames) or None,
        })
        return (chain,)


# ---------------------------------------------------------------------------
# Clip Shots (round 2, step 3): a real video as a piece of the timeline
# ---------------------------------------------------------------------------

class MiniMaxH3Clip:
    """Prepare a video clip for the timeline: resize and centre-crop to the
    generation size, trim to a 17k + 5 frame length, and encode it with the H3
    video VAE (and its sound, or silence, with the H3 audio VAE). Frames are
    expected at 24 fps (VHS Load Video with force_rate 24)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames at 24 fps."}),
                "vae": ("VAE", {"tooltip": "H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
                "frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Use this many "
                           "frames (snapped down to 17k + 5). 0: as many as the clip has."}),
                "audio_mode": (["clip", "mute"], {"default": "clip", "tooltip": "The clip's own "
                               "sound, or silence."}),
            },
            "optional": {"audio": ("AUDIO", {"tooltip": "The clip's sound (VHS Load Video's audio)."})},
        }

    RETURN_TYPES = ("MMH3_CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3"

    def prepare(self, images, vae, audio_vae, width, height, frames=0, audio_mode="clip", audio=None):
        h3, _ = _native()
        if width % 32 or height % 32:
            raise ValueError("width and height must be multiples of 32")
        have = int(images.shape[0])
        want = min(have, int(frames)) if frames else have
        n = planner.nearest_valid(want)
        while n > want and n > 5:
            n -= planner.GROUP_FRAMES
        if n < 5 or n > have:
            raise ValueError(f"The clip has {have} frames; it needs at least 5 at 24 fps.")
        pixels = h3._resize(images[:n], width, height, "center")
        video = vae.encode(pixels)
        if video.shape[2] != planner.video_tokens(n):
            raise RuntimeError(f"The video VAE returned {video.shape[2]} tokens for {n} frames; "
                               f"expected {planner.video_tokens(n)}.")
        need = planner.audio_at(n)
        sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if audio_mode == "clip" and audio is not None:
            a, _ = h3._encode_ref_audio(audio_vae, audio)
        else:
            a = None
        if a is None or a.shape[-1] < need - 2:
            # silence for a muted clip, and to pad a clip whose sound is short
            silent, _ = h3._encode_ref_audio(audio_vae, {
                "waveform": torch.zeros(1, 2, int(round(n / planner.FPS * sr)) + sr // 10),
                "sample_rate": sr})
            a = silent if a is None else torch.cat([a, silent[..., a.shape[-1]:]], -1)
        audio_lat = _fit_audio(a[..., :need + 2], need) if a.shape[-1] >= need else _fit_audio(a, need)
        clip = {"video": video.detach().cpu(), "audio": audio_lat.detach().cpu(), "frames": n,
                "width": width, "height": height,
                "pixels": pixels.detach().cpu().to(torch.float16)}
        src_audio = audio["waveform"] if (audio_mode == "clip" and audio is not None) else None
        clip["digest"] = _digest(["clip", clip["pixels"], src_audio, audio_mode, n, width, height])
        return (clip,)


class MiniMaxH3ClipShot:
    """Put a prepared clip into the Shot chain, like a Shot. It is never
    sampled; the Shots next to it lead into it or out of it ('bridge'), or
    meet it with a hard cut ('cut')."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("MMH3_CLIP",),
                "shot_id": ("STRING", {"default": ""}),
                "join": (["bridge", "cut"], {"default": "bridge", "tooltip": "How the clip joins "
                         "the piece before it."}),
            },
            "optional": {"shots": ("MMH3_SHOTS",)},
        }

    RETURN_TYPES = ("MMH3_SHOTS",)
    RETURN_NAMES = ("shots",)
    FUNCTION = "add"
    CATEGORY = "MiniMax H3"

    def add(self, clip, shot_id, join="bridge", shots=None):
        chain = list(shots) if shots else []
        chain.append({
            "kind": "clip",
            "id": str(shot_id).strip() or f"clip{len(chain) + 1}",
            "seconds": clip["frames"] / planner.FPS,
            "text": "",
            "cut_verb": "the camera cuts to",
            "seed": 0,
            "lock": None,
            "join": join,
            "frames": clip["frames"],
            "clip": clip,
        })
        return (chain,)


class MiniMaxH3ClipPixels:
    """Final video option: put each clip's original (resized) frames back in
    place of their decoded version. Sharper clips; a faint seam can show where
    a clip meets generated footage."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "VAE Decode of Long Shot's latent."}),
            "latent": ("LATENT", {"tooltip": "Long Shot's latent (it carries where the clips are)."}),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "swap"
    CATEGORY = "MiniMax H3"

    def swap(self, images, latent, enabled=True):
        spans = (latent or {}).get("mmh3_clips") or []
        if not enabled or not spans:
            return (images,)
        out = images.clone()
        for sp in spans:
            px = sp["pixels"][sp["offset"]:sp["offset"] + sp["frames"]]
            a = sp["start"]
            b = min(out.shape[0], a + px.shape[0])
            if b <= a or px.shape[1:3] != out.shape[1:3]:
                continue
            out[a:b] = px[:b - a].to(out)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SongTrack": MiniMaxH3SongTrack,
    "MiniMaxH3RefModCarrier": MiniMaxH3RefModCarrier,
    "MiniMaxH3TimelineShot": MiniMaxH3TimelineShot,
    "MiniMaxH3Clip": MiniMaxH3Clip,
    "MiniMaxH3ClipShot": MiniMaxH3ClipShot,
    "MiniMaxH3ClipPixels": MiniMaxH3ClipPixels,
}
if MiniMaxH3LongShot is not None:
    NODE_CLASS_MAPPINGS["MiniMaxH3LongShot"] = MiniMaxH3LongShot
else:
    logger.error("MiniMax H3 Long Shot needs a newer ComfyUI (V3 node API). Update ComfyUI.")
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LongShot": "MiniMax H3 Long Shot",
    "MiniMaxH3SongTrack": "MiniMax H3 Song Track",
    "MiniMaxH3RefModCarrier": "MiniMax H3 RefMod Carrier",
    "MiniMaxH3TimelineShot": "MiniMax H3 Timeline Shot",
    "MiniMaxH3Clip": "MiniMax H3 Clip",
    "MiniMaxH3ClipShot": "MiniMax H3 Clip Shot",
    "MiniMaxH3ClipPixels": "MiniMax H3 Clip Pixels",
}
