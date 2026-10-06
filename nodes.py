"""MiniMax H3 Long Shot — a chain of Shot nodes rendered as one continuous shot,
with optional lip-sync to a song."""

from __future__ import annotations

import inspect
import logging

import torch

from . import planner

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


class _RefEncoder:
    """Per-segment conditioning through ComfyUI's own H3 nodes.

    References go to the native Reference to Video node exactly as connected,
    so <Picture N> / <Video N> / <Audio N> numbering matches the native node.

    The text encoder has to run per segment (each has its own prompt), but the
    VAE encodes of the references don't: they depend only on the references and
    the window length. Those blocks are encoded once per window length and
    reused, which saves re-encoding every reference video on every segment."""

    def __init__(self, clip, vae, audio_vae, width, height, refs, ref_image_size):
        self.clip, self.vae, self.audio_vae = clip, vae, audio_vae
        self.width, self.height = width, height
        self.refs = refs
        self.ref_image_size = ref_image_size
        self._blocks = {}   # window_frames -> minimax_refs blocks

    @property
    def has_refs(self):
        return any(self.refs.values())

    def encode(self, prompt, frames, first_frame=None, last_frame=None):
        h3, _ = _native()
        if not self.has_refs:
            out = h3.MiniMaxH3ImageToVideo.execute(
                self.clip, self.vae, prompt, self.width, self.height, frames,
                first_frame=first_frame, last_frame=last_frame)
            return _unwrap(out)[0]

        cached = self._blocks.get(frames)
        out = h3.MiniMaxH3ReferenceToVideo.execute(
            self.clip, prompt, self.width, self.height, frames,
            ref_image_size=self.ref_image_size,
            vae=None if cached is not None else self.vae,
            audio_vae=None if cached is not None else self.audio_vae,
            **self.refs)
        positive = _unwrap(out)[0]
        if cached is None:
            self._blocks[frames] = list(positive[0][1].get("minimax_refs", []) or [])
            return positive
        import node_helpers
        return node_helpers.conditioning_set_values(positive, {"minimax_refs": list(cached)})


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


def _segment_noise(noise, seed_mode, index):
    """Only RandomNoise is stepped — stepping DisableNoise's seed would
    silently turn it into real noise."""
    _, cs = _native()
    if seed_mode == "increment" and isinstance(noise, cs.Noise_RandomNoise):
        return cs.Noise_RandomNoise(noise.seed + index - 1)
    return noise


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


def run_long_shot(model, clip, vae, noise, sampler, sigmas, prompt, width, height,
                  overlap_frames, seed_mode, dry_run, ref_image_size="match", audio_vae=None,
                  song=None, first_frame=None, last_frame=None, extra_refs=None,
                  ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
    """The whole Long Shot run. Returns (latent, total_frames, total_seconds, plan)."""
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
    if native_refs and (first_frame is not None or last_frame is not None):
        raise ValueError(
            "first_frame / last_frame use H3's image-to-video path and can't be combined "
            "with reference inputs. Connect one or the other.")
    if (refs["ref_audios"] or refs["ref_video_audios"]) and audio_vae is None:
        raise ValueError("Reference audio is connected — connect the H3 audio VAE to audio_vae.")
    for name in refs["ref_video_audios"]:
        if "ref_video_" + name.rsplit("_", 1)[-1] not in refs["ref_videos"]:
            raise ValueError(
                f"{name} has no matching {'ref_video_' + name.rsplit('_', 1)[-1]}. A video "
                f"soundtrack only counts when its video is connected too.")

    plan = planner.plan_from_durations([float(s.get("seconds", 0)) for s in shots],
                                       overlap_frames)
    if song is not None:
        need, have = plan.total_audio_tokens, song["latent"].shape[-1]
        if have < need:
            raise ValueError(
                f"The song covers {have / planner.AUDIO_LATENT_FPS:.2f}s but the shots add up to "
                f"{planner.seconds(plan.total_frames):.2f}s. Shorten the shots or give Song "
                f"Track a longer duration.")

    extra_blocks = _extra_ref_blocks(extra_refs)
    prompts = planner.build_prompts(
        shots, plan, bundle, reference_mode=native_refs or bool(extra_blocks),
        audio_reuse=song is not None,
        use_first_frame=first_frame is not None, use_last_frame=last_frame is not None)
    seeds = None
    if hasattr(noise, "seed"):
        step = 1 if seed_mode == "increment" else 0
        seeds = [noise.seed + step * (s.index - 1) for s in plan.segments]
    report = planner.render_plan(plan, prompts, seeds,
                                 song_offset=song["offset"] if song is not None else None)
    # Console gets the short version; the full plan, prompts included, is the
    # plan output, for a text preview node.
    logger.info("\n%s", planner.render_plan(
        plan, prompts, seeds, song_offset=song["offset"] if song is not None else None,
        include_prompts=False))
    total_s = planner.seconds(plan.total_frames)

    if dry_run:
        # Block the latent so nothing downstream of it runs — no decode, no saved
        # clip — while plan, total_frames and total_seconds still go through.
        from comfy_execution.graph_utils import ExecutionBlocker
        return (ExecutionBlocker(None), plan.total_frames, total_s, report)

    _native()
    _require_arbitrary_guides()
    import comfy.model_management

    encoder = _RefEncoder(clip, vae, audio_vae, width, height, refs, ref_image_size)
    if extra_blocks:
        logger.info("MiniMax H3 Long Shot: adding %d external reference block(s) to every "
                    "segment", len(extra_blocks))

    cumulative = None
    for sp in prompts:
        seg = sp.segment
        comfy.model_management.throw_exception_if_processing_interrupted()
        logger.info("MiniMax H3 Long Shot: segment %d/%d, %d-frame window",
                    seg.index, len(prompts), seg.window_frames)

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
        sampled = _sample_window(model, _segment_noise(noise, seed_mode, seg.index),
                                 sampler, sigmas, positive, target)
        cumulative = ({"samples": sampled["samples"]} if cumulative is None
                      else _append(cumulative, sampled, seg))

    from comfy.nested_tensor import NestedTensor
    v, a = cumulative["samples"].tensors
    if v.shape[2] != plan.total_video_tokens or a.shape[-1] != plan.total_audio_tokens:
        raise RuntimeError(
            f"Stitched latent is {v.shape[2]} video / {a.shape[-1]} audio tokens; expected "
            f"{plan.total_video_tokens} / {plan.total_audio_tokens}.")
    if song is not None:
        # The song is the source of truth: output it exactly.
        a = song["latent"][..., :plan.total_audio_tokens].to(a)
        cumulative = {"samples": NestedTensor((v, a))}
    return (cumulative, plan.total_frames, total_s, report)


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
                    io.Model.Input("model"),
                    io.Clip.Input("clip"),
                    io.Vae.Input("vae", tooltip="H3 video VAE."),
                    io.Noise.Input("noise"),
                    io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"),
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
                    io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                                   tooltip="'max' holds identity better but is slower — paid on "
                                   "every segment."),
                    io.Vae.Input("audio_vae", optional=True,
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
                    io.Conditioning.Input(
                        "extra_refs", optional=True, tooltip="Reference blocks added by another "
                        "node, e.g. Apply H3 RefMod. Feed it a conditioning (a native Reference "
                        "to Video with an empty prompt works); its reference blocks are added to "
                        "every segment and its prompt text is ignored."),
                ] + _ref_inputs(io),
                outputs=[
                    io.Latent.Output(display_name="latent"),
                    io.Int.Output(display_name="total_frames"),
                    io.Float.Output(display_name="total_seconds"),
                    io.String.Output(display_name="plan"),
                ],
            )

        @classmethod
        def execute(cls, **kwargs) -> io.NodeOutput:
            return io.NodeOutput(*run_long_shot(**kwargs))

    return MiniMaxH3LongShot


try:
    MiniMaxH3LongShot = _make_long_shot_node()
except ImportError:   # ComfyUI too old for the V3 node API
    MiniMaxH3LongShot = None


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SongTrack": MiniMaxH3SongTrack,
    "MiniMaxH3RefModCarrier": MiniMaxH3RefModCarrier,
}
if MiniMaxH3LongShot is not None:
    NODE_CLASS_MAPPINGS["MiniMaxH3LongShot"] = MiniMaxH3LongShot
else:
    logger.error("MiniMax H3 Long Shot needs a newer ComfyUI (V3 node API). Update ComfyUI.")
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LongShot": "MiniMax H3 Long Shot",
    "MiniMaxH3SongTrack": "MiniMax H3 Song Track",
    "MiniMaxH3RefModCarrier": "MiniMax H3 RefMod Carrier",
}
