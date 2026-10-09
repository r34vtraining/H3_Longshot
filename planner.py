"""Segment planning for MiniMax H3 Long Shot.

Pure Python — no torch, no ComfyUI — so every rule is unit-testable.

H3 temporal facts (verified against ComfyUI's comfy/ldm/minimax/model.py and
comfy_extras/nodes_minimax_h3.py):

* Video runs at 24 fps and frame counts must satisfy 17k + 5.
* Video tokens follow FRAME_PER_TOKEN = (1, 4, 4, 4, 4): each group of five
  tokens covers 17 frames and opens with a 1-frame token, so a 17k + 5 frame
  latent has 5k + 2 tokens.
* Audio is a 40 Hz latent stream on the same timeline; its index at a frame
  boundary is round(frame * 40 / 24).

Each Shot node in the chain is one segment and one generation. Continuation
windows open with a hidden overlap pinned by a guide copied from the tail of
everything generated so far; the overlap is dropped after sampling and only
new tokens are appended. With a 17k + 5 overlap the tail slice starts on a
17-frame group boundary, so one causal group straddles each join and a
single decode of the stitched latent sees one continuous sequence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FPS = 24
AUDIO_LATENT_FPS = 40
GROUP_FRAMES = 17
TRAINED_MIN_FRAMES = 124   # ComfyUI: "trained range is ~124-362"
TRAINED_MAX_FRAMES = 362
MAX_SEGMENTS = 16


# ---------------------------------------------------------------------------
# Grid math
# ---------------------------------------------------------------------------

def is_valid_frame_count(frames: int) -> bool:
    return frames >= 5 and frames % GROUP_FRAMES == 5


def align_up(frames: int) -> int:
    """Smallest 17k + 5 >= frames — mirrors ComfyUI's align_frame_count."""
    frames = max(5, int(frames))
    while frames % GROUP_FRAMES != 5:
        frames += 1
    return frames


def nearest_valid(frames: float) -> int:
    """Closest 17k + 5 to a requested frame count."""
    k = max(0, round((frames - 5) / GROUP_FRAMES))
    return GROUP_FRAMES * k + 5


def video_tokens(frames: int) -> int:
    if not is_valid_frame_count(frames):
        raise ValueError(f"{frames} frames is not on the 17k + 5 grid")
    return ((frames - 5) // GROUP_FRAMES) * 5 + 2


def audio_at(frame_index: int) -> int:
    """Cumulative 40 Hz audio index at a 24 fps frame boundary. Always taken
    from the global position so rounding can't drift over a long chain."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return round(frame_index * AUDIO_LATENT_FPS / FPS)


def seconds(frames: int) -> float:
    return frames / FPS


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    return f"{m:02d}:{t - m * 60:04.1f}"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    index: int                 # 1-based
    requested_seconds: float
    window_frames: int         # this generation's length (17k + 5)
    overlap_frames: int        # hidden leading context (0 for segment 1)
    new_frames: int            # frames this segment adds to the result
    visible_start: int         # global frame where its new content begins
    window_start: int          # global frame of the window's first frame
    overlap_video_tokens: int
    overlap_audio_tokens: int
    new_video_tokens: int
    new_audio_tokens: int
    window_audio_tokens: int

    @property
    def visible_end(self) -> int:
        return self.visible_start + self.new_frames

    @property
    def window_end(self) -> int:
        return self.window_start + self.window_frames

    @property
    def window_audio_span(self) -> tuple[int, int]:
        """Global 40 Hz audio indices this window covers — the song slice."""
        return audio_at(self.window_start), audio_at(self.window_end)


@dataclass
class Plan:
    overlap_frames: int
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return self.segments[-1].visible_end

    @property
    def total_video_tokens(self) -> int:
        return video_tokens(self.total_frames)

    @property
    def total_audio_tokens(self) -> int:
        return audio_at(self.total_frames)


def plan_from_durations(durations: list[float], overlap_frames: int) -> Plan:
    """One segment per requested duration, snapped to the H3 grid.

    Segment 1 becomes its own 17k + 5 clip. Every later segment adds a
    multiple of 17 new frames on top of a 17k + 5 hidden overlap, so each
    window stays on the grid and each tail slice lands on a group boundary.
    """
    if not durations:
        raise ValueError("Connect at least one Shot node — one per segment.")
    if len(durations) > MAX_SEGMENTS:
        raise ValueError(f"At most {MAX_SEGMENTS} segments are supported.")
    if not is_valid_frame_count(overlap_frames):
        raise ValueError(
            f"overlap_frames must be on the 17k + 5 grid (5, 22, 39, 56 ...), got {overlap_frames}"
        )
    for i, d in enumerate(durations, 1):
        if d <= 0:
            raise ValueError(f"Shot {i} has a duration of {d}s; it must be positive.")

    # Snap each segment's END on the global timeline, not its length. Snapping
    # lengths independently lets rounding accumulate — three 6.67s segments
    # would come out 0.67s short — which pulls a lip-synced video out of step
    # with its song. Snapping ends bounds total error to half a group.
    ends, t = [], 0.0
    for dur in durations:
        t += dur
        end = nearest_valid(t * FPS)
        if ends and end < ends[-1] + GROUP_FRAMES:
            end = ends[-1] + GROUP_FRAMES
        ends.append(end)

    segments: list[Segment] = []
    cursor = 0
    for i, (dur, end) in enumerate(zip(durations, ends), start=1):
        if i == 1:
            window = end
            if window < overlap_frames:
                raise ValueError(
                    f"Shot 1 is {seconds(window):.2f}s, shorter than the "
                    f"{overlap_frames}-frame overlap it has to supply. Lengthen it or "
                    f"reduce overlap_frames."
                )
            new, ov, ws = window, 0, 0
            ov_v = ov_a = 0
            new_v, new_a = video_tokens(window), audio_at(window)
            win_a = new_a
        else:
            new = end - cursor
            ov = overlap_frames
            window = ov + new
            ws = cursor - ov
            ov_v = video_tokens(ov)
            new_v = new // GROUP_FRAMES * 5
            ov_a = audio_at(cursor) - audio_at(cursor - ov)
            new_a = audio_at(cursor + new) - audio_at(cursor)
            win_a = ov_a + new_a
            if ov_v + new_v != video_tokens(window):
                raise AssertionError("window does not satisfy the H3 video grid")

        segments.append(Segment(
            index=i, requested_seconds=dur, window_frames=window, overlap_frames=ov,
            new_frames=new, visible_start=cursor, window_start=ws,
            overlap_video_tokens=ov_v, overlap_audio_tokens=ov_a,
            new_video_tokens=new_v, new_audio_tokens=new_a, window_audio_tokens=win_a,
        ))
        cursor += new

    plan = Plan(overlap_frames=overlap_frames, segments=segments)
    _window_warnings(plan)
    return plan


def _window_warnings(plan: Plan, skip=()) -> None:
    """Notes on windows outside H3's trained range. Locked takes and clips are
    skipped: they aren't sampled."""
    for s in plan.segments:
        if s.index in skip:
            continue
        span = f"{s.window_frames} frames ({seconds(s.window_frames):.1f}s)"
        if s.window_frames > TRAINED_MAX_FRAMES:
            plan.warnings.append(
                f"Segment {s.index} window is {span}, above H3's trained ~{TRAINED_MAX_FRAMES}. "
                f"Split that Shot in two."
            )
        elif s.window_frames < TRAINED_MIN_FRAMES:
            plan.warnings.append(
                f"Segment {s.index} window is {span}, below H3's trained ~{TRAINED_MIN_FRAMES}. "
                f"Consider merging it with a neighbour."
            )


# ---------------------------------------------------------------------------
# Timeline of pieces (round 2)
#
# Every piece's window is 17k + 5 frames. The first piece shows all of it;
# every later piece drops its first `overlap` frames (the shared zone, owned by
# the left piece's tail) and shows window - overlap = 17j frames. So any take
# fits anywhere: a take rendered as the first piece just drops its head when
# something is put before it, and a take rendered after another piece shows
# its head when it becomes first.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PieceSpec:
    requested_seconds: float
    window: int | None = None     # fixed window (a locked take, or keep-length re-render)
    locked: bool = False


def plan_pieces(pieces: list[PieceSpec], overlap_frames: int) -> Plan:
    """Lay out pieces with fixed or requested lengths.

    Requested lengths snap their END on the timeline like plan_from_durations,
    so rounding never accumulates; the running target restarts at every fixed
    piece, whose length is exact. With no fixed pieces this is exactly
    plan_from_durations."""
    if not pieces:
        raise ValueError("Connect at least one Shot node — one per segment.")
    if len(pieces) > MAX_SEGMENTS:
        raise ValueError(f"At most {MAX_SEGMENTS} segments are supported.")
    if not is_valid_frame_count(overlap_frames):
        raise ValueError(
            f"overlap_frames must be on the 17k + 5 grid (5, 22, 39, 56 ...), got {overlap_frames}")
    segments: list[Segment] = []
    cursor = 0
    ideal = 0.0                    # where requested lengths say we should be, in seconds
    for i, p in enumerate(pieces, start=1):
        first = i == 1
        if p.window is not None:
            w = int(p.window)
            if not is_valid_frame_count(w):
                raise ValueError(f"Shot {i}'s take is {w} frames, which is not on the 17k + 5 grid.")
            if first and w < overlap_frames:
                raise ValueError(
                    f"Shot 1 is {seconds(w):.2f}s, shorter than the {overlap_frames}-frame "
                    f"overlap it has to supply. Lengthen it or reduce overlap_frames.")
            if not first and w - overlap_frames < GROUP_FRAMES:
                raise ValueError(
                    f"Shot {i}'s take is {seconds(w):.2f}s; after the {overlap_frames}-frame "
                    f"shared zone it would show nothing. Re-render it longer.")
            new = w if first else w - overlap_frames
            end = cursor + new
            ideal = seconds(end)
        else:
            if p.requested_seconds <= 0:
                raise ValueError(f"Shot {i} has a duration of {p.requested_seconds}s; it must be positive.")
            ideal += p.requested_seconds
            end = nearest_valid(ideal * FPS)
            if first:
                if end < overlap_frames:
                    raise ValueError(
                        f"Shot 1 is {seconds(end):.2f}s, shorter than the {overlap_frames}-frame "
                        f"overlap it has to supply. Lengthen it or reduce overlap_frames.")
            elif end < cursor + GROUP_FRAMES:
                end = cursor + GROUP_FRAMES
            new = end - cursor
        if first:
            ov, ws, window = 0, 0, new
            ov_v = ov_a = 0
            new_v, new_a = video_tokens(window), audio_at(window)
            win_a = new_a
        else:
            ov, ws = overlap_frames, cursor - overlap_frames
            window = ov + new
            ov_v = video_tokens(ov)
            new_v = new // GROUP_FRAMES * 5
            ov_a = audio_at(cursor) - audio_at(cursor - ov)
            new_a = audio_at(cursor + new) - audio_at(cursor)
            win_a = ov_a + new_a
            if ov_v + new_v != video_tokens(window):
                raise AssertionError("window does not satisfy the H3 video grid")
        segments.append(Segment(
            index=i, requested_seconds=p.requested_seconds, window_frames=window,
            overlap_frames=ov, new_frames=new, visible_start=cursor, window_start=ws,
            overlap_video_tokens=ov_v, overlap_audio_tokens=ov_a,
            new_video_tokens=new_v, new_audio_tokens=new_a, window_audio_tokens=win_a))
        cursor += new
    plan = Plan(overlap_frames=overlap_frames, segments=segments)
    _window_warnings(plan, skip={i + 1 for i, p in enumerate(pieces) if p.locked})
    return plan


def tail_audio_tokens(seg: Segment, overlap_frames: int) -> int:
    """Audio tokens in this window's last `overlap_frames` frames, by global position."""
    return audio_at(seg.window_end) - audio_at(seg.window_end - overlap_frames)


def head_audio_tokens(seg: Segment, overlap_frames: int) -> int:
    """Audio tokens in this window's first `overlap_frames` frames, by global position."""
    return audio_at(seg.window_start + overlap_frames) - audio_at(seg.window_start)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def alignment_line(kind: str, window_frames: int) -> str:
    if kind == "i2va":
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    if kind == "l2va":
        return ("How the reference pictures align with the target video — "
                f"<Picture 1> (from [Shot 1]) aligns with the "
                f"{seconds(window_frames - 1):.2f}-second mark of the target video.")
    raise ValueError(kind)


def _na(text: str) -> str:
    text = (text or "").strip()
    return text if text else "N/A"


_SHOT_REF = re.compile(r"\[Shot\s+\d+\]", re.I)
_SHOT_ONE_RUN = re.compile(r"\[Shot 1\](?:\s*,\s*\[Shot 1\])+")


def as_single_shot(text: str) -> str:
    """Every segment is its own one-shot generation, so shot references in the
    shared sections all become [Shot 1]. '[Shot 1], [Shot 3]' collapses to one."""
    return _SHOT_ONE_RUN.sub("[Shot 1]", _SHOT_REF.sub("[Shot 1]", text or ""))


def summary_block(bundle: dict, audio_reuse: bool) -> str:
    text = as_single_shot((bundle.get("summary") or "").strip())
    types = bundle.get("task_types")
    if types is None:
        return text   # the user typed their own [prefix]; keep it verbatim
    types = list(types) or ["reference generation"]
    if audio_reuse and "audio reuse" not in types:
        types.append("audio reuse")
    return f"[{' + '.join(types)}] {text}".strip()


def compose_prompt(text: str, bundle: dict, *, reference_mode: bool = False,
                   audio_reuse: bool = False, alignment: str = "") -> str:
    """One segment's prompt from the shared bundle plus this Shot's text.

    Full-reference format puts the style line before [Shot 1] and carries the
    subject, summary and retention sections. With no references at all, the
    base format is used and only style, soundscape and music apply."""
    text = (text or "").strip()
    style = (bundle.get("style_line") or "").strip()
    soundscape = bundle.get("overall_soundscape", "")
    music = bundle.get("non_diegetic_music", "")

    if reference_mode:
        body = f"{style}\n[Shot 1] {text}" if style else f"[Shot 1] {text}"
        sections = []
        subjects = as_single_shot((bundle.get("subject_definitions") or "").strip())
        if subjects:
            sections.append(f"subject_definitions:\n{subjects}")
        sections.append(f"summary:\n{summary_block(bundle, audio_reuse)}")
        retention = as_single_shot((bundle.get("retention_analysis") or "").strip())
        if retention:
            sections.append(f"retention_analysis:\n{retention}")
        sections += [
            f"detailed_description:\n{body}",
            f"overall_soundscape:\n{_na(soundscape)}",
            f"non_diegetic_music:\n{_na(music)}",
        ]
        out = "\n\n".join(sections)
    else:
        body = " ".join(p for p in (style, text) if p)
        out = "\n\n".join([
            f"integrated_multimodal_description: [Shot 1] {body}",
            f"overall_soundscape: {_na(soundscape)}",
            f"non_diegetic_music: {_na(music)}",
        ])
    return f"{alignment}\n\n{out}" if alignment else out


@dataclass
class SegmentPrompt:
    segment: Segment
    prompt: str
    first_frame: bool
    last_frame: bool


def build_prompts(shots: list[dict], plan: Plan, bundle: dict, *, reference_mode=False,
                  audio_reuse=False, use_first_frame=False,
                  use_last_frame=False) -> list[SegmentPrompt]:
    out = []
    last = len(plan.segments)
    for seg, shot in zip(plan.segments, shots):
        first = use_first_frame and seg.index == 1
        final = use_last_frame and seg.index == last
        align = ""
        if not reference_mode:
            if first:
                align = alignment_line("i2va", seg.window_frames)
            elif final:
                align = alignment_line("l2va", seg.window_frames)
        prompt = compose_prompt(shot.get("text", ""), bundle, reference_mode=reference_mode,
                                audio_reuse=audio_reuse, alignment=align)
        out.append(SegmentPrompt(seg, prompt, first, final))
    return out


# ---------------------------------------------------------------------------
# Human-readable plan
# ---------------------------------------------------------------------------

def render_plan(plan: Plan, prompts: list[SegmentPrompt], seeds: list[int] | None = None,
                song_offset: float | None = None, include_prompts: bool = True,
                statuses: list[str] | None = None, own_seeds: list[bool] | None = None) -> str:
    """The plan as text. include_prompts=False gives the short console version:
    one line per segment plus warnings, without the prompt bodies."""
    total = seconds(plan.total_frames)
    lines = [
        f"MiniMax H3 Long Shot — {len(plan.segments)} segments, "
        f"{plan.total_frames} frames ({total:.2f}s)",
        f"overlap {plan.overlap_frames} frames"
        + (" · lip-sync song connected" if song_offset is not None else ""),
        "",
    ]
    for sp in prompts:
        s = sp.segment
        v0, v1 = seconds(s.visible_start), seconds(s.visible_end)
        head = (f"Segment {s.index}: asked {s.requested_seconds:g}s, got "
                f"{seconds(s.new_frames):.2f}s · shows {fmt_time(v0)}–{fmt_time(v1)} · "
                f"window {s.window_frames}f")
        if s.overlap_frames:
            head += f" ({s.overlap_frames}f hidden overlap)"
        if seeds is not None:
            head += f" · seed {seeds[s.index - 1]}"
            if own_seeds and own_seeds[s.index - 1]:
                head += " (Shot seed)"
        extras = [x for x, on in (("first frame", sp.first_frame),
                                  ("last frame", sp.last_frame)) if on]
        if extras:
            head += " · " + " + ".join(extras)
        if statuses is not None:
            head += f" · {statuses[s.index - 1]}"
        lines.append(head)
        if song_offset is not None:
            w0 = song_offset + seconds(s.window_start)
            w1 = song_offset + seconds(s.window_end)
            lines.append(f"  lyrics for this Shot: song {fmt_time(w0)}–{fmt_time(w1)}"
                         + (f" (includes {seconds(s.overlap_frames):.2f}s overlap)"
                            if s.overlap_frames else ""))
        if include_prompts:
            lines += ["-" * len(head), sp.prompt, ""]
    for w in plan.warnings:
        lines.append(f"NOTE: {w}")
    return "\n".join(lines).rstrip()
