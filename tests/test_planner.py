"""Planner tests. Grid math is cross-checked against ComfyUI's own H3 code."""
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)
PKG = os.path.basename(PKG_DIR)
# Import as a package, the way ComfyUI loads custom nodes. Putting the package
# folder itself on sys.path would shadow ComfyUI's own nodes.py.
sys.path.insert(0, os.path.dirname(PKG_DIR))
sys.path.insert(0, HERE)

import importlib  # noqa: E402

P = importlib.import_module(PKG + ".planner")


@pytest.fixture(scope="module")
def upstream():
    import comfy_env
    comfy_env.boot()
    import comfy_extras.nodes_minimax_h3 as h3
    return h3


def test_align_matches_upstream(upstream):
    for n in range(1, 800):
        assert P.align_up(n) == upstream.align_frame_count(max(5, n))


def test_tokens_and_audio_match_upstream(upstream):
    for n in range(5, 800, 17):
        frames, vt, at = upstream.temporal_shape(n)
        assert P.video_tokens(frames) == vt and P.audio_at(frames) == at


def _check(plan):
    cursor = 0
    for s in plan.segments:
        assert P.is_valid_frame_count(s.window_frames)
        assert s.visible_start == cursor
        cursor += s.new_frames
    segs = plan.segments
    assert P.is_valid_frame_count(plan.total_frames)
    assert sum(s.new_video_tokens for s in segs) == plan.total_video_tokens
    assert sum(s.new_audio_tokens for s in segs) == plan.total_audio_tokens, "audio drift"
    for s in segs[1:]:
        assert s.window_start % P.GROUP_FRAMES == 0, "tail slice off a group boundary"
        assert s.overlap_video_tokens + s.new_video_tokens == P.video_tokens(s.window_frames)
        a0, a1 = s.window_audio_span
        assert a1 - a0 == s.window_audio_tokens, "song slice length != window audio"


def test_exhaustive_random_chains_hold_invariants():
    rng = random.Random(0)
    checked = 0
    for _ in range(6000):
        n = rng.randint(2, 10)
        durs = [round(rng.uniform(1.5, 16), 1) for _ in range(n)]
        ov = rng.choice([5, 22, 39, 56])
        try:
            plan = P.plan_from_durations(durs, ov)
        except ValueError:
            continue
        _check(plan)
        checked += 1
    assert checked > 5000


HALF_GROUP = P.GROUP_FRAMES / P.FPS / 2 + 1e-9


def test_every_boundary_stays_within_half_a_group_of_the_song():
    """Lip sync depends on this: rounding must not accumulate along the chain."""
    rng = random.Random(1)
    for _ in range(3000):
        durs = [round(rng.uniform(2, 12), 2) for _ in range(rng.randint(2, 16))]
        try:
            plan = P.plan_from_durations(durs, 22)
        except ValueError:
            continue
        t = 0.0
        for s, d in zip(plan.segments, durs):
            t += d
            assert abs(P.seconds(s.visible_end) - t) <= HALF_GROUP, (durs, s.index)


def test_even_20s_in_three():
    plan = P.plan_from_durations([20 / 3] * 3, 22)
    assert abs(P.seconds(plan.total_frames) - 20) <= HALF_GROUP
    assert [s.new_frames for s in plan.segments] == [158, 170, 153]


def test_many_short_segments_do_not_drift():
    plan = P.plan_from_durations([1.3] * 16, 5)
    assert abs(P.seconds(plan.total_frames) - 1.3 * 16) <= HALF_GROUP


def test_trained_range_warnings():
    assert any("above" in w for w in P.plan_from_durations([6, 16], 22).warnings)
    assert any("below" in w for w in P.plan_from_durations([6, 2], 22).warnings)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least one"):
        P.plan_from_durations([], 22)
    with pytest.raises(ValueError, match="17k"):
        P.plan_from_durations([6, 6], 20)
    with pytest.raises(ValueError, match="shorter than"):
        P.plan_from_durations([0.5, 6], 39)


def bundle(**kw):
    """The shape MiniMax H3 Ref Prompt Builder r2v emits on its long_shot output."""
    b = {"version": 1, "subject_definitions": "", "summary": "", "task_types": [
        "reference generation"], "retention_analysis": "", "style_line": "",
        "overall_soundscape": "", "non_diegetic_music": "", "shots": []}
    b.update(kw)
    return b


def test_base_prompt_format():
    p = P.compose_prompt("Luma skips.", bundle(style_line="Soft 3D CG.",
                                               overall_soundscape="Birdsong."))
    assert p == ("integrated_multimodal_description: [Shot 1] Soft 3D CG. Luma skips.\n\n"
                 "overall_soundscape: Birdsong.\n\nnon_diegetic_music: N/A")


def test_base_format_drops_reference_only_sections():
    p = P.compose_prompt("x", bundle(subject_definitions="<Luma> is gold.", summary="s",
                                     retention_analysis="r"))
    assert "subject_definitions" not in p and "summary" not in p and "retention" not in p


def test_reference_prompt_format():
    p = P.compose_prompt("Luma skips.", bundle(
        style_line="Soft 3D CG.", subject_definitions="<Luma> is the golden creature.",
        summary="The target video shows <Luma> skipping.",
        retention_analysis="<Luma> (appears in [Shot 1]): fully_preserved - fur retained."),
        reference_mode=True, audio_reuse=True)
    assert p.startswith("subject_definitions:\n<Luma> is the golden creature.")
    assert ("summary:\n[reference generation + audio reuse] The target video shows "
            "<Luma> skipping.") in p
    assert "retention_analysis:\n<Luma> (appears in [Shot 1]): fully_preserved" in p
    assert "detailed_description:\nSoft 3D CG.\n[Shot 1] Luma skips." in p
    order = [p.index(k) for k in ("subject_definitions:", "summary:", "retention_analysis:",
                                  "detailed_description:", "overall_soundscape:",
                                  "non_diegetic_music:")]
    assert order == sorted(order), "sections out of spec order"


def test_task_types_follow_the_builder_and_add_audio_reuse_once():
    b = bundle(task_types=["reference generation", "audio reference"])
    p = P.compose_prompt("x", b, reference_mode=True, audio_reuse=True)
    assert "summary:\n[reference generation + audio reference + audio reuse]" in p
    b = bundle(task_types=["audio reuse"])
    assert "[audio reuse]" in P.compose_prompt("x", b, reference_mode=True, audio_reuse=True)


def test_custom_summary_prefix_is_kept_verbatim():
    b = bundle(summary="[video editing] My own prefix.", task_types=None)
    p = P.compose_prompt("x", b, reference_mode=True, audio_reuse=True)
    assert "summary:\n[video editing] My own prefix." in p


def test_shot_references_collapse_to_shot_one():
    b = bundle(retention_analysis="<Luma> (appears in [Shot 1], [Shot 3]): fully_preserved - x.",
               summary="Shown across [Shot 2] and [Shot 4].")
    p = P.compose_prompt("x", b, reference_mode=True)
    assert "(appears in [Shot 1]):" in p
    assert "Shot 3" not in p and "Shot 2" not in p and "Shot 4" not in p


def test_alignment_lines_land_on_first_and_last():
    plan = P.plan_from_durations([7, 7, 7], 22)
    shots = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    ps = P.build_prompts(shots, plan, bundle(), use_first_frame=True, use_last_frame=True)
    assert ps[0].prompt.startswith("For the target video")
    assert "For the target" not in ps[1].prompt and "How the reference" not in ps[1].prompt
    last = plan.segments[-1]
    assert f"{P.seconds(last.window_frames - 1):.2f}-second mark" in ps[2].prompt


def test_render_plan_shows_song_ranges():
    plan = P.plan_from_durations([7, 7, 7], 22)
    ps = P.build_prompts([{"text": x} for x in "abc"], plan, bundle())
    text = P.render_plan(plan, ps, seeds=[1, 2, 3], song_offset=42.0)
    assert "song 00:42.0" in text and "overlap)" in text and "seed 3" in text


def test_single_shot_is_one_plain_generation():
    plan = P.plan_from_durations([6.5], 22)
    (seg,) = plan.segments
    assert seg.overlap_frames == 0 and seg.window_start == 0
    assert seg.window_frames == plan.total_frames == P.nearest_valid(6.5 * P.FPS)
    assert seg.new_video_tokens == plan.total_video_tokens
    assert seg.new_audio_tokens == plan.total_audio_tokens
