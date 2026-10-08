"""Integration tests: Long Shot and Song Track against ComfyUI's real H3 code.

The 32B model can't run here, so sampling is simulated as a "perfect
continuation": it returns the matching slice of a random ground-truth
latent after asserting every guide it was handed is exactly right. Any
off-by-one in slicing, overlap trimming, audio boundaries, song slicing, or
append order breaks bit-exact equality with ground truth.

Everything else is upstream: the native H3 conditioning nodes, the audio
encode helper, the PackedLayout the model builds, real noise generation, and
the real SamplerCustomAdvanced wrapper.
"""
import importlib
import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)
PKG = os.path.basename(PKG_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(PKG_DIR))

import comfy_env  # noqa: E402

comfy_env.boot()

import torch  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402
from comfy.ldm.minimax.model import PackedLayout  # noqa: E402
import comfy.latent_formats as latent_formats  # noqa: E402
import comfy_extras.nodes_custom_sampler as cs  # noqa: E402

nodes = importlib.import_module(PKG + ".nodes")
P = importlib.import_module(PKG + ".planner")

W, H = 256, 128
SR = 32000   # mock audio VAE rate, so the native encoder never resamples


class MockClip:
    def __init__(self):
        self.prompts = []

    def tokenize(self, prompt, images=None, minimax_ref_items=None):
        self.prompts.append(prompt)
        return {}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 7, 16), {}]]


class MockVae:
    def encode(self, pixels):
        return torch.zeros(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)


class MockAudioVae:
    """Encodes to 40 Hz tokens whose value IS their time index, so any slice
    reveals exactly which part of the song it came from."""
    audio_sample_rate = SR

    def encode(self, wave):              # [1, L, C] as the native helper passes it
        t = wave.shape[1] * 40 // SR
        idx = torch.arange(t, dtype=torch.float32)
        return idx.view(1, 1, 1, t).expand(1, 32, 2, t).clone()


def shots_of(durations, texts=None):
    texts = texts or [f"beat {i}" for i in range(1, len(durations) + 1)]
    return [{"seconds": d, "text": t, "cut_verb": "the camera cuts to"}
            for d, t in zip(durations, texts)]


def make_song(seconds):
    wave = torch.zeros(1, 2, int(round(seconds * SR)))
    (song,) = nodes.MiniMaxH3SongTrack().load({"waveform": wave, "sample_rate": SR},
                                              MockAudioVae())
    return song


def make_simulator(plan, song=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt_v = torch.randn(1, 24, plan.total_video_tokens, H // 16, W // 16, generator=g)
    gt_a = torch.randn(1, 32, 2, plan.total_audio_tokens, generator=g)
    st = SimpleNamespace(v=0, a=0, calls=[], seg=0)

    def sample(model, noise, sampler, sigmas, positive, latent):
        seg = plan.segments[st.seg]
        st.seg += 1
        tv, ta = latent["samples"].tensors
        wv, wa = tv.shape[2], ta.shape[-1]
        meta = positive[0][1]
        kfs = meta.get("minimax_keyframes", [])
        st.calls.append(SimpleNamespace(seed=getattr(noise, "seed", None), meta=meta))
        head = [k for k in kfs if k["resolved_frame_index"] == 0]

        if song is not None:
            # every window must carry exactly its own slice of the song
            a0, a1 = seg.window_audio_span
            audio_guides = [k["audio_latent"] for k in head if k.get("audio_latent") is not None]
            assert len(audio_guides) == 1, "exactly one audio guide per window"
            assert torch.equal(audio_guides[0], song["latent"][..., a0:a1]), \
                f"segment {seg.index} got the wrong part of the song"

        if st.v == 0:
            assert not any(k.get("latent") is not None and k.get("audio_latent") is not None
                           and k is not head[0] for k in head[1:])
            start_v, start_a = 0, 0
        else:
            vids = [k for k in head if k.get("latent") is not None]
            assert len(vids) == 1, "continuation needs exactly one video tail guide"
            gv = vids[0]["latent"]
            ovt = gv.shape[2]
            assert torch.equal(gv, gt_v[:, :, st.v - ovt:st.v]), "video tail guide != tail"
            oat = seg.overlap_audio_tokens
            if song is None:
                assert torch.equal(vids[0]["audio_latent"], gt_a[..., st.a - oat:st.a]), \
                    "audio tail guide != tail"
            start_v, start_a = st.v - ovt, st.a - oat
        assert start_v + wv <= plan.total_video_tokens and start_a + wa <= plan.total_audio_tokens
        out = NestedTensor((gt_v[:, :, start_v:start_v + wv].clone(),
                            gt_a[..., start_a:start_a + wa].clone()))
        st.v, st.a = start_v + wv, start_a + wa
        return {"samples": out}

    return sample, gt_v, gt_a, st


BUNDLE_KEYS = ("subject_definitions", "summary", "task_types", "retention_analysis",
               "style_line", "overall_soundscape", "non_diegetic_music")


def make_bundle(durations, texts=None, **fields):
    """What MiniMax H3 Ref Prompt Builder r2v emits on its long_shot output."""
    b = {"version": 1, "subject_definitions": "", "summary": "",
         "task_types": ["reference generation"], "retention_analysis": "", "style_line": "",
         "overall_soundscape": "", "non_diegetic_music": "",
         "shots": shots_of(durations, texts) if durations else []}
    b.update(fields)
    return b


def call_node(**kwargs):
    """Through the real V3 node class, as ComfyUI would call it."""
    return nodes.MiniMaxH3LongShot.execute(**kwargs).args


def node_kwargs(durations, overlap=22, noise=None, texts=None, seed_mode="increment",
                dry_run=False, clip=None, vae=None, **extra):
    fields = {k: extra.pop(k) for k in list(extra) if k in BUNDLE_KEYS}
    return dict(model=None, clip=clip or MockClip(), vae=vae or MockVae(),
                noise=noise or cs.Noise_RandomNoise(100), sampler=None, sigmas=None,
                prompt=make_bundle(durations, texts, **fields), width=W, height=H,
                overlap_frames=overlap, seed_mode=seed_mode, dry_run=dry_run, **extra)


def run(monkeypatch, durations, overlap=22, song=None, seed_mode="increment",
        dry_run=False, noise=None, texts=None, **extra):
    plan = P.plan_from_durations(durations, overlap)
    sim, gt_v, gt_a, st = make_simulator(plan, song)
    monkeypatch.setattr(nodes, "_sample_window", sim)
    clip = extra.pop("clip", None) or MockClip()
    out = call_node(**node_kwargs(durations, overlap, noise, texts, seed_mode, dry_run,
                                  clip=clip, song=song, **extra))
    return SimpleNamespace(out=out, plan=plan, gt_v=gt_v, gt_a=gt_a, st=st, clip=clip)


# ---------------------------------------------------------------------------
# 1. Stitching is bit-exact
# ---------------------------------------------------------------------------

CHAINS = [[7, 7, 7], [6.5, 5, 8, 4.5], [10, 3], [4, 4, 4, 4, 4, 4], [20 / 3] * 3, [12, 9.5]]


@pytest.mark.parametrize("durations", CHAINS)
@pytest.mark.parametrize("overlap", [5, 22, 39])
def test_stitch_is_bit_exact(monkeypatch, durations, overlap):
    try:
        P.plan_from_durations(durations, overlap)
    except ValueError:
        pytest.skip("not splittable")
    r = run(monkeypatch, durations, overlap)
    v, a = r.out[0]["samples"].tensors
    assert torch.equal(v, r.gt_v) and torch.equal(a, r.gt_a)
    assert r.out[1] == r.plan.total_frames
    assert len(r.st.calls) == len(durations)


# ---------------------------------------------------------------------------
# 2. Lip sync
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("durations", CHAINS)
@pytest.mark.parametrize("overlap", [5, 22, 39])
def test_each_segment_gets_its_exact_song_slice(monkeypatch, durations, overlap):
    try:
        plan = P.plan_from_durations(durations, overlap)
    except ValueError:
        pytest.skip("not splittable")
    song = make_song(P.seconds(plan.total_frames) + 2)
    r = run(monkeypatch, durations, overlap, song=song)   # simulator asserts every slice
    v, a = r.out[0]["samples"].tensors
    assert torch.equal(v, r.gt_v), "video stitch broke with a song connected"
    # output audio is exactly the song, token for token
    assert torch.equal(a, song["latent"][..., :r.plan.total_audio_tokens])


def test_song_track_takes_audio_as_is(monkeypatch):
    """Trimming is the loader's job; Song Track encodes exactly what it's given."""
    info = nodes.MiniMaxH3SongTrack
    assert list(info.INPUT_TYPES()["required"]) == ["audio", "audio_vae"]
    assert info.RETURN_NAMES == ("song",)
    (song,) = info().load({"waveform": torch.zeros(1, 2, 25 * SR), "sample_rate": SR},
                          MockAudioVae())
    assert song["seconds"] == pytest.approx(25.0) and song["latent"].shape[-1] == 25 * 40
    r = run(monkeypatch, [7, 7, 7], song=song)
    assert "lyrics for this Shot: song 00:00.0" in r.out[3], "ranges timed from the clip start"


def test_song_too_short_is_a_clear_error(monkeypatch):
    with pytest.raises(ValueError, match="song covers"):
        run(monkeypatch, [7, 7, 7], song=make_song(10))


def test_song_with_first_frame_shares_one_frame_zero_guide(monkeypatch):
    img = torch.zeros(1, H, W, 3)
    r = run(monkeypatch, [7, 7, 7], song=make_song(25), first_frame=img)
    seg1 = r.st.calls[0].meta["minimax_keyframes"]
    assert len(seg1) == 1, "first frame and song should merge into one frame-0 guide"
    assert seg1[0].get("latent") is not None and seg1[0].get("audio_latent") is not None


# ---------------------------------------------------------------------------
# 3. Prompts, seeds, frames, refs
# ---------------------------------------------------------------------------

def test_each_shot_becomes_its_own_prompt(monkeypatch):
    r = run(monkeypatch, [7, 7, 7], texts=["walks.", "skips.", "floats."],
            style_line="Soft 3D CG.")
    assert [p.split("\n\n")[0] for p in r.clip.prompts] == [
        "integrated_multimodal_description: [Shot 1] Soft 3D CG. walks.",
        "integrated_multimodal_description: [Shot 1] Soft 3D CG. skips.",
        "integrated_multimodal_description: [Shot 1] Soft 3D CG. floats."]


def test_seeds(monkeypatch):
    assert [c.seed for c in run(monkeypatch, [7, 7, 7]).st.calls] == [100, 101, 102]
    assert [c.seed for c in run(monkeypatch, [7, 7, 7], seed_mode="same").st.calls] == [100] * 3


def test_disabled_noise_stays_disabled(monkeypatch):
    seen = []
    plan = P.plan_from_durations([7, 7, 7], 22)
    sim, *_ = make_simulator(plan)
    monkeypatch.setattr(nodes, "_sample_window",
                        lambda m, n, *a: (seen.append(type(n)), sim(m, n, *a))[1])
    call_node(**node_kwargs([7, 7, 7], noise=cs.Noise_EmptyNoise()))
    assert seen == [cs.Noise_EmptyNoise] * 3


def test_first_and_last_frame_positions(monkeypatch):
    img = torch.zeros(1, H, W, 3)
    r = run(monkeypatch, [7, 7, 7], first_frame=img, last_frame=img)
    idx = [[k["resolved_frame_index"] for k in c.meta["minimax_keyframes"]] for c in r.st.calls]
    assert idx[0] == [0] and idx[1] == [0]
    assert idx[2] == [0, r.plan.segments[-1].window_frames - 1]


def test_refs_ride_through_every_segment(monkeypatch):
    ref = torch.zeros(1, H, W, 3)
    r = run(monkeypatch, [7, 7, 7], ref_images={"ref_image_0": ref, "ref_image_1": ref},
            subject_definitions="<Luma> is the golden creature.")
    for c in r.st.calls:
        assert len(c.meta["minimax_refs"]) == 2
    assert all(p.startswith("subject_definitions:") for p in r.clip.prompts)


def test_refs_with_frames_rejected(monkeypatch):
    img = torch.zeros(1, H, W, 3)
    with pytest.raises(ValueError, match="can't be combined"):
        run(monkeypatch, [7, 7, 7], first_frame=img, ref_images={"ref_image_0": img})


def test_dry_run_never_samples(monkeypatch):
    monkeypatch.setattr(nodes, "_sample_window",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("sampled")))
    latent, frames, secs, report = call_node(**node_kwargs([7, 7, 7], dry_run=True))
    from comfy_execution.graph_utils import ExecutionBlocker
    assert isinstance(latent, ExecutionBlocker) and latent.message is None
    assert frames == P.plan_from_durations([7, 7, 7], 22).total_frames
    assert "Segment 3:" in report


# ---------------------------------------------------------------------------
# 4. The model's own layout
# ---------------------------------------------------------------------------

def _layout(positive, target):
    vs = target["samples"].tensors[0].shape
    at = target["samples"].tensors[1].shape[-1]
    m = positive[0][1]
    return PackedLayout(7, vs[2], (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2, at,
                        keyframes=m.get("minimax_keyframes"), refs=m.get("minimax_refs"))


@pytest.mark.parametrize("with_refs", [False, True])
@pytest.mark.parametrize("with_song", [False, True])
@pytest.mark.parametrize("overlap", [5, 22, 39])
def test_real_layout_aligns_guides_with_the_window(with_refs, with_song, overlap):
    plan = P.plan_from_durations([7, 7, 7], overlap)
    seg, prev = plan.segments[1], plan.segments[0]
    cumulative = {"samples": NestedTensor((
        torch.randn(1, 24, P.video_tokens(prev.window_frames), H // 16, W // 16),
        torch.randn(1, 32, 2, prev.window_audio_tokens)))}
    meta = {}
    if with_refs:
        meta["minimax_refs"] = [{"kind": "image", "latent_h": H // 16, "latent_w": W // 16,
                                 "latent": torch.zeros(1, 24, 1, H // 16, W // 16)}]
    song_audio = None
    if with_song:
        song = make_song(30)
        song_audio = nodes._song_slice(song, seg)
    positive = nodes._add_tail_guide([[torch.zeros(1, 7, 16), meta]], cumulative, seg, song_audio)
    layout = _layout(positive, nodes._empty_window(W, H, seg))

    kinds = {k: (s, e) for s, e, k in layout.segments}
    rows = (H // 16 // 2) * (W // 16 // 2)
    cs_, ce = kinds["cond"]
    ts, _ = kinds["video"]
    assert ce - cs_ == seg.overlap_video_tokens * rows
    assert torch.equal(layout.position_ids[cs_:ce, 0], layout.position_ids[ts:ts + ce - cs_, 0])

    acs, ace = kinds["cond_audio"]
    aus, aue = kinds["audio"]
    expect = seg.window_audio_tokens if with_song else seg.overlap_audio_tokens
    assert ace - acs == expect * 2, "audio guide should span the window with a song, else the overlap"
    # Stereo rows are channel-major (all left, then all right), so compare each
    # channel of the guide with the start of the same channel in the target.
    g_t, t_t = ace - acs, aue - aus
    guide, target = layout.position_ids[acs:ace], layout.position_ids[aus:aue]
    gn, tn = g_t // 2, t_t // 2
    for ch in (0, 1):
        assert torch.equal(guide[ch * gn:(ch + 1) * gn], target[ch * tn:ch * tn + gn]), \
            f"audio guide channel {ch} is off the window's timeline"
    assert not layout.img_update[:seg.overlap_video_tokens * rows].any()


def test_conflicting_head_guide_rejected():
    plan = P.plan_from_durations([7, 7, 7], 22)
    prev = plan.segments[0]
    cumulative = {"samples": NestedTensor((
        torch.randn(1, 24, P.video_tokens(prev.window_frames), H // 16, W // 16),
        torch.randn(1, 32, 2, prev.window_audio_tokens)))}
    pos = [[torch.zeros(1, 7, 16), {"minimax_keyframes": [{"resolved_frame_index": 3}]}]]
    with pytest.raises(ValueError, match="hidden overlap"):
        nodes._add_tail_guide(pos, cumulative, plan.segments[1])


# ---------------------------------------------------------------------------
# 5. Real sampler wrapper, real noise on a nested AV latent
# ---------------------------------------------------------------------------

def test_real_sampler_custom_advanced_path(monkeypatch):
    fmt = latent_formats.MiniMaxH3AV()
    got = {}

    class StubGuider:
        def __init__(self, model):
            self.model_patcher = SimpleNamespace(load_device=torch.device("cpu"),
                                                 model=SimpleNamespace(latent_format=fmt))

        def set_conds(self, positive):
            got["positive"] = positive

        def sample(self, noise, latent_image, sampler, sigmas, **kw):
            got["noise"], got["seed"] = noise, kw.get("seed")
            return latent_image

    monkeypatch.setattr(cs, "Guider_Basic", StubGuider)
    seg = P.plan_from_durations([7, 7], 22).segments[1]
    target = nodes._empty_window(W, H, seg)
    out = nodes._sample_window(None, cs.Noise_RandomNoise(7), None, torch.tensor([1.0, 0.0]),
                               [[torch.zeros(1, 7, 16), {}]], target)
    v, a = out["samples"].tensors
    assert v.shape == target["samples"].tensors[0].shape
    assert a.shape == target["samples"].tensors[1].shape
    nv, na = got["noise"].tensors
    assert nv.shape == v.shape and na.shape == a.shape and nv.abs().sum() > 0
    assert got["seed"] == 7


# ---------------------------------------------------------------------------
# 6. External reference blocks (Apply H3 RefMod)
# ---------------------------------------------------------------------------

def refmod_blocks():
    """Blocks in the exact shape Apply H3 RefMod appends to minimax_refs."""
    hh, ww = H // 16, W // 16
    return [
        {"kind": "image", "latent_h": hh, "latent_w": ww,
         "latent": torch.randn(1, 24, 1, hh, ww)},
        {"kind": "video", "latent_h": hh, "latent_w": ww, "latent_t": 7,
         "latent": torch.randn(1, 24, 7, hh, ww), "ref_audio_t": 0, "audio_latent": None},
        {"kind": "audio", "ref_audio_t": 40, "audio_latent": torch.randn(1, 32, 2, 40)},
    ]


def refmod_conditioning(blocks):
    return [[torch.zeros(1, 7, 16), {"minimax_refs": blocks}]]


def test_extra_refs_reach_every_segment_and_stitch_stays_exact(monkeypatch):
    blocks = refmod_blocks()
    r = run(monkeypatch, [7, 7, 7], extra_refs=refmod_conditioning(blocks))
    for c in r.st.calls:
        got = c.meta["minimax_refs"]
        assert len(got) == 3 and all(a is b for a, b in zip(got, blocks))
    v, a = r.out[0]["samples"].tensors
    assert torch.equal(v, r.gt_v) and torch.equal(a, r.gt_a)


def test_extra_refs_come_after_native_refs(monkeypatch):
    blocks = refmod_blocks()
    ref = torch.zeros(1, H, W, 3)
    r = run(monkeypatch, [7, 7], ref_images={"ref_image_0": ref},
            extra_refs=refmod_conditioning(blocks))
    for c in r.st.calls:
        got = c.meta["minimax_refs"]
        assert len(got) == 4
        assert got[0]["kind"] == "image" and got[0] is not blocks[0]   # native <Picture 1>
        assert all(a is b for a, b in zip(got[1:], blocks))            # RefMod after it


def test_extra_refs_work_with_a_song_and_first_frame(monkeypatch):
    img = torch.zeros(1, H, W, 3)
    r = run(monkeypatch, [7, 7, 7], song=make_song(25), first_frame=img,
            extra_refs=refmod_conditioning(refmod_blocks()))
    for c in r.st.calls:
        assert len(c.meta["minimax_refs"]) == 3


def test_extra_refs_prompt_text_is_ignored(monkeypatch):
    r = run(monkeypatch, [7, 7], extra_refs=[[torch.ones(1, 99, 16), {"minimax_refs": []}]])
    for c in r.st.calls:
        assert "minimax_refs" not in c.meta or c.meta["minimax_refs"] == []


@pytest.mark.parametrize("overlap", [5, 22, 39])
def test_real_layout_with_refmod_blocks_keeps_guides_aligned(overlap):
    plan = P.plan_from_durations([7, 7, 7], overlap)
    seg, prev = plan.segments[1], plan.segments[0]
    cumulative = {"samples": NestedTensor((
        torch.randn(1, 24, P.video_tokens(prev.window_frames), H // 16, W // 16),
        torch.randn(1, 32, 2, prev.window_audio_tokens)))}
    pos = nodes._add_ref_blocks([[torch.zeros(1, 7, 16), {}]], refmod_blocks())
    pos = nodes._add_tail_guide(pos, cumulative, seg)
    layout = _layout(pos, nodes._empty_window(W, H, seg))

    kinds = {}
    for s, e, k in layout.segments:
        kinds.setdefault(k, (s, e))
    rows = (H // 16 // 2) * (W // 16 // 2)
    cs_, ce = kinds["cond"]
    ts, _ = kinds["video"]
    assert ce - cs_ == seg.overlap_video_tokens * rows
    assert torch.equal(layout.position_ids[cs_:ce, 0], layout.position_ids[ts:ts + ce - cs_, 0]), \
        "RefMod blocks shifted the tail guide off the window's timeline"


# ---------------------------------------------------------------------------
# 7. Growing reference inputs, matching the native Reference to Video node
# ---------------------------------------------------------------------------

import comfy_extras.nodes_minimax_h3 as native_h3  # noqa: E402


class RecordingClip(MockClip):
    def __init__(self):
        super().__init__()
        self.items = []

    def tokenize(self, prompt, images=None, minimax_ref_items=None):
        self.items.append([i["type"] for i in (minimax_ref_items or [])])
        return super().tokenize(prompt, images=images)


class CountingVae(MockVae):
    def __init__(self):
        self.calls = []

    def encode(self, pixels):
        self.calls.append(pixels.shape[0])
        return super().encode(pixels)


class CountingAudioVae(MockAudioVae):
    def __init__(self):
        self.calls = 0

    def encode(self, wave):
        self.calls += 1
        return super().encode(wave)


def ref_set():
    img = torch.rand(1, H, W, 3)
    vid = torch.rand(30, H, W, 3)                       # 1.25s at 24fps -> trimmed to 22 frames
    snd = {"waveform": torch.zeros(1, 2, SR), "sample_rate": SR}
    voice = {"waveform": torch.zeros(1, 2, 2 * SR), "sample_rate": SR}
    return dict(ref_images={"ref_image_0": img}, ref_videos={"ref_video_0": vid},
                ref_video_audios={"ref_video_audio_0": snd}, ref_audios={"ref_audio_0": voice})


def test_schema_matches_native_reference_inputs():
    info = nodes.MiniMaxH3LongShot.GET_NODE_INFO_V1()
    mine = nodes.MiniMaxH3LongShot.GET_SCHEMA()
    native = native_h3.MiniMaxH3ReferenceToVideo.GET_SCHEMA()

    def grows(schema):
        out = {}
        for inp in schema.inputs:
            t = getattr(inp, "template", None)
            if t is not None and hasattr(t, "prefix"):
                out[inp.id] = (t.prefix, t.max)
        return out

    assert grows(mine) == grows(native)
    assert info["display_name"] == "MiniMax H3 Long Shot"
    assert info["output_name"] == ["latent", "total_frames", "total_seconds", "plan"]


def test_labels_reach_the_tokenizer_in_native_order(monkeypatch):
    """Long Shot must number references exactly as the native node does."""
    refs = ref_set()
    clip = RecordingClip()
    run(monkeypatch, [7, 7], clip=clip, vae=MockVae(), audio_vae=MockAudioVae(), **refs)

    native_clip = RecordingClip()
    native_h3.MiniMaxH3ReferenceToVideo.execute(
        native_clip, "x", W, H, 175, vae=MockVae(), audio_vae=MockAudioVae(), **refs)
    expected = native_clip.items[0]
    assert expected == ["image", "audio", "video", "audio"]   # soundtrack is <Audio 1>
    assert clip.items == [expected, expected], "every segment presents the same labels"


def test_reference_encodes_are_cached_per_window_length(monkeypatch):
    refs = ref_set()
    vae, avae = CountingVae(), CountingAudioVae()
    # Segment 1 has no overlap, so equal windows need it 22 frames longer than the rest.
    durs = [175 / 24, 153 / 24, 153 / 24]
    plan = P.plan_from_durations(durs, 22)
    assert {s.window_frames for s in plan.segments} == {175}, "need equal windows for this test"
    r = run(monkeypatch, durs, vae=vae, audio_vae=avae, **refs)
    # one image + one video encode, once — not once per segment
    assert sorted(vae.calls) == [1, 22]
    assert avae.calls == 2                                     # soundtrack + standalone, once
    blocks = [c.meta["minimax_refs"] for c in r.st.calls]
    assert all(len(b) == 3 for b in blocks)                    # image, video+audio, audio
    for b in blocks[1:]:
        assert all(torch.equal(x["latent" if x["kind"] != "audio" else "audio_latent"],
                               y["latent" if y["kind"] != "audio" else "audio_latent"])
                   for x, y in zip(b, blocks[0]))


def test_different_window_lengths_get_their_own_encode(monkeypatch):
    refs = ref_set()
    vae = CountingVae()
    plan = P.plan_from_durations([7, 4, 7], 22)
    n_lengths = len({s.window_frames for s in plan.segments})
    assert n_lengths > 1
    run(monkeypatch, [7, 4, 7], vae=vae, audio_vae=MockAudioVae(), **refs)
    assert vae.calls.count(22) == n_lengths


def test_reference_audio_needs_the_audio_vae(monkeypatch):
    with pytest.raises(ValueError, match="audio_vae"):
        run(monkeypatch, [7, 7], ref_audios={"ref_audio_0": ref_set()["ref_audios"]["ref_audio_0"]})


def test_soundtrack_without_its_video_is_a_clear_error(monkeypatch):
    snd = ref_set()["ref_video_audios"]["ref_video_audio_0"]
    with pytest.raises(ValueError, match="no matching ref_video_0"):
        run(monkeypatch, [7, 7], audio_vae=MockAudioVae(),
            ref_video_audios={"ref_video_audio_0": snd})


def test_missing_shots_is_a_clear_error(monkeypatch):
    with pytest.raises(ValueError, match="No shots"):
        call_node(**node_kwargs(None))


# ---------------------------------------------------------------------------
# 8. End to end with the real prompt pack: Shot chain -> r2v builder -> Long Shot
# ---------------------------------------------------------------------------

PROMPT_PACK = os.environ.get("MMH3_PROMPT_PACK")


def _load_prompt_pack():
    if not PROMPT_PACK or not os.path.isfile(os.path.join(PROMPT_PACK, "__init__.py")):
        pytest.skip("set MMH3_PROMPT_PACK to the comfyui-minimax-h3 folder to run this test")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mmh3_prompt_pack",
                                                  os.path.join(PROMPT_PACK, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_r2v_builder_drives_long_shot(monkeypatch):
    pack = _load_prompt_pack()
    shot, builder = pack.MiniMaxH3Shot(), pack.MiniMaxH3RefPromptBuilder()
    c, = shot.add("the camera cuts to", 5.0, "She reads from the page.")
    c, = shot.add("the shot cuts to", 5.0, "Her voice catches.", c)
    outs = builder.build(
        "<Subject 1> is Kat, the young woman whose appearance comes from <Picture 1>.",
        "reference generation", "-", "-",
        "The target video shows <Subject 1> reading a poem aloud.",
        "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - her face is retained.",
        "Live-action, cinematic, soft classroom light.", "", "Quiet classroom room tone.", "",
        shots=c)
    assert len(outs) == 4
    bundle = outs[3]
    plan = P.plan_from_durations([5.0, 5.0], 22)
    sim, *_ = make_simulator(plan)
    monkeypatch.setattr(nodes, "_sample_window", sim)
    clip = RecordingClip()
    ref = torch.rand(1, H, W, 3)
    latent, frames, secs, report = call_node(
        model=None, clip=clip, vae=MockVae(), noise=cs.Noise_RandomNoise(1), sampler=None,
        sigmas=None, prompt=bundle, width=W, height=H, overlap_frames=22,
        seed_mode="increment", dry_run=False, ref_images={"ref_image_0": ref})
    assert frames == plan.total_frames
    p1, p2 = clip.prompts
    for p, text in ((p1, "She reads from the page."), (p2, "Her voice catches.")):
        assert p.startswith("subject_definitions:\n<Subject 1> is Kat")
        assert "summary:\n[reference generation] The target video shows" in p
        assert "(appears in [Shot 1]):" in p and "Shot 2" not in p
        assert f"detailed_description:\nLive-action, cinematic, soft classroom light.\n[Shot 1] {text}" in p
        assert "overall_soundscape:\nQuiet classroom room tone." in p



# ---------------------------------------------------------------------------
# 9. A single Shot is one ordinary generation
# ---------------------------------------------------------------------------

def test_single_shot_runs_with_no_tail_guide(monkeypatch):
    r = run(monkeypatch, [6.5])
    assert len(r.st.calls) == 1
    assert not r.st.calls[0].meta.get("minimax_keyframes")
    v, a = r.out[0]["samples"].tensors
    assert torch.equal(v, r.gt_v) and torch.equal(a, r.gt_a)


def test_single_shot_with_song_refs_and_refmod(monkeypatch):
    plan = P.plan_from_durations([6.5], 22)
    song = make_song(10)
    refs = ref_set()
    r = run(monkeypatch, [6.5], song=song, audio_vae=MockAudioVae(),
            extra_refs=refmod_conditioning(refmod_blocks()), **refs)
    meta = r.st.calls[0].meta
    kfs = meta["minimax_keyframes"]
    assert len(kfs) == 1 and kfs[0]["resolved_frame_index"] == 0
    assert torch.equal(kfs[0]["audio_latent"], song["latent"][..., :plan.total_audio_tokens])
    assert len(meta["minimax_refs"]) == 3 + 3      # native refs + RefMod blocks
    v, a = r.out[0]["samples"].tensors
    assert torch.equal(a, song["latent"][..., :plan.total_audio_tokens])


# ---------------------------------------------------------------------------
# 10. The console gets a short summary; the plan output keeps the prompts
# ---------------------------------------------------------------------------

def test_console_log_is_a_summary_without_prompts(monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="MiniMaxH3LongShot"):
        r = run(monkeypatch, [7, 7, 7], texts=["SECRET-ONE.", "SECRET-TWO.", "SECRET-THREE."],
                style_line="STYLE-MARKER")
    logged = caplog.text
    for i in (1, 2, 3):
        assert f"Segment {i}: asked" in logged
    assert "SECRET" not in logged and "STYLE-MARKER" not in logged
    assert "integrated_multimodal_description" not in logged
    report = r.out[3]
    assert "SECRET-TWO." in report and "STYLE-MARKER" in report, "plan output keeps everything"


def test_console_summary_keeps_warnings(monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="MiniMaxH3LongShot"):
        run(monkeypatch, [6, 16])
    assert "NOTE: Segment 2 window" in caplog.text


# ---------------------------------------------------------------------------
# RefMods through the refmods input: presented to the text encoder
# ---------------------------------------------------------------------------

class FakeRefMod:
    """Duck-types the RefMod pack's H3RefMod: name, kind, ref_block(strength)."""

    def __init__(self, name, kind, latent_t=1):
        self.name, self.kind, self.latent_t = name, kind, latent_t
        hh, ww = H // 16, W // 16
        if kind == "audio":
            self.latent = torch.randn(1, 32, 2, 40)
        else:
            self.latent = torch.randn(1, 24, latent_t, hh, ww)

    def ref_block(self, strength=1.0):
        z = self.latent * strength
        if self.kind == "audio":
            return {"kind": "audio", "ref_audio_t": z.shape[-1], "audio_latent": z}
        b = {"kind": self.kind, "latent_h": z.shape[3], "latent_w": z.shape[4], "latent": z}
        if self.kind == "video":
            b.update(latent_t=self.latent_t, ref_audio_t=0, audio_latent=None)
        return b


class DecodingVae(CountingVae):
    """Decodes a latent to BTHWC frames (17k+5 frames per 5k+2 tokens is not
    needed here — only the shape contract RefMod Text Encode relies on)."""

    def __init__(self):
        super().__init__()
        self.decodes = 0

    def decode(self, z):
        self.decodes += 1
        t = 1 if z.shape[2] == 1 else (z.shape[2] - 1) * 4 + 1
        return torch.rand(1, t, z.shape[3] * 16, z.shape[4] * 16, 3)


class ItemClip(MockClip):
    def __init__(self):
        super().__init__()
        self.items = []

    def tokenize(self, prompt, images=None, minimax_ref_items=None):
        self.items.append(list(minimax_ref_items or []))
        return super().tokenize(prompt, images=images)


def mods_set():
    return [(FakeRefMod("hero_face", "image"), 1.0),
            (FakeRefMod("hero_walk", "video", latent_t=7), 0.8),
            (FakeRefMod("hero_voice", "audio"), 1.0)]


def test_refmods_take_the_next_labels_after_native_refs(monkeypatch):
    refs, mods = ref_set(), mods_set()
    clip, vae = ItemClip(), DecodingVae()
    r = run(monkeypatch, [7, 7], clip=clip, vae=vae, audio_vae=MockAudioVae(),
            refmods=mods, **refs)
    plan_text = r.out[3]
    # native: 1 image, 1 video, soundtrack + standalone audio -> RefMods continue from there
    for line in ("<Picture 1> = ref_image_0", "<Audio 1> = ref_video_audio_0",
                 "<Video 1> = ref_video_0", "<Audio 2> = ref_audio_0",
                 "<Picture 2> = hero_face (RefMod)", "<Video 2> = hero_walk (RefMod)",
                 "<Audio 3> = hero_voice (RefMod)"):
        assert line in plan_text
    native = ["image", "audio", "video", "audio"]
    for seg_items in clip.items:
        assert [i["type"] for i in seg_items] == native + ["image", "video", "audio"]
        img, vid = seg_items[4], seg_items[5]
        assert img["data"].shape[0] == 1 and img["data"].shape[-1] == 3
        assert vid["timestamps"][0] == 0.0 and vid["data"].shape[0] == len(vid["timestamps"])
    for c in r.st.calls:
        blocks = c.meta["minimax_refs"]
        assert [b["kind"] for b in blocks] == ["image", "video_audio", "audio",
                                               "image", "video", "audio"]
        assert [bool(b.get("refmod")) for b in blocks] == [False] * 3 + [True] * 3
    # strength reached the block the model sees
    assert torch.allclose(r.st.calls[0].meta["minimax_refs"][4]["latent"],
                          mods[1][0].latent * 0.8)
    v, a = r.out[0]["samples"].tensors
    assert torch.equal(v, r.gt_v) and torch.equal(a, r.gt_a)


def test_refmods_alone_use_the_reference_path(monkeypatch):
    clip, vae = ItemClip(), DecodingVae()
    mods = [(FakeRefMod("hero_face", "image"), 1.0)]
    r = run(monkeypatch, [7, 7, 7], clip=clip, vae=vae, refmods=mods,
            subject_definitions="<hero> comes from <Picture 1>.")
    assert "<Picture 1> = hero_face (RefMod)" in r.out[3]
    assert all([i["type"] for i in items] == ["image"] for items in clip.items)
    assert all(len(c.meta["minimax_refs"]) == 1 for c in r.st.calls)
    assert all(p.startswith("subject_definitions:") for p in clip.prompts), "r2v format"


def test_refmods_are_decoded_once_per_run(monkeypatch):
    vae = DecodingVae()
    run(monkeypatch, [6, 6, 6, 6], vae=vae, refmods=mods_set())
    assert vae.decodes == 2          # image + video; audio is never decoded


def test_zero_strength_refmod_gets_no_label_or_block(monkeypatch):
    clip = ItemClip()
    mods = [(FakeRefMod("off", "image"), 0.0), (FakeRefMod("on", "image"), 1.0)]
    r = run(monkeypatch, [7], clip=clip, vae=DecodingVae(), refmods=mods)
    assert "<Picture 1> = on (RefMod)" in r.out[3] and "off" not in r.out[3].split("Segment")[0]
    assert len(clip.items[0]) == 1 and len(r.st.calls[0].meta["minimax_refs"]) == 1


def test_dry_run_lists_refmod_labels_without_decoding(monkeypatch):
    vae = DecodingVae()
    out = call_node(**node_kwargs([7, 7], dry_run=True, vae=vae, refmods=mods_set()))
    assert "<Picture 1> = hero_face (RefMod)" in out[3] and vae.decodes == 0


def test_refmods_cannot_combine_with_first_frame():
    with pytest.raises(ValueError, match="RefMods"):
        call_node(**node_kwargs([7, 7], vae=DecodingVae(), refmods=mods_set(),
                                first_frame=torch.rand(1, H, W, 3)))


def test_refmods_and_tail_guides_line_up_in_the_real_layout(monkeypatch):
    """The model's own PackedLayout must accept keyframes + native refs + RefMods."""
    refs = ref_set()
    r = run(monkeypatch, [7, 7], vae=DecodingVae(), audio_vae=MockAudioVae(),
            refmods=mods_set(), **refs)
    seg = r.plan.segments[1]
    meta = r.st.calls[1].meta
    layout = PackedLayout(7, P.video_tokens(seg.window_frames), H // 16, W // 16,
                          seg.window_audio_tokens, keyframes=meta["minimax_keyframes"],
                          refs=meta["minimax_refs"])
    kinds = [k for _a, _b, k in layout.segments]
    assert kinds.count("ref_img") == 4          # native image + video, RefMod image + video
    assert kinds.count("cond") == 1             # the continuation's video tail


def test_unconnected_or_muted_refs_take_no_label(monkeypatch):
    """A muted loader's link never reaches the node; slot names don't matter."""
    img = torch.rand(1, H, W, 3)
    r = run(monkeypatch, [7], vae=DecodingVae(), refmods=[(FakeRefMod("hero", "image"), 1.0)],
            ref_images={"ref_image_0": None, "ref_image_5": img})
    assert "<Picture 1> = ref_image_5" in r.out[3]
    assert "<Picture 2> = hero (RefMod)" in r.out[3]
