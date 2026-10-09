"""Timeline mode (round 2): pieces, shared zones, end pins with exact
write-back, locked takes, pin fingerprints, seams and the takes store.

The sampler stub behaves like a real sampler with fixed settings: its output
depends only on seed, prompt and the guides it is handed. It reproduces a
start guide in the window's head (a real continuation does, closely) and
only approximately honours an end guide, so the exact join can only come
from Long Shot's write-back."""
import hashlib
import os
import random
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_integration as T  # noqa: E402  (boots ComfyUI, loads the pack)
from comfy.nested_tensor import NestedTensor  # noqa: E402

nodes, P = T.nodes, T.P
store = __import__(T.PKG + ".store", fromlist=["store"])
RECIPE = {"model": "m" * 32, "sampler": "s" * 32}
OV = 22
OVT = P.video_tokens(OV)


class PinStub:
    def __init__(self):
        self.calls = []

    def __call__(self, model, noise, sampler, sigmas, positive, latent):
        meta = positive[0][1]
        kfs = meta.get("minimax_keyframes", [])
        tv, ta = latent["samples"].tensors
        head = [k for k in kfs if k["resolved_frame_index"] == 0 and k.get("latent") is not None]
        tail = [k for k in kfs if k["resolved_frame_index"] > 0 and k.get("latent") is not None]
        prompt = CLIP.prompts[-1]
        self.calls.append({"seed": noise.seed, "prompt": prompt, "keyframes": kfs})
        h = hashlib.sha256(f"{noise.seed}|{prompt}".encode())
        for k in head + tail:
            h.update(str(k["resolved_frame_index"]).encode())
            h.update(k["latent"].float().numpy().tobytes())
        g = torch.Generator().manual_seed(int.from_bytes(h.digest()[:8], "little"))
        v = torch.randn(tv.shape, generator=g)
        a = torch.randn(ta.shape, generator=g)
        if head:
            v[:, :, :head[0]["latent"].shape[2]] = head[0]["latent"]
        if tail:     # close, but not exact: write-back has to make it exact
            t = tail[0]["latent"]
            v[:, :, -t.shape[2]:] = t + 0.01 * torch.randn(t.shape, generator=g)
        return {"samples": NestedTensor((v, a))}


CLIP, VAE = T.MockClip(), T.MockVae()


def shot(sid, seconds=5.0, text=None, seed=-1, lock=None, join="bridge", frames=None):
    return {"kind": "shot", "id": sid, "seconds": seconds, "text": text or f"beat {sid}",
            "cut_verb": "the camera cuts to", "seed": seed, "lock": lock, "join": join,
            "frames": frames}


def run(monkeypatch, shots, stub=None, recipe=RECIPE, dry_run=False, song=None, **extra):
    stub = stub or PinStub()
    monkeypatch.setattr(nodes, "_sample_window", stub)
    kw = T.node_kwargs([], clip=CLIP, vae=VAE, noise=T.cs.Noise_RandomNoise(100),
                       dry_run=dry_run, song=song)
    kw["prompt"]["shots"] = shots
    kw.update(save_to_disk=True, cache_name="proj", **extra)
    outputs, rows = nodes._run_long_shot(**kw, _recipe=recipe)
    return outputs, rows, stub


def lat(outputs):
    return outputs[0]["samples"].tensors


def takes_dir():
    return os.path.join(nodes.STORE_ROOT, "proj", "takes")


def locked_from(shots, rows, keep):
    """The Studio's next step: rendered Shots lock to their takes and keep
    the seed they used."""
    out = []
    for sh, row in zip(shots, rows):
        sh = dict(sh, seed=row["seed"], frames=row["window_frames"])
        if sh["id"] in keep:
            sh["lock"] = row["take"]
        out.append(sh)
    return out


# ---------------------------------------------------------------------------
# Planner: pieces and shared zones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(40))
def test_pieces_without_fixed_windows_plan_exactly_like_round_1(seed):
    rnd = random.Random(seed)
    durs = [round(rnd.uniform(1.5, 12), 2) for _ in range(rnd.randint(1, 8))]
    ov = rnd.choice([5, 22, 39])
    try:
        a = P.plan_from_durations(durs, ov)
    except ValueError:
        with pytest.raises(ValueError):
            P.plan_pieces([P.PieceSpec(d) for d in durs], ov)
        return
    b = P.plan_pieces([P.PieceSpec(d) for d in durs], ov)
    assert a.segments == b.segments


@pytest.mark.parametrize("seed", range(60))
def test_grid_holds_for_any_mix_of_fixed_and_requested_pieces(seed):
    rnd = random.Random(1000 + seed)
    ov = rnd.choice([5, 22, 39])
    specs = []
    for _ in range(rnd.randint(1, 10)):
        if rnd.random() < 0.5:
            specs.append(P.PieceSpec(rnd.uniform(2, 9), window=17 * rnd.randint(3, 12) + 5, locked=True))
        else:
            specs.append(P.PieceSpec(round(rnd.uniform(1.5, 9), 2)))
    plan = P.plan_pieces(specs, ov)
    assert P.is_valid_frame_count(plan.total_frames)
    total_a = 0
    for k, s in enumerate(plan.segments):
        assert P.is_valid_frame_count(s.window_frames)
        if k == 0:
            assert s.new_frames == s.window_frames and s.window_start == 0
        else:
            assert s.new_frames % 17 == 0 and s.new_frames > 0
            assert s.window_start % 17 == 0, "every shared zone starts on a group boundary"
            assert s.overlap_frames == ov
        if specs[k].window is not None:
            assert s.window_frames == specs[k].window, "a take keeps its exact window"
        a0, a1 = s.window_audio_span
        assert a1 - a0 == s.window_audio_tokens
        assert s.overlap_audio_tokens + s.new_audio_tokens == s.window_audio_tokens
        total_a += s.new_audio_tokens
    assert total_a == plan.total_audio_tokens, "song slices by global position, no drift"
    assert plan.segments[-1].visible_end == plan.total_frames


def test_a_take_from_the_first_place_drops_its_head_further_down():
    first = P.plan_pieces([P.PieceSpec(5)], OV).segments[0]
    later = P.plan_pieces([P.PieceSpec(3), P.PieceSpec(5, window=first.window_frames)], OV)
    assert later.segments[1].window_frames == first.window_frames
    assert later.segments[1].new_frames == first.window_frames - OV


# ---------------------------------------------------------------------------
# Running a timeline
# ---------------------------------------------------------------------------

def test_ids_without_locks_render_like_a_round_1_chain(monkeypatch):
    shots = [shot("a", 5), shot("b", 4), shot("c", 6)]
    out, rows, stub = run(monkeypatch, shots)
    assert len(stub.calls) == 3
    nodes.clear_segment_cache()
    # the same chain without ids, through round 1's path, with the same stub
    plain = [{k: v for k, v in s.items() if k in ("seconds", "text", "cut_verb", "seed")}
             for s in shots]
    out2, rows2, stub2 = run(monkeypatch, plain)
    assert all(torch.equal(x, y) for x, y in zip(lat(out), lat(out2)))
    assert [r["window_frames"] for r in rows] == [r["window_frames"] for r in rows2]


def test_every_finished_piece_is_saved_as_a_take(monkeypatch):
    out, rows, _ = run(monkeypatch, [shot("a"), shot("b", 4)])
    names = sorted(os.listdir(takes_dir()))
    assert names == sorted(r["take"] for r in rows)
    assert rows[0]["take"].startswith("a__100__") and rows[1]["take"].startswith("b__101__")
    meta = store.TakeStore("proj", nodes.STORE_ROOT).meta(rows[1]["take"])
    assert meta["window_frames"] == rows[1]["window_frames"] and meta["overlap"] == OV
    assert meta["left_take"] == rows[0]["take"]
    assert meta["width"] == T.W and meta["height"] == T.H


def test_all_locked_samples_nothing_and_reproduces_the_film(monkeypatch):
    shots = [shot("a"), shot("b", 4), shot("c", 6)]
    out, rows, _ = run(monkeypatch, shots)
    nodes.clear_segment_cache()
    locked = locked_from(shots, rows, {"a", "b", "c"})
    out2, rows2, stub = run(monkeypatch, locked)
    assert stub.calls == []
    assert [r["status"] for r in rows2] == ["locked"] * 3
    assert all(torch.allclose(x, y.to(x), atol=2e-3) for x, y in zip(lat(out), lat(out2)))


def test_missing_take_is_a_clear_error(monkeypatch):
    with pytest.raises(ValueError, match="Shot 2's approved take is missing"):
        run(monkeypatch, [shot("a"), shot("b", lock="b__1__deadbeef.safetensors")])


def visible_ranges(rows):
    """Video token ranges [start, end) each piece shows in the stitched latent."""
    out, cur = [], 0
    for k, r in enumerate(rows):
        n = P.video_tokens(r["window_frames"]) - (0 if k == 0 else OVT)
        out.append((cur, cur + n))
        cur += n
    return out


def test_reroll_in_place_keeps_every_other_shot_bit_identical(monkeypatch):
    shots = [shot(x, s) for x, s in zip("abcde", (5, 4, 5, 6, 5))]
    out, rows, _ = run(monkeypatch, shots)
    before = lat(out)
    # approve all, then re-roll Shot 3 in place: new seed, same length
    locked = locked_from(shots, rows, set("abde"))
    locked[2] = dict(locked[2], seed=999, frames=rows[2]["window_frames"])
    out2, rows2, stub = run(monkeypatch, locked)
    assert len(stub.calls) == 1, "one render"
    kf = stub.calls[0]["keyframes"]
    start = [k for k in kf if k["resolved_frame_index"] == 0][0]
    end = [k for k in kf if k["resolved_frame_index"] > 0][0]
    assert end["resolved_frame_index"] == rows[2]["window_frames"] - OV
    old3 = store.TakeStore("proj", nodes.STORE_ROOT).load(rows[2]["take"])[0]
    assert torch.equal(end["latent"].to(old3), old3[:, :, -OVT:]), "end pin = Shot 3's old tail"
    assert rows2[2]["pins"]["end"]["kind"] == "old_tail" and rows2[2]["pins"]["start"] == {"from": 2}
    after = lat(out2)
    assert after[0].shape == before[0].shape and after[1].shape == before[1].shape
    s3, e3 = visible_ranges(rows2)[2]
    v0, v1 = before[0].float(), after[0].float()
    assert torch.allclose(v1[:, :, :s3], v0[:, :, :s3], atol=2e-3), "Shots 1-2 unchanged"
    assert torch.allclose(v1[:, :, e3 - OVT:], v0[:, :, e3 - OVT:], atol=2e-3), \
        "the shared zone and Shots 4-5 unchanged"
    assert not torch.allclose(v1[:, :, s3:e3 - OVT], v0[:, :, s3:e3 - OVT]), "Shot 3 is new"
    assert [r["seam"] for r in rows2] == [None, "ok", "ok", "ok", "ok"]


def test_write_back_makes_the_join_exact_even_when_the_sampler_drifts(monkeypatch):
    shots = [shot("a"), shot("b"), shot("c")]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "c"})
    locked[1] = dict(locked[1], seed=7, frames=rows[1]["window_frames"])
    run(monkeypatch, locked)
    ts = store.TakeStore("proj", nodes.STORE_ROOT)
    new_b = [m for m in ts.list("b") if m["seed"] == 7][0]
    old_b = ts.meta(rows[1]["take"])
    assert new_b["tail"] == old_b["tail"], "tail written back bit-exactly"
    assert new_b["end_src"] == old_b["tail"]


def test_prepend_keeps_old_shot_1_visible_frames(monkeypatch):
    shots = [shot("a", 5), shot("b", 4)]
    out, rows, _ = run(monkeypatch, shots)
    before = lat(out)[0].float()
    locked = locked_from(shots, rows, {"a", "b"})
    out2, rows2, stub = run(monkeypatch, [shot("p", 4)] + locked)
    assert len(stub.calls) == 1
    end = [k for k in stub.calls[0]["keyframes"] if k["resolved_frame_index"] > 0][0]
    old_a = store.TakeStore("proj", nodes.STORE_ROOT).load(rows[0]["take"])[0]
    assert torch.equal(end["latent"].to(old_a), old_a[:, :, :OVT]), "end pin = old Shot 1's head"
    assert rows2[0]["pins"]["end"]["kind"] == "head"
    after = lat(out2)[0].float()
    p_end = visible_ranges(rows2)[0][1]
    # old Shot 1's head is now the new piece's tail; everything after it is the old film
    assert torch.allclose(after[:, :, p_end - OVT:], before, atol=2e-3)
    assert rows2[1]["seam"] == "ok" and rows2[1]["seconds"] < rows[0]["seconds"]


def test_moving_pieces_does_not_re_render_unchanged_ones(monkeypatch):
    shots = [shot("a"), shot("b"), shot("c")]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "b"})       # c still under review (unlocked)
    _, _, stub = run(monkeypatch, [shot("p", 4)] + locked)
    assert [c["prompt"].count("beat p") for c in stub.calls] == [1], "only the new piece renders"


def test_changing_a_pin_source_re_renders(monkeypatch):
    shots = [shot("a"), shot("b"), shot("c")]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a"})
    # Shot 2 gets a new seed: Shot 3 follows a different tail, so it renders too
    locked[1] = dict(locked[1], seed=55)
    _, rows2, stub = run(monkeypatch, locked)
    assert len(stub.calls) == 2
    assert rows2[2]["reason"] in ("start pin changed", "first run", "new take") or \
        rows2[2]["status"] == "render"


def test_switching_to_an_older_take_samples_nothing(monkeypatch):
    shots = [shot("a"), shot("b")]
    out, rows, _ = run(monkeypatch, shots)
    first_b = rows[1]["take"]
    locked = locked_from(shots, rows, {"a"})
    locked[1] = dict(locked[1], seed=77)
    run(monkeypatch, locked)                                   # a second take of b
    assert len(store.TakeStore("proj", nodes.STORE_ROOT).list("b")) == 2
    back = locked_from(shots, rows, {"a"})
    back[1] = dict(back[1], lock=first_b)
    out3, rows3, stub = run(monkeypatch, back)
    assert stub.calls == [] and rows3[1]["take"] == first_b
    assert torch.allclose(lat(out3)[0], lat(out)[0].to(lat(out3)[0]), atol=2e-3)


def test_removing_a_middle_shot_leaves_a_reported_hard_cut(monkeypatch):
    shots = [shot("a"), shot("b"), shot("c")]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "b", "c"})
    _, rows2, stub = run(monkeypatch, [locked[0], locked[2]])
    assert stub.calls == []
    assert rows2[1]["seam"] == "mismatch"
    # re-rolling the Shot after the gap bridges it: pinned to a's tail and its own old tail
    fixed = [locked[0], dict(locked[2], lock=None, seed=5, frames=rows[2]["window_frames"])]
    _, rows3, stub = run(monkeypatch, fixed)
    assert len(stub.calls) == 1 and rows3[1]["seam"] == "ok"
    assert rows3[1]["pins"]["start"] == {"from": 1}


def test_bypassed_first_shot_makes_the_next_take_first(monkeypatch):
    shots = [shot("a"), shot("b"), shot("c")]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"b", "c"})
    out2, rows2, stub = run(monkeypatch, locked[1:])
    assert stub.calls == []
    assert rows2[0]["frames"] == rows[1]["window_frames"], "a take that becomes first shows its head"
    assert P.is_valid_frame_count(out2[1])


def test_cut_join_starts_fresh(monkeypatch):
    _, rows, stub = run(monkeypatch, [shot("a"), shot("b", join="cut")])
    assert [k for k in stub.calls[1]["keyframes"] if k.get("latent") is not None] == []
    assert rows[1]["join"] == "cut" and rows[1]["seam"] == "cut"


def test_too_short_to_pin_at_both_ends(monkeypatch):
    shots = [shot("a"), shot("b", 3), shot("c")]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "c"})
    locked[1] = dict(locked[1], seconds=0.8, seed=3, frames=None)
    with pytest.raises(ValueError, match="too short to be pinned at both ends"):
        run(monkeypatch, locked)


def test_take_made_at_another_size_is_refused(monkeypatch):
    out, rows, _ = run(monkeypatch, [shot("a")])
    with pytest.raises(ValueError, match="width"):
        run(monkeypatch, [shot("a", lock=rows[0]["take"])], width=512)


def test_dry_run_plans_pins_from_metadata_only(monkeypatch):
    shots = [shot(x) for x in "abc"]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "c"})
    locked[1] = dict(locked[1], seed=4, frames=rows[1]["window_frames"])
    loads = []
    real = store.TakeStore.load
    monkeypatch.setattr(store.TakeStore, "load", lambda self, n: loads.append(n) or real(self, n))
    out2, rows2, stub = run(monkeypatch, locked, dry_run=True)
    assert stub.calls == [] and loads == []
    assert [r["status"] for r in rows2] == ["locked", "render", "locked"]
    assert "end → its old tail" in out2[3] and "start ← Shot 1 tail" in out2[3]


def test_lazy_inputs_skip_the_model_when_everything_is_locked(monkeypatch):
    shots = [shot(x) for x in "ab"]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "b"})
    kw = T.node_kwargs([], noise=T.cs.Noise_RandomNoise(100))
    kw["prompt"]["shots"] = locked
    kw.update(save_to_disk=True, cache_name="proj", model=None, clip=None, vae=None)
    monkeypatch.setattr(nodes.disk, "linked_inputs", lambda *a: {"model", "clip", "vae", "sigmas"})
    monkeypatch.setattr(nodes.disk, "recipe_components", lambda *a: RECIPE)
    assert nodes.lazy_requests(kw, {}, "1") == []
    locked[1] = dict(locked[1], lock=None, seed=9)
    kw["prompt"]["shots"] = locked
    assert set(nodes.lazy_requests(kw, {}, "1")) == {"model", "clip", "vae", "sigmas"}


def test_song_rides_on_the_start_guide_and_the_end_pin_is_video_only(monkeypatch):
    song = T.make_song(40)
    shots = [shot(x) for x in "abc"]
    out, rows, _ = run(monkeypatch, shots, song=song)
    locked = locked_from(shots, rows, {"a", "c"})
    locked[1] = dict(locked[1], seed=8, frames=rows[1]["window_frames"])
    out2, rows2, stub = run(monkeypatch, locked, song=song)
    kfs = stub.calls[0]["keyframes"]
    audio = [k for k in kfs if k.get("audio_latent") is not None]
    assert len(audio) == 1 and audio[0]["resolved_frame_index"] == 0
    seg = nodes.planner.plan_pieces(
        [P.PieceSpec(5, window=r["window_frames"]) for r in rows], OV).segments[1]
    a0, a1 = seg.window_audio_span
    assert torch.equal(audio[0]["audio_latent"], song["latent"][..., a0:a1])
    assert torch.equal(lat(out2)[1], song["latent"][..., :lat(out2)[1].shape[-1]].to(lat(out2)[1]))


def test_round_1_saved_segments_are_adopted_as_takes(monkeypatch):
    plain = [{"seconds": 5.0, "text": f"beat {x}", "cut_verb": "the camera cuts to", "seed": -1}
             for x in "ab"]
    out, rows, stub = run(monkeypatch, plain)                 # round 1 path, saves segments
    assert len(stub.calls) == 2
    nodes.clear_segment_cache()
    out2, rows2, stub2 = run(monkeypatch, [shot("a"), shot("b")])
    assert stub2.calls == [], "adopted, not re-rendered"
    assert all(r["take"] for r in rows2)
    assert all(torch.allclose(x, y.to(x), atol=2e-3) for x, y in zip(lat(out), lat(out2)))


def test_audio_slots_shift_by_at_most_one_token_and_stitch_cleanly(monkeypatch):
    shots = [shot("a", 5), shot("b", 5)]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "b"})
    for secs in (1.3, 2.1, 3.7, 4.4):
        nodes.clear_segment_cache()
        out2, rows2, _ = run(monkeypatch, [shot("p", secs)] + locked)
        v, a = lat(out2)
        assert a.shape[-1] == P.audio_at(out2[1]) and v.shape[2] == P.video_tokens(out2[1])


def test_timeline_shot_node_builds_the_chain():
    node = nodes.MiniMaxH3TimelineShot()
    (chain,) = node.add("s1", 5.0, "  walks  ", 12, "", "bridge", 0)
    (chain,) = node.add("s2", 4.0, "runs", -1, "s2__3__abcdef01.safetensors", "cut", 107, shots=chain)
    assert chain[0] == {"kind": "shot", "id": "s1", "seconds": 5.0, "text": "walks",
                        "cut_verb": "the camera cuts to", "seed": 12, "lock": None,
                        "join": "bridge", "frames": None}
    assert chain[1]["lock"] == "s2__3__abcdef01.safetensors" and chain[1]["frames"] == 107
    assert nodes.is_timeline(chain)


def test_take_names_are_validated():
    ts = store.TakeStore("proj", "/tmp/nowhere")
    for bad in ("../x.safetensors", "a__1__xyz.safetensors", "a.safetensors", None):
        with pytest.raises(ValueError):
            ts.path(bad)
    assert ts.take_name("c 1/../x", 5, "0123456789abcdef") == "c_1_.._x__5__01234567.safetensors" \
        or store.TAKE_NAME_RE.fullmatch(ts.take_name("c 1/../x", 5, "0123456789abcdef"))


def test_short_first_take_is_a_clear_error(monkeypatch):
    with pytest.raises(ValueError, match="shorter than the 22-frame overlap"):
        run(monkeypatch, [shot("a", frames=5), shot("b")])


def test_unchanged_shot_between_locked_ones_keeps_its_take(monkeypatch):
    shots = [shot(x) for x in "abc"]
    out, rows, _ = run(monkeypatch, shots)
    nodes.clear_segment_cache()
    locked = locked_from(shots, rows, {"a", "c"})
    out2, rows2, stub = run(monkeypatch, locked)
    assert stub.calls == [] and rows2[1]["take"] == rows[1]["take"]
    assert rows2[1]["pins"]["end"] is None and rows2[1]["status"] == "reused"


def test_new_end_pin_is_the_reason_not_the_start(monkeypatch):
    shots = [shot(x) for x in "abc"]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "c"})
    locked[1] = dict(locked[1], seed=31)
    _, rows2, _ = run(monkeypatch, locked)
    assert rows2[1]["reason"] == "seed changed"


def test_take_deleted_mid_run_is_a_clear_error(monkeypatch):
    shots = [shot(x) for x in "ab"]
    out, rows, _ = run(monkeypatch, shots)
    locked = locked_from(shots, rows, {"a", "b"})
    real = store.TakeStore.meta
    monkeypatch.setattr(store.TakeStore, "load", lambda self, n: (_ for _ in ()).throw(FileNotFoundError(n)))
    with pytest.raises(ValueError, match="Shot 1's take is missing"):
        run(monkeypatch, locked)


def test_ids_containing_double_underscores_do_not_mix_takes(monkeypatch):
    run(monkeypatch, [shot("a"), shot("a__1")])
    ts = store.TakeStore("proj", nodes.STORE_ROOT)
    assert [store.take_shot(m["name"]) for m in ts.list("a")] == ["a"]
    assert [store.take_shot(m["name"]) for m in ts.list("a__1")] == ["a__1"]


def test_all_locked_with_reference_audio_runs_without_the_audio_vae(monkeypatch):
    """A fully locked / reused run skips the lazy audio VAE; reference audio
    connected to the node must not turn that into an error."""
    voice = {"ref_audios": {"ref_audio_0": {"waveform": torch.zeros(1, 2, 2 * T.SR),
                                            "sample_rate": T.SR}}}
    shots = [shot("a"), shot("b")]
    out, rows, _ = run(monkeypatch, shots, audio_vae=T.MockAudioVae(), **voice)
    locked = locked_from(shots, rows, {"a", "b"})
    out2, rows2, stub = run(monkeypatch, locked, audio_vae=None, **voice)
    assert stub.calls == [] and [r["status"] for r in rows2] == ["locked", "locked"]
    # memory reuse, the other way a run needs nothing loaded
    out3, rows3, stub3 = run(monkeypatch, shots, audio_vae=None, **voice)
    assert stub3.calls == []


def test_rendering_with_reference_audio_still_needs_the_audio_vae(monkeypatch):
    voice = {"ref_audios": {"ref_audio_0": {"waveform": torch.zeros(1, 2, 2 * T.SR),
                                            "sample_rate": T.SR}}}
    with pytest.raises(ValueError, match="audio_vae"):
        run(monkeypatch, [shot("a")], **voice)
