"""Clip Shots: a real video as a piece of the timeline. Clips are never
sampled; the generated Shots next to them are pinned into and out of them
('bridge') or meet them with a hard cut ('cut')."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_integration as T  # noqa: E402
import test_timeline as TL  # noqa: E402

nodes, P = T.nodes, T.P
store = TL.store
OV, OVT = TL.OV, TL.OVT
shot, run, lat, locked_from = TL.shot, TL.run, TL.lat, TL.locked_from


def make_clip(frames=107, seed=1, w=T.W, h=T.H):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(1, 24, P.video_tokens(frames), h // 16, w // 16, generator=g)
    a = torch.randn(1, 32, 2, P.audio_at(frames), generator=g)
    px = torch.rand(frames, h, w, 3, generator=g).to(torch.float16)
    return {"video": v, "audio": a, "frames": frames, "width": w, "height": h, "pixels": px,
            "digest": nodes._digest([v, a])}


def clip_shot(sid, clip, join="bridge"):
    (chain,) = nodes.MiniMaxH3ClipShot().add(clip, sid, join)
    return chain[0]


def kf(stub, call, at_end):
    ks = [k for k in stub.calls[call]["keyframes"] if k.get("latent") is not None]
    return [k for k in ks if (k["resolved_frame_index"] > 0) == at_end]


def test_a_clip_between_two_new_shots_bridges_both_ways(monkeypatch):
    c = make_clip()
    out, rows, stub = run(monkeypatch, [shot("a"), clip_shot("c", c), shot("b")])
    assert len(stub.calls) == 2, "the clip is never sampled"
    assert [r["kind"] for r in rows] == ["shot", "clip", "shot"]
    assert rows[1]["status"] == "clip" and rows[1]["take"] is None
    end = kf(stub, 0, True)[0]
    assert torch.equal(end["latent"].to(c["video"]), c["video"][:, :, :OVT]), "a leads into the clip's head"
    start = kf(stub, 1, False)[0]
    assert torch.equal(start["latent"].to(c["video"]), c["video"][:, :, -OVT:]), "b comes out of its tail"
    # the stitched film holds the clip's own tokens (after its shared zone) bit-exactly
    s, e = TL.visible_ranges(rows)[1]
    assert torch.equal(lat(out)[0][:, :, s:e], c["video"][:, :, OVT:].to(lat(out)[0]))
    assert [r["seam"] for r in rows] == [None, "ok", "ok"]


def test_replace_a_middle_shot_with_a_clip_renders_only_its_neighbours(monkeypatch):
    shots = [shot(x) for x in "abcde"]
    out, rows, _ = run(monkeypatch, shots)
    before = lat(out)[0].float()
    c = make_clip(frames=rows[2]["window_frames"])
    timeline = locked_from(shots, rows, set("ae"))
    timeline[2] = clip_shot("clip", c)
    out2, rows2, stub = run(monkeypatch, timeline)
    assert len(stub.calls) == 2, "Shot 2 and Shot 4 re-render; 1 and 5 stay"
    assert rows2[1]["pins"]["end"]["kind"] == "head"
    assert rows2[3]["pins"]["end"]["kind"] == "old_tail", "Shot 4 still meets Shot 5's old start"
    after = lat(out2)[0].float()
    r = TL.visible_ranges(rows2)
    assert torch.allclose(after[:, :, :r[1][0]], before[:, :, :r[1][0]], atol=2e-3), "Shot 1 unchanged"
    assert torch.allclose(after[:, :, r[3][1] - OVT:], before[:, :, r[3][1] - OVT:], atol=2e-3), \
        "Shot 5 (and the zone it starts from) unchanged"
    assert [x["seam"] for x in rows2] == [None, "ok", "ok", "ok", "ok"]


def test_clip_first_keeps_every_frame_and_the_next_shot_follows_it(monkeypatch):
    c = make_clip(frames=73)
    out, rows, stub = run(monkeypatch, [clip_shot("c", c), shot("a")])
    assert rows[0]["frames"] == 73 and len(stub.calls) == 1
    assert torch.equal(lat(out)[0][:, :, :P.video_tokens(73)], c["video"].to(lat(out)[0]))
    assert torch.equal(kf(stub, 0, False)[0]["latent"].to(c["video"]), c["video"][:, :, -OVT:])


def test_clip_last(monkeypatch):
    c = make_clip()
    out, rows, stub = run(monkeypatch, [shot("a"), clip_shot("c", c)])
    assert len(stub.calls) == 1 and rows[1]["seam"] == "ok"
    assert P.is_valid_frame_count(out[1])


def test_adjacent_clips_always_cut(monkeypatch):
    c1, c2 = make_clip(seed=1), make_clip(seed=2)
    out, rows, stub = run(monkeypatch, [shot("a"), clip_shot("c1", c1), clip_shot("c2", c2)])
    assert rows[2]["join"] == "cut" and rows[2]["seam"] == "cut"
    assert len(stub.calls) == 1


def test_cut_join_keeps_the_neighbour_and_hides_the_clips_first_zone(monkeypatch):
    shots = [shot("a")]
    out, rows, _ = run(monkeypatch, shots)
    a_take = rows[0]["take"]
    c = make_clip()
    out2, rows2, stub = run(monkeypatch, [shot("a", lock=a_take, seed=rows[0]["seed"]),
                                          clip_shot("c", c, join="cut")])
    assert stub.calls == [] and rows2[1]["seam"] == "cut" and rows2[0]["pins"]["end"] is None
    assert rows2[1]["frames"] == c["frames"] - OV


def test_an_unbridged_clip_after_a_locked_shot_is_a_reported_hard_cut(monkeypatch):
    out, rows, _ = run(monkeypatch, [shot("a")])
    out2, rows2, stub = run(monkeypatch, [shot("a", lock=rows[0]["take"], seed=rows[0]["seed"]),
                                          clip_shot("c", make_clip())])
    assert stub.calls == [] and rows2[1]["seam"] == "mismatch"


def test_clip_at_another_size_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="prepared at"):
        run(monkeypatch, [shot("a"), clip_shot("c", make_clip(w=T.W * 2))])


def test_lazy_inputs_skip_the_model_when_only_clips_and_takes(monkeypatch):
    out, rows, _ = run(monkeypatch, [shot("a"), clip_shot("c", make_clip())])
    locked = [shot("a", lock=rows[0]["take"], seed=rows[0]["seed"]), clip_shot("c", make_clip())]
    kw = T.node_kwargs([], noise=T.cs.Noise_RandomNoise(100))
    kw["prompt"]["shots"] = locked
    kw.update(save_to_disk=True, cache_name="proj", model=None, clip=None, vae=None)
    monkeypatch.setattr(nodes.disk, "linked_inputs", lambda *a: {"model", "clip", "vae", "sigmas"})
    monkeypatch.setattr(nodes.disk, "recipe_components", lambda *a: TL.RECIPE)
    assert nodes.lazy_requests(kw, {}, "1") == []


def test_clip_pixels_swaps_the_original_frames_back_in(monkeypatch):
    c = make_clip(frames=73)
    out, rows, _ = run(monkeypatch, [shot("a"), clip_shot("c", c)])
    latent = out[0]
    total = out[1]
    decoded = torch.zeros(total, T.H, T.W, 3)
    (swapped,) = nodes.MiniMaxH3ClipPixels().swap(decoded, latent, True)
    start = rows[1]["start_frame"]
    assert torch.equal(swapped[start:start + rows[1]["frames"]], c["pixels"][OV:].float())
    assert torch.equal(swapped[:start], decoded[:start])
    (off,) = nodes.MiniMaxH3ClipPixels().swap(decoded, latent, False)
    assert off is decoded


class FrameVae:
    """Encodes frames to the right number of tokens, each the mean of its frames."""
    def encode(self, pixels):
        n = pixels.shape[0]
        k = P.video_tokens(n)
        means = torch.tensor([float(pixels[min(n - 1, t * 4)].mean()) for t in range(k)])
        return means.view(1, 1, k, 1, 1).expand(1, 24, k, pixels.shape[1] // 16,
                                                  pixels.shape[2] // 16).clone()


def test_clip_node_trims_resizes_and_encodes(monkeypatch):
    frames = torch.rand(130, 90, 160, 3)
    (c,) = nodes.MiniMaxH3Clip().prepare(frames, FrameVae(), T.MockAudioVae(), T.W, T.H, 0, "mute")
    assert c["frames"] == 124 and c["pixels"].shape == (124, T.H, T.W, 3)
    assert c["video"].shape[2] == P.video_tokens(124)
    assert c["audio"].shape[-1] == P.audio_at(124)
    (c2,) = nodes.MiniMaxH3Clip().prepare(frames, FrameVae(), T.MockAudioVae(), T.W, T.H, 80, "clip",
                                          audio={"waveform": torch.zeros(1, 2, T.SR * 6),
                                                 "sample_rate": T.SR})
    assert c2["frames"] == 73, "trimmed to 80 frames, snapped down to 17k + 5"
    assert c2["audio"].shape[-1] == P.audio_at(73)
    with pytest.raises(ValueError, match="at least 5"):
        nodes.MiniMaxH3Clip().prepare(frames[:3], FrameVae(), T.MockAudioVae(), T.W, T.H, 0, "mute")


def test_a_short_clip_sound_is_padded_with_silence(monkeypatch):
    frames = torch.rand(107, 64, 64, 3)
    (c,) = nodes.MiniMaxH3Clip().prepare(frames, FrameVae(), T.MockAudioVae(), T.W, T.H, 0, "clip",
                                         audio={"waveform": torch.zeros(1, 2, T.SR), "sample_rate": T.SR})
    assert c["audio"].shape[-1] == P.audio_at(107)


@pytest.mark.parametrize("ov", [5, 22, 56])
def test_the_audio_across_a_bridged_clip_join_is_continuous(monkeypatch, ov):
    c = make_clip(frames=158)
    out, rows, _ = run(monkeypatch, [shot("a", 6), clip_shot("c", c)], overlap_frames=ov)
    a = lat(out)[1]
    seg = nodes.planner.plan_pieces([P.PieceSpec(6, window=rows[0]["window_frames"]),
                                     P.PieceSpec(0, window=158, locked=True)], ov).segments[1]
    zone0 = P.audio_at(seg.window_start)
    n = a.shape[-1] - zone0 - 1          # (the very last token can be the ±1 fit at the film's end)
    assert torch.equal(a[..., zone0:zone0 + n], c["audio"][..., :n].to(a)), \
        "the shared zone and the clip after it play the clip's own tokens, none repeated or lost"


def test_clips_get_no_trained_range_warnings(monkeypatch):
    out, rows, _ = run(monkeypatch, [shot("a"), clip_shot("c", make_clip(frames=600))])
    assert "H3's trained" not in out[3]


def test_first_frame_on_a_clip_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="starts with a clip"):
        run(monkeypatch, [clip_shot("c", make_clip()), shot("a")],
            first_frame=torch.zeros(1, T.H, T.W, 3))


def test_a_clip_re_encoded_after_a_restart_is_still_the_same_clip(monkeypatch):
    c = make_clip()
    out, rows, _ = run(monkeypatch, [shot("a"), clip_shot("c", c), shot("b")])
    again = dict(c, video=c["video"] + 1e-3)            # same source, slightly different encode
    locked = [shot("a", lock=rows[0]["take"], seed=rows[0]["seed"]), clip_shot("c", again),
              shot("b", lock=rows[2]["take"], seed=rows[2]["seed"])]
    out2, rows2, stub = run(monkeypatch, locked)
    assert stub.calls == [] and [r["seam"] for r in rows2] == [None, "ok", "ok"]
    assert rows2[1]["pins"]["start"] is None, "a clip has no start pin"
