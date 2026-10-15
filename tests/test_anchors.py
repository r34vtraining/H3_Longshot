"""Frame anchors: a Shot can begin from / end on an image. The image is that
Shot's next <Picture N> and is pinned at the join, unless a real pin from the
neighbouring Shot already covers it."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_integration as T  # noqa: E402
import test_timeline as TL  # noqa: E402

nodes, P = T.nodes, T.P
shot, run, OV = TL.shot, TL.run, TL.OV


class ItemsClip(T.MockClip):
    def __init__(self):
        super().__init__()
        self.items = []

    def tokenize(self, prompt, images=None, minimax_ref_items=None):
        self.items.append(len(minimax_ref_items or []))
        return super().tokenize(prompt, images, minimax_ref_items)


def img(v):
    return torch.full((1, 64, 96, 3), float(v))


def anchored(sid, a=None, b=None, **kw):
    return dict(shot(sid, **kw), start_image=a, end_image=b)


def idx(call):
    return sorted(int(k["resolved_frame_index"]) for k in call["keyframes"] if k.get("latent") is not None)


def test_a_shot_on_its_own_starts_and_ends_on_its_images(monkeypatch):
    clip = ItemsClip()
    monkeypatch.setattr(TL, "CLIP", clip)
    out, rows, stub = run(monkeypatch, [anchored("b", img(0.2), img(0.8), seconds=5)], clip=clip)
    w = rows[0]["window_frames"]
    assert idx(stub.calls[0]) == [0, w - 1]
    assert clip.items[-1] == 2, "both anchors are presented as pictures"


def test_continuing_from_the_shot_before_skips_the_start_image(monkeypatch):
    shots = [shot("a", 5), anchored("b", img(0.2), img(0.8), seconds=5), shot("c", 5)]
    out, rows, stub = run(monkeypatch, shots)
    b = stub.calls[1]
    wb = rows[1]["window_frames"]
    assert 0 in idx(b), "the start pin from Shot a"
    assert (OV - 1) not in idx(b), "no anchor keyframe inside the shared zone"
    assert wb - 1 in idx(b), "ends on its image (nothing pinned after it yet)"


def test_after_a_cut_the_image_sits_where_the_shot_begins(monkeypatch):
    shots = [shot("a", 5), anchored("b", img(0.2), None, seconds=5, join="cut")]
    out, rows, stub = run(monkeypatch, shots)
    assert idx(stub.calls[1]) == [OV - 1], "the Shot's own frames begin at the end of its shared zone"


def test_changing_an_anchor_makes_a_new_take(monkeypatch):
    _, rows1, _ = run(monkeypatch, [anchored("b", img(0.2), None, seconds=5)])
    _, rows2, stub = run(monkeypatch, [anchored("b", img(0.3), None, seconds=5)])
    assert len(stub.calls) == 1 and rows1[0]["take"] != rows2[0]["take"]
    _, rows3, stub3 = run(monkeypatch, [anchored("b", img(0.3), None, seconds=5)])
    assert len(stub3.calls) == 0 and rows3[0]["take"] == rows2[0]["take"], "same anchor: reused"


def test_timeline_shot_node_carries_the_images():
    (chain,) = nodes.MiniMaxH3TimelineShot().add("x", 5.0, "t", start_image=torch.zeros(3, 8, 8, 3))
    assert chain[0]["start_image"].shape[0] == 1 and "end_image" not in chain[0]


def test_a_context_image_is_a_picture_but_never_pinned(monkeypatch):
    clip = ItemsClip()
    monkeypatch.setattr(TL, "CLIP", clip)
    shots = [shot("a", 5), dict(shot("b", 5, join="cut"), context_image=img(0.5))]
    out, rows, stub = run(monkeypatch, shots, clip=clip)
    assert idx(stub.calls[1]) == [], "nothing pinned: a cut, and the context image is only a reference"
    assert clip.items[-1] == 1, "presented as one more picture"
    assert rows[1]["seam"] == "cut"


def test_changing_the_context_image_makes_a_new_take(monkeypatch):
    def sh(v):
        return [dict(shot("b", 5, join="cut"), context_image=img(v))]
    _, r1, _ = run(monkeypatch, sh(0.2))
    _, r2, stub = run(monkeypatch, sh(0.4))
    assert len(stub.calls) == 1 and r1[0]["take"] != r2[0]["take"]
    _, r3, stub3 = run(monkeypatch, sh(0.4))
    assert len(stub3.calls) == 0, "same context: reused"


def test_timeline_shot_node_carries_the_context_image():
    (chain,) = nodes.MiniMaxH3TimelineShot().add("x", 5.0, "t", context_image=torch.zeros(4, 8, 8, 3))
    assert chain[0]["context_image"].shape[0] == 1 and "start_image" not in chain[0]


def test_timeline_shot_node_carries_its_own_retention_only_when_given():
    (chain,) = nodes.MiniMaxH3TimelineShot().add("x", 5.0, "t", retention="  <Picture 2> (x): weak_reference.  ")
    assert chain[0]["retention"] == "<Picture 2> (x): weak_reference."
    (chain,) = nodes.MiniMaxH3TimelineShot().add("x", 5.0, "t")
    assert "retention" not in chain[0]
