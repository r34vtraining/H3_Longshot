"""Carrier -> the real Apply H3 RefMod -> Long Shot extra_refs.

Runs against an installed copy of Luisacaotica/ComfyUI-MiniMaxH3Mod. Set
REFMOD_PACK to its folder; skipped otherwise.
"""
import asyncio
import logging
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REFMOD_PACK = os.environ.get("REFMOD_PACK")
pytestmark = pytest.mark.skipif(
    not REFMOD_PACK or not os.path.isdir(REFMOD_PACK or ""),
    reason="set REFMOD_PACK to the ComfyUI-MiniMaxH3Mod folder to run")

import test_integration as T  # noqa: E402  (boots ComfyUI, shares the mocks)
import torch  # noqa: E402


@pytest.fixture(scope="module")
def refmod():
    import nodes as comfy_nodes
    assert asyncio.run(comfy_nodes.load_custom_node(REFMOD_PACK))
    apply = comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3RefModApply"]
    core = sys.modules[next(k for k in sys.modules if k.endswith(".core")
                            and hasattr(sys.modules[k], "H3RefMod"))]
    return apply, core.H3RefMod


def _mods(H3RefMod):
    hh, ww = T.H // 16, T.W // 16
    return [(H3RefMod(name="luma_face", kind="image", latent=torch.randn(1, 24, 1, hh, ww),
                      latent_h=hh, latent_w=ww, mode="encode"), 1.0),
            (H3RefMod(name="luma_walk", kind="video", latent=torch.randn(1, 24, 7, hh, ww),
                      latent_h=hh, latent_w=ww, latent_t=7, mode="encode"), 1.0)]


def _apply(apply, conditioning, mods):
    return apply.execute(conditioning=conditioning, mods=mods).args[0]


def test_carrier_feeds_refmod_without_double_counting(refmod, monkeypatch, caplog):
    apply, H3RefMod = refmod
    carrier, = T.nodes.MiniMaxH3RefModCarrier().carry()
    refmod_out = _apply(apply, carrier, _mods(H3RefMod))

    ref = torch.rand(1, T.H, T.W, 3)
    with caplog.at_level(logging.WARNING):
        r = T.run(monkeypatch, [7, 7], ref_images={"ref_image_0": ref}, extra_refs=refmod_out)
    assert "counted twice" not in caplog.text
    for c in r.st.calls:
        kinds = [b["kind"] for b in c.meta["minimax_refs"]]
        assert kinds == ["image", "image", "video"], "native <Picture 1>, then RefMod's two"
    v, a = r.out[0]["samples"].tensors
    assert torch.equal(v, r.gt_v) and torch.equal(a, r.gt_a)


def test_feeding_refmod_from_a_loaded_reference_node_warns(refmod, monkeypatch, caplog):
    """The wiring to avoid: a Reference to Video node with images connected."""
    apply, H3RefMod = refmod
    import comfy_extras.nodes_minimax_h3 as native
    ref = torch.rand(1, T.H, T.W, 3)
    loaded = native.MiniMaxH3ReferenceToVideo.execute(
        T.MockClip(), "", T.W, T.H, 124, vae=T.MockVae(),
        ref_images={"ref_image_0": ref}).args[0]
    refmod_out = _apply(apply, loaded, _mods(H3RefMod))
    with caplog.at_level(logging.WARNING):
        r = T.run(monkeypatch, [7, 7], ref_images={"ref_image_0": ref}, extra_refs=refmod_out)
    assert "counted twice" in caplog.text
    assert len(r.st.calls[0].meta["minimax_refs"]) == 4   # the image really is in twice
