"""Dry run through ComfyUI's real executor.

A real graph: stub sources -> Long Shot -> a recording node on the latent and
another on the plan. With dry_run on, the latent branch must not run at all
(no decode, no saved clip) while the plan branch still does.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_integration as T  # noqa: E402  (boots ComfyUI, loads the pack)
import comfy_extras.nodes_custom_sampler as cs  # noqa: E402

RAN = []


class _Source:
    """Hands Long Shot stand-ins for model, clip, vae, noise, sampler, sigmas."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "NOISE", "SAMPLER", "SIGMAS", "MMH3_LONGSHOT")
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self):
        return (None, T.MockClip(), T.MockVae(), cs.Noise_RandomNoise(5), None, None,
                T.make_bundle([7, 7, 7]))


def _recorder(kind):
    class _Rec:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": (kind,)}}
        RETURN_TYPES = ()
        OUTPUT_NODE = True
        FUNCTION = "go"
        CATEGORY = "test"

        def go(self, value):
            RAN.append(kind)
            return ()
    return _Rec


class _Server:
    client_id = None
    last_node_id = None

    def send_sync(self, *a, **k):
        pass


@pytest.fixture
def executor(monkeypatch):
    import execution
    import nodes as comfy_nodes
    for name, cls in {"TestSource": _Source, "TestLatentSink": _recorder("LATENT"),
                      "TestPlanSink": _recorder("STRING")}.items():
        monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, name, cls)
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3LongShot",
                        T.nodes.MiniMaxH3LongShot)
    RAN.clear()
    # Same construction as ComfyUI's main.py, with its default cache settings.
    return execution.PromptExecutor(_Server(), cache_args={"lru": 0, "ram": 0,
                                                           "ram_inactive": 0})


def _graph(dry_run):
    src = ["1", 0]
    return {
        "1": {"class_type": "TestSource", "inputs": {}},
        "2": {"class_type": "MiniMaxH3LongShot", "inputs": {
            "model": ["1", 0], "clip": ["1", 1], "vae": ["1", 2], "noise": ["1", 3],
            "sampler": ["1", 4], "sigmas": ["1", 5], "prompt": ["1", 6],
            "width": 256, "height": 128, "overlap_frames": 22, "seed_mode": "increment",
            "dry_run": dry_run, "ref_image_size": "match"}},
        "3": {"class_type": "TestLatentSink", "inputs": {"value": ["2", 0]}},
        "4": {"class_type": "TestPlanSink", "inputs": {"value": ["2", 3]}},
    }


def test_dry_run_skips_everything_downstream_of_the_latent(executor):
    executor.execute(_graph(True), "dry", {}, ["3", "4"])
    assert executor.success, executor.status_messages
    assert RAN == ["STRING"], "plan reached its node; the latent branch never ran"


def test_real_run_feeds_the_latent_downstream(executor, monkeypatch):
    plan = T.P.plan_from_durations([7, 7, 7], 22)
    sim, *_ = T.make_simulator(plan)
    monkeypatch.setattr(T.nodes, "_sample_window", sim)
    executor.execute(_graph(False), "real", {}, ["3", "4"])
    assert executor.success, executor.status_messages
    assert sorted(RAN) == ["LATENT", "STRING"]
