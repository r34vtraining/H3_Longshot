"""Segment reuse and per-Shot seeds.

The sampler stub is deterministic: a window's content depends only on its
seed, its prompt, its size, and the tail guide it continues from — like a real
sampler with fixed settings. So a run that reuses segments must produce
exactly what a fresh run would."""
import hashlib

import pytest
import torch

import test_integration as T
from comfy.nested_tensor import NestedTensor

nodes, P = T.nodes, T.P


class Stub:
    def __init__(self, fail_on=None):
        self.calls = []          # (seed, prompt) per sampled segment
        self.fail_on = fail_on

    def __call__(self, model, noise, sampler, sigmas, positive, latent):
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise KeyboardInterrupt("interrupted")
        meta = positive[0][1]
        tail = [k["latent"] for k in meta.get("minimax_keyframes", []) if k.get("latent") is not None]
        prompt = self.clip.prompts[-1]
        self.calls.append((getattr(noise, "seed", None), prompt))
        h = hashlib.sha256(f"{noise.seed}|{prompt}".encode())
        if tail:
            h.update(tail[0].numpy().tobytes())
        g = torch.Generator().manual_seed(int.from_bytes(h.digest()[:8], "little"))
        tv, ta = latent["samples"].tensors
        v = torch.randn(tv.shape, generator=g)
        a = torch.randn(ta.shape, generator=g)
        if tail:                      # a real continuation reproduces its guide
            v[:, :, :tail[0].shape[2]] = tail[0]
        return {"samples": NestedTensor((v, a))}


# Loader outputs are the same objects from one queue to the next in ComfyUI,
# so the tests reuse one CLIP and one VAE across runs too.
CLIP, VAE = T.MockClip(), T.MockVae()


def go(monkeypatch, durations, texts=None, seeds=None, reuse=True, noise_seed=100,
       stub=None, model=None, **extra):
    stub = stub or Stub()
    stub.clip = CLIP
    monkeypatch.setattr(nodes, "_sample_window", stub)
    kw = T.node_kwargs(durations, texts=texts, clip=CLIP, vae=VAE,
                       noise=T.cs.Noise_RandomNoise(noise_seed),
                       reuse_segments=reuse, **extra)
    kw["model"] = model
    if seeds:
        for shot, sd in zip(kw["prompt"]["shots"], seeds):
            shot["seed"] = sd
    out = T.call_node(**kw)
    return out, stub


def latent(out):
    return out[0]["samples"].tensors


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(latent(a), latent(b)))


def test_identical_rerun_samples_nothing(monkeypatch):
    first, s1 = go(monkeypatch, [5, 5, 5])
    again, s2 = go(monkeypatch, [5, 5, 5])
    assert len(s1.calls) == 3 and s2.calls == []
    assert same(first, again)
    assert again[3].count("· reused") == 3


def test_building_shot_by_shot_renders_only_the_new_shot(monkeypatch):
    texts = ["a", "b", "c", "d"]
    stubs = []
    for n in (2, 3, 4):
        out, st = go(monkeypatch, [5] * n, texts=texts[:n])
        stubs.append(st)
    assert [len(s.calls) for s in stubs] == [2, 1, 1]
    nodes.clear_segment_cache()
    fresh, _ = go(monkeypatch, [5] * 4, texts=texts)
    assert same(out, fresh), "reused chain must equal a fresh render"


def test_changing_a_middle_shot_renders_it_and_everything_after(monkeypatch):
    texts = ["a", "b", "c", "d"]
    go(monkeypatch, [5] * 4, texts=texts)
    out, st = go(monkeypatch, [5] * 4, texts=["a", "b", "C!", "d"])
    assert len(st.calls) == 2
    plan = out[3]
    assert "Segment 1:" in plan
    lines = [l for l in plan.splitlines() if l.startswith("Segment ")]
    assert lines[0].endswith("reused (memory)") and lines[1].endswith("reused (memory)")
    assert lines[2].endswith("will render — prompt changed")
    assert lines[3].endswith("will render — follows a changed segment")
    nodes.clear_segment_cache()
    fresh, _ = go(monkeypatch, [5] * 4, texts=["a", "b", "C!", "d"])
    assert same(out, fresh)


def test_shot_seed_rerolls_only_that_shot(monkeypatch):
    go(monkeypatch, [5] * 3)
    out, st = go(monkeypatch, [5] * 3, seeds=[-1, -1, 777])
    assert [s for s, _p in st.calls] == [777]
    lines = [l for l in out[3].splitlines() if l.startswith("Segment ")]
    assert "seed 777 (Shot seed)" in lines[2] and lines[2].endswith("seed changed")
    assert "seed 101" in lines[1]


def test_base_seed_change_renders_everything(monkeypatch):
    go(monkeypatch, [5] * 3)
    out, st = go(monkeypatch, [5] * 3, noise_seed=5)
    assert len(st.calls) == 3
    assert out[3].count("will render — seed changed") == 3


def test_reuse_off_renders_everything_and_remembers_nothing(monkeypatch):
    go(monkeypatch, [5] * 2, reuse=False)
    assert not nodes._SEGMENT_CACHE
    out, st = go(monkeypatch, [5] * 2, reuse=False)
    assert len(st.calls) == 2 and out[3].count("reuse off") == 2


def test_dry_run_reports_without_sampling_or_forgetting(monkeypatch):
    go(monkeypatch, [5] * 3)
    last = list(nodes._LAST_RUN)
    out, st = go(monkeypatch, [5] * 3, texts=["beat 1", "beat 2", "new"], dry_run=True)
    assert st.calls == []
    assert out[3].count("· reused") == 2 and "will render — prompt changed" in out[3]
    assert nodes._LAST_RUN == last


def test_interrupted_run_resumes_from_where_it_stopped(monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        go(monkeypatch, [5] * 4, stub=Stub(fail_on=2))
    out, st = go(monkeypatch, [5] * 4)
    assert len(st.calls) == 2       # segments 1-2 were kept
    nodes.clear_segment_cache()
    fresh, _ = go(monkeypatch, [5] * 4)
    assert same(out, fresh)


def test_a_different_model_object_invalidates(monkeypatch):
    go(monkeypatch, [5] * 2, model=object())
    out, st = go(monkeypatch, [5] * 2, model=object())
    assert len(st.calls) == 2
    assert "model, CLIP or VAE changed" in out[3]


def test_song_and_references_are_part_of_the_fingerprint(monkeypatch):
    img = torch.rand(1, T.H, T.W, 3)
    go(monkeypatch, [5] * 2, ref_images={"ref_image_0": img})
    out, st = go(monkeypatch, [5] * 2, ref_images={"ref_image_0": img + 0.01})
    assert len(st.calls) == 2 and "references changed" in out[3]


def test_large_tensors_hash_by_sample_and_still_detect_changes():
    big = torch.zeros(nodes._FULL_HASH_ELEMENTS * 2 + 7)
    d0 = nodes._digest(big)
    big[123] = 1.0           # off the sampled stride, still changes the sum
    assert nodes._digest(big) != d0


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(nodes, "SEGMENT_CACHE_MAX", 3)
    go(monkeypatch, [5] * 5)
    assert len(nodes._SEGMENT_CACHE) == 3


class CustomNoise:
    """A custom-node noise source: not ComfyUI's class, but seeded the same way."""
    def __init__(self, seed):
        self.seed = seed
        self.flavor = "pink"

    def generate_noise(self, latent):
        return torch.zeros(1)


def test_custom_noise_nodes_get_increment_and_shot_seeds(monkeypatch):
    stub = Stub()
    stub.clip = CLIP
    monkeypatch.setattr(nodes, "_sample_window", stub)
    kw = T.node_kwargs([5] * 3, clip=CLIP, vae=VAE, noise=CustomNoise(40))
    kw["prompt"]["shots"][2]["seed"] = 0
    out = T.call_node(**kw)
    assert [s for s, _p in stub.calls] == [40, 41, 0]
    assert "seed 0 (Shot seed)" in out[3]


def test_disable_noise_is_left_alone(monkeypatch):
    seen = []
    def sample(model, noise, *a):
        seen.append(noise)
        return Stub.__call__(stub, model, noise, *a)
    stub = Stub()
    stub.clip = CLIP
    monkeypatch.setattr(nodes, "_sample_window", sample)
    empty = T.cs.Noise_EmptyNoise()
    kw = T.node_kwargs([5] * 2, clip=CLIP, vae=VAE, noise=empty)
    kw["prompt"]["shots"][1]["seed"] = 9
    T.call_node(**kw)
    assert all(n is empty for n in seen)


def test_random_noise_from_comfyuis_own_loader_is_recognised(monkeypatch):
    """ComfyUI loads comfy_extras/*.py under a path-based module name, so the
    RandomNoise a real workflow passes in is a different class object from
    comfy_extras.nodes_custom_sampler.Noise_RandomNoise. Long Shot must not rely
    on class identity (it used to, and silently ignored Shot seeds)."""
    import importlib.util
    import os
    import sys
    path = T.cs.__file__
    name = os.path.splitext(path)[0]           # how ComfyUI's load_custom_node names it
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    noise = mod.RandomNoise.execute(1722).args[0]
    assert not isinstance(noise, T.cs.Noise_RandomNoise)   # the trap

    stub = Stub()
    stub.clip = CLIP
    monkeypatch.setattr(nodes, "_sample_window", stub)
    kw = T.node_kwargs([5] * 3, clip=CLIP, vae=VAE, noise=noise)
    kw["prompt"]["shots"][2]["seed"] = 0
    out = T.call_node(**kw)
    assert [s for s, _p in stub.calls] == [1722, 1723, 0]
    assert "seed 0 (Shot seed)" in out[3]
