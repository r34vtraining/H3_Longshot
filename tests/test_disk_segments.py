"""Saved segments (crash-proof resume): recipe fingerprints, the on-disk
store, its switches, and lazy model loading through the real executor."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_integration as T  # noqa: E402  (boots ComfyUI, loads the pack)
import test_reuse as R  # noqa: E402
import comfy_extras.nodes_custom_sampler as cs  # noqa: E402

nodes, P = T.nodes, T.P
store = __import__(T.PKG + ".store", fromlist=["store"])
RECIPE = {"model": "m" * 32, "sampler": "s" * 32}


def stat_stub(name):
    return [1234, 1700000000]


# ---------------------------------------------------------------------------
# A.1 Recipe fingerprints
# ---------------------------------------------------------------------------

def user_graph(ids=None, **change):
    """The user's model chain as an API prompt, with node ids from `ids`."""
    i = {k: k for k in ("unet", "sage", "shift", "turbo", "lora", "clip", "vv", "va", "sched",
                        "samp", "noise", "ls")}
    i.update(ids or {})
    v = dict(unet="H3\\minimax_h3_fl2va_pruned_bf16.safetensors", sage="auto", shift_video=12,
             turbo="H3\\Speed\\turbo.safetensors", lora_strength=0.8, steps=8,
             scheduler="beta57", sampler="er_sde", seed=1722)
    v.update(change)
    L = lambda k, slot=0: [i[k], slot]  # noqa: E731
    return {
        i["unet"]: {"class_type": "UNETLoader", "inputs": {"unet_name": v["unet"], "weight_dtype": "default"}},
        i["sage"]: {"class_type": "PathchSageAttentionKJ", "inputs": {"model": L("unet"), "sage_attention": v["sage"], "allow_compile": False}},
        i["shift"]: {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": L("sage"), "shift_video": v["shift_video"], "shift_audio": 3}},
        i["turbo"]: {"class_type": "MiniMaxH3TurboLoRA", "inputs": {"model": L("shift"), "lora_name": v["turbo"], "strength": 1.0}},
        i["lora"]: {"class_type": "LoraLoaderModelOnly", "inputs": {"model": L("turbo"), "lora_name": "style.safetensors", "strength_model": v["lora_strength"]}},
        i["clip"]: {"class_type": "CLIPLoader", "inputs": {"clip_name": "H3\\qwen.safetensors", "type": "minimax", "device": "default"}},
        i["vv"]: {"class_type": "VAELoader", "inputs": {"vae_name": "H3\\video_vae.safetensors"}},
        i["va"]: {"class_type": "VAELoader", "inputs": {"vae_name": "H3\\audio_vae.safetensors"}},
        i["sched"]: {"class_type": "BasicScheduler", "inputs": {"model": L("lora"), "scheduler": v["scheduler"], "steps": v["steps"], "denoise": 1}},
        i["samp"]: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": v["sampler"]}},
        i["noise"]: {"class_type": "RandomNoise", "inputs": {"noise_seed": v["seed"]}},
        i["ls"]: {"class_type": "MiniMaxH3LongShot", "inputs": {
            "model": L("lora"), "clip": L("clip"), "vae": L("vv"), "audio_vae": L("va"),
            "noise": L("noise"), "sampler": L("samp"), "sigmas": L("sched"),
            "width": 256, "height": 128}},
    }, i["ls"]


def recipe(graph_and_id, file_stat=stat_stub):
    graph, ls = graph_and_id
    return store.recipe_components(graph, ls, file_stat)


def test_recipe_is_identical_with_different_node_ids():
    a = recipe(user_graph())
    renumbered = {k: str(n) for n, k in enumerate(["unet", "sage", "shift", "turbo", "lora",
                                                   "clip", "vv", "va", "sched", "samp",
                                                   "noise", "ls"], start=100)}
    b = recipe(user_graph(renumbered))
    nested = {k: f"359:{n}" for k, n in renumbered.items()}      # inside a subgraph
    c = recipe(user_graph(nested))
    assert a == b == c and a["model"] and a["sampler"]


@pytest.mark.parametrize("change,part", [
    ({"unet": "H3\\other.safetensors"}, "model"), ({"lora_strength": 0.9}, "model"),
    ({"shift_video": 8}, "model"), ({"sage": "disabled"}, "model"),
    ({"turbo": "H3\\Speed\\turbo_v2.safetensors"}, "model"),
    ({"steps": 20}, "sampler"), ({"scheduler": "simple"}, "sampler"),
    ({"sampler": "euler"}, "sampler")])
def test_any_upstream_literal_changes_the_recipe(change, part):
    a, b = recipe(user_graph()), recipe(user_graph(**change))
    assert a[part] != b[part]


def test_noise_seed_is_not_part_of_the_recipe():
    # seeds are fingerprinted per segment already; the base seed lives there
    assert recipe(user_graph()) == recipe(user_graph(seed=99))


def test_a_replaced_model_file_changes_the_recipe():
    a = recipe(user_graph(), lambda n: [1234, 1700000000])
    b = recipe(user_graph(), lambda n: [1234, 1700000999] if "fl2va" in n else [1234, 1700000000])
    c = recipe(user_graph(), lambda n: [9999, 1700000000] if "fl2va" in n else [1234, 1700000000])
    assert a["model"] != b["model"] and a["model"] != c["model"]
    # (the sigmas come from the model too, so the sampler part changes with it)


def test_windows_and_linux_file_names_give_the_same_recipe():
    g, ls = user_graph()
    g2, _ = user_graph(unet="H3/minimax_h3_fl2va_pruned_bf16.safetensors")
    assert store.recipe_components(g, ls, stat_stub) == store.recipe_components(g2, ls, stat_stub)


def test_no_recipe_without_the_prompt():
    assert store.recipe_components(None, "ls") is None
    g, _ = user_graph()
    assert store.recipe_components(g, "nope") is None
    del g["unet"]                                  # dangling link (ephemeral node)
    assert store.recipe_components(g, "ls", stat_stub) is None


# ---------------------------------------------------------------------------
# A.2 / A.3 Saving, loading, switches
# ---------------------------------------------------------------------------

def run(monkeypatch, durations, stub=None, recipe=RECIPE, **extra):
    stub = stub or R.Stub()
    stub.clip = R.CLIP
    monkeypatch.setattr(nodes, "_sample_window", stub)
    kw = T.node_kwargs(durations, clip=R.CLIP, vae=R.VAE, noise=cs.Noise_RandomNoise(100), **extra)
    out, rows = nodes._run_long_shot(**kw, _recipe=recipe)
    return out, rows, stub


def seg_files():
    folder = os.path.join(nodes.STORE_ROOT, "default", "segments")
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def test_save_clear_memory_resume_from_disk(monkeypatch):
    first, rows, s1 = run(monkeypatch, [5, 5, 9])
    assert len(s1.calls) == 3 and len(seg_files()) == 3
    assert [r["source"] for r in rows] == [None] * 3

    nodes.clear_segment_cache()                    # = ComfyUI restarted
    again, rows, s2 = run(monkeypatch, [5, 5, 9])
    assert s2.calls == [], "nothing previously rendered is sampled again"
    assert [(r["status"], r["source"]) for r in rows] == [("reused", "disk")] * 3
    assert again[3].count("reused (disk)") == 3
    for x, y in zip(first[0]["samples"].tensors, again[0]["samples"].tensors):
        assert x.dtype == y.dtype
        assert torch.equal(y, x.half().to(x.dtype)), "exactly the fp16-rounded original"

    third, rows, s3 = run(monkeypatch, [5, 5, 9])  # now from memory
    assert s3.calls == [] and [r["source"] for r in rows] == ["memory"] * 3


def test_resumed_chain_continues_within_fp16_rounding(monkeypatch):
    """Shot 4 after a restart is guided by the fp16 copy of Shot 3's tail."""
    run(monkeypatch, [5, 5, 5])
    guides = {}

    class Rec(R.Stub):
        def __call__(self, model, noise, sampler, sigmas, positive, latent):
            kf = [k for k in positive[0][1].get("minimax_keyframes", []) if k.get("latent") is not None]
            guides[self.tag] = kf[0]["latent"].clone()
            return super().__call__(model, noise, sampler, sigmas, positive, latent)

    a = Rec(); a.tag = "memory"
    run(monkeypatch, [5, 5, 5, 5], stub=a, save_to_disk=False)   # Shot 4 not saved
    nodes.clear_segment_cache()
    b = Rec(); b.tag = "disk"
    run(monkeypatch, [5, 5, 5, 5], stub=b)
    assert len(a.calls) == len(b.calls) == 1
    assert torch.allclose(guides["memory"], guides["disk"], rtol=1e-3, atol=1e-3)
    assert torch.equal(guides["disk"], guides["memory"].half().float())


def test_progress_events_say_where_a_segment_came_from(monkeypatch):
    got = []
    run(monkeypatch, [5, 5])
    nodes.clear_segment_cache()
    monkeypatch.setattr(nodes, "_notify", got.append)
    run(monkeypatch, [5, 5, 5])
    assert [(e["segment"], e["status"], e.get("source")) for e in got] == [
        (1, "reused", "disk"), (2, "reused", "disk"), (3, "rendering", None), (3, "done", None)]


def test_files_are_fp16_with_metadata(monkeypatch):
    from safetensors import safe_open
    run(monkeypatch, [5])
    path = os.path.join(nodes.STORE_ROOT, "default", "segments", seg_files()[0])
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        assert f.get_tensor("video").dtype == torch.float16
        assert f.get_tensor("audio").dtype == torch.float16
    assert meta["segment"] == "1" and meta["seed"] == "100" and meta["longshot_version"]
    assert meta["dtype"] == "float32" and "video_shape" in meta and "created" in meta


def test_interrupted_write_is_ignored_and_cleaned_up(monkeypatch):
    run(monkeypatch, [5, 5])
    folder = os.path.join(nodes.STORE_ROOT, "default", "segments")
    victim = sorted(os.listdir(folder))[0]
    # a crash mid-write: only the .tmp exists for that fingerprint
    os.replace(os.path.join(folder, victim), os.path.join(folder, victim + ".tmp"))
    nodes.clear_segment_cache()
    _, rows, stub = run(monkeypatch, [5, 5], dry_run=True)
    assert "disk" in [r["source"] for r in rows] and None in [r["source"] for r in rows]
    _, rows, stub = run(monkeypatch, [5, 5])
    assert not [f for f in os.listdir(folder) if f.endswith(".tmp")], ".tmp cleaned up"
    assert len(os.listdir(folder)) == 2


def test_corrupt_or_wrong_shape_file_is_ignored(monkeypatch):
    run(monkeypatch, [5])
    folder = os.path.join(nodes.STORE_ROOT, "default", "segments")
    path = os.path.join(folder, os.listdir(folder)[0])
    with open(path, "r+b") as fh:
        fh.truncate(100)
    nodes.clear_segment_cache()
    _, rows, stub = run(monkeypatch, [5])
    assert len(stub.calls) == 1, "a broken file is never used; the segment renders"
    nodes.clear_segment_cache()
    _, rows, stub = run(monkeypatch, [5])
    assert stub.calls == [] and rows[0]["source"] == "disk", "and the new file replaced it"


def test_memory_only_writes_nothing(monkeypatch):
    run(monkeypatch, [5, 5], save_to_disk=False)
    assert seg_files() == []


def test_off_reads_and_writes_nothing(monkeypatch):
    run(monkeypatch, [5, 5])
    before = seg_files()
    nodes.clear_segment_cache()
    _, rows, stub = run(monkeypatch, [5, 5], reuse_segments=False)
    assert len(stub.calls) == 2 and seg_files() == before
    assert [r["reason"] for r in rows] == ["reuse off"] * 2


def test_no_recipe_means_memory_only(monkeypatch):
    run(monkeypatch, [5, 5], recipe=None)
    assert seg_files() == []


def test_each_project_has_its_own_folder(monkeypatch):
    run(monkeypatch, [5], cache_name="mara-spaceport-chase")
    nodes.clear_segment_cache()
    run(monkeypatch, [5], cache_name="../../escape me")
    root = nodes.STORE_ROOT
    assert sorted(os.listdir(root)) == ["escape_me", "mara-spaceport-chase"]


def test_a_changed_model_recipe_does_not_reuse_disk(monkeypatch):
    run(monkeypatch, [5, 5])
    nodes.clear_segment_cache()
    _, rows, stub = run(monkeypatch, [5, 5], recipe={"model": "other", "sampler": "s" * 32})
    assert len(stub.calls) == 2


@pytest.mark.parametrize("name,want", [("default", "default"), ("mara-spaceport-chase", "mara-spaceport-chase"),
                                       ("..", "default"), ("a/b\\c", "a_b_c"), ("", "default"),
                                       ("Mara — chase", "Mara_chase")])
def test_cache_name_is_sanitised(name, want):
    assert store.safe_cache_name(name) == want


# ---------------------------------------------------------------------------
# Lazy model loading through the real executor
# ---------------------------------------------------------------------------

LOADED = []


class _Models:
    """Stands in for the model chain; records whether ComfyUI ran it."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": ("STRING", {"default": "x.safetensors"})}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "SIGMAS")
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, unet_name):
        LOADED.append(unet_name)
        return (object(), T.MockClip(), T.MockVae(), torch.linspace(1, 0, 5))


class _Rest:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("NOISE", "SAMPLER", "MMH3_LONGSHOT", "IMAGE")
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self):
        return (cs.Noise_RandomNoise(5), None, T.make_bundle([5, 5]), torch.full((1, 64, 64, 3), 0.5))


class _Sink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("LATENT",)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, value):
        LOADED.append("latent")
        return ()


class _Server:
    client_id = None
    last_node_id = None

    def send_sync(self, *a, **k):
        pass


def make_executor(monkeypatch):
    import execution
    import nodes as comfy_nodes
    for name, cls in {"TestModels": _Models, "TestRest": _Rest, "TestSink": _Sink}.items():
        monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, name, cls)
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3LongShot", nodes.MiniMaxH3LongShot)
    return execution.PromptExecutor(_Server(), cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})


def graph(dry_run, unet="x.safetensors"):
    return {
        "m": {"class_type": "TestModels", "inputs": {"unet_name": unet}},
        "r": {"class_type": "TestRest", "inputs": {}},
        "ls": {"class_type": "MiniMaxH3LongShot", "inputs": {
            "model": ["m", 0], "clip": ["m", 1], "vae": ["m", 2], "sigmas": ["m", 3],
            "noise": ["r", 0], "sampler": ["r", 1], "prompt": ["r", 2],
            "ref_images.ref_image_0": ["r", 3],
            "width": 256, "height": 128, "overlap_frames": 22, "seed_mode": "increment",
            "dry_run": dry_run, "reuse_segments": True, "ref_image_size": "match",
            "save_to_disk": True, "cache_name": "lazy"}},
        "out": {"class_type": "TestSink", "inputs": {"value": ["ls", 0]}},
    }


def test_model_is_only_loaded_when_a_segment_renders(monkeypatch):
    plan = P.plan_from_durations([5, 5], 22)
    sim, *_ = T.make_simulator(plan)
    monkeypatch.setattr(nodes, "_sample_window", sim)
    LOADED.clear()

    ex = make_executor(monkeypatch)
    ex.execute(graph(True), "dry", {}, ["out"])
    assert ex.success, ex.status_messages
    assert LOADED == [], "a dry run never loads the model"

    ex = make_executor(monkeypatch)
    ex.execute(graph(False), "first", {}, ["out"])
    assert ex.success, ex.status_messages
    assert LOADED == ["x.safetensors", "latent"], "rendering loads it once"
    assert len(os.listdir(os.path.join(nodes.STORE_ROOT, "lazy", "segments"))) == 2

    # restart: RAM and ComfyUI's caches are gone, the files remain
    nodes.clear_segment_cache()
    LOADED.clear()
    ex = make_executor(monkeypatch)
    ex.execute(graph(False), "after-restart", {}, ["out"])
    assert ex.success, [m[1].get("exception_message") or m for m in ex.status_messages[-1:]]
    assert LOADED == ["latent"], "fully reused from disk: the model is never loaded"
    rows = ex.history_result["outputs"]["ls"]["plan_json"]
    assert [(r["status"], r["source"]) for r in rows] == [("reused", "disk")] * 2

    # a different model file: everything renders, so the model loads
    sim, *_ = T.make_simulator(plan)
    monkeypatch.setattr(nodes, "_sample_window", sim)
    LOADED.clear()
    ex = make_executor(monkeypatch)
    ex.execute(graph(False, unet="y.safetensors"), "new-model", {}, ["out"])
    assert ex.success, ex.status_messages
    assert LOADED == ["y.safetensors", "latent"]
