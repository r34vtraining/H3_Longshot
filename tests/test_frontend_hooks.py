"""Hooks for front ends (H3 Long Shot Studio): the plan as a UI output, the
plan as data, and per-segment progress events over the websocket."""
import pytest

import test_integration as T
import test_reuse as R
import test_executor as E

nodes, P = T.nodes, T.P


@pytest.fixture
def events(monkeypatch):
    got = []
    monkeypatch.setattr(nodes, "_notify", got.append)
    return got


def ui_of(kw):
    return nodes.MiniMaxH3LongShot.execute(**kw).ui


def test_plan_is_a_ui_output_with_rows(monkeypatch, events):
    stub = R.Stub()
    stub.clip = R.CLIP
    monkeypatch.setattr(nodes, "_sample_window", stub)
    kw = T.node_kwargs([5, 5, 9], clip=R.CLIP, vae=R.VAE, noise=T.cs.Noise_RandomNoise(40))
    kw["prompt"]["shots"][2]["seed"] = 777
    out = nodes.MiniMaxH3LongShot.execute(**kw)
    ui = out.ui
    assert ui["text"] == [out.args[3]]
    rows = ui["plan_json"]
    plan = P.plan_from_durations([5, 5, 9], 22)
    assert [r["index"] for r in rows] == [1, 2, 3]
    assert [r["seed"] for r in rows] == [40, 41, 777]
    assert [r["own_seed"] for r in rows] == [False, False, True]
    assert all(r["status"] == "render" and r["reason"] == "first run" for r in rows)
    # rows tile the timeline with no gaps, matching the planner
    assert rows[0]["start"] == 0
    for a, b in zip(rows, rows[1:]):
        assert a["end"] == b["start"]
    assert rows[-1]["end"] == pytest.approx(P.seconds(plan.total_frames), abs=1e-3)
    assert sum(r["frames"] for r in rows) == plan.total_frames
    assert [r["start_frame"] for r in rows] == [s.visible_start for s in plan.segments]


def test_progress_events_report_render_and_reuse(monkeypatch, events):
    R.go(monkeypatch, [5, 5])
    assert [(e["segment"], e["status"]) for e in events] == [
        (1, "rendering"), (1, "done"), (2, "rendering"), (2, "done")]
    assert all(e["of"] == 2 for e in events)
    assert [e["seed"] for e in events] == [100, 100, 101, 101]

    events.clear()
    R.go(monkeypatch, [5, 5, 5])      # add a Shot: 1-2 come from memory
    assert [(e["segment"], e["status"]) for e in events] == [
        (1, "reused"), (2, "reused"), (3, "rendering"), (3, "done")]


def test_reroll_plan_rows_say_why(monkeypatch, events):
    R.go(monkeypatch, [5, 5, 5])
    events.clear()
    kw = T.node_kwargs([5, 5, 5], clip=R.CLIP, vae=R.VAE, dry_run=True,
                       noise=T.cs.Noise_RandomNoise(100))
    kw["prompt"]["shots"][1]["seed"] = 9
    rows = ui_of(kw)["plan_json"]
    assert [r["status"] for r in rows] == ["reused", "render", "render"]
    assert rows[1]["reason"] == "seed changed"
    assert rows[2]["reason"] == "follows a changed segment"
    assert events == [], "a dry run sends no progress"


def test_notify_without_a_server_is_harmless():
    nodes._notify({"segment": 1})   # no PromptServer.instance in tests


def test_dry_run_plan_reaches_history_through_the_executor(E_executor):
    ex = E_executor
    ex.execute(E._graph(True), "dry", {}, ["3", "4"])
    assert ex.success, ex.status_messages
    ui = ex.history_result["outputs"]["2"]
    assert ui["text"][0].startswith("MiniMax H3 Long Shot — 3 segments")
    assert [r["status"] for r in ui["plan_json"]] == ["render"] * 3


@pytest.fixture
def E_executor(monkeypatch):
    return E.executor.__wrapped__(monkeypatch)


def test_plan_ui_is_resent_when_the_run_is_cached(E_executor):
    """Queue the same dry run twice: the second is served whole from ComfyUI's
    cache, and the plan must still be in /history (has_intermediate_output)."""
    from comfy_api.latest import io
    if "has_intermediate_output" not in getattr(io.Schema, "__dataclass_fields__", {}):
        pytest.skip("this ComfyUI predates has_intermediate_output")
    ex = E_executor
    ex.execute(E._graph(True), "first", {}, ["3", "4"])
    ex.execute(E._graph(True), "again", {}, ["3", "4"])
    assert ex.success, ex.status_messages
    cached = [m for m in ex.status_messages if m[0] == "execution_cached"][-1][1]["nodes"]
    assert "2" in cached, "the second run should come from the cache"
    assert ex.history_result["outputs"]["2"]["plan_json"]
