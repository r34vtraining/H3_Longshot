"""Segment reuse is process-wide; start every test with an empty memory so
tests can't reuse each other's segments, and give every test its own folder
for saved segments so nothing is written into the ComfyUI tree."""
import sys

import pytest


@pytest.fixture(autouse=True)
def _fresh_segment_memory(tmp_path, monkeypatch):
    for name, mod in list(sys.modules.items()):
        if name.endswith(".nodes") and hasattr(mod, "clear_segment_cache"):
            mod.clear_segment_cache()
            if hasattr(mod, "STORE_ROOT"):
                monkeypatch.setattr(mod, "STORE_ROOT", str(tmp_path / "longshot"))
    yield
