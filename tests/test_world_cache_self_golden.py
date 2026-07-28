"""Tier-1 self-golden gate, run through the binary world cache (ticket 03).

The spec's default behavior contract (``.scratch/simulation-performance/spec.md``)
demands episode outputs stay bit-exact under a pure-mechanical optimization; the
world cache is exactly that (it changes *how* the world is loaded, never its
values). This re-runs ``scripts/capture_self_golden.py``'s exact protocol, but
with the world reconstructed from a cache hit instead of freshly parsed, and
requires the identical bit-exact match to ``tests/fixtures/self_golden/
mini_fixture.json`` that ``tests/test_self_golden.py`` requires for the
uncached path.

**Environment guard**, identical to ``test_self_golden.py``: numpy's Ziggurat
float draws are not guaranteed bit-identical across CPUs / libm / numpy
versions, so this gate skips (rather than falsely fails) when the running
environment differs from the one recorded in the capture. See that module and
ADR-0003 for the full rationale.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "self_golden" / "mini_fixture.json"


def _load_capture_module() -> ModuleType:
    """scripts/capture_self_golden.py: the protocol, the fixture-world builder."""
    spec = importlib.util.spec_from_file_location(
        "capture_self_golden", REPO_ROOT / "scripts" / "capture_self_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture_self_golden = _load_capture_module()
benchmark = capture_self_golden.benchmark


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    if not SELF_GOLDEN_PATH.exists():
        pytest.skip("self-golden not captured (scripts/capture_self_golden.py)")
    return json.loads(SELF_GOLDEN_PATH.read_text(encoding="utf-8"))


def test_self_golden_is_bit_exact_when_the_world_loads_from_a_cache_hit(
    golden: dict[str, Any], tmp_path: Path
) -> None:
    captured_env = golden["meta"]["environment"]
    current_env = capture_self_golden.environment_fingerprint()
    if current_env != captured_env:
        pytest.skip(
            "self-golden captured on a different environment "
            f"({captured_env} != {current_env}); Ziggurat float draws are not "
            "guaranteed bit-identical across them — re-run the capture here to pin it"
        )

    fixture_dir = tmp_path / "fixture_world"
    data_dir = benchmark.build_fixture_data_dir(fixture_dir)
    config = benchmark.fixture_config(data_dir)
    cache_dir = tmp_path / "cache"

    # Cold: parses the CSVs and writes the cache. Warm: the object under test,
    # reconstructed entirely from the pickle via TravelTimeModel._from_cached_state.
    benchmark.load_world(config, cache_dir=cache_dir)
    world, _ = benchmark.load_world(config, cache_dir=cache_dir)

    produced = capture_self_golden.run_protocol(world)

    assert [e["seed"] for e in produced["training"]] == [e["seed"] for e in golden["training"]]
    for expected, got in zip(golden["training"], produced["training"], strict=True):
        assert got["w"] == expected["w"], f"seed {expected['seed']}: W trajectory drifted"
        assert got["metrics"] == expected["metrics"], (
            f"seed {expected['seed']}: training metrics drifted"
        )
    assert produced["final_w"] == golden["final_w"]
    assert [e["seed"] for e in produced["evaluation"]] == [e["seed"] for e in golden["evaluation"]]
    for expected, got in zip(golden["evaluation"], produced["evaluation"], strict=True):
        assert got["metrics"] == expected["metrics"], (
            f"seed {expected['seed']}: evaluation metrics drifted"
        )
