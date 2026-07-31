"""Tier-1 self-golden gate: the package still computes its captured fixture outputs.

The simulation-performance effort's default behavior contract
(``.scratch/simulation-performance/spec.md``): every pure-mechanical optimization
must reproduce ``tests/fixtures/self_golden/mini_fixture.json`` *float-for-float*.
This test re-runs the exact capture protocol
(``scripts/capture_self_golden.py``) and asserts equality with ``==`` — no
tolerance. When an optimization legitimately reorders float arithmetic (Tier 2),
it re-captures and documents why, per the tiered contract.

**Environment guard.** numpy's ``Generator`` reproduces its integer stream per
seed, but its Ziggurat-based float draws (per-arc velocities, the client count)
are not guaranteed bit-identical across CPUs / libm / numpy versions. So this
gate *skips* — never falsely fails — when the running environment differs from
the one recorded in the capture. Re-run the capture script to pin a new
environment (e.g. the CI image). See the capture module docstring and ADR-0003.
"""

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "self_golden" / "mini_fixture.json"


def _load_capture_module() -> ModuleType:
    """scripts/capture_self_golden.py: the single source of the protocol + world build."""
    spec = importlib.util.spec_from_file_location(
        "capture_self_golden", REPO_ROOT / "scripts" / "capture_self_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture_self_golden = _load_capture_module()


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    if not SELF_GOLDEN_PATH.exists():
        pytest.skip("self-golden not captured (scripts/capture_self_golden.py)")
    return json.loads(SELF_GOLDEN_PATH.read_text(encoding="utf-8"))


def test_self_golden_gate_is_active_on_this_environment(golden: dict[str, Any]) -> None:
    """Always runs: makes an off-platform (inert) Tier-1 gate LOUD, not silent.

    The four bit-exact tests below *skip* when the environment differs from the
    capture (numpy float draws are not guaranteed bit-identical across CPUs / libm
    / numpy versions — user-ratified, ADR-0003). A silently-green CI could hide
    that the default behavior gate never actually ran, so this test emits a warning
    when the gate is inert here. Re-capture on this environment
    (``scripts/capture_self_golden.py``) to activate it.
    """
    captured = golden["meta"]["environment"]
    assert set(captured) == {"numpy", "python", "system", "machine"}, "malformed fingerprint"
    current = capture_self_golden.environment_fingerprint()
    if current != captured:
        warnings.warn(
            f"Tier-1 self-golden gate is INERT here: captured on {captured}, running on "
            f"{current}; the bit-exact tests skipped. Re-capture on this environment to "
            "make the default behavior gate live.",
            stacklevel=2,
        )


@pytest.fixture(scope="module")
def live(golden: dict[str, Any]) -> dict[str, Any]:
    """Re-run the capture protocol against the live package (guarded on environment)."""
    captured_env = golden["meta"]["environment"]
    current_env = capture_self_golden.environment_fingerprint()
    if current_env != captured_env:
        pytest.skip(
            "self-golden captured on a different environment "
            f"({captured_env} != {current_env}); Ziggurat float draws are not "
            "guaranteed bit-identical across them — re-run the capture here to pin it"
        )
    return capture_self_golden.capture()


def test_training_w_trajectory_is_bit_exact(golden: dict[str, Any], live: dict[str, Any]) -> None:
    expected = golden["training"]
    produced = live["training"]
    assert [e["seed"] for e in produced] == [e["seed"] for e in expected]
    for exp, got in zip(expected, produced, strict=True):
        assert got["w"] == exp["w"], f"seed {exp['seed']}: W trajectory drifted"


def test_training_metrics_are_bit_exact(golden: dict[str, Any], live: dict[str, Any]) -> None:
    for exp, got in zip(golden["training"], live["training"], strict=True):
        assert got["metrics"] == exp["metrics"], f"seed {exp['seed']}: training metrics drifted"


def test_final_w_is_bit_exact(golden: dict[str, Any], live: dict[str, Any]) -> None:
    assert live["final_w"] == golden["final_w"]


def test_evaluation_metrics_are_bit_exact(golden: dict[str, Any], live: dict[str, Any]) -> None:
    expected = golden["evaluation"]
    produced = live["evaluation"]
    assert [e["seed"] for e in produced] == [e["seed"] for e in expected]
    for exp, got in zip(expected, produced, strict=True):
        assert got["metrics"] == exp["metrics"], f"seed {exp['seed']}: evaluation metrics drifted"


def test_frozen_w_eval_metrics_are_bit_exact(golden: dict[str, Any], live: dict[str, Any]) -> None:
    """Ticket 01's third block: EVAL_SEEDS run with the literal, never-recomputed FROZEN_W.

    The only block a diff against is attributable purely to a production
    change (spec.md "Behavior contract") — no cost-function change can move
    this W, so nothing here can move except through the simulator's own
    behavior under a fixed decision.
    """
    expected = golden["frozen_w_eval"]
    produced = live["frozen_w_eval"]
    assert [e["seed"] for e in produced] == [e["seed"] for e in expected]
    for exp, got in zip(expected, produced, strict=True):
        assert got["metrics"] == exp["metrics"], (
            f"seed {exp['seed']}: frozen-W eval metrics drifted"
        )
