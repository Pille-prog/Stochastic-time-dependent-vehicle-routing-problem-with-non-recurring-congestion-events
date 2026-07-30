"""Tier-1 self-golden capture: the current package's exact fixture outputs (ticket 01).

The simulation-performance effort's default behavior gate
(``.scratch/simulation-performance/spec.md``): a bit-exact snapshot of what the
package computes *today* on the committed mini fixture, so every pure-mechanical
optimization can prove it changed nothing. This script runs a fixed, fully
specified protocol and writes it to ``tests/fixtures/self_golden/mini_fixture.json``;
``tests/test_self_golden.py`` re-runs the identical protocol and asserts equality
float-for-float (``==``, no tolerance).

Protocol (deterministic — per-Episode Generators are seeded from the seed alone,
ticket 13):

* **Training** — ``TRAIN_SEEDS`` run in order carrying W, warm-up learning rate on
  the first Episode exactly as the Trainer does. Each Episode records its W vector
  (the "per-episode W trajectory") and all nine ``EPISODE_METRICS``.
* **Evaluation** — ``EVAL_SEEDS`` run greedily with the *final* trained W (so the
  eval golden also pins the training outcome). Each records the ten metrics.

**Environment sensitivity.** numpy's ``Generator`` guarantees a reproducible
*integer* stream per seed, but float distribution methods (the per-arc velocity
``normal`` draws, the client-count ``normal`` draw) are Ziggurat-based and NOT
guaranteed bit-identical across CPUs / libm / numpy versions (numpy's own
compatibility policy). The capture therefore records an environment fingerprint,
and the gate *skips* — rather than falsely fails — when the running environment
differs from the one that produced the file. Re-run this script to re-capture on a
new canonical environment. See the ticket ``## Comments`` and ADR-0003.

**Tier-2 check mode.** An optimization that inherently reorders float arithmetic
(vectorized sums, BLAS batching) cannot meet the bit-exact gate; the spec then
asks it to prove the fixture outputs still agree to a tight tolerance before
re-capturing. ``--check`` re-runs the protocol against the *committed* capture and
reports the worst relative deviation instead of overwriting it::

    uv run python scripts/capture_self_golden.py --check --rtol 1e-9

``--check`` also prints a per-seed, per-metric diff report (simulator-correctness
ticket 01): every metric that moved, on every seed of every block, not only the
single worst deviation. The three-outcome rule (spec.md decision 10 — matches,
explained, or reverts) needs to see *what* changed, not just *how much*.

**Frozen-W block** (simulator-correctness ticket 01, spec.md "Behavior
contract"). The training and evaluation blocks above both hang off learning: a
cost-function change moves W from the first affected training episode, so a
diff against either block conflates the fix's direct effect with everything
downstream of a different W. This third block runs ``EVAL_SEEDS`` greedily
against :data:`FROZEN_W` — a literal snapshot of ``final_w``, fixed once and
never recomputed — so no cost change can move a decision here: the diff is
the fix's direct effect and nothing else. Written under the ``"frozen_w_eval"``
key; the two blocks above are untouched by this ticket.

Usage (writes tests/fixtures/self_golden/mini_fixture.json)::

    uv run python scripts/capture_self_golden.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from stdvrp.simulation import run_evaluation_episode, run_training_episode
from stdvrp.simulation.episode import EpisodeResult
from stdvrp.training.trainer import EPISODE_METRICS

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "self_golden" / "mini_fixture.json"

# The fixed protocol seeds. Small and deterministic; more eval than train because
# eval episodes are cheap and each one widens the regression net.
TRAIN_SEEDS: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)
EVAL_SEEDS: tuple[int, ...] = tuple(range(100000, 100010))

# The frozen-W block's literal W (simulator-correctness ticket 01, spec.md
# "Behavior contract"): a fixed snapshot of this protocol's own ``final_w`` as
# committed in ``tests/fixtures/self_golden/mini_fixture.json`` on 2026-07-28.
# Deliberately a literal, not a live read of that file — the whole point of this
# block is a W that never moves, including when the training block above is
# re-captured for an unrelated reason. Do not recompute this from a fresh run;
# update it only if the effort deliberately re-freezes (record why in the ticket
# that does it).
FROZEN_W: tuple[float, ...] = (
    0.20606397239619353,
    0.6936753052485881,
    0.6574030403042306,
    0.0059087832347860594,
    0.0001757044221968135,
    0.0001634297971536607,
    1.0025790590600254e-05,
    0.06788189945396003,
    0.10840152110368309,
    0.11454814031505155,
    0.0,
    1.281648973406756,
    0.00014615139718725904,
    0.0,
    0.21336738281676795,
    0.7759134231437905,
    0.0007870017157424951,
    0.0002292898670024438,
    0.0,
)


def _load_benchmark() -> ModuleType:
    """scripts/benchmark_episodes.py: the single owner of the fixture-world build.

    Loaded by importlib file-path, mirroring how ``rebaseline_golden_master``
    reuses ``capture_golden_master`` — scripts share helpers this way rather than
    duplicating them (``tests/`` is not importable from a standalone script run).
    """
    spec = importlib.util.spec_from_file_location(
        "benchmark_episodes", REPO_ROOT / "scripts" / "benchmark_episodes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()


def _metrics(episode: EpisodeResult) -> dict[str, float]:
    """The ten golden-pinned metrics, each coerced to a round-trippable float/int."""
    return {name: _as_number(getattr(episode, name)) for name in EPISODE_METRICS}


def _as_number(value: Any) -> Any:
    """int metrics stay int; float metrics become Python float (exact JSON round-trip)."""
    return value if isinstance(value, int) else float(value)


def run_protocol(world: Any) -> dict[str, Any]:
    """Run the training + evaluation protocol; return the capture document body.

    ``world`` is a ``benchmark_episodes.World``; its ``config`` carries the warm-up
    learning-rate quirk and its ``episode_kwargs()`` the shared runner arguments.
    """
    config = world.config
    kwargs = world.episode_kwargs()
    learning_rate = (
        config.warmup_learning_rate
        if config.warmup_learning_rate is not None
        else config.learning_rate
    )
    w: np.ndarray | None = None
    training = []
    for seed in TRAIN_SEEDS:
        result = run_training_episode(
            seed=seed, W=w, learning_rate=learning_rate, depot=0, **kwargs
        )
        learning_rate = config.learning_rate
        w = result.w
        training.append(
            {"seed": seed, "w": [float(x) for x in result.w], "metrics": _metrics(result.episode)}
        )

    assert w is not None, "TRAIN_SEEDS must be non-empty"
    final_w = w
    evaluation = []
    for seed in EVAL_SEEDS:
        episode = run_evaluation_episode(seed=seed, W=final_w, **kwargs)
        evaluation.append({"seed": seed, "metrics": _metrics(episode)})

    return {
        "final_w": [float(x) for x in final_w],
        "training": training,
        "evaluation": evaluation,
        "frozen_w_eval": run_frozen_w_eval(kwargs),
    }


def run_frozen_w_eval(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """The third capture block (ticket 01): ``EVAL_SEEDS`` run with the literal :data:`FROZEN_W`.

    Unlike ``evaluation`` above, this W never moves — it does not come from this
    run's training block, so a cost-function change cannot move it. That makes
    this the only block where a self-golden diff is attributable purely to a
    production change: no decision here can shift because *this* W changed, only
    because the simulator's behavior under a fixed W did.
    """
    frozen_w = np.array(FROZEN_W)
    frozen_w_eval = []
    for seed in EVAL_SEEDS:
        episode = run_evaluation_episode(seed=seed, W=frozen_w, **kwargs)
        frozen_w_eval.append({"seed": seed, "metrics": _metrics(episode)})
    return frozen_w_eval


def environment_fingerprint() -> dict[str, str]:
    """What the bit-exactness of the float draws depends on (see module docstring)."""
    return {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def capture() -> dict[str, Any]:
    """Build the fixture world and run the protocol, returning the full document."""
    with tempfile.TemporaryDirectory(prefix="self_golden_world_") as tmp:
        data_dir = benchmark.build_fixture_data_dir(Path(tmp))
        config = benchmark.fixture_config(data_dir)
        world, _ = benchmark.load_world(config)
        body = run_protocol(world)
    return {
        "meta": {
            "description": "Tier-1 self-golden: exact fixture outputs of the current package",
            "environment": environment_fingerprint(),
            "train_seeds": list(TRAIN_SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
        },
        **body,
    }


def numeric_pairs(expected: Any, produced: Any, path: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Walk two capture documents in lockstep, yielding their leaf numbers."""
    if isinstance(expected, dict):
        for key in expected:
            yield from numeric_pairs(expected[key], produced[key], f"{path}.{key}")
    elif isinstance(expected, list):
        for index, item in enumerate(expected):
            yield from numeric_pairs(item, produced[index], f"{path}[{index}]")
    else:
        yield path, expected, produced


def _relative_deviation(expected: Any, produced: Any) -> float:
    """How far ``produced`` sits from ``expected``, relative to ``expected``.

    An expected value of exactly zero admits no relative measure, so a produced
    value that is not also exactly zero counts as infinite deviation — the metrics
    that are structurally zero (an Episode with no earliness cost, the padding
    feature) must stay zero however the float sums are reassociated.
    """
    return math.inf if expected == 0 else abs(produced - expected) / abs(expected)


def worst_deviation(golden: dict[str, Any], live: dict[str, Any]) -> tuple[float, str]:
    """The largest relative deviation between two captures, and where it sits."""
    worst, where = 0.0, "(identical)"
    for path, expected, produced in numeric_pairs(golden, live):
        if produced == expected:
            continue
        deviation = _relative_deviation(expected, produced)
        if deviation > worst:
            worst, where = deviation, path
    return worst, where


def _diff_line(where: str, expected: Any, produced: Any) -> str:
    deviation = _relative_deviation(expected, produced)
    return f"  {where}: expected={expected!r} produced={produced!r} rel={deviation:.3e}"


def diff_report(golden: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Every metric that moved, per seed, per block (ticket 01's diff reporter).

    ``worst_deviation`` answers "how far off is the single worst number"; this
    answers "what, exactly, changed" — the spec's three-outcome rule (decision
    10: matches / explained-to-a-mechanism / reverts) needs the second question
    answered, because a bare "worst rtol 3e-2" cannot be checked against a
    ticket's written prediction of *which* metrics move and in *which*
    direction. One line per differing leaf, grouped by block and seed rather
    than by JSON path, so a ticket's Comments can read it directly.
    """
    lines: list[str] = []
    golden_final_w = golden.get("final_w", [])
    live_final_w = live.get("final_w", [])
    for index, (expected, produced) in enumerate(zip(golden_final_w, live_final_w, strict=True)):
        if produced != expected:
            lines.append(_diff_line(f"final_w[{index}]", expected, produced))

    for block_name in ("training", "evaluation", "frozen_w_eval"):
        golden_block = golden.get(block_name, [])
        live_block = live.get(block_name, [])
        for golden_entry, live_entry in zip(golden_block, live_block, strict=True):
            label = f"{block_name} seed={golden_entry['seed']}"
            if "w" in golden_entry:
                golden_w, live_w = golden_entry["w"], live_entry["w"]
                for index, (expected, produced) in enumerate(zip(golden_w, live_w, strict=True)):
                    if produced != expected:
                        lines.append(_diff_line(f"{label} w[{index}]", expected, produced))
            for name, expected in golden_entry["metrics"].items():
                produced = live_entry["metrics"][name]
                if produced != expected:
                    lines.append(_diff_line(f"{label} metrics.{name}", expected, produced))
    return lines


def check(rtol: float) -> int:
    """Compare a fresh run against the committed capture; return a process exit code."""
    if not SELF_GOLDEN_PATH.exists():
        print(f"no capture to check against at {SELF_GOLDEN_PATH}")
        return 1
    golden = json.loads(SELF_GOLDEN_PATH.read_text(encoding="utf-8"))
    captured_env = golden["meta"]["environment"]
    current_env = environment_fingerprint()
    if current_env != captured_env:
        print(f"WARNING: captured on {captured_env}, running on {current_env}")

    print("re-running the self-golden protocol against the committed capture ...", flush=True)
    live = capture()
    body = {key: value for key, value in golden.items() if key != "meta"}
    worst, where = worst_deviation(body, live)
    print(f"  worst relative deviation: {worst:.3e} at {where}")
    print(f"  tolerance:                {rtol:.3e}")

    diff_lines = diff_report(body, live)
    if diff_lines:
        print(f"  per-seed per-metric diff ({len(diff_lines)} leaves moved):")
        for line in diff_lines:
            print(line)
    else:
        print("  per-seed per-metric diff: (identical - nothing moved)")

    if worst > rtol:
        print("FAIL: the live package is outside the tolerance")
        return 1
    print("OK: within tolerance")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed capture instead of overwriting it",
    )
    parser.add_argument("--rtol", type=float, default=1e-9, help="--check tolerance")
    arguments = parser.parse_args()
    if arguments.check:
        raise SystemExit(check(arguments.rtol))

    print("building the mini-fixture world and running the self-golden protocol ...", flush=True)
    document = capture()
    SELF_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELF_GOLDEN_PATH.write_text(
        json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    env = document["meta"]["environment"]
    n_train = len(document["training"])
    n_eval = len(document["evaluation"])
    n_frozen = len(document["frozen_w_eval"])
    fingerprint = f"numpy {env['numpy']}, python {env['python']}, {env['system']}/{env['machine']}"
    print(f"written: {SELF_GOLDEN_PATH}")
    print(f"  environment: {fingerprint}")
    print(
        f"  training episodes: {n_train}, evaluation episodes: {n_eval}, "
        f"frozen-W episodes: {n_frozen}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
