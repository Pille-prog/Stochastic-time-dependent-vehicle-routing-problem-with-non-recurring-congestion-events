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
  eval golden also pins the training outcome). Each records the nine metrics.

**Environment sensitivity.** numpy's ``Generator`` guarantees a reproducible
*integer* stream per seed, but float distribution methods (the per-arc velocity
``normal`` draws, the client-count ``normal`` draw) are Ziggurat-based and NOT
guaranteed bit-identical across CPUs / libm / numpy versions (numpy's own
compatibility policy). The capture therefore records an environment fingerprint,
and the gate *skips* — rather than falsely fails — when the running environment
differs from the one that produced the file. Re-run this script to re-capture on a
new canonical environment. See the ticket ``## Comments`` and ADR-0003.

Usage (writes tests/fixtures/self_golden/mini_fixture.json)::

    uv run python scripts/capture_self_golden.py
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import tempfile
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
    """The nine golden-pinned metrics, each coerced to a round-trippable float/int."""
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
    }


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


def main() -> None:
    print("building the mini-fixture world and running the self-golden protocol ...", flush=True)
    document = capture()
    SELF_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELF_GOLDEN_PATH.write_text(
        json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    env = document["meta"]["environment"]
    n_train = len(document["training"])
    n_eval = len(document["evaluation"])
    fingerprint = f"numpy {env['numpy']}, python {env['python']}, {env['system']}/{env['machine']}"
    print(f"written: {SELF_GOLDEN_PATH}")
    print(f"  environment: {fingerprint}")
    print(f"  training episodes: {n_train}, evaluation episodes: {n_eval}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
