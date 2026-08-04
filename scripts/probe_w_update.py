"""Decompose the per-episode W update, on either side, into the terms that make it.

The third measurement of ``.scratch/linear-policy-learning/issues/01-legacy-w-trajectory-diff.md``,
and the premise ticket 02 starts from. :mod:`scripts.compare_w_trajectories` says
*which* weights end up inflated; this says *what the update was made of* when
they did.

Both sides run the same arithmetic::

    W += lr * (U_t - already_acquired - X @ W) * X

so the per-episode step is a sum of ``T`` such terms, and the size of that sum
has only a few possible sources: how many terms there are (``T``), how big each
feature vector is (``‖X‖``), how big the residual is, and how much the terms
point the same way rather than cancelling. This records all four, plus
``X[13]`` — the earliness bin that is live here and structurally zero in the
legacy, and the single largest named contributor to the ``‖X‖`` gap.

**How the instrumentation stays honest.** Neither side is edited. The legacy's
``policy.actualize_W`` and our ``MonteCarloPolicy.learn`` are each replaced, at
runtime, by a transcription that computes the identical arithmetic and records
the intermediates. The check that the transcription is faithful is built in:
``--verify`` re-runs the same episodes uninstrumented and requires the final
``‖W‖`` to match bit for bit. Ticket 01's published figures were taken with that
check passing on both sides.

Usage::

    .venv/Scripts/python.exe -u scripts/probe_w_update.py repo \\
        experiments/chengdu/config_linear_congestion_10k.yaml --episodes 25
    .venv/Scripts/python.exe -u scripts/probe_w_update.py legacy \\
        --data-dir "C:/Users/ferna/OneDrive/Documentos/Mega city" --episodes 25

The legacy side reuses :mod:`scripts.capture_legacy_w_trajectory`'s shim and its
warm world cache, so it costs ~15s to load and ~4s/episode.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The legacy's ``actualize_W`` clips the assembled feature vector at 3 before it
# computes anything (Con_Clip line 4374) -- see ``MonteCarloPolicy``'s
# ``UPDATE_FEATURE_CEILING``, which is the same constant on our side. Spelled
# out here rather than imported so the legacy branch does not need our package.
FEATURE_CEILING = 3

# ``earliness_bin3``: assigned here, never assigned in either legacy source
# (B10). Recorded by name because it is the largest single named contributor to
# the ``‖X‖`` difference the probe exists to attribute.
EARLINESS_BIN3 = 13


@dataclasses.dataclass
class UpdateStats:
    """Every per-update intermediate, accumulated across a run."""

    updates_per_episode: list[int] = dataclasses.field(default_factory=list)
    feature_norm: list[float] = dataclasses.field(default_factory=list)
    nonzero_features: list[int] = dataclasses.field(default_factory=list)
    earliness_bin3: list[float] = dataclasses.field(default_factory=list)
    abs_residual: list[float] = dataclasses.field(default_factory=list)
    signed_residual: list[float] = dataclasses.field(default_factory=list)
    step_coherence: list[float] = dataclasses.field(default_factory=list)
    """Per episode: ``‖ΔW‖ / Σ‖step‖``. 1.0 if every step pointed the same way."""

    episode_step_norm: list[float] = dataclasses.field(default_factory=list)

    def record_update(self, X: NDArray[np.float64], residual: float) -> None:
        self.feature_norm.append(float(np.linalg.norm(X)))
        self.nonzero_features.append(int(np.count_nonzero(X)))
        self.earliness_bin3.append(float(X[EARLINESS_BIN3]))
        self.abs_residual.append(abs(residual))
        self.signed_residual.append(residual)

    def record_episode(self, updates: int, step_norm: float, step_norm_sum: float) -> None:
        self.updates_per_episode.append(updates)
        self.episode_step_norm.append(step_norm)
        self.step_coherence.append(step_norm / step_norm_sum if step_norm_sum else 0.0)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for field in dataclasses.fields(self):
            values = np.asarray(getattr(self, field.name), dtype=float)
            if values.size == 0:
                continue
            out[field.name] = {
                "n": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            }
        return out


def probe_repo(config_path: Path, episodes: int, spread: int | None) -> tuple[UpdateStats, float]:
    """Instrument ``MonteCarloPolicy.learn`` and run ``episodes`` training Episodes."""
    import stdvrp.simulation  # noqa: F401
    from stdvrp.config import ExperimentConfig
    from stdvrp.policies.monte_carlo import MonteCarloPolicy
    from stdvrp.simulation import run_training_episode
    from stdvrp.traffic import world_cache
    from stdvrp.training.episode_pool import EpisodeWorld

    stats = UpdateStats()

    def instrumented_learn(self, snapshots, actions, rewards):  # type: ignore[no-untyped-def]
        """``MonteCarloPolicy.learn``, transcribed, recording its intermediates."""
        T = len(actions)
        before = None if self.W is None else self.W.copy()
        step_norm_sum = 0.0
        U_t = 0.0
        lr = self.learning_rate
        for t in range(T - 1, -1, -1):
            U_t += rewards[t + 1]
            snapshot = snapshots[t]
            acquired = self._already_acquired_cost(snapshot)
            X = self.feature_extractor.action_features(
                self.feature_extractor.state_features(snapshot), actions[t]
            )
            X = np.clip(X, a_min=None, a_max=FEATURE_CEILING)
            assert self.W is not None
            residual = U_t - acquired - float(np.dot(X, self.W))
            stats.record_update(X, residual)
            step = lr * (residual * X)
            step_norm_sum += float(np.linalg.norm(step))
            self.W = self.W + step
        if before is not None:
            stats.record_episode(T, float(np.linalg.norm(self.W - before)), step_norm_sum)

    config = ExperimentConfig.from_yaml(config_path)
    if spread is not None:
        config = dataclasses.replace(config, time_window_spread=spread)

    original = MonteCarloPolicy.learn
    MonteCarloPolicy.learn = instrumented_learn  # type: ignore[method-assign]
    try:
        world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())
        kwargs = world.episode_kwargs()
        w = None
        lr = config.warmup_learning_rate or config.learning_rate
        for index in range(episodes):
            result = run_training_episode(
                seed=config.first_train_seed + index, W=w, learning_rate=lr, **kwargs
            )
            lr = config.learning_rate
            w = result.w
    finally:
        MonteCarloPolicy.learn = original  # type: ignore[method-assign]
    return stats, float(np.linalg.norm(w))


def probe_legacy(data_dir: Path, episodes: int, diff_tw: int | None) -> tuple[UpdateStats, float]:
    """Instrument the monolith's ``policy.actualize_W`` and run ``episodes`` episodes."""
    import capture_legacy_w_trajectory as cap

    stats = UpdateStats()

    def instrumented_actualize_W(self, states, actions, rewards):  # type: ignore[no-untyped-def]
        """``actualize_W`` (Con_Clip 4318-4394), transcribed without its dead diagnostics."""
        T = len(actions)
        before = np.array(self.W, dtype=float).copy()
        step_norm_sum = 0.0
        U_t = 0
        lr = self.learning_rate
        for t in range(T - 1, -1, -1):
            U_t += rewards[t + 1]
            self.X = []
            self.state = states[t]
            self.calculate_already_acquired_cost()
            self.extract_general_state_features()
            self.extract_state_action_features(actions[t])
            self.X = np.array(
                np.clip(
                    list(itertools.chain(self.X_general_state, self.X_state_action)),
                    a_min=None,
                    a_max=FEATURE_CEILING,
                )
            )
            residual = U_t - self.total_cost_acquired - float(np.dot(self.X, self.W))
            stats.record_update(self.X, residual)
            step = lr * (residual * self.X)
            step_norm_sum += float(np.linalg.norm(step))
            self.W = self.W + step
        after = np.array(self.W, dtype=float)
        stats.record_episode(T, float(np.linalg.norm(after - before)), step_norm_sum)

    import capture_golden_master as golden

    legacy_path = data_dir / cap.DEFAULT_LEGACY_NAME
    legacy = golden.load_legacy(legacy_path)
    legacy.policy.actualize_W = instrumented_actualize_W

    protocol = dict(cap.DEFAULT_PROTOCOL)
    if diff_tw is not None:
        protocol["diff_TW"] = diff_tw
    sha = cap.script_sha256(legacy_path)
    trajectory = cap.capture_trajectory(
        legacy, data_dir, sha, protocol, episodes, cap.default_cache_path(sha)
    )
    return stats, float(np.linalg.norm(trajectory.w[-1]))


def verify_repo(config_path: Path, episodes: int, spread: int | None) -> float:
    """The same episodes with nothing patched: its final ``‖W‖`` must match the probe's."""
    import stdvrp.simulation  # noqa: F401
    from stdvrp.config import ExperimentConfig
    from stdvrp.simulation import run_training_episode
    from stdvrp.traffic import world_cache
    from stdvrp.training.episode_pool import EpisodeWorld

    config = ExperimentConfig.from_yaml(config_path)
    if spread is not None:
        config = dataclasses.replace(config, time_window_spread=spread)
    world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())
    kwargs = world.episode_kwargs()
    w = None
    lr = config.warmup_learning_rate or config.learning_rate
    for index in range(episodes):
        result = run_training_episode(
            seed=config.first_train_seed + index, W=w, learning_rate=lr, **kwargs
        )
        lr = config.learning_rate
        w = result.w
    return float(np.linalg.norm(w))


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    lines = ["| statistic | n | mean | median | max |", "| --- | --- | --- | --- | --- |"]
    for name, values in summary.items():
        lines.append(
            f"| `{name}` | {values['n']:.0f} | {values['mean']:.4f} "
            f"| {values['median']:.4f} | {values['max']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("side", choices=("repo", "legacy"))
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "experiments" / "chengdu" / "config_linear_congestion_10k.yaml",
        help="repo side only",
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--data-dir", type=Path, default=None, help="legacy side only")
    parser.add_argument(
        "--time-window-spread",
        type=int,
        default=None,
        help="override the time-window width on either side (the legacy's diff_TW)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="repo side only: re-run uninstrumented and require the same final |W|",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the summary as JSON")
    args = parser.parse_args()

    if args.side == "repo":
        stats, final_norm = probe_repo(args.config, args.episodes, args.time_window_spread)
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required for the legacy side")
        stats, final_norm = probe_legacy(args.data_dir, args.episodes, args.time_window_spread)

    summary = stats.summary()
    print(format_summary(summary))
    print(f"\nfinal |W| after {args.episodes} episodes: {final_norm:.2f}")

    if args.verify:
        if args.side != "repo":
            parser.error(
                "--verify is repo-side only (the legacy patch is not reversible in-process)"
            )
        expected = verify_repo(args.config, args.episodes, args.time_window_spread)
        if expected != final_norm:
            print(f"VERIFY FAILED: uninstrumented |W| {expected:.6f} != probed {final_norm:.6f}")
            return 1
        print(f"verified: uninstrumented run reproduces |W| = {expected:.2f} exactly")

    if args.out is not None:
        document: dict[str, Any] = {
            "side": args.side,
            "episodes": args.episodes,
            "time_window_spread": args.time_window_spread,
            "final_w_norm": final_norm,
            "summary": summary,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
