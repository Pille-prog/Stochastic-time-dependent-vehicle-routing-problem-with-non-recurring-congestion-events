"""Diff two W trajectories component by component.

The analysis half of ``.scratch/linear-policy-learning/issues/01-legacy-w-trajectory-diff.md``.
:mod:`scripts.capture_legacy_w_trajectory` and
:mod:`scripts.capture_repo_w_trajectory` each write one 24-component vector per
training episode; this reads both and answers the ticket's question — *which
components diverge in ours and stay bounded in the legacy*.

Everything below the ``main`` guard is a pure function over lists of floats,
unit-tested in ``tests/unit/test_w_trajectory_comparison.py``. The two capture
drivers need the full Chengdu dataset and hours of CPU; this arithmetic needs
neither, so it is not allowed to hide inside them.

**The measure.** Every trajectory starts from the zero vector, so "growth since
the first episode" is undefined for a component that starts at zero and stays
small. What discriminates a diverging weight from a settled one is instead
whether its *magnitude* keeps climbing: :func:`component_stats` takes the peak
``|W[i]|`` over the first half of the run and over the second half, and reports
their ratio. A converged component sits near 1 — the legacy is flat from its
50th episode, so essentially all of its components should. A diverging one is
unbounded above.

Rank, deliberately, by ``late_peak - early_peak`` rather than by the ratio: a
component that goes from 1e-6 to 1e-4 has a ratio of 100 and contributes
nothing to ``|W|``, while one that goes from 500 to 5000 is what the norm is
made of.

Usage::

    .venv/Scripts/python.exe scripts/compare_w_trajectories.py \\
        .scratch/linear-policy-learning/legacy_w_trajectory.json \\
        .scratch/linear-policy-learning/repo_w_trajectory.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# The 24 components, in the order both the legacy's ``actualize_W`` and our
# ``FeatureExtractor.candidate_features`` assemble them: 16 general-state
# features then 8 state-action ones. Verified append-by-append against
# ``Main_Chengdu_Sirve_Con_Clip.py`` (``extract_general_state_features`` 2859,
# the live ``extract_state_action_features`` 3573) — the two files agree on
# every position, which is what makes an index-wise diff meaningful at all.
FEATURE_NAMES: tuple[str, ...] = (
    "sqrt(clients_left)",
    "time_left",
    "time_left^2",
    "clients_left^2",
    "clients_left^2*time",
    "time^2*clients_left",
    "time^2*clients_left^2",
    "depot_dist_total",
    "depot_dist_min",
    "depot_dist_max",
    "earliness_bin0",
    "earliness_bin1",
    "earliness_bin2",
    "earliness_bin3",
    "mean_earliness_diff",
    "congestion_signal",
    "late_count/13",
    "zero_pad",
    "distance_cost/100",
    "earliness_cost/60",
    "delay_cost/60",
    "future_delay/2500",
    "(future_delay/2500)^2",
    "overtime_cost/180",
)

# A component whose second-half peak is this many times its first-half peak has
# not settled. Chosen loose on purpose: the legacy's cost is flat from episode
# 50, so anything genuinely converged should sit near 1 and a factor of two is
# already a long way from that.
DEFAULT_GROWTH_THRESHOLD = 2.0


@dataclass(frozen=True)
class ComponentStats:
    """One weight's behaviour over a whole run."""

    index: int
    name: str
    final: float
    """``W[i]`` after the last episode, signed."""

    peak: float
    """``max |W[i]|`` over the run."""

    early_peak: float
    """``max |W[i]|`` over the first half of the episodes."""

    late_peak: float
    """``max |W[i]|`` over the second half."""

    growth_ratio: float
    """``late_peak / early_peak``; 1.0 when both are zero, ``inf`` when only the first is."""

    norm_share: float
    """``W[i]^2 / |W|^2`` at the last episode — how much of the norm this carries."""


@dataclass(frozen=True)
class MagnitudeRatio:
    """How much larger one component's weight gets here than in the legacy.

    The measure that actually answered ticket 01. :func:`divergence_report` asks
    "does this component keep climbing *within* its own run", which is the right
    question when one side converges and the other runs away — but both runs
    oscillate hard enough that their within-run peaks land early on both sides,
    and it came back empty. Comparing the peaks *across* sides does not care
    when the peak happened, and showed every live component inflated here.
    """

    index: int
    name: str
    legacy_peak: float
    repo_peak: float
    ratio: float
    """``repo_peak / legacy_peak``; ``inf`` when the legacy holds it at zero."""


@dataclass(frozen=True)
class Divergence:
    """One component that runs away on the repo side while the legacy holds it."""

    index: int
    name: str
    legacy: ComponentStats
    repo: ComponentStats
    excess: float
    """``late_peak - early_peak`` on the repo side: magnitude the second half added."""


@dataclass
class TrajectoryLog:
    """What both capture drivers accumulate: one row per training episode.

    Lives here, with :func:`load_trajectory`, so the schema has one owner rather
    than a writer in each driver and a reader here — the four lists always travel
    together and were being declared, appended, checkpointed and returned in two
    places.
    """

    w: list[list[float]] = field(default_factory=list)
    episode_costs: list[float] = field(default_factory=list)
    vehicle_counts: list[int] = field(default_factory=list)
    client_counts: list[int] = field(default_factory=list)

    def record(self, w: Sequence[float], cost: float, vehicles: int, clients: int) -> None:
        self.w.append([float(value) for value in w])
        self.episode_costs.append(float(cost))
        self.vehicle_counts.append(int(vehicles))
        self.client_counts.append(int(clients))

    def __len__(self) -> int:
        return len(self.w)

    def document(self, **head: object) -> dict[str, object]:
        """The JSON document, with ``head`` (meta/protocol) in front of the data."""
        return {
            **head,
            "w_trajectory": self.w,
            "episode_costs": self.episode_costs,
            "vehicle_counts": self.vehicle_counts,
            "client_counts": self.client_counts,
        }

    def write(self, path: Path, **head: object) -> None:
        """Write the finished document to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.document(**head), indent=1)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")

    def checkpoint(self, path: Path) -> None:
        """Overwrite ``path`` with the run so far.

        Called after every episode: these runs are hours long, and a partial
        trajectory still answers the ticket.
        """
        self.write(path, meta={"partial": True, "episodes": len(self)})


def _as_matrix(trajectory: Sequence[Sequence[float]]) -> NDArray[np.float64]:
    if not trajectory:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    widths = {len(row) for row in trajectory}
    if len(widths) != 1:
        raise ValueError(f"trajectory rows of inconsistent width: {sorted(widths)}")
    return np.asarray(trajectory, dtype=np.float64)


def load_trajectory(path: Path) -> list[list[float]]:
    """Read the ``w_trajectory`` written by either capture driver."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    trajectory = document.get("w_trajectory")
    if not trajectory:
        raise ValueError(f"{path}: no w_trajectory in this file")
    return [[float(value) for value in row] for row in trajectory]


def step_norms(trajectory: Sequence[Sequence[float]]) -> list[float]:
    """``|W_t - W_{t-1}|`` per episode, measuring the first step from the zero vector.

    W is zero before any episode runs, so the first entry is ``|W_1|`` itself —
    one norm per episode, not one fewer. The ticket's lead ("logging ``|dW|`` per
    episode would find it in one run") is this list.
    """
    matrix = _as_matrix(trajectory)
    if matrix.shape[0] == 0:
        return []
    previous = np.vstack([np.zeros((1, matrix.shape[1])), matrix[:-1]])
    return [float(value) for value in np.linalg.norm(matrix - previous, axis=1)]


def norm_trajectory(trajectory: Sequence[Sequence[float]]) -> list[float]:
    """``|W|`` after each episode."""
    matrix = _as_matrix(trajectory)
    return [float(value) for value in np.linalg.norm(matrix, axis=1)]


def component_stats(trajectory: Sequence[Sequence[float]]) -> list[ComponentStats]:
    """One :class:`ComponentStats` per weight, in index order."""
    matrix = _as_matrix(trajectory)
    episodes, width = matrix.shape
    if episodes == 0:
        raise ValueError("cannot summarise an empty trajectory")
    magnitude = np.abs(matrix)
    # A one-episode run has no second half; splitting at max(1, n//2) would leave
    # it empty and report every component as having collapsed to zero. The single
    # episode belongs to both halves instead.
    split = max(1, episodes // 2)
    early = magnitude[:split].max(axis=0)
    late = magnitude[split:].max(axis=0) if episodes > 1 else early

    final = matrix[-1]
    squared_norm = float(np.dot(final, final))

    stats = []
    for index in range(width):
        early_peak = float(early[index])
        late_peak = float(late[index])
        if early_peak == 0.0:
            ratio = 1.0 if late_peak == 0.0 else math.inf
        else:
            ratio = late_peak / early_peak
        stats.append(
            ComponentStats(
                index=index,
                name=FEATURE_NAMES[index] if index < len(FEATURE_NAMES) else f"X[{index}]",
                final=float(final[index]),
                peak=float(magnitude[:, index].max()),
                early_peak=early_peak,
                late_peak=late_peak,
                growth_ratio=ratio,
                norm_share=(
                    0.0 if squared_norm == 0.0 else float(final[index] ** 2) / squared_norm
                ),
            )
        )
    return stats


def divergence_report(
    legacy: Sequence[Sequence[float]],
    repo: Sequence[Sequence[float]],
    growth_threshold: float = DEFAULT_GROWTH_THRESHOLD,
) -> list[Divergence]:
    """Components growing in ``repo`` past ``growth_threshold`` while ``legacy`` holds them.

    A component the legacy grows just as hard is shared behaviour, not the
    residual this ticket is chasing, so it is excluded rather than ranked low.
    """
    legacy_stats = component_stats(legacy)
    repo_stats = component_stats(repo)
    rows = [
        Divergence(
            index=ours.index,
            name=ours.name,
            legacy=theirs,
            repo=ours,
            excess=ours.late_peak - ours.early_peak,
        )
        for theirs, ours in zip(legacy_stats, repo_stats, strict=True)
        if ours.growth_ratio > growth_threshold and theirs.growth_ratio <= growth_threshold
    ]
    return sorted(rows, key=lambda row: row.excess, reverse=True)


def magnitude_ratios(
    legacy: Sequence[Sequence[float]], repo: Sequence[Sequence[float]]
) -> list[MagnitudeRatio]:
    """Each component's peak magnitude here against the legacy's, largest ratio first.

    Components the legacy holds at exactly zero (``zero_pad``, and
    ``earliness_bin3`` which its ``client_counts_earliness[3]`` never assigns)
    sort last rather than to the top: an infinite ratio off a zero baseline says
    "structurally absent there", not "runs away here".
    """
    legacy_stats = component_stats(legacy)
    repo_stats = component_stats(repo)
    rows = [
        MagnitudeRatio(
            index=ours.index,
            name=ours.name,
            legacy_peak=theirs.peak,
            repo_peak=ours.peak,
            ratio=(
                math.inf
                if theirs.peak == 0.0 and ours.peak > 0.0
                else 1.0
                if theirs.peak == 0.0
                else ours.peak / theirs.peak
            ),
        )
        for theirs, ours in zip(legacy_stats, repo_stats, strict=True)
    ]
    return sorted(rows, key=lambda row: (math.isinf(row.ratio), -row.ratio))


# --- Reporting ----------------------------------------------------------------


def _ratio(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.2f}"


def format_component_table(
    legacy_stats: Sequence[ComponentStats], repo_stats: Sequence[ComponentStats]
) -> str:
    """A markdown table of both sides' growth, one row per weight."""
    lines = [
        "| i | feature | legacy final | legacy peak | legacy growth "
        "| repo final | repo peak | repo growth | repo norm share |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for theirs, ours in zip(legacy_stats, repo_stats, strict=True):
        lines.append(
            f"| {ours.index} | `{ours.name}` "
            f"| {theirs.final:.4g} | {theirs.peak:.4g} | {_ratio(theirs.growth_ratio)} "
            f"| {ours.final:.4g} | {ours.peak:.4g} | {_ratio(ours.growth_ratio)} "
            f"| {ours.norm_share * 100:.1f}% |"
        )
    return "\n".join(lines)


def format_divergences(rows: Sequence[Divergence]) -> str:
    if not rows:
        return "No component diverges in ours while staying bounded in the legacy."
    lines = [
        "| i | feature | repo early peak | repo late peak | repo growth | legacy growth |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.index} | `{row.name}` "
            f"| {row.repo.early_peak:.4g} | {row.repo.late_peak:.4g} "
            f"| {_ratio(row.repo.growth_ratio)} | {_ratio(row.legacy.growth_ratio)} |"
        )
    return "\n".join(lines)


def format_magnitude_ratios(rows: Sequence[MagnitudeRatio]) -> str:
    lines = [
        "| i | feature | legacy peak | repo peak | repo/legacy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.index} | `{row.name}` | {row.legacy_peak:.4g} "
            f"| {row.repo_peak:.4g} | {_ratio(row.ratio)} |"
        )
    finite = [row.ratio for row in rows if math.isfinite(row.ratio) and row.legacy_peak > 0]
    if finite:
        lines.append("")
        lines.append(
            f"Over the {len(finite)} components the legacy actually trains: "
            f"median {np.median(finite):.2f}x, range "
            f"{min(finite):.2f}x-{max(finite):.2f}x."
        )
    return "\n".join(lines)


def format_step_spikes(trajectory: Sequence[Sequence[float]], top: int = 10) -> str:
    """The episodes with the largest ``|dW|``: the ticket's "which episode injects it"."""
    norms = step_norms(trajectory)
    ranked = sorted(enumerate(norms, start=1), key=lambda pair: pair[1], reverse=True)[:top]
    lines = ["| episode | \\|dW\\| | \\|W\\| after |", "| --- | --- | --- |"]
    norms_after = norm_trajectory(trajectory)
    for episode, value in ranked:
        lines.append(f"| {episode} | {value:.4g} | {norms_after[episode - 1]:.4g} |")
    return "\n".join(lines)


def build_report(legacy: Sequence[Sequence[float]], repo: Sequence[Sequence[float]]) -> str:
    legacy_stats = component_stats(legacy)
    repo_stats = component_stats(repo)
    legacy_norms = norm_trajectory(legacy)
    repo_norms = norm_trajectory(repo)
    diverging = divergence_report(legacy, repo)

    return "\n\n".join(
        [
            # ASCII only below this line: the report is printed to a Windows
            # console (cp1252) as well as written to a UTF-8 file.
            f"# W trajectory diff: {len(legacy)} legacy episodes vs {len(repo)} repo episodes",
            "Same seed integers, different worlds (ticket 13 replaced the legacy's two shared "
            "global streams with per-Episode PCG64 spawning). Compare the *shape* of each "
            "component's trajectory, never its values.",
            "## Norms",
            f"- legacy `|W|`: {legacy_norms[0]:.4g} after episode 1 -> "
            f"{legacy_norms[-1]:.4g} after episode {len(legacy_norms)}",
            f"- repo `|W|`: {repo_norms[0]:.4g} after episode 1 -> "
            f"{repo_norms[-1]:.4g} after episode {len(repo_norms)}",
            "## Peak magnitude here against the legacy's",
            format_magnitude_ratios(magnitude_ratios(legacy, repo)),
            "## Diverging in ours, bounded in the legacy",
            format_divergences(diverging),
            "## Every component, both sides",
            format_component_table(legacy_stats, repo_stats),
            "## Largest single-episode steps (repo)",
            format_step_spikes(repo),
            "## Largest single-episode steps (legacy)",
            format_step_spikes(legacy),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("legacy", type=Path, help="legacy_w_trajectory.json")
    parser.add_argument("repo", type=Path, help="repo_w_trajectory.json")
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="truncate both trajectories to this many episodes before comparing",
    )
    args = parser.parse_args()

    legacy = load_trajectory(args.legacy)
    repo = load_trajectory(args.repo)
    episodes = args.episodes or min(len(legacy), len(repo))
    report = build_report(legacy[:episodes], repo[:episodes])
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
