"""ReferenceCard: the linear MonteCarloPolicy's frozen per-seed baseline (ticket 01).

Retires the scalar ``static_policy_mean_cost`` — one hardcoded YAML number that
only supported comparing *means*, hiding real effects the cost distribution's
variance is wide enough to swallow. A ``ReferenceCard`` is the winning cell
(best training budget x best ``test_action_count``) of a completed linear
``MonteCarloPolicy`` run: its per-seed ``total_cost`` over ``test_seeds`` (the
verdict set) and over ``evaluation_seeds`` (what the live training report reads
from, ticket 07), plus ``best_w``, the config snapshot and wall-clock per phase.

Every later Policy in this effort is measured against this same frozen vector,
paired seed-for-seed (spec.md, "Why the paired comparison is valid") — never
recomputed, never a moving target.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceCard:
    """The fixed opponent every later run in this effort is measured against."""

    winning_budget: int
    winning_test_action_count: int
    test_seeds: tuple[int, ...]
    test_seed_costs: tuple[float, ...]
    evaluation_seeds: tuple[int, ...]
    evaluation_seed_costs: tuple[float, ...]
    best_w: tuple[float, ...]
    config: dict[str, Any]
    wall_clock_seconds: dict[str, float]

    def __post_init__(self) -> None:
        if len(self.test_seeds) != len(self.test_seed_costs):
            raise ValueError("test_seeds and test_seed_costs must pair one cost per seed")
        if len(self.evaluation_seeds) != len(self.evaluation_seed_costs):
            raise ValueError(
                "evaluation_seeds and evaluation_seed_costs must pair one cost per seed"
            )

    @property
    def test_mean_cost(self) -> float:
        """Mean ``total_cost`` over the verdict set (``test_seeds``)."""
        return sum(self.test_seed_costs) / len(self.test_seed_costs)

    @property
    def evaluation_mean_cost(self) -> float:
        """Mean ``total_cost`` over the tuning set (``evaluation_seeds``)."""
        return sum(self.evaluation_seed_costs) / len(self.evaluation_seed_costs)

    def test_cost_by_seed(self) -> dict[int, float]:
        """The verdict-set vector as a seed -> cost mapping, for paired lookups."""
        return dict(zip(self.test_seeds, self.test_seed_costs, strict=True))

    def evaluation_cost_by_seed(self) -> dict[int, float]:
        """The tuning-set vector as a seed -> cost mapping, for paired lookups."""
        return dict(zip(self.evaluation_seeds, self.evaluation_seed_costs, strict=True))

    def to_json(self) -> dict[str, Any]:
        """A plain-JSON document: tuples become lists, everything else already is."""
        return {
            "winning_budget": self.winning_budget,
            "winning_test_action_count": self.winning_test_action_count,
            "test_seeds": list(self.test_seeds),
            "test_seed_costs": list(self.test_seed_costs),
            "evaluation_seeds": list(self.evaluation_seeds),
            "evaluation_seed_costs": list(self.evaluation_seed_costs),
            "best_w": list(self.best_w),
            "config": self.config,
            "wall_clock_seconds": self.wall_clock_seconds,
        }

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> ReferenceCard:
        return cls(
            winning_budget=document["winning_budget"],
            winning_test_action_count=document["winning_test_action_count"],
            test_seeds=tuple(document["test_seeds"]),
            test_seed_costs=tuple(document["test_seed_costs"]),
            evaluation_seeds=tuple(document["evaluation_seeds"]),
            evaluation_seed_costs=tuple(document["evaluation_seed_costs"]),
            best_w=tuple(document["best_w"]),
            config=document["config"],
            wall_clock_seconds=document["wall_clock_seconds"],
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> ReferenceCard:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
