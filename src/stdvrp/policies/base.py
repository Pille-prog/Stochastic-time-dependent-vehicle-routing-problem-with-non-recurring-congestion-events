"""Policy: a rule that maps a State to a decision (variation axis 1; ADR-0002)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from stdvrp.simulation.state import State, TrainingSnapshot


class Policy(ABC):
    """Maps a State to a decision: the Client node each vehicle serves next."""

    @abstractmethod
    def decide(self, state: State) -> list[int]:
        """The next node per vehicle (a Client node or the depot)."""


class TrainablePolicy(Protocol):
    """A Policy that can also drive and learn from a training Episode.

    Structural, not nominal (ticket 02): ``MonteCarloPolicy`` satisfies this by
    shape, without inheriting from it, and
    :meth:`~stdvrp.simulation.model.Model.run_training_episode` depends only on
    this Protocol — never on ``MonteCarloPolicy`` itself — so a second
    trainable Policy is pluggable without ``Model`` knowing it exists.
    """

    def decide_train(self, state: State) -> list[int]:
        """The epsilon-greedy next node per vehicle for one training decision epoch."""
        ...

    def learn(
        self,
        snapshots: list[TrainingSnapshot],
        actions: list[list[int]],
        rewards: list[float],
    ) -> None:
        """Update from one Episode's snapshots, decisions and rewards.

        ``MonteCarloPolicy`` aliases this as ``update_W`` — the name ADR-0001's
        change log and tickets 06/08/09 name it by; that legacy name stays
        callable, it just is not the protocol-facing one.
        """
        ...
