"""Policy: a rule that maps a State to a decision (variation axis 1; ADR-0002)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stdvrp.simulation.state import State


class Policy(ABC):
    """Maps a State to a decision: the Client node each vehicle serves next."""

    @abstractmethod
    def decide(self, state: State) -> list[int]:
        """The next node per vehicle (a Client node or the depot)."""
