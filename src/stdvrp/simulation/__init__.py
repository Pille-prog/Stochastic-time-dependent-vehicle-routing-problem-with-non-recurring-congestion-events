"""State, Model (transition function) and Episode runners: the sequential decision core.

The Model's three collaborators (``CostLedger``, ``EpisodeVelocities``,
``FleetRoutes``) are deliberately *not* re-exported here: they are reached
through the Model that owns them, and the few places that name them directly —
tests, mostly — import them from their own modules.
"""

from stdvrp.simulation.episode import (
    EpisodeResult,
    TrainingEpisodeResult,
    run_evaluation_episode,
    run_training_episode,
)
from stdvrp.simulation.model import Model
from stdvrp.simulation.state import State, TrainingSnapshot

__all__ = [
    "EpisodeResult",
    "Model",
    "State",
    "TrainingEpisodeResult",
    "TrainingSnapshot",
    "run_evaluation_episode",
    "run_training_episode",
]
