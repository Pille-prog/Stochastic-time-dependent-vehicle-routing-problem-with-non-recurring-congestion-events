"""Checkpoint save/load for a neural training run (ticket 07, neural-policy).

Written every evaluation block, atomically: :func:`save_checkpoint` writes to
a temporary file in the same directory and then ``os.replace``s it into
place, a single filesystem operation that either fully lands or does not
happen at all. A ``Ctrl-C`` (or any other interruption) during the
potentially-slow write can only ever corrupt the temporary file — the
checkpoint at ``path`` is either the previous, fully-valid write, or the new
one; never a half-written one.

Deliberately does **not** persist any RNG generator state — see
:mod:`stdvrp.training.neural_episode`'s module docstring for why: every
stochastic stream is spawned fresh from each Episode's own seed, so resuming
correctly needs only :attr:`TrainingCheckpoint.episodes_completed`, which
determines every subsequent Episode's seed.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

from stdvrp.training.neural_episode import NeuralPolicyState
from stdvrp.training.neural_report import ConvergenceState, EvaluationReport

__all__ = ["TrainingCheckpoint", "load_checkpoint", "save_checkpoint"]


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Everything needed to resume a run identically, minus the network/optimizer.

    The network and optimizer state live separately, on :class:`
    ~stdvrp.training.neural_episode.NeuralPolicyState` — :func:`save_checkpoint`
    reads them from there at save time and :func:`load_checkpoint` writes them
    back via ``load_state_dict`` at load time, rather than duplicating them
    onto this dataclass too.
    """

    episodes_completed: int
    elapsed_seconds: float
    convergence: ConvergenceState
    evaluations: tuple[EvaluationReport, ...]


def save_checkpoint(
    path: Path, checkpoint: TrainingCheckpoint, policy_state: NeuralPolicyState
) -> None:
    document = {
        "episodes_completed": checkpoint.episodes_completed,
        "elapsed_seconds": checkpoint.elapsed_seconds,
        "convergence": dataclasses.asdict(checkpoint.convergence),
        "evaluations": checkpoint.evaluations,
        "encoder_state": policy_state.encoder.state_dict(),
        "head_state": policy_state.head.state_dict(),
        "optimizer_state": policy_state.optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(document, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, policy_state: NeuralPolicyState) -> TrainingCheckpoint:
    """Restore ``policy_state`` in place and return the rest of the run's history.

    ``weights_only=False``: the document carries plain dataclasses
    (:class:`~stdvrp.training.neural_report.EvaluationReport`) alongside the
    state dicts, which ``weights_only=True`` (the safe default for
    downloaded, untrusted checkpoints) refuses to unpickle. Every checkpoint
    this module writes is produced by :func:`save_checkpoint` from a run on
    this machine, not loaded from an untrusted source.
    """
    document = torch.load(path, weights_only=False)
    policy_state.encoder.load_state_dict(document["encoder_state"])
    policy_state.head.load_state_dict(document["head_state"])
    policy_state.optimizer.load_state_dict(document["optimizer_state"])
    return TrainingCheckpoint(
        episodes_completed=document["episodes_completed"],
        elapsed_seconds=document["elapsed_seconds"],
        convergence=ConvergenceState(**document["convergence"]),
        evaluations=tuple(document["evaluations"]),
    )
