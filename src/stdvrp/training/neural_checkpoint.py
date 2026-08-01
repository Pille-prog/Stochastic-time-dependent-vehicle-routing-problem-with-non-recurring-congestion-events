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

**The device the weights were written on is part of the document (ticket
12).** CPU and CUDA do not agree bit for bit, so ticket 07's bit-identical
resume guarantee only holds when resuming on the *same* device the checkpoint
was written on. :func:`load_checkpoint` refuses a cross-device resume with a
loud error rather than silently loading anyway (which would produce a
trajectory that *looks* resumed but quietly diverges from the interrupted
run).
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
        "device": str(policy_state.device),
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

    ``policy_state`` must already be built for *this* run's resolved device
    (:func:`~stdvrp.training.neural_episode.build_neural_policy_state` runs
    before this is ever called) -- a mismatch against the device the
    checkpoint was saved on (ticket 12) raises rather than loading anyway,
    since CPU and CUDA do not agree bit for bit and a cross-device resume
    would silently break ticket 07's bit-identical resume guarantee.
    """
    document = torch.load(path, weights_only=False, map_location=policy_state.device)
    checkpoint_device = document["device"]
    resumed_device = str(policy_state.device)
    if checkpoint_device != resumed_device:
        raise RuntimeError(
            f"{path} was written on device={checkpoint_device!r} but this run resolved "
            f"device={resumed_device!r} -- refusing a cross-device resume. CPU and CUDA "
            "do not agree bit for bit, so loading anyway would silently break the "
            "bit-identical resume guarantee. Resume on the original device, or start a "
            "fresh run on this one."
        )
    policy_state.encoder.load_state_dict(document["encoder_state"])
    policy_state.head.load_state_dict(document["head_state"])
    policy_state.optimizer.load_state_dict(document["optimizer_state"])
    return TrainingCheckpoint(
        episodes_completed=document["episodes_completed"],
        elapsed_seconds=document["elapsed_seconds"],
        convergence=ConvergenceState(**document["convergence"]),
        evaluations=tuple(document["evaluations"]),
    )
