"""RidgeAccumulator: the estimator ticket 16 (neural-policy) replaces per-Episode
Adam SGD with -- normal equations accumulated **across** Episodes, with
exponential forgetting, re-solved every ``N`` Episodes rather than every
decision epoch.

Plain numpy, not torch: the accumulator is a ``dim x dim`` matrix (``dim =
515`` at the shipped architecture) and a length-``dim`` vector, small enough
that GPU placement buys nothing and independent of whatever device the
encoder/head happen to live on (``transformer_policy.py`` converts to/from
``torch.Tensor`` at the one boundary where the two meet). Kept in its own
module, not inlined into ``transformer_policy.py``, because the linear algebra
here has nothing to do with tokens, embeddings or Episodes -- it is a generic
accumulated-ridge-regression primitive, independently testable against known
least-squares results.

## The recurrence (spec.md decision 9's amendment; this ticket's own text)

::

    A <- gamma * A + sum_t phi_t phi_t^T
    b <- gamma * b + sum_t phi_t * y_t
    W <- (A + lambda I)^-1 b                # re-solved every N episodes

``gamma`` (exponential forgetting) is applied **once per Episode**, not once
per decision epoch: the ticket's own "effective window ~= 1/(1-gamma)
Episodes" is only true under that reading, and it is applied via
:meth:`observe_episode` on *every* call -- aborted or not, so memory decays
with wall-clock Episodes elapsed rather than with how many of them turned out
usable. A cadence counter (``N``, the number of Episodes between solves) is
the caller's concern, not this class's: ``episodes_since_solve`` here is
informational, read by ``transformer_policy.py`` to decide *when* to call
:meth:`solve`.

## Aborted Episodes are excluded, not merely down-weighted

The heavy tail (research note F10, this ticket's "The heavy tail" section):
``ABORT_PENALTY - 200 * served`` enters every decision epoch's target for the
whole Episode it terminates, dominated by a constant rather than by anything
the candidate ranking did. :meth:`observe_episode`'s ``aborted=True`` branch
still applies the forgetting step (the window keeps moving) but contributes
nothing to ``A``/``b`` -- the exclusion the ticket calls "mandatory", counted
in :attr:`episodes_excluded` for the caller's per-block exclusion-rate report.

## Standardize, then freeze -- the main technical risk (this ticket's own
section of that name)

``A`` is dominated by whichever columns of `phi` happen to run at a larger
natural scale (the encoder's own hidden units and the raw arc facts, not
necessarily the action-conditional ones the argmin actually reads). A single
``lambda`` shrinks every column's *raw* coefficient by an amount that depends
on that column's natural scale -- if the action columns are small, `lambda`
shrinks them disproportionately, `W . phi` goes flat across candidates, and
the argmin has nothing left to rank on: "a clean-looking solve and no
divergence to warn you" (this ticket's own words).

The fix is the standard one -- standardize columns before ridge, so `lambda`
shrinks every *standardized* coefficient by the same relative amount -- with
one adaptation for an accumulator that never buffers samples: the scale is
computed once, from the raw (unscaled) sum-of-squares this class has already
been accumulating for exactly this purpose, at the *first* :meth:`solve` call
that has any data at all, and then **frozen forever** (never recomputed).
"Frozen once" is deliberate, not merely convenient: a moving scale would make
the accumulated ``A``/``b`` non-stationary in a way exponential forgetting
does not account for, undermining the very idea of a running accumulator.
Freezing retroactively rescales the raw ``A``/``b`` accumulated so far
(``A /= outer(scale, scale)``, ``b /= scale`` -- valid because dividing every
sample's `phi` by a fixed per-column constant commutes with summing outer
products) rather than requiring the scale to be known in advance, so ordinary
episodes need carry no separate un-standardized buffer.

A column that is identically zero across every observed sample (``claimed``
is structurally ``False`` for the chosen action of every replayed epoch --
see ``transformer_policy.py``, "ADR-0011") has zero raw variance; a fixed
floor on the scale (:data:`_SCALE_FLOOR`) is what keeps that column's
standardized value finite (0 / floor, not 0 / 0) instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["RidgeAccumulator"]

#: Floor under a column's frozen scale (module docstring, "Standardize, then
#: freeze"). ``claimed`` is identically zero across every accumulated sample by
#: construction, and a handful of other columns can be near-constant on a small
#: or degenerate world (the unit tests' tiny fixtures among them); this keeps
#: standardization finite rather than dividing by (near) zero. Chosen well
#: below the tokenizer's own smallest nonzero natural scale (``tokenizer.py``'s
#: fields run O(1e-3) to O(1) after its own normalization), so it never
#: competes with a column that actually carries signal.
_SCALE_FLOOR = 1e-3


@dataclass(slots=True)
class RidgeAccumulator:
    """Accumulated normal equations for one residual regression, across Episodes.

    Constructed via :meth:`zeros`, not the dataclass constructor directly --
    every array field must share ``dim`` and start at the same zero state, and
    the constructor cannot enforce that on its own.
    """

    dim: int
    forgetting: float
    ridge: float
    A: NDArray[np.float64]
    b: NDArray[np.float64]
    raw_sum_sq: NDArray[np.float64]
    scale: NDArray[np.float64] | None = None
    effective_n: float = 0.0
    episodes_included: int = 0
    episodes_excluded: int = 0
    episodes_since_solve: int = 0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError(f"forgetting (gamma) must be in (0, 1], got {self.forgetting}")
        if self.ridge <= 0.0:
            raise ValueError(f"ridge (lambda) must be positive, got {self.ridge}")

    @classmethod
    def zeros(cls, dim: int, *, forgetting: float, ridge: float) -> RidgeAccumulator:
        return cls(
            dim=dim,
            forgetting=forgetting,
            ridge=ridge,
            A=np.zeros((dim, dim), dtype=np.float64),
            b=np.zeros(dim, dtype=np.float64),
            raw_sum_sq=np.zeros(dim, dtype=np.float64),
        )

    def observe_episode(
        self, phi_rows: NDArray[np.float64], y_rows: NDArray[np.float64], *, aborted: bool
    ) -> None:
        """Fold one Episode's ``(Phi_t, y_t)`` rows in, with forgetting applied once.

        ``phi_rows``: ``[T, dim]``, one joint feature vector per decision epoch
        (``Phi_t = sum_v phi(s, v, a_v)``, the action row actually taken).
        ``y_rows``: ``[T]``, the matching residual targets
        (``y_t = (U_t - acquired) / return_scale - sum_v c(s, v, a_v)``).

        Forgetting (``self.forgetting`` applied to ``A``/``b``/``raw_sum_sq``/
        ``effective_n`` uniformly) happens on *every* call, aborted or not --
        the window is measured in Episodes elapsed (module docstring), not in
        Episodes that contributed data. An aborted Episode contributes
        nothing beyond that decay (module docstring, "Aborted Episodes are
        excluded"). ``raw_sum_sq`` must decay exactly like ``effective_n``,
        its own denominator in :meth:`_freeze_scale`'s ``mean_square =
        raw_sum_sq / effective_n`` -- leaving it undecayed would grow the
        numerator's effective window without bound while the denominator's
        stays at ``~1/(1-gamma)``, silently inflating the frozen scale (and
        therefore silently shrinking ``W`` beyond the configured ``lambda``)
        by more the longer training runs before the first solve.
        """
        if phi_rows.shape != (phi_rows.shape[0], self.dim):
            raise ValueError(f"phi_rows must be [T, {self.dim}], got {phi_rows.shape}")
        if phi_rows.shape[0] != y_rows.shape[0]:
            raise ValueError("phi_rows and y_rows must have the same length")

        self.A *= self.forgetting
        self.b *= self.forgetting
        self.raw_sum_sq *= self.forgetting
        self.effective_n *= self.forgetting
        self.episodes_since_solve += 1

        if aborted:
            self.episodes_excluded += 1
            return

        if self.scale is None:
            self.raw_sum_sq += np.square(phi_rows).sum(axis=0)
            self.A += phi_rows.T @ phi_rows
            self.b += phi_rows.T @ y_rows
        else:
            scaled = phi_rows / self.scale
            self.A += scaled.T @ scaled
            self.b += scaled.T @ y_rows
        self.effective_n += phi_rows.shape[0]
        self.episodes_included += 1

    def _freeze_scale(self) -> None:
        """Compute and freeze the per-column scale from the raw data accumulated
        so far (module docstring, "Standardize, then freeze"). A no-op once
        already frozen, or while there is no data yet to compute it from.
        """
        if self.scale is not None or self.effective_n <= 0.0:
            return
        mean_square = self.raw_sum_sq / self.effective_n
        scale = np.sqrt(np.maximum(mean_square, _SCALE_FLOOR * _SCALE_FLOOR))
        self.A /= np.outer(scale, scale)
        self.b /= scale
        self.scale = scale

    def solve(self) -> NDArray[np.float64]:
        """``W = (A + lambda*I)^-1 b`` in the *physical* (unstandardized) units
        :meth:`observe_episode`'s raw ``phi_rows`` are expressed in -- callers
        never need to know standardization happened at all.

        Freezes the column scale on the first call that has any data
        (``episodes_included > 0`` at some point); before that, or if nothing
        has ever been observed, returns the zero vector -- ``W = 0`` reproduces
        ticket 15's null exactly, which is also this class's own start state.
        """
        self._freeze_scale()
        if self.scale is None:
            return np.zeros(self.dim, dtype=np.float64)
        regularized = self.A + self.ridge * np.eye(self.dim)
        w_standardized = np.linalg.solve(regularized, self.b)
        w: NDArray[np.float64] = w_standardized / self.scale
        self.episodes_since_solve = 0
        return w

    def state_dict(self) -> dict[str, Any]:
        """Plain-value snapshot for checkpointing (``neural_checkpoint.py``)."""
        return {
            "dim": self.dim,
            "forgetting": self.forgetting,
            "ridge": self.ridge,
            "A": self.A.copy(),
            "b": self.b.copy(),
            "raw_sum_sq": self.raw_sum_sq.copy(),
            "scale": None if self.scale is None else self.scale.copy(),
            "effective_n": self.effective_n,
            "episodes_included": self.episodes_included,
            "episodes_excluded": self.episodes_excluded,
            "episodes_since_solve": self.episodes_since_solve,
        }

    @classmethod
    def from_state_dict(cls, document: dict[str, Any]) -> RidgeAccumulator:
        return cls(
            dim=document["dim"],
            forgetting=document["forgetting"],
            ridge=document["ridge"],
            A=document["A"],
            b=document["b"],
            raw_sum_sq=document["raw_sum_sq"],
            scale=document["scale"],
            effective_n=document["effective_n"],
            episodes_included=document["episodes_included"],
            episodes_excluded=document["episodes_excluded"],
            episodes_since_solve=document["episodes_since_solve"],
        )
