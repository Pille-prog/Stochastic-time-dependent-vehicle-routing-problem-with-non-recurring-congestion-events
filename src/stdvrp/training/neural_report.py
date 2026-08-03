"""The live paired report and convergence stopping rule (ticket 07, neural-policy).

Every evaluation block is a **paired** comparison against the reference card
(ticket 01): the same ``evaluation_seeds``, seed for seed, never a difference
of means. With this cost distribution's variance, two means can hide a real
few-percent effect — the whole point of the live report is to tell
"improving" from "nothing is happening" within the first couple of blocks,
not after a run that took hours (spec.md, "The live report").

**These seeds select checkpoints and hyperparameters; they never render a
verdict.** ``evaluation_seeds`` are contaminated by construction — everything
in this module reads them — so nothing here may be presented as the effort's
answer to "does it win." That question is ``test_seeds``, held out until
tickets 08/09.

Pure and dependency-free from the rest of the training stack on purpose: no
import of ``stdvrp.training.trainer`` (would cycle back, since ``Trainer``
imports this module) or of anything that touches torch, so this module's
formatting and convergence-state logic are testable without a simulator, a
network, or even a paired dataset larger than what a test writes by hand.

Genuinely torch-free at *runtime* (importable with torch never installed —
verified directly, not merely by inspection: ``uv sync`` without ``--extra
neural`` then importing this module cleanly). **Not testable via
``monkeypatch.setitem(sys.modules, "torch", None)``** the way
``test_torch_support.py`` checks ``stdvrp.policies`` — ``scipy.stats``'s own
``array_api_compat`` layer treats a ``None``-valued ``sys.modules["torch"]``
entry (present but not a real module) differently from the key being absent
entirely, and crashes on an unrelated ``AttributeError`` during its own
import-time example-doc generation. That is an artifact of the injection
technique meeting scipy's own torch detection, not a real gap: confirmed by
actually uninstalling torch (``uv sync`` without the extra) and importing this
module successfully, twice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from scipy import stats

#: Evaluation blocks without an improved best mean before the learning rate
#: is cut. Frozen in spec.md's "Frozen parameters" table, not a config knob.
PATIENCE_BLOCKS = 5

#: What the learning rate is multiplied by on a patience trigger.
LR_REDUCTION_FACTOR = 0.3

#: Reductions already applied when a further patience trigger declares the
#: run converged, rather than cutting the rate a fourth time.
MAX_REDUCTIONS = 3

#: The hard safety net (spec.md: "not the budget"). A run that reaches either
#: bound is reported "did not converge" and may never be presented as clean.
MAX_EPISODES = 10_000
MAX_HOURS = 24.0

#: Evaluation cadence's floor (spec.md: "~every 50 episodes").
MIN_EVALUATION_CADENCE = 50

#: Cadence grows so a run stays at roughly this many evaluation blocks total,
#: however long it runs — the mechanism behind "scales with run length".
TARGET_EVALUATION_BLOCKS = 40

#: Ticket 16's mandatory mitigation for the ridge estimator's blind spot: an
#: Episode the accumulator excluded (aborted at ``CLOCK_CEILING``, research
#: note F10) supplies no gradient *away* from the abort region, so a policy
#: that starts aborting degrades silently -- "the same 'small loss, destroyed
#: policy' signature as ticket 08's failure, with a different cause" (this
#: ticket's own words). A block whose exclusion rate reaches this ceiling is
#: exactly that signal: healthy training excludes close to 0% of Episodes, so
#: a run into double digits is already well past "rising" and into "acting on
#: it is overdue." Not a config knob -- a protocol-level stop condition, the
#: same status as :data:`PATIENCE_BLOCKS`/:data:`MAX_REDUCTIONS` above.
EXCLUSION_RATE_STOP_THRESHOLD = 0.10


def evaluation_cadence(
    episodes_completed: int,
    *,
    minimum: int = MIN_EVALUATION_CADENCE,
    target_blocks: int = TARGET_EVALUATION_BLOCKS,
) -> int:
    """Episodes between evaluation blocks, at the current point in the run.

    A fixed cadence (or the linear baseline's ``test_frequency``) either
    stays too tight for a long run (spec.md's own example: ``test_frequency:
    10`` on a 2000-episode run spends 10 000 evaluation episodes evaluating —
    more than the training itself) or too loose for a short one. Recomputing
    this from episodes completed *so far* — not a total the convergence-based
    loop does not have in advance — keeps the count of evaluation blocks
    roughly constant regardless of how long the run turns out to run.

    ``minimum``/``target_blocks`` default to the frozen spec values; a caller
    only ever overrides them to shrink a test's wall-clock (real training
    runs use the defaults).
    """
    return max(minimum, episodes_completed // target_blocks)


def paired_wilcoxon_p(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Two-sided Wilcoxon signed-rank p-value over two paired sequences.

    ``NaN`` when every pair is exactly tied (``scipy`` has nothing to rank) —
    reported as such rather than raising. Shared by :attr:`EvaluationReport.wilcoxon_p`
    (paired against the reference card) and Gate A's own trained-vs-untrained
    comparison (``stdvrp.training.gate_a``, ticket 08) — the same idiom, two
    different pairs of things being compared.
    """
    differences = [x - y for x, y in zip(a, b, strict=True)]
    if all(difference == 0 for difference in differences):
        return float("nan")
    result = stats.wilcoxon(a, b)
    return float(result.pvalue)


def candidate_spread_ratio(pairs: tuple[tuple[float, float], ...]) -> float:
    """``r = sd_candidates(W.phi) / sd_candidates(c)`` (ticket 17, Gate A''s
    companion diagnostic, spec.md decision 10's amendment).

    ``pairs`` is one ``(sd_residual, sd_cost)`` per (decision epoch, vehicle)
    with at least two candidates
    (``TransformerMonteCarloPolicy.spread_samples``) -- pooled here as the
    mean residual spread over the mean cost spread, across every sample given.
    ``NaN`` on no samples, or when the mean cost spread is exactly zero (a
    degenerate pool where every candidate's myopic base agreed, so the ratio
    is undefined rather than infinite) -- reported as such rather than
    crashing a live training report or a gate run over a diagnostic.

    ``r ~= 0``: the learned term never touches the ranking (or `lambda`
    shrank it away). ``r`` in ``0.1..0.5``: correcting the base without
    overwriting it -- the target. ``r >> 1``: overwriting ``c(s, a)`` --
    ticket 08's failure mode, returning.
    """
    if not pairs:
        return float("nan")
    residual_sds = [residual_sd for residual_sd, _cost_sd in pairs]
    cost_sds = [cost_sd for _residual_sd, cost_sd in pairs]
    mean_cost_sd = float(np.mean(cost_sds))
    if mean_cost_sd == 0.0:
        return float("nan")
    return float(np.mean(residual_sds)) / mean_cost_sd


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """One evaluation block, paired seed-by-seed against the reference card.

    ``training_episodes``/``excluded_episodes`` (ticket 16) are the training
    Episodes run *since the previous block* (not the cumulative
    ``episodes_completed``) and how many of those the ridge estimator excluded
    as aborted (``TransformerMonteCarloPolicy._is_aborted`` /
    ``RidgeAccumulator.episodes_excluded``). Both default to ``0`` so a
    caller building a report without them (every test predating ticket 16)
    keeps working unchanged, at :attr:`exclusion_rate` ``0.0``.

    ``spread_samples`` (ticket 17) is every evaluation Episode's
    candidate-spread pairs this block ran, pooled
    (``TransformerMonteCarloPolicy.spread_samples`` from every
    ``evaluation_seeds`` Episode -- ``Trainer._run_neural_evaluation_block``).
    Defaults to ``()`` so a report built without it (every test predating
    ticket 17) keeps working unchanged, at :attr:`candidate_spread_ratio`
    ``NaN``.
    """

    episodes_completed: int
    seed_costs: tuple[float, ...]
    reference_seed_costs: tuple[float, ...]
    training_episodes: int = 0
    excluded_episodes: int = 0
    spread_samples: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if len(self.seed_costs) != len(self.reference_seed_costs):
            raise ValueError("seed_costs and reference_seed_costs must be the same paired length")
        if not self.seed_costs:
            raise ValueError("an evaluation block must have at least one paired seed")
        if self.excluded_episodes > self.training_episodes:
            raise ValueError("excluded_episodes cannot exceed training_episodes")

    @property
    def exclusion_rate(self) -> float:
        """Fraction of this block's training Episodes the estimator excluded.

        ``0.0`` when ``training_episodes`` is ``0`` (a report built without
        that count at all, or the degenerate case of a block following zero
        training Episodes) rather than raising on a division by zero -- the
        live report must never crash a training run over this diagnostic.
        """
        if self.training_episodes == 0:
            return 0.0
        return self.excluded_episodes / self.training_episodes

    @property
    def candidate_spread_ratio(self) -> float:
        """``r`` (ticket 17, Gate A''s companion diagnostic) over this block's
        pooled ``spread_samples`` -- see :func:`candidate_spread_ratio`."""
        return candidate_spread_ratio(self.spread_samples)

    @property
    def mean_cost(self) -> float:
        return sum(self.seed_costs) / len(self.seed_costs)

    @property
    def reference_mean_cost(self) -> float:
        return sum(self.reference_seed_costs) / len(self.reference_seed_costs)

    @property
    def delta(self) -> float:
        """This block's mean minus the reference's — negative is an improvement.

        Reads like "a difference of means" but is not one in the sense the
        module docstring warns against: it equals the mean of the per-seed
        differences exactly, because ``seed_costs`` and
        ``reference_seed_costs`` are always built in the same
        ``evaluation_seeds`` order (``__post_init__`` requires equal length;
        callers are responsible for the shared order, verified by
        ``Trainer.train_neural`` pairing both from the same
        ``config.evaluation_seeds`` iteration). The paired quantities this
        class exists to expose — :attr:`wins`, :attr:`wilcoxon_p` — read
        ``seed_costs``/``reference_seed_costs`` directly, seed by seed; this
        property is a convenience view of the same pairing, not a shortcut
        that drops it.
        """
        return self.mean_cost - self.reference_mean_cost

    @property
    def delta_pct(self) -> float:
        return 100.0 * self.delta / self.reference_mean_cost

    @property
    def wins(self) -> int:
        """Seeds where this block beat the reference card, strictly."""
        return sum(
            cost < reference
            for cost, reference in zip(self.seed_costs, self.reference_seed_costs, strict=True)
        )

    @property
    def wilcoxon_p(self) -> float:
        """Two-sided Wilcoxon signed-rank p-value over the paired per-seed costs.

        ``NaN`` when every pair is exactly tied (``scipy`` has nothing to rank)
        — reported as such rather than raising, since a live report must never
        crash a training run over a degenerate block.
        """
        return paired_wilcoxon_p(self.seed_costs, self.reference_seed_costs)


class ConvergenceAction(Enum):
    """What :func:`update_convergence` decided this block should do."""

    CONTINUE = "continue"
    REDUCE_LR = "reduce_lr"
    CONVERGED = "converged"


@dataclass(slots=True)
class ConvergenceState:
    """Mutable patience/best-so-far bookkeeping, updated one evaluation block at a time."""

    current_lr: float
    best_mean_cost: float = float("inf")
    best_episode: int = 0
    best_delta_pct: float = 0.0
    blocks_without_improvement: int = 0
    reductions_applied: int = 0


def update_convergence(state: ConvergenceState, report: EvaluationReport) -> ConvergenceAction:
    """Advance ``state`` from one evaluation block; mutates ``state`` in place.

    Patience (spec.md, "Frozen parameters"): :data:`PATIENCE_BLOCKS` blocks
    without an improved best mean multiplies ``current_lr`` by
    :data:`LR_REDUCTION_FACTOR`. After :data:`MAX_REDUCTIONS` reductions, a
    further patience trigger means "still no improvement after cutting the
    rate three times" — that is convergence, not a fourth cut.
    """
    if report.mean_cost < state.best_mean_cost:
        state.best_mean_cost = report.mean_cost
        state.best_episode = report.episodes_completed
        state.best_delta_pct = report.delta_pct
        state.blocks_without_improvement = 0
        return ConvergenceAction.CONTINUE

    state.blocks_without_improvement += 1
    if state.blocks_without_improvement < PATIENCE_BLOCKS:
        return ConvergenceAction.CONTINUE

    state.blocks_without_improvement = 0
    if state.reductions_applied >= MAX_REDUCTIONS:
        return ConvergenceAction.CONVERGED

    state.reductions_applied += 1
    state.current_lr *= LR_REDUCTION_FACTOR
    return ConvergenceAction.REDUCE_LR


def hit_safety_cap(
    *,
    episodes_completed: int,
    elapsed_seconds: float,
    max_episodes: int = MAX_EPISODES,
    max_hours: float = MAX_HOURS,
) -> bool:
    """The hard net (spec.md: "not the budget") — 10 000 episodes or 24 hours.

    ``max_episodes``/``max_hours`` default to the frozen spec values; a
    caller only ever overrides them to make a test's cap reachable quickly.
    """
    return episodes_completed >= max_episodes or elapsed_seconds >= max_hours * 3600.0


def exclusion_rate_stop_signal(
    report: EvaluationReport, *, threshold: float = EXCLUSION_RATE_STOP_THRESHOLD
) -> bool:
    """Ticket 16's mandatory mitigation: "a rising exclusion rate is a stop
    signal."

    A run that starts aborting is exactly what the ridge estimator is blind
    to (module docstring at :data:`EXCLUSION_RATE_STOP_THRESHOLD`), so this is
    checked every block, unconditionally -- not only when something else
    already looks wrong. ``threshold`` defaults to the frozen protocol value;
    a caller only ever overrides it to make a test's trigger reachable
    without needing a genuinely diverging policy.
    """
    return report.exclusion_rate >= threshold


# --- Formatting ------------------------------------------------------------------


def format_lr(lr: float) -> str:
    """``3.0e-4``, not Python's default two-digit-exponent ``3.0e-04``.

    Matches spec.md's own printed example exactly; used everywhere an ``lr``
    value is logged (this module's two formatters and ``Trainer.train_neural``'s
    patience/resume log lines), so a reduced rate reads the same way a fresh
    one does.
    """
    mantissa, exponent = f"{lr:.1e}".split("e")
    sign, digits = exponent[0], exponent[1:].lstrip("0") or "0"
    return f"{mantissa}e{sign}{digits}"


def format_episode_line(
    *, episode: int, seed: int, cost: float, loss: float, lr: float, wall_clock_seconds: float
) -> str:
    """``[ep  348] train seed 1348   cost 2691.3   loss 3.9e-3   lr 3.0e-4   8.9s``.

    ``loss`` is printed in scientific notation, not spec.md's illustrative
    ``%.3f``. The quantity is a Huber over ``targets[t] / _return_scale``
    (``transformer_policy.py``), which for a healthy Chengdu episode lands
    around ``2500 / 127500 ~ 0.02`` — squared, that is ``2e-4``, and every
    value a converging run produces rounds to ``0.000`` at three decimals.
    Ticket 08's Gate A log printed exactly that for 1 338 consecutive
    episodes, so the one number in the live report that could have shown the
    fit collapsing showed nothing at all. Scientific notation costs the same
    column width and distinguishes ``2e-4`` from ``2e-8``.
    """
    return (
        f"[ep {episode:4d}] train seed {seed}   cost {cost:.1f}   "
        f"loss {loss:.1e}   lr {format_lr(lr)}   {wall_clock_seconds:.1f}s"
    )


_BLOCK_RULE = "-" * 74


def format_evaluation_block(report: EvaluationReport, state: ConvergenceState) -> str:
    """The multi-line paired block report (spec.md, "The live report").

    Read ``state`` *after* :func:`update_convergence` has already advanced
    it for this same ``report`` — "best so far" and the patience counter both
    describe the state resulting from this block, not the block before it.
    """
    p_value = report.wilcoxon_p
    p_display = "nan" if p_value != p_value else f"{p_value:.3f}"  # NaN != NaN
    lines = [
        _BLOCK_RULE,
        f"  eval @{report.episodes_completed}   mean {report.mean_cost:.1f}  |  "
        f"MC ref {report.reference_mean_cost:.1f}  |  "
        f"delta {report.delta:+.1f}  ({report.delta_pct:+.1f}%)",
        f"              wins on {report.wins}/{len(report.seed_costs)} seeds"
        f"    Wilcoxon p={p_display}",
        f"              best so far: {state.best_delta_pct:+.1f}% @ep{state.best_episode}"
        f"     no improvement: {state.blocks_without_improvement}/{PATIENCE_BLOCKS} blocks",
    ]
    # Ticket 16: "log the exclusion count and rate per evaluation block" --
    # only printed once there is a training-episode count to rate against
    # (every real run; some pre-ticket-16 reports built for other tests omit
    # it, at its 0/0 default).
    if report.training_episodes > 0:
        lines.append(
            f"              excluded {report.excluded_episodes}/{report.training_episodes} "
            f"training episodes ({report.exclusion_rate * 100:.1f}%)"
        )
    # Ticket 17's companion diagnostic (spec.md decision 10's amendment):
    # "reported every block", not gated -- only printed once there is at
    # least one candidate-spread sample to pool (every real evaluation block;
    # pre-ticket-17 reports built for other tests omit it, at its NaN default).
    if report.spread_samples:
        r_value = report.candidate_spread_ratio
        r_display = "nan" if r_value != r_value else f"{r_value:.3f}"  # NaN != NaN
        lines.append(f"              r = sd(W.phi)/sd(c) = {r_display}")
    lines.append(_BLOCK_RULE)
    return "\n".join(lines)


def open_training_log(path: Path, *, echo: bool = True) -> Callable[[str], None]:
    """A ``Trainer(log=...)`` callback that appends every message to ``path``.

    Runs are hours long on a laptop (ticket 07): a log file is what lets one
    survive the terminal it was started in. Opens in **append** mode and
    flushes after every write, so a resumed run's log is continuous with the
    interrupted one and a hard kill loses at most the message currently being
    written, never an earlier one. ``echo`` also prints each message to
    stdout — the default, since a file-only log defeats "watchable".

    The returned callback keeps the file handle open for the caller's whole
    process lifetime rather than as a context manager: a training run's log
    is meant to outlive any single call into this function, and the process
    exiting closes it naturally.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", newline="\n")

    def log(message: str) -> None:
        handle.write(message + "\n")
        handle.flush()
        if echo:
            print(message)

    return log
