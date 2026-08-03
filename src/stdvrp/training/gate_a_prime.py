"""Gate A' -- does training add anything on top of the base? (ticket 17, neural-policy).

Rewritten from ticket 08's Gate A (preserved, untouched, in ``gate_a.py``) for
the residual decomposition ticket 15 introduced: the untrained network is no
longer a nearest-Client placeholder, it **is** ``Q = c(s, a)`` -- a genuinely
strong cost-greedy dispatcher over the linear baseline's own candidate set
(ticket 08 measured it beating 1150 episodes of training by 13.8%). "Does it
learn" therefore sharpens into "does training add value on top of a good
initialization", and two of ticket 08's three parts break if reused verbatim
(spec.md, "Gate A'"):

1. **The null model's threshold stays at >= 5%** (:data:`GATE_A_PRIME_EFFECT_THRESHOLD_PCT`,
   :data:`GATE_A_PRIME_SIGNIFICANCE_THRESHOLD`) -- unchanged from ticket 08,
   but now measured against the much harder myopic-base null rather than
   nearest-neighbour, and deliberately *not* coupled to the linear baseline
   (spec.md: the gap to the baseline is policy-dependent and reads 41.3% on
   ``evaluation_seeds`` against 11.2% on ``test_seeds`` -- "a threshold that
   cannot be computed on one seed set is not a threshold").
2. **Reproducibility stays at >= 3 init seeds** (:data:`GATE_A_PRIME_MIN_INIT_SEEDS`),
   now **per arm** -- 3 seeds x 2 arms, six trained networks per real run.
3. **Calibration is redefined onto the residual**: ``rho(W . phi, y~)``
   (:class:`~stdvrp.policies.transformer_policy.ResidualCalibrationPair`), not
   ``rho(Q, U_t)`` (ticket 08's pairing) -- the latter passes at ``W = 0``
   with no parameter having moved, since ``Q == c`` there and ``c`` alone
   already correlates with the return. A guaranteed false PASS under the
   residual decomposition, which is why this is a *redefinition*, not a
   port.

## The two arms, both run unconditionally

``frozen`` (``train_encoder=False``, ticket 16's arm: ridge only, no learning
rate anywhere) and ``trained`` (``train_encoder=True``, ticket 17's own:
ridge + SGD over ``encoder``/``head.layer1``, spec.md's "Two timescales").
Both run for every one of :data:`GATE_A_PRIME_MIN_INIT_SEEDS` init seeds,
regardless of whether the other already lost -- running ``trained`` only
when ``frozen`` loses would mean the effort's original thesis ("a transformer
over raw State beats a 19-feature linear VFA") never gets a number if
``frozen`` happens to win (spec.md, "Both run whatever the first says").
:attr:`GateAPrimeResult.representation_learning_delta_pct` is that number:
``delta(trained) - delta(frozen)``, signed.

## The companion diagnostic, ``r`` (reported, never gated)

``r = sd_candidates(W . phi) / sd_candidates(c)``
(:func:`~stdvrp.training.neural_report.candidate_spread_ratio`, pooled from
:class:`~stdvrp.policies.transformer_policy.TransformerMonteCarloPolicy.spread_samples`
over every measurement seed). ``r ~= 0``: the ranking was never touched (or
`lambda` shrank it away). ``r`` in ``0.1..0.5``: correcting the base without
overwriting it. ``r >> 1``: overwriting ``c(s, a)`` -- ticket 08's failure
mode, returning.

Torch-free at module scope, exactly like ``gate_a.py``/``trainer.py``: every
import that reaches torch is deferred into the functions that actually run a
network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import stats

from stdvrp.training.neural_report import candidate_spread_ratio, paired_wilcoxon_p
from stdvrp.training.reference_card import ReferenceCard
from stdvrp.training.trainer import Trainer

if TYPE_CHECKING:
    from stdvrp.training.episode_pool import EpisodeWorld
    from stdvrp.training.neural_episode import NeuralPolicyState

__all__ = [
    "BASELINE_BEST_CELL_COST",
    "BASELINE_LIKE_FOR_LIKE_COST",
    "GATE_A_PRIME_CALIBRATION_THRESHOLD",
    "GATE_A_PRIME_EFFECT_THRESHOLD_PCT",
    "GATE_A_PRIME_MIN_INIT_SEEDS",
    "GATE_A_PRIME_SIGNIFICANCE_THRESHOLD",
    "ArmResult",
    "ArmSummary",
    "GateAPrimeResult",
    "format_gate_a_prime_report",
    "run_gate_a_prime",
]

#: Null-model effect size Gate A' requires: mean cost reduction, trained vs.
#: the myopic base (W = 0), over test_seeds (spec.md's "Frozen parameters",
#: unchanged from ticket 08 -- what changed is the null, not this number).
GATE_A_PRIME_EFFECT_THRESHOLD_PCT = 5.0

#: Wilcoxon signed-rank significance threshold for the null-model test.
GATE_A_PRIME_SIGNIFICANCE_THRESHOLD = 0.05

#: Minimum acceptable Spearman rho(W.phi, y~) on a trained network.
GATE_A_PRIME_CALIBRATION_THRESHOLD = 0.5

#: Reproducibility needs at least this many independent network-init seeds, per arm.
GATE_A_PRIME_MIN_INIT_SEEDS = 3

#: The two arms this gate always runs, both places -- shared so
#: ``ArmResult``/``ArmSummary`` validate against one definition rather than
#: two copies that could drift apart.
Arm = Literal["frozen", "trained"]
_ARM_NAMES: tuple[Arm, ...] = ("frozen", "trained")

#: The linear baseline's own frozen numbers (ticket 01's sweep,
#: ``results/baseline_null_50.py``), named alongside every trained number
#: per spec.md's standing obligation -- context for Gate B (ticket 09), never
#: a threshold Gate A' itself gates on (spec.md: the gap to either number is
#: policy-dependent and not computable on one seed set alone).
BASELINE_BEST_CELL_COST = 3384.82  # budget 100, m + 40 -- Gate B's own verdict cell
BASELINE_LIKE_FOR_LIKE_COST = 3458.4  # budget 100, m + 2 -- identical action set to this Policy


def _spearman(pairs: tuple[tuple[float, float], ...]) -> float:
    """Spearman rho over ``(predicted, realised)`` pairs; NaN below 2 samples
    or when either side is constant (scipy warns and returns NaN itself) --
    reported as such, matching ``gate_a.py``'s own "NaN rather than crash"
    discipline. At ``W = 0`` the predicted half (``W . phi``) is identically
    zero for every sample, so this is NaN there by construction -- the
    guaranteed non-PASS spec.md's redefinition needs (module docstring).
    """
    if len(pairs) < 2:
        return float("nan")
    predicted = [p for p, _ in pairs]
    realised = [u for _, u in pairs]
    result = stats.spearmanr(predicted, realised)
    return float(result.statistic)


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm's one network-init seed's complete Gate A' measurement.

    ``residual_calibration`` is ``(W.phi, y~)`` pairs pooled across every
    seed's decision epochs
    (:meth:`~stdvrp.policies.transformer_policy.TransformerMonteCarloPolicy.residual_calibration_pairs`).
    ``spread_samples`` is ``(sd_candidates(W.phi), sd_candidates(c))`` pairs,
    one per (decision epoch, vehicle) with >= 2 candidates
    (``TransformerMonteCarloPolicy.spread_samples``) -- the ``r`` diagnostic's
    raw material.
    """

    arm: Arm
    init_seed: int
    trained_seed_costs: tuple[float, ...]
    untrained_seed_costs: tuple[float, ...]
    residual_calibration: tuple[tuple[float, float], ...]
    spread_samples: tuple[tuple[float, float], ...]
    episodes_completed: int
    converged: bool

    def __post_init__(self) -> None:
        if self.arm not in _ARM_NAMES:
            raise ValueError(f"arm must be one of {_ARM_NAMES}, got {self.arm!r}")
        if len(self.trained_seed_costs) != len(self.untrained_seed_costs):
            raise ValueError(
                "trained_seed_costs and untrained_seed_costs must be the same paired length"
            )
        if not self.trained_seed_costs:
            raise ValueError("an arm must measure at least one seed")

    @property
    def seed_reductions_pct(self) -> tuple[float, ...]:
        """Per-seed percent cost reduction, trained vs. the myopic base; positive improves."""
        return tuple(
            100.0 * (untrained - trained) / untrained
            for trained, untrained in zip(
                self.trained_seed_costs, self.untrained_seed_costs, strict=True
            )
        )

    @property
    def mean_reduction_pct(self) -> float:
        return float(np.mean(self.seed_reductions_pct))

    @property
    def median_reduction_pct(self) -> float:
        """Report alongside :attr:`mean_reduction_pct` -- the right-tailed cost
        distribution (F10) makes the two genuinely differ (ticket 17's own
        work item: "report mean and median improvement")."""
        return float(np.median(self.seed_reductions_pct))

    @property
    def wilcoxon_p(self) -> float:
        """Two-sided paired Wilcoxon signed-rank p-value, trained vs. the myopic base."""
        return paired_wilcoxon_p(self.trained_seed_costs, self.untrained_seed_costs)

    @property
    def null_model_passes(self) -> bool:
        """This seed alone: significant *and* at least the required effect size."""
        p = self.wilcoxon_p
        return (
            p == p  # not NaN
            and p < GATE_A_PRIME_SIGNIFICANCE_THRESHOLD
            and self.mean_reduction_pct >= GATE_A_PRIME_EFFECT_THRESHOLD_PCT
        )

    @property
    def calibration_spearman(self) -> float:
        return _spearman(self.residual_calibration)

    @property
    def calibration_passes(self) -> bool:
        rho = self.calibration_spearman
        return rho == rho and rho >= GATE_A_PRIME_CALIBRATION_THRESHOLD  # not NaN

    @property
    def candidate_spread_ratio(self) -> float:
        """``r`` (spec.md decision 10's amendment) over this seed's own
        measurement -- reported, never gated."""
        return candidate_spread_ratio(self.spread_samples)


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """One arm's full reproducibility protocol: every init seed, plus the combined verdict."""

    arm: Arm
    results: tuple[ArmResult, ...]

    def __post_init__(self) -> None:
        if self.arm not in _ARM_NAMES:
            raise ValueError(f"arm must be one of {_ARM_NAMES}, got {self.arm!r}")
        if not self.results:
            raise ValueError("an arm summary needs at least one init seed's result")
        if any(result.arm != self.arm for result in self.results):
            raise ValueError("every ArmResult in this summary must match its own arm")

    @property
    def mean_reductions_pct(self) -> tuple[float, ...]:
        """One number per init seed -- the distribution reproducibility is measured over."""
        return tuple(result.mean_reduction_pct for result in self.results)

    @property
    def reproducibility_mean_pct(self) -> float:
        return float(np.mean(self.mean_reductions_pct))

    @property
    def reproducibility_sd_pct(self) -> float:
        return float(np.std(self.mean_reductions_pct))

    @property
    def reproducibility_spread_straddles_zero(self) -> bool:
        """``mean +/- sd`` crossing zero means "did not learn", even if every
        individual seed cleared its own bar (ticket 08's own wording, carried
        forward unchanged by ticket 17)."""
        mean = self.reproducibility_mean_pct
        sd = self.reproducibility_sd_pct
        return (mean - sd) <= 0.0 <= (mean + sd)

    @property
    def null_model_passes(self) -> bool:
        """Every init seed individually passes, and there are enough of them
        to call it reproducibility rather than one lucky init.

        Ticket 08's original aggregation rule (``gate_a.py``'s
        ``GateAResult.null_model_passes``), carried forward unchanged: this
        ticket revises Part 1's *null* (the myopic base, not
        nearest-neighbour) and Part 3's *target* (the residual, not raw
        ``Q``), not how Part 1 aggregates across seeds. A single pooled test
        over every seed of every init seed was not considered here, for the
        same reason ticket 08 did not consider it: reproducibility is
        specifically about *each* independent network clearing the bar on its
        own, not about a large enough pooled sample making a small aggregate
        effect look significant.
        """
        return len(self.results) >= GATE_A_PRIME_MIN_INIT_SEEDS and all(
            result.null_model_passes for result in self.results
        )

    @property
    def reproducibility_passes(self) -> bool:
        return self.null_model_passes and not self.reproducibility_spread_straddles_zero

    @property
    def calibration_passes(self) -> bool:
        return all(result.calibration_passes for result in self.results)

    @property
    def passes(self) -> bool:
        """All three parts of Gate A', for this arm alone."""
        return self.reproducibility_passes and self.calibration_passes

    @property
    def mean_candidate_spread_ratio(self) -> float:
        """Mean ``r`` across this arm's init seeds -- reported, never gated:
        NaN-safe (a seed whose measurement produced no >= 2-candidate
        decision epoch, or a zero cost spread, is dropped from the mean
        rather than poisoning it)."""
        ratios = [result.candidate_spread_ratio for result in self.results]
        finite = [ratio for ratio in ratios if ratio == ratio]  # not NaN
        return float(np.mean(finite)) if finite else float("nan")


@dataclass(frozen=True, slots=True)
class GateAPrimeResult:
    """Both arms' full protocol, plus the decomposition the effort has been
    trying to produce since it started: the value of representation learning.
    """

    frozen: ArmSummary
    trained: ArmSummary

    def __post_init__(self) -> None:
        if self.frozen.arm != "frozen":
            raise ValueError("frozen summary must carry arm='frozen'")
        if self.trained.arm != "trained":
            raise ValueError("trained summary must carry arm='trained'")

    @property
    def representation_learning_delta_pct(self) -> float:
        """``delta(trained) - delta(frozen)``, signed (ticket 17's Acceptance:
        "stated explicitly as a number with a sign, not left to be inferred
        from two tables"). Positive: training the encoder adds value beyond
        the ridge-only fit. Negative: it spends what the ridge alone bought."""
        return self.trained.reproducibility_mean_pct - self.frozen.reproducibility_mean_pct

    @property
    def passes(self) -> bool:
        """Gate A' as a whole passes if **either** arm clears all three parts
        on its own.

        Not specified verbatim by spec.md's three-part table (which grades
        each arm, "per arm"), but consistent with the ladder that follows it
        (decision 11/14: "Gate B with the better of the two") -- the effort
        only needs one viable arm to proceed, and both are measured
        regardless so the decomposition above always exists. If neither
        passes, the gate fails outright and the "If the gate fails" table
        (rho, r) is what explains why.
        """
        return self.frozen.passes or self.trained.passes

    @property
    def null_mean_cost(self) -> float:
        """The shared untrained (myopic-base) network's mean cost over the
        measurement seeds -- identical across every arm and every init seed
        by construction (:func:`run_gate_a_prime` builds it once, before
        either arm trains). Read off ``frozen`` arbitrarily; ``trained``
        carries the identical tuple.
        """
        return float(np.mean(self.frozen.results[0].untrained_seed_costs))


def _evaluate_null_over_seeds(
    world: EpisodeWorld, policy_state: NeuralPolicyState, seeds: tuple[int, ...]
) -> tuple[float, ...]:
    """Greedy cost only, for the shared untrained (myopic-base) network."""
    from stdvrp.training.neural_episode import run_neural_evaluation_episode

    costs: list[float] = []
    for seed in seeds:
        result, _spread = run_neural_evaluation_episode(
            seed=seed,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=policy_state,
            config=world.config,
        )
        costs.append(result.total_cost)
    return tuple(costs)


def _evaluate_arm_over_seeds(
    world: EpisodeWorld, policy_state: NeuralPolicyState, seeds: tuple[int, ...]
) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Greedy cost + residual calibration pairs + candidate-spread samples,
    for one trained network, over ``seeds``."""
    from stdvrp.training.neural_episode import run_neural_residual_calibration_episode

    costs: list[float] = []
    residual_pairs: list[tuple[float, float]] = []
    spread_samples: list[tuple[float, float]] = []
    for seed in seeds:
        result, pairs, seed_spread = run_neural_residual_calibration_episode(
            seed=seed,
            client_generator=world.client_generator,
            travel_time_model=world.travel_time_model,
            shortest_path_cache=world.shortest_path_cache,
            congestion_generator=world.congestion_generator,
            policy_state=policy_state,
            config=world.config,
        )
        costs.append(result.total_cost)
        residual_pairs.extend(pairs)
        spread_samples.extend(seed_spread)
    return tuple(costs), tuple(residual_pairs), tuple(spread_samples)


def run_gate_a_prime(
    world: EpisodeWorld,
    *,
    reference_card: ReferenceCard,
    checkpoint_dir: Path,
    init_seeds: tuple[int, ...] = (0, 1, 2),
    test_seeds: tuple[int, ...] | None = None,
    untrained_init_seed: int = 0,
    max_episodes: int | None = None,
    max_hours: float | None = None,
    evaluation_cadence_minimum: int | None = None,
    log: Callable[[str], None] | None = None,
) -> GateAPrimeResult:
    """Run the whole Gate A' protocol: both arms, ``len(init_seeds)`` networks
    each, measured against one shared untrained (null) network.

    ``test_seeds`` defaults to ``world.config.test_seeds`` -- the verdict set,
    never ``evaluation_seeds`` (spec.md's anti-contamination rule; those only
    ever feed ``reference_card``'s live report during training below).

    The untrained network is built **once**, not once per arm or per seed:
    ticket 15's myopic base makes ``Q == c`` at init regardless of
    ``init_rng`` (which arm's encoder gets trained afterwards has no bearing
    on what the untrained network computes) -- ``test_network.py``/
    ``test_transformer_policy.py`` pin this numerically.

    Each seed's own checkpoint lands at
    ``checkpoint_dir / f"gate_a_prime_{arm}_init{seed}.pt"``.
    ``max_episodes``/``max_hours``/``evaluation_cadence_minimum`` default to
    spec.md's frozen convergence parameters (``None``, forwarded to
    :meth:`~stdvrp.training.trainer.Trainer.train_neural`); pass them only to
    make a dev run's cap reachable quickly, never for the real gate.
    """
    from stdvrp.training.neural_episode import build_neural_policy_state

    config = world.config
    seeds = test_seeds if test_seeds is not None else config.test_seeds
    emit = log if log is not None else lambda _message: None

    emit(f"Gate A': building the shared untrained (myopic-base) network, {len(seeds)} test seeds")
    untrained_state = build_neural_policy_state(config, np.random.default_rng(untrained_init_seed))
    untrained_costs = _evaluate_null_over_seeds(world, untrained_state, seeds)
    emit(f"Gate A': untrained (myopic base) mean cost {float(np.mean(untrained_costs)):.4f}")

    summaries: dict[Arm, ArmSummary] = {}
    arm_encoder_flags: tuple[tuple[Arm, bool], ...] = (("frozen", False), ("trained", True))
    for arm, train_encoder in arm_encoder_flags:
        arm_results: list[ArmResult] = []
        for init_seed in init_seeds:
            emit(f"Gate A' arm={arm}: init_seed={init_seed} -- training to convergence")
            trainer = Trainer(world, log=log)
            training = trainer.train_neural(
                reference_card=reference_card,
                checkpoint_path=checkpoint_dir / f"gate_a_prime_{arm}_init{init_seed}.pt",
                max_episodes=max_episodes,
                max_hours=max_hours,
                evaluation_cadence_minimum=evaluation_cadence_minimum,
                init_seed=init_seed,
                train_encoder=train_encoder,
            )
            trained_costs, residual_pairs, spread_samples = _evaluate_arm_over_seeds(
                world, training.policy_state, seeds
            )
            result = ArmResult(
                arm=arm,
                init_seed=init_seed,
                trained_seed_costs=trained_costs,
                untrained_seed_costs=untrained_costs,
                residual_calibration=residual_pairs,
                spread_samples=spread_samples,
                episodes_completed=training.episodes_completed,
                converged=training.converged,
            )
            arm_results.append(result)
            emit(
                f"Gate A' arm={arm} init_seed={init_seed}: {training.episodes_completed} episodes "
                f"({'converged' if training.converged else 'DID NOT CONVERGE'}), "
                f"mean reduction {result.mean_reduction_pct:+.1f}% "
                f"(median {result.median_reduction_pct:+.1f}%), "
                f"Wilcoxon p={result.wilcoxon_p:.4g}, "
                f"calibration rho={result.calibration_spearman:.3f}, "
                f"r={result.candidate_spread_ratio:.3f}"
            )
        summaries[arm] = ArmSummary(arm=arm, results=tuple(arm_results))

    return GateAPrimeResult(frozen=summaries["frozen"], trained=summaries["trained"])


def _format_arm(summary: ArmSummary) -> list[str]:
    lines = [f"Arm: {summary.arm}", "-" * 60]
    for result in summary.results:
        lines.append(
            f"  init_seed={result.init_seed}: {result.episodes_completed} episodes "
            f"({'converged' if result.converged else 'DID NOT CONVERGE'})"
        )
        lines.append(
            f"    reduction: mean {result.mean_reduction_pct:+.2f}%  "
            f"median {result.median_reduction_pct:+.2f}%  "
            f"Wilcoxon p={result.wilcoxon_p:.4g}  "
            f"{'PASS' if result.null_model_passes else 'FAIL'}"
        )
        lines.append(
            f"    calibration: rho(W.phi, y~)={result.calibration_spearman:+.3f}  "
            f"{'PASS' if result.calibration_passes else 'FAIL'}   "
            f"r=sd(W.phi)/sd(c)={result.candidate_spread_ratio:.3f}"
        )
    lines.append(
        f"  1. Null model:      {'PASS' if summary.null_model_passes else 'FAIL'} "
        f"(>= {GATE_A_PRIME_MIN_INIT_SEEDS} seeds, each p<{GATE_A_PRIME_SIGNIFICANCE_THRESHOLD} "
        f"and >= {GATE_A_PRIME_EFFECT_THRESHOLD_PCT:.0f}% reduction)"
    )
    spread_note = (
        "straddles zero" if summary.reproducibility_spread_straddles_zero else "consistent sign"
    )
    lines.append(
        f"  2. Reproducibility: {'PASS' if summary.reproducibility_passes else 'FAIL'} "
        f"(mean {summary.reproducibility_mean_pct:+.2f}% "
        f"+/- sd {summary.reproducibility_sd_pct:.2f}%, {spread_note})"
    )
    lines.append(
        f"  3. Calibration:     {'PASS' if summary.calibration_passes else 'FAIL'} "
        f"(>= {GATE_A_PRIME_CALIBRATION_THRESHOLD} required)"
    )
    lines.append(f"  mean r across seeds: {summary.mean_candidate_spread_ratio:.3f}")
    lines.append(f"  ARM {summary.arm.upper()}: {'PASS' if summary.passes else 'FAIL'}")
    return lines


def format_gate_a_prime_report(result: GateAPrimeResult) -> str:
    """The three-part report per arm, plus the decomposition ticket 17's
    Acceptance requires stated explicitly (never left to be inferred)."""
    n_seeds = len(result.frozen.results[0].untrained_seed_costs)
    lines = [
        "Gate A' -- does training add anything on top of the base?",
        "=" * 60,
        f"null: the myopic base c(s, a) (ticket 15) -- Q == c at W == 0, the same "
        "network regardless of arm or init seed. Not a warm start: "
        "neural_warm_start has been dead/inert since ticket 15 (c is always the "
        f"'cost' formula). mean cost over {n_seeds} test_seeds: {result.null_mean_cost:.4f}",
        "=" * 60,
    ]
    lines.extend(_format_arm(result.frozen))
    lines.append("=" * 60)
    lines.extend(_format_arm(result.trained))
    lines.append("=" * 60)
    lines.append(
        f"delta(frozen)  vs. the myopic base  = {result.frozen.reproducibility_mean_pct:+.2f}%  "
        "(the value of the cost features + the estimator)"
    )
    lines.append(
        f"delta(trained) vs. the myopic base  = {result.trained.reproducibility_mean_pct:+.2f}%"
    )
    lines.append(
        f"delta(trained) - delta(frozen)      = {result.representation_learning_delta_pct:+.2f}%  "
        "(the value of representation learning)"
    )
    lines.append(
        "named for context (not gated by Gate A' -- Gate B's own question, tickets 09/10): "
        f"linear baseline best cell {BASELINE_BEST_CELL_COST:.2f}, "
        f"like-for-like (m+2) {BASELINE_LIKE_FOR_LIKE_COST:.2f}"
    )
    lines.append("=" * 60)
    lines.append(f"GATE A': {'PASS' if result.passes else 'FAIL'}")
    return "\n".join(lines)
