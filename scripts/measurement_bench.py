"""Measurement bench: the review's numbers as a rerunnable command (ticket 01).

The simulator-correctness effort's instrumentation foundation
(``.scratch/simulator-correctness/spec.md``). Runs a fixed 60-seed set
(``BENCH_SEEDS = range(60)``, the review's own N) on the committed mini fixture
and reports, per seed and aggregated: mean total cost and its 4 components, km
driven, final tau, decision count, unserved-Client count split by
in-window/overdue, and a violation counter per invariant in the spec's
catalogue (``spec.md``, "The invariant catalogue"). Output is plain,
deterministically ordered text — diffable with ``diff``, not prose — so
tickets 02-09 can save a "before" and an "after" run and diff them directly.

**Decision-stable by construction.** Episodes run greedy evaluation with a
*fixed* W (``--w zero``, the default, or ``--w frozen`` for the literal W
``scripts/capture_self_golden.py`` freezes) rather than a freshly trained one,
for the same reason the self-golden's frozen-W block exists (spec, "Behavior
contract"): a fix's cost change must not itself move the decisions being
measured. ``--vehicle-count`` controls fleet size; ``None`` (default) uses the
demand-generated fleet, matching how the review measured B11 (duplicate
assignment needs >=2 vehicles); B1b and B3's headline numbers were measured by
the review on forced single-vehicle fleets (``--vehicle-count 1``).

**Capture-seeds mode** (``--capture-seeds``) replays the exact frozen-W capture
protocol from ``capture_self_golden.py`` — ``TRAIN_SEEDS`` carrying W,
``EVAL_SEEDS`` with the final trained W, generated fleet sizes — instrumented
the same way, so tickets 03/04/05 can read the per-capture-seed unserved,
teleport and duplicate-assignment counts their predicted diffs depend on
(ticket 01's ``## Comments``).

Usage (writes to stdout; redirect or use ``--output`` to save a snapshot)::

    uv run python scripts/measurement_bench.py
    uv run python scripts/measurement_bench.py --vehicle-count 1
    uv run python scripts/measurement_bench.py --capture-seeds
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

import stdvrp.simulation.episode as episode_module
from stdvrp.congestion import ArcProbabilityCongestionGenerator, CongestedArcs, CongestionGenerator
from stdvrp.simulation import run_evaluation_episode, run_training_episode
from stdvrp.simulation.fleet_routes import PARKED
from stdvrp.simulation.model import EMERGENCY_HORIZON, Model

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The fixed 60-seed set every before/after comparison in this effort uses
#: (spec.md decision 11: "the review's own N, so the numbers stay comparable
#: to the report").
BENCH_SEEDS: tuple[int, ...] = tuple(range(60))


def _load_module(name: str, relative_path: str) -> ModuleType:
    """Load a sibling ``scripts/`` module by file path (mirrors capture_self_golden.py)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


benchmark = _load_module("benchmark_episodes", "scripts/benchmark_episodes.py")
capture_self_golden = _load_module("capture_self_golden", "scripts/capture_self_golden.py")

# The literal frozen-W capture uses (the current final_w) — kept in sync by
# ticket 01's frozen-W addition to capture_self_golden.py, not duplicated here.
FROZEN_W: tuple[float, ...] = capture_self_golden.FROZEN_W


# --- Probe instrumentation: recording subclasses, behavior-identical to the originals --


class BenchCongestionGenerator(CongestionGenerator):
    """Wraps the real generator; counts B7 — a write that shortens an active event.

    "Active" means the prior entry's expiry is still ahead of ``minute_start``;
    "shortens" means the new entry's expiry is earlier than that. The compose
    fix (ticket 07) keeps the more severe multiplier and the later expiry, so
    this counts writes that fail that rule.
    """

    def __init__(self, inner: CongestionGenerator) -> None:
        self.inner = inner
        self.shorten_violations = 0

    def generate(
        self, minute_start: float, congested_arcs: CongestedArcs, rng: np.random.Generator
    ) -> None:
        before = dict(congested_arcs)
        self.inner.generate(minute_start, congested_arcs, rng)
        for arc, value in congested_arcs.items():
            prior = before.get(arc)
            if prior is None or prior is value:
                continue
            _prior_multiplier, prior_expiry = prior
            if prior_expiry > minute_start and value[1] < prior_expiry:
                self.shorten_violations += 1


class BenchModel(Model):
    """Model instrumenting B1a/B1b, B10, B11, B17 (ticket 01's violation counters).

    A wrapper in the spirit of ``tests/test_invariants.py``'s ``RecordingModel``:
    every override calls straight through to the real behavior and only adds
    counting, so the Episode this runs is bit-identical to an uninstrumented one.
    """

    last_instance: BenchModel | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.mid_arc_park_violations = 0
        self._counted_mid_arc_park: set[int] = set()
        self.duplicate_assignment_transitions = 0
        self.earliness_bin_violations = 0
        self.congestion_book_size_at_end = 0
        self.congestion_book_expired_at_end = 0
        super().__init__(*args, **kwargs)
        # Capture the congestion book just before it is released (both Episode
        # runners call ``velocities.release()`` at termination, clearing it) —
        # B17's "book holds no expired entries" is checked at exactly that point.
        self._real_release = self.velocities.release
        self.velocities.release = self._capture_book_then_release  # type: ignore[method-assign]
        BenchModel.last_instance = self

    def _capture_book_then_release(self) -> None:
        """A single end-of-episode snapshot, not a continuous check.

        B17's catalogue row is "the book holds no expired entries **after a
        clock advance**" — every advance, not only the last one. This samples
        only the terminal state (cheap, and exactly what the ticket's bullet 4
        asks for: "the book reaches N arcs, all N expired"). It is real evidence
        that the book carries stale entries this far, not a substitute for the
        per-advance property ticket 07 needs to write once it composes and
        purges.
        """
        book = self.velocities.congested_arcs
        tau = self.state.tau_episode
        self.congestion_book_size_at_end = len(book)
        self.congestion_book_expired_at_end = sum(
            1 for _mult, expiry in book.values() if tau >= expiry
        )
        self._real_release()

    def _reroute_for(self, action: list[int]) -> None:
        """B1a/B1b: a vehicle mid-arc, sent to the depot, that ends up ``PARKED``.

        One counter standing in for **two** catalogue rows — "no vehicle becomes
        ``PARKED`` while travelling" and "a vehicle's recorded node never changes
        without charged distance (no teleporting)". The review measured these as
        the same event, never independent (``docs/simulator-review.md``, B1b:
        "comparten disparador y en las mediciones nunca aparecen por separado" —
        of 71 mid-arc retirements, all 71 came from the one depot-idle trigger).
        A count here is evidence for both rows; it is not two independent
        measurements. Ticket 04 (ADR-0005) gave ``State`` the fact this bench
        used to reach into ``fleet`` internals for (``state.vehicle_standing``);
        both this probe and ``tests/test_invariants.py``'s dedicated property
        now agree it stays at zero.

        Mirrors the review's own reproduction probe (``docs/simulator-review.md``,
        "Reproducción"): watch vehicles that, before rerouting, are travelling
        (``departure_tau < tau < arrival_tau``) with ``last_node_reached == depot``
        and ``action == depot``; count the ones ``PARKED`` afterward.

        A wrongly-parked vehicle never revives (``is_travelling`` is permanently
        ``False`` — the review's own finding), and the Policy keeps re-offering
        it the depot every later transition, so this branch fires again on every
        subsequent transition as a harmless no-op (``PARKED`` set to ``PARKED``
        again). Without ``_counted``, that no-op would be recounted as a fresh
        violation every transition for the rest of the Episode — one real
        retirement inflated into dozens. Each vehicle counts at most once.
        """
        fleet = self.fleet
        tau = self.state.tau_episode
        watch = [
            v
            for v in range(len(action))
            if v not in self._counted_mid_arc_park
            and action[v] == self.depot
            and self.state.last_node_reached[v] == self.depot
            and fleet.is_travelling(v)
            and fleet.departure_tau[v] < tau < fleet.arrival_tau[v]
        ]
        super()._reroute_for(action)
        for v in watch:
            if fleet.arrival_tau[v] == PARKED:
                self.mid_arc_park_violations += 1
                self._counted_mid_arc_park.add(v)

    def transition_function(self, action: list[int]) -> float:
        """B11 (duplicate non-depot assignment) and B10 (earliness-bin partition)."""
        non_depot = [target for target in action if target != self.depot]
        if len(non_depot) != len(set(non_depot)):
            self.duplicate_assignment_transitions += 1
        self._check_earliness_partition()
        return super().transition_function(action)

    def _check_earliness_partition(self) -> None:
        """B10: the four earliness bins (``general[7:11]``) should sum to pending demand.

        Re-derives what the Policy is about to compute for this decision
        (read-only, no side effect) rather than hooking ``FeatureExtractor``
        directly — that class has no seam (ADR-0002) and this Model has no
        business reaching into the Policy's internals for anything but a count.
        """
        extractor = getattr(self.policy, "feature_extractor", None)
        if extractor is None:
            return
        remaining_count = len(self.state.clients_not_visited)
        if remaining_count == 0:
            return
        features = extractor.state_features(self.state)
        total_binned = round(float(features.general[7:11].sum()) * extractor._number_clients)
        if total_binned != remaining_count:
            self.earliness_bin_violations += 1


# --- Per-seed measurement -----------------------------------------------------


@dataclass
class SeedMeasurement:
    """Everything the bench reports for one Episode."""

    seed: int
    crashed: bool = False
    total_cost: float = float("nan")
    distance_cost: float = float("nan")
    delay_cost: float = float("nan")
    earliness_cost: float = float("nan")
    overtime_cost: float = float("nan")
    km_driven: float = float("nan")
    final_tau: float = float("nan")
    decisions: int = 0
    unserved_in_window: int = 0
    unserved_overdue: int = 0
    mid_arc_park_violations: int = 0
    duplicate_assignment_transitions: int = 0
    earliness_bin_violations: int = 0
    congestion_shorten_violations: int = 0
    congestion_book_size_at_end: int = 0
    congestion_book_expired_at_end: int = 0
    negative_cost_component_violations: int = 0
    money_without_counter_violations: int = 0


FIELDS: tuple[str, ...] = tuple(SeedMeasurement.__dataclass_fields__)


def _money_without_counter_violations(costs: Any) -> int:
    """B14: money charged > 0 must imply its counter > 0, and vice versa.

    Ticket 03 gave the termination delay charge its own counter
    (``unserved_clients``, never folded into ``late_clients`` — see
    ``CostLedger``), so ``delay_cost`` now agrees with *either* counter being
    positive, not just ``late_clients``.
    """
    violations = 0
    if (costs.delay_cost > 0) != (costs.late_clients > 0 or costs.unserved_clients > 0):
        violations += 1
    if (costs.overtime_cost > 0) != (costs.overtime_vehicles > 0):
        violations += 1
    if (costs.earliness_cost > 0) != (costs.early_clients > 0):
        violations += 1
    return violations


def _measure_model(seed: int, model: Model) -> SeedMeasurement:
    """Read every counter off a finished ``BenchModel`` instance."""
    assert isinstance(model, BenchModel)
    state = model.state
    costs = model.costs
    tau = state.tau_episode
    unserved = state.clients_not_visited
    overdue = sum(1 for client in unserved if tau > model.time_windows[client][1])  # type: ignore[index]
    components = (costs.distance_cost, costs.delay_cost, costs.earliness_cost, costs.overtime_cost)
    negative_components = sum(1 for value in components if value < 0)
    return SeedMeasurement(
        seed=seed,
        crashed=False,
        total_cost=costs.total_cost,
        distance_cost=costs.distance_cost,
        delay_cost=costs.delay_cost,
        earliness_cost=costs.earliness_cost,
        overtime_cost=costs.overtime_cost,
        km_driven=sum(state.total_vehicle_distance_travelled.values()),
        final_tau=tau,
        decisions=model.total_state_counter,
        unserved_in_window=len(unserved) - overdue,
        unserved_overdue=overdue,
        mid_arc_park_violations=model.mid_arc_park_violations,
        duplicate_assignment_transitions=model.duplicate_assignment_transitions,
        earliness_bin_violations=model.earliness_bin_violations,
        congestion_shorten_violations=0,  # filled by the caller (owns the generator wrapper)
        congestion_book_size_at_end=model.congestion_book_size_at_end,
        congestion_book_expired_at_end=model.congestion_book_expired_at_end,
        negative_cost_component_violations=negative_components,
        money_without_counter_violations=_money_without_counter_violations(costs),
    )


def _run_one(
    world: Any,
    seed: int,
    *,
    vehicle_count: int | None,
    w: np.ndarray | None,
    train: bool,
    learning_rate: float = 0.0,
) -> tuple[SeedMeasurement, np.ndarray | None]:
    """Run one instrumented Episode; return its measurement and (if training) the new W.

    Patches ``episode_module.Model`` to :class:`BenchModel` for the duration of
    the call — the same technique ``tests/test_invariants.py`` uses — so
    ``run_evaluation_episode`` / ``run_training_episode`` need no changes.
    """
    kwargs = dict(world.episode_kwargs())
    generator = BenchCongestionGenerator(kwargs["congestion_generator"])
    kwargs["congestion_generator"] = generator

    original_model = episode_module.Model
    episode_module.Model = BenchModel  # type: ignore[misc]
    try:
        try:
            if train:
                result = run_training_episode(
                    seed=seed, W=w, learning_rate=learning_rate, depot=0, **kwargs
                )
                new_w: np.ndarray | None = result.w
            else:
                if vehicle_count is not None:
                    kwargs["vehicle_count"] = vehicle_count
                run_evaluation_episode(seed=seed, W=w, **kwargs)
                new_w = w
        except ValueError:
            return SeedMeasurement(seed=seed, crashed=True), w
    finally:
        episode_module.Model = original_model

    model = BenchModel.last_instance
    assert model is not None
    measurement = _measure_model(seed, model)
    measurement.congestion_shorten_violations = generator.shorten_violations
    return measurement, new_w


def run_bench(
    world: Any, seeds: tuple[int, ...], *, vehicle_count: int | None, w: np.ndarray | None
) -> list[SeedMeasurement]:
    """The reusable 60-seed bench: every seed evaluated once, greedily, with fixed W."""
    measurements = []
    for seed in seeds:
        measurement, _ = _run_one(world, seed, vehicle_count=vehicle_count, w=w, train=False)
        measurements.append(measurement)
    return measurements


def run_capture_seeds(world: Any) -> list[SeedMeasurement]:
    """Bullet 5: the exact frozen-W capture protocol, instrumented.

    Mirrors ``capture_self_golden.run_protocol`` (training carries W across
    ``TRAIN_SEEDS``, evaluation runs ``EVAL_SEEDS`` with the final W) so the
    per-seed counts line up with the 15 Episodes the self-golden capture pins.
    """
    config = world.config
    learning_rate = (
        config.warmup_learning_rate
        if config.warmup_learning_rate is not None
        else config.learning_rate
    )
    w: np.ndarray | None = None
    measurements = []
    for seed in capture_self_golden.TRAIN_SEEDS:
        measurement, w = _run_one(
            world, seed, vehicle_count=None, w=w, train=True, learning_rate=learning_rate
        )
        learning_rate = config.learning_rate
        measurements.append(measurement)

    for seed in capture_self_golden.EVAL_SEEDS:
        measurement, _ = _run_one(world, seed, vehicle_count=None, w=w, train=False)
        measurements.append(measurement)
    return measurements


# --- Structural checks: independent of any Episode (B9, B16) -----------------


def _reference_bfs_depths(
    successors: dict[int, list[int]], start: int, max_depth: int
) -> dict[int, int]:
    """True minimum-depth BFS, for comparison against the DFS-by-recursion spread (B9)."""
    depths = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        if depths[node] >= max_depth:
            continue
        for neighbor in successors.get(node, []):
            if neighbor not in depths:
                depths[neighbor] = depths[node] + 1
                queue.append(neighbor)
    return depths


@dataclass
class B9Check:
    """B9's two distinct symptoms: nodes reached at the wrong depth, and nodes missed entirely.

    A node the recursion discovers first via a long path keeps that depth (wrong
    damping factor) and, if that depth is already ``max_depth``, stops expanding
    from it — which can leave a node genuinely within ``max_depth`` (via a
    shorter path the recursion never takes because the intermediate node is
    already ``visited``) unreached altogether. The review's headline "~15% never
    congested" is the second symptom, not the first.
    """

    wrong_depth: int
    reached_total: int
    missed_entirely: int
    genuinely_reachable_total: int


def check_b9_spread_depths(successors: dict[int, list[int]], max_depth: int = 3) -> B9Check:
    """B9: DFS-by-recursion vs. true-BFS reachability and depth, over every trigger node."""
    generator = ArcProbabilityCongestionGenerator(
        event_probability={},
        successors=successors,
        congestion_lower_bound=0.3,
        congestion_upper_bound=0.4,
        max_congestion_duration=120,
        max_depth=max_depth,
    )
    wrong_depth = 0
    reached_total = 0
    missed_entirely = 0
    genuinely_reachable_total = 0
    for node in successors:
        reached, dfs_depth = generator._reachable_nodes(node, 0, max_depth)
        bfs_depth = _reference_bfs_depths(successors, node, max_depth)
        reached_total += len(reached)
        genuinely_reachable_total += len(bfs_depth)
        for other in reached:
            if dfs_depth[other] != bfs_depth.get(other):
                wrong_depth += 1
        missed_entirely += len(set(bfs_depth) - reached)
    return B9Check(wrong_depth, reached_total, missed_entirely, genuinely_reachable_total)


#: Durations the review names as tested/shipped (fine) plus the two it flags as
#: breaking (50, 70) — B16.
B16_DURATIONS: tuple[int, ...] = (30, 45, 50, 60, 70, 90, 120, 150, 180, 200, 240)


def check_b16_cadence(
    horizon_start: int = 300,
    episode_end: int = EMERGENCY_HORIZON,
    durations: tuple[int, ...] = B16_DURATIONS,
) -> dict[int, tuple[int, int]]:
    """B16: the shipped float-modulo cadence vs. the integer-arithmetic intended one.

    ``tau`` at every ``_congestion_epoch_due`` check is ``next_decision_tau``,
    which starts at ``horizon_start_minute + 2`` (both ints) and is incremented
    by the int ``DECISION_EPOCH_MINUTES`` — so it is always integer-valued, which
    is what makes an integer-arithmetic reference formula meaningful here (see
    ticket 02's note to verify this before asserting it).

    Returns ``{duration: (fires_as_shipped, fires_intended)}``.
    """
    taus = range(horizon_start + 2, episode_end, 2)
    results = {}
    for duration in durations:
        hours_max_duration = duration / 60
        fires_shipped = sum(1 for tau in taus if (tau + 180 - 2) / 60 % hours_max_duration == 0)
        fires_intended = sum(1 for tau in taus if (tau + 178) % duration == 0)
        results[duration] = (fires_shipped, fires_intended)
    return results


# --- Reporting: plain, diffable text ------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}" if value == value else "nan"  # nan != nan
    return str(value)


def format_report(
    measurements: list[SeedMeasurement],
    *,
    title: str,
    successors: dict[int, list[int]] | None = None,
) -> str:
    lines = [f"# {title}", f"# n_seeds={len(measurements)}", ""]

    lines.append("\t".join(FIELDS))
    for m in measurements:
        lines.append("\t".join(_fmt(getattr(m, field_name)) for field_name in FIELDS))
    lines.append("")

    crashed = [m for m in measurements if m.crashed]
    live = [m for m in measurements if not m.crashed]
    n = len(measurements)

    lines.append(f"# aggregate (N={n}, crashed={len(crashed)})")
    if crashed:
        lines.append("crashed_seeds\t" + ",".join(str(m.seed) for m in crashed))
    if live:
        for name in (
            "total_cost",
            "distance_cost",
            "delay_cost",
            "earliness_cost",
            "overtime_cost",
            "km_driven",
            "final_tau",
            "decisions",
        ):
            values = [getattr(m, name) for m in live]
            lines.append(f"mean_{name}\t{sum(values) / len(values):.6f}")

        def episodes_with(predicate: Any) -> int:
            return sum(1 for m in live if predicate(m) > 0)

        def total_of(name: str) -> int:
            return sum(getattr(m, name) for m in live)

        lines.append(
            f"unserved_in_window\tepisodes={episodes_with(lambda m: m.unserved_in_window)}"
            f"/{len(live)}\tclients={total_of('unserved_in_window')}"
        )
        lines.append(
            f"unserved_overdue\tepisodes={episodes_with(lambda m: m.unserved_overdue)}/{len(live)}"
            f"\tclients={total_of('unserved_overdue')}"
        )
        lines.append(
            "B1a_B1b_mid_arc_park\tepisodes="
            f"{episodes_with(lambda m: m.mid_arc_park_violations)}/{len(live)}"
            f"\tevents={total_of('mid_arc_park_violations')}"
            f"\tper_episode={total_of('mid_arc_park_violations') / len(live):.3f}"
        )
        lines.append(
            "B11_duplicate_assignment\tepisodes="
            f"{episodes_with(lambda m: m.duplicate_assignment_transitions)}/{len(live)}"
            f"\ttransitions={total_of('duplicate_assignment_transitions')}"
        )
        lines.append(
            "B10_earliness_bin_partition\tepisodes="
            f"{episodes_with(lambda m: m.earliness_bin_violations)}/{len(live)}"
            f"\tdecisions={total_of('earliness_bin_violations')}"
        )
        lines.append(
            "B7_congestion_shorten\tepisodes="
            f"{episodes_with(lambda m: m.congestion_shorten_violations)}/{len(live)}"
            f"\twrites={total_of('congestion_shorten_violations')}"
        )
        lines.append(
            "B17_book_purge\tfinal_book_size_total="
            f"{total_of('congestion_book_size_at_end')}"
            f"\texpired_at_end_total={total_of('congestion_book_expired_at_end')}"
        )
        lines.append(
            "B14_money_without_counter\tepisodes="
            f"{episodes_with(lambda m: m.money_without_counter_violations)}/{len(live)}"
            f"\tviolations={total_of('money_without_counter_violations')}"
        )
        lines.append(
            "B15_negative_cost_component\tepisodes="
            f"{episodes_with(lambda m: m.negative_cost_component_violations)}/{len(live)}"
            f"\tviolations={total_of('negative_cost_component_violations')}"
        )
        lines.append(
            "B3_termination_charge_clock_independence\tNOT_DIRECTLY_COUNTABLE"
            "\tsee unserved_in_window above (that demand is what B3 prices at zero pre-fix)"
        )
    lines.append("")

    if successors is not None:
        lines.append("# structural checks (independent of seed)")
        b9 = check_b9_spread_depths(successors)
        wrong_pct = 100 * b9.wrong_depth / b9.reached_total if b9.reached_total else 0.0
        missed_pct = (
            100 * b9.missed_entirely / b9.genuinely_reachable_total
            if b9.genuinely_reachable_total
            else 0.0
        )
        lines.append(
            f"B9_spread_wrong_depth\tcount={b9.wrong_depth}\treached_total={b9.reached_total}"
            f"\tpct={wrong_pct:.2f}"
        )
        lines.append(
            f"B9_spread_missed_entirely\tcount={b9.missed_entirely}"
            f"\tgenuinely_reachable_total={b9.genuinely_reachable_total}\tpct={missed_pct:.2f}"
        )
        for duration, (shipped, intended) in check_b16_cadence().items():
            flag = "match" if shipped == intended else "MISMATCH"
            lines.append(
                f"B16_cadence_duration_{duration}\tfires_shipped={shipped}"
                f"\tfires_intended={intended}\t{flag}"
            )
        lines.append("")

    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------------


def _load_world() -> Any:
    with tempfile.TemporaryDirectory(prefix="measurement_bench_world_") as tmp:
        data_dir = benchmark.build_fixture_data_dir(Path(tmp))
        config = benchmark.fixture_config(data_dir)
        world, _ = benchmark.load_world(config)
        # world_cache holds no reference to tmp after load; the CSVs are parsed
        # into arrays, not memory-mapped, so it is safe to let tmp be removed.
    return world


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=None,
        help="force a fleet size (default: demand-generated, matching the review's B11 numbers)",
    )
    parser.add_argument(
        "--w",
        choices=("zero", "frozen"),
        default="zero",
        help="fixed W the bench evaluates with: the zero vector (default, matches how the "
        "review's own probes ran) or the literal frozen W (scripts/capture_self_golden.py)",
    )
    parser.add_argument(
        "--capture-seeds",
        action="store_true",
        help="run the frozen-W capture protocol's 15 seeds instead of the 60-seed bench "
        "(ticket 01 bullet 5: sizes tickets 03/04/05's blast radius)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write the report here instead of stdout"
    )
    args = parser.parse_args(argv)

    world = _load_world()
    w = None if args.w == "zero" else np.array(FROZEN_W)

    if args.capture_seeds:
        measurements = run_capture_seeds(world)
        title = "capture-seeds bench (frozen-W protocol: 5 train + 10 eval)"
    else:
        measurements = run_bench(world, BENCH_SEEDS, vehicle_count=args.vehicle_count, w=w)
        title = f"60-seed bench (vehicle_count={args.vehicle_count}, w={args.w})"

    report = format_report(measurements, title=title, successors=world.travel_time_model.successors)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"written: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
