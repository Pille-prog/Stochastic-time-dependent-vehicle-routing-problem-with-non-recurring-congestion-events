"""Ticket 13 statistical regression: the new package's costs stay near the
pre-migration baseline once RNG modernization changes the exact numbers.

Ticket 12 pinned the new package's behavior bit-for-bit against
``chengdu_full_phase2.json`` (full Chengdu data, protocol captured from the
legacy run). Ticket 13 (RNG modernization, ADR-0001 phase 2) replaced every
global ``random``/``np.random`` consumption with injected per-concern
``np.random.Generator`` streams (PCG64) — exact bit-for-bit equality with that
baseline is neither expected nor achievable (different bit-generator algorithm,
different draw order), exactly as ADR-0001 anticipated. This file is the
retirement ADR-0001 called for: ``chengdu_full_phase2.json`` is repurposed as
the **pre-migration statistical baseline** (it is, after all, the last capture
of the new package before ticket 13 touched a single RNG call site), and the
exact-equality assertions are replaced with mean-cost-over-N-seeds tolerance
checks. ``tests/test_golden_master.py`` (legacy monolith vs its own frozen
capture) is unaffected — it never imports ``stdvrp`` and was already
legacy-only documentation before this ticket.

Pre-registered tolerance: 40% relative to the baseline mean. The baseline's own
per-episode costs have a ~40% coefficient of variation across both the 10 eval
seeds and the 10 test seeds (computed from ``chengdu_full_phase2.json`` itself),
so a single new sample of N draws from the same underlying distributions can
land 40% away from the old sample's mean by chance alone; this band is sized to
absorb that resampling noise while still catching a genuine regression (a sign
error, a swapped bound, a broken stream) at the confirmed ADR-0001 fix
magnitudes (fix 7 alone shifted a single-episode cost 2342.5 -> 4016.6, ~70%).

Skips when the local dataset is absent (e.g. CI); building the world through the
new package re-reads the 88 speed files and the 907 MB path cache (~15 minutes).
"""

import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stdvrp.config import ExperimentConfig
from stdvrp.congestion import ArcProbabilityCongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.simulation import run_evaluation_episode
from stdvrp.traffic import CsvDataSource, TravelTimeModel
from stdvrp.training import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_master" / "chengdu_full_phase2.json"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))

# Relative tolerance for mean-cost-over-N-seeds comparisons (see module docstring).
RELATIVE_TOLERANCE = 0.40

spec = importlib.util.spec_from_file_location(
    "capture_golden_master", REPO_ROOT / "scripts" / "capture_golden_master.py"
)
assert spec is not None and spec.loader is not None
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    if not GOLDEN_PATH.exists():
        pytest.skip("pre-migration baseline not generated (scripts/rebaseline_golden_master.py)")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data_dir() -> Path:
    directory = capture.default_data_dir()
    missing = capture.missing_data_files(directory)
    if missing:
        pytest.skip(
            f"full Chengdu dataset not available under {directory} "
            f"({len(missing)} files missing, first: {missing[0]})"
        )
    return directory


@pytest.fixture(scope="module")
def world(golden: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    protocol = golden["protocol"]
    source = CsvDataSource(data_dir, "link.csv", 601, LEGACY_DAYS, "all_shortest_paths.csv")
    travel_time_model = TravelTimeModel(
        source.load_road_network(),
        source.load_traffic_history(),
        protocol["max_congestion_duration"],
        horizon_start_minute=protocol["horizon_start_time"],
    )
    return {
        "travel_time_model": travel_time_model,
        "cache": ShortestPathCache.from_csv(data_dir / "all_shortest_paths.csv"),
        # The legacy ClientGenerator hardcodes: gauss stddev 30, 60-client floor,
        # universe range(1, 1900), and 28 clients per vehicle at 150 mean clients.
        "client_generator": ClientGenerator(
            mean_number_clients=protocol["mean_number_clients"],
            client_count_stddev=30.0,
            min_number_clients=60,
            client_universe_node_range=(1, 1900),
            clients_per_vehicle=28,
            time_window_spread=protocol["diff_TW"],
            horizon_start_minute=protocol["horizon_start_time"],
            horizon_end_minute=protocol["horizon_end_time"],
        ),
        "congestion_generator": ArcProbabilityCongestionGenerator(
            event_probability=travel_time_model.event_probability,
            successors=travel_time_model.successors,
            congestion_lower_bound=protocol["congestion_lower_bound"],
            congestion_upper_bound=protocol["congestion_upper_bound"],
            max_congestion_duration=protocol["max_congestion_duration"],
        ),
    }


def _relative_diff(actual: float, expected: float) -> float:
    return abs(actual - expected) / expected


def config_from_protocol(protocol: dict[str, Any], data_dir: Path) -> ExperimentConfig:
    """The stored capture protocol expressed as one ExperimentConfig (ticket 09).

    The Trainer derives its seed sequences from start + count, so the protocol's
    explicit lists must be contiguous ranges — asserted here rather than assumed.
    """
    train_seeds = protocol["train_seeds"]
    eval_seeds = protocol["eval_seeds"]
    assert train_seeds == list(range(train_seeds[0], train_seeds[0] + len(train_seeds)))
    assert eval_seeds == list(range(eval_seeds[0], eval_seeds[0] + len(eval_seeds)))
    return ExperimentConfig(
        data_dir=data_dir,
        links_file="link.csv",
        shortest_paths_file="all_shortest_paths.csv",
        instance_day=601,
        traffic_days=LEGACY_DAYS,
        horizon_start_minute=protocol["horizon_start_time"],
        horizon_end_minute=protocol["horizon_end_time"],
        # The legacy ClientGenerator hardcodes (see the ``world`` fixture).
        mean_number_clients=protocol["mean_number_clients"],
        client_count_stddev=30.0,
        min_number_clients=60,
        clients_per_vehicle=28,
        time_window_spread=protocol["diff_TW"],
        client_universe_seed=0,
        client_universe_size=150,
        client_universe_node_range=(1, 1900),
        congestion_lower_bound=protocol["congestion_lower_bound"],
        congestion_upper_bound=protocol["congestion_upper_bound"],
        max_congestion_duration=protocol["max_congestion_duration"],
        total_train_iterations=len(train_seeds),
        # One evaluation block right after the last training episode, as captured.
        test_frequency=len(train_seeds),
        learning_rate=protocol["learning_rate"],
        warmup_learning_rate=protocol["warmup_learning_rate"],
        epsilon=protocol["epsilon"],
        n_observed_arcs=protocol["n_arcs"],
        first_train_seed=train_seeds[0],
        evaluation_seed_start=eval_seeds[0],
        evaluation_seed_count=len(eval_seeds),
        test_episodes=1,
        test_action_counts=tuple(protocol["test_actions"]),
        test_seeds=tuple(protocol["test_seeds"]),
        test_vehicle_counts=tuple(protocol["test_vehicles"]),
        static_policy_mean_cost=None,
    )


def test_trainer_run_produces_finite_costs_near_the_baseline(
    golden: dict[str, Any], data_dir: Path, world: dict[str, Any], tmp_path: Path
) -> None:
    """Ticket 09 wiring survives the RNG migration: Trainer.run() end to end."""
    protocol = golden["protocol"]
    config = config_from_protocol(protocol, data_dir)
    trainer = Trainer(
        config,
        client_generator=world["client_generator"],
        travel_time_model=world["travel_time_model"],
        shortest_path_cache=world["cache"],
        congestion_generator=world["congestion_generator"],
    )
    result = trainer.run(tmp_path / "run")

    assert all(np.isfinite(w).all() for w in result.training.w_trajectory)
    # Training moved the weights off the initial zero vector.
    assert any(component != 0 for component in result.training.w_trajectory[-1])
    assert [block.episodes_completed for block in result.training.evaluations] == [
        len(protocol["train_seeds"])
    ]
    assert result.training.best_w is not None

    baseline_mean = statistics.mean(golden["training"]["eval_costs"])
    produced_mean = result.training.evaluations[0].mean_cost
    assert _relative_diff(produced_mean, baseline_mean) <= RELATIVE_TOLERANCE, (
        f"evaluation mean cost {produced_mean:.1f} vs baseline {baseline_mean:.1f} "
        f"({_relative_diff(produced_mean, baseline_mean):.0%} > {RELATIVE_TOLERANCE:.0%})"
    )
    assert (tmp_path / "run" / "results.json").is_file()
    assert (tmp_path / "run" / "training_plot.png").stat().st_size > 0


def test_evaluation_mean_cost_is_within_tolerance_of_the_baseline(
    golden: dict[str, Any], world: dict[str, Any]
) -> None:
    """Mean episode cost over the captured eval seeds, same W, new RNG streams."""
    protocol = golden["protocol"]
    best_w = np.array(golden["training"]["w_trajectory"][-1])

    produced_costs = [
        run_evaluation_episode(
            seed=seed,
            client_generator=world["client_generator"],
            travel_time_model=world["travel_time_model"],
            shortest_path_cache=world["cache"],
            congestion_generator=world["congestion_generator"],
            W=best_w,
            epsilon=protocol["epsilon"],
            max_congestion_duration=protocol["max_congestion_duration"],
            horizon_start_minute=protocol["horizon_start_time"],
            horizon_end_minute=protocol["horizon_end_time"],
            n_observed_arcs=protocol["n_arcs"],
        ).total_cost
        for seed in protocol["eval_seeds"]
    ]

    baseline_mean = statistics.mean(golden["training"]["eval_costs"])
    produced_mean = statistics.mean(produced_costs)
    assert _relative_diff(produced_mean, baseline_mean) <= RELATIVE_TOLERANCE, (
        f"mean cost over {len(produced_costs)} eval seeds {produced_mean:.1f} vs "
        f"baseline {baseline_mean:.1f} "
        f"({_relative_diff(produced_mean, baseline_mean):.0%} > {RELATIVE_TOLERANCE:.0%})"
    )


def test_final_test_mean_cost_is_within_tolerance_of_the_baseline(
    golden: dict[str, Any], world: dict[str, Any]
) -> None:
    """Mean total cost over each action-count's test seeds, same fixed-fleet table."""
    protocol = golden["protocol"]
    best_w = np.array(golden["training"]["w_trajectory"][-1])

    for actions, episodes in golden["test"].items():
        produced_costs = [
            run_evaluation_episode(
                seed=entry["seed"],
                client_generator=world["client_generator"],
                travel_time_model=world["travel_time_model"],
                shortest_path_cache=world["cache"],
                congestion_generator=world["congestion_generator"],
                W=best_w,
                epsilon=protocol["epsilon"],
                max_congestion_duration=protocol["max_congestion_duration"],
                horizon_start_minute=protocol["horizon_start_time"],
                horizon_end_minute=protocol["horizon_end_time"],
                n_observed_arcs=protocol["n_arcs"],
                vehicle_count=entry["vehicles"],
                number_actions_test=entry["vehicles"] + int(actions),
            ).total_cost
            for entry in episodes
        ]

        baseline_mean = statistics.mean(entry["total_cost"] for entry in episodes)
        produced_mean = statistics.mean(produced_costs)
        assert _relative_diff(produced_mean, baseline_mean) <= RELATIVE_TOLERANCE, (
            f"actions={actions}: mean cost over {len(produced_costs)} test seeds "
            f"{produced_mean:.1f} vs baseline {baseline_mean:.1f} "
            f"({_relative_diff(produced_mean, baseline_mean):.0%} > {RELATIVE_TOLERANCE:.0%})"
        )
