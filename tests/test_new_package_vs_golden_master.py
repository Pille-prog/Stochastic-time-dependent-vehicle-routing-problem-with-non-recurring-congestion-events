"""Tickets 07/08/09 acceptance: the new package reproduces the golden master exactly.

Ticket 07 — for every golden-master test episode (full Chengdu data, marker
``golden``): the episode total cost and all four components — distance, delay,
earliness, overtime — plus tau, state count and the delay/earliness client counts
must equal the stored values bit-for-bit. W is injected from the stored
post-training trajectory, exactly as the legacy's ``Best_W`` drives its test
episodes.

Ticket 08 — re-running the stored training protocol through the new package
(warm-up learning rate on the first Episode, capture-convention exploration
seeding) must reproduce the stored W trajectory bit-for-bit after every Episode.

Ticket 09 — one Trainer.run() driven by an ExperimentConfig assembled from the
stored protocol must reproduce the whole golden master — W trajectory, per-seed
evaluation costs and every test episode — and write the per-run outputs.

Skips when the local dataset is absent (e.g. CI); building the world through the
new package re-reads the 88 speed files and the 907 MB path cache (~15 minutes).
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stdvrp.config import ExperimentConfig
from stdvrp.congestion import ArcProbabilityCongestionGenerator
from stdvrp.demand import ClientGenerator
from stdvrp.network import ShortestPathCache
from stdvrp.simulation import run_evaluation_episode, run_training_episode
from stdvrp.traffic import CsvDataSource, TravelTimeModel
from stdvrp.training import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_master" / "chengdu_full.json"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))

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
        pytest.skip("golden master not captured yet (scripts/capture_golden_master.py)")
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


def test_w_trajectory_matches_exactly(golden: dict[str, Any], world: dict[str, Any]) -> None:
    """Ticket 08: the training W sequence equals the stored trajectory bit-for-bit."""
    protocol = golden["protocol"]
    expected_trajectory = golden["training"]["w_trajectory"]

    w = None
    # Legacy warm-up quirk (pending ticket 12 triage): the first training Episode
    # always runs with the hardcoded tiny rate, later ones with the configured one.
    learning_rate = protocol["warmup_learning_rate"]
    produced = []
    for train_seed in protocol["train_seeds"]:
        result = run_training_episode(
            seed=train_seed,
            client_generator=world["client_generator"],
            travel_time_model=world["travel_time_model"],
            shortest_path_cache=world["cache"],
            congestion_generator=world["congestion_generator"],
            W=w,
            learning_rate=learning_rate,
            epsilon=protocol["epsilon"],
            max_congestion_duration=protocol["max_congestion_duration"],
            horizon_start_minute=protocol["horizon_start_time"],
            horizon_end_minute=protocol["horizon_end_time"],
            n_observed_arcs=protocol["n_arcs"],
            exploration_seed=protocol["train_exploration_seed_offset"] + train_seed,
            repair_seed=protocol["train_repair_seed_offset"] + train_seed,
        )
        learning_rate = protocol["learning_rate"]
        w = result.w
        produced.append([float(component) for component in result.w])

    mismatches = [
        f"episode {i} (seed {seed}): {diff}"
        for i, (seed, expected, actual) in enumerate(
            zip(protocol["train_seeds"], expected_trajectory, produced, strict=True)
        )
        for diff in capture.compare_results(expected, actual)
    ]
    assert not mismatches, "W trajectory mismatch:\n" + "\n".join(mismatches[:50])


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
        train_exploration_seed_offset=protocol["train_exploration_seed_offset"],
        train_repair_seed_offset=protocol["train_repair_seed_offset"],
        static_policy_mean_cost=None,
    )


def test_trainer_run_reproduces_the_whole_golden_master(
    golden: dict[str, Any], data_dir: Path, world: dict[str, Any], tmp_path: Path
) -> None:
    """Ticket 09: one config-driven Trainer.run() equals the stored capture bit-for-bit."""
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

    produced_trajectory = [[float(x) for x in w] for w in result.training.w_trajectory]
    mismatches = capture.compare_results(golden["training"]["w_trajectory"], produced_trajectory)

    # The single evaluation block is the capture's eval pass: same per-seed costs,
    # and (having beaten the initial best-cost sentinel) it pins Best_W = Newest_W.
    assert [block.episodes_completed for block in result.training.evaluations] == [
        len(protocol["train_seeds"])
    ]
    mismatches += capture.compare_results(
        golden["training"]["eval_costs"], list(result.training.evaluations[0].seed_costs)
    )
    assert result.training.best_w is not None
    assert list(result.tested_w) == list(result.training.best_w) == produced_trajectory[-1]

    # test_episodes=1: each per-seed mean IS the single captured episode.
    for report in result.test:
        for entry, expected in zip(
            report.per_seed, golden["test"][str(report.action_count)], strict=True
        ):
            produced = {"seed": entry.seed, "vehicles": entry.vehicle_count, **entry.metrics}
            for key, value in expected.items():
                if produced[key] != value:
                    mismatches.append(
                        f"actions={report.action_count} seed={entry.seed} {key}: "
                        f"{value!r} != {produced[key]!r}"
                    )

    assert not mismatches, "golden mismatch:\n" + "\n".join(mismatches[:50])
    assert (tmp_path / "run" / "results.json").is_file()
    assert (tmp_path / "run" / "training_plot.png").stat().st_size > 0


def test_every_golden_test_episode_matches_exactly(
    golden: dict[str, Any], world: dict[str, Any]
) -> None:
    protocol = golden["protocol"]
    best_w = np.array(golden["training"]["w_trajectory"][-1])

    mismatches = []
    for actions, episodes in golden["test"].items():
        for entry in episodes:
            result = run_evaluation_episode(
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
            )
            produced = {
                "seed": entry["seed"],
                "vehicles": entry["vehicles"],
                "total_cost": result.total_cost,
                "distance_cost": result.distance_cost,
                "delay_cost": result.delay_cost,
                "earliness_cost": result.earliness_cost,
                "overtime_cost": result.overtime_cost,
                "tau": result.tau,
                "state_count": result.state_count,
                "delay_clients": result.delay_clients,
                "earliness_clients": result.earliness_clients,
            }
            for key, expected in entry.items():
                if produced[key] != expected:
                    mismatches.append(
                        f"actions={actions} seed={entry['seed']} {key}: "
                        f"{expected!r} != {produced[key]!r}"
                    )

    assert not mismatches, "golden mismatch:\n" + "\n".join(mismatches[:50])
