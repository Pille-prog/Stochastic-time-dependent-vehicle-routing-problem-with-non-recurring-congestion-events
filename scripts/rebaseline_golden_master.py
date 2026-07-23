"""Re-baseline the golden master from the NEW package (ticket 12, phase 2).

Phase-2 fixes (ADR-0001 change log) deliberately change episode outcomes, so the
new package can no longer match the legacy capture bit-for-bit. This script runs
the capture protocol stored in ``chengdu_full.json`` through the ported Trainer
on the full local dataset and writes the outcome to ``chengdu_full_phase2.json``.

Ticket 13 (RNG modernization) repurposed that file as the pre-migration
statistical baseline: ``tests/test_new_package_vs_golden_master.py`` now compares
its mean-cost-over-N-seeds against it within a tolerance, instead of asserting
exact equality (ADR-0001 phase-2 addendum) — this script is unchanged, since it
still captures exactly the moment right before ticket 13 touched any RNG call
site.

The legacy capture is NOT touched: ``chengdu_full.json`` remains the frozen
evidence of what the monolith computed, and ``tests/test_golden_master.py``
keeps re-verifying the legacy against it.

Usage (writes tests/fixtures/golden_master/chengdu_full_phase2.json):

    uv run python scripts/rebaseline_golden_master.py

Expect ~25-40 minutes: the world build re-reads the 88 speed files and the
907 MB path cache, then the training/evaluation/test episodes run.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_BASELINE = REPO_ROOT / "tests" / "fixtures" / "golden_master" / "chengdu_full.json"
PHASE2_BASELINE = REPO_ROOT / "tests" / "fixtures" / "golden_master" / "chengdu_full_phase2.json"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))


def _load_capture_module() -> ModuleType:
    """scripts/capture_golden_master.py: data-dir and comparison helpers."""
    spec = importlib.util.spec_from_file_location(
        "capture_golden_master", REPO_ROOT / "scripts" / "capture_golden_master.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_from_protocol(protocol: dict[str, Any], data_dir: Path) -> Any:
    """The stored capture protocol as one ExperimentConfig.

    Mirrors ``config_from_protocol`` in tests/test_new_package_vs_golden_master.py
    (tests/ is not importable from a standalone script run).
    """
    from stdvrp.config import ExperimentConfig

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
        # The legacy ClientGenerator hardcodes.
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


def run_rebaseline() -> None:
    from stdvrp.congestion import ArcProbabilityCongestionGenerator
    from stdvrp.demand import ClientGenerator
    from stdvrp.network import ShortestPathCache
    from stdvrp.traffic import CsvDataSource, TravelTimeModel
    from stdvrp.training import Trainer

    capture = _load_capture_module()
    golden = json.loads(LEGACY_BASELINE.read_text(encoding="utf-8"))
    protocol = golden["protocol"]

    data_dir = capture.default_data_dir()
    missing = capture.missing_data_files(data_dir)
    if missing:
        raise SystemExit(
            f"full Chengdu dataset not available under {data_dir} "
            f"({len(missing)} files missing, first: {missing[0]})"
        )

    started = time.monotonic()
    print(f"building the world from {data_dir} (~15 min cold) ...", flush=True)
    source = CsvDataSource(data_dir, "link.csv", 601, LEGACY_DAYS, "all_shortest_paths.csv")
    travel_time_model = TravelTimeModel(
        source.load_road_network(),
        source.load_traffic_history(),
        protocol["max_congestion_duration"],
        horizon_start_minute=protocol["horizon_start_time"],
    )
    config = config_from_protocol(protocol, data_dir)
    trainer = Trainer(
        config,
        client_generator=ClientGenerator(
            mean_number_clients=protocol["mean_number_clients"],
            client_count_stddev=30.0,
            min_number_clients=60,
            client_universe_node_range=(1, 1900),
            clients_per_vehicle=28,
            time_window_spread=protocol["diff_TW"],
            horizon_start_minute=protocol["horizon_start_time"],
            horizon_end_minute=protocol["horizon_end_time"],
        ),
        travel_time_model=travel_time_model,
        shortest_path_cache=ShortestPathCache.from_csv(data_dir / "all_shortest_paths.csv"),
        congestion_generator=ArcProbabilityCongestionGenerator(
            event_probability=travel_time_model.event_probability,
            successors=travel_time_model.successors,
            congestion_lower_bound=protocol["congestion_lower_bound"],
            congestion_upper_bound=protocol["congestion_upper_bound"],
            max_congestion_duration=protocol["max_congestion_duration"],
        ),
        log=lambda message: print(message, flush=True),
    )
    print(f"world ready after {time.monotonic() - started:.0f}s; running the protocol", flush=True)

    with tempfile.TemporaryDirectory(prefix="rebaseline_run_") as run_dir:
        result = trainer.run(Path(run_dir))

    assert [block.episodes_completed for block in result.training.evaluations] == [
        len(protocol["train_seeds"])
    ]
    document = {
        "meta": {
            **golden["meta"],
            "rebaselined_from": LEGACY_BASELINE.name,
            "rebaseline_reason": (
                "ticket 12 phase-2 fixes change episode outcomes deliberately "
                "(ADR-0001 phase-2 change log); values produced by the new package"
            ),
            "rebaseline_date": datetime.date.today().isoformat(),
        },
        "protocol": protocol,
        "training": {
            "w_trajectory": [[float(x) for x in w] for w in result.training.w_trajectory],
            "eval_costs": [float(c) for c in result.training.evaluations[0].seed_costs],
        },
        "test": {
            str(report.action_count): [
                {"seed": entry.seed, "vehicles": entry.vehicle_count, **entry.metrics}
                for entry in report.per_seed
            ]
            for report in result.test
        },
    }
    PHASE2_BASELINE.write_text(
        json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    diffs = capture.compare_results(
        {"training": golden["training"], "test": golden["test"]},
        {"training": document["training"], "test": document["test"]},
    )
    print(f"\nwritten: {PHASE2_BASELINE}")
    print(f"values differing from the legacy capture: {len(diffs)}")
    for line in diffs[:20]:
        print(f"  {line}")
    print(f"total {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    run_rebaseline()
