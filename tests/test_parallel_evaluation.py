"""Ticket 08 (simulation-performance): parallel evaluation and the final test.

The **Tier 1 gate**: a fixture experiment's ``results.json`` must be
byte-identical whether its evaluation blocks and final test ran serially
in-process or on the persistent worker pool. Around it, the pool's own
contracts: seed-ordered aggregation regardless of completion order, one
executor for the whole run (workers are persistent, each loading the world once
from the ticket-03 binary cache), and the serial fallback.

The world is the usual 44-day copy of the mini fixture (see
``characterization_world``), pre-warmed into a cache directory so the spawned
workers read the binary snapshot instead of re-parsing 88 speed CSVs each.
"""

import dataclasses
import shutil
from pathlib import Path

import pytest

import characterization_world
from stdvrp.config import ExperimentConfig
from stdvrp.simulation import EpisodeResult
from stdvrp.traffic import world_cache
from stdvrp.training import Trainer
from stdvrp.training import episode_pool as episode_pool_module
from stdvrp.training.episode_pool import (
    EpisodePool,
    EpisodeRequest,
    EpisodeWorld,
    default_worker_count,
)

FIXTURE_DIR = characterization_world.FIXTURE_DIR


@pytest.fixture(scope="module")
def parallel_config(tmp_path_factory: pytest.TempPathFactory) -> ExperimentConfig:
    """A tiny experiment on the 44-day fixture world: 2 train, 2 eval, 2 test seeds."""
    world = characterization_world.build_legacy_world(tmp_path_factory.mktemp("parallel_world"))
    shutil.copyfile(FIXTURE_DIR / "all_shortest_paths.csv", world / "all_shortest_paths.csv")
    return dataclasses.replace(
        ExperimentConfig.from_yaml(FIXTURE_DIR / "config.yaml"),
        data_dir=world,
        traffic_days=characterization_world.LEGACY_DAYS,
        total_train_iterations=2,
        test_frequency=1,
        evaluation_seed_count=3,
        test_action_counts=(2, 4),
        test_seeds=(100, 101),
        test_vehicle_counts=(4, 5),
    )


@pytest.fixture(scope="module")
def warm_cache_dir(
    parallel_config: ExperimentConfig, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A world cache pre-warmed in-process, so spawned workers never parse CSVs."""
    cache_dir = tmp_path_factory.mktemp("parallel_cache")
    world_cache.load_world(parallel_config, cache_dir=cache_dir)
    return cache_dir


# --- Tier 1 gate --------------------------------------------------------------


def test_results_json_is_bit_identical_serial_versus_parallel(
    parallel_config: ExperimentConfig, warm_cache_dir: Path, tmp_path: Path
) -> None:
    """The ticket's Tier 1 gate: parallelism must not move a single bit.

    The serial run deliberately parses the CSVs (``cache_dir=None``) while the
    workers unpickle the binary cache, so this also covers the asymmetry a real
    run has: the first process to touch a cold cache builds its world from the
    parse, and every worker gets the round-tripped copy.
    """
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    Trainer.from_config(parallel_config, cache_dir=None, worker_count=1).run(serial)
    Trainer.from_config(parallel_config, cache_dir=warm_cache_dir, worker_count=3).run(parallel)

    assert (parallel / "results.json").read_bytes() == (serial / "results.json").read_bytes()


# --- Seed-ordered aggregation --------------------------------------------------


def test_the_pool_reduces_in_request_order_not_completion_order(
    parallel_config: ExperimentConfig, warm_cache_dir: Path
) -> None:
    """Results come back in request order even when the first request finishes last.

    The batch is deliberately front-loaded with the slowest Episode — a big fleet
    on a wide action pool — and given a worker each, so the two cheap ones behind
    it really do finish first. In completion order the answer would be reversed.
    """
    requests = (
        EpisodeRequest(100002, vehicle_count=8, number_actions_test=20),
        EpisodeRequest(100000, vehicle_count=4, number_actions_test=5),
        EpisodeRequest(100001, vehicle_count=4, number_actions_test=5),
    )
    world = EpisodeWorld.load(parallel_config, cache_dir=warm_cache_dir)
    expected = world.run_episodes(None, requests)

    with EpisodePool(parallel_config, cache_dir=warm_cache_dir, worker_count=3) as pool:
        assert pool.run(None, requests) == expected
    # Distinct requests must give distinct episodes, or the ordering claim is vacuous.
    assert len({result.total_cost for result in expected}) == len(requests)


def test_the_pool_runs_the_requested_overrides(
    parallel_config: ExperimentConfig, warm_cache_dir: Path
) -> None:
    """Fleet-size and action-pool overrides travel with the request, not the world."""
    requests = (
        EpisodeRequest(100, vehicle_count=4, number_actions_test=6),
        EpisodeRequest(100, vehicle_count=6, number_actions_test=8),
    )
    world = EpisodeWorld.load(parallel_config, cache_dir=warm_cache_dir)
    expected = world.run_episodes(None, requests)
    assert expected[0] != expected[1]

    with EpisodePool(parallel_config, cache_dir=warm_cache_dir, worker_count=2) as pool:
        assert pool.run(None, requests) == expected


def test_an_empty_batch_never_starts_a_worker(
    parallel_config: ExperimentConfig, warm_cache_dir: Path
) -> None:
    with EpisodePool(parallel_config, cache_dir=warm_cache_dir, worker_count=2) as pool:
        assert pool.run(None, ()) == ()
        assert not pool.is_started


# --- The pool is persistent ----------------------------------------------------


def test_one_executor_serves_every_batch(
    parallel_config: ExperimentConfig,
    warm_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers are started once and reused, so the world is loaded once per worker."""
    real_executor = episode_pool_module.ProcessPoolExecutor
    created = 0

    def counting_executor(*args: object, **kwargs: object) -> object:
        nonlocal created
        created += 1
        return real_executor(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(episode_pool_module, "ProcessPoolExecutor", counting_executor)

    with EpisodePool(parallel_config, cache_dir=warm_cache_dir, worker_count=2) as pool:
        first = pool.run(None, (EpisodeRequest(100000),))
        second = pool.run(None, (EpisodeRequest(100001),))

    assert created == 1
    assert isinstance(first[0], EpisodeResult) and isinstance(second[0], EpisodeResult)


def test_a_closed_pool_starts_fresh_workers_for_a_later_batch(
    parallel_config: ExperimentConfig, warm_cache_dir: Path
) -> None:
    """``Trainer.run`` closes the pool when it finishes; running twice must still work."""
    pool = EpisodePool(parallel_config, cache_dir=warm_cache_dir, worker_count=2)
    request = (EpisodeRequest(100000),)
    try:
        first = pool.run(None, request)
        pool.close()
        assert not pool.is_started
        assert pool.run(None, request) == first
        assert pool.is_started
    finally:
        pool.close()


# --- Serial fallback and the worker-count switch --------------------------------


def test_worker_count_one_keeps_everything_in_process(
    parallel_config: ExperimentConfig, warm_cache_dir: Path
) -> None:
    trainer = Trainer.from_config(parallel_config, cache_dir=warm_cache_dir, worker_count=1)
    assert trainer.episode_pool is None


def test_parallel_evaluation_requires_a_world_cache_directory(
    parallel_config: ExperimentConfig,
) -> None:
    """Without a cache every worker would re-parse the CSVs; fail loudly instead."""
    with pytest.raises(ValueError, match="cache_dir"):
        Trainer.from_config(parallel_config, cache_dir=None, worker_count=4)


def test_worker_count_must_be_positive(parallel_config: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="worker_count"):
        Trainer.from_config(parallel_config, cache_dir=None, worker_count=0)


def test_default_worker_count_reads_the_environment_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STDVRP_WORKERS", "3")
    assert default_worker_count() == 3

    # Unset: serial. Parallel evaluation is opt-in because a worker is a whole
    # world in memory (8.0 GB on the full dataset), not because cores are scarce.
    monkeypatch.delenv("STDVRP_WORKERS")
    assert default_worker_count() == 1

    monkeypatch.setenv("STDVRP_WORKERS", "nonsense")
    with pytest.raises(ValueError, match="STDVRP_WORKERS"):
        default_worker_count()

    monkeypatch.setenv("STDVRP_WORKERS", "0")
    with pytest.raises(ValueError, match="STDVRP_WORKERS"):
        default_worker_count()
