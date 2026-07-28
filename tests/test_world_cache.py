"""WorldCache: binary snapshot of the loaded world (ticket 03, simulation-performance).

A cache miss parses the fixture CSVs and writes a pickle; a cache hit must
reconstruct a ``TravelTimeModel``/``ShortestPathCache`` pair exactly equal to
what the cold parse produced (the spec's default Tier-1 behavior contract), and
a change to either a source file or a world-shaping config field must never
silently reuse a stale entry.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from stdvrp.config import ExperimentConfig
from stdvrp.traffic import world_cache

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "chengdu_mini"


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig.from_yaml(FIXTURE_DIR / "config.yaml")


def test_cache_miss_then_hit_reproduces_the_parsed_world_exactly(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    # First call: no cache file yet, so this parses the CSVs and writes one.
    built = world_cache.load_world(config, cache_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 1

    # Second call: reconstructed entirely from the pickle (_from_cached_state),
    # never touching the CSVs again.
    cached = world_cache.load_world(config, cache_dir=tmp_path)

    built_model, cached_model = built.travel_time_model, cached.travel_time_model
    assert cached_model.travel_data == built_model.travel_data
    assert cached_model.speed_std == built_model.speed_std
    assert cached_model.successors == built_model.successors
    assert cached_model.node_coordinates == built_model.node_coordinates
    assert cached_model.event_probability == built_model.event_probability
    # Insertion order matters: the live CongestionGenerator draws one random
    # number per event_probability key in dict order (ADR-0001).
    assert list(cached_model.event_probability) == list(built_model.event_probability)
    pd.testing.assert_frame_equal(
        cached_model.mean_arc_data, built_model.mean_arc_data, check_exact=True
    )

    assert cached.shortest_path_cache.as_dict() == built.shortest_path_cache.as_dict()
    assert len(cached.shortest_path_cache) == len(built.shortest_path_cache)


def test_no_cache_dir_parses_fresh_every_call(config: ExperimentConfig, tmp_path: Path) -> None:
    # cache_dir=None (the default) is the pre-ticket-03 behavior: every call
    # re-parses the CSVs independently, so it never writes anywhere, including
    # a cache_dir the caller happens to have lying around unused.
    first = world_cache.load_world(config, cache_dir=None)
    second = world_cache.load_world(config, cache_dir=None)
    assert first.travel_time_model.travel_data == second.travel_time_model.travel_data
    assert list(tmp_path.iterdir()) == []


def test_a_second_config_with_different_world_params_gets_its_own_cache_entry(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    other = dataclasses.replace(config, max_congestion_duration=config.max_congestion_duration + 30)

    world_cache.load_world(config, cache_dir=tmp_path)
    world_cache.load_world(other, cache_dir=tmp_path)

    # Content-addressed by key: two distinct world-shaping configs coexist
    # rather than evicting each other (unlike a single fixed cache slot).
    assert len(list(tmp_path.iterdir())) == 2


def test_touching_a_source_file_invalidates_the_cache(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    world_cache.load_world(config, cache_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 1

    # Same bytes, later mtime: the size+mtime signature still changes (the
    # documented invalidation basis — content hashing the CSVs on every load
    # would itself cost real seconds, defeating the cache's purpose).
    links_path = config.data_dir / config.links_file
    stat = links_path.stat()
    # +1s, not +1ns: filesystem mtime granularity (e.g. NTFS's 100ns ticks) can
    # round a sub-tick nudge back down to the original value.
    os.utime(links_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    world_cache.load_world(config, cache_dir=tmp_path)

    # A stale cache is rebuilt, never silently reused: the new key writes a
    # second file rather than being served from (or overwriting) the first.
    assert len(list(tmp_path.iterdir())) == 2
