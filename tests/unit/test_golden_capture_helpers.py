"""Unit tests for the pure helpers of scripts/capture_golden_master.py.

These run everywhere (no Chengdu data needed). The expensive re-run test that
uses the real dataset lives in tests/test_golden_master.py behind the `golden`
marker.
"""

import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestToJsonable:
    def test_converts_numpy_scalars_to_python_floats(self, capture):
        out = capture.to_jsonable({"a": np.float64(1.25), "b": np.int64(3)})
        assert type(out["a"]) is float
        assert type(out["b"]) is int

    def test_converts_numpy_arrays_to_lists(self, capture):
        out = capture.to_jsonable({"w": np.array([0.1, 0.2])})
        assert out["w"] == [0.1, 0.2]
        assert all(type(x) is float for x in out["w"])

    def test_recurses_into_nested_containers(self, capture):
        out = capture.to_jsonable({"outer": [{"inner": np.float64(2.5)}]})
        assert type(out["outer"][0]["inner"]) is float

    def test_rejects_non_finite_floats(self, capture):
        # A golden master with NaN/inf could never be compared exactly; fail loudly
        # at capture time instead of storing a poisoned value.
        with pytest.raises(ValueError, match="non-finite"):
            capture.to_jsonable({"bad": float("nan")})


class TestJsonRoundTrip:
    def test_awkward_floats_survive_bit_exactly(self, capture, tmp_path):
        # Shortest-repr JSON floats round-trip float64 exactly; this is what makes
        # "exact golden equality" via a JSON file sound.
        values = {
            "floats": [0.1, 1 / 3, math.pi, 1e-300, 123456789.123456789, 5e-324],
            "int": 42,
        }
        path = tmp_path / "gm.json"
        capture.write_json(path, values)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        for original, reloaded in zip(values["floats"], loaded["floats"], strict=True):
            assert original == reloaded

    def test_numpy_values_round_trip_exactly(self, capture, tmp_path):
        rng_free = np.linspace(0.0, 1.0, 7) ** 3  # deterministic awkward decimals
        path = tmp_path / "gm.json"
        capture.write_json(path, {"w": rng_free})
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["w"] == list(rng_free)


class TestWorldCache:
    KEY: ClassVar[dict] = {
        "cache_format": 1,
        "legacy_sha256": "abc",
        "data_signature": {"link.csv": 10},
    }

    def test_round_trips_the_world_on_matching_key(self, capture, tmp_path):
        path = tmp_path / "world.pkl"
        world = ({"travel_data": [1.5, 2.5]}, {"paths": {(0, 1): ([0.0, 1.0], 3.5, 1.2)}})
        capture.write_world_cache(path, self.KEY, world)
        assert capture.read_world_cache(path, self.KEY) == world

    def test_key_mismatch_misses(self, capture, tmp_path):
        path = tmp_path / "world.pkl"
        capture.write_world_cache(path, self.KEY, "world")
        stale = {**self.KEY, "legacy_sha256": "different"}
        assert capture.read_world_cache(path, stale) is None

    def test_missing_file_misses(self, capture, tmp_path):
        assert capture.read_world_cache(tmp_path / "absent.pkl", self.KEY) is None

    def test_corrupt_file_misses_instead_of_raising(self, capture, tmp_path):
        path = tmp_path / "world.pkl"
        path.write_bytes(b"not a pickle")
        assert capture.read_world_cache(path, self.KEY) is None

    def test_write_is_atomic_no_tmp_file_left_behind(self, capture, tmp_path):
        path = tmp_path / "world.pkl"
        capture.write_world_cache(path, self.KEY, "world")
        assert [p.name for p in tmp_path.iterdir()] == ["world.pkl"]

    def test_default_cache_path_honors_env_override(self, capture, monkeypatch, tmp_path):
        monkeypatch.setenv("STDVRP_GOLDEN_CACHE", str(tmp_path / "custom.pkl"))
        assert capture.default_cache_path() == tmp_path / "custom.pkl"

    def test_default_cache_path_stays_outside_the_repo(self, capture, monkeypatch):
        # The pickle can reach gigabytes; it must never land in the OneDrive-synced repo.
        monkeypatch.delenv("STDVRP_GOLDEN_CACHE", raising=False)
        default = capture.default_cache_path()
        assert not default.is_relative_to(REPO_ROOT)


class TestCompareResults:
    def test_identical_dicts_yield_no_differences(self, capture):
        a = {"x": [1.0, {"y": 2.0}], "z": "s"}
        assert capture.compare_results(a, a) == []

    def test_reports_path_to_a_float_mismatch(self, capture):
        a = {"training": {"w_trajectory": [[0.1, 0.2]]}}
        b = {"training": {"w_trajectory": [[0.1, 0.2000000001]]}}
        diffs = capture.compare_results(a, b)
        assert len(diffs) == 1
        assert "training.w_trajectory[0][1]" in diffs[0]

    def test_reports_missing_and_extra_keys(self, capture):
        diffs = capture.compare_results({"a": 1}, {"b": 1})
        assert any("a" in d for d in diffs)
        assert any("b" in d for d in diffs)

    def test_reports_length_mismatch(self, capture):
        diffs = capture.compare_results({"l": [1, 2]}, {"l": [1]})
        assert len(diffs) == 1
        assert "l" in diffs[0]
