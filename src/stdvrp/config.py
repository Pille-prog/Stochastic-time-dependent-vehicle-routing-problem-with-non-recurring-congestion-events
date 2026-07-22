"""Frozen, validated experiment configuration.

Replaces the legacy comma-separated ``sys.argv`` string plus the values that were
hardcoded across ``main()``, ``training_and_testing`` and ``model`` (horizon 300-780,
``n_arcs=3``, warm-up learning rate 1e-6, evaluation seeds 100000-100049, data file
paths, the ``mean_static_policy`` plot baseline). One YAML file per experiment,
versioned next to the experiment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Every knob of one experiment; immutable once loaded."""

    # Data (formerly hardcoded relative file names resolved against the CWD).
    data_dir: Path
    links_file: str
    shortest_paths_file: str
    instance_day: int
    traffic_days: tuple[int, ...]

    # Horizon in minutes since 03:00 (formerly hardcoded 300 and 780).
    horizon_start_minute: int
    horizon_end_minute: int

    # Demand (former argv: mean_number_clients, diff_TW; former main() hardcodes:
    # random.seed(0); random.sample(range(1, 1900), 150)).
    mean_number_clients: int
    time_window_spread: int
    client_universe_seed: int
    client_universe_size: int
    client_universe_node_range: tuple[int, int]

    # Congestion (former argv).
    congestion_lower_bound: float
    congestion_upper_bound: float
    max_congestion_duration: int

    # Policy and training (former argv plus hardcoded values).
    total_train_iterations: int
    test_frequency: int
    learning_rate: float
    warmup_learning_rate: float
    epsilon: float
    n_observed_arcs: int
    first_train_seed: int
    evaluation_seed_start: int
    evaluation_seed_count: int
    test_episodes: int

    # Plot baseline (was a hardcoded lookup table keyed by experiment parameters).
    static_policy_mean_cost: float | None

    def __post_init__(self) -> None:
        if not self.traffic_days:
            raise ValueError("traffic_days must not be empty")
        if self.instance_day not in self.traffic_days:
            raise ValueError(f"instance_day {self.instance_day} must be one of traffic_days")
        if not 0 <= self.horizon_start_minute < self.horizon_end_minute:
            raise ValueError("horizon must satisfy 0 <= horizon_start_minute < horizon_end_minute")
        if self.mean_number_clients <= 0:
            raise ValueError("mean_number_clients must be positive")
        if self.time_window_spread < 0:
            raise ValueError("time_window_spread must be >= 0")
        lo, hi = self.client_universe_node_range
        if lo >= hi:
            raise ValueError("client_universe_node_range must be (low, high) with low < high")
        if not 0 < self.client_universe_size <= hi - lo:
            raise ValueError(
                f"client_universe_size must be in 1..{hi - lo} for node range ({lo}, {hi})"
            )
        if not 0 <= self.congestion_lower_bound <= self.congestion_upper_bound:
            raise ValueError("congestion bounds must satisfy 0 <= lower <= upper")
        # The legacy draws congestion durations with random.randint(30, max_duration).
        if self.max_congestion_duration < 30:
            raise ValueError("max_congestion_duration must be >= 30 minutes")
        for name in (
            "total_train_iterations",
            "test_frequency",
            "test_episodes",
            "evaluation_seed_count",
            "n_observed_arcs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.warmup_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.static_policy_mean_cost is not None and self.static_policy_mean_cost <= 0:
            raise ValueError("static_policy_mean_cost must be positive or null")

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        """The fixed seeds used for every evaluation pass (legacy range(100000, 100050))."""
        return tuple(
            range(
                self.evaluation_seed_start, self.evaluation_seed_start + self.evaluation_seed_count
            )
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load and validate a config; relative data_dir resolves against the YAML's folder."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: config must be a YAML mapping")
        field_names = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(set(raw) - set(field_names))
        if unknown:
            raise ValueError(f"{path}: unknown config keys {unknown}")
        missing = sorted(set(field_names) - set(raw))
        if missing:
            raise ValueError(f"{path}: missing config keys {missing}")

        values: dict[str, Any] = dict(raw)
        data_dir = Path(str(values["data_dir"]))
        if not data_dir.is_absolute():
            data_dir = path.parent / data_dir
        values["data_dir"] = data_dir
        values["traffic_days"] = tuple(
            _require_int_list(path, "traffic_days", values["traffic_days"])
        )
        node_range = _require_int_list(
            path, "client_universe_node_range", values["client_universe_node_range"]
        )
        if len(node_range) != 2:
            raise ValueError(f"{path}: client_universe_node_range must have exactly 2 entries")
        values["client_universe_node_range"] = (node_range[0], node_range[1])
        for name in (
            "congestion_lower_bound",
            "congestion_upper_bound",
            "learning_rate",
            "warmup_learning_rate",
            "epsilon",
        ):
            values[name] = _require_float(path, name, values[name])
        if values["static_policy_mean_cost"] is not None:
            values["static_policy_mean_cost"] = _require_float(
                path, "static_policy_mean_cost", values["static_policy_mean_cost"]
            )
        for name in ("links_file", "shortest_paths_file"):
            if not isinstance(values[name], str) or not values[name]:
                raise ValueError(f"{path}: {name} must be a non-empty string")
        for name in (
            "instance_day",
            "horizon_start_minute",
            "horizon_end_minute",
            "mean_number_clients",
            "time_window_spread",
            "client_universe_seed",
            "client_universe_size",
            "max_congestion_duration",
            "total_train_iterations",
            "test_frequency",
            "n_observed_arcs",
            "first_train_seed",
            "evaluation_seed_start",
            "evaluation_seed_count",
            "test_episodes",
        ):
            values[name] = _require_int(path, name, values[name])
        return cls(**values)


def _require_int(path: Path, name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: {name} must be an integer, got {value!r}")
    return value


def _require_int_list(path: Path, name: str, value: Any) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{path}: {name} must be a list of integers, got {value!r}")
    return list(value)


def _require_float(path: Path, name: str, value: Any) -> float:
    # PyYAML parses "1e-6" (no dot) as a string; accept it as a float for ergonomics.
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{path}: {name} must be a number, got {value!r}")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{path}: {name} must be a number, got {value!r}") from error
