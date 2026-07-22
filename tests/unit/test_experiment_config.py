"""ExperimentConfig: YAML loading, freezing, validation."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from stdvrp.config import ExperimentConfig

FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "config.yaml"


def valid_values() -> dict:
    return {
        "data_dir": ".",
        "links_file": "link.csv",
        "shortest_paths_file": "all_shortest_paths.csv",
        "instance_day": 601,
        "traffic_days": [601],
        "horizon_start_minute": 300,
        "horizon_end_minute": 780,
        "mean_number_clients": 20,
        "client_count_stddev": 4.0,
        "min_number_clients": 8,
        "clients_per_vehicle": 4,
        "time_window_spread": 60,
        "client_universe_seed": 0,
        "client_universe_size": 10,
        "client_universe_node_range": [1, 45],
        "congestion_lower_bound": 0.3,
        "congestion_upper_bound": 0.4,
        "max_congestion_duration": 120,
        "total_train_iterations": 10,
        "test_frequency": 5,
        "learning_rate": 1.0e-5,
        "warmup_learning_rate": 1.0e-6,
        "epsilon": 0.1,
        "n_observed_arcs": 3,
        "first_train_seed": 1000,
        "evaluation_seed_start": 100000,
        "evaluation_seed_count": 50,
        "test_episodes": 10,
        "static_policy_mean_cost": None,
    }


def write_config(tmp_path: Path, values: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_loads_the_committed_fixture_config() -> None:
    config = ExperimentConfig.from_yaml(FIXTURE_CONFIG)
    assert config.instance_day == 601
    assert config.traffic_days == (601,)
    assert config.horizon_start_minute == 300
    assert config.horizon_end_minute == 780
    assert config.n_observed_arcs == 3
    assert config.warmup_learning_rate == 1.0e-6
    assert config.data_dir == FIXTURE_CONFIG.parent / "."
    assert (config.data_dir / config.links_file).is_file()


def test_covers_every_field_with_no_extras(tmp_path: Path) -> None:
    config = ExperimentConfig.from_yaml(write_config(tmp_path, valid_values()))
    assert {f.name for f in dataclasses.fields(config)} == set(valid_values())


def test_is_frozen(tmp_path: Path) -> None:
    config = ExperimentConfig.from_yaml(write_config(tmp_path, valid_values()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.epsilon = 0.5  # type: ignore[misc]


def test_evaluation_seeds_expand_start_and_count(tmp_path: Path) -> None:
    config = ExperimentConfig.from_yaml(write_config(tmp_path, valid_values()))
    assert config.evaluation_seeds == tuple(range(100000, 100050))


def test_relative_data_dir_resolves_against_the_yaml_folder(tmp_path: Path) -> None:
    values = valid_values() | {"data_dir": "sub/data"}
    config = ExperimentConfig.from_yaml(write_config(tmp_path, values))
    assert config.data_dir == tmp_path / "sub" / "data"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    values = valid_values() | {"surprise": 1}
    with pytest.raises(ValueError, match=r"unknown config keys.*surprise"):
        ExperimentConfig.from_yaml(write_config(tmp_path, values))


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    values = valid_values()
    del values["epsilon"]
    with pytest.raises(ValueError, match=r"missing config keys.*epsilon"):
        ExperimentConfig.from_yaml(write_config(tmp_path, values))


def test_scientific_notation_without_dot_still_parses_as_float(tmp_path: Path) -> None:
    # PyYAML parses "1e-6" as a string; the loader must still accept it.
    path = tmp_path / "config.yaml"
    values = valid_values()
    del values["warmup_learning_rate"]
    path.write_text(yaml.safe_dump(values) + "warmup_learning_rate: 1e-6\n", encoding="utf-8")
    assert ExperimentConfig.from_yaml(path).warmup_learning_rate == 1.0e-6


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"traffic_days": []}, "traffic_days"),
        ({"instance_day": 602}, "instance_day"),
        ({"horizon_start_minute": 780}, "horizon"),
        ({"epsilon": 1.5}, "epsilon"),
        ({"congestion_lower_bound": 0.5}, "congestion bounds"),
        ({"max_congestion_duration": 29}, "max_congestion_duration"),
        ({"client_universe_size": 100}, "client_universe_size"),
        ({"client_universe_node_range": [45, 1]}, "client_universe_node_range"),
        ({"total_train_iterations": 0}, "total_train_iterations"),
        ({"learning_rate": 0.0}, "learning rates"),
        ({"time_window_spread": -1}, "time_window_spread"),
        ({"time_window_spread": 500}, "time_window_spread"),
        ({"client_count_stddev": -1.0}, "client_count_stddev"),
        ({"min_number_clients": 0}, "min_number_clients"),
        ({"min_number_clients": 100}, "min_number_clients"),
        ({"clients_per_vehicle": 0}, "clients_per_vehicle"),
        ({"static_policy_mean_cost": -3.0}, "static_policy_mean_cost"),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.from_yaml(write_config(tmp_path, valid_values() | overrides))


def test_type_errors_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="instance_day must be an integer"):
        ExperimentConfig.from_yaml(write_config(tmp_path, valid_values() | {"instance_day": "601"}))
    with pytest.raises(ValueError, match="epsilon must be a number"):
        ExperimentConfig.from_yaml(
            write_config(tmp_path, valid_values() | {"epsilon": "not a number"})
        )
