"""Unit tests for ``ReferenceCard`` (ticket 01, neural-policy).

The frozen per-seed baseline that retires the scalar ``static_policy_mean_cost``:
a completed linear ``MonteCarloPolicy`` run's per-seed costs over ``test_seeds``
and ``evaluation_seeds``, which is what makes the neural-policy effort's paired
comparison (spec.md, "Why the paired comparison is valid") possible at all.
"""

import dataclasses
import json

import pytest

from stdvrp.training.reference_card import ReferenceCard


def make_card(**overrides: object) -> ReferenceCard:
    values: dict[str, object] = {
        "winning_budget": 500,
        "winning_test_action_count": 20,
        "test_seeds": (100, 101, 102),
        "test_seed_costs": (2000.0, 2200.0, 1800.0),
        "evaluation_seeds": (100000, 100001),
        "evaluation_seed_costs": (2100.0, 1900.0),
        "best_w": (0.1, 0.2, 0.3),
        "config": {"mean_number_clients": 150},
        "wall_clock_seconds": {"world_load": 39.1, "train": 1200.0, "final_test": 400.0},
    }
    values.update(overrides)
    return ReferenceCard(**values)  # type: ignore[arg-type]


class TestConstruction:
    def test_is_frozen(self) -> None:
        card = make_card()
        with pytest.raises(dataclasses.FrozenInstanceError):
            card.winning_budget = 100  # type: ignore[misc]

    def test_rejects_mismatched_test_vector_lengths(self) -> None:
        with pytest.raises(ValueError, match="test_seeds"):
            make_card(test_seed_costs=(1.0, 2.0))

    def test_rejects_mismatched_evaluation_vector_lengths(self) -> None:
        with pytest.raises(ValueError, match="evaluation_seeds"):
            make_card(evaluation_seed_costs=(1.0,))


class TestMeanCosts:
    def test_test_mean_cost(self) -> None:
        card = make_card()
        assert card.test_mean_cost == pytest.approx(2000.0)

    def test_evaluation_mean_cost(self) -> None:
        card = make_card()
        assert card.evaluation_mean_cost == pytest.approx(2000.0)


class TestCostBySeed:
    def test_test_cost_by_seed_pairs_in_order(self) -> None:
        card = make_card()
        assert card.test_cost_by_seed() == {100: 2000.0, 101: 2200.0, 102: 1800.0}

    def test_evaluation_cost_by_seed_pairs_in_order(self) -> None:
        card = make_card()
        assert card.evaluation_cost_by_seed() == {100000: 2100.0, 100001: 1900.0}


class TestJsonRoundTrip:
    def test_to_json_is_plain_json_serializable(self) -> None:
        card = make_card()
        document = card.to_json()
        assert json.loads(json.dumps(document)) == document

    def test_from_json_round_trips_to_an_equal_card(self) -> None:
        card = make_card()
        assert ReferenceCard.from_json(card.to_json()) == card

    def test_save_and_load_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        card = make_card()
        path = tmp_path / "reference_card.json"

        card.save(path)
        loaded = ReferenceCard.load(path)

        assert loaded == card

    def test_saved_file_ends_with_a_trailing_newline(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "reference_card.json"
        make_card().save(path)
        assert path.read_text(encoding="utf-8").endswith("\n")
