"""Unit tests for :mod:`stdvrp.training.neural_checkpoint` (ticket 07, neural-policy).

``TestAtomicWrite`` is the ticket's real deliverable ("Ctrl-C leaves a usable
checkpoint, not a corrupt file"): a write that fails partway through must
never disturb an existing, previously-valid checkpoint.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stdvrp.simulation  # noqa: E402, F401 -- circular-import landmine, see test_tokenizer.py
from stdvrp.config import ExperimentConfig  # noqa: E402
from stdvrp.training.neural_checkpoint import (  # noqa: E402
    TrainingCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from stdvrp.training.neural_episode import build_neural_policy_state  # noqa: E402
from stdvrp.training.neural_report import ConvergenceState, EvaluationReport  # noqa: E402

pytestmark = pytest.mark.neural

FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "chengdu_mini" / "config.yaml"


def make_config(**overrides: object) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(FIXTURE_CONFIG)
    values: dict[str, object] = {"neural_d_model": 8, "neural_n_layers": 1, "neural_n_heads": 2}
    values.update(overrides)
    return dataclasses.replace(config, **values)


def make_checkpoint(episodes_completed: int = 10) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        episodes_completed=episodes_completed,
        elapsed_seconds=123.4,
        convergence=ConvergenceState(
            current_lr=1e-4, best_mean_cost=500.0, best_episode=8, reductions_applied=1
        ),
        evaluations=(
            EvaluationReport(
                episodes_completed=8, seed_costs=(1.0, 2.0), reference_seed_costs=(1.5, 2.5)
            ),
        ),
    )


class TestRoundTrip:
    def test_save_then_load_restores_weights_and_history_exactly(self, tmp_path: Path) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        with torch.no_grad():
            for param in state.encoder.parameters():
                param.add_(1.0)
        saved_params = [p.clone() for p in state.encoder.parameters()]

        checkpoint = make_checkpoint()
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, checkpoint, state)

        fresh_state = build_neural_policy_state(config, np.random.default_rng(1))
        loaded = load_checkpoint(path, fresh_state)

        assert loaded.episodes_completed == checkpoint.episodes_completed
        assert loaded.elapsed_seconds == checkpoint.elapsed_seconds
        assert loaded.convergence == checkpoint.convergence
        assert loaded.evaluations == checkpoint.evaluations
        for saved, restored in zip(saved_params, fresh_state.encoder.parameters(), strict=True):
            torch.testing.assert_close(saved, restored, atol=0.0, rtol=0.0)

    def test_optimizer_momentum_survives_the_round_trip(self, tmp_path: Path) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        # One optimizer step so Adam has real momentum/variance state to restore.
        loss = sum(p.sum() for p in state.encoder.parameters())
        loss.backward()
        state.optimizer.step()
        original_state_dict = state.optimizer.state_dict()

        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, make_checkpoint(), state)

        fresh_state = build_neural_policy_state(config, np.random.default_rng(1))
        load_checkpoint(path, fresh_state)

        restored_keys = fresh_state.optimizer.state_dict()["state"].keys()
        assert restored_keys == original_state_dict["state"].keys()

    def test_the_ridge_accumulator_survives_the_round_trip(self, tmp_path: Path) -> None:
        """Ticket 16: without this, resuming a run would restart accumulation
        from an empty ``A``/``b`` while the loaded network already reflects
        everything accumulated before the interruption -- this must catch
        that regression, not just confirm ``load_checkpoint`` doesn't crash
        on the new key."""
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        rng = np.random.default_rng(2)
        phi = rng.normal(size=(7, state.ridge.dim))
        y = rng.normal(size=7)
        state.ridge.observe_episode(phi, y, aborted=False)
        state.ridge.observe_episode(np.empty((0, state.ridge.dim)), np.empty(0), aborted=True)
        state.ridge.solve()  # freezes the column scale
        original = state.ridge.state_dict()

        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, make_checkpoint(), state)

        fresh_state = build_neural_policy_state(config, np.random.default_rng(1))
        load_checkpoint(path, fresh_state)

        restored = fresh_state.ridge.state_dict()
        np.testing.assert_array_equal(restored["A"], original["A"])
        np.testing.assert_array_equal(restored["b"], original["b"])
        np.testing.assert_array_equal(restored["scale"], original["scale"])
        assert restored["effective_n"] == original["effective_n"]
        assert restored["episodes_included"] == original["episodes_included"] == 1
        assert restored["episodes_excluded"] == original["episodes_excluded"] == 1
        assert restored["episodes_since_solve"] == original["episodes_since_solve"]
        np.testing.assert_allclose(fresh_state.ridge.solve(), state.ridge.solve())


class TestAtomicWrite:
    def test_a_failed_write_does_not_disturb_the_previous_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        path = tmp_path / "checkpoint.pt"

        good = make_checkpoint(episodes_completed=5)
        save_checkpoint(path, good, state)
        assert path.read_bytes()  # sanity: something was actually written

        import stdvrp.training.neural_checkpoint as checkpoint_module

        def exploding_save(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated interruption mid-write")

        monkeypatch.setattr(checkpoint_module.torch, "save", exploding_save)

        with pytest.raises(RuntimeError, match="simulated interruption"):
            save_checkpoint(path, make_checkpoint(episodes_completed=6), state)

        # The temp file (if any) never replaced the real, previously-valid one.
        loaded = load_checkpoint(path, state)
        assert loaded.episodes_completed == 5
        assert not (tmp_path / "checkpoint.pt.tmp").exists()

    def test_writes_to_a_temp_file_first_and_replaces_atomically(self, tmp_path: Path) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        path = tmp_path / "checkpoint.pt"

        save_checkpoint(path, make_checkpoint(), state)

        assert path.exists()
        assert not (tmp_path / "checkpoint.pt.tmp").exists()


class TestDeviceGuard:
    """Ticket 12: a checkpoint records the device its weights were written on;
    resuming on a different device is refused, never silently allowed (CPU and
    CUDA do not agree bit for bit, so loading anyway would silently break
    ticket 07's bit-identical resume guarantee). Tampers with the saved
    document's "device" field directly, so this is real hardware-independent
    coverage of the guard itself -- no second GPU needed to prove it fires."""

    def test_save_records_the_device(self, tmp_path: Path) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        path = tmp_path / "checkpoint.pt"

        save_checkpoint(path, make_checkpoint(), state)

        document = torch.load(path, weights_only=False)
        assert document["device"] == str(state.device) == "cpu"

    def test_a_mismatched_device_refuses_to_load(self, tmp_path: Path) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, make_checkpoint(), state)

        # Tamper with the saved document to simulate a checkpoint written on a
        # different device than this run resolved -- the scenario the guard
        # exists for, reproduced without needing a second physical device.
        document = torch.load(path, weights_only=False)
        document["device"] = "cuda"
        torch.save(document, path)

        with pytest.raises(RuntimeError, match="cross-device"):
            load_checkpoint(path, state)

    def test_a_matching_device_loads_normally(self, tmp_path: Path) -> None:
        config = make_config()
        state = build_neural_policy_state(config, np.random.default_rng(0))
        path = tmp_path / "checkpoint.pt"
        checkpoint = make_checkpoint(episodes_completed=7)
        save_checkpoint(path, checkpoint, state)

        loaded = load_checkpoint(path, state)

        assert loaded.episodes_completed == 7
