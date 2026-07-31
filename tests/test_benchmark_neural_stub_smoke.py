"""CI smoke for the stub network benchmark (ticket 03): it runs, no timing assertions.

Mirrors ``tests/test_benchmark_smoke.py``'s contract: a measurement tool, not a
correctness gate, so this only checks the tiny-shape path executes end to end
and produces finite, positive timings — never a wall-clock threshold. Skips
cleanly (via the ``neural_stub_module`` fixture) when torch is not installed.
"""

from __future__ import annotations

from types import ModuleType

import pytest

pytestmark = pytest.mark.neural


def test_stub_benchmark_runs_and_times_are_finite(neural_stub_module: ModuleType) -> None:
    torch = pytest.importorskip("torch")
    module = neural_stub_module
    model = module.build_stub_network(d_model=16, n_layers=1, n_heads=2)
    device = torch.device("cpu")
    model = model.to(device)

    acting = module.time_acting_path(
        model, device, seq_len=5, d_model=16, decisions_per_episode=3, warmup=1
    )
    assert acting.total_seconds > 0
    assert acting.per_unit > 0

    learning = module.time_learning_path(
        model,
        device,
        seq_len=5,
        d_model=16,
        decisions_per_episode=6,
        minibatch_size=2,
        k_passes=2,
        warmup=1,
    )
    assert learning.total_seconds > 0


def test_main_runs_end_to_end_on_cpu(
    neural_stub_module: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    neural_stub_module.main(
        [
            "--d-model",
            "16",
            "--n-layers",
            "1",
            "--n-heads",
            "2",
            "--client-tokens",
            "3",
            "--vehicle-tokens",
            "2",
            "--decisions-per-episode",
            "4",
            "--minibatch-size",
            "2",
            "--k-passes",
            "1",
            "--warmup",
            "1",
            "--cpu-only",
        ]
    )
    out = capsys.readouterr().out
    assert "cpu" in out
    assert "acting" in out
    assert "learning" in out


def test_build_stub_network_param_count_is_positive(neural_stub_module: ModuleType) -> None:
    model = neural_stub_module.build_stub_network(d_model=16, n_layers=1, n_heads=2)
    assert sum(p.numel() for p in model.parameters()) > 0
