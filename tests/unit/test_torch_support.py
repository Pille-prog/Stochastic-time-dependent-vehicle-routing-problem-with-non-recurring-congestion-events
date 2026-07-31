"""torch's lazy-import boundary (ticket 03, neural-policy).

Proves the ticket's two load-bearing claims as executable tests rather than a
docstring: ``stdvrp.policies`` stays importable with torch unimportable, and
``resolve_device`` fails with an actionable message rather than a bare
``ModuleNotFoundError`` when it is. Both are tested by forcing
``sys.modules["torch"] = None`` (Python then raises ``ImportError`` on any
``import torch``) rather than by actually uninstalling torch, so the "torch
absent" branch is exercised deterministically regardless of this machine's own
environment.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# stdvrp.simulation.episode imports stdvrp.policies.monte_carlo, and
# stdvrp.policies.monte_carlo (via feature_extraction) imports
# stdvrp.simulation.state — a pre-existing circular import that only resolves
# if stdvrp.simulation finishes initializing first (every other test file
# reaches this module after something else has already imported it; this is
# the first to import stdvrp.policies directly, so it must do so explicitly).
import stdvrp.simulation  # noqa: F401
from stdvrp.policies.torch_support import TORCH_INSTALL_HINT, resolve_device

POLICY_MODULES = (
    "stdvrp.policies",
    "stdvrp.policies.base",
    "stdvrp.policies.monte_carlo",
    "stdvrp.policies.feature_extraction",
    "stdvrp.policies.torch_support",
)


def test_resolve_device_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="uv sync --extra neural"):
        resolve_device("cpu")


def test_stdvrp_policies_importable_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A future ``import torch`` at module scope anywhere in ``policies`` would fail this."""
    monkeypatch.setitem(sys.modules, "torch", None)
    for name in POLICY_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in POLICY_MODULES:
        importlib.import_module(name)


@pytest.mark.neural
def test_resolve_device_with_torch() -> None:
    torch = pytest.importorskip("torch")
    assert resolve_device("cpu") == torch.device("cpu")
    if torch.cuda.is_available():
        assert resolve_device("cuda") == torch.device("cuda")


@pytest.mark.neural
def test_resolve_device_cpu_does_not_force_deterministic_algorithms() -> None:
    """ "cuda" opts into determinism (see module docstring); "cpu" needs no such
    global side effect (stdvrp.policies.network's forward pass has no
    stochastic op at all -- see its own docstring), so resolve_device("cpu")
    must not flip a process-wide torch setting nobody asked for."""
    torch = pytest.importorskip("torch")
    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        resolve_device("cpu")
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(was_enabled)


@pytest.mark.neural
def test_resolve_device_cuda_enables_deterministic_algorithms() -> None:
    """Ticket 05's determinism requirement: resolve_device("cuda") must set
    torch.use_deterministic_algorithms(True) even without CUDA hardware present
    -- torch.device("cuda") itself never touches the GPU, so this is safe to
    assert on CPU-only CI."""
    torch = pytest.importorskip("torch")
    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        resolve_device("cuda")
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(was_enabled)


@pytest.mark.neural
def test_resolve_device_rejects_an_invalid_device_name_like_torch_itself() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError):
        resolve_device("not-a-real-device")
    # Sanity: resolve_device does not swallow torch's own validation.
    with pytest.raises(RuntimeError):
        torch.device("not-a-real-device")


def test_install_hint_names_the_extra() -> None:
    assert "neural" in TORCH_INSTALL_HINT
