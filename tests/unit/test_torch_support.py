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


@pytest.fixture
def restore_deterministic_algorithms():
    """Requested explicitly (never autouse) by tests that resolve "cuda" and so
    flip torch.use_deterministic_algorithms as a side effect -- this file also
    has non-neural tests that must pass with torch genuinely absent, which an
    autouse fixture importing torch would break."""
    torch = pytest.importorskip("torch")
    was_enabled = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(was_enabled)


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
def test_resolve_device_with_torch(restore_deterministic_algorithms: None) -> None:
    """On a machine with a real GPU (ticket 12), the "cuda" branch below
    actually runs and flips torch.use_deterministic_algorithms as a side
    effect, which must not leak into later tests in the same session."""
    torch = pytest.importorskip("torch")
    assert resolve_device("cpu") == torch.device("cpu")
    if torch.cuda.is_available():
        assert resolve_device("cuda") == torch.device("cuda")


@pytest.mark.neural
def test_resolve_device_auto_matches_real_hardware(restore_deterministic_algorithms: None) -> None:
    """No monkeypatching: whatever this machine actually has, "auto" must agree."""
    torch = pytest.importorskip("torch")
    expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert resolve_device("auto") == expected


@pytest.mark.neural
def test_resolve_device_auto_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch, restore_deterministic_algorithms: None
) -> None:
    """Forces torch.cuda.is_available() so this is deterministic across CI (no GPU)
    and this machine (real GPU) alike, mirroring test_resolve_device_cuda_enables_
    deterministic_algorithms below."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


@pytest.mark.neural
def test_resolve_device_auto_falls_back_to_cpu_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


@pytest.mark.neural
def test_resolve_device_cpu_does_not_force_deterministic_algorithms(
    restore_deterministic_algorithms: None,
) -> None:
    """ "cuda" opts into determinism (see module docstring); "cpu" needs no such
    global side effect (stdvrp.policies.network's forward pass has no
    stochastic op at all -- see its own docstring), so resolve_device("cpu")
    must not flip a process-wide torch setting nobody asked for."""
    torch = pytest.importorskip("torch")
    torch.use_deterministic_algorithms(False)
    resolve_device("cpu")
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.neural
def test_resolve_device_cuda_enables_deterministic_algorithms(
    monkeypatch: pytest.MonkeyPatch, restore_deterministic_algorithms: None
) -> None:
    """Ticket 05's determinism requirement: resolve_device("cuda") must set
    torch.use_deterministic_algorithms(True). Forces torch.cuda.is_available()
    True so this is deterministic across CI (no GPU) and this machine (real
    GPU) alike -- ticket 12 makes an explicit "cuda" with no GPU raise instead
    of resolving (see test_resolve_device_cuda_without_a_gpu_fails_loudly
    below), so this test can no longer rely on "cuda" being accepted
    regardless of hardware the way it did before that change."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    torch.use_deterministic_algorithms(False)
    resolve_device("cuda")
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.neural
def test_resolve_device_cuda_without_a_gpu_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ticket 12: an explicit "cuda" with no GPU available must raise, never
    silently downgrade to cpu (a silent downgrade would not fail -- it would
    just run ~3.4x slower, trip the 24h safety cap, and be recorded as a
    non-result)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")


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
