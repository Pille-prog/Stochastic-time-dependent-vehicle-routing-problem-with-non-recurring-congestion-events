"""Lazy torch import boundary (ticket 03, neural-policy).

torch is an optional extra (``pyproject.toml``'s ``neural`` group): the
4000+-test suite never touches a network, so ``stdvrp.policies`` must stay
importable without it installed. Every access to torch for device selection
goes through :func:`resolve_device`, which imports torch **inside the
function**, never at module scope — the one place this package is allowed to
need it, and only when a caller actually asks for a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

TORCH_INSTALL_HINT = (
    "torch is not installed; install the optional dependency with `uv sync --extra neural`"
)


def resolve_device(name: str) -> torch.device:
    """``ExperimentConfig.device`` (``"cpu"`` or ``"cuda"``) as a ``torch.device``.

    Raises a clear, actionable error rather than a bare ``ModuleNotFoundError``
    if the ``neural`` extra was never installed.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(TORCH_INSTALL_HINT) from error
    return torch.device(name)
