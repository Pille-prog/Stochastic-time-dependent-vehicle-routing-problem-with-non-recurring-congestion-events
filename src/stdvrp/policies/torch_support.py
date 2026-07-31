"""Lazy torch import boundary (ticket 03, neural-policy).

torch is an optional extra (``pyproject.toml``'s ``neural`` group): the
4000+-test suite never touches a network, so ``stdvrp.policies`` must stay
importable without it installed. Every access to torch for device selection
goes through :func:`resolve_device`, which imports torch **inside the
function**, never at module scope — the one place this package is allowed to
need it, and only when a caller actually asks for a device.

Ticket 05's determinism requirement — a fixed init seed and fixed input must
agree bit-for-bit on the configured device — holds on CPU with no extra work
(:mod:`~stdvrp.policies.network` uses no dropout, so its forward pass has no
stochastic op at all). CUDA is a separate, real concern: kernel/algorithm
selection (e.g. which attention backend, which reduction order) is not always
deterministic by default. ``resolve_device("cuda")`` therefore calls
``torch.use_deterministic_algorithms(True)`` — a **global** torch setting,
paid for by every subsequent torch call in the process, not just this
Policy's. The cost: some ops fall back to slower deterministic kernels, and
any future op with no deterministic CUDA implementation raises
``RuntimeError`` instead of silently running nondeterministically — a loud
failure is the point, not a bug to work around. Not exercised against real
CUDA hardware by this ticket (none was available); recorded here rather than
assumed away.
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
    if the ``neural`` extra was never installed. Selecting ``"cuda"`` also
    enables ``torch.use_deterministic_algorithms`` process-wide — see the
    module docstring for what that costs.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(TORCH_INSTALL_HINT) from error
    device = torch.device(name)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
    return device
