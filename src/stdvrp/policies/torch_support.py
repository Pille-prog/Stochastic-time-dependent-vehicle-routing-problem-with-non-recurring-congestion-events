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
failure is the point, not a bug to work around. Exercised against real CUDA
hardware for the first time in ticket 12 (RTX 4060 Laptop, 8 GB).

**"auto", and the two failure modes it must not have (ticket 12).**
``ExperimentConfig.device`` defaults to ``"auto"``: ``"cuda"`` if
``torch.cuda.is_available()``, else ``"cpu"``. Two things ``resolve_device``
must never do with it:

1. **Silently prefer the slow device.** Not a risk for ``"auto"`` itself (it
   picks whichever device actually exists), but it would be if an *explicit*
   ``"cuda"`` fell back to CPU when no GPU is present — a run that does that
   does not fail, it just takes ~3.4x longer, trips the 24 h safety cap, and
   is recorded as "did not converge". So explicit ``"cuda"`` with no GPU
   available raises ``RuntimeError`` instead of resolving to CPU.
2. **Pretend the config still pins the result.** CPU and CUDA do not agree
   bit for bit (floating-point reduction order), so under ``"auto"`` the
   config no longer determines what ran — the machine does. Callers that care
   (the checkpoint, the results record) must read the *resolved* device back
   off this function's return value and record it themselves; this module has
   no run-record concept of its own to write it into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

TORCH_INSTALL_HINT = (
    "torch is not installed; install the optional dependency with `uv sync --extra neural`"
)


def resolve_device(name: str) -> torch.device:
    """``ExperimentConfig.device`` (``"cpu"``, ``"cuda"``, or ``"auto"``) as a ``torch.device``.

    Raises a clear, actionable error rather than a bare ``ModuleNotFoundError``
    if the ``neural`` extra was never installed. ``"auto"`` resolves to
    ``"cuda"`` if available, else ``"cpu"`` — call this once per run and keep
    the result; it is not guaranteed to agree between calls. An explicit
    ``"cuda"`` with no GPU available raises ``RuntimeError`` rather than
    degrading to CPU (see the module docstring). Selecting ``"cuda"`` (whether
    named explicitly or reached via ``"auto"``) also enables
    ``torch.use_deterministic_algorithms`` process-wide — see the module
    docstring for what that costs.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(TORCH_INSTALL_HINT) from error

    if name == "auto":
        resolved_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device: cuda was requested explicitly but torch.cuda.is_available() is "
            "False on this machine -- refusing to fall back to cpu silently. A silent "
            "fallback would not fail; it would just run ~3.4x slower, trip the 24h "
            "safety cap, and be recorded as a non-result. Pass device: cpu (or auto) "
            "instead, or run on a machine with a CUDA GPU."
        )
    else:
        resolved_name = name

    device = torch.device(resolved_name)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
    return device
