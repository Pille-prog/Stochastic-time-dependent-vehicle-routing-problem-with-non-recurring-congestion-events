"""Stub transformer cost measurement (ticket 03, neural-policy).

spec.md's compute budget rested on a napkin estimate of PyTorch dispatch
overhead, not a measurement. This script replaces the estimate: a stub
encoder of the spec's "Starting architecture" shape (d=128, 3 layers, 4
heads) timed on the two paths the real Policy (tickets 04-06) will exercise
per episode:

**Acting path (latency-bound).** One encoder forward per decision epoch
(spec.md decision 6: "one encoder pass per decision epoch, then m cheap head
passes") — ``--decisions-per-episode`` (default 400) sequential batch=1
forwards over a ``--client-tokens + --vehicle-tokens + 1`` (global) token
sequence. Hundreds of tiny forwards is where CPU kernel-launch overhead can
beat a GPU's per-launch latency, so this path is measured, not assumed.

**Learning path (throughput-bound).** One batch per episode, ``--k-passes``
shuffled minibatch passes over it (spec.md decision 9), each a real
forward + backward + Adam step against a dummy per-token regression target —
a stand-in for the real Q-head loss ticket 06 will build.

Usage (mini shape, fast)::

    uv run python scripts/benchmark_neural_stub.py --decisions-per-episode 20

Real-config shape (the numbers that go in the ticket's Comments and
spec.md's budget table)::

    uv run python scripts/benchmark_neural_stub.py

Both CPU and CUDA run by default when CUDA is available; ``--cpu-only`` skips
CUDA. Needs the optional ``neural`` extra: ``uv sync --extra neural``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

# stdvrp.simulation.episode imports stdvrp.policies.monte_carlo, which (via
# feature_extraction) imports stdvrp.simulation.state — a pre-existing
# circular import (independent of this ticket) that only resolves if
# stdvrp.simulation finishes initializing first. Importing it before
# stdvrp.policies avoids the landmine for whoever runs this script directly.
import stdvrp.simulation  # noqa: F401
from stdvrp.policies.torch_support import TORCH_INSTALL_HINT

try:
    import torch
    from torch import nn
except ImportError:
    print(f"scripts/benchmark_neural_stub.py needs torch: {TORCH_INSTALL_HINT}", file=sys.stderr)
    raise SystemExit(1) from None

# spec.md's target shape: ~150 pending Clients + up to ~8 vehicles + 1 global
# summary token (decisions 1/4 and the "Starting architecture" row).
DEFAULT_CLIENT_TOKENS = 150
DEFAULT_VEHICLE_TOKENS = 8
GLOBAL_TOKENS = 1


def build_stub_network(
    d_model: int, n_layers: int, n_heads: int, dim_feedforward: int | None = None
) -> nn.Module:
    """A minimal encoder + per-token Q head of the target shape — nothing else.

    Not the real tokenizer or network (tickets 04/05); this exists only to be
    timed, on the same shape the real thing will use.
    """
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=dim_feedforward or 4 * d_model,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
    q_head = nn.Linear(d_model, 1)
    return nn.Sequential(encoder, q_head)


@dataclass(frozen=True)
class PathTiming:
    """Wall-clock for one measured path, plus what it scales to per episode."""

    total_seconds: float
    unit_count: int

    @property
    def per_unit(self) -> float:
        return self.total_seconds / self.unit_count if self.unit_count else 0.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_acting_path(
    model: nn.Module,
    device: torch.device,
    *,
    seq_len: int,
    d_model: int,
    decisions_per_episode: int,
    warmup: int,
) -> PathTiming:
    """``decisions_per_episode`` sequential batch=1 encoder forwards (no grad).

    Token tensors are pre-generated outside the timed region: a real decision
    epoch's tokens come from the tokenizer (ticket 04), not from ``randn``, so
    random-tensor generation is not part of the cost being measured.
    """
    model.eval()
    warmup_tokens = [torch.randn(1, seq_len, d_model, device=device) for _ in range(warmup)]
    timed_tokens = [
        torch.randn(1, seq_len, d_model, device=device) for _ in range(decisions_per_episode)
    ]
    with torch.no_grad():
        for tokens in warmup_tokens:
            model(tokens)
        _sync(device)
        started = time.perf_counter()
        for tokens in timed_tokens:
            model(tokens)
        _sync(device)
        elapsed = time.perf_counter() - started
    return PathTiming(elapsed, decisions_per_episode)


def time_learning_path(
    model: nn.Module,
    device: torch.device,
    *,
    seq_len: int,
    d_model: int,
    decisions_per_episode: int,
    minibatch_size: int,
    k_passes: int,
    warmup: int,
) -> PathTiming:
    """``k_passes`` shuffled minibatch passes over one episode's collected batch.

    A stand-in for the real learn() step: the whole episode's decisions batched
    once, K on-policy passes over shuffled minibatches, then discarded (spec.md
    decision 9) — real forward + backward + Adam step against a dummy per-token
    regression target.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    episode_tokens = torch.randn(decisions_per_episode, seq_len, d_model, device=device)
    episode_targets = torch.randn(decisions_per_episode, seq_len, device=device)

    def one_pass() -> None:
        order = torch.randperm(decisions_per_episode, device=device)
        for start in range(0, decisions_per_episode, minibatch_size):
            batch_idx = order[start : start + minibatch_size]
            optimizer.zero_grad()
            prediction = model(episode_tokens[batch_idx]).squeeze(-1)
            loss = nn.functional.mse_loss(prediction, episode_targets[batch_idx])
            loss.backward()
            optimizer.step()

    for _ in range(warmup):
        one_pass()
    _sync(device)
    started = time.perf_counter()
    for _ in range(k_passes):
        one_pass()
    _sync(device)
    elapsed = time.perf_counter() - started
    # One episode's learning step, however many passes it took.
    return PathTiming(elapsed, 1)


def run_on_device(
    device_name: str,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    client_tokens: int,
    vehicle_tokens: int,
    decisions_per_episode: int,
    minibatch_size: int,
    k_passes: int,
    warmup: int,
) -> None:
    device = torch.device(device_name)
    seq_len = client_tokens + vehicle_tokens + GLOBAL_TOKENS
    model = build_stub_network(d_model, n_layers, n_heads).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    acting = time_acting_path(
        model,
        device,
        seq_len=seq_len,
        d_model=d_model,
        decisions_per_episode=decisions_per_episode,
        warmup=warmup,
    )
    learning = time_learning_path(
        model,
        device,
        seq_len=seq_len,
        d_model=d_model,
        decisions_per_episode=decisions_per_episode,
        minibatch_size=minibatch_size,
        k_passes=k_passes,
        warmup=max(1, warmup // k_passes) if k_passes else 0,
    )

    minibatches_per_pass = (decisions_per_episode + minibatch_size - 1) // minibatch_size
    print(f"\n{device_name} (seq_len={seq_len}, params={param_count:,})")
    _print_table(
        [
            (
                "acting (encoder fwd)",
                f"{acting.per_unit * 1000:8.3f} ms/decision   "
                f"{acting.per_unit * decisions_per_episode:8.3f} s/ep "
                f"over {decisions_per_episode} decisions",
            ),
            (
                "learning (K passes)",
                f"{learning.total_seconds:8.3f} s/ep "
                f"({k_passes} passes x {minibatches_per_pass} minibatches)",
            ),
        ]
    )
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        print(f"  cuda memory allocated (this stub model): {allocated / 1e6:.1f} MB")


def _print_table(rows: list[tuple[str, str]]) -> None:
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label.ljust(width)}  {value}")


EPISODE_POOL_NOTE = """
EpisodePool interaction (not measured here, documented per the ticket):
  Each EpisodePool worker holds its own 8.0 GB resident world (ticket 08,
  simulation-performance), so 32 GB RAM caps worker_count at ~3 regardless of
  GPU. A CUDA device context is a separate, roughly 500 MB-1 GB fixed cost per
  process that touches it (this stub model itself is a few MB, see above) --
  3 worker processes each opening their own CUDA context on an 8 GB laptop GPU
  is a second, independent ceiling. Evaluation parallelism (workers) and a
  single shared GPU do not compose for free; the acting-path numbers above
  answer only "CPU or CUDA per decision", not "how many workers can share one
  GPU".
""".strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--client-tokens", type=int, default=DEFAULT_CLIENT_TOKENS)
    parser.add_argument("--vehicle-tokens", type=int, default=DEFAULT_VEHICLE_TOKENS)
    parser.add_argument(
        "--decisions-per-episode",
        type=int,
        default=400,
        help="sequential acting-path forwards per simulated episode (spec.md's ~400 estimate)",
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=32, help="learning-path minibatch size"
    )
    parser.add_argument(
        "--k-passes",
        type=int,
        default=4,
        help="shuffled minibatch passes over one episode's batch (spec.md decision 9's K)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="untimed warmup iterations")
    parser.add_argument(
        "--cpu-only", action="store_true", help="skip CUDA even if torch reports it available"
    )
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    devices = ["cpu"]
    if not args.cpu_only and torch.cuda.is_available():
        devices.append("cuda")

    print(
        f"stub network: d_model={args.d_model} n_layers={args.n_layers} n_heads={args.n_heads}",
        flush=True,
    )
    for device_name in devices:
        run_on_device(
            device_name,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            client_tokens=args.client_tokens,
            vehicle_tokens=args.vehicle_tokens,
            decisions_per_episode=args.decisions_per_episode,
            minibatch_size=args.minibatch_size,
            k_passes=args.k_passes,
            warmup=args.warmup,
        )
    if "cuda" not in devices and not args.cpu_only:
        print("\n(CUDA not available on this machine; ran CPU only)")
    print(f"\n{EPISODE_POOL_NOTE}")


if __name__ == "__main__":
    main()
