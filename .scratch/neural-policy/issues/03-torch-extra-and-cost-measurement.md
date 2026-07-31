# 03 — torch as an optional extra, and what it actually costs

**What to build:** Add torch as an **optional** dependency that the existing
suite never needs, and then *measure* the per-episode cost of a stub network so
the effort's compute budget stops being a guess.

**Blocked by:** —

**Status:** resolved

## Why optional

torch is not installed today (verified: numpy 2.4.6, no torch / jax / sklearn).
Making it mandatory would put a ~200 MB (CPU) or ~2.5 GB (CUDA) wheel in front
of the 4039-test suite and CI, which run the linear baseline and never touch a
network.

- [x] `pyproject.toml`: `[project.optional-dependencies] neural = ["torch>=2.4"]`
- [x] The new policy module imports torch **lazily**, inside the module that
      needs it — never at package import time. `stdvrp.policies` must stay
      importable without torch.
- [x] `ExperimentConfig` gains `device: str` (default `"cpu"`) and the neural
      hyperparameters. `from_yaml`'s explicit key validation rejects unknown
      keys and requires every known one — **every shipped YAML needs the new
      fields**, including the mini fixture and the sweep configs.
- [x] Tests that need torch skip cleanly when it is absent (a marker, like the
      existing `golden` marker).

## The measurement (the actual point of this ticket)

spec.md's budget rests on an estimate of ~10 s/ep training, ~3.3 s/ep
evaluation. That is a napkin calculation about PyTorch dispatch overhead, not a
measurement. Correct it **before** committing to long runs.

- [x] Stub network of the target shape (d=128, 3 layers, 4 heads, ~150 client
      tokens + 8 vehicle + 1 global). Measure, on the real dataset:
      - one encoder forward (the acting path — ~400 sequential tiny passes per
        episode, which is **latency-bound**, not throughput-bound);
      - one training episode's learning step (K passes × minibatches — the
        batched path, which is throughput-bound);
      - both on **CPU and CUDA** (RTX 4060 Laptop, 8 GB).
- [x] Expect the answer to be **hybrid or CPU-favouring**: the acting loop is
      hundreds of 158-token forwards where kernel-launch overhead can dominate,
      while the learning step is a real batch. If CUDA wins only on the learning
      step, say so and configure accordingly. **Measured: it did not — CUDA won
      both paths on this hardware. See Comments.**
- [x] Note the interaction with `EpisodePool`: each spawned worker holds an
      8 GB world, so 32 GB RAM caps workers at ~3 — and 3 CUDA contexts on an
      8 GB laptop GPU is a separate problem. Evaluation parallelism and GPU do
      not compose for free.
- [x] **Update spec.md's budget table with the measured numbers.**

## Acceptance

- [x] `uv run pytest` passes with torch **not** installed.
- [x] `uv run pytest` passes with torch installed.
- [x] A measured s/ep table lands in this ticket's Comments and spec.md is
      corrected against it.
- [x] Predicted self-golden diff: **zero.**

## Comments

**Measured (2026-07-30), reference hardware exactly** (RTX 4060 Laptop 8 GB /
Ryzen 7 8845HS / 32 GB RAM / Windows), torch 2.6.0+cu124, driver reporting
CUDA 12.9. Stub network at spec.md's exact target shape —
`nn.TransformerEncoderLayer` d_model=128, nhead=4, dim_feedforward=512 (the
standard 4x ratio), 3 layers, `batch_first=True` — over seq_len=159 (150
client tokens + 8 vehicle + 1 global), 594,945 params
(`scripts/benchmark_neural_stub.py`; three independent runs to check the
result wasn't warmup noise — this machine's timing is known to drift run to
run):

| Path | CPU | CUDA | CUDA speedup |
|---|---|---|---|
| Acting (400 sequential batch=1 forwards) | 1.00-1.08 s/ep (2.5-2.7 ms/decision) | 0.75-0.89 s/ep (1.9-2.2 ms/decision) | ~1.2-1.3x |
| Learning (K=4 passes × 13 minibatches of 32) | 8.48-8.64 s/ep | 0.78-0.81 s/ep | ~10.8x |

**Reproducing the CUDA numbers:** `uv sync --extra neural` alone resolves
plain PyPI `torch>=2.4`, which on Windows is the **CPU-only** build (`uv.lock`
pins `torch==2.13.0`, no `+cuXXX` suffix) — deliberately, so the checked-in
dependency spec stays portable and never requires a GPU or a non-default
package index just to install the project (spec.md decision 7: "CPU by
default"). The CUDA measurement above used a separate, local-only step not
recorded in `pyproject.toml`/`uv.lock`: `uv pip install torch --reinstall
--index-url https://download.pytorch.org/whl/cu124 --python .venv`, which
installed `torch==2.6.0+cu124` into the venv directly. Whoever re-runs this
measurement needs that same manual step first.
| **Training episode (acting + learning)** | **~9.5-9.7 s/ep** | **~1.5-1.7 s/ep** | **~6x** |
| **Evaluation episode (acting only)** | **~1.0-1.08 s/ep** | **~0.75-0.89 s/ep** | **~1.2-1.3x** |

**Correction to this ticket's own prediction:** it expected "hybrid or
CPU-favouring" — kernel-launch overhead dominating the acting path's batch=1
forwards, CUDA winning only the batched learning step. Measured: **CUDA won
both paths** on this hardware. 159 tokens × 128 dim × 3 layers is enough real
matmul work per decision that launch overhead does not dominate here, even at
batch=1. Recorded as found, not forced into the anticipated shape — the
learning-path result (~10.8x) is still the larger and more reliable effect.

**`device: cpu` stays the config default** (spec.md decision 7 is a
structural choice; this measurement doesn't overturn it): the benchmark above
is single-process and never combines a GPU with live `EpisodePool` workers.
An operator who wants CUDA on this class of hardware can pass `device: cuda`
explicitly; whether to change the *default* is a question for whoever
measures the worker+CUDA interaction below, not this ticket.

**`EpisodePool` interaction (documented per the ticket, not itself measured
here):** each spawned worker holds its own 8.0 GB resident world (ticket 08,
simulation-performance), so this machine's 32 GB RAM caps `worker_count` at
~3 regardless of GPU. A CUDA context is a separate, roughly 500 MB-1 GB fixed
cost per process that touches it — this stub model itself allocates only
21.8 MB, so the model is never the constraint. 3 worker processes each
opening their own CUDA context on one 8 GB laptop GPU is an independent
ceiling nothing here tests: evaluation parallelism (workers) and a single
shared GPU do not compose for free. The table above answers "CPU or CUDA for
one decision", not "how many workers can share one GPU" — a follow-on
question for whoever wires `EpisodePool` + `device: cuda` together.

**Also found, not fixed (out of scope for this ticket):** `stdvrp.policies`
has a pre-existing circular import with `stdvrp.simulation`
(`stdvrp.simulation.episode` imports `MonteCarloPolicy`;
`stdvrp.policies.feature_extraction` imports `stdvrp.simulation.state`) that
only resolves if `stdvrp.simulation` finishes initializing first. Any fresh
process importing `stdvrp.policies` (or a submodule) as its *first* `stdvrp`
import fails with `ImportError: cannot import name 'MonteCarloPolicy' from
partially initialized module...`. Reproduces on an unmodified checkout
(`uv run python -c "from stdvrp.policies import Policy"` in a fresh
interpreter), so it predates this ticket and is not this ticket's to fix.
Worked around locally by importing `stdvrp.simulation` first, with a comment
recording why, in `tests/unit/test_torch_support.py` and
`scripts/benchmark_neural_stub.py` — the first two things in the tree to
import `stdvrp.policies` without something else having already pulled in
`stdvrp.simulation`. Whoever next restructures `policies`/`simulation` should
know this landmine exists.

**Actual params (594,945) vs. spec.md's "~200k" ballpark:** at
`dim_feedforward=4*d_model` (the standard Transformer ratio), the three-layer
encoder alone is already ~595k params — roughly 2-3x the spec's own estimate.
Recorded for whoever builds the real network in ticket 05, who should not be
surprised by the actual parameter count at this shape.

**Predicted self-golden diff: zero, verified.** Nothing in this ticket
touches the linear baseline's execution path; `tests/test_self_golden.py`
passes unchanged.
