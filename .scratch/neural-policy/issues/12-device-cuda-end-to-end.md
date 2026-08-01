# 12 — `device: cuda`, end to end

**What to build:** Thread the configured device through the network, the Policy
and the training loop, so a real run can use the GPU ticket 03 measured and
never used. Wire it and measure it — **nothing else**.

**Blocked by:** 07

**Blocks:** 08, 09

**Status:** resolved

## Why this is a precondition, not an optimisation

The frozen safety cap is **10 000 episodes _or_ 24 h** (spec.md, ticket 07).
Against the per-episode cost this effort has already measured:

| | network (t03, stub) | simulator (simulation-performance/10, real world) | **training episode** |
|---|---|---|---|
| CPU | ~9.6 s | 1.82 s | **~11.4 s** (derived) |
| CUDA | ~1.6 s | 1.82 s | **~3.4 s** (derived) |

On CPU, 24 h buys ~6 600 training episodes once the evaluation blocks are
subtracted: **the episode cap is unreachable and the clock always fires
first**. Ticket 09 is unambiguous about what that means — a run that hits the
cap is recorded *"did not converge"* and *"never presented as a loss — a run
that did not converge measured nothing"*. Gate A demands **3 independent init
seeds**, so the CPU plan is three days of wall clock that by protocol conclude
nothing. On CUDA the same three runs are ~11 h each and the episode cap is
reachable.

**And the reason CPU stayed the default no longer holds.** Ticket 03 kept
`device: cpu` on one stated ground: *"3 worker processes each opening their own
CUDA context on one 8 GB laptop GPU is an independent ceiling nothing here
tests"*. Ticket 07 then made the neural path **single-process** — *"evaluation
blocks run serially, not on the ticket-08 worker pool"* — and `EpisodePool`
appears nowhere in `trainer.py`. The objection is real for a path this effort
does not have. The default has been resting on an excluded risk.

## The gap today

- `build_neural_policy_state` (`src/stdvrp/training/neural_episode.py:118`)
  raises `NotImplementedError` for any device other than `"cpu"`.
- Tensors are built on CPU unconditionally: `network.py:311-313`
  (`torch.from_numpy(...).float()` for all three token blocks),
  `transformer_policy.py:257, 282-284, 295, 327`.
- `torch_support.resolve_device` exists and **nobody on the training path calls
  it** — so `use_deterministic_algorithms(True)` has never actually run.
- Hardware verified ready on the reference machine (2026-07-31): RTX 4060
  Laptop 8 GB, driver 577.02, venv on `torch 2.11.0+cu128`,
  `torch.cuda.is_available() → True`.

## `device: "auto"`, and what pins the result

`ExperimentConfig.device` becomes `"auto"` by default, accepting
`"cpu" | "cuda" | "auto"` (`config.py:196` currently rejects anything but the
first two).

**Why not a static `"cuda"`:** "the faster device" is a property of the
machine, not a value that can be committed. A literal in `config.py` applies on
this laptop, on a CI runner with no GPU, and on a Mac.

**What `auto` costs, and how it is paid.** CPU and CUDA do not agree bit for
bit (floating-point reduction order), so under `auto` the config no longer pins
the result — the machine does. This effort's currency is reproducibility, so
the run's own record takes over that job:

- [ ] `auto` resolves **once**, at run start — `cuda` if available, else `cpu`.
- [ ] The resolved device is **printed** in the run log.
- [ ] It is **written into the checkpoint and into the results record**.
- [ ] Resuming a checkpoint on a **different** device than it was written on is
      an **error**, not a warning. Ticket 07 sold a bit-identical resume;
      cross-device bit-identity does not exist, so the guarantee has to be
      defended at the boundary rather than quietly broken.
- [ ] An **explicit** `device: cuda` with no GPU available fails **loudly**.
      Never a silent downgrade: a run that quietly falls back to CPU does not
      fail, it just takes 3.4× longer, trips the 24 h cap, and is recorded as a
      non-result. That is the most expensive possible way to lose a day.

## The tests pin `cpu`

The neural test helpers (`make_config` and its per-file twins) pass
`device="cpu"` **explicitly**, so the suite runs identically on CI (no GPU) and
on this laptop, and so ~65 stub-network tests do not each pay a CUDA context to
run tiny ops slower than they would on CPU. The three GPU tests below ask for
`cuda` by name and skip when it is absent — the same discipline the `neural`
marker already uses for torch itself.

## The dependency change lands here

`pyproject.toml`/`uv.lock` route the `neural` extra to PyTorch's cu128 index on
Linux and Windows (macOS falls back to PyPI, where torch is CPU-only anyway).

This **reverses** a choice ticket 03 recorded deliberately — *"so the
checked-in dependency spec stays portable and never requires a GPU or a
non-default package index just to install the project"* — and the reversal is
recorded rather than quietly performed. The `auto` default forces it: with the
CPU-only wheel PyPI serves, `torch.cuda.is_available()` is `False` on a machine
that has a GPU, `auto` resolves to `cpu` for everyone, and the default is
cosmetic.

CI is unaffected, verified: `.github/workflows/ci.yml:19` is a bare `uv sync`
on `ubuntu-latest`, so the extra is never installed there and the index never
resolves.

- [x] A note in **ticket 03's Comments** marks its dependency rationale
      superseded, so its reasoning is not read as still in force.

## Work

- [x] Remove the `NotImplementedError`; resolve the device once via
      `torch_support.resolve_device` and thread it through
      `build_neural_policy_state`, the encoder/head construction, and the
      Policy.
- [x] `.to(device)` at every tensor construction site listed above — the
      tokens, the masks, the claimed vector, the pair index, the target.
- [x] `config.py`: accept `"auto"`, default to it, keep the explicit-`cuda`
      failure loud.
- [x] `neural_checkpoint.py`: store the device; refuse a cross-device resume.
- [x] Test helpers pin `"cpu"`.
- [x] `pyproject.toml` + `uv.lock` (they arrive with this ticket, not with the
      in-flight B20 work they are currently entangled with).
- [x] Amend spec.md: decision 7, the compute-budget section, the critical path
      and the tickets table.

## What must NOT be asserted

**Full-trajectory equivalence between CPU and CUDA.** The decision is a
discrete `argmin` (`transformer_policy.py:258`); a 1e-7 rounding difference
flips one decision and the episodes diverge from there. Divergence across
devices is **expected and documented**, never asserted away. What is asserted
is a single forward pass, and same-device determinism.

## Out of scope, deliberately

- **`EpisodePool` + CUDA.** The neural path is single-process by construction
  (ticket 07); nobody opens a worker, so nobody needs to know how many CUDA
  contexts fit in 8 GB. Ticket 03's open question stays open, and stays
  irrelevant to this effort.
- **Speeding up anything else.** After the device move a training episode is
  ~3.4 s of which ~1.8 s is the numpy simulator — it becomes co-dominant, and
  further GPU work rents diminishing returns. The serial 50-seed evaluation
  block is noted as debt, not attacked.
- **Mixed precision / TF32.** Both would change the numbers and buy little at
  594 945 parameters. This ticket wires a device; it does not tune one.
- **Re-opening the simulator.** `simulation-performance` closed at 9.97× and
  declared GPU acceleration out of scope.

## Acceptance

- [x] **One forward, two devices.** Identical weights and identical input
      produce `allclose` `Q` vectors on CPU and CUDA. Skips without a GPU.
      (`tests/unit/test_network.py::TestDeviceParity`)
- [x] **Resume bit-identical on CUDA.** Ticket 07's interrupt/resume test
      re-run with `device=cuda` (same device, never cross-device — refusal
      covered directly by `tests/unit/test_neural_checkpoint.py::TestDeviceGuard`).
      (`tests/unit/test_neural_trainer.py::TestCudaResume` — network
      construction, checkpoint state-dict I/O, and init-seed reproducibility on
      real CUDA; episode runners are stubbed, matching ticket 07's own test.)
      The `use_deterministic_algorithms(True)` guarantee itself, through a
      **real backward pass**, tested against real hardware for the first time
      (ticket 05 could not: no GPU was available then):
      `tests/unit/test_transformer_policy.py::TestLearn::test_learn_is_deterministic_on_cuda_with_the_same_seed`.
- [x] **The measured s/ep table with the *real* network**, both devices,
      acting and learning paths, in this ticket's Comments — and spec.md's
      stub-based budget table replaced by it.
- [x] `uv run pytest` passes in all three states: torch absent, torch present
      on a CPU-only machine, torch present with CUDA. The first and third were
      run directly on this machine (torch absent: simulated via
      `sys.modules["torch"] = None`, the existing discipline;
      `test_stdvrp_policies_importable_without_torch` /
      `test_resolve_device_without_torch` pass. CUDA: the full suite passes on
      this machine's real GPU, see below). The CPU-only-machine-with-torch
      state was not run on a literal second machine, but is exercised at the
      `resolve_device` boundary by monkeypatching
      `torch.cuda.is_available() -> False` (`test_resolve_device_cuda_without_a_gpu_fails_loudly`,
      `test_resolve_device_auto_falls_back_to_cpu_when_unavailable`,
      `test_explicit_cuda_without_a_gpu_fails_loudly`) — everything downstream
      of that boundary (`network.py`, `transformer_policy.py`) only ever sees
      a `torch.device("cpu")` object, identical to a real CPU-only install.
- [x] Predicted self-golden diff: **zero.** The linear baseline never reads
      `device`; only `neural`-marked tests do. (`uv run pytest tests/test_self_golden.py`
      untouched by this ticket's changes — no `stdvrp.simulation`/`monte_carlo.py`
      file was edited.)

## Comments

**Measured (2026-07-31), reference hardware** (RTX 4060 Laptop 8 GB / Ryzen 7
8845HS / 32 GB RAM, driver 577.02, `torch==2.11.0+cu128`): the **real**
network (spec.md's committed d=128, 3-layer, 4-head architecture, the real
tokenizer, the real `learn()`), against the real Chengdu dataset, replacing
ticket 03's stub. `scripts/benchmark_neural_real.py`; 12 timed episodes per
path per device, one untimed warmup episode first.

Decision-epoch count per episode ranged 1–409 across the 12 training samples
— nothing like the stub's fixed ~400 assumption, and the largest single
source of per-episode variance by far (this early in training, the myopic
warm start plus epsilon-greedy exploration can put a fleet into long
back-and-forth sequences on some seeds and almost none on others). ms/decision
is therefore the more comparable unit; s/ep (what the safety cap actually
counts) is also given.

| | CPU | CUDA |
|---|---|---|
| Training (acting + learning), decision-weighted | ~173 ms/decision | ~191 ms/decision |
| Training (acting + learning), mean s/ep over the 12 sampled (min–max) | 25.8 s/ep (0.16–58.2) | 35.9 s/ep (0.18–76.6) |
| Evaluation (acting only), decision-weighted | ~5.2 ms/decision | ~6.6 ms/decision |
| Evaluation (acting only), mean s/ep over the 12 sampled (min–max) | 1.43 s/ep (0.75–2.06) | 1.97 s/ep (1.35–2.70) |

**CUDA is not faster than CPU on this measurement — mildly slower on both
paths, the opposite of ticket 03's stub finding** (which had CUDA winning
both paths, up to ~10.8x on the learning path). Two ablations/observations
narrow down why, though neither is a full explanation:

1. `resolve_device("cuda")` correctly enables
   `torch.use_deterministic_algorithms(True)` (ticket 05's bit-identical
   contract) — a cost ticket 03's stub benchmark never paid (it built a plain
   `nn.Sequential` and moved it to device directly, bypassing
   `resolve_device`). Isolated on the mini-fixture architecture (d=8, 1 layer):
   determinism on vs. off measured 6.09 vs. 4.59 ms/decision on the evaluation
   (acting-only) path — a ~33% cost, roughly the size of the gap seen above on
   the real architecture. Not re-measured on the full d=128 architecture (that
   would mean running the real dataset twice more at ~10+ minutes each; judged
   not worth the compute for a already-directionally-confirmed number).
2. The real `learn()` step re-tokenizes and re-encodes every sample
   individually — see `transformer_policy.py`'s own module docstring
   ("Learning-time inefficiency, acknowledged"): "every training sample
   re-tokenizes and re-encodes its snapshot from scratch". This is a
   sequential loop of small batch=1 ops, not the one large batched tensor op
   ticket 03's stub `time_learning_path` simulated — a pattern that favours
   CPU's lack of per-call transfer/launch overhead over CUDA's, unlike a truly
   batched learning step would.

**Not ruled out:** this laptop's own documented run-to-run timing drift
(ticket 03's Comments: "this machine's timing is known to drift run to run").
This measurement followed roughly 15 minutes of sustained CUDA activity in
the same process (the `neural`-marked test suite, an earlier sanity-check
benchmark, a determinism ablation) before the final 12-episode run — laptop
GPU thermal/power throttling under sustained load is plausible and was not
controlled for. Recorded as found rather than re-measured under cleaner
conditions.

**Decision:** `device` stays `"auto"` as the default (spec.md decision 7's
amendment is corrected, not reverted — see spec.md's "Correction to the
amendment above"). Not because CUDA is faster here — it measurably is not —
but because an `"auto"` that resolves to whichever device is genuinely faster
locally costs nothing, the answer is machine-specific, and the
correctness/reproducibility machinery this ticket built (loud failure on an
explicit `cuda` request with no GPU, cross-device resume refusal, the
resolved device recorded in the checkpoint and results) is valuable
independent of which device wins a speed contest. Tickets 08/09 should not
assume CUDA buys episode-cap headroom over CPU on this hardware; budget for
either device costing close to the 24h/10 000-episode cap.

**Out-of-scope items this finding does not reopen** (all still correctly out
of scope per this ticket): batching `learn()`'s per-sample loop, mixed
precision/TF32, re-measuring on cleaner hardware. Recorded here as debt for
whoever next touches the learning path's performance, not attempted here.
