# 05 — The network, the Q head and the myopic warm start

**What to build:** The transformer encoder over the ticket-04 tokens, the Q head
that scores (vehicle, Client) pairs, and the warm start that makes the untrained
network a *sensible* policy rather than a random one.

**Blocked by:** 04

**Status:** resolved

## Shape

- Encoder over the concatenated token set (clients + vehicles + global), with a
  learned type embedding per token kind. Starting point: **d=128, 3 layers,
  4 heads (~200k params)**, tuned on the **evaluation** seeds only — never on
  `test_seeds`.
- Head: `Q(vehicle_embedding_i, client_embedding_j, claimed_ij)`. It takes
  `claimed` as an input so the encoder runs **once per decision epoch** and the
  per-vehicle sweep is cheap (ticket 06).
- Output is a **cost** to minimize, matching the baseline's `argmin` convention.
  Do not silently flip the sign somewhere and make everything else confusing.

## The warm start (this is the load-bearing part)

Dropping the candidate set (ticket 06) means a randomly-initialized network
scores ~150 Clients with no prior — the farthest Client is as likely as the
nearest. The linear baseline never had this problem because
`_select_vehicle_possible_actions` handed it a nearest-first shortlist.

- [x] Initialize so that untrained `Q(i, j) ≈ minutes_from_vehicle_i_to_j`.
      This is `docs/research/rl-methodology-for-stdvrp.md` **F2**'s own
      recommendation ("a myopic warm start … would be strictly better and costs
      nothing"), and it is an **initialization, not a feature** — the value is
      not fed in as an input, the weights merely start somewhere sensible.
- [x] Verify it: the untrained network's greedy policy must be
      *nearest-feasible-Client*, checked directly against the geometry over
      random states.

**Consequence, and it is a good one:** the null model of Gate A is therefore a
nearest-neighbour policy, not a random one. "It learns" comes to mean "it beats
going to the nearest Client", which is a claim with content. Do not weaken the
warm start to make the gate easier — that would be gaming the null.

## Reproducibility

- [x] The network's init draws from an **injected** generator, never a global —
      the same discipline ticket 13 of `generic-stdvrp-refactor` established for
      the Episode streams. Gate A needs 3 independent init seeds and they must
      be reproducible.
- [x] Determinism: with a fixed init seed and fixed input, two forward passes
      agree bit-for-bit on the configured device. If CUDA cannot deliver that
      without `torch.use_deterministic_algorithms(True)`, set it and say what it
      costs.

## Acceptance

- [x] Untrained-network policy == nearest-feasible-Client, asserted.
- [x] Init is reproducible from an injected seed.
- [x] Predicted self-golden diff: **zero.**

## Comments

Resolved. `src/stdvrp/policies/network.py` — two `nn.Module`s: `TokenEncoder`
(runs once per decision epoch: `Tokens` -> `Embeddings`) and `QHead` (runs
once per vehicle, scoring every pending Client — a **cost to minimize**,
matching the baseline's `argmin` convention, never flipped).

**The variable-fleet-size problem, solved before the warm start could even be
attempted:** ticket 04's `client_tokens` row is `3 + 2*m` wide, and `m` varies
across configs and across `ExperimentConfig.test_vehicle_counts` within one
trained network's own evaluation sweep — a plain `nn.Linear(3+2*m, d_model)`
cannot have a fixed weight shape under that. Solved by giving the per-vehicle
"arc" pair (`minutes_from_vehicle[v]`, `path_length_from_vehicle[v]`) its own
`nn.Linear(2, d_model)`, weight-shared across `v` (parameter count independent
of `m`), applied once for all `(client, vehicle)` pairs and kept **outside**
the shared self-attention transformer entirely — cheap (`O(n_pending*m)`
linear ops), and, as it turned out, exactly what makes the warm start below
*exact* rather than approximate (an un-mixed pathway is trivial to reconstruct
a specific raw fact from; a self-attention-mixed one is not).

**The warm start is bit-exact, not statistical**, verified against real
tokenizer/geometry output before the design was finalized (see the module
docstring's numbered construction): (1) zero-initialising
`self_attn.out_proj`/`linear2` (weight and bias) makes every
`nn.TransformerEncoderLayer` an exact identity at init — a `Linear` with zero
weight *and* zero bias is a constant-zero function of any finite input,
regardless of what its own upstream `LayerNorm`/`in_proj` computed, so each
residual branch contributes exactly zero; (2) `arc_embed`'s row 0 is hand-set
to reconstruct `minutes_from_vehicle` exactly; (3)/(4) `QHead`'s two layers are
hand-set so `Q` reads only that one reconstructed value at init. A prototype
script (`torch.manual_seed`, no repo dependency) confirmed both the identity
claim and the exact reconstruction empirically before any production code was
written, since a plausible-sounding residual argument is exactly the kind of
thing worth checking rather than trusting.

**A real bug caught by the same prototyping, before it shipped:** the first
draft zeroed `QHead.layer1` *entirely* except its one reconstruction row,
reasoning that the "background" hidden units contribute nothing at init and
can stay dormant. That is a permanent deadlock, not a transient one — with a
zero weight row, `hidden[row]` is identically zero for every input, so
`layer2`'s matching column has zero gradient forever (its gradient is
`hidden[row] * dL/dQ`, and `dL/dQ` never rescues an operand that is exactly
zero by construction), which means `layer1`'s row never receives gradient
either. A one-off backward-pass check surfaced this before it became a
training-time surprise months later. Fixed by giving those background rows
ordinary Xavier-random weights instead of zero: `hidden[row]` is then a real,
nonzero, input-dependent value, so `layer2`'s column *does* get a nonzero
gradient from step one (its own gradient formula never depended on its current
value), and `layer1`'s row unlocks the following step once that column moves
off zero — the same mechanism as zero-gamma residual-block init in ResNet-
style networks. `tests/unit/test_network.py::TestQHeadBackgroundUnitsAreTrainable`
is a regression guard for exactly this: the warm-start test alone (which only
checks values/argmin *at init*) would never have caught it, since the bug is
invisible until the first training step.

**Determinism:** `dropout=0.0` throughout (no config knob for it exists yet,
and none of tickets 04-06 ask for one), so the network's forward pass has no
stochastic op at all — two calls on the same instance agree bit-for-bit on CPU
with no special handling. CUDA's kernel-selection nondeterminism (attention-
backend choice, reduction order) is a separate, real concern the ticket calls
out by name; `torch_support.resolve_device("cuda")` now calls
`torch.use_deterministic_algorithms(True)` — a **global** torch setting, so
the cost is paid once per process rather than per network, and documented in
that module's docstring: some ops fall back to slower deterministic kernels,
and any future op with no deterministic CUDA implementation raises
`RuntimeError` (a loud failure, not silent nondeterminism) instead of running
anyway. Not exercised against real CUDA hardware — none was available on this
machine; recorded as an assumption rather than papered over.
`tests/unit/test_torch_support.py`'s two new tests assert the flag flips
correctly for `"cuda"` and stays untouched for `"cpu"`, both safely on
CPU-only hardware (`torch.device("cuda")` never touches a GPU by itself).

**Reproducibility:** every learned weight is drawn from an injected
`init_rng: np.random.Generator` via a small Xavier-uniform-via-numpy helper,
copied into the `nn.Parameter` — never from torch's own global default
generator, which `nn.Linear.reset_parameters()` would otherwise silently
consume. Biases are always zero (a bias's own gradient never depends on its
current value, so this costs nothing and needs no extra draws).
`tests/unit/test_network.py::TestReproducibility` asserts two fresh
same-seed constructions are parameter- and output-bit-identical, and that
different seeds diverge (guards against an accidental fallback to the global
generator going unnoticed).

**Out of scope, deferred to ticket 06 as planned:** scoring the depot as a
candidate action. Ticket 05's `QHead` signature is `(vehicle_embedding,
client_embeddings, claimed)` — Client scoring only, per spec.md's own
pseudocode; the depot ("always legal", spec.md decision 4) is ticket 06's
action-set/mask concern, not this ticket's.

**Predicted self-golden diff: zero, verified**
(`tests/test_self_golden.py` + `tests/test_world_cache_self_golden.py`, 7/7
passed) — two new modules (`network.py`, `test_network.py`) plus a
`torch_support.py` addition that only activates for `device="cuda"`; nothing
in the linear baseline's execution path is touched. Full suite green both
with torch installed (4089 passed, 3 deselected — the previously-`neural`-
skipped tests now run) and with it uninstalled (verified via `uv pip
uninstall torch`, matching ticket 03's own discipline: `test_network.py`
skips as a whole module at collection, `test_torch_support.py`'s non-neural
tests still pass). `mypy` clean on `stdvrp.*`, `ruff check`/`ruff format
--check` clean on every touched file.

`/code-review` (both axes, against `98260e8`): no hard standards violations.
Two Standards judgement calls, both applied: (1) `TokenEncoder.forward` used a
bare `m` for vehicle count where the rest of this package (`tokenizer.py`,
`monte_carlo.py`) always spells out `number_vehicles` — renamed; (2)
`QHead._arc_dim0_index`'s `d_model + d_model` arithmetic and
`TokenEncoder.forward`'s concatenation order were two independently-written
pieces of the same layout knowledge, tied together only by parallel prose
comments — extracted into one shared `_arc_dim0_index(d_model)` function both
classes call, so the offset formula exists in exactly one place (the
concatenation *order* itself still needs a human to keep in sync, but
`TestWarmStart` is an end-to-end check across both classes that would catch a
future mismatch immediately, not silently).

Spec axis found two real gaps, both fixed: (1) `TestWarmStart` compared `Q`
against `tokens.client_tokens` — the tokenizer's *own* output — rather than
"checked directly against the geometry" as the ticket's text literally asks;
a bug shared between `tokenizer.py` and `network.py` (e.g. both transposing
the same field) would not have been caught. Rewritten to recompute
`minutes_from_vehicle` independently from `EpisodeGeometry.average_minutes_rows`/
`.column_positions`, with the tolerance tightened from `atol=1e-4` (looser
than the actual float32-cast error by four orders of magnitude, observed
~1e-8) to `1e-5`. (2) `claimed` was never exercised with a nonzero value or
checked to actually participate in the computation graph — added
`TestClaimedIsWired`, which caught a real bug in the *test* (not the
production code) on first write: the naive version asserted `claimed.grad`
is nonzero after a single backward pass at init, which is false by the same
one-step-delayed-unlock mechanism already documented for `layer1`'s
background rows (`layer2`'s background columns are exactly zero until an
optimizer step moves them, so no gradient reaches *any* input — vehicle
embedding, client context, or claimed — through those rows on the very first
call). Corrected to assert zero gradient at init, then a real optimizer step,
then nonzero gradient — stress-tested across 30 seeds outside the suite before
landing. Reverified after both fixes: `mypy`/`ruff check`/`ruff format --check`
clean, `tests/unit/test_network.py` + `tests/unit/test_torch_support.py`
(28/28) and self-golden (7/7) all green.
