---
status: accepted
---

# Array-oriented hot path behind concrete domain facades

Profiling (2026-07-23, mini fixture) showed ~82% of episode time in the
Policy's decision path — 66% in `_extract_state_action_features` alone, driven
by 1.13M per-episode `path_between` tuple-dict lookups — while the Model's
transition function is ~5%. To reach the simulation-performance effort's ≥10×
serial episode-throughput target in Python, the hot path's core becomes
**data-oriented**: dense numpy geometry matrices (travel minutes and lengths,
node × episode-client) built once per Episode, struct-of-arrays state on the
hot path, vectorized feature extraction and candidate selection, and candidate
Q evaluation as one matrix–vector product. The object-oriented elegance the
lab wants lives **at the boundary**: cohesive, concrete, domain-named classes
(geometry facade, feature extractor, cost ledger) encapsulate the arrays;
ADR-0002's three-seams rule stands — none of these become interfaces.

Behavior is protected by a tiered contract (spec, `.scratch/simulation-performance/`):
bit-exact per-seed equality with a pre-effort self-golden capture by default;
optimizations that inherently reorder float arithmetic validate against the
existing statistical golden gate plus a tight-tolerance fixture check, with the
reordering documented per ticket; deliberate behavior changes stay in separate
ticket-12-style tickets with their own re-baseline.

## Considered options

- **Pure domain objects** (Vehicle/Client/Route with behavior, loops over
  objects): rejected — maximally idiomatic OO, but Python object overhead in
  the hot loop cannot deliver the measured-speed objective; likely no faster
  than today.
- **Conservative optimization** (keep dicts/lists, memoize per epoch, hoist
  loop invariants): rejected as the end state — an estimated 2–5× that leaves
  the vectorization headroom on the table; its techniques are still used where
  they compose with the array core.
- **Compiled kernels first** (numba/cython before restructuring): rejected as
  the default — JIT/toolchain complexity before the data layout is right
  optimizes the wrong representation. Per the effort's "anything goes"
  decision, compiled kernels remain available tactically, justified by
  benchmark inside individual tickets.

## Consequences

- Hot-path code reads as array programs; reviewers should expect numpy idioms
  inside the facades and domain vocabulary outside them.
- Feature definitions (normalization literals, the permanently-zero feature,
  the duplicate-append quirk's inflated semantics) are **unchanged** — W keeps
  19 components and stored weight vectors stay valid; the quirk's faithful
  vectorization is a correctness requirement, not an accident.
- Last-bit float divergence from vectorized reductions is expected and
  gated statistically (tier 2); bit-exact debugging of tier-2 code compares
  against the tight-tolerance fixture check, not `==`.
- Episode construction gains a per-Episode matrix build (~tens of MB, ~ms);
  memory stays trivial next to the 907 MB path cache already resident.
- The Tier-1 self-golden gate (ticket 01) is **environment-guarded**: it asserts
  bit-exact `==` only when the running numpy version and platform match the
  capture's recorded fingerprint, and skips otherwise. numpy's `Generator`
  guarantees a reproducible integer stream per seed but not bit-identical float
  draws (the per-arc velocity `normal` draws are Ziggurat-based; numpy's own
  compatibility policy excludes cross-CPU/libm/version float identity). The gate
  is therefore live on the capture machine (where optimization tickets are
  worked) and inert elsewhere; an always-run test warns loudly when it is inert
  so a green CI never hides a dormant gate, and CI relies on the Tier-2
  statistical gate. To make Tier-1 live on another platform (e.g. the CI image),
  re-run `scripts/capture_self_golden.py` there. User-ratified 2026-07-23.
