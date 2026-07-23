# 05 — Vectorized feature extraction

**What to build:** Rebuild `_extract_state_action_features` (66% of episode
time) and `_extract_general_state_features` + `_classify_delayed_clients` as
vectorized numpy computations over the ticket-04 geometry matrices, extracted
from `MonteCarloPolicy` into a cohesive concrete class (no new seam —
ADR-0002). Candidate Q evaluation becomes one matrix–vector product per
vehicle instead of per-candidate Python loops.

**Blocked by:** 04.

**Status:** open

- [ ] Feature extraction lives in a concrete collaborator (working name
      `FeatureExtractor`) with no hidden instance-state handoff: the
      `X_general_state`/`X_state_action`/`possible_actions` mutable-attribute
      coupling inside `MonteCarloPolicy` is dissolved; data flows through
      arguments and return values.
- [ ] The **duplicate-append quirk is reproduced faithfully** (spec decision):
      the inflated `vehicle_to_clients` semantics — per-remaining-vehicle
      appends with evolving travel times — are expressed as a deterministic
      vectorized construction with the same contents and the same effect on
      `delayed_clients` and the `future_delay` feature.
- [ ] The permanently-zero feature and all normalization literals stay exactly
      as documented (W keeps 19 components; stored weight vectors remain
      valid).
- [ ] Candidate evaluation batches all candidates of a vehicle into one
      feature matrix and a single `X @ W`; the argmin tie-break (first
      candidate in iteration order wins) is preserved exactly.
- [ ] Tier 2 gate (float sums reorder): fixture self-golden within
      `rtol=1e-9` **and** the statistical golden gate green; the ticket
      Comments state precisely which reductions reordered.
- [ ] Benchmark note in Comments: episode throughput before/after and new
      profile top-10.
