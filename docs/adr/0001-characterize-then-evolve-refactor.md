---
status: accepted
---

# Characterize-then-evolve refactor, with two-phase RNG modernization

The legacy monolith (`Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`, ~6,580 lines, no tests) backs existing research results, so we refactor against golden-master characterization tests captured from the original script (frozen at git tag `legacy-monolith`) running on a small committed fixture sub-network with fixed seeds. Phase 1 is purely structural: the new code must consume randomness from the global `random`/`np.random` state in the exact same order so the golden master matches bit-for-bit — this deliberately preserves shadowed-method behavior and any latent bugs. Phase 2 then evolves deliberately: documented bug fixes and migration to injected `np.random.Generator` instances, each re-baselined via statistical regression (mean episode cost over N seeds within tolerance) because exact golden-master equality cannot survive RNG reordering.

## Considered options

- **Bug-for-bug forever**: rejected — permanently freezes accidental behavior (e.g., Python silently keeping only the last of 3 duplicate method definitions).
- **Clean redesign validated only by new tests**: rejected — loses the evidence that the refactor did not change research results.

## Consequences

- Phase 1 code may look deliberately unidiomatic in places (global-RNG call order preserved); do not "fix" this before Phase 2.
- Deleted experiment variants (~15 dead or shadowed methods) live in the `legacy-monolith` tag, not in the working tree; resurrect them deliberately as named strategies with their own tests.

## Addendum (2026-07-21, ticket 03): golden-master venue

The original plan ("golden master captured on a small committed fixture") proved impossible: the legacy hardcodes a 1,900-node client universe (`random.sample(range(1, 1900))`), a 60-client minimum, a vehicle-ratio table defined only for 150/250 mean clients, and aggregates over 88 speed files plus a 907 MB precomputed path file — no sub-megabyte fixture can satisfy that. Revised split: the committed mini fixture (45 nodes, day 601, <0.5 MB, regenerable via `scripts/make_fixture.py`) serves the config-driven new code's tests in CI; the exact golden master (ticket 04) runs the legacy on the full local dataset (or a large gitignored fixture from the same script), with the exact-equality test skipping when the data is absent. CI therefore guards structure and new-code behavior; exact legacy equality is verified locally where the data lives. All legacy data paths are relative, so ticket 04 can redirect them by running the script with a different working directory instead of editing code.
