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
