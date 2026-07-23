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

## Addendum (2026-07-23, ticket 09): Trainer deviations from the legacy loops

Two documented deviations in the ported Trainer (`src/stdvrp/training/trainer.py`), both outside the golden-master path:

- **Best-W fallback.** When fewer training episodes than `test_frequency` run, no evaluation block ever executes; the legacy would then run its final test with `Best_W = []` and crash. The Trainer runs the final test with the last trained W instead. Any run whose iteration count reaches `test_frequency` is unaffected.
- **Best-cost sentinel.** The legacy initializes `Q_pred = 1e11`; the Trainer uses `inf`. Behavior differs only if a mean evaluation cost exceeded 1e11, where the legacy would keep `Best_W = []` and crash in the final test.

Also noted: the legacy `test_model` report includes three mean-time metrics (`mean_delay_time`, `mean_earliness_time`, `mean_overtime`) that ticket 07's Model port does not expose — the golden master does not pin them. The Trainer reports the nine golden-pinned metrics; resurrect the other three deliberately (with Model support and tests) if a study needs them.

## Phase-2 change log (ticket 12): deliberate, documented bug fixes

Each entry is a deliberate behavior change with its own test, applied after the user triaged the full candidate list (2026-07-23). Where a fix changes episode outcomes, the golden master is re-baselined from the new code (the exact legacy capture remains in git history) and the vs-legacy comparisons compensate exactly computable deltas.

1. **Single-observation speed std is 0.0, not NaN** (`_aggregate_speed_statistics`). A (Period, Link) seen on one day has a NaN pandas std (ddof=1); the legacy `dropna()` that should have removed those rows discarded its result, so `random.gauss(speed, nan)` produced NaN velocities. A single observation now means a deterministic speed. Multi-observation stds are bit-identical; golden outcomes are unaffected unless an episode traversed a NaN-std arc (the golden capture's finite costs show none did).
2. **The 420-540 std window blends like the other two** (`_build_speed_std_lookup`). The legacy computed the interpolation into a discarded row copy and stored the raw gap-filled std for minutes 421-539. The window now stores the blend of the (off-by-two, preserved) 418/542 endpoints, mirroring the 660-840 and 960-1080 branches. Changes every stochastic velocity drawn in that window on real multi-day data → golden re-baseline; the 44-copy characterization world is unaffected (all stds 0.0), so the vs-legacy gates stay bit-exact.

## Addendum (2026-07-23, ticket 14): the monolith left the working tree

`Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py` is deleted from the working tree; the `legacy-monolith` tag is now its only home and must stay frozen forever — the golden master's `legacy_sha256` (`bfbe14b5…`) pins the tag's content. The characterization venue survives the deletion: `tests/characterization_world.py` and `scripts/capture_golden_master.py` extract the script from the tag at run time (`git show legacy-monolith:<file>`), so the vs-legacy suites and the golden re-run keep working unchanged. CI fetches tags for this (`fetch-tags: true` in the checkout step); a tagless shallow clone fails those tests with instructions rather than skipping silently.
