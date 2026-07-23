# Golden master (legacy monolith, full Chengdu data)

`chengdu_full.json` pins the exact behavior of the untouched legacy script
(`Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`, frozen at the
`legacy-monolith` tag; since ticket 14 the capture and tests extract it from
that tag — it is no longer in the working tree) so the Phase 1 refactor can
prove it changed nothing (ADR-0001). It was captured on the **full local dataset** — the legacy cannot run
on the committed mini fixture (see the ADR-0001 addendum) — by:

```
uv run python scripts/capture_golden_master.py
```

## What is stored

- `meta` — sha256 of the legacy script, byte sizes of all 92 consumed data
  files, and the Python/numpy/pandas/networkx versions used at capture time.
- `protocol` — every parameter and seed list of the capture run. The re-run test
  reads the protocol from this file, so the stored values and the procedure that
  produced them cannot drift apart.
- `training.w_trajectory` — the linear-approximation weight vector W after each
  training episode (full float precision): the "W trajectory for a small
  training run".
- `training.eval_costs` — per-seed greedy-policy episode costs with the
  post-training W (the legacy's `cont % test_frequency` evaluation block).
- `test` — per-seed test episodes keyed by the `actions` setting: total cost,
  the four cost components (distance, delay, earliness, overtime), plus tau,
  state count and delay/earliness client counts.

## Why subsets of the legacy's loops are exact

Every legacy episode begins with `client_generator_function(seed)` (which calls
`random.seed(seed)`) followed by `np.random.seed(seed)`, and test/eval episodes
never mutate W. So each episode's outcome depends only on (seed, W, data), and
running a *prefix* of the legacy's hardcoded seed tables produces values
identical to the full run's — the capture just stops earlier.

## Why training is reproducible at all

The legacy policy's exploration RNGs (`local_rng`, `local_rng_2`, used only by
the training path) are **unseeded** `random.Random()` instances — a plain legacy
run trains differently every time. The capture seeds them per episode
(`protocol` offsets + train seed) right after constructing the policy, without
modifying the legacy script. The stored W trajectory is therefore one exactly
re-runnable realization of the legacy's random training; the ported Trainer
(ticket 08/09) must reproduce the same seeding convention.

## Verifying / re-capturing

- Verify (needs the dataset in the repo's parent folder or `STDVRP_DATA_DIR`):
  `uv run pytest -m golden` — re-runs the whole protocol and requires bit-exact
  equality.
- Re-capture (only after a *deliberate*, documented behavior change — Phase 2):
  rerun the capture script and commit the new JSON together with the change.

## The world cache

Loading the data through the legacy classes costs ~25 minutes of pure CPU
(88 speed files re-read, plus a pure-Python parse of the 907 MB
`all_shortest_paths.csv`). Capture and verification therefore pickle the built
world objects (`DataCalculations`, `shortest_path_memory`) to
`%LOCALAPPDATA%/stdvrp/golden_world_cache.pkl` (override with
`STDVRP_GOLDEN_CACHE`) and reuse them on later runs. The cache is invalidated
by any change to the legacy script, the data files' sizes, the world-shaping
parameters (horizon, max congestion duration) or the Python/numpy/pandas
versions. It stores the pristine post-`__init__` state, so a cache hit is
state-identical to a fresh build. To force a cold rebuild: delete the cache
file, or pass `--no-cache` to the capture script.

Known limitation (accepted): the data signature records byte sizes, not
content hashes — a same-size in-place edit of a data CSV would go unnoticed by
both the cache key and `test_dataset_matches_captured_signature`, and a warm
cache skips the legacy load path entirely. After any deliberate data change,
delete the cache and re-verify cold.
