# Measurement artifacts (tickets 08 and 13-17)

The raw evidence behind issue `08-gate-a-does-it-learn.md`'s Comments and the
redesign it handed to. Kept here and not under `runs/`, which is gitignored:
the numbers are transcribed into the tickets, but a transcription is not a
result — anyone re-reading this in six months should be able to recompute the
tables rather than trust them.

## The linear baseline's own null (ticket 14)

- **`baseline_null_50.py`** / **`baseline_null_50.json`** /
  **`baseline_null_50.log`** — three arms over the 50 `evaluation_seeds`,
  through `EpisodeWorld.run_episode`, which is the identical path
  `Trainer._run_evaluation_block` takes. `best_w @ m+2` reproduces ticket 01's
  own episode-50 figure (2483.24) to the cent, which is what says the harness
  is right.

  | arm | mean | median |
  |---|---|---|
  | `W = 0` @ `m+2` | **30 791.43** | 30 627.72 |
  | `best_w` @ `m+2` | 2483.24 | 2390.14 |
  | `best_w` @ `m+40` (the card's winning cell) | 2168.39 | 2036.78 |

  Two findings, one of them a refutation:

  1. **The action count is worth 12.68%** (`m+40` vs `m+2`, 36/50 seeds,
     Wilcoxon p = 8.24e-05) — the measured basis for ticket 14 reversing
     ADR-0007. On `test_seeds` the same axis reads 2.1%: the effect
     reproduces, its magnitude does not transfer.
  2. **`W = 0` is not a myopic null.** It was predicted to be
     "nearest allowed Client" (`X · W = 0` → `argmin` takes index 0 →
     `_closest_allowed_clients` is nearest-first). Branch 3's
     `list(set(possible_actions))` runs *after* the sort and node ids are
     arbitrary ints, so the dedup returns hash-table order: `W = 0` picks an
     **arbitrary** feasible Client. The linear baseline has no cheap myopic
     null, and the candidate-set-versus-ranking decomposition has to wait for
     ticket 14's own 2×2.

  Read beside `warm_start_50.py` below — same 50 seeds — it also measures
  **F12's winner's curse, and that it is policy-dependent**: a `W` selected on
  `evaluation_seeds` reads 2168.39 there against 3384.82 on `test_seeds`
  (×1.56), while cost-greedy reads 3693.23 against 3811.28 (×1.03). Any
  quantity defined across the two seed sets is not well-defined; a Gate A′
  threshold defined as "the gap to the baseline" was drafted and reverted on
  exactly this.

## The action set adopted: the missing 2×2 cell (ticket 14)

- **`action_set_m2_50.py`** / **`action_set_m2_50.json`** /
  **`action_set_m2_50.log`** — post-ticket-14 (the `m+2` `action_set.py`
  shortlist live in `TransformerMonteCarloPolicy._sweep`), the untrained
  `cost` warm start over the same 50 `evaluation_seeds`, two arms:

  | arm | mean | median |
  |---|---|---|
  | cost @ `m+2`, `DEPOT_WARM_START_PENALTY = 1.0` (as shipped) | **3365.09** | 3376.23 |
  | cost @ `m+2`, `DEPOT_WARM_START_PENALTY = 0.0` | 3364.52 | 3376.23 |

  The first row is the cell `neural-policy` ticket 14's own table was missing
  — cost-greedy ranking, the baseline's candidate set — completing the 2×2
  that separates candidate set from ranking rule (full table in
  `docs/adr/0011-the-action-set-is-shared-not-owned.md` and ticket 14's
  Comments). Read against `warm_start_50.py`'s 3693.23 (cost-greedy at ~151
  candidates, ranking rule held fixed): restricting to `m+2` is **8.9%
  better** for the myopic dispatcher too, not only for a linear model that
  cannot see far. Read against `baseline_null_50.py`'s 2168.39 (the linear
  baseline's best cell, same seed set): the myopic base does **not** beat a
  tuned 19-weight VFA at zero training here — the pre-declared branch for
  "if it does" did not fire.

  The two rows decide `is_depot`/`DEPOT_WARM_START_PENALTY`: **-0.02%, 1/50
  seeds differ, Wilcoxon p = 0.317** — a null result. The penalty is kept
  (costs nothing where it no longer matters, still correct where it does),
  confirming the structural prediction that the depot now enters the `m+2`
  candidate list only where `select_vehicle_possible_actions` itself admits
  it, rather than by argument alone.

## Gate A

- **`gate_a_init0.json`** — the official arm 0 result, copied verbatim from
  `runs/gate_a_v2/results_init0.json`. 1150 episodes, converged, best block
  (episode 650) restored for measurement per protocol. Carries the per-seed
  `trained_seed_costs` / `untrained_seed_costs` vectors over the 50 held-out
  `test_seeds`, so every statistic in the ticket is recomputable:
  null 5299.48 -> trained 4423.73, +15.49% mean reduction, Wilcoxon
  p = 6.21e-05, calibration Spearman 0.5427 against -0.3487 untrained.
  **This is `n = 1`.** Arms 1 and 2 were killed at episode 162 on 2026-08-02
  when the learning rule they were testing was replaced (ticket 08's closing
  comment); `results_init12.json` was never written, and Gate A part 2
  (reproducibility) has never been measured. Cite the +15.49% with that
  attached — `runs/gate_a_v2/gate_a_init1.pt` and `log_init12.txt` are what is
  left of the arm.

- **`gate_a_cost_warm_start_87ep.log`** — the arm launched with
  `neural_warm_start: cost`, stopped at episode 87 once it had made its point.
  This is where the level-mismatch diagnosis comes from: null on `test_seeds`
  3811.28 (the warm start's -28% holds on the verdict set, not just on
  `evaluation_seeds`), first evaluation block at episode 50 **6168.5 — +62%
  worse than its own null** — training costs climbing monotonically, and
  episode 1's loss of `3.2e-01` reproducing `0.5 * 0.8^2` exactly.

## Mini fixture A/B (`mini-fixture-ab/`)

One JSON per arm, each 200 episodes with a greedy evaluation every 10 over
held-out seeds 100..109, `init_seed=0`, paired against that arm's *own*
same-architecture untrained null. Produced by `mini_ab.py` (or
`mini_ab_ablate.py` where a `QHead.forward` ablation was needed). Every file
carries `null_mean`, `null_costs` and per-block `costs`, so the ticket's
tables are recomputable from them.

| file | arm | null |
|---|---|---|
| `control.json` | before the cost features landed | 577.22 |
| `costfeat.json` | the committed learning rule | 577.22 |
| `execfilter.json` | executed-actions-only regression (rejected earlier) | 577.22 |
| `dueling.json` | dueling V + centred advantage (rejected) | 577.22 |
| `dueling_v10.json` | same, value branch scaled 10x (rejected) | 577.22 |
| `dueling_cost.json` | dueling + `cost` warm start (rejected) | 542.82 |
| `dueling_cost_huber.json` | + Huber delta 0.02 (rejected; stopped at ep 140) | 542.82 |
| `min_lvl100.json` | `minutes` + `neural_level_gain: 100` | 577.22 |
| `cost_only.json` | `cost` warm start, committed learning rule | 542.82 |
| `cost_lvl100.json` | `cost` + `neural_level_gain: 100` | 542.82 |

Note the two nulls, and that they are *identical* within each group: that is
the measured proof that neither the dueling decomposition nor the level gain
moved the untrained policy, only what training does to it.

## Probes

- **`warm_start_probe.py`** — cost-component decomposition of the null, the
  trained checkpoint and the degraded checkpoint over 8 `evaluation_seeds`,
  plus the warm-start weight-vector screen. This is what showed `unserved` at
  0.0 while delay doubled, ruling out fleet retirement.
- **`warm_start_50.py`** — the headline warm-start number over all 50
  `evaluation_seeds`, paired, with the Wilcoxon: `minutes` 5484.25 vs `cost`
  3693.23, -32.7%, 47/50 seeds, p = 2.5e-14.

Both take the dataset path explicitly. The repository moved out of
`OneDrive\\Documentos\\Mega city`, so `experiments/chengdu/config.yaml`'s
relative `data_dir: ../../..` no longer resolves — every real-dataset run
needs `--data-dir "C:/Users/ferna/OneDrive/Documentos/Mega city"` (the gate
script has the flag for exactly this).
