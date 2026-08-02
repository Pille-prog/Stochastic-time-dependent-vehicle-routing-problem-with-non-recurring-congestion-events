# Ticket 08 measurement artifacts

The raw evidence behind issue `08-gate-a-does-it-learn.md`'s Comments. Kept
here and not under `runs/`, which is gitignored: the numbers are transcribed
into the ticket, but a transcription is not a result — anyone re-reading this
in six months should be able to recompute the tables rather than trust them.

## Gate A

- **`gate_a_init0.json`** — the official arm 0 result, copied verbatim from
  `runs/gate_a_v2/results_init0.json`. 1150 episodes, converged, best block
  (episode 650) restored for measurement per protocol. Carries the per-seed
  `trained_seed_costs` / `untrained_seed_costs` vectors over the 50 held-out
  `test_seeds`, so every statistic in the ticket is recomputable:
  null 5299.48 -> trained 4423.73, +15.49% mean reduction, Wilcoxon
  p = 6.21e-05, calibration Spearman 0.5427 against -0.3487 untrained.
  **Arms 1 and 2 land in `runs/gate_a_v2/results_init12.json`** — copy them
  here alongside this file when they finish.

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
