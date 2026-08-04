# 04 — Everything measured at `time_window_spread: 60` needs re-reading

**Status:** open, unclaimed

## What changed

Ticket 01 found that every Chengdu config in this repo set
`time_window_spread: 60` while the reference runs used **150**
(`scripts/capture_golden_master.py`'s `diff_TW`, and the `Tw150` in the legacy's
own stored result filenames). The configs are now 150:

- `experiments/chengdu/config.yaml`
- `experiments/chengdu/config_linear_congestion_10k.yaml`
- `experiments/chengdu/config_linear_congestion_10k_eps0_lr1e-5.yaml`
- `experiments/chengdu/baseline_scaled.yaml`
- `experiments/chengdu/parallel_scaled.yaml`

`tests/fixtures/chengdu_mini/config.yaml` is deliberately **not** changed: it is
a synthetic 20-Client miniature, not a reproduction of the Chengdu runs, and its
window width is its own knob. Every test builds from that fixture, so the config
change breaks nothing in the suite — which is exactly why it needs this ticket
instead.

## Why it is not cosmetic

`diff_TW` sets each Client's window width *and* the latest minute a window can
open (`randint(300, 780 - diff_TW)`), so 60 is a strictly harder problem. Over
200 training episodes it moves the two sides in **opposite** directions:

| | legacy `‖W‖` @200 | legacy train cost | repo `‖W‖` @200 | repo train cost |
| --- | --- | --- | --- | --- |
| TW 60 | 3 732 | 6 671 | 7 502 | 11 229 |
| TW 150 | 2 895 | 3 534 | 15 754 | 9 450 |

The legacy's own numbers confirm which is right: at TW 150 its training cost is
mean 3 534 / median 2 695, inside the [2 798, 4 316] band `spec.md` records and
matching its stored result file for lr 0.001 / ε 0 / lc 0.1 / uc 0.4 /
duration 120 (mean cost 4 182). At TW 60 it comes out at 6 671, outside it.

## What is now stale

- **`experiments/chengdu/reference_card.json`.** Left untouched on purpose: it is
  a *measurement record* that embeds the config it was measured under, and
  editing that config would misreport what was measured. Its `best_w` is already
  19 components wide against today's 24, so it predates the restored features
  too. `scripts/run_baseline_reference_card.py`'s docstring now carries the
  warning.
- **The neural effort's Gate A and Gate A' results** (`.scratch/neural-policy/`),
  which are paired against that card.
- **Everything under `runs/`**, including the runs `spec.md`'s "The symptom"
  section quotes.
- **Every table in `spec.md` above the "Corrected first" section**, and the
  TW 60 trajectory JSONs beside it (kept, clearly labelled, not deleted).

## Done when

- [ ] A decision per stale artifact: re-measure, or annotate as historical
- [ ] `reference_card.json` re-run at TW 150 if the neural effort still needs a
      live opponent — and if so, at 24 features, not 19
- [ ] `spec.md`'s superseded tables either re-measured or explicitly dated
- [ ] A check on whether any *other* config field drifted from the reference the
      same way `diff_TW` did — this one was found by accident, not by an audit
