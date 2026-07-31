# 08 — Gate A: does it learn

**What to build:** The three-part acceptance gate that decides whether the
network learned anything at all. This is the **hard landing gate** of the
effort — independent of whether it beats the baseline.

**Blocked by:** 07

**Status:** open

## The three parts

All three, on the held-out `test_seeds` (100..153). **Never on the evaluation
seeds** — those selected the checkpoint and the hyperparameters.

| Part | Test | Threshold |
|---|---|---|
| **Null model** | Trained vs. the same architecture **untrained**, paired per seed | Wilcoxon signed-rank **p < 0.05** and **≥ 5%** mean cost reduction |
| **Reproducibility** | ≥ **3** independent network-init seeds | Improvement reported as mean ± sd |
| **Calibration** | Spearman ρ(predicted `Q`, realised `U_t`) on held-out episodes | **≥ 0.5**, against ≈ 0 untrained |

## Why each one is there

**The null model is nearest-neighbour, not noise.** Ticket 05's myopic warm
start means the untrained network already goes to the nearest feasible Client.
Beating that by 5% is a real claim. Do not weaken the warm start to make this
gate easier — that games the null.

**Reproducibility** guards against one lucky init. A single run that clears the
bar is not evidence; three runs with a reported spread are. If the spread
straddles zero, it did not learn, whatever the best run says.

**Calibration is the part that cannot be faked.** A policy can improve by
accident — a shifted argmin that happens to route better without the value
function meaning anything. Correlation between predicted `Q` and the realised
cost-to-go measures whether the network learned *the thing it was trained to
learn*, independent of whether the policy improved. Spearman and not Pearson
because the cost distribution's right tail is brutal (research note **F10**).

## Work

- [ ] Develop and debug on the **mini fixture** (20 Clients, ~72 decisions/ep,
      no 8 GB world load). Run the gate itself on the real dataset.
- [ ] Wilcoxon signed-rank over the 50 paired seeds. Report **both** mean and
      median improvement — with this distribution they will differ, and the
      difference is informative.
- [ ] Report the numbers **whatever they are**, including a failure. Ticket 09
      cannot be interpreted without them.

## If the gate fails

The effort does not proceed to a verdict on a network that did not learn. The
failure is diagnosed first: the calibration number distinguishes "the value
function is not fitting" (a learning-rule or target-scaling problem) from "the
value function fits but the policy does not improve" (a decision-structure or
action-space problem). Those have different fixes and the gate is designed to
tell them apart.

## Acceptance

- [ ] All three parts pass, with the numbers recorded in this ticket's
      Comments.
- [ ] Predicted self-golden diff: **zero.**

## Comments
