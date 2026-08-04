# 17 — Gate A′: does training add anything on top of the base?

**What to build:** The landing gate, rewritten for the residual decomposition —
and the two encoder arms that produce the number this effort has been trying to
produce since it started: **the value of representation learning.**

**Blocked by:** 16

**Status:** open

## Why the gate had to be rewritten, not reused

Ticket 08's Gate A asked "does the trained network beat the same architecture
untrained". Under the residual decomposition the untrained network **is** the
myopic base — a genuinely strong cost-greedy dispatcher, not a nearest-Client
placeholder — so the question changes into the one that actually matters:
**does training add value on top of a good initialization?** Ticket 08 answered
that question by accident and the answer was no. This gate asks it on purpose.

Two of the three parts break if reused verbatim.

**Part 1's threshold.** ≥5% over the null was written when the null was
nearest-neighbour. It is now the wrong size in both directions, and the honest
threshold is the gap the effort actually has to close.

**Part 3 is vacuous.** `ρ(Q, U_t)` with `c` frozen and additive would pass at
`W = 0` — `Q` correlates with the return without a single parameter having
moved. Ticket 08 already flagged the weaker version of this ("part 3 can pass
on an action-blind network"); under the decomposition it stops being a blind
spot and becomes a **guaranteed false PASS.**

## The three parts

All on the held-out `test_seeds` (100..153). **Never on the evaluation seeds** —
those chose the checkpoint, `λ`, `γ` and the architecture.

| Part | Test | Threshold |
|---|---|---|
| **Null model** | Trained vs. `W = 0` (the myopic base), paired per seed | Wilcoxon **p < 0.05** and **≥ 5%** mean cost reduction |
| **Reproducibility** | **≥ 3** independent init seeds, per arm | reported mean ± sd |
| **Calibration** | Spearman **ρ(W·φ, ỹ)** on held-out episodes — the learned term against the residual it is actually regressed onto | **≥ 0.5**, against ≈ 0 at `W = 0` |

### Part 1's threshold stays at 5%, and stays decoupled from the baseline

It was amended to *"≥ the gap against the baseline's 3384.82"* on 2026-08-02
and amended back the same day. The measurement that reversed it
(`results/baseline_null_50.py`, 50 `evaluation_seeds`):

| policy | `evaluation_seeds` | `test_seeds` | ratio |
|---|---|---|---|
| linear `best_w` @ `m+40` — **selected on `evaluation_seeds`** | **2168.39** | 3384.82 | **×1.56** |
| neural cost-greedy — no selected parameters | 3693.23 | 3811.28 | **×1.03** |

That is F12's winner's curse, measured, and the part that matters is that it is
**policy-dependent**: the selection set flatters a *fitted* `W` by ~36% and an
arithmetic rule by ~3%. So "the gap to the baseline" computes to **41.3% on
selection data and 11.2% on verdict data**, and any formula mixing the two seed
sets — as that amendment did — is incoherent rather than merely imprecise.
A threshold that cannot be computed on one seed set is not a threshold.

So Gate A′ asks its own self-contained question — **does training improve on
its own null, by at least 5%** — and Gate B alone decides whether that is
enough to matter. The two gates stay independent, which is what spec.md said
they were for in the first place ("Three gates, three different questions. None
substitutes for another").

**What did change is the null, and it is much harder.** The 5% was written when
the untrained network was a nearest-Client placeholder. It is now the myopic
base — a cost-greedy dispatcher over the baseline's own candidate set, which
ticket 08 measured beating 1150 episodes of training by 13.8%. Five percent
over *that* is a different claim from five percent over nearest-neighbour, and
every report of this gate names which null produced it.

### Part 2 stays at three init seeds, and it is not optional

Ticket 08 closed with **n = 1**. Arm 0 gave +15.49% (p = 6.21e-05) and arms 1
and 2 were killed at episode 162 when the learning rule they were testing was
replaced. The effort's headline — *"It learns — that question is closed"* — has
never been reproduced. Under the frozen encoder the init seed still varies the
random feature map, so the test keeps its meaning.

If the spread straddles zero it did not learn, whatever the best run says.

### Part 3's companion diagnostic, reported every block

```
r = sd_candidates(W·φ) / sd_candidates(c)

r ≈ 0        the learned term does not touch the ranking — or ticket 16's λ
             shrank it away (the failure this effort has been measuring)
r ≈ 0.1–0.5  correcting the base without overwriting it        ← the target
r ≫ 1        overwriting c(s, a) — the ticket-08 failure mode, returning
```

Ticket 08 had to measure this by hand three times with scratchpad probes. It is
cheap; it belongs in the live report, not in a script someone rewrites.

## The two arms — both run, unconditionally

| arm | encoder | learned by |
|---|---|---|
| **frozen** | random, never trained (real attention, not an identity map — ticket 15) | ridge only. **No learning rate anywhere in the system** |
| **trained** | same architecture, slow SGD (ticket 16's two timescales) | ridge + SGD |

```
delta(frozen)  vs baseline        = value of the cost features + the estimator
delta(trained) − delta(frozen)    = value of representation learning
```

**Both run whatever the first says.** If `trained` only ran when `frozen` lost,
and `frozen` won, the second number would never exist — and it is the effort's
original thesis ("a transformer over raw State beats a 19-feature linear VFA")
finally reduced to a measurement. Running them conditionally would be winning
without being able to say why.

## Work

- [ ] Develop on the mini fixture; run the gate on the real dataset.
- [ ] 3 init seeds × 2 arms. Report **mean and median** improvement — with this
      cost distribution they differ, and the difference is informative.
- [ ] Report the numbers **whatever they are**, including a failure. Ticket 09
      cannot be interpreted without them.
- [ ] Name the null and its warm-start-equivalent alongside every trained
      number, per spec.md's standing obligation.

## If the gate fails

The failure protocol is sharper than ticket 08's, because the instruments now
separate three cases instead of two:

| ρ(W·φ, ỹ) | `r` | reading |
|---|---|---|
| low | ≈ 0 | the residual does not fit — estimator or `λ` over-shrinkage (ticket 16) |
| low | ≫ 1 | it is fitting noise and spending the base — `γ` too short, or the target is not attributable |
| **high** | ≈ 0 | **it fits and the policy still does not improve** — the residual is real but orthogonal to the ranking. That is a decision-structure result, and it points at rollout (research note #2), not at another estimator |

## Acceptance

- [ ] All three parts, both arms, numbers recorded in this ticket's Comments.
- [ ] The decomposition `delta(trained) − delta(frozen)` stated explicitly as a
      number with a sign, not left to be inferred from two tables.
- [ ] Predicted self-golden diff: **zero.**

## Comments
