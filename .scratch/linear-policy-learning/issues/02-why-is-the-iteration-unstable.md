# 02 — Why the legacy's W iteration is stable and ours is not

**Status:** open, unclaimed

**Blocked by:** none (ticket 01 is resolved and left the tooling warm)

## The question ticket 01 leaves

Ticket 01 showed there is no misbehaving component to name: all 22 weights the
legacy trains are inflated here by 2.6x - 13.8x, median 4.84x. It also showed the
update's *inputs* are close on both sides — over 25 episodes at TW 150, `T` 377.8
vs 413.4, mean `‖X‖` 1.803 vs 2.170, mean `|residual|` 1380 vs 2398, and our
steps are actually *less* coherent (0.868 vs 0.784). Nothing off by more than
1.74x.

Yet `‖ΔW‖` comes out 4.68x and `‖W‖` after 25 episodes 9.16x (1658 vs 15189).
The same-sized steps land in a settling place there and a drifting one here: the
legacy's `‖W‖` rises to 4249 by episode 50 and comes back down (2895 at 200),
ours never settles (15189 at 25, 7910 at 100, 15754 at 200).

## What to measure

`W += lr·(U_t − acquired − XᵀW)·X` is linear stochastic approximation. Its
stability margin is set by `lr·λmax(E[XXᵀ])`, not by any single feature's size,
which is exactly why a whole-vector inflation and no named culprit is the shape
ticket 01 found.

So: **measure the spectrum of `E[XXᵀ]` on both sides**, over the same update rows
ticket 01's probes already walk. Report `λmax`, the condition number, and
`lr·λmax` against the stability bound. If ours crosses where the legacy's does
not, that is the answer, and it names the fix (`lr`, feature scaling, or which
features to drop) rather than the symptom.

Worth splitting the accumulation too: is `E[XXᵀ]` inflated by many features
sharing one direction (collinearity) or by one direction being genuinely bigger?
The 24 features include several exact functions of each other
(`time_left`/`time_left^2`, `clients_left^2`/`clients_left^2*time`,
`future_delay`/`future_delay^2`), so a near-singular `E[XXᵀ]` is plausible on
both sides — the question is whether ours is worse.

## The machinery is warm

Ticket 01 left `scripts/capture_legacy_w_trajectory.py` (the shim, plus a warm
world cache at `%LOCALAPPDATA%/stdvrp/legacy_w_world_c8aadb53ae7a.pkl` — ~15s to
load, ~4s/episode) and `scripts/capture_repo_w_trajectory.py` (~1.3s/episode).
Run both at TW 150; ticket 01 also found the configs had been set to 60
(`issues/04`), and the two settings are not comparable.
Both were instrumented for ticket 01 by wrapping `actualize_W` / `learn` without
editing either side; each instrumented run reproduced its uninstrumented `‖W‖`
exactly, so that is a validated way in.

## Done when

- [ ] `λmax(E[XXᵀ])`, the condition number, and `lr·λmax` measured on both sides
      over a comparable set of update rows
- [ ] A statement of whether ours crosses the stability bound where the legacy's
      does not — and if it does, which directions carry `λmax`
- [ ] The answer written into `spec.md`
