# 07 — Trainer: live paired report, convergence stopping, checkpoints

**What to build:** Make a training run **watchable and resumable**. Every
evaluation block prints its paired comparison against the reference card, so you
know within two blocks whether anything is happening. The run stops when it has
converged, not when a counter runs out.

**Blocked by:** 01, 06

**Status:** open

## The live paired report

```
[ep  348] train seed 1348   cost 2691.3   loss 0.387   lr 3.0e-4   8.9s
[ep  349] train seed 1349   cost 2402.8   loss 0.371   lr 3.0e-4   9.1s
[ep  350] train seed 1350   cost 2555.0   loss 0.369   lr 3.0e-4   8.7s
--------------------------------------------------------------------------
  eval @350   mean 2087.4  |  MC ref 2314.9  |  delta -227.5  (-9.8%)
              wins on 37/50 seeds    Wilcoxon p=0.003
              best so far: -9.8% @ep350     no improvement: 0/5 blocks
--------------------------------------------------------------------------
```

- [ ] Per-episode line: seed, cost, loss, lr, wall-clock.
- [ ] Per-block line: mean, the reference card's mean, the **paired** delta,
      the win count, the Wilcoxon p, the best-so-far, the patience counter.
- [ ] The delta is paired **seed by seed** against the reference card's
      per-seed vector — not a difference of means. This is why ticket 01 froze a
      vector: with this cost variance, two means hide a real 3% effect, and the
      whole point of the live report is to tell "improving" from "nothing is
      happening" early.
- [ ] `write_training_plot` draws the reference **curve** (per-seed spread, or
      at minimum its mean) instead of the retired `static_policy_mean_cost`
      line.
- [ ] The blocks print against `evaluation_seeds`. **The verdict never uses
      them** — they select checkpoints and hyperparameters, so they are
      contaminated by construction. `test_seeds` are touched only by tickets
      08/09.

## Convergence stopping

- [ ] Patience **5** evaluation blocks without improving the best mean →
      `lr × 0.3`, logged loudly. After **3** reductions with no improvement:
      **converged**, stop.
- [ ] Hard safety cap: **10 000 episodes or 24 h**. It is a net, not a budget —
      if it fires, the run is recorded **"did not converge"** and may never be
      presented as a clean result.
- [ ] Evaluation cadence scales with run length (~every 50 episodes). At
      `test_frequency: 10`, a 2000-episode run spends more time evaluating than
      training.

## Checkpoints and resumption

Runs are hours long on a laptop.

- [ ] Checkpoint every evaluation block: network weights, optimizer state, RNG
      states, episode index, patience state, the evaluation history.
- [ ] Resume from a checkpoint and continue **identically** — assert it: a run
      interrupted and resumed produces the same trajectory as an uninterrupted
      one.
- [ ] `Ctrl-C` leaves a usable checkpoint, not a corrupt file.
- [ ] Long runs go to a log file so the run survives the terminal.

## Note on the stopping rule and the anti-p-hacking clause

The stopping rule is **in the spec, frozen**. It is not "stop when it looks
good" — that is a decision made after seeing the numbers, which spec.md's
anti-p-hacking clause forbids. If you find yourself wanting to stop a run early
because it looks promising, that is exactly the impulse the rule exists to
overrule.

## CONTEXT.md

Add **Reference card**: a completed Policy's frozen per-seed costs, the fixed
opponent every later run is compared against.

## Acceptance

- [ ] Interrupt/resume produces an identical trajectory.
- [ ] A deliberately hopeless run (e.g. lr = 0) is visibly hopeless in the log
      by the second block. This is the feature working.
- [ ] Predicted self-golden diff: **zero.**

## Comments
