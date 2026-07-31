# 07 — Trainer: live paired report, convergence stopping, checkpoints

**What to build:** Make a training run **watchable and resumable**. Every
evaluation block prints its paired comparison against the reference card, so you
know within two blocks whether anything is happening. The run stops when it has
converged, not when a counter runs out.

**Blocked by:** 01, 06

**Status:** resolved

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

- [x] Per-episode line: seed, cost, loss, lr, wall-clock.
- [x] Per-block line: mean, the reference card's mean, the **paired** delta,
      the win count, the Wilcoxon p, the best-so-far, the patience counter.
- [x] The delta is paired **seed by seed** against the reference card's
      per-seed vector — not a difference of means. This is why ticket 01 froze a
      vector: with this cost variance, two means hide a real 3% effect, and the
      whole point of the live report is to tell "improving" from "nothing is
      happening" early.
- [x] `write_training_plot` draws the reference **curve** (per-seed spread, or
      at minimum its mean) instead of the retired `static_policy_mean_cost`
      line.
- [x] The blocks print against `evaluation_seeds`. **The verdict never uses
      them** — they select checkpoints and hyperparameters, so they are
      contaminated by construction. `test_seeds` are touched only by tickets
      08/09.

## Convergence stopping

- [x] Patience **5** evaluation blocks without improving the best mean →
      `lr × 0.3`, logged loudly. After **3** reductions with no improvement:
      **converged**, stop.
- [x] Hard safety cap: **10 000 episodes or 24 h**. It is a net, not a budget —
      if it fires, the run is recorded **"did not converge"** and may never be
      presented as a clean result.
- [x] Evaluation cadence scales with run length (~every 50 episodes). At
      `test_frequency: 10`, a 2000-episode run spends more time evaluating than
      training.

## Checkpoints and resumption

Runs are hours long on a laptop.

- [x] Checkpoint every evaluation block: network weights, optimizer state, RNG
      states, episode index, patience state, the evaluation history.
- [x] Resume from a checkpoint and continue **identically** — assert it: a run
      interrupted and resumed produces the same trajectory as an uninterrupted
      one.
- [x] `Ctrl-C` leaves a usable checkpoint, not a corrupt file.
- [x] Long runs go to a log file so the run survives the terminal.

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

- [x] Interrupt/resume produces an identical trajectory.
- [x] A deliberately hopeless run (e.g. lr = 0) is visibly hopeless in the log
      by the second block. This is the feature working.
- [x] Predicted self-golden diff: **zero.**

## Comments

Implemented as new methods on the existing `Trainer` class
(`Trainer.train_neural`, `src/stdvrp/training/trainer.py`) plus four new
modules: `neural_report.py` (live report + convergence state machine, pure
and torch-free), `neural_episode.py` (episode runners parallel to
`episode.py`'s, built around a mutable `NeuralPolicyState` rather than a
functional `W`), `neural_checkpoint.py` (atomic checkpoint save/load), and
`transformer_policy.py` gained one small addition (`last_loss`, read by the
per-episode report — `learn()` still returns `None`, matching the
`TrainablePolicy` protocol; this is an extra attribute, not a signature
change).

**torch stays optional at the `Trainer` import boundary, not just at
`stdvrp.policies`'s.** `trainer.py` is imported by hundreds of existing,
non-neural tests, so its own module-scope imports must never reach torch.
Every neural-policy import inside `train_neural`/its helper methods is
deferred (`import` inside the method body, matching `torch_support.py`'s own
discipline) — the `TYPE_CHECKING` block at the top of the file supplies type
hints only, resolved lazily by `from __future__ import annotations`.
Confirmed by actually uninstalling torch (`uv sync` without `--extra
neural`) and importing `stdvrp.training` cleanly — not merely by inspection.
One real surprise along the way: `monkeypatch.setitem(sys.modules, "torch",
None)` (the technique `test_torch_support.py` uses to simulate absence)
crashes `scipy.stats`'s own import-time torch detection with an unrelated
`AttributeError` — an artifact of that injection technique meeting scipy's
`array_api_compat` layer, not a real gap (documented at
`neural_report.py`'s own module docstring, confirmed by the real-uninstall
test above).

**RNG state needs no separate checkpointing.** Every stochastic stream —
congestion, velocity, exploration, and the new minibatch-shuffle stream
`learn` needs — is spawned fresh from each Episode's own seed
(`spawn_neural_episode_rngs`, extending ticket 13's `_spawn_episode_rngs`
with a fourth child). Resuming therefore only needs the next episode index;
verified, not assumed — the interrupt/resume identical-trajectory test
compares final network weights bit-for-bit (`atol=0.0, rtol=0.0`) between an
uninterrupted run and one stopped mid-run and resumed.

**Evaluation blocks run serially, not on the ticket-08 worker pool.** That
pool batches a fixed `W` array across processes; the transformer's weights
change every episode, and broadcasting fresh weights to N workers every
block is a real, unsolved parallelism question of its own — out of scope
here, matching the `worker_count=1` default and serial-fallback convention
already used everywhere else in this codebase.

**Evaluation cadence**: `max(50, episodes_completed // 40)`, recomputed at
the current point in the run rather than derived from a total the
convergence-based loop does not have in advance (unlike the linear
baseline's fixed `test_frequency`) — keeps evaluation overhead bounded as a
run gets longer without needing to guess its eventual length.
`evaluation_cadence_minimum`/`max_episodes`/`max_hours` are optional
`train_neural` overrides purely so tests can reach the cadence/cap in a
handful of episodes instead of thousands — production callers never pass
them, and the frozen defaults (`MIN_EVALUATION_CADENCE=50`,
`MAX_EPISODES=10_000`, `MAX_HOURS=24`) live in `neural_report.py`.

**Hopeless-run demonstration** (real simulator, real network, mini fixture,
`neural_learning_rate=1e-10` — Adam's update step is ~lr-proportional, so
this freezes the weights in practice without violating the config's
`neural_learning_rate > 0` validation):

```
Computing a real reference card (untrained linear baseline) ...
reference mean cost: 501.5

[ep    1] train seed 1000   cost 4308.8   loss 0.037   lr 1.0e-10   3.8s
[ep    2] train seed 1001   cost 13841.8   loss 0.250   lr 1.0e-10   0.4s
[ep    3] train seed 1002   cost 12643.4   loss 0.277   lr 1.0e-10   0.2s
[ep    4] train seed 1003   cost 8599.0   loss 0.227   lr 1.0e-10   0.1s
--------------------------------------------------------------------------
  eval @4   mean 11295.4  |  MC ref 501.5  |  delta +10793.8  (+2152.1%)
              wins on 0/50 seeds    Wilcoxon p=0.000
              best so far: +2152.1% @ep4     no improvement: 0/5 blocks
--------------------------------------------------------------------------
[ep    5] train seed 1004   cost 10845.7   loss 0.203   lr 1.0e-10   0.3s
[ep    6] train seed 1005   cost 9813.0   loss 0.231   lr 1.0e-10   0.1s
[ep    7] train seed 1006   cost 14292.4   loss 0.245   lr 1.0e-10   0.2s
[ep    8] train seed 1007   cost 7748.0   loss 0.185   lr 1.0e-10   0.0s
--------------------------------------------------------------------------
  eval @8   mean 11295.4  |  MC ref 501.5  |  delta +10793.8  (+2152.1%)
              wins on 0/50 seeds    Wilcoxon p=0.000
              best so far: +2152.1% @ep4     no improvement: 1/5 blocks
--------------------------------------------------------------------------
SAFETY CAP REACHED at episode 8 (0.0h) -- run did NOT converge
```

The two blocks' means are bit-identical (11295.4 == 11295.4, frozen weights
→ bit-identical greedy decisions on the same 50 evaluation seeds), 0/50
wins, and `Wilcoxon p=0.000` in the wrong direction — unmistakably hopeless
by the second block, exactly the acceptance criterion.

**Interrupt/resume identical trajectory**: verified two ways. A stubbed,
fast pytest (`test_neural_trainer.py`, no real simulation) compares an
8-episode uninterrupted run against one stopped after episode 2 and resumed,
asserting identical per-block paired seed costs and bit-identical final
weights. Also verified manually against the **real** mini-fixture simulator
before writing that test (recorded here since the manual run is not itself
committed): identical evaluation means at every block and bit-identical
final parameters, `atol=0.0, rtol=0.0`.

**Design decisions not spelled out in the ticket text, made here:**

- The depot's `decide`/`learn` participation and the target-scaling scheme
  belong to ticket 06 (see that ticket's Comments) — nothing new for this
  ticket beyond consuming `TransformerMonteCarloPolicy` as-is.
- `write_training_plot` gained a shaded ±1 population-standard-deviation
  band around the reference line (in addition to the mean line ticket 01
  already drew) — the "at minimum its mean" bar was already met before this
  ticket; the band is the "or spread" half.
- `NeuralTrainingResult` deliberately has no final-test field, unlike the
  linear baseline's `ExperimentResult`: this ticket's blocks only ever touch
  `evaluation_seeds`, and adding a `test_seeds`-driven final test here would
  itself be exactly the kind of premature verdict-reading spec.md's
  anti-p-hacking clause forbids. That is tickets 08/09's job.

**Verification**: 65 new tests across five files (`test_neural_report.py`
29 — torch-free, runs in the default suite; `test_neural_checkpoint.py` 4,
`test_neural_episode.py` 15, `test_neural_trainer.py` 10 — all `neural`-marked)
plus one addition to `test_transformer_policy.py`. Full suite (excluding
`golden`) after landing: 4166 passed, 3 deselected. `mypy`/`ruff
check`/`ruff format --check` clean on every file this ticket touched (the
same four pre-existing E501 violations noted in ticket 06's Comments,
untouched by this ticket either).

**`/code-review` (both axes), and what it found**: Standards found no hard
violations — the deferred-import discipline (torch stays optional at
`Trainer`'s own import boundary, not just `stdvrp.policies`'s) was confirmed
followed correctly in every method that needs it, not only `train_neural`
itself. Judgement calls, left as-is: `train_neural` is long (~190 lines) but
reads as one coherent narrative with no internal duplication; the
demand/rng/state/geometry setup repeated between `run_neural_training_episode`
and `run_neural_evaluation_episode` mirrors the identical, already-unshared
shape in the pinned `episode.py` (deliberate non-sharing, not an oversight);
per-file test-helper duplication (`make_config` et al.) matches
`test_trainer.py`'s own established no-shared-test-helpers convention. Fixed:
a stray local `import dataclasses` in `test_neural_checkpoint.py` moved to
the module top, matching its sibling files.

Spec found no missing or partial requirements and cleared two apparent
scope-creep concerns on inspection (the three `neural_*` config fields
predate this diff, landed in ticket 06; the `max_episodes`/`max_hours`/
`evaluation_cadence_minimum` test-seam overrides have no production call
site — `grep -rn "train_neural("` outside the test files confirms it).
It also found one **real bug**, confirmed by measurement before fixing:
the Comments above claimed the network's weight-init seed is "distinct from
every per-episode stream", but `SeedSequence(first_train_seed).spawn(1)[0]`
and episode 0's congestion stream (the first child `spawn_neural_episode_rngs`
draws from `SeedSequence(first_train_seed + 0)`) were bit-identical —
numpy's spawn-keying depends only on child index, not on how many children a
given call asks for, so two independently-constructed `SeedSequence`s built
from the same bare-int entropy collide on their first spawned child
regardless. Fixed by seeding the network init from a two-element entropy
sequence (`[first_train_seed, salt]`) structurally unreachable from any
plain per-episode seed — reverified with zero collisions against every
spawned stream over the full 10 000-episode safety-cap range. Did not
corrupt determinism or the resume-identical-trajectory guarantee (both
streams are simply independent generators feeding unrelated computations),
but was a real, worth-fixing confound for ticket 08's "≥3 independent
network-init seeds" requirement, since varying the init seed by varying
`first_train_seed` would previously also have shifted episode 0's
congestion draws. Also fixed on the same pass: `format_lr` (new, shared by
`neural_report.py` and `Trainer.train_neural`'s own log lines) matches
spec.md's single-digit-exponent example (`3.0e-4`) instead of Python's
default two-digit `3.0e-04`; `EvaluationReport.delta`'s docstring now states
why it is not the "difference of means" spec.md's paired-comparison
requirement forbids, despite reading like one in isolation.
