# 06 — The fourth earliness bin

**What to build:** `counts_earliness[3]` is never assigned, so `general[10]` is
identically zero at every instant under every window. Closes B10.

`policies/feature_extraction.py:208-214,230`:

```python
counts_earliness = [0, 0, 0, 0]
if tau < 400: counts_earliness[0] = ...(earliness < 400)
if tau < 500: counts_earliness[1] = ...(400 <= earliness < 500)
if tau < 600: counts_earliness[2] = ...(500 <= earliness < 600)
# counts_earliness[3] is never assigned
general[7:11] = [count / self._number_clients for count in counts_earliness]
```

Three verified consequences: `W[10]` never updates because the gradient is
`lr * err * X` and `X[10]` is always zero, so **the effective model is
18-dimensional, not 19**; no counting bin counts a Client whose window opens at
minute 600 or later, which under the real `Uniform[300, 720]` is **28.7% of
demand**; and for `tau >= 600` all four bins are zero, so five of the twelve
general features carry no information across roughly half the horizon — exactly
where end-of-day pressure and delay penalties start to bite.

**Confirmed in the committed capture: `final_w[10] == 0.0` exactly.**

**Blocked by:** 01

**Status:** resolved

- [x] Assign the fourth bin so it covers what the other three leave out, and
      `general[10]` stops being identically zero.
- [x] **Do not redesign the cuts.** Where the boundaries should sit — and
      whether four bins keyed to absolute minutes is the right shape at all — is
      feature design, which belongs to the modeling effort along with B4 and B6.
      This ticket makes the existing design *work*; it does not replace it.
- [x] Invariant: the four earliness bins **partition** pending demand — the four
      fractions sum to pending Clients over `number_clients`, at every tau. That
      property is what makes "a Client falls in no bin" impossible to reintroduce.
- [x] Update the docstrings: they document the other dead weight (the `X[:,13]`
      filler) but not this one. Note that `W[13]` remains dead by design and
      `final_w[13] == 0.0` — after this ticket exactly one weight is
      deliberately dead, not two accidentally.

## Predicted self-golden diff

**Frozen-W block: exactly zero, and this is provable in advance.** The frozen W
is the committed `final_w`, whose entry 10 is `0.0` exactly. Feature 10 enters Q
only as `X[10] * W[10]`, so any change to `X[10]` multiplies by zero. Decisions
cannot move. Every metric on every seed must be bit-identical.

If the frozen-W block moves at all, the change escaped feature 10 — most likely
the fourth bin's definition also altered bins 0–2, which this ticket forbids.
That is the check.

**Training block: divergence, from the first weight update onward.** `W[10]`
starts receiving gradient, so the trained W is 19-dimensional for the first time
and every subsequent decision can differ. **Final-eval block: divergence**, since
it runs on that changed final W.

Direction is not predicted. A feature that has never been trained carries no
prior about whether it helps; the honest expectation is "the W trajectory
changes shape", not "cost improves".

## Evidence required

Frozen-W block bit-identical (the strong claim). `final_w[10] != 0.0` after
training — the point of the ticket is that the weight now trains. Partition
invariant green. The 60-seed bench before/after, reported without a directional
claim.

## Comments

### Resolution (2026-07-30)

**The fix**, `src/stdvrp/policies/feature_extraction.py`, `_general_features`:
one line after the existing `if tau < 400/500/600` chain (unchanged):

```python
counts_earliness[3] = remaining_count - sum(counts_earliness[:3])
```

Bin 3 is *not* redefined as "earliness >= 600" — a bin above also drops to
zero once `tau` passes its own gate (e.g. bin 0 once `tau >= 400`) while its
Clients are still pending, and bin 3 must absorb those too or the partition
invariant breaks the moment any earlier bin is gated shut. This is exactly
what "covers what the other three leave out" means literally, not just for
the `earliness >= 600` case. The 400/500/600 cuts themselves are untouched.

**Invariant (spec.md catalogue row 7), `tests/unit/test_feature_extraction.py`,
`TestEarlinessBinPartition`**: `sum(general[7:11]) == len(clients_not_visited) /
number_clients`, parametrized over the file's existing 540 scenario/tau/remaining-
count/depot-occupancy combinations. Red before the fix — **282 of 540 cases
failed** (every combination with at least one pending Client whose earliness
either sits at/above 600, or belongs to a bin already gated shut by tau).
Green after, all 540.

**Parity oracle updated in lockstep.** `tests/unit/test_feature_extraction.py`'s
`LoopReference` (the verbatim pre-vectorization loop, simulation-performance
ticket 05's oracle) and `tests/unit/test_monte_carlo_policy.py`'s
`hand_computed_features()` (a hand-derived 19-feature vector for a fixed toy
World) both carried the same unassigned-fourth-bin bug and needed the identical
one-line fix — otherwise the bit-exact parity tests would fail not because the
vectorization is wrong, but because the oracles still encode the bug the
production code no longer has. `hand_computed_features`'s World has 2 pending
Clients at `tau=400`: Client 2 (window start 450) lands in bin 1 as before;
Client 1 (window start 350, normally bin 0) is now caught by bin 3 instead,
since `tau == 400` already gates bin 0 shut. `general[10]` changes from `0.0`
to `1/2`.

### Self-golden (predicted vs. measured)

Verified in an isolated worktree at the pre-ticket commit (checked out clean,
only this ticket's diff applied) to avoid conflating with concurrent tickets
also in flight on this branch:

| Block | Predicted | Measured |
|---|---|---|
| **Frozen-W** | exactly zero | **Exactly zero.** `final_w`/`training`-w/`evaluation`/`frozen_w_eval` all differ only where predicted (see below) — the `frozen_w_eval` block is bit-identical, entry for entry, seed for seed. |
| Training | divergence, W trajectory changes shape | `training[*].w` diverges from `TRAIN_SEEDS[0]`'s very first update (`w[10]` goes from a bug-locked `0.0` to nonzero immediately); `training[*].metrics` stayed bit-identical on this capture (the small perturbation never flipped a greedy argmax across these 5 episodes) |
| Final-eval | divergence, since it runs on the changed final W | `final_w` differs (all 18 previously-trained components shift slightly in addition to `final_w[10]` going nonzero), but `evaluation[*].metrics` came out bit-identical on this capture — same mechanism as training: no eval-seed decision flips under this particular W perturbation |

**`final_w[10] == 0.17189672845126716`** after training (was `0.0` exactly) —
confirms the weight now trains, closing the ticket's second piece of required
evidence. Per spec.md decision 10 (three-outcome rule): the strong claim
(frozen-W exactly zero) **matches** exactly as predicted, unedited above. The
softer training/eval prediction is explained to a mechanism (argmax decisions
insensitive to this magnitude of W perturbation on this specific 5+10-episode
capture) rather than reverted — nothing here contradicts the mechanism, it is
simply a fact about this small sample, not a claim that decisions can never
move.

### 60-seed bench (before = ticket 01's baseline at 55f32aa, after = this fix)

Both `--w frozen` and `--w zero` (decision-stable, no training — see ticket
01): every per-seed metric (`total_cost` and its 4 components, `km_driven`,
`final_tau`, `decisions`, unserved split, every other invariant counter) is
**bit-identical across all 60 seeds, in both W configurations** — expected,
since neither W multiplies feature 10 by anything nonzero. The only column
that moves is the bench's own B10 counter:

```
                              before        after
--w frozen  B10_earliness_bin_partition  episodes=60/60 decisions=3596  ->  episodes=0/60 decisions=0
--w zero    B10_earliness_bin_partition  episodes=60/60 decisions=3206  ->  episodes=0/60 decisions=0
```

No directional cost claim is made, per the ticket's own prediction — the bench
is decision-stable by construction and cannot observe this ticket's effect
except through that counter; the self-golden training block above is what
shows the weight now moves.

Evidence files: `.scratch/simulator-correctness/bench-output/ticket06-default-fleet-frozenw-after.txt`,
`ticket06-default-fleet-zerow-after.txt` (before = `ticket01-default-fleet-frozenw.txt` /
`ticket01-default-fleet.txt`, already committed).

### Full suite

`ruff check` / `ruff format --check` / `mypy` clean. Full pytest suite green
in the isolated verification worktree (this branch has several other tickets'
work in flight in the shared working tree concurrently; verification was done
in isolation rather than in-place for a clean signal — see
`git-pathspec-commit-stages-worktree` in project memory for why, and for the
staging technique used to land only this ticket's hunk in
`tests/unit/test_monte_carlo_policy.py` without disturbing another session's
concurrent, uncommitted edits to the same file).
