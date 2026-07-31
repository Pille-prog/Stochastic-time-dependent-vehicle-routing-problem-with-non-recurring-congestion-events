# 02 — The `TrainablePolicy` seam

**What to build:** Replace `Model.run_training_episode`'s
`isinstance(policy, MonteCarloPolicy)` with a `TrainablePolicy` protocol, so a
second trainable Policy is pluggable without `Model` knowing it exists. Add the
one field `TrainingSnapshot` is missing.

**Blocked by:** —

**Status:** resolved

## The leak

`model.py` currently does:

```python
if not isinstance(policy, MonteCarloPolicy):
    raise TypeError("training episodes require a MonteCarloPolicy")
...
policy.update_W(self.episode_states, self.episode_actions, self.episode_rewards)
```

`docs/research/rl-methodology-for-stdvrp.md` §6.3 names this and recommends
moving the loop to the `Trainer` outright. **That is not this ticket** — see
"Debt recorded" below.

## Work

- [x] `policies/base.py` gains `TrainablePolicy` (a `Protocol`, so
      `MonteCarloPolicy` satisfies it without inheriting): `decide_train(state)`
      and `learn(snapshots, actions, rewards)`.
- [x] `MonteCarloPolicy.update_W` gains `learn` as its protocol-facing name.
      Prefer an alias/rename over a wrapper — `update_W` is named in ADR-0001's
      change log and in three tickets, so whichever way it goes, the docstring
      must keep the legacy name findable.
- [x] `model.py` calls through the protocol. The `isinstance` and its
      `TypeError` go.
- [x] `TrainingSnapshot` gains **`vehicle_completing_service`**. This is the one
      field ticket 04's tokenizer needs that the snapshot does not already
      capture; the other five (`tau_episode`, `clients_not_visited`,
      `last_node_reached`, `vehicle_standing`, `observed_velocity`) are exactly
      right already. Copy it, do not alias it — `State` mutates it in place.
- [x] mypy is strict on `stdvrp.*` (`disallow_untyped_defs`,
      `disallow_incomplete_defs`). The protocol must type-check without
      `# type: ignore`.

## Blast radius (GitNexus, verified)

`Policy` upstream: 8 symbols, risk **LOW**, 4 direct.
`TrainingSnapshot` upstream: 10 symbols, risk **MEDIUM**, 6 direct.
`tests/unit/test_training_snapshot.py` and `tests/unit/test_monte_carlo_policy.py`
both assert on these directly.

## Debt recorded, not paid

Moving the training loop from `Model` to `Trainer` is architecturally correct
(`CONTEXT.md`: "**Trainer**: Runs training and evaluation episodes over the
Model to fit and compare Policies") and is what the research note recommends.
It is **out of scope here** because it relocates the exact code path
`test_self_golden` pins, which would turn this ticket's zero-diff prediction
from a structural guarantee into something that has to be demonstrated. Record
it as a follow-on effort; do not do it inside this one.

## Acceptance

- [x] Predicted self-golden diff: **exactly zero.** The loop does not move; a
      method is renamed and a type check is deleted. This is the strongest
      claim in the effort — if a rename moves a single float, it was not a
      rename.
- [x] Full suite green, mypy clean, no new `type: ignore`.

## Comments

Resolved. `MonteCarloPolicy.update_W`'s body was renamed to `learn` (its
`states` param renamed `snapshots` to match the protocol literally), with
`update_W = learn` kept as a true alias (same function object, not a wrapper)
so ADR-0001's change log and tickets 06/08/09 still find it; both names are
callable and both docstrings cross-reference the other. `TrainablePolicy`
(`policies/base.py`) declares `decide_train`/`learn` structurally —
`MonteCarloPolicy` satisfies it with no inheritance and no `# type: ignore`.
`Model.run_training_episode` drops the `isinstance`/`TypeError` gate entirely
and calls through `cast(TrainablePolicy, self.policy)` — `model.py` no longer
imports `MonteCarloPolicy` at all, only `TrainablePolicy` and `TimeWindows`.
`TrainingSnapshot` gained `vehicle_completing_service` (copied via
`tuple(state.vehicle_completing_service)`, same pattern as the other five
fields); no current read site touches it, ticket 04's tokenizer is the first.

Self-golden diff verified exactly zero (`tests/test_self_golden.py` +
`tests/test_world_cache_self_golden.py`, 7/7 passed) — the change is a pure
rename/alias plus a protocol/cast substitution, no arithmetic or control flow
moved. Full suite (4039 passed, 3 deselected) and `mypy`/`ruff check`/`ruff
format --check` all verified clean in an isolated `git worktree` at this
ticket's base commit (`d1ac8e5`), rather than the shared working tree — another
session was concurrently landing unrelated neural-policy ticket work (a
dependency add plus ticket 01's config changes) in the same directory while
this ticket was in flight, and briefly left the shared tree in a
non-representative broken intermediate state; the isolated worktree run is the
one that counts. `/code-review` (both axes) came back with no hard violations
— only judgement calls already anticipated by this ticket's own text (a
fourth `Policy`-family seam beyond ADR-0002's three, justified the same way
the ADR requires: a real second implementation, this effort's neural policy,
is planned) or explicitly out of scope (bundling
`snapshots`/`actions`/`rewards` into one type — pre-existing shape, not this
ticket's to fix). One docstring tense fix applied (`state.py`: "ticket 04's
tokenizer does" → "will", since ticket 04 hasn't landed yet).
