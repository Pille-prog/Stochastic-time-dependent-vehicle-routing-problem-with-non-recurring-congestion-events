# 15 — The myopic base leaves the network

**What to build:** `c(s, v, a)` — the projected cost the tokenizer already
computes exactly — stops being a weight inside `arc_embed` and becomes an
additive term **no gradient can reach**. The network is left with one job:
the residual.

```
Q(s, v, a) = c(s, v, a)  +  W · φ(s, v, a)
             ↑ tokenizer     ↑ the only term with parameters, W = 0 at init
             a float, not a Parameter
```

**Blocked by:** 14 (the null this ticket freezes is the null over the shared
action set, not over ~151 candidates)

**Status:** resolved

## Why: four measured attempts share one signature

Ticket 08's §4 is the strongest evidence in the effort, and it was read as four
separate results when it is one:

| attempt | from `minutes` (a poor start) | from `cost` (a good start) |
|---|---|---|
| level gain 100 | better every column | **worse** every column |
| dueling `V` + centred advantage | best single block of any arm | **+10.73%** mean over blocks |
| Huber knee at 0.02 | — | **+129%** by episode 140 |
| `cost` warm start itself | −32.7% at init | training then **spends** it |

**Each makes the optimizer more effective, and each only helps when the
starting policy is bad.** That is not four findings about three techniques. It
is one finding about where `c(s, a)` lives: in ordinary trainable weights, so
an optimizer that moves faster destroys it faster. `WARM_START_WEIGHTS["cost"]
= (1, 0, 1, 1, 0, 1)` is literally the projected cost of the assignment, and
gradient descent was being handed a licence to unlearn an arithmetic identity
the tokenizer had already computed exactly.

Structural unreachability is the fix, and it is not a config flag:
`requires_grad = False` on `arc_embed` row 0 would keep the coupling and keep
the encoder pinned as an identity map (below). Taking `c` out of the network
entirely is what makes the guarantee hold and what unlocks the rest.

## What it unlocks

The warm start's exactness required **every `TransformerEncoderLayer` to be an
exact identity at init** (`out_proj` and `linear2` zeroed). Three layers of
attention that compute nothing. Once `c` is additive and outside:

- `arc_embed` row 0 stops being special → ordinary Xavier like every other row.
- The encoder no longer has to be an identity → **real attention at
  initialization**, which is what makes ticket 17's frozen-encoder arm a random
  *feature map* rather than a linear embedding.
- `W = 0` gives the cost-greedy policy **by construction**, exactly, with no
  weight tuned to make it so. Training can only add.

## What dies, and what that means

| knob | after |
|---|---|
| `neural_warm_start` | **dead on this path.** `WARM_START_WEIGHTS` stops being an initialization and becomes the definition of `c` — the same six-field vector, in a place where nothing can move it |
| `neural_level_gain` | **dead.** The level is solved exactly in one ridge solve (ticket 16); there is nothing left to walk and nothing to accelerate |
| `neural_huber_delta` | **dead.** Least squares, not Huber. The heavy tail is handled by ticket 16's abort exclusion instead |

None are deleted — they still describe the path ticket 08 measured, and its
numbers stay citeable. They are documented as no-ops of the new path, in the
modules that own them.

## Naming — and one thing this is *not*

This is a **myopic base**, not a warm start. A warm start is a point you move
away from; nothing moves away from this. Naming it a warm start invites exactly
the failure this ticket exists to close.

It is also **not Powell's post-decision state**, and `CONTEXT.md` follows
Powell's vocabulary explicitly. `c(s, a)` is built from `tau`, the time windows
and `EpisodeGeometry.average_minutes` — a *static historical prior*, so it is a
**projection**, not the immediate cost the simulator will actually charge.
`Q = c + W·φ` is a residual VFA over a known myopic base; `W·φ` absorbs both
the future cost-to-go *and* the error in the projection. Writing "post-decision"
anywhere near this would plant a wrong idea that outlives the ticket.

## Observability — unchanged, and provably

Every field of `c` is already in the arc token and was already admitted by
spec.md decision 1's 2026-08-01 amendment: computed from `tau`, the time
windows, `EpisodeGeometry.average_minutes` and the simulator's hardcoded rate
constants (configuration, not observation — ADR-0006's dated clarification).
Moving where a number is *added* changes nothing about what is *read*.

- [x] `tokenize`'s five arguments unchanged; ADR-0006's structural test passes
      untouched.

## Work

- [x] `c(s, v, a)` computed per candidate (including the synthetic depot row)
      and added outside `QHead`. Not a `Parameter`, not in any `state_dict`.
- [x] `arc_embed` row 0 → ordinary Xavier; drop the identity-at-init
      construction from `TokenEncoder` and the warm-start construction from
      `QHead._init_weights`.
- [x] Verify `W = 0` reproduces ticket 14's frozen null **to the cent**, on the
      mini fixture and on the real dataset. If it does not, the decomposition
      is not equivalent and nothing downstream is interpretable.
- [x] Document the three dead knobs where they live.

## Acceptance

- [x] `W = 0` mean cost == ticket 14's frozen null, exactly.
- [x] `network.py`'s "A dueling decomposition — tried, measured, and rejected"
      and "The level term" docstrings updated to say **why they are moot now**
      rather than deleted — both were rejected for reasons that this ticket
      removes, and a future reader will otherwise re-derive them.
- [x] Predicted self-golden diff: **zero.**

## Comments

**Landed as planned**, with one refinement over the ticket's own sketch:
`DEPOT_WARM_START_PENALTY` is not deleted — it moves from a hand-set column of
`QHead.linear` to an addend `TokenEncoder.forward` applies when it builds
`Embeddings.depot_cost`, so `c(s, v, depot)` reproduces the exact same number
the pre-ticket-15 architecture computed via `is_depot`'s weight. Ticket 14's
frozen null (`action_set_m2_50.py`'s arm 1, `DEPOT_WARM_START_PENALTY = 1.0`)
depends on that addend, so it had to move with `c`, not disappear.

`QHead` keeps its exact forward-pass code (`linear(x) + level +
layer2(ReLU(layer1(x)))`); only `_init_weights` changes, and it gets simpler:
`linear.weight`/`linear.bias`/`layer2.weight`/`layer2.bias` are now zero with
no exception (previously `linear` was zero *except* two hand-set columns).
`layer1` stays Xavier-random, unchanged, for the deadlock reason the module
docstring already documented. `Embeddings` gained two fields, `cost` and
`depot_cost` — `TokenEncoder.forward` computes them straight from
`arc_tokens`/`depot_arc_tokens` (a registered, non-persistent buffer holding
`WARM_START_WEIGHTS["cost"]`, never a `Parameter`), and
`transformer_policy.py`'s `_score` adds them to `QHead`'s (now purely
residual) output — the one place `c` and `W · φ` actually meet, exactly once,
outside `QHead`.

**Verified `W = 0` reproduces ticket 14's frozen null, both ways the ticket
asked for**
(`.scratch/neural-policy/results/myopic_base_null_50.py`,
`myopic_base_null_50.json` alongside it):

- **Mini fixture:** the pre-ticket-15 architecture (a sibling git worktree
  checked out at this ticket's parent commit) against this ticket's
  architecture, same 50 `evaluation_seeds`, `neural_warm_start="cost"`,
  `device="cpu"`, zero training. Both: mean **461.287099**, and every one of
  the 50 per-seed costs agreed — `max |old - new| = 0.0000000000`. Not
  "close"; the two architectures produce bit-identical decisions end to end.
- **Real dataset**, same 50 `evaluation_seeds` `action_set_m2_50.py`'s arm 1
  used: this ticket's architecture reads **3365.092529**, against ticket 14's
  recorded **3365.09** — matching to the cent, the deviation being exactly
  the rounding `action_set_m2_50.py`'s own printed number carried.

The equivalence is not a numerical coincidence: it follows from `Q(s, v, a) ==
c(s, v, a)` holding *by construction* (linear/layer2 are the zero tensor, so
`W · φ == 0` regardless of what the encoder or `layer1` compute — see
`network.py`, "The myopic base"), so the two architectures' random weights
never influence `Q` at init at all. The comparison above is the empirical
check that this reasoning has no gap, not the reasoning itself.

**Verification:** `mypy`/`ruff` clean (project-wide, not just the changed
files); full suite (`uv run pytest -q -m "not golden"`) 4279 passed, 3
`golden`-marked deselected (up from ticket 14's 4274 — five new tests, no
regressions); `tests/test_self_golden.py` run explicitly, 6/6 passed
(predicted-zero self-golden diff confirmed — this ticket never touches
`monte_carlo.py`, `action_set.py`, or `tokenizer.py`); `-m neural`, 119 passed
(up from 114).

`tests/unit/test_network.py` and `tests/unit/test_transformer_policy.py`
rewritten around the new decomposition: `TestWarmStart`/`TestWarmStartWeights`
→ `TestQHeadResidualIsZeroAtInit` (the residual is exactly zero for arbitrary
input, independent of any real world) + `TestUntrainedQEqualsMyopicBase` (the
combined `Q` matches `c` read off the token, end to end, argmin included);
`TestIdentityAtInit` → `TestRealAttentionAtInit` (asserts the *opposite* —
the encoder is no longer an identity map); `TestWarmStartIsNotBehindAnActivation`/
`TestCostFeaturesAreWired` → `TestEncoderGradientAtInit` (no gradient reaches
any encoder parameter on the first backward pass now, not just the arc cost
columns — a strictly stronger structural guarantee than before — but gradient
does reach it after one optimizer step, mirroring `TestClaimedIsWired`'s
two-phase pattern). `TestDeviceParity` extended to check `Embeddings`'
cross-device numerics directly, since `QHead`'s own output is trivially
zero-on-every-device now and no longer carries that comparison's weight.

**`/code-review`, two axes, both run and both fed back:** the Spec pass caught
a real gap the checklist above had hidden — "The level term" section's
Acceptance-required "why it's moot now" note had been written for "A dueling
decomposition" only and never actually added to "The level term" itself,
leaving `config.py`'s `neural_level_gain` docstring pointing at a note that
did not exist. Added. The Standards pass flagged three things: a stray
`.. code-block::` reST directive with no precedent elsewhere in this
docstring-heavy file (reverted to the plain-text convention already used two
paragraphs away); `myopic_base_null_50.py` hardcoding a path into this
session's own ephemeral scratchpad worktree, making the script unrunnable by
anyone once that directory is gone (rewritten: the script now checks the new
architecture against a recorded constant by default, with the worktree
comparison behind an optional `--old-src` flag for anyone who wants to
re-derive that constant from scratch); and `TestDepotWarmStart` being the one
test class this pass missed renaming despite the ticket's own naming
argument, alongside `DEPOT_WARM_START_PENALTY` itself, which was judged
correctly *not* worth renaming — it is a public name three other files and
one already-committed historical script (`action_set_m2_50.py`) reference by
exact spelling, and `WARM_START_WEIGHTS` (its sibling in `tokenizer.py`) is
deliberately kept under its old name for the same citability reason (module
docstring, "The myopic base": "the same six-field vector ticket 08
introduced"). Renamed the test class; left the constant.
