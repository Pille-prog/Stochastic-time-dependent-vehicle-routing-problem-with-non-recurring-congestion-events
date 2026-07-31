# 09 — Honest documentation

**What to build:** Three places where the prose claims something the code does
not do. No behavior changes. Closes B18 and B19, and records B8's inertness.

This ticket exists because of what ticket 04 demonstrates: a name that lies is
the mechanism by which the same bug gets written three times. These three have
not caused a defect yet. Fix them before they do.

**Blocked by:** 01

**Status:** resolved

## B18 — "N arcs" is really "the last N observed velocities"

`simulation/state.py:38`. **The code is right and needs no change.** `begin_arc`
adds an entry per new arc and `resample_arc` adds one per decision epoch, even on
the same arc — a sliding window of the last *N* observed velocities, whatever
their origin. That is the intended design.

Only the wording is wrong. The docstring ("velocities observed on the last
`n_arcs` arcs") and the parameter name (`n_arcs` on `State`, `n_observed_arcs`
in config) promise *N distinct arcs* where the design delivers *N velocity
observations in time*. Measured: of 4928 insertions, 1586 were new arcs and 3342
were resamples of the same arc — with `n_observed_arcs: 3` the window covers
roughly 4–6 recent minutes, not three distinct arcs.

- [x] Correct the docstring to describe a recency window.
- [x] Rename the parameter in `State` and `ExperimentConfig` (and every
      committed YAML) so it no longer says "arcs".
- [x] Note for the modeling effort: as a congestion proxy the temporal reading is
      arguably *better* than the per-arc one — a congestion event lasts tens of
      minutes, which a time window captures and a distinct-arc window would not.
      Relevant if B4 ever connects `mean_velocities`.

## B19 — the std window's right endpoint is not the first observation

`traffic/travel_time_model.py:285,292,299`. The docstring justifies the shifted
endpoints (418/542, 658/842, 958/1082) as "the last/first minutes *observed*
around each data gap". In the real archive the left ones (418/658/958) are, but
the first observed minutes after each gap are **540/840/1080**, not 542/842/1082
— the right anchor sits one observation past the edge.

- [x] **Fix the docstring, not the anchors.** ADR-0001's phase-2 fix 2 already
      made this call, describing them as "the (off-by-two, preserved)
      endpoints". The decision to keep them is taken and documented; only this
      docstring still sells them as an empirical justification. Make it say what
      the ADR says: a preserved legacy quirk.
- [x] **Do not move the anchors.** That is a modeling change — it alters every
      stochastic velocity drawn in those windows on real multi-day data (median
      10% difference in the stored std, p90 ~25%), with nobody claiming 540 is
      better than 542. The left shift is inert anyway, since the data gap makes
      it equivalent.

## B8 — record that distance damping is inert

ADR-0001's fix 7 states its purpose was to resurrect the depth-3 damping factor
(0.73) that was dead code. The saturation that same fix introduced kills it
again under the shipped bounds: with `p ~ U(0.3, 0.4)`, multipliers saturate at
depth 1 for 68% of draws, at depth 2 for 88%, and at depth 3 **always**. Per
epoch, more than half the network sits at ≤40% of free-flow speed and about
three quarters of those arcs carry the identical 0.4 regardless of whether they
are 1, 2 or 3 hops from the epicentre.

- [x] **ADR-0001 addendum**: under the shipped 0.3/0.4 bounds the distance
      damping table is inert, and the 0.73 factor fix 7 resurrected cannot be
      observed. So that nobody reads fix 7 and believes damping is live.
- [x] No code change. Correcting it is a modeling decision (saturate? rescale?
      move the bounds?) and belongs to the effort that reopens the generator.

## Predicted self-golden diff

**Exactly zero on all three blocks.** Docstrings, an ADR addendum, and a
parameter rename that carries the same value cannot move a float.

The rename is the only part that touches executable code. If it moves anything,
a call site was reading the old name through a path the rename missed — which is
a bug introduced by this ticket, not a finding.

## Evidence required

Zero diff. Full suite green. The renamed parameter present in every committed
YAML with its value unchanged.

## Comments

### Resolution (2026-07-30)

All three blocks landed, commits `061343e` + `e603c87` (rename + docstrings +
ADR addendum) and `970275a` (B18's third checklist item, added after
`/code-review`'s Spec pass flagged it missing).

- **B18.** `State.__init__`'s `n_arcs` and `ExperimentConfig.n_observed_arcs`
  both renamed to `n_observed_velocities`; docstring at `state.py:38` now reads
  "Sliding window of the last n_observed_velocities velocity observations...
  one entry per decision epoch, not per distinct arc". Renamed at every call
  site (`episode.py`, `model.py`'s two internal reads, `episode_pool.py`) and
  every committed YAML (`experiments/chengdu/{config,baseline_scaled,parallel_scaled}.yaml`,
  `tests/fixtures/chengdu_mini/config.yaml`) — value `3` unchanged everywhere.
  The two remaining `n_arcs` mentions (`config.py`'s and `trainer.py`'s module
  docstrings) are deliberately untouched: they describe the *legacy
  monolith's* own hardcoded variable name (confirmed against the
  `legacy-monolith` tag), not the renamed field. The third checklist item (a
  forward note for whoever connects `mean_velocities`, B4) is now at
  `feature_extraction.py`'s module docstring and `StateFeatures.mean_velocities`
  — the place that ticket would actually read.
- **B19.** `_build_speed_std_lookup`'s docstring in `travel_time_model.py`
  rewritten to state the off-by-two as a preserved quirk per ADR-0001's
  already-made call, not an empirical justification. The anchors
  (418/542, 658/842, 958/1082) are byte-identical before and after.
- **B8.** ADR-0001 gained a dated addendum recording fix 7's depth-3 damping
  as inert again under the shipped 0.3/0.4 bounds. No code change, as scoped.

**Predicted self-golden diff (exactly zero) holds**: every non-doc, non-YAML
hunk is a bare identifier rename (`n_arcs`→`n_observed_velocities`,
`last_arc`→`last_slot` in `model.py`); the YAML values are unchanged. Verified
in an isolated `git worktree` at the final commit (uncontaminated by the two
other simulator-correctness sessions concurrently working tickets 05/07/08 in
the same shared working tree): `mypy` clean, `ruff check`/`ruff format --check`
clean, full suite `-m "not golden"` **3025 passed, 6 deselected** (0:07:55).

**A rename this wide needs a verification pass before commit, not after**: the
first commit's `n_arcs`→`n_observed_velocities` rename missed
`tests/unit/test_monte_carlo_policy.py`'s four `State(...)` call sites — that
file was mid-edit by a concurrent session's ticket 05 at the time, so a plain
`git diff` on it showed both sessions' hunks entangled together and the miss
wasn't obvious until an isolated-worktree test run raised
`TypeError: State.__init__() got an unexpected keyword argument 'n_arcs'`.
Landed as a follow-up commit (`e603c87`) built via git plumbing
(`hash-object` + a scratch `GIT_INDEX_FILE` + `commit-tree` + `update-ref`)
directly against the branch tip, specifically so the fix could land without
touching the shared working tree or index the other session had files staged
in — see `git-pathspec-commit-stages-worktree` in project memory for why a
pathspec'd `git add`/`git commit` on an entangled file is unsafe here.
