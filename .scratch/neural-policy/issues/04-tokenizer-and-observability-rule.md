# 04 — The tokenizer, and the observability rule as a test

**What to build:** The pure function `State` + `EpisodeGeometry` → tokens, and
the executable test that the Policy cannot see anything else. Writes
**ADR-0006**.

**Blocked by:** 03

**Status:** resolved

## The tokens

Three kinds, one tensor each. Every value is a **raw fact the State or the
geometry already holds**. Nothing here is a cost, a polynomial or a bin.

```
client_token[j]  = [tw_start, tw_end, tw_end - tau,
                    minutes_from_vehicle[0..m-1],
                    path_length_from_vehicle[0..m-1]]
vehicle_token[i] = [standing, completing_service, minutes_to_depot,
                    observed_velocity[0..n-1]]
global_token     = [tau, shift_end - tau, episode_end - tau,
                    n_pending, n_vehicles]
```

`claimed` is **not** a token field — it enters at the head (ticket 06), so one
encoder pass serves the whole per-vehicle sweep.

**The forbidden list, verbatim from spec.md** — these must not appear, in any
form, however tempting:

> `earliness_cost`, `delay_cost`, `future_delay`, `overtime_cost`, the
> 400/500/600 earliness bins, `clients_left²·time` and its siblings, the
> `late_count / 13` normalizer, the `310`/`350` depot-idle literals.

If a future reader wants a feature added here, the question to answer first is:
*is this a fact, or is it an opinion about the facts?* Only facts.

## Normalization

Raw minutes and raw minute-counts differ by orders of magnitude across fields.
Standardize per field with **fixed, config-derived scales** (horizon length,
episode length), not with running statistics — running statistics make an
Episode's tokens depend on which Episodes came before, which breaks the
per-seed reproducibility the paired comparison depends on. Record the chosen
scales in the module docstring; they are part of the token definition the way
the legacy's normalizers were part of the feature definition.

## The observability rule, as a test

This is the ticket's real deliverable.

- [x] The tokenizer's signature takes **only** `State`/`TrainingSnapshot`, the
      time windows, the `EpisodeGeometry` and config scalars. It cannot reach
      `EpisodeVelocities`, `congested_arcs`, `TravelTimeModel` or `FleetRoutes`
      because they are not arguments.
- [x] An executable test that pins that: construct the tokenizer, assert its
      module's imports and its call signature admit nothing from the forbidden
      list. A structural test, not a docstring — the failure mode this guards
      against is a future ticket "just adding" a congestion field because it
      would obviously help.
- [x] Property test: tokens are a **pure function** of the snapshot. Same
      snapshot twice ⇒ bit-identical tokens.
- [x] Property test: **permutation equivariance.** Reordering
      `clients_not_visited` reorders the client tokens identically and changes
      nothing else. This is the property that makes attention the right tool
      and an MLP the wrong one, so it should be asserted, not assumed.
- [x] Property test: variable set sizes. 60 and 180 Clients, 4 and 8 vehicles,
      all produce well-formed token tensors.

## ADR-0006 — What the Policy is allowed to see

Records:

- the rule itself (spec.md, "The observability rule, precisely");
- **why the live traffic feed was rejected** despite
  `docs/research/rl-methodology-for-stdvrp.md` ranking it the single
  highest-leverage change (#1, F1). The reason is not that it would not help —
  it probably would. It is that the transformer must be **exactly as
  congestion-blind as the linear baseline** for the comparison to attribute a
  win to the approximator rather than to information one side was handed;
- **why `EpisodeGeometry.average_minutes` is permitted**: an offline historical
  prior, not an observation of this Episode, and the identical object the
  baseline reads. Without it the network has no notion of distance;
- the **only admissible congestion-aware arm**: the fleet's *shared observation
  memory* — what these vehicles measured, pooled. A dispatcher may aggregate its
  own vehicles' reports; it may not read the world's velocity field.

## CONTEXT.md

Add a clause under **Policy** stating the observability rule. Glossary only —
no implementation detail.

## Acceptance

- [x] Predicted self-golden diff: **zero.** New module, nothing existing is
      called differently.

## Comments

Resolved. `src/stdvrp/policies/tokenizer.py` — a pure `tokenize(snapshot,
geometry, time_windows, *, horizon_start_minute, shift_end_minute,
episode_end_minute) -> Tokens` function, deliberately not a class: unlike
`FeatureExtractor` there is no per-Episode setup to cache, so a plain function
is the honest shape. `Tokens` is a frozen dataclass of exactly the three arrays
spec.md names (`client_tokens`, `vehicle_tokens`, `global_token`); `claimed` is
not a field, per spec.md decision 6 — it enters at ticket 06's per-vehicle head.

**Signature, and why `depot` isn't one of its five parameters:** `State`/
`TrainingSnapshot`, `TimeWindows`, `EpisodeGeometry`, and the three config
clocks. No `depot` argument: `EpisodeGeometry.build`'s own contract places the
depot at column 0 always ("columns are depot first, then this Episode's
Clients"), already relied on unchanged elsewhere (`FeatureExtractor._overtime_costs`'s
`depot_column = 0`), so `minutes_to_depot` reads `vehicle_minutes[:, 0]`
directly — one fewer argument for the observability test to have to admit.

**The observability rule test** (`tests/unit/test_tokenizer.py::TestObservabilityRule`,
3 tests): `test_signature_admits_only_the_permitted_inputs` pins the parameter
set to an **exact** allow-list (not just "nothing forbidden") — a future ticket
that adds a sixth argument must edit this test, not slip past it, which is
exactly the failure mode the ticket calls out. `test_module_imports_nothing_forbidden`
parses `tokenizer.py`'s own AST (not a docstring claim) and asserts neither the
forbidden names (`EpisodeVelocities`, `congested_arcs`, `TravelTimeModel`,
`FleetRoutes`) nor the forbidden modules
(`stdvrp.simulation.episode_velocities`, `stdvrp.traffic.travel_time_model`,
`stdvrp.simulation.fleet_routes`) appear anywhere in its imports.
`test_no_parameter_or_annotation_names_a_forbidden_type` is the redundant third
check most likely to survive a future refactor that restructures the import
list without touching parameter annotations.

**Normalization** — fixed, config-derived, documented in the module docstring
per the ticket's ask (not running statistics, so tokens stay independent of
Episode history — spec.md, "Why the paired comparison is valid"):
`horizon_length = shift_end_minute - horizon_start_minute` scales every
minute-valued field whose natural range is "within one shift" (`tw_start`,
`tw_end`, `tw_end - tau`, `minutes_from_vehicle`, `minutes_to_depot`, `tau`,
`shift_end - tau`); `episode_length = episode_end_minute - horizon_start_minute`
scales `episode_end - tau`, whose range extends past the shift end.
`path_length_from_vehicle` (kilometres, not minutes) has no minute-based
config scalar to borrow — it is divided by `horizon_length` too, documented
honestly as a magnitude-parity choice, not a unit conversion. `n_pending` is
divided by `total_clients = len(geometry.columns) - 1` (this Episode's total
Client count, read off the `EpisodeGeometry` already in hand, not a new
config scalar) — turning a raw count into "fraction of this Episode's demand
still pending", bounded like the scaled time fields. Left unscaled,
deliberately: `standing`/`completing_service` (already 0/1), `observed_velocity`
(already O(1) km/min — verified against `TravelTimeModel`'s own docstring:
"speeds in km/min"), and `n_vehicles` (a small raw integer; this problem's
fleet sizes run single digits to the low tens per `ClientGenerator.generate`,
already the same order of magnitude as the scaled fields, unlike `n_pending`
which reaches into the hundreds).

**Property tests** (`tests/unit/test_tokenizer.py`, hypothesis,
`max_examples=50, deadline=None, derandomize=True`, matching this repo's
existing convention in `test_monte_carlo_policy.py`): `TestPureFunction` calls
`tokenize` twice on the same snapshot and asserts bit-identical output —
trivially true by construction (no internal state, no randomness), but
asserted rather than assumed, per the ticket's own framing.
`TestPermutationEquivariance` reorders `clients_not_visited` by a
Hypothesis-drawn permutation and asserts `client_tokens` reorders identically
while `vehicle_tokens`/`global_token` do not move at all.
`TestVariableSetSizes` is not Hypothesis-driven — four `pytest.mark.parametrize`
cells at the ticket's exact numbers (60/180 Clients × 4/8 vehicles) over a
vectorized dense-geometry builder (`make_dense_world`), asserting the three
tensors' shapes and that every value is finite. Also added, ahead of the
property tests: `TestHandComputedExample` (4 tests), one fully worked
by-hand world (one vehicle away from the depot, two Clients, hand-computed
expected values for all three tensors) — the fastest thing to catch a
transposed field or a wrong divisor, and the property tests alone would not
have caught either (they hold for any consistent field order/scale, correct
or not).

**Circular-import landmine** (ticket 03's Comments, not this ticket's to fix):
`test_tokenizer.py` is the first test module to import
`stdvrp.policies.tokenizer` directly, so it needs the same defensive
`import stdvrp.simulation` before that import that `test_torch_support.py`
already carries, for the same reason.

**Predicted self-golden diff: zero, verified**
(`tests/test_self_golden.py` + `tests/test_world_cache_self_golden.py`, 7/7
passed) — a new module nothing existing calls. Full suite green (4062 passed,
5 skipped, 3 deselected), `mypy` clean on `stdvrp.*`, `ruff check`/`ruff
format --check` clean on the new files (the 4 pre-existing `E501` hits in
`monte_carlo.py`/`model.py` predate this ticket, verified via `git stash`
against the clean HEAD — not touched here). `CONTEXT.md` gained one sentence
under **Policy** stating the rule, glossary-only, cross-referencing this
ticket and ADR-0006 — no implementation detail, per the ticket's ask.

`/code-review` (both axes, against `3656ff8`): no hard standards violations
(no banned Powell-vocabulary terms, no seam added without a second
implementation per ADR-0002, ADR/docstring conventions matched against
ADR-0005 and `feature_extraction.py`). Spec axis found one real defect, fixed
before landing: the module docstring claimed `tw_start`/`tw_end` shared
`tau`'s clock zero point (subtracting `horizon_start_minute`), but the code
divided them by `horizon_length` unadjusted — the two absolute-clock fields
most likely to be compared against `tau` downstream were the two the fix
missed. Corrected in both directions (code now subtracts
`horizon_start_minute` from `tw_start`/`tw_end` in `_client_tokens`; the
docstring's normalization section reworded to state the rule once, for all
three absolute-clock reads, instead of leaving it ambiguous which fields a
trailing parenthetical covered) and the hand-computed example's expected
values updated to match. Standards axis also flagged single-letter locals
(`m`, `n`) in `_client_tokens`/`_vehicle_tokens` against `feature_extraction.py`'s
spelled-out precedent — renamed to `number_vehicles`/`n_observed_velocities`.
Both fixes reverified: `ruff check`/`format --check` and `mypy` clean, all 13
tokenizer tests still pass.
