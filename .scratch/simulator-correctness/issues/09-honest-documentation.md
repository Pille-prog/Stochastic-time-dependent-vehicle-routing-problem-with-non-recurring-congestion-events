# 09 — Honest documentation

**What to build:** Three places where the prose claims something the code does
not do. No behavior changes. Closes B18 and B19, and records B8's inertness.

This ticket exists because of what ticket 04 demonstrates: a name that lies is
the mechanism by which the same bug gets written three times. These three have
not caused a defect yet. Fix them before they do.

**Blocked by:** 01

**Status:** open

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

- [ ] Correct the docstring to describe a recency window.
- [ ] Rename the parameter in `State` and `ExperimentConfig` (and every
      committed YAML) so it no longer says "arcs".
- [ ] Note for the modeling effort: as a congestion proxy the temporal reading is
      arguably *better* than the per-arc one — a congestion event lasts tens of
      minutes, which a time window captures and a distinct-arc window would not.
      Relevant if B4 ever connects `mean_velocities`.

## B19 — the std window's right endpoint is not the first observation

`traffic/travel_time_model.py:285,292,299`. The docstring justifies the shifted
endpoints (418/542, 658/842, 958/1082) as "the last/first minutes *observed*
around each data gap". In the real archive the left ones (418/658/958) are, but
the first observed minutes after each gap are **540/840/1080**, not 542/842/1082
— the right anchor sits one observation past the edge.

- [ ] **Fix the docstring, not the anchors.** ADR-0001's phase-2 fix 2 already
      made this call, describing them as "the (off-by-two, preserved)
      endpoints". The decision to keep them is taken and documented; only this
      docstring still sells them as an empirical justification. Make it say what
      the ADR says: a preserved legacy quirk.
- [ ] **Do not move the anchors.** That is a modeling change — it alters every
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

- [ ] **ADR-0001 addendum**: under the shipped 0.3/0.4 bounds the distance
      damping table is inert, and the 0.73 factor fix 7 resurrected cannot be
      observed. So that nobody reads fix 7 and believes damping is live.
- [ ] No code change. Correcting it is a modeling decision (saturate? rescale?
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
