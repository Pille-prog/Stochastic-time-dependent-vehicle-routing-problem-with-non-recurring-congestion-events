# 02 — Deduplicate the final test's identical episodes

**What to build:** Stop re-running bit-identical episodes in
`Trainer.final_test`. With per-seed Generators (ticket 13 of the refactor),
every one of the `test_episodes: 50` iterations for a given (seed, vehicles,
action count) is deterministic and identical — the trainer docstring already
admits "the mean equals a single episode's value". 15,000 episodes → 300.

**Blocked by:** 01.

**Status:** open

- [ ] `final_test` runs each (action count, seed) episode **once**; the
      reported per-seed metrics are that episode's values (the mean of k
      identical values is the value itself — division-order float noise from
      the legacy `sum/k` must not leak into the report, so compare against the
      self-golden and document if the old report differed in last-bit ulps).
- [ ] `test_episodes` stays in `ExperimentConfig` for config compatibility but
      is documented as inert (or validated == deduplicated behavior); decide
      and record which in the ticket Comments.
- [ ] Tier 1 gate: `results.json` for a fixture run is bit-identical to the
      self-golden capture (or the documented ulp-level report difference is
      pinned by a test).
- [ ] Benchmark note in Comments: measured final-test speedup.
