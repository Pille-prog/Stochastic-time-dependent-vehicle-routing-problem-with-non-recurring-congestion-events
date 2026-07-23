# 08 — Parallel evaluation and final test

**What to build:** Run the Trainer's evaluation blocks and final test on a
persistent `ProcessPoolExecutor` (16 logical cores on the target machine).
Training stays sequential — W is a serial dependency.

**Blocked by:** 02, 03.

**Status:** open

- [ ] A persistent pool whose worker initializer loads the world once per
      worker from the ticket-03 binary cache; episodes are submitted as
      (seed, W, overrides) tasks. Windows `spawn` is the required start
      method; nothing may rely on fork semantics.
- [ ] Seed-ordered aggregation: results are collected and reduced in the same
      seed order as the serial loops, so evaluation means and `results.json`
      are bit-identical to serial execution regardless of completion order.
- [ ] Serial fallback (worker count 1 or a config/env switch) for debugging
      and CI, exercised by tests; the pool is exercised by at least one
      multi-worker test on the fixture.
- [ ] Tier 1 gate: fixture `results.json` bit-identical serial vs parallel.
- [ ] Benchmark note in Comments: full-experiment wall-clock at 1, 8, 16
      workers on the real dataset.
