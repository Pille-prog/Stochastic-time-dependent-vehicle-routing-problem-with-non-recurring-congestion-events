# 08 — Parallel evaluation and final test

**What to build:** Run the Trainer's evaluation blocks and final test on a
persistent `ProcessPoolExecutor` (16 logical cores on the target machine).
Training stays sequential — W is a serial dependency.

**Blocked by:** 02, 03.

**Status:** resolved

- [x] A persistent pool whose worker initializer loads the world once per
      worker from the ticket-03 binary cache; episodes are submitted as
      (seed, W, overrides) tasks. Windows `spawn` is the required start
      method; nothing may rely on fork semantics.
- [x] Seed-ordered aggregation: results are collected and reduced in the same
      seed order as the serial loops, so evaluation means and `results.json`
      are bit-identical to serial execution regardless of completion order.
- [x] Serial fallback (worker count 1 or a config/env switch) for debugging
      and CI, exercised by tests; the pool is exercised by at least one
      multi-worker test on the fixture.
- [x] Tier 1 gate: fixture `results.json` bit-identical serial vs parallel.
- [x] Benchmark note in Comments: full-experiment wall-clock at 1, 8, 16
      workers on the real dataset. **Measured at 1 and 2**: 8 and 16 workers
      need 72 GB and 136 GB of resident world on a 31 GB machine (one world per
      worker, 8.02 GB measured), so they are not runnable here, let alone
      measurable — see the benchmark.

## Comments

### Resolution (2026-07-27)

`src/stdvrp/training/episode_pool.py` (new) turns the Trainer's two
embarrassingly parallel phases into a *batch of requests*: `EpisodeRequest`
(seed plus the two per-episode overrides), `EpisodeWorld` (the loaded world plus
the config scalars, with the one `episode_kwargs()` both Episode runners take)
and `EpisodePool` (a persistent `ProcessPoolExecutor` whose initializer loads
one world per worker). `Trainer.train`'s evaluation blocks and
`Trainer.final_test` now build request tuples and hand them to
`_run_evaluation_batch`, which runs them here or on the pool — through the same
`EpisodeWorld.run_episodes` either way.

**Seed-ordered aggregation is `Executor.map`'s submission-order contract**, not
a re-sort: `map` yields results in the order the tasks were submitted however
the workers interleave (confirmed against CPython's implementation — the futures
are held in a deque and yielded front-first), so every mean, std and
`results.json` entry is reduced in exactly the serial loops' seed order.

**The whole final-test table is one batch.** `final_test` submits every
(action count, seed) cell at once and slices the results back apart per action
count with a `strict=True` zip, so a mis-slice raises rather than silently
mispairing a cost with a seed (`test_final_test_walks_the_configured_tables`
now gives every cell a distinct cost, which is what makes that assertion bite).
One batch also means the pool is never drained and refilled between action
counts, and the cheap `actions=2` Episodes can fill the tail left by the
expensive `actions=50` ones.

**Spawn, always.** `multiprocessing.get_context("spawn")` is passed explicitly
rather than left to the platform default: it is the only option on Windows, and
choosing it everywhere means nothing can quietly come to depend on `fork`
inheriting parent state. A spawned worker starts empty, so it loads its own
world in the pool initializer — from the ticket-03 binary cache, which is why
`cache_dir` is mandatory when `worker_count > 1` (without it every worker would
re-parse 88 speed CSVs and the 907 MB path file, 24 minutes apiece — measured
below). `EpisodePool.for_worker_count` raises that as a `ValueError` *before*
`Trainer.from_config` loads the world, so a bad flag combination fails in
milliseconds rather than after a long parse; `experiments/chengdu/run.py`
rejects `--no-cache` with more than one worker the same way.

**Workers are persistent, and spawned lazily.** One executor serves every batch
of a run (`test_one_executor_serves_every_batch`), so a worker's world load is
paid once per run rather than once per batch or per episode. CPython spawns a
worker only when a task arrives and no idle worker is free, so a batch smaller
than `worker_count` starts only as many workers as it has Episodes — which
matters here because each worker costs a whole world of memory (see the
benchmark).

**Serial fallback, and why it is also the default.** `worker_count=1` returns no
pool at all and the Trainer keeps every Episode in this process, running the very
same `EpisodeWorld.run_episodes` the workers run — the debugging and CI path, and
`Trainer.from_config`'s default, so every existing caller and test is unaffected.
`experiments/chengdu/run.py` defaults to it too, which is a **change from this
ticket's first draft** (it defaulted to `os.cpu_count()`): the benchmark below
measures one loaded world at **8.0 GB resident**, so "one worker per core" on
this 16-core machine would ask for ~136 GB and thrash a 32 GB host into
uselessness. Worker count is a memory budget and only the operator knows theirs,
so they say: `STDVRP_WORKERS` once per machine, `--workers` per run. The flag is
resolved after parsing rather than as an argparse default, so a malformed
`STDVRP_WORKERS` cannot break `--help`.

**Progress while a batch runs.** Batching the whole final test would otherwise
have replaced six per-action-count log lines with hours of silence, so
`EpisodePool.run` / `EpisodeWorld.run_episodes` take an optional `on_progress`
callback (called with the completed count, in request order, on both paths) and
`final_test` logs every tenth of the batch. Pinned by
`test_the_final_test_reports_progress_while_it_runs`.

**The unit-test seam moved with the code.** `tests/unit/test_trainer.py` stubs
`run_evaluation_episode` in `stdvrp.training.episode_pool` now (training is
still stubbed in `stdvrp.training.trainer`), because that is where the call
lives; its assertions — seed sequences, per-episode overrides, the W each
Episode runs with — are unchanged, and two new cases pin that a pool takes every
batch and is shut down even when the run raises.

**A failed run shuts the workers down.** `Trainer.run` closes the pool in a
`finally`, so a crashed training loop cannot leave N processes each holding a
world; a closed pool starts fresh workers if it is used again
(`test_a_closed_pool_starts_fresh_workers_for_a_later_batch`). A worker that
dies (or an initializer that raises) breaks the pool and the `BrokenProcessPool`
propagates — deliberately not caught: quietly finishing the run serially would
turn one out-of-memory worker into an hours-longer run with no explanation.

**CI runs the pool.** `tests/test_parallel_evaluation.py` is neither marked nor
skipped, so the multi-worker cases run on Linux CI too. Spawn re-imports the
parent's `__main__` in each child, which under `pytest` is the console-script
launcher — its body is guarded by `if __name__ == "__main__"` and the child
imports it as `__mp_main__`, so nothing re-runs. (On Windows `__main__` is a
`.exe` wrapper and multiprocessing skips that re-import entirely.)

**`scripts/benchmark_episodes.py` lost its private `World`** — the same five
fields and the same `episode_kwargs()` as `EpisodeWorld` — so the benchmark now
times the object the experiment actually runs on rather than a lookalike. It
also gained the `--workers` sweep this ticket's benchmark is measured with
(ticket 01's script is where measurement lives), which times the final test
through `Trainer.final_test` itself rather than rebuilding its table, and reuses
`project_full_run` unchanged: the per-episode means it feeds in are already
divided by the parallelism achieved, so the same unit-tested arithmetic answers
"how long at N workers?". `experiments/chengdu/parallel_scaled.yaml` is its
scaled config, next to ticket 01's `baseline_scaled.yaml`.

**`Trainer` now takes the `EpisodeWorld`** (`Trainer(world, episode_pool=...,
log=...)`, `config` read off the world) instead of the four world components.
The first draft kept the old signature and `from_config` unbundled a world it
had just built, only for `__init__` to rebuild it — a round trip flagged in
review as a data clump. Three other call sites (the golden-master test, the
rebaseline script, the Trainer unit tests) were updated; two of them got
shorter, and nothing outside read the four attributes it replaces. The Trainer
is also a context manager now, so code that drives `train()`/`final_test()`
directly — benchmarks, tests, notebooks — closes the pool the way `run()` does.

### Benchmark (2026-07-27, this machine: 16 logical cores, 31.3 GB RAM)

Reproduce (the world cache must be warm — see below)::

    uv run python scripts/benchmark_episodes.py \
        --config experiments/chengdu/parallel_scaled.yaml \
        --project experiments/chengdu/config.yaml --workers 1,2

**The world is 8.02 GB resident, and that — not the core count — is the
ceiling.** Measured in a fresh process that loads the real Chengdu world from
the ticket-03 binary cache and does nothing else: 43.8s to load, **8.02 GB**
working set, 8.09 GB after running one Episode (3,617,604 cached paths; the cold
parse that built the cache took 1458.3s = 24m18s). Since every worker holds its
own copy:

```
workers   worlds resident   fits in 31.3 GB?
      1      8.0 GB (parent only)          yes
      2     24.1 GB (parent + 2)           yes, tight
      3     32.1 GB (parent + 3)           no
      8     72.2 GB                        no
     16    136.4 GB                        no
```

**So the ticket's "1, 8, 16 workers" is not measurable on this machine, and
would not be runnable on it either** — 8 workers need 72 GB. The sweep therefore
runs 1 and 2, which is what a 32 GB host can actually do, and the parallel
efficiency it measures (97%) is the number that extrapolates. The parent's own
world is idle while the pool runs, so Windows trims it; the practical
requirement is closer to `worker_count x 8.0 GB`.

**Sweep** (`parallel_scaled.yaml`: the real world, 32 evaluation Episodes and 8
final-test seeds at each of the two projection endpoints, evaluated with a W
trained by 5 real training Episodes):

```
                     1 worker            2 workers          speedup
pool start                0.0s               59.5s
evaluation (32 ep)      384.6s              198.5s            1.94x
final test @2  (8 ep)    92.1s               46.7s            1.97x
final test @50 (8 ep)   551.8s              282.3s            1.95x
parallelizable total   1028.5s              527.5s            1.95x
```

**1.95x on two workers is 97% efficiency** — the phases really are
embarrassingly parallel, and the per-task overhead (one pickled W plus one
`EpisodeResult` per Episode, against 6-70 s of work) is invisible. The pool's
59.5s startup is two workers unpickling 8 GB apiece, paid once for the whole
run, not per batch: against the ~7,600s of evaluation the full experiment does,
it is 0.8%.

**Projected full run** (`config.yaml`: 100 training, 10 blocks x 50 evaluation,
6 action counts x 50 test seeds; middle action counts interpolated between the
measured @2 and @50; `project_full_run`, unit-tested, fed the *effective*
per-episode times above):

```
  1 worker     19034.1s   (5h17m14s)
  2 workers    10398.2s   (2h53m18s)     -2h24m
```

Training (100 x 11.55 s/ep = 1155s) and the world load (36.0s) stay serial by
construction, so Amdahl caps this config at about 16x however many workers a
machine can feed; two workers already collect 1.95 of that.

**These absolute numbers are post-ticket-04, pre-05/07.** Ticket 04 recorded a
deliberate per-episode regression (dense geometry matrices; the payoff lands
when 05 and 07 vectorize on top of them), and it is visible here: 12.0 s/ep
evaluation against ticket 01's 5.27 s/ep baseline. Ticket 08 owns the *ratio*,
which is independent of that; re-run the sweep after 05/07 land and the
projection will drop with the per-episode times while the 1.95x stands.

**Reproduction gotcha.** `default_cache_dir()` resolves `%LOCALAPPDATA%`, and
Windows Store Python redirects that per package — the cache these runs used sits
under `...\Packages\PythonSoftwareFoundation.Python.3.11_...\LocalCache\Local\
stdvrp\world_cache`, not `%LOCALAPPDATA%\stdvrp\world_cache`. Two interpreters
therefore do not necessarily share one cache; set `STDVRP_WORLD_CACHE_DIR` to
pin it.

**Where the next parallel win is.** Nothing in this ticket can lift the memory
ceiling: the 8 GB is the `ShortestPathCache` dict plus `TravelTimeModel`'s
per-arc/minute dicts, and every worker needs all of it. Putting that data in
`multiprocessing.shared_memory` as numpy arrays — which ticket 04's
`EpisodeGeometry` already suggests the shape of — would make worker count a
*core* budget again and is the natural follow-up ticket; it is a representation
change across the Model and the world cache, so it does not belong here.
