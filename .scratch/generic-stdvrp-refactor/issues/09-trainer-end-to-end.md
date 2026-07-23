# 09 — Trainer end-to-end and experiment outputs

**What to build:** The complete experiment as a single command: Trainer runs the training loop with periodic evaluation and best-W tracking, then the final test, writing results and plots — all driven by ExperimentConfig alone, reproducing the golden master on the fixture.

**Blocked by:** 08.

**Status:** resolved

- [x] Trainer runs train + periodic evaluation + final test entirely from ExperimentConfig (no hardcoded horizon, seeds, or the static-policy plot lookup table — all moved to config)
- [x] A full fixture run reproduces golden-master values; results and the training plot are written to a per-run output directory
- [x] One documented command runs the Chengdu experiment; CI runs a tiny smoke-sized version

## Answer

`Trainer` (`src/stdvrp/training/trainer.py`) ports `training_and_testing.training_model`/`test_model`: the training loop (warm-up-lr quirk preserved), evaluation blocks every `test_frequency` episodes with best-W tracking, and the final test over configured seed/vehicle/action tables, writing `results.json` + the training plot to a per-run directory. `ExperimentConfig` gained `test_action_counts`/`test_seeds`/`test_vehicle_counts` (the legacy `test_model` tables, now in `experiments/chengdu/config.yaml`) and nullable `train_exploration_seed_offset`/`train_repair_seed_offset` (golden-capture seeding convention; null = legacy nondeterminism).

Verification: `test_trainer_run_reproduces_the_whole_golden_master` drives one config-assembled `Trainer.run()` through the captured protocol on the full local dataset — W trajectory, per-seed eval costs and every test episode match bit-for-bit (ADR-0001 addendum venue: golden equality runs on the full data, not the mini fixture). CI runs `tests/test_trainer_smoke.py` (2-episode end-to-end on the 44-copy mini-fixture world via `Trainer.from_config`) plus 8 stub-based loop-logic unit tests. The documented command is `uv run python experiments/chengdu/run.py` (README).

Deviations documented in the ADR-0001 ticket-09 addendum: best-W fallback to the final trained W when no evaluation block ran (legacy would crash on `Best_W = []`), `inf` best-cost sentinel replacing `1e11`, and the three legacy mean-time report metrics (`mean_delay_time`/`mean_earliness_time`/`mean_overtime`) not ported — ticket 07's Model never exposed them and the golden master does not pin them.
