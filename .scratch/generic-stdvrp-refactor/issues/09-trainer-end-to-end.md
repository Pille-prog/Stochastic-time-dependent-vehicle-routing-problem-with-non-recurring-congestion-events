# 09 — Trainer end-to-end and experiment outputs

**What to build:** The complete experiment as a single command: Trainer runs the training loop with periodic evaluation and best-W tracking, then the final test, writing results and plots — all driven by ExperimentConfig alone, reproducing the golden master on the fixture.

**Blocked by:** 08.

**Status:** ready-for-agent

- [ ] Trainer runs train + periodic evaluation + final test entirely from ExperimentConfig (no hardcoded horizon, seeds, or the static-policy plot lookup table — all moved to config)
- [ ] A full fixture run reproduces golden-master values; results and the training plot are written to a per-run output directory
- [ ] One documented command runs the Chengdu experiment; CI runs a tiny smoke-sized version
