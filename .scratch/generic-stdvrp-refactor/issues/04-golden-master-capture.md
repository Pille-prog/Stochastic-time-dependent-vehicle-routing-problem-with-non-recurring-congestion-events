# 04 — Golden master capture from the legacy script

**What to build:** The frozen behavioral reference that guards the whole refactor: exact episode outcomes of the original script running on the fixture with fixed seeds, checked by an automated test.

**Blocked by:** 03.

**Status:** ready-for-agent

- [ ] A minimal shim parameterizes ONLY the three hardcoded data-file paths of the legacy script — nothing else changes (ADR-0001)
- [ ] Golden master captured on the fixture: per-seed episode total cost and the four components (distance, delay, earliness, overtime), plus the W trajectory for a small training run
- [ ] A pytest test re-runs the legacy script on the fixture and matches the stored values exactly
- [ ] Captured values and the capture procedure committed alongside the fixtures
