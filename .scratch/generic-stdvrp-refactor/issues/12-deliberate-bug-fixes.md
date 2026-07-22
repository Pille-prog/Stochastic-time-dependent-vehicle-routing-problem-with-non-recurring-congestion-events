# 12 — Deliberate, documented bug fixes

**What to build:** Phase 2 begins: triage the defect list accumulated while porting (tickets 05–09 record every oddity they preserve), and fix each one as a deliberate behavior change with its own test.

**Blocked by:** 09, 10, 11.

**Status:** ready-for-agent

- [ ] Every fix follows: failing test first → fix → entry in the ADR-0001 phase-2 change log
- [ ] Golden master re-baselined only where a fix intentionally changes outcomes, with the justification recorded
- [ ] Shadowed/dead legacy variants are NOT resurrected — they stay in the `legacy-monolith` tag (ADR-0001)
- [ ] Candidate list triaged with the user before fixing (some "bugs" may be intended research behavior)
