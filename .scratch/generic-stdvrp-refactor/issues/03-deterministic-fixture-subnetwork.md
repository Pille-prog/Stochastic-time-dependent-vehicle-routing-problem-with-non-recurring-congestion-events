# 03 — Deterministic fixture sub-network

**What to build:** A small committed sub-network (tens of nodes) extracted reproducibly from the real Chengdu data, so every test in every later ticket runs in seconds on any machine, including CI, without the full dataset.

**Blocked by:** 01. **Also blocked on the user**: the Chengdu CSVs' location and size are still unknown (asked three times during grilling/spec) — extraction cannot start without them.

**Status:** ready-for-agent

- [ ] Extraction script produces the fixture deterministically from the full data, with the command documented
- [ ] Fixture (links subset + matching morning/afternoon speed subsets) committed under the test fixtures directory, total well under 1 MB
- [ ] Fixture network is strongly connected enough for routing, includes the depot node (node 0), and supports at least a handful of Clients
- [ ] The unmodified legacy script can consume the fixture files (same format/columns as the real data)
