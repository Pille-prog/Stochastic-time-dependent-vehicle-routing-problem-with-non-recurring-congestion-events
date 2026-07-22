# 01 — Tag legacy monolith and establish data conventions

**What to build:** The permanent reference point for all characterization work: the untouched legacy script frozen under a git tag, and the convention by which instance data enters the repo without being committed.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Git tag `legacy-monolith` exists on a commit containing the unmodified legacy script, pushed to origin
- [ ] `data/` directory is gitignored; a placeholder file inside documents the expected Chengdu inputs (links file, morning and afternoon speed files)
- [ ] Root README documents where the data comes from and where to place it
- [ ] No code changes of any kind
