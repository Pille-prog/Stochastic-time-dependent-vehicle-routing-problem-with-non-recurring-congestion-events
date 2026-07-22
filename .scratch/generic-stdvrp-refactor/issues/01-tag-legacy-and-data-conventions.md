# 01 — Tag legacy monolith and establish data conventions

**What to build:** The permanent reference point for all characterization work: the untouched legacy script frozen under a git tag, and the convention by which instance data enters the repo without being committed.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] Git tag `legacy-monolith` exists on a commit containing the unmodified legacy script, pushed to origin
- [x] `data/` directory is gitignored; a placeholder file inside documents the expected Chengdu inputs (links file, morning and afternoon speed files)
- [x] Root README documents where the data comes from and where to place it
- [x] No code changes of any kind

## Comments

Resolved 2026-07-21. Tag `legacy-monolith` is an annotated tag on `main` (1e8d42e), where the script is byte-identical to the feature branch. Data confirmed: `link.csv` (0.4 MB) + `speed[6XX/7XX]_[0/1].csv` (~23 MB each) in the `Mega city` folder one level above the repo; also `all_shortest_paths.csv` (907 MB, precomputed path cache — relevant to ticket 06).
