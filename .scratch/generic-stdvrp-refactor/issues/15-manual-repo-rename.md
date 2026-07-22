# 15 — Manual repo rename (user action)

**What to build:** The rename approved during ticketing, split out of ticket 02 because it requires actions outside an agent session: the GitHub CLI is not installed, and the local folder cannot be renamed while a session holds it as its working directory.

**Blocked by:** None — user can do it any time. Nothing depends on it.

**Status:** ready-for-agent (user)

- [ ] GitHub repo renamed to `stdvrp_orchestrator` (Settings → General → Repository name, or `gh repo rename stdvrp_orchestrator`). GitHub redirects the old URL automatically.
- [ ] Local folder renamed `stdvrp_orquestator` → `stdvrp_orchestrator` (close editors/sessions first; folder is inside OneDrive, so let sync settle afterwards)
- [ ] `git remote -v` afterwards; if the remote still points at the long historical name, optionally `git remote set-url origin https://github.com/Pille-prog/stdvrp_orchestrator.git`
