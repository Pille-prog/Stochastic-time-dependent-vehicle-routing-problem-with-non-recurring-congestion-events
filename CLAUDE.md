## Context gathering (obligatorio antes de implementar)

Antes de implementar cualquier feature o fix (toda skill de desarrollo, `/implement` incluido), contextualizarse es obligatorio:

- **GitNexus MCP** — `query`/`context` para símbolos y flujos, `impact` antes de tocar código.
- **DeepWiki MCP** — `ask_question` sobre las librerías involucradas.

Sin ese contexto no se genera código.

El índice y sus instrucciones viven en `AGENTS.md` (bloque `<!-- gitnexus:start/end -->`). Reindexar siempre con `npx gitnexus analyze --index-only` — mantenerlo fresco tras cada merge. Sin ese flag, `analyze` reinyecta su bloque en `CLAUDE.md` (y regenera skills); si el bloque reaparece aquí, moverlo de vuelta a `AGENTS.md` — este archivo es la brújula metodológica y no debe ser pisado por herramientas.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
