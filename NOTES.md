# Notas del profesor

## Preferencias del alumno

- Español siempre; términos técnicos sin traducir.
- Le gustan los entregables "rich markdown" con diagramas mermaid y tablas
  implementación-vs-teoría. El documento central vive en
  `reference/estudio-dominio-transformer-stdvrp.md`.
- Perfil: desarrollador senior (coderhub.cl), cómodo con código; el dominio nuevo es
  RL/ADP para VRP estocástico.

## Contexto operativo

- El código a estudiar está en `origin/FEATURE_DEV_total_code_refactorization`
  (NO en main). Worktree de solo lectura en el scratchpad de la sesión
  (`.../scratchpad/fernando-branch`); si no existe, recrearlo con
  `git worktree add <scratchpad>/fernando-branch origin/FEATURE_DEV_total_code_refactorization`.
- Fernando corre su propio ultracode review — no lanzar code reviews propios.
- El propio repo trae una investigación excelente y con fuentes verificadas:
  `docs/research/rl-methodology-for-stdvrp.md` (fecha 2026-07-22). Los flaws F1–F14 que
  lista son anteriores al transformer; varios siguen vigentes en la ruta neuronal.
- Workspace excluido de git vía `.git/info/exclude` (no tocar el repo de Fernando).

## Estado de la enseñanza

- 2026-08-01: workspace creado; estudio de dominio + lección 0001 (mapa del sistema)
  entregados. Quiz L01: 3/3 (ver learning-records/0001).
- 2026-08-01: lección 0002 (anatomía de learn: señal/ruido del target, baseline de costo
  hundido, sin replay, Adam vs. warm start, tabla de firmas diagnósticas) entregada.
  UX: componente .twocol/.panel agregado a course.css tras feedback (columnas separadas
  con encabezado; font .72em para evitar cortes). Próxima: lección 0003 — "cómo diseñaría
  la literatura el experimento de diagnóstico" (Gate A, corr(Q, minutos), ε=0, n-step).
