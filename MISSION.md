# Misión

## Por qué

Este repo es de Fernando (hermano de Guillermo). Fernando refactorizó el proyecto en torno a DDD
y está incorporando una policy neuronal con Transformers (`TransformerMonteCarloPolicy`, rama
`FEATURE_DEV_total_code_refactorization`) para el STDVRP con eventos de congestión no recurrentes.
**El transformer no está aprendiendo** durante el entrenamiento. Fernando lanzó su propio
code review multi-agente (ultracode) — nosotros NO lo repetimos; nuestro rol es distinto.

## Objetivo de aprendizaje

Que Guillermo:

1. **Entienda el código actual** — el pipeline completo: tokenizer → TokenEncoder → QHead →
   sweep por vehículo → `learn` con retornos Monte Carlo — al nivel de poder leerlo y
   cuestionarlo con Fernando.
2. **Identifique divergencias entre la implementación y la literatura científica pionera**
   del dominio (RL para VRP dinámico/estocástico: Ulmer, Powell, Bertsekas, Kool, Mnih…),
   con hipótesis concretas de por qué el transformer no aprende.

## Éxito se ve como

Guillermo puede explicar, sin mirar el código, (a) cómo fluye una decisión y un update de
gradiente por el sistema, y (b) nombrar las 3–5 divergencias más plausibles vs. la teoría,
con la fuente primaria que respalda cada una — y discutirlas con Fernando de igual a igual.

## Restricciones

- Idioma: español (términos técnicos en su forma original).
- Entregables "rich markdown" con diagramas de flujo (mermaid) — ver `reference/`.
- No modificar el código de Fernando; el estudio es de solo lectura (worktree en scratchpad).
