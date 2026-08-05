# 0001 — Mapa del sistema dominado

**Fecha:** 2026-08-01 · **Lección:** 0001-mapa-del-sistema.html · **Resultado:** quiz 3/3

## Qué quedó consolidado

- Los dos circuitos (decidir por epoch / aprender por episodio) y su mapa a archivos:
  `tokenizer.py → network.py → transformer_policy.py (_sweep/learn) → trainer.py`.
- Una pasada de encoder por epoch; el QHead es lo que corre m veces.
- El target de `learn` es el retorno global `U_t − costo_hundido`, compartido por los
  m vehículos del epoch (semilla de la divergencia D1).
- Factibilidad = máscara `+inf` antes del argmin (ADR-0007), no aprendizaje.

## Zona de desarrollo próximo

Listo para la mecánica cuantitativa de `learn()`: varianza del target, escalado,
Adam vs. warm start (D1, D2, D4, D6 del estudio). La L02 debe usar números concretos
del config (`neural_learning_rate=3e-4`, K=4, batch=32, ε=0.1, escala ≈ 1.8e5).

## A revisar más adelante

- Aún no se ha tocado en profundidad: el simulador (`Model.transition_function`),
  la estructura de costos, ni Gate A — candidatos para L03+.
