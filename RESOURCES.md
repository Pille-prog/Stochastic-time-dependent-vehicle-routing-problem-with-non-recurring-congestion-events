# Recursos

Fuentes primarias para el estudio "transformer que no aprende" en el STDVRP.
Las marcadas ✅ fueron verificadas con URL primaria por el research note del propio repo
(`docs/research/rl-methodology-for-stdvrp.md`, 2026-07-22); ⭐ = máxima prioridad de lectura.

## Dominio: RL / ADP para VRP dinámico y estocástico

- ⭐✅ **Hildebrandt, Thomas & Ulmer — "Opportunities for reinforcement learning in stochastic dynamic vehicle routing"** — <https://arxiv.org/abs/2103.00507> — EL survey del dominio; nombra explícitamente la descomposición por vehículo ("coarse policies") y las VFA lineales/neuronales.
- ⭐✅ **Chen, Ulmer & Thomas — DQN para same-day delivery (SDVRP)** — <https://arxiv.org/abs/1910.11901> — la receta neuronal más cercana a este repo que sí funcionó: MLP + experience replay + ε decay 1.0→0.01 + Adam con lr decreciente; +22% sobre policy paramétrica.
- ✅ **Powell — post-decision state / ADP unificado** — <https://arxiv.org/abs/2002.06238> y libro RLSO <https://castle.princeton.edu/rlso/>.
- ✅ **Bertsekas — Multiagent rollout (Prop. 2.1, mejora garantizada, O(s·m))** — <https://arxiv.org/abs/1910.00120>.
- ✅ **Ulmer, Goodson, Mattfeld & Hennig — offline VFA + online rollout** — DOI 10.1287/trsc.2017.0767.

## Policies neuronales constructivas (y por qué no transfieren directo)

- ✅ **Kool, van Hoof & Welling — "Attention, Learn to Solve Routing Problems!"** — <https://arxiv.org/abs/1803.08475> — REINFORCE + greedy-rollout baseline; supuestos: euclídeo, tamaño fijo, determinista.
- ✅ **Kwon et al. — POMO** — <https://arxiv.org/abs/2010.16011> — baseline compartida multi-arranque.
- ✅ **Nazari et al. — RL para VRP** — <https://arxiv.org/abs/1802.04240>.
- ✅ **SVRPBench** — <https://arxiv.org/abs/2505.21887> — POMO/AM degradan >20% bajo shift distribucional con tiempos estocásticos (preprint 2025).
- ✅ **SED2AM — transformer time-dependent (Edmonton/Calgary)** — <https://arxiv.org/abs/2503.04085>.

## Estabilidad del deep RL basado en valor

- ⭐ **Mnih et al. 2015 — "Human-level control through deep reinforcement learning"** (Nature 518:529–533) — experience replay (decorrelación) + target network como LOS dos estabilizadores canónicos del Q-learning profundo. Registro: <https://www.researchgate.net/publication/272837232_Human-level_control_through_deep_reinforcement_learning>.
- ✅ **Lagoudakis & Parr — LSPI/LSTD-Q** — <https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf> — estimadores sin learning rate.
- ✅ **Ryzhov, Frazier & Powell — step sizes (Robbins–Monro)** — <https://arxiv.org/abs/1407.2676>.
- **Sutton & Barto — Reinforcement Learning: An Introduction (2ª ed.)** — <http://incompleteideas.net/book/the-book-2nd.html> — Monte Carlo vs TD, varianza del retorno, semi-gradiente (base conceptual; libre acceso).

## Del propio repo (leer primero)

- ⭐ `docs/research/rl-methodology-for-stdvrp.md` — la investigación de Fernando; flaws F1–F14 y ranking de opciones.
- `docs/research/reading-path.md` — su ruta de lectura para `network.py`/`transformer_policy.py`.
- `docs/simulator-review.md` — bugs confirmados vs. quirks preservados del simulador.
- `.scratch/` — specs/tickets del esfuerzo neural-policy (spec.md citado por los docstrings).
