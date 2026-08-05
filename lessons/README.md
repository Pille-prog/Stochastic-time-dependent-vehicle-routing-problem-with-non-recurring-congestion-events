# Aula: por qué el transformer no está aprendiendo — resumen ejecutivo

> **Para:** Fernando. **De:** Guillermo (con Claude como profesor).
> **Rama:** `aula/transformer-que-no-aprende` — solo material de estudio, no toca `src/` ni `tests/`.
> **Código estudiado:** `FEATURE_DEV_total_code_refactorization` @ `c875ef4`
> (`tokenizer.py`, `network.py`, `transformer_policy.py`, `neural_episode.py`, `trainer.py`, config de Chengdu).
> Detalle completo con fuentes: [`../reference/estudio-dominio-transformer-stdvrp.md`](../reference/estudio-dominio-transformer-stdvrp.md).

## Lo que está bien construido (y no hay que tocar a ciegas)

- Equivarianza por permutación de los client tokens; regla de observabilidad pineada por test (ADR-0006).
- Factibilidad como máscara dura `+inf` antes del argmin (ADR-0007), nunca heurística aprendida.
- Warm start miópico exacto (`Q == minutos/horizon` en init) con la solución correcta al deadlock de gradiente.
- Disciplina RNG: 4 streams por episodio derivados de la seed (arregla el F5 del baseline para la ruta neuronal).
- Normalización fija derivada del config (preserva la comparación pareada por seed).

## Causas probables, en orden de sospecha

| # | Causa | Mecanismo | Chequeo barato |
|---|---|---|---|
| **D1** | **Sin crédito por acción** (crítica) | Cada par `(epoch t, vehículo v)` se regresa contra el MISMO `target_t = U_t − costo_hundido` — retorno global de miles de unidades (+ cola terminal 40000, F10). La diferencia real entre candidatos es ~minutos: ≈5e-5 en escala del target. La red puede aprender V(s) perfecto y el argmin seguir siendo ruido. | Si `last_loss` baja pero el costo greedy no mejora → D1 confirmada. Gate A ayuda: calibración ↑ + costo plano = aprende valor, no ranking. |
| **D2** | **Sin experience replay** (crítica) | Un batch por episodio (K=4 pasadas sobre las mismas ~T·m muestras correlacionadas) y se descarta. Cada episodio es otra instancia → interferencia catastrófica episodio a episodio. Mnih 2015 y Chen/Ulmer/Thomas (el éxito más cercano a este problema) usan replay; el spec lo dejó como "knob a medir". | Loss oscilando sin tendencia entre episodios. Probar buffer de N episodios muestreando mezclado. |
| **D3** | **ε = 0.1 fijo, uniforme sobre ~150 candidatos** | Un desvío exploratorio absurdo contamina los retornos MC de TODOS los epochs anteriores del episodio (no hay bootstrap que lo corte). Es el F4 del baseline, heredado sin decay (la literatura decae 1.0→0.01). | Run corto con ε=0: si mejora notablemente, D3 pesa. |
| **D4** | **Adam borra el warm start** | El paso de Adam es ~lr (3e-4) por parámetro sin importar la magnitud del gradiente → los ceros exactos (out_proj, linear2, columnas de fondo de layer2) se mueven desde el update 1; el prior "vecino más cercano" puede morir antes de que exista señal aprendible. | `corr(Q(v,j), minutos(v,j))` en checkpoints de episodio 1/10/100. Si cae a ~0 rápido → D4. Mitigaciones: lr menor para los pesos estructurales, o SGD al inicio. |
| **D5** | **Features aún ciegas a congestión** (techo) | Los arcos (`minutes_from_vehicle`, `path_length`) son promedios estáticos del CSV; lo único dinámico son las `observed_velocity` y los relojes. Es el F1 de tu propio research note ("the binding constraint"): aunque D1–D4 se arreglen, el margen sobre el baseline lineal queda acotado. | Ya está rankeada #1 en `docs/research/rl-methodology-for-stdvrp.md`. |

Secundarias: **D6** escala del target vs. escala del prior (todos los Q convergen al mismo valor grande, el ranking queda en residuos); **D8** el criterio de convergencia (patience → 3 cortes de lr) lee medias de pocos seeds con varianza enorme — puede declarar CONVERGED por ruido.

**No son el problema:** falta de target network (targets MC, no bootstrap), Huber, la convención argmin de costo, re-encodear por muestra en `learn` (ineficiencia documentada, no bug).

## La historia causal más plausible

ε contamina retornos (D3) → targets ya ruidosos y sin contrafactual por acción (D1) → Adam
funde el prior miópico en los primeros cientos de updates (D4) → cada episodio nuevo
sobrescribe lo anterior (D2) → argmin ≈ ruido: costo plano o peor que el baseline. Y aunque
todo eso se arregle, D5 acota el techo.

## Las 5 preguntas para tu ultracode review

1. ¿Qué muestra Gate A? (calibración ↑ + costo plano = D1/D6; nada sube = D2/D4)
2. ¿Sobrevive el warm start? (`corr(Q, minutos)` por checkpoint)
3. ¿Qué pasa con ε=0 puro?
4. ¿El loss baja mientras el costo no? (la firma más informativa)
5. ¿Por qué target MC completo y no n-step/TD, si `rewards[t+1]` ya se captura por transición?

## Material del aula

- [`0001-mapa-del-sistema.html`](0001-mapa-del-sistema.html) — los dos circuitos (decidir/aprender) y su mapa a archivos.
- [`0002-anatomia-de-learn.html`](0002-anatomia-de-learn.html) — la mecánica de `learn()` con números reales y la tabla de firmas diagnósticas.
- [`../reference/estudio-dominio-transformer-stdvrp.md`](../reference/estudio-dominio-transformer-stdvrp.md) — estudio completo: diagramas, divergencias D1–D9 con severidad y fuentes primarias.
- [`../RESOURCES.md`](../RESOURCES.md) — bibliografía verificada (Hildebrandt/Ulmer, Chen/Ulmer/Thomas, Mnih, Bertsekas, Kool, POMO…).
