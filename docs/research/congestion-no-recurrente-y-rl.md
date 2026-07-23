# Congestión no recurrente en el STDVRP: fidelidad del modelo generativo, mejoras y estado del arte con RL

> **Estado:** nota de investigación, no una decisión. No se ha escrito ningún ADR a partir de ella.
> **Alcance:** el generador de eventos de congestión no recurrente (`src/stdvrp/congestion/generator.py`,
> `src/stdvrp/traffic/travel_time_model.py::_compute_event_probabilities`) y el capítulo 7.2 de la tesis
> (`Tésis_Pilleux_Fernando.md`, "Desarrollo del Simulador").
> **Fecha de la investigación:** 2026-07-23. Toda afirmación externa está anclada a fuentes primarias
> (PDF de la agencia emisora, página del editor, DOI, arXiv) con URLs en [Bibliografía](#7-bibliografía).
> **Anclas de línea:** el monolito legado `Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py` fue
> **borrado del árbol de trabajo** durante esta sesión (`git status` lo reporta como `D` en el índice).
> Las líneas que se citan de él se verificaron sobre `git show HEAD:Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`
> en el commit `636d28c`. Las líneas de `src/stdvrp/` se verificaron sobre el árbol de trabajo del
> 2026-07-23. Trate los nombres de símbolo como autoritativos y los números como pistas.
> **Complementa** a [`rl-methodology-for-stdvrp.md`](./rl-methodology-for-stdvrp.md), que cubre el
> *estimador* de RL (MC lineal, LSTD, rollout, VFA neuronal). Esta nota cubre el *modelo generativo* del
> entorno y el hueco de literatura. Los dos hallazgos se cruzan en un punto y se señala explícitamente.

---

## Pregunta

*¿El modelo de generación de congestión no recurrente implementado es fiel a lo que la ciencia del
transporte establece sobre incidentes, obras, clima y eventos especiales; qué habría que cambiar para
acercarlo a la realidad; y sigue siendo el STDVRP con congestión no recurrente resuelto con RL un
problema abierto en 2025-2026?*

---

## Resumen ejecutivo

- **Veredicto global (Q1): "simplificado pero defendible" en la estructura, con tres defectos que sí
  comprometen la validez de la afirmación de realismo.** El esqueleto —probabilidad por arco calibrada
  con datos reales, penalización multiplicativa de velocidad, duración aleatoria, propagación con
  atenuación por vecindario— es una abstracción razonable y está calibrado con datos, no inventado. Pero
  (i) el umbral de detección de "congestión" usa la **media de todo el día** como referencia
  (`travel_time_model.py:329`), de modo que lo que el código llama *no recurrente* es en gran medida
  **congestión recurrente de hora punta**; (ii) la propagación se hace **aguas abajo** siguiendo
  sucesores (`generator.py:76-107`), mientras que la teoría de flujo dice que las colas se propagan
  **aguas arriba**; (iii) hay **doble conteo**: la σ usada para muestrear velocidades es la desviación
  *entre los 44 días*, que ya contiene los días con incidentes, y encima se suman los eventos.
- **La magnitud del evento no está anclada a nada físico.** φ ~ U(0.1, 0.4) es un multiplicador de
  velocidad elegido a mano (`generator.py:66-68`); la literatura estándar (HCM 2010 Exhibit 10-17,
  SHRP 2 L08 Exhibit 36-16) parametriza incidentes por **fracción de capacidad residual según carriles
  bloqueados** (0.35 / 0.49 / 0.58… para 1 carril bloqueado en vías de 2 / 3 / 4 carriles), no por
  reducción directa de velocidad.
- **La duración es uniforme; la literatura es unánime en que no lo es.** `δ ~ U(30, max_duration)`
  (`generator.py:69`). La revisión primaria de Li, Pereira & Ben-Akiva (2018, *European Transport
  Research Review* 10:22) documenta que las duraciones de incidente se ajustan a **lognormal,
  log-logística, Weibull o F generalizada**, casi siempre mediante modelos AFT de riesgo, nunca uniforme.
  Una uniforme elimina la cola pesada, que es exactamente el fenómeno que hace difícil el problema.
- **El impacto del evento es lineal en la duración; en la realidad es cuadrático.** Con el diagrama de
  colas determinista de Morales (FHWA), el retardo total de un incidente es una **forma cuadrática en las
  duraciones** dividida por `2(S1−S3)`. Truncar la cola de duración y usar duración uniforme suprime
  precisamente los eventos que dominan el costo esperado.
- **Un `factor = 0.73` es código muerto.** `max_depth = 3` se pasa como `_reachable_nodes(node, 0,
  max_depth - 1)` (`generator.py:77`), y la recursión expande solo mientras `depth < 2`, así que la
  profundidad máxima alcanzada es 2. La rama `depth == 3 → 0.73` (`generator.py:91-92`) nunca se ejecuta.
  Esto importa porque la tesis justifica el umbral del 40 % diciendo que "los arcos adyacentes alcanzan
  penalizaciones de hasta 54,7 %" — 0.4/0.73 = 0.548, es decir el **tercer** vecindario, que el código
  nunca genera (el máximo real es 0.4/0.78 = 0.513).
- **El evento no es un incidente en un arco: es el bloqueo completo de dos estrellas de salida.** El
  generador congestiona *todas* las aristas salientes de ambos extremos del arco incidentado con el
  factor 1 (`generator.py:76-107`), no solo el arco afectado. El footprint espacial es mucho mayor que un
  incidente real y no está justificado por ninguna fuente.
- **Ironía de observabilidad:** la política dinámica de referencia *sí* reacciona a la congestión
  (umbral `v < 0.5·v̄`, legado `:1880-1894`), pero la política de Monte Carlo **no la ve**: el promedio
  de velocidades observadas se calcula y se descarta antes de entrar al vector de features (legado
  `:2596-2607`; ver también `rl-methodology-for-stdvrp.md` §"Bottom line"). El agente de RL está siendo
  entrenado a **ignorar** la variable exógena que da nombre a la tesis.
- **Veredicto de brecha (Q3): la brecha existe pero es más estrecha de lo que afirma la tesis, y hay que
  reformularla.** La combinación exacta *(multi-vehículo + ventanas de tiempo blandas + tiempos de viaje
  estocásticos dependientes del tiempo sobre red real + eventos no recurrentes explícitos con
  propagación espacial + política de RL)* no la encontré publicada. Pero cada par de esos ingredientes
  sí está resuelto, y **Chen, Imdahl, Lai & Van Woensel** (TR-C 172:105022, 2025; y TR-C 182:105387,
  2026) cubren RL + tiempos de viaje dependientes del tiempo *y* estocásticos, con generalización, en el
  mismo horizonte temporal en que se escribió la tesis. La contribución defendible ya **no** es "RL para
  STDVRP" (eso está siendo resuelto) sino "**RL bajo eventos no recurrentes explícitos, espacialmente
  correlacionados y parcialmente observables sobre una red urbana real**" — y para sostenerla hay que
  arreglar la observabilidad y el realismo del generador.

---

## 1. Qué hace el código hoy — reconstrucción del modelo generativo

### 1.1 Dónde vive

| Pieza | Legado (`HEAD:Main_Chengdu…py`) | Vivo (`src/stdvrp/`) |
|---|---|---|
| Probabilidad de evento por arco | `store_probability_for_event_of_all_arcs` `:1302-1325` | `travel_time_model.py::_compute_event_probabilities` `:294-335` |
| Generación de eventos | `create_random_unexpected_event_with_probability_and_2_nodes` `:1392-1450` | `congestion/generator.py::ArcProbabilityCongestionGenerator.generate` `:58-107` |
| Vecindario afectado | `get_all_node_starts` `:851-871` | `generator.py::_reachable_nodes` `:109-137` |
| Cadencia del sorteo | `model.transition_function` `:5850-5855` | `simulation/model.py:267-271` |
| Aplicación a la velocidad | `create_random_velocity` `:606-636` | `simulation/model.py:520-543` |
| Expiración / re-cálculo FIFO | `_next_congestion_end` `:5681-5702` | `simulation/model.py:499-517` |
| Parámetros | `main` `:6540-6545`, `model.__init__` `:5246-5254` | `config.py:49-51`, `:107-111` |

Existen cinco generadores alternativos en el legado (`create_random_unexpected_event` `:639`,
`create_one_random_unexpected_event` `:759`, `create_random_unexpected_event_with_radius` `:945`,
`create_random_unexpected_event_with_probability` `:1345`, `create_one_congestion` `:1452`). **Solo el de
"probability_and_2_nodes" se invoca** desde `transition_function` (`:5855`); el resto es código muerto.
El refactor lo documenta explícitamente (`generator.py:3-8`).

### 1.2 Calibración de la probabilidad por arco

`_compute_event_probabilities` (`travel_time_model.py:294-335`):

1. Agrega las observaciones de velocidad de cada uno de los **44 días** (archivos `speed[601..630]` y
   `speed[701..714]`) a **medias de 30 minutos por link**, filtrando `hour >= 8`.
2. Calcula `avg_speed` como la media por link **sobre todo el rango horario disponible** (`:314-317`;
   en el legado, `get_mean_of_all_intervals` `:1279-1281`).
3. Cuenta, por arco, el número de medias de 30 minutos que caen en el rango
   `0.1·avg_speed ≤ v ≤ 0.4·avg_speed` (`:329`).
4. Normaliza:

   ```python
   unit_of_time_of_congestions = max_congestion_duration / 60
   hours = 8 / unit_of_time_of_congestions          # travel_time_model.py:319-320
   probability[key] = probability[key] * 2 / (day_count * hours * 3)   # :334
   ```

Con `max_congestion_duration = 60` esto es `P_ij = n_ij · 2 / (44 · 8 · 3)`. La aritmética **es
internamente coherente**: el dataset cubre cinco ventanas de dos horas (3-5, 8-10, 12-14, 17-19, 21-23;
Guo et al. 2019), de las cuales las de `hour >= 8` son cuatro → **16 medias de 30 min = 8 horas
observadas por día**. Entonces

$$P_{ij} \;=\; \frac{2}{3}\cdot\frac{n_{ij}}{44\cdot 16}\cdot 2 \;=\; \frac{2}{3}\cdot\big(\text{fracción de horas observadas en que el arco estuvo "congestionado"}\big).$$

El factor `2/3` es la cifra de Falcocchio & Levinson (2015) que la tesis cita (§7.2, "hasta dos tercios
del tiempo total perdido en áreas metropolitanas corresponden a congestiones no recurrentes"), aplicada
como si fuera una proporción de *frecuencia de ocurrencia*.

`P_ij` es **constante a lo largo del día** y **heterogénea entre arcos** (calibrada arco a arco).

### 1.3 El generador de eventos

`ArcProbabilityCongestionGenerator.generate` (`generator.py:58-107`), invocado con
`minute_start = tau_episode`:

```python
for key in self.event_probability:                       # :60  un sorteo por arco, en orden de inserción
    probability_for_congestion = np.random.uniform(0, 1)  # :61
    if probability_for_congestion < self.event_probability[key]:   # :62
        velocity_penalization = np.random.uniform(lower_bound, upper_bound)   # :66-68
        state_time_elimination = np.random.uniform(30, max_congestion_duration)  # :69
        congested_arcs[(start, end)] = [φ, minute_start + δ]        # :71-74
        for node in [start, end]:                                   # :76
            node_starts, depth = self._reachable_nodes(node, 0, max_depth - 1)  # :77
            for node_start in node_starts:
                for affected_node in self.successors.get(node_start, []):       # :79-81
                    factor = {0: 1.0, 1: 0.83, 2: 0.78, 3: 0.73}[depth[node_start]]  # :85-92
                    congested_arcs[(node_start, affected_node)] = [φ/factor, minute_start + δ]  # :104-107
```

Propiedades exactas:

- **Un sorteo Bernoulli independiente por arco y por época.** No hay proceso de Poisson explícito, pero
  con `|L| = 5943` arcos y `P_ij` pequeña, la suma de Bernoullis independientes ≈ Poisson de intensidad
  `Λ = Σ_ij P_ij` por época. El número esperado de eventos por episodio **no está reportado en ningún
  lugar del código, de la tesis ni de los experimentos** (*no verificado*, ver §8).
- **Magnitud** φ ~ U(`congestion_lower_bound`, `congestion_upper_bound`). Los escenarios de la tesis
  (Cuadro 7.3) son `[0.1, 0.4]`, `[0.1, 0.3]`, `[0.1, 0.2]`, `[0.1, 0.1]`. Es un **multiplicador directo
  de la velocidad media**.
- **Duración** δ ~ U(30, `max_congestion_duration`) minutos. Con `max_congestion_duration = 60`, δ ~ U(30, 60);
  media 45 min, sin cola.
- **Extensión espacial:** la doble iteración sobre `[start, end]` y luego sobre `successors` hace que se
  congestionen **todas las aristas salientes** de todo nodo alcanzable en ≤ 2 saltos *hacia adelante*
  desde cualquiera de los dos extremos del arco incidentado. Los arcos cuyo nodo origen está a distancia
  0 reciben φ íntegro; los de distancia 1, φ/0.83; los de distancia 2, φ/0.78.
- **La rama `depth == 3 → factor 0.73` es inalcanzable.** `_reachable_nodes(node, 0, max_depth-1)` con
  `max_depth = 3` (`model.__init__` legado `:5254`; `generator.py:55`) expande solo mientras `depth < 2`
  (`:131`), de modo que `node_depth` ∈ {0, 1, 2}. El propio refactor dejó un `raise AssertionError` en el
  `else` (`:93-94`) que documenta el tope de 3 pero no detecta que el 3 no ocurre.
- **Superposición de eventos:** si el arco ya está congestionado y el evento vigente es *más severo*
  (multiplicador menor o igual), se conserva el vigente (`:97-102`); en caso contrario se sobrescribe
  severidad **y** hora de término.
- Los factores 0.83 / 0.78 / 0.73 provienen, según la tesis (§7.2), de Liu, Long, Deng, Tang & Huang
  (2022), *Revealing spatiotemporal correlation of urban roads via traffic perturbation simulation*,
  *Sustainable Cities and Society* 77:103545. La existencia y el tema del paper están verificados; **los
  valores 0.83/0.78/0.73 no pude verificarlos contra el texto** (paywall ScienceDirect) — ver §8.

### 1.4 De evento a tiempo de viaje

`Model.create_random_velocity` (`model.py:520-543`):

```python
if key_arc not in self.congested_arcs:      → muestra normal memoizada
if tau_episode >= event_end:                → muestra normal memoizada
else:
    length, speed = travel_data[key_minute]          # media de 44 días en ese arco-minuto
    velocity = max(speed * congestion_multiplier, 0.0001)
```

Dos consecuencias no obvias:

1. **Durante un evento la velocidad es determinista.** `speed` es la media histórica del arco en ese
   minuto y φ un escalar; no se suma ruido. La varianza condicional de la velocidad **colapsa a cero**
   justo cuando el evento ocurre. En los datos empíricos ocurre lo contrario: la variabilidad crece con
   la congestión.
2. **Fuera de evento, la velocidad se muestrea `random.gauss(μ_arco,minuto, σ_arco,minuto)** con
   truncamiento a `[0.001, 2]` km/min (legado `generate_normal_velocity` `:585-597`). σ es la desviación
   estándar **entre los 44 días** en ese arco-minuto (`process_all_data` legado `:354-357`). Esa σ ya
   incorpora los días con incidentes, obras y clima: **el ruido "recurrente" ya contiene la congestión no
   recurrente histórica**, y el generador la añade otra vez encima.

### 1.5 Cadencia, expiración y FIFO

- **Sorteo:** `model.py:267-271`, `if (tau + 178) / 60 % hours_max_duration == 0`. Con
  `hours_max_duration = 1` la condición se cumple para `tau ≡ 2 (mod 60)`, es decir τ ∈ {302, 362, 422,
  …, 782}: **una época horaria**, 8 sorteos sobre el horizonte [300, 780]. Con
  `max_congestion_duration = 120` la cadencia pasa a 2 horas y `P_ij` se duplica (`hours = 4`).
- **Acoplamiento paramétrico:** `max_congestion_duration` controla simultáneamente (a) la cota superior
  de la duración, (b) el intervalo entre sorteos y (c) la normalización de la probabilidad. Tres
  cantidades físicas distintas atadas a un parámetro; no se pueden variar independientemente en un
  diseño experimental.
- **Expiración:** `_next_congestion_end` (`model.py:499-517`) adelanta el reloj al instante en que expira
  una congestión que afecta a un vehículo en tránsito, y recalcula velocidades. Es una implementación
  correcta de la propiedad FIFO en el sentido de Ichoua, Gendreau & Potvin (2003) para el caso en que la
  velocidad cambia mientras el vehículo está dentro del arco. **La recuperación es instantánea**: al
  minuto `end` el arco vuelve de golpe a su velocidad normal.

### 1.6 Observabilidad

El planificador no recibe la lista `congested_arcs`. Lo único observable es el vector de las últimas
`n_arcs = 3` velocidades por vehículo (`state.observed_velocity`). De ahí:

- La **política dinámica de referencia** sí lo usa: si `v_última < 0.5·v̄` reencamina al segundo cliente
  más barato (legado `:1876-1900`).
- La **política de Monte Carlo no lo usa**: el promedio se calcula en `mean_velocities` y las dos líneas
  que lo insertarían en el vector de features están comentadas (legado `:2604`, `:2607`). Confirmado
  independientemente en `rl-methodology-for-stdvrp.md` ("not one of the 19 features … can see any of it").

### 1.7 Semillas y varianza

- `client_generator_function` hace `random.seed(random_seed)` (legado `:1542`) y el runner hace
  `np.random.seed(seed)` (legado `:6079`, `:6121`; vivo `episode.py:86-87`, `:182-183`).
- Los **eventos de congestión** consumen exclusivamente `np.random` → dado el seed del episodio, la
  secuencia de sorteos por arco es idéntica para todas las políticas. **Esto sí logra números aleatorios
  comunes (CRN) para el proceso exógeno de congestión.**
- Las **velocidades** consumen `random.gauss` (stream de `random`), y se muestrean **solo cuando un
  vehículo entra al arco**. Como la trayectoria depende de la política, la secuencia de extracciones
  diverge entre políticas: **no hay acoplamiento CRN sobre las velocidades**. Comparar políticas sobre
  "las mismas condiciones" (tesis §7.4.2) es cierto para el proceso de eventos, pero **no** para el campo
  de velocidades.
- Los dos RNG de exploración de la política de entrenamiento estaban **sin sembrar** en el legado; el
  refactor lo documenta y lo repara opcionalmente (`episode.py:26-29`, `config.py:71-76`).

### 1.8 Resumen formal del proceso generativo implementado

Para cada época τ_c ∈ {302, 362, …} y cada arco (i,j) ∈ L, independientemente:

$$
\begin{aligned}
&B_{ij}(\tau_c) \sim \mathrm{Bernoulli}(P_{ij}),\quad P_{ij}=\tfrac{2}{3}\,\widehat{f}_{ij},\ \widehat{f}_{ij}=\text{frecuencia horaria empírica de } v\le 0.4\bar v_{ij}\\
&\text{si } B_{ij}=1:\quad \phi \sim U(\phi_{\min},\phi_{\max}),\quad \delta \sim U(30, \delta_{\max})\\
&V_{kl}(\tau) = \bar v_{kl}(\tau)\cdot \phi / h(d_{kl}),\ \ h\in\{1,\,0.83,\,0.78\},\ \ \tau\in[\tau_c,\tau_c+\delta)\\
&V_{kl}(\tau) \sim \mathcal N(\bar v_{kl}(\tau), \sigma_{kl}(\tau))\ \text{truncada}, \ \ \text{en otro caso}
\end{aligned}
$$

donde `d_kl` es la distancia en saltos *hacia adelante* desde alguno de los dos extremos del arco
incidentado, y `(k,l)` recorre toda la estrella de salida de esos nodos.

---

## 2. Qué dice la literatura — por dimensión

### 2.1 Cuánto de la congestión es no recurrente

La FHWA (*Office of Operations, Reducing Non-Recurring Congestion*) atribuye **~50 % de la congestión**
a interrupciones temporales, desagregadas en **incidentes ≈ 25 %, clima ≈ 15 %, zonas de trabajo ≈ 10 %**;
el resto es cuello de botella recurrente, semaforización deficiente y eventos especiales
([FHWA](https://ops.fhwa.dot.gov/program_areas/reduce-non-cong.htm); informe subyacente:
*Traffic Congestion and Reliability: Trends and Advanced Strategies for Congestion Mitigation*,
FHWA/Cambridge Systematics, 2005,
[PDF](https://ops.fhwa.dot.gov/congestion_report/congestion_report_05.pdf)).

La cifra de **dos tercios** que usa la tesis (Falcocchio & Levinson 2015) es del mismo orden pero
**más alta que la cifra oficial de la FHWA**, y — punto crítico — se refiere a **participación en el
retardo total**, no a **frecuencia de ocurrencia**. El código la usa como si fuera lo segundo
(`travel_time_model.py:334`).

### 2.2 Proceso de ocurrencia

- La literatura de **frecuencia de choques** trabaja con conteos: Poisson, **binomial negativa
  (Poisson-gamma)** para sobredispersión, modelos de parámetros aleatorios, etc. La revisión canónica es
  Lord & Mannering (2010), *The statistical analysis of crash-frequency data*, *TR Part A* 44:291-305
  ([TRID](https://trid.trb.org/view/917816)). La conclusión metodológica relevante: la exposición
  (VMT/flujo) es la variable explicativa dominante y la sobredispersión es la regla, no la excepción.
- Para la **llegada temporal** de incidentes, la formulación aceptada es un **proceso de Poisson no
  homogéneo (NHPP)** con intensidad λ(t) modulada por el flujo. Ejemplo primario reciente y explícito:
  Mouhous, Aissani & Farhi (2025), *A Stochastic Model for Traffic Incidents and Free Flow Recovery in
  Road Networks*, *Mathematics* 13(3):520 ([MDPI](https://www.mdpi.com/2227-7390/13/3/520)) — incidentes
  como NHPP, impacto como proceso de *shot noise*, y recuperación del flujo libre con **decaimiento
  exponencial**.
- **Qué hace el código:** Bernoulli independiente por arco, con `P_ij` **constante en el tiempo**. Es un
  Poisson-binomial homogéneo. No hay dependencia del flujo ni de la hora del día.

### 2.3 Duración

La fuente primaria de síntesis es **Li, Pereira & Ben-Akiva (2018)**, *Overview of traffic incident
duration analysis and prediction*, *European Transport Research Review* 10(2):22, DOI
[10.1186/s12544-018-0300-1](https://doi.org/10.1186/s12544-018-0300-1). Del §2.4 del propio paper
(texto leído directamente del PDF de acceso abierto):

> "Several studies reveal that the traffic duration time meets the log-normal distribution […] or
> log-logistic distribution […]. Weibull distribution (or with gamma heterogeneity or random parameters)
> provides the best likelihood ratio statistics for the used dataset in some other studies […]. Several
> other studies report that the generalized F distribution is the best type […]."

Y del §3:

> "Most of these models are parametric accelerated failure time (AFT) models, which can determine the
> significant variables that affect the traffic incident duration time."

También establece la **descomposición en fases**: detección/reporte → preparación/despacho → viaje de la
respuesta → limpieza/tratamiento (§2.1), y que la mayoría de los estudios miden solo las tres últimas.

Referencias primarias que fundan esa tradición y que la tesis puede citar directamente:

- Golob, Recker & Leonard (1987), *An analysis of the severity and incident duration of truck-involved
  freeway accidents*, *Accident Analysis & Prevention* 19(5):375-395
  ([PubMed](https://pubmed.ncbi.nlm.nih.gov/3675808/)) — 9.000+ accidentes con camión en Los Ángeles.
- Giuliano (1989), *Incident characteristics, frequency, and duration on a high volume urban freeway*,
  *TR Part A* 23(5):387-396
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0191260789900861)) — la
  referencia clásica del ajuste **lognormal** de la duración, con tipo de incidente, hora del día,
  involucramiento de camión y cierre de carriles como factores.
- Nam & Mannering (2000), *An exploratory hazard-based analysis of highway incident duration*,
  *TR Part A* 34(2):85-102
  ([EconPapers](https://econpapers.repec.org/article/eeetransa/v_3a34_3ay_3a2000_3ai_3a2_3ap_3a85-102.htm))
  — modelos AFT por fase (detección, respuesta, limpieza).

**Qué hace el código:** δ ~ U(30, 60). No es ninguna de las formas aceptadas y **no tiene cola derecha**:
el soporte está acotado por construcción, de modo que el evento largo y raro —el que domina el retardo
esperado, §2.6— es imposible por diseño. El piso de 30 minutos tampoco tiene respaldo en el código ni
en la tesis (*no verificado* si es plausible como mínimo empírico para Chengdu).

### 2.4 Magnitud: reducción de capacidad, no de velocidad

La forma canónica de parametrizar el impacto de un incidente es la **fracción de capacidad que queda
disponible** según carriles bloqueados. Tabla verificada en
[FHWA-HOP-13-042, Apéndice C](https://ops.fhwa.dot.gov/publications/fhwahop13042/appc.htm):

**Tabla 40 — Capacidad residual en zona de incidente (HCM 2010, Exhibit 10-17)**

| Carriles (un sentido) | Vehículo detenido en berma | Accidente en berma | 1 carril bloqueado | 2 carriles | 3 carriles |
|---|---|---|---|---|---|
| 2 | 0.95 | 0.81 | **0.35** | 0 | — |
| 3 | 0.99 | 0.83 | **0.49** | 0.17 | 0 |
| 4 | 0.99 | 0.85 | **0.58** | 0.25 | 0.13 |
| 5 | 0.99 | 0.87 | 0.65 | 0.40 | 0.20 |
| 6 | 0.99 | 0.89 | 0.71 | 0.50 | 0.26 |

**Tabla 41 — Factores de ajuste de capacidad (SHRP 2 L08, Exhibit 36-16, borrador de capítulo HCM)**

| Carriles | Sin incidente | Berma cerrada | 1 carril | 2 carriles | 3 carriles |
|---|---|---|---|---|---|
| 2 | 1.00 | 0.81 | 0.70 | — | — |
| 3 | 1.00 | 0.83 | 0.74 | 0.51 | — |
| 4 | 1.00 | 0.85 | 0.77 | 0.50 | 0.52 |

Y la tabla original de Morales (FHWA), *Analytical Procedures for Estimating Freeway Traffic Congestion*,
TRB Circular 344 (leída directamente del
[PDF](https://onlinepubs.trb.org/Onlinepubs/trcircular/344/344-007.pdf), Tabla 1):

| Carriles/sentido | Capacidad S₁ (veh/h) | 1 carril bloqueado S₃ | Berma bloqueada S₃ |
|---|---|---|---|
| 2 | 3.700 | 1.300 (0.35) | 3.000 (0.81) |
| 3 | 5.550 | 2.700 (0.49) | 4.600 (0.83) |
| 4 | 7.400 | 4.300 (0.58) | 6.300 (0.85) |

Nótese además el fenómeno de **capacity drop**: una vez formada la cola, la tasa de descarga del cuello
de botella cae respecto del flujo pre-cola. Los valores empíricos reportados van de ~5 % a ~20 %, con
Cassidy & Bertini (1999) en el rango 8-10 %.

**Qué hace el código:** φ multiplica directamente la **velocidad media**, sin pasar por capacidad ni
por un diagrama fundamental. `φ ∈ [0.1, 0.4]` implica velocidades del 10 % al 40 % de la media — es
decir, un rango de severidad *muy* alto aplicado uniformemente a arcos urbanos heterogéneos.

### 2.5 Extensión espacial y propagación

- Las colas se propagan **aguas arriba** desde el cuello de botella. Es teoría de ondas de choque
  elemental; el diagrama de colas de Morales lo formaliza para el caso de un incidente.
- En redes urbanas, la propagación tiene estructura de **percolación** y de **contagio**:
  - Li, Fu, Wang, Lu, Berezin, Stanley & Havlin (2015), *Percolation transition in dynamical traffic
    network with evolving critical bottlenecks*, *PNAS* 112(3):669-672, DOI
    [10.1073/pnas.1419185112](https://www.pnas.org/doi/10.1073/pnas.1419185112) — el tráfico global se
    descompone en clústeres de flujo local unidos por enlaces cuello de botella que **evolucionan con la
    hora del día** y difieren de los cuellos de botella estructurales.
  - Saberi, Hamedmoghadam, Ashfaq et al. (2020), *A simple contagion process describes spreading of
    traffic jams in urban networks*, *Nature Communications* 11:1616, DOI
    [10.1038/s41467-020-15353-2](https://www.nature.com/articles/s41467-020-15353-2) — dos parámetros
    macroscópicos, tasa de propagación β y tasa de disipación μ, en un sistema tipo SIR, validados
    empíricamente en varias ciudades.

**Qué hace el código:** propaga **hacia adelante** (`successors`), con atenuación determinista por número
de saltos y sin dinámica temporal (todo el vecindario se enciende y se apaga simultáneamente). La
dirección es la contraria a la física de colas; la atenuación por saltos es un sustituto razonable de
una función de decaimiento espacial, pero no está calibrada contra β/μ ni contra un modelo de percolación.

### 2.6 Correlación con la congestión recurrente y con la demanda

Tres hechos establecidos:

1. La **frecuencia** de incidentes escala con la exposición (VMT/flujo) — Lord & Mannering (2010).
2. El **impacto** de un incidente crece de forma no lineal con la relación demanda/capacidad: en el
   diagrama de Morales, el retardo total contiene el factor `(S₂−S₃)(S₁−S₂)/[2(S₁−S₃)]`, que se dispara
   cuando la demanda `S₂` se acerca a la capacidad `S₁`.
3. Por eso SHRP 2 L03 (*Analytical Procedures for Determining the Impacts of Reliability Mitigation
   Strategies*, TRB, 2012, DOI [10.17226/22806](https://doi.org/10.17226/22806)) construye modelos de
   fiabilidad en los que las métricas (buffer index, planning time index, percentiles de TTI) son función
   conjunta de demanda, capacidad, incidentes y clima.

**Qué hace el código:** `P_ij` no depende del tiempo ni del estado de congestión recurrente, y φ es
independiente de la hora. La única correlación presente es la espacial estática (arcos históricamente
congestionados tienen `P_ij` alta), que se acumula además con el problema del umbral (§3).

### 2.7 Clima

FHWA Road Weather Management ([Rain & Flooding](https://ops.fhwa.dot.gov/weather/weather_events/rain_flooding.htm),
[Snow & Ice](https://ops.fhwa.dot.gov/weather/weather_events/snow_ice.htm)):

- Lluvia ligera: velocidad en autopista −2 % a −13 %; lluvia fuerte: −3 % a −17 %.
- Arterias: −10 % a −25 % con lluvia.
- Nieve/aguanieve: velocidades medias en arteria −30 % a −40 %.
- "Snow, ice and fog alone cause 15 percent of non-recurring delay".

Lo esencial para el simulador: el clima es un evento **correlacionado a nivel de toda la red**, de
duración larga (horas) y magnitud moderada — exactamente el complemento del incidente (local, corto,
severo).

**Qué hace el código:** no existe.

### 2.8 Observabilidad y detección

La fase de **detección/reporte** es explícitamente parte de la duración del incidente en la taxonomía de
Li, Pereira & Ben-Akiva (2018, §2.1), y la mayoría de los datasets **empiezan a contar cuando el
incidente se reporta**, no cuando ocurre. Es decir: en la realidad hay un retardo entre ocurrencia y
disponibilidad de la información para un planificador.

**Qué hace el código:** el evento es totalmente invisible salvo por el efecto en la velocidad del
vehículo que ya entró al arco. Es el extremo pesimista (información cero, ni siquiera con retardo) y es
una elección **defendible como escenario**, pero no está justificada en la tesis como tal ni comparada
con el caso con información.

### 2.9 Recuperación

Mouhous, Aissani & Farhi (2025) modelan la recuperación del flujo libre con **decaimiento exponencial**
del tiempo de recuperación entre incidentes, con incrementos gamma/exponenciales. Saberi et al. (2020)
la modelan con una tasa de disipación μ tipo SIR. Ambas implican que **el retorno a la normalidad es
gradual y toma más tiempo que la limpieza del incidente** — la fase "recuperación" que Li et al. señalan
que casi nadie mide.

**Qué hace el código:** recuperación **instantánea** al minuto `τ_c + δ` (`model.py:536-537`).

### 2.10 Chengdu y los datos usados

- El dataset es Guo, Zhang, Dong & Guo (2019), *Urban link travel speed dataset from a megacity road
  network*, *Scientific Data* 6:61, DOI
  [10.1038/s41597-019-0060-3](https://doi.org/10.1038/s41597-019-0060-3). Verificado directamente:
  **1.902 nodos, 5.943 links dirigidos**, 45 días (1 jun – 15 jul 2015), periodos de 2 minutos, y
  **cinco ventanas horarias representativas: 3-5, 8-10, 12-14, 17-19, 21-23**, construido a partir de
  3.01 mil millones de muestras GPS de >12.000 taxis.
- **Dos lecturas de la tesis que hay que corregir:**
  1. La tesis dice "información detallada de 44 días sobre la velocidad promedio en cada arco, registrada
     en intervalos de dos minutos **desde las 3:00 a.m. hasta las 11:00 p.m.**" (§7.1). La cobertura
     **no es continua**: son cinco bloques de dos horas. Esto es precisamente lo que obliga a la
     interpolación por tramos de `get_interpolated_speed` (legado `:420-457`, intervalos 420-540,
     660-840, 960-1080, que son exactamente los huecos entre ventanas). Las velocidades usadas en gran
     parte del horizonte del episodio son **interpoladas linealmente**, no observadas.
  2. La tesis dice "las velocidades promedio entre calles adyacentes presentan una **correlación del
     49,75 %**" (§7.2). El paper dice que **"49.75 % de los links tienen correlaciones significativas"**
     con sus vecinos, que es otra cosa. La afirmación en la tesis debe reescribirse.
  3. La cifra "97,81 % de las calles cumplen la distribución normal" **sí es correcta**: 97,81 % de las
     variables de velocidad pasan un test Kolmogorov-Smirnov de normalidad al 0,05 (y 1,53 % lognormal).
- Sobre distribuciones de velocidad: Maghrour Zefreh & Török (2020), *Distribution of traffic speed in
  different traffic conditions: an empirical study in Budapest*, *Transport* 35(1):68-86, DOI
  [10.3846/transport.2019.11725](https://doi.org/10.3846/transport.2019.11725), reportan **normal en
  flujo libre**, gamma en congestión, exponencial en congestión sobresaturada, lognormal en flujo
  subsaturado. La tesis (§1) cita este trabajo diciendo que "bajo condiciones de flujo libre, las
  velocidades siguen distribuciones **log-normales**" — según el resumen del propio editor, la lognormal
  corresponde al régimen **subsaturado**, no al de flujo libre. Corregir.
- Existe además el **DiDi Chuxing GAIA Open Dataset** con trayectorias y Travel Time Index de Chengdu
  (nov-2016 y 2018), que sería la fuente natural para calibrar frecuencias e impactos de eventos; no se
  usa en este trabajo (*no verificado* si el acceso sigue abierto).

---

## 3. Veredicto dimensión por dimensión

| Dimensión | Qué hace el código | Qué dice la literatura | Veredicto | Fuente |
|---|---|---|---|---|
| **Cuota de congestión no recurrente** | Factor `2/3` aplicado a la frecuencia de episodios (`travel_time_model.py:334`) | ~50 % del **retardo** (incidentes 25 %, clima 15 %, obras 10 %); dos tercios es cota alta y se refiere a retardo, no a frecuencia | **No soportado** (error de categoría: cuota de retardo usada como cuota de frecuencia) | [FHWA](https://ops.fhwa.dot.gov/program_areas/reduce-non-cong.htm) |
| **Umbral de identificación del evento** | `0.1·v̄ ≤ v ≤ 0.4·v̄` con `v̄` = media de **todo el día** (`:329`) | Un evento no recurrente es una desviación respecto del **perfil esperado a esa hora**, no respecto de la media diaria | **No soportado / irrealista**: etiqueta congestión recurrente de punta como no recurrente | Li, Pereira & Ben-Akiva (2018) §1; SHRP 2 L03 |
| **Proceso de ocurrencia** | Bernoulli i.i.d. por arco y época, `P_ij` constante en el tiempo (`generator.py:60-62`) | NHPP con λ(t) modulada por flujo; conteos sobre-dispersos (Poisson-gamma) | **Simplificado pero defendible** en estructura; **no soportado** en la homogeneidad temporal | Lord & Mannering (2010); Mouhous et al. (2025) |
| **Heterogeneidad espacial de la tasa** | `P_ij` calibrada arco a arco con 44 días de datos (`:294-335`) | La exposición varía por arco; calibrar por arco es correcto | **Consistente** (con la salvedad del umbral) | Lord & Mannering (2010) |
| **Duración** | `δ ~ U(30, δ_max)` (`generator.py:69`) | Lognormal / log-logística / Weibull / F generalizada, vía AFT; con cola pesada y fases | **No soportado** | Li, Pereira & Ben-Akiva (2018) §2.4, §3; Giuliano (1989); Nam & Mannering (2000) |
| **Magnitud** | `φ ~ U(0.1, 0.4)` multiplicando la velocidad media (`:66-68`) | Fracción de **capacidad** residual por carriles bloqueados (0.35/0.49/0.58…); capacity drop 5-20 % | **No soportado** en la forma; **simplificado pero defendible** como *reducción de velocidad* si se calibrara | HCM 2010 Exh. 10-17; SHRP 2 L08 Exh. 36-16; Morales, TRB Circ. 344 |
| **Varianza durante el evento** | Colapsa a 0 (velocidad determinista, `model.py:539-541`) | La variabilidad de velocidad **crece** en régimen congestionado | **No soportado** | Maghrour Zefreh & Török (2020) |
| **Extensión espacial** | Toda la estrella de salida de ambos extremos, ≤2 saltos hacia adelante (`:76-107`) | Un incidente bloquea carriles de **un** enlace; el efecto se extiende por cola | **No soportado** en el footprint (bloquea arcos que el incidente no toca) | Morales, TRB Circ. 344 |
| **Dirección de la propagación** | Aguas **abajo** (`successors`) | Las colas se propagan aguas **arriba** | **No soportado** | Morales (diagrama de colas); Saberi et al. (2020) |
| **Atenuación con la distancia** | Determinista, 1 / 0.83 / 0.78 (0.73 muerto) | Existe decaimiento espacial; parámetros dependen de topología y de β/μ | **Simplificado pero defendible**; la rama 0.73 es **código muerto** (`generator.py:91-92`) | Liu et al. (2022) *(valores no verificados)*; Saberi et al. (2020) |
| **Dinámica de propagación** | Instantánea, simultánea en todo el vecindario | Contagio con tasa β; percolación con umbral | **Simplificado** | Saberi et al. (2020); Li et al. (2015) |
| **Correlación con congestión recurrente / demanda** | Ninguna (φ y `P_ij` independientes de la hora) | Frecuencia ∝ exposición; impacto superlineal cerca de capacidad | **No soportado** | Lord & Mannering (2010); Morales |
| **Doble conteo con el ruido recurrente** | σ = desviación **entre 44 días**, que ya contiene los eventos, y encima se suman eventos (legado `:354-357`) | Los perfiles empíricos día-a-día ya incorporan la variabilidad no recurrente | **No soportado** (sobreestima la incertidumbre total) | Guo et al. (2019); SHRP 2 L03 |
| **Clima** | No modelado | 15 % del retardo no recurrente; evento correlacionado en toda la red | **Ausente** | [FHWA RWM](https://ops.fhwa.dot.gov/weather/weather_events/snow_ice.htm) |
| **Zonas de trabajo / eventos especiales** | No modelados | 10 % + ~5 % del total | **Ausente** | [FHWA](https://ops.fhwa.dot.gov/program_areas/reduce-non-cong.htm) |
| **Observabilidad** | Cero información sobre el evento; solo velocidad experimentada | Existe una fase de detección/reporte medible; los TMC publican incidentes con retardo | **Simplificado pero defendible** como escenario extremo; no justificado como tal | Li, Pereira & Ben-Akiva (2018) §2.1 |
| **Recuperación** | Instantánea al expirar δ (`model.py:536-537`) | Decaimiento exponencial / disipación con tasa μ | **No soportado** | Mouhous et al. (2025); Saberi et al. (2020) |
| **Efecto de la duración sobre el costo** | Lineal (arco degradado durante δ) | Retardo total ~ forma **cuadrática** en las duraciones | **No soportado** (subestima la cola de costos) | Morales, TRB Circ. 344, ec. general |
| **FIFO / no adelantamiento** | Respetada: se recalcula al expirar el evento (`model.py:499-517`) | Propiedad exigida desde Ichoua et al. (2003) | **Consistente** | Ichoua, Gendreau & Potvin (2003) |
| **CRN entre políticas** | Sí para eventos (`np.random` sembrado); **no** para velocidades (`random.gauss` dependiente de trayectoria) | La comparación pareada exige acoplar todo el proceso exógeno | **Parcialmente soportado** | práctica estándar de simulación |

---

## 4. Mejoras propuestas priorizadas

### Tabla de prioridades

| # | Mejora | Efecto esperado sobre la dificultad del MDP | Costo | Riesgo de romper la comparación con la tesis |
|---|---|---|---|---|
| **M1** | Umbral de detección relativo al **perfil por hora**, no a la media diaria | Reduce drásticamente la tasa de eventos espurios; el evento pasa a ser realmente "no recurrente" | Bajo (una línea + recalibración) | Alto: cambia `P_ij`, hay que re-entrenar |
| **M2** | Eliminar el doble conteo: separar σ *dentro del día* de σ *entre días* | Baja la varianza total; hace que el efecto del evento sea identificable | Medio | Alto |
| **M3** | Duración lognormal (o AFT) con covariables, en vez de U(30, δmax) | Introduce cola pesada → el valor de anticipar crece; problema **más** difícil y más informativo | Bajo | Medio |
| **M4** | Magnitud desde tablas de capacidad + diagrama fundamental, no φ arbitrario | Severidad heterogénea y físicamente interpretable | Medio | Medio |
| **M5** | Propagación **aguas arriba** con formación de cola y spillback | Correlación espacial realista; recompensa la evitación anticipada de zonas | Alto | Alto |
| **M6** | NHPP con λ(t) proporcional al flujo / inverso a la velocidad esperada | Eventos concentrados en punta → interacción con las ventanas de tiempo | Bajo-medio | Medio |
| **M7** | Correlación evento-recurrencia: severidad creciente con demanda/capacidad | Hace que el mismo evento sea peor a las 8:30 que a las 14:00 | Bajo | Medio |
| **M8** | Clima como evento correlacionado a nivel red | Añade un modo de fallo de escala global; obliga a políticas robustas | Bajo | Bajo (escenario adicional) |
| **M9** | Observabilidad parcial con retardo de detección | Convierte el problema en POMDP explícito; permite estudiar el valor de la información | Medio | Bajo (escenario adicional) |
| **M10** | Recuperación gradual (decaimiento exponencial) | Cola de impacto realista; elimina el salto discontinuo | Bajo | Bajo |
| **M11** | CRN completo + diseño de escenarios (SAA) para evaluación | Reduce varianza del estimador de la diferencia entre políticas | Medio | Ninguno (mejora el rigor) |
| **M12** | Validación contra datos: distribución de tiempos de viaje, BI y PTI | Convierte "realismo" de afirmación en medición | Medio-alto | Ninguno |
| **M0** | *(pre-requisito, ver `rl-methodology-for-stdvrp.md`)* hacer observable la congestión en los features | Sin esto, ninguna mejora del generador cambia la política aprendida | Bajo | Alto |

> **M0 es la que más importa.** Hoy la política de Monte Carlo no puede ver la congestión (legado
> `:2604`, `:2607`). Enriquecer el generador sin arreglar los features solo hace que el entorno sea más
> ruidoso para un agente que no puede reaccionar. Esa mejora está detallada en la otra nota; aquí solo se
> declara como dependencia.

---

### M1 — Umbral relativo al perfil por hora del día

**Qué cambiar.** En `_compute_event_probabilities` (`travel_time_model.py:329`), sustituir

```python
if row["Speed"] <= 0.4 * row["avg_speed"] and row["Speed"] >= 0.1 * row["avg_speed"]:
```

por una comparación contra la **media del mismo arco en el mismo intervalo de 30 minutos a través de los
44 días** (que ya se calcula en el pipeline como `aggregated_data`, legado `:354-357`), y expresar el
umbral en desviaciones estándar o en percentiles:

```python
z = (row["Speed"] - mu[link, slot]) / sigma[link, slot]
is_event = z <= -2.0          # o: row["Speed"] <= q05[link, slot]
```

**Por qué.** Un evento no recurrente es por definición una desviación respecto de lo *esperado a esa
hora*. Usar la media diaria hace que el criterio se dispare sistemáticamente en la punta de la mañana y
de la tarde, que es congestión **recurrente**. Ese es el mismo principio con el que se construyen los
modelos de fiabilidad de SHRP 2 L03: la parte recurrente se explica con demanda/capacidad y lo que queda
es lo no recurrente (DOI [10.17226/22806](https://doi.org/10.17226/22806)). Es también la definición
operativa que usan Li, Pereira & Ben-Akiva (2018) al separar congestión recurrente de no recurrente.

**Parámetros sugeridos.** Umbral `z ≤ −2` (≈ percentil 2,3 bajo normalidad, coherente con el 97,81 % de
normalidad reportado por Guo et al. 2019), o percentil empírico 5 %. Reportar `Λ = Σ P_ij` resultante.

**Efecto en la dificultad.** Baja la tasa de eventos (probablemente mucho), pero los que quedan son
genuinamente informativos. El agente deja de "aprender el ruido".

**Costo.** Bajo (una función). **Requiere recalibrar y re-entrenar todos los escenarios.**

---

### M2 — Eliminar el doble conteo entre σ y eventos

**Qué cambiar.** Hoy `σ(arco, minuto)` es la desviación estándar **entre días** (legado `:354-357`), y
sobre esa dispersión se superponen los eventos (`generator.py`). Descomponer:

- Ajustar el perfil recurrente `μ(arco, slot)` con los **días limpios** (los que no contienen episodios
  marcados como evento por M1).
- Estimar `σ_recurrente` sobre esos mismos días limpios.
- Dejar que **toda** la variabilidad extrema la produzca el generador de eventos.

**Por qué.** Con la implementación actual el modelo suma dos veces la misma incertidumbre: la
variabilidad histórica ya *es* el resultado de incidentes, clima y obras pasados. El resultado es un
entorno más ruidoso de lo que Chengdu realmente es, lo que infla artificialmente la ventaja de una
política reactiva sobre una estática (que es el resultado principal de la tesis).

**Efecto.** Reduce la brecha reportada entre MC y estática. Es un resultado *menos favorable* pero
**mucho más defendible ante el comité**.

**Costo.** Medio (un paso extra de limpieza en el pipeline de calibración).

---

### M3 — Duración lognormal (o AFT) con covariables

**Qué cambiar.** En `generator.py:69`:

```python
# antes
state_time_elimination = np.random.uniform(30, self.max_congestion_duration)
# después
state_time_elimination = float(np.random.lognormal(mean=mu_ln, sigma=sigma_ln))
state_time_elimination = min(state_time_elimination, hard_cap)   # solo por estabilidad numérica
```

**Por qué.** Li, Pereira & Ben-Akiva (2018, §2.4) documentan lognormal / log-logística / Weibull /
F generalizada como las formas aceptadas; Giuliano (1989) es la referencia clásica del ajuste lognormal;
Nam & Mannering (2000) el estándar AFT por fases.

**Parametrización concreta y honesta.** Yo **no** puedo darle a usted valores de μ_ln y σ_ln extraídos de
una fuente primaria verificada (ver §8: no pude leer el texto completo de Giuliano 1989 ni de Nam &
Mannering 2000). Lo defendible es **calibrar por momentos** contra los propios datos de Chengdu:

- Detecte episodios con M1, mida su duración empírica `d_1..d_n` (minutos).
- Ajuste `μ_ln = mean(log d)`, `σ_ln = sd(log d)` por máxima verosimilitud.
- Reporte mediana `exp(μ_ln)`, media `exp(μ_ln + σ_ln²/2)` y percentil 95.
- Como *sanity check* de orden de magnitud, la literatura de incidentes en autopista sitúa la mediana en
  decenas de minutos con colas de varias horas; si su ajuste da una mediana de 45 min y un P95 de ~3 h,
  está en territorio plausible. **Trate cualquier número que no venga de su ajuste como no verificado.**

**Alternativa mejor si tiene covariables.** Un AFT lognormal
`log δ = β₀ + β₁·(hora punta) + β₂·(clase de vía) + β₃·(severidad) + ε`, que es exactamente la forma que
Nam & Mannering (2000) proponen. Le da además la correlación de M7 gratis.

**Efecto en la dificultad.** Aumenta la varianza del retorno y crea eventos raros de alto costo. Combinado
con M0 (observabilidad), es el cambio que más valor le da a una política anticipativa frente a una
estática — y por tanto el que más fortalece la hipótesis de la tesis.

**Costo.** Bajo en código; medio en calibración.

---

### M4 — Magnitud anclada a reducción de capacidad

**Qué cambiar.** Reemplazar `φ ~ U(0.1, 0.4)` por un pipeline en dos pasos:

1. **Muestrear el tipo de bloqueo**, no la severidad: `{berma, 1 carril, 2 carriles}` con probabilidades
   que reflejen la mezcla de incidentes.
2. **Traducir a capacidad residual** con la tabla del HCM/SHRP 2 según el número de carriles del arco
   (que en Chengdu no está en `link.csv`; habría que imputarlo por clase de vía o por longitud/velocidad
   libre — ver §8).
3. **Traducir capacidad a velocidad** con un diagrama fundamental (Greenshields o triangular) o, más
   simple y suficiente para el simulador, con un mapeo monótono `v/v̄ = g(c/c̄)` calibrado sobre los
   propios datos de Chengdu (velocidad observada vs. cuantil de velocidad).

Valores de referencia verificados (HCM 2010 Exhibit 10-17): 1 carril bloqueado en vía de 2 carriles →
**0.35** de la capacidad; 3 carriles → 0.49; 4 carriles → 0.58. Berma: 0.81-0.85. Y añadir un **capacity
drop** adicional del 8-10 % una vez formada la cola (Cassidy & Bertini 1999).

**Por qué.** Es la parametrización canónica y auditable. Además elimina la arbitrariedad de que "el
escenario más congestionado" sea `φ_max = 0.1` (velocidad al 10 % de la media, ~2-3 km/h sostenidos
durante 30-60 minutos en toda una vecindad de dos saltos), que es una situación de bloqueo total, no un
incidente típico.

**Efecto.** Severidades heterogéneas y correlacionadas con la jerarquía vial. La dificultad se concentra
en pocos eventos graves en vez de repartirse en muchos moderados.

**Costo.** Medio: hay que imputar carriles/clase de vía.

---

### M5 — Propagación aguas arriba con cola y spillback

**Qué cambiar.** Invertir la dirección de `_reachable_nodes` (usar **predecesores** en vez de
`successors`, `generator.py:79`, `:132`) y hacer que la extensión de la cola sea **función de la
duración transcurrida y del exceso de demanda**, no un radio fijo:

```
L_cola(t) = (λ − μ_incidente) · (t − t_0) / k_jam      [longitud de cola en km]
```

congestionando los arcos aguas arriba hasta cubrir `L_cola(t)`, y aplicando **spillback** cuando la cola
excede la longitud del arco (entonces se propaga al arco anterior).

**Por qué.** Es teoría de flujo elemental y está en el diagrama de Morales; empíricamente, Saberi et al.
(2020) muestran que la propagación de atascos en red urbana se describe con una tasa β de contagio y una
μ de disipación, y Li et al. (2015) que los cuellos de botella críticos evolucionan con la hora.

**Parametrización mínima viable** (si no quiere implementar teoría de colas): sustituir el radio fijo de
2 saltos por un **radio creciente en el tiempo** `r(t) = min(r_max, ⌊β·(t−t_0)⌋)` sobre **predecesores**,
con β calibrado para que la velocidad de propagación esté en el orden de 1-2 km/h contra el flujo (rango
clásico de las ondas de choque de arranque/parada; *magnitud no verificada contra fuente primaria en esta
revisión*).

**Efecto.** El agente puede aprender que hay que abandonar una zona **antes** de que la cola lo alcance;
esto es exactamente el tipo de anticipación que justifica usar RL.

**Costo.** Alto. Es el cambio más caro y el que más cambia los resultados.

---

### M6 — NHPP con intensidad dependiente del tiempo

**Qué cambiar.** Sustituir `P_ij` constante por `P_ij(τ)` proporcional a un proxy de exposición:

```
λ_ij(τ) = λ̄_ij · [ v̄_ij / v_ij(τ) ]^γ        (más incidentes cuando la vía está más cargada)
```

normalizado para que `∫λ_ij(τ)dτ` sobre el día iguale la frecuencia empírica de M1. `γ ∈ [0.5, 1.5]`
como barrido de sensibilidad.

**Por qué.** Mouhous, Aissani & Farhi (2025) usan explícitamente un NHPP para incidentes; Lord &
Mannering (2010) establecen que la exposición es la variable dominante de la frecuencia.

**Efecto.** Los eventos se concentran en las horas en que las ventanas de tiempo están más apretadas →
la política tiene que aprender un compromiso temporal, no solo espacial.

**Costo.** Bajo-medio. **Importante:** desacople primero `max_congestion_duration` de la cadencia de
sorteo y de la normalización (`travel_time_model.py:319-320`, `model.py:80-81`), que hoy están atados.

---

### M7 — Correlación recurrente ↔ no recurrente

**Qué cambiar.** Hacer que la **severidad** dependa del estado recurrente:

```
φ_efectivo(τ) = φ_base · (v_recurrente(τ) / v̄)^θ ,   θ > 0
```

de modo que un incidente en hora punta reduzca la velocidad proporcionalmente más (en tiempo de viaje)
que el mismo incidente a mediodía.

**Por qué.** En el diagrama de colas de Morales el retardo contiene `(S₂−S₃)(S₁−S₂)/[2(S₁−S₃)]`: cuando
la demanda `S₂` se acerca a la capacidad `S₁`, el mismo bloqueo produce mucho más retardo. Es también la
razón por la que Adler et al. (citado en Li et al. 2018, §1) estiman una ganancia de ~57 € por minuto de
reducción de duración en general, pero **~1.200 € por minuto en ubicaciones altamente congestionadas**.

**Efecto.** El costo esperado deja de ser separable entre "hora del día" y "evento"; el agente debe
aprender la interacción. Sube la dificultad de forma sustantiva y realista.

**Costo.** Bajo (una multiplicación), una vez que M6 está en su sitio.

---

### M8 — Clima como evento correlacionado a nivel de red

**Qué cambiar.** Añadir un `WeatherCongestionGenerator` que implemente la misma interfaz
`CongestionGenerator` (`generator.py:28-33` ya define el seam) y que, con probabilidad `p_w` por
episodio, active un multiplicador **global** de velocidad durante una duración larga.

**Parámetros verificados (FHWA Road Weather Management):**

| Condición | Multiplicador de velocidad en arteria | Duración típica |
|---|---|---|
| Lluvia ligera | 0.90 – 0.75 (−10 % a −25 %) | 1-3 h |
| Lluvia fuerte | 0.83 – 0.75 | 1-3 h |
| Nieve / aguanieve | 0.70 – 0.60 (−30 % a −40 %) | 2-6 h |

(Para Chengdu, la nieve es marginal; la lluvia no.) `p_w` debería salir de datos climáticos de Chengdu
jun-jul 2015 (*no verificado*).

**Por qué.** El clima es el 15 % del retardo no recurrente según la FHWA y es el único modo de fallo
**correlacionado en toda la red**: ninguna política de reencaminamiento local lo puede evitar, solo
puede replanificar la secuencia y aceptar horas extra. Es un escenario que discrimina fuertemente entre
políticas estáticas y adaptativas.

**Costo.** Bajo — el seam ya existe.

---

### M9 — Observabilidad parcial y retardo de detección

**Qué cambiar.** Introducir un canal de información con retardo: el evento ocurre en `τ_c`, pero se
publica al planificador en `τ_c + D`, con `D` una variable aleatoria (la **fase de detección/reporte** de
Li, Pereira & Ben-Akiva 2018, §2.1). Tres escenarios a comparar:

| Escenario | `D` | Interpretación |
|---|---|---|
| **Oráculo** | 0 | cota superior del valor de la información |
| **Realista** | `D ~ Lognormal`, calibrada | feed de un TMC / proveedor de tráfico |
| **Ciego** | ∞ | el escenario actual de la tesis |

**Por qué.** (a) Es lo que ocurre en la práctica; (b) permite **medir el valor de la información**, que
es un resultado publicable por sí mismo y mucho más interesante que "MC gana a la estática"; (c) hace
explícito que el problema es un POMDP, lo que fortalece la justificación metodológica del uso de RL.

**Costo.** Medio. Requiere M0 (que los features puedan ver algo).

---

### M10 — Recuperación gradual

**Qué cambiar.** En `model.py:536-537`, en vez del salto binario `tau >= event_end`, aplicar

```
mult(τ) = 1 − (1 − φ)·exp(−(τ − τ_fin)/T_rec)   para τ ≥ τ_fin
```

con `T_rec` del orden de la longitud de la cola dividida por la tasa de descarga (o simplemente
`T_rec ≈ 0.5·δ` como primera aproximación).

**Por qué.** Mouhous et al. (2025) modelan exactamente esto con decaimiento exponencial; Saberi et al.
(2020) con una tasa de disipación μ; Li, Pereira & Ben-Akiva (2018, §2.1) señalan que la fase de
recuperación existe y casi nunca se mide.

**Costo.** Bajo.

---

### M11 — Reproducibilidad, CRN y generación de escenarios

Tres cambios, todos de rigor experimental:

1. **CRN completo.** Hoy los eventos están acoplados entre políticas (`np.random` sembrado) pero las
   velocidades no, porque `random.gauss` se consume **en el orden en que los vehículos entran a los
   arcos** (`generate_normal_velocity`, legado `:585`). Solución: **pre-muestrear el campo de velocidades
   completo al inicio del episodio** — un `dict[(arco, minuto)] → v` generado desde un RNG dedicado
   sembrado con el seed del episodio — y que el simulador solo lo consulte. Así, dos políticas distintas
   sobre el mismo seed ven **exactamente el mismo mundo**. Esto es acoplamiento perfecto y reduce la
   varianza del estimador de la *diferencia* de costos, que es el estimador que la tesis reporta.
2. **RNG independientes por fuente de aleatoriedad** (demanda, velocidades, eventos, exploración) en vez
   de dos streams globales compartidos. `numpy.random.Generator` con `SeedSequence.spawn()` lo da
   directamente. Hoy el stream `random` se comparte entre generación de clientes y velocidades, lo que
   hace que el campo de velocidades dependa del número de clientes sorteados.
3. **Diseño de escenarios tipo SAA para la evaluación.** En vez de 50 semillas por escenario de
   congestión, generar un **árbol de escenarios común** con estratificación por número de eventos
   (0, 1-2, 3-5, >5) y reportar el costo condicional a cada estrato. Esto responde la pregunta que
   realmente importa —*¿cuánto de la ventaja de MC viene de los episodios con eventos?*— que hoy queda
   oculta en la media. La nota `rl-methodology-for-stdvrp.md` §5.2 discute SAA como método de decisión;
   aquí se propone como método de **evaluación**.

---

### M12 — Validación contra datos reales

Hoy la tesis afirma realismo pero no lo mide. Propuesta concreta de protocolo de validación:

| Métrica | Cómo se calcula en el simulador | Contra qué se compara |
|---|---|---|
| Distribución de velocidad por arco y slot | Histograma de `sampled_arc_velocities` | Distribución empírica de los 44 días (test KS) |
| **Buffer Index** = (TT₉₅ − TT̄)/TT̄ | Sobre un conjunto fijo de pares OD y horas de salida | Mismo cálculo sobre los datos de Chengdu |
| **Planning Time Index** = TT₉₅/TT_libre | Ídem | Ídem |
| Fracción de arco-horas con `v ≤ 0.4·v̄(slot)` | Conteo en simulación | Conteo empírico (es el estadístico que M1 calibra) |
| Nº de eventos por episodio, y fracción de red afectada | Instrumentar `generate` | Sin contraparte directa; **reportarlo** es ya un avance |

Las definiciones de BI y PTI son las de la FHWA
([*Travel Time Reliability: Making It There On Time, All The Time*](https://ops.fhwa.dot.gov/publications/tt_reliability/ttr_report.htm)):
`BI = (TT₉₅ − TT_medio)/TT_medio`, `PTI = TT₉₅ / TT_flujo_libre`. Son exactamente las métricas con las
que SHRP 2 L03 evalúa modelos de fiabilidad, así que reportarlas conecta el simulador con el estándar de
la ciencia del transporte y no solo con el de investigación operativa.

**Este es el entregable que convierte "usamos datos reales de Chengdu" en una afirmación verificable.**

---

## 5. Estado del arte: STDVRP + RL + congestión no recurrente

### 5.1 Lo que está resuelto

**(a) TDVRP determinista con propiedad FIFO.** Cerrado desde Ichoua, Gendreau & Potvin (2003),
*Vehicle dispatching with time-dependent travel times*, *EJOR* 144:379-396, DOI
[10.1016/S0377-2217(02)00147-9](https://doi.org/10.1016/S0377-2217(02)00147-9), que introduce el modelo
de velocidades por tramos que satisface FIFO y sobre el que se apoya casi toda la literatura posterior.
Hay incluso benchmarks modernos con datos reales (Blauth, Held, Müller, Schlomberg, Traub, Tröbst &
Vygen, [arXiv:2205.00889](https://arxiv.org/abs/2205.00889)) — explícitamente **deterministas**.

**(b) Rutas/caminos óptimos en redes estocásticas y dependientes del tiempo.** Miller-Hooks &
Mahmassani (2000), *Least expected time paths in stochastic, time-varying transportation networks*,
*Transportation Science* 34(2):198-215; Gao & Chabini (2006), *Optimal routing policy problems in
stochastic time-dependent networks*, *TR Part B* 40(2):93-122, DOI
[10.1016/j.trb.2005.02.001](https://www.sciencedirect.com/science/article/abs/pii/S0191261505000391)
([PDF de autor](https://people.umass.edu/sgao/orp.pdf)). Estos trabajos ya establecen que la solución
óptima bajo información en línea es una **política**, no un camino — el argumento conceptual que la
tesis usa para justificar RL ya está hecho desde 2006, en el nivel de camino.

**(c) STDVRP estático con metaheurísticas.** Toda la línea que la tesis revisa en §2.1.4 (Lecluyse, Van
Woensel & Peremans 2009; Taş, Dellaert, Van Woensel & de Kok 2014; Huang, Zhao, Van Woensel & Gross
2017; Jie, Liu & Sun 2022).

**(d) RL/ADP para VRP dinámico y estocástico.** Cubierto en detalle en
[`rl-methodology-for-stdvrp.md`](./rl-methodology-for-stdvrp.md) §2 (línea de Ulmer, Thomas, Soeffker,
Mattfeld). Las revisiones de referencia son:
- Hildebrandt, Thomas & Ulmer (2023), *Opportunities for reinforcement learning in stochastic dynamic
  vehicle routing*, **Computers & Operations Research** 150:106071 — *no* *Networks*, como sugería el
  encargo; verificar la cita en la tesis.
- Bogyrbayeva, Meraliyev, Mustakhov & Dauletbayev (2024), *Machine Learning to Solve Vehicle Routing
  Problems: A Survey*, **IEEE T-ITS** 25(6), DOI
  [10.1109/TITS.2023.3334976](https://doi.org/10.1109/TITS.2023.3334976).
- Bai et al. (2023), *Analytics and machine learning in vehicle routing research*, **International
  Journal of Production Research** 61(1):4-30, DOI
  [10.1080/00207543.2021.2013566](https://doi.org/10.1080/00207543.2021.2013566) — **no** EJOR;
  corregir si la tesis lo cita así.
- Zhou, Lischka, Kulcsár, Wu, Haghir Chehreghani & Laporte (2025), *Learning for routing: A guided
  review of recent developments and future directions*, **TR Part E** 202:104278,
  [arXiv:2507.00218](https://arxiv.org/abs/2507.00218) — 253 artículos revisados, 2016-2025.

**(e) RL con tiempos de viaje simultáneamente dependientes del tiempo y estocásticos.** **Esto ya está
hecho**, y es el hallazgo que más obliga a reformular la tesis:

- **Chen, Imdahl, Lai & Van Woensel (2025)**, *The Dynamic Traveling Salesman Problem with
  Time-Dependent and Stochastic travel times: A deep reinforcement learning approach*, **TR Part C**
  172:105022, DOI [10.1016/j.trc.2025.105022](https://doi.org/10.1016/j.trc.2025.105022)
  ([preprint SSRN 4809480](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4809480)). Modelo
  *Dynamic Graph Temporal Attention* con atención multi-cabeza para extraer información de los tiempos de
  viaje estocásticos; supera heurísticas de horizonte rodante y otros modelos de RL, hasta **32 %** de
  mejora en duración media de ruta en escenarios grandes.
- **Chen, Imdahl, Lai & Van Woensel (2026)**, *A reinforcement learning approach for the dynamic vehicle
  routing and scheduling problem with stochastic request times and time-dependent, stochastic travel
  times*, **TR Part C** 182:105387, DOI
  [10.1016/j.trc.2025.105387](https://doi.org/10.1016/j.trc.2025.105387). Modelo *dynamic
  spatial-temporal multi-pointer* con RL, salidas flexibles y rutas dinámicas, minimizando costo
  temporal, tardanza y clientes no servidos, con generalización entre escenarios sin reentrenamiento.

### 5.2 Los trabajos más cercanos, y en qué se quedan cortos

| Trabajo | TD | Estocástico | Evento no recurrente **explícito** | Propagación espacial | Multi-vehículo | RL | Red real | Dónde se queda corto para esta tesis |
|---|---|---|---|---|---|---|---|---|
| Ichoua, Gendreau & Potvin (2003), EJOR | ✔ | ✘ | ✘ | ✘ | ✔ | ✘ | ✘ | Determinista; tabú estático |
| Gao & Chabini (2006), TR-B | ✔ | ✔ | ✘ | ✘ | ✘ (camino) | ✘ (DP) | ✘ | No hay VRP, no hay flota, no hay ventanas |
| Taş et al. (2014), TR-C | ✔ | ✔ (gamma) | ✘ | ✘ | ✔ | ✘ | ✘ | Estático, sin eventos, sin correlación |
| Y. Li, Gao & Liu (2011) | ✔ | parcial | ✔ (perturbación por accidente) | ✘ (solo el arco) | ✔ | ✘ (GA + Dijkstra) | ✔ | Reoptimiza el camino, **no la secuencia**; sin aprendizaje |
| Jie, Liu & Sun (2022) | ✔ | ✔ (U + N) | ✔ (nominalmente) | ✘ | ✔ | ✘ (PSO) | ✘ | Solución **estática**; el "evento" es ruido en la velocidad |
| Huang et al. (2017) | ✔ | ✔ | ✘ | ✘ (independencia entre arcos explícita) | ✔ | ✘ | ✔ | Sin eventos ni correlación espacial |
| **Chen, Imdahl, Lai & Van Woensel (2025), TR-C 172** | ✔ | ✔ | ✘ | ✘ | ✘ (**TSP**, un vehículo) | ✔ (DRL, atención) | parcial | Un solo vehículo; sin eventos discretos; sin ventanas blandas |
| **Chen, Imdahl, Lai & Van Woensel (2026), TR-C 182** | ✔ | ✔ | ✘ | ✘ | ✔ | ✔ (DRL) | parcial | El dinamismo son **solicitudes**, no eventos de red; sin propagación espacial |
| **Kamphuis, Levering & Mandjes (2025), C&OR 183:107148** | ✔ | ✔ | ✔ (**proceso de fondo markoviano** para efectos recurrentes y no recurrentes, calibrado con registros de incidentes y espiras) | ✔ (a nivel de enlace/red) | ✘ | ✘ | ✔ | **Consejo de hora de salida** para un OD; no hay VRP ni política de asignación |
| Mozhdehi, Wang, Sun & Wang (2025), SED2AM, [arXiv:2503.04085](https://arxiv.org/abs/2503.04085) | ✔ | ✘ | ✘ | ✘ | ✔ (multi-viaje) | ✔ (DRL) | ✔ (2 ciudades canadienses) | Determinista |
| Arani, Rezvani, Davarikia & Chan (2020), [arXiv:2001.08587](https://arxiv.org/abs/2001.08587) | ✔ | ✔ | ✔ (título) | ✘ | ✘ (un EV) | ✘ (heurística) | parcial | Camino de un vehículo; sin RL; venue menor |

**Lectura del cuadro.** Nadie combina las seis columnas. Pero dos trabajos están a **una columna** de
distancia:

- **Chen et al. (2026)** tiene todo excepto el **evento no recurrente explícito con estructura
  espacial**: su estocasticidad es ruido en los tiempos de viaje, no un proceso de eventos.
- **Kamphuis et al. (2025)** tiene un modelo de eventos no recurrentes genuinamente bueno (proceso de
  fondo markoviano, calibrado con bases de incidentes reales) pero **no hay problema de ruteo**: es
  consejo de hora de salida para un origen-destino.

### 5.3 Lo que no está resuelto

1. Ningún trabajo que yo haya podido verificar acopla **un proceso de eventos con estructura espacial**
   (propagación, correlación entre arcos) con **una política de RL multi-vehículo** sobre una red urbana
   real con ventanas de tiempo.
2. Ningún trabajo estudia el **valor de la información sobre incidentes** en un contexto de VRP: cuánto
   se gana pasando de ciego → detección con retardo → oráculo.
3. La literatura de RL para VRP evalúa sobre instancias sintéticas o sobre tiempos de viaje reales pero
   **sin escenarios de disrupción**; la única evidencia de *stress test* que conozco (SVRPBench,
   [arXiv:2505.21887](https://arxiv.org/abs/2505.21887), discutida en `rl-methodology-for-stdvrp.md`
   §3.3) reporta que los solvers neuronales se degradan >20 % bajo cambio de distribución.

---

## 6. Declaración de brecha defendible y preguntas abiertas

### 6.1 Lo que **no** se puede seguir afirmando

La tesis afirma en las Conclusiones (§8) que "la incorporación de congestiones no recurrentes que afectan
a arcos adyacentes constituye una novedad en la literatura del stochastic time-dependent vehicle routing
problem", y en §2.3-2.4 que "las metodologías de RL han sido escasamente exploradas en la literatura
relacionada con el TDVRP y STDVRP".

- La segunda afirmación **era razonable en junio de 2025 y ya no lo es**: Chen, Imdahl, Lai & Van Woensel
  publicaron en TR-C 172 (marzo 2025) y TR-C 182 (2026) exactamente RL para ruteo con tiempos de viaje
  dependientes del tiempo *y* estocásticos.
- La primera es **defendible pero frágil**: la novedad no es "congestión no recurrente" (Y. Li et al.
  2011 y Jie et al. 2022 ya la tienen, y la tesis los cita), sino la **propagación a arcos adyacentes**.
  Y esa propagación, tal como está implementada, va en la dirección física equivocada y tiene un footprint
  no justificado (§3). Un miembro de comité que lea el código puede desarmar la novedad.

### 6.2 Declaración de brecha que **sí** se sostiene

> Existen, por separado, (i) modelos de ruteo con tiempos de viaje dependientes del tiempo y estocásticos
> resueltos con deep RL —Chen, Imdahl, Lai & Van Woensel, TR-C 172 (2025) y TR-C 182 (2026)—, y
> (ii) modelos de red con congestión recurrente **y** no recurrente explícitamente separadas y calibradas
> con registros de incidentes —Kamphuis, Levering & Mandjes, C&OR 183 (2025)—, además de (iii) una
> literatura madura sobre generación de incidentes (ocurrencia NHPP, duración lognormal/AFT, reducción de
> capacidad tabulada por HCM/SHRP 2, propagación por percolación/contagio).
> **No existe, a la fecha de esta revisión, un trabajo que integre los tres:** una política aprendida
> (RL) para un problema de ruteo **multi-vehículo con ventanas de tiempo blandas**, sobre una red urbana
> **real**, donde los eventos no recurrentes se generan con un modelo **anclado en la ciencia del
> transporte** (ocurrencia condicionada al flujo, duración de cola pesada, severidad derivada de
> reducción de capacidad, propagación aguas arriba con spillback) y donde la **observabilidad parcial**
> de esos eventos es una variable de diseño y no un supuesto implícito.

Esa formulación es honesta, es verificable y **no depende** de que nadie más haya hecho "RL para STDVRP".

### 6.3 Cómo reencuadrar la contribución para que siga siendo defendible

Tres reencuadres, en orden de esfuerzo:

1. **Barato (reescritura, sin experimentos nuevos).** Cambiar la afirmación de novedad de "RL para
   STDVRP con NRC" a "**primer estudio que cuantifica la ventaja de una política adaptativa frente a una
   metaheurística estática de estado del arte bajo eventos no recurrentes espacialmente correlacionados,
   sobre red urbana real**", citando a Chen et al. como concurrente y a Kamphuis et al. como el
   antecedente del modelo de eventos. Añadir una sección de limitaciones que reconozca los puntos de §3.
2. **Medio (M0 + M3 + M11 + M12).** Hacer observable la congestión, poner duración lognormal, acoplar CRN
   y validar con BI/PTI. Con eso, la contribución pasa a ser "**una política de RL que aprende a
   anticipar eventos no recurrentes a partir de señales parciales de velocidad, validada contra métricas
   de fiabilidad estándar de la ciencia del transporte**". Esto es publicable.
3. **Caro (además M4 + M5 + M9).** El resultado que realmente falta en la literatura: **la curva del
   valor de la información sobre incidentes en ruteo de última milla**, de ciego a oráculo, con eventos
   físicamente correctos. Ese es un paper de TR-C / *Transportation Science*.

### 6.4 Preguntas de investigación abiertas

**PI-1. ¿Cuánto de la ventaja de una política adaptativa frente a una estática proviene de los eventos
no recurrentes, y cuánto del ruido recurrente?**
Hoy la tesis reporta mejoras de 9,35 %-36,40 % sobre la política estática sin descomponerlas. Con M2
(eliminar doble conteo) y M11 (estratificación por número de eventos) se puede aislar. *Hipótesis
falsable: si la mayor parte de la ventaja proviene del ruido gaussiano y no de los eventos, la
contribución de la tesis no es sobre congestión no recurrente.*

**PI-2. ¿Cuál es la curva del valor de la información sobre incidentes en un STDVRPTW?**
Comparar política óptima bajo (a) información nula, (b) detección con retardo lognormal, (c) oráculo.
Métrica: costo esperado normalizado. Nadie lo ha hecho para ruteo multi-vehículo con ventanas.

**PI-3. ¿La estructura de propagación importa para la política aprendida, o basta con la marginal por
arco?**
Comparar tres generadores con **la misma marginal de degradación por arco-minuto** pero distinta
estructura de correlación espacial (independiente / vecindario hacia adelante como hoy / cola aguas
arriba con spillback). Si la política aprendida es indistinguible, la propagación no es una contribución;
si no lo es, se cuantifica exactamente cuánto aporta.

**PI-4. ¿Se puede aprender una política robusta a *cambio de distribución* en el proceso de eventos?**
Entrenar con `φ ∈ [0.1, 0.4]`, δ uniforme, y evaluar bajo duración lognormal y severidad por capacidad
HCM. Conecta directamente con el hallazgo de SVRPBench (degradación >20 % de solvers neuronales bajo
cambio de distribución) y con la afirmación de generalización de Chen et al. (2026).

---

## 7. Bibliografía

**Fuentes oficiales de agencias (congestión, incidentes, capacidad, clima, fiabilidad)**

- Federal Highway Administration. *Reducing Non-Recurring Congestion* (Office of Operations).
  <https://ops.fhwa.dot.gov/program_areas/reduce-non-cong.htm> — desglose incidentes 25 % / clima 15 % /
  zonas de trabajo 10 %; "about half of congestion" es no recurrente. *(leído)*
- Cambridge Systematics / FHWA (2005). *Traffic Congestion and Reliability: Trends and Advanced
  Strategies for Congestion Mitigation*.
  <https://ops.fhwa.dot.gov/congestion_report/congestion_report_05.pdf> *(PDF descargado; el extractor de
  texto no lo parseó — se cita como referencia del informe subyacente, no por una cifra concreta)*
- FHWA (2013). *Guide for Highway Capacity and Operations Analysis of Active Transportation and Demand
  Management Strategies*, FHWA-HOP-13-042, Apéndice C.
  <https://ops.fhwa.dot.gov/publications/fhwahop13042/appc.htm> — Tabla 40 (HCM 2010 Exhibit 10-17,
  capacidad residual) y Tabla 41 (SHRP 2 L08 Exhibit 36-16, factores de ajuste). *(leído; tablas
  reproducidas en §2.4)*
- Morales, J. M. (FHWA). *Analytical Procedures for Estimating Freeway Traffic Congestion*. TRB
  Transportation Research Circular 344, pp. 38-46.
  <https://onlinepubs.trb.org/Onlinepubs/trcircular/344/344-007.pdf> — diagrama de colas determinista,
  ecuación general del retardo (forma cuadrática en las duraciones dividida por `2(S₁−S₃)`), Tabla 1 de
  capacidades de cuello de botella. *(leído directamente del PDF; el OCR de la ecuación está degradado —
  ver §8)*
- FHWA Road Weather Management. *Rain & Flooding*.
  <https://ops.fhwa.dot.gov/weather/weather_events/rain_flooding.htm> · *Snow & Ice*.
  <https://ops.fhwa.dot.gov/weather/weather_events/snow_ice.htm> — reducciones de velocidad por
  condición meteorológica. *(leído)*
- FHWA. *Travel Time Reliability: Making It There On Time, All The Time*.
  <https://ops.fhwa.dot.gov/publications/tt_reliability/ttr_report.htm> — definiciones de Buffer Index y
  Planning Time Index.
- Transportation Research Board / SHRP 2 (2012). *Analytical Procedures for Determining the Impacts of
  Reliability Mitigation Strategies* (Report S2-L03-RR-1). DOI
  [10.17226/22806](https://doi.org/10.17226/22806) ·
  <https://nap.nationalacademies.org/catalog/22806/analytical-procedures-for-determining-the-impacts-of-reliability-mitigation-strategies>
  *(catálogo y tabla de contenidos leídos; ecuaciones "data-poor"/"data-rich" no obtenidas — §8)*
- Transportation Research Board. *Highway Capacity Manual*, 6.ª ed. (2016) y 7.ª ed. (2022/2025).
  <https://nap.nationalacademies.org/read/24798/> *(registro bibliográfico; los valores de capacidad
  residual se citan vía FHWA-HOP-13-042, que reproduce Exhibit 10-17 del HCM 2010)*

**Duración y frecuencia de incidentes**

- Li, R., Pereira, F. C., Ben-Akiva, M. E. (2018). *Overview of traffic incident duration analysis and
  prediction.* **European Transport Research Review** 10(2):22. DOI
  [10.1186/s12544-018-0300-1](https://doi.org/10.1186/s12544-018-0300-1) ·
  [PDF](https://backend.orbit.dtu.dk/ws/files/149877717/filestore_2_.pdf) *(leído completo)*
- Golob, T. F., Recker, W. W., Leonard, J. D. (1987). *An analysis of the severity and incident duration
  of truck-involved freeway accidents.* **Accident Analysis & Prevention** 19(5):375-395.
  <https://pubmed.ncbi.nlm.nih.gov/3675808/> *(registro bibliográfico)*
- Giuliano, G. (1989). *Incident characteristics, frequency, and duration on a high volume urban
  freeway.* **Transportation Research Part A** 23(5):387-396.
  <https://www.sciencedirect.com/science/article/abs/pii/0191260789900861> *(registro bibliográfico;
  texto completo no obtenido — §8)*
- Nam, D., Mannering, F. (2000). *An exploratory hazard-based analysis of highway incident duration.*
  **Transportation Research Part A** 34(2):85-102.
  <https://econpapers.repec.org/article/eeetransa/v_3a34_3ay_3a2000_3ai_3a2_3ap_3a85-102.htm>
  *(registro bibliográfico; texto completo no obtenido — §8)*
- Lord, D., Mannering, F. (2010). *The statistical analysis of crash-frequency data: A review and
  assessment of methodological alternatives.* **Transportation Research Part A** 44(5):291-305.
  <https://trid.trb.org/view/917816> *(registro bibliográfico)*
- Mouhous, F., Aissani, D., Farhi, N. (2025). *A Stochastic Model for Traffic Incidents and Free Flow
  Recovery in Road Networks.* **Mathematics** 13(3):520.
  <https://www.mdpi.com/2227-7390/13/3/520> *(abstract y descripción del modelo vía búsqueda; la página
  devolvió 403 al fetch directo — §8)*

**Flujo, propagación y correlación espacial**

- Li, D., Fu, B., Wang, Y., Lu, G., Berezin, Y., Stanley, H. E., Havlin, S. (2015). *Percolation
  transition in dynamical traffic network with evolving critical bottlenecks.* **PNAS**
  112(3):669-672. DOI [10.1073/pnas.1419185112](https://www.pnas.org/doi/10.1073/pnas.1419185112)
- Saberi, M., Hamedmoghadam, H., Ashfaq, M. et al. (2020). *A simple contagion process describes
  spreading of traffic jams in urban networks.* **Nature Communications** 11:1616. DOI
  [10.1038/s41467-020-15353-2](https://www.nature.com/articles/s41467-020-15353-2)
- Cassidy, M. J., Bertini, R. L. (1999). *Some traffic features at freeway bottlenecks.*
  **Transportation Research Part B** 33(1):25-42. *(citado por la literatura de capacity drop revisada;
  registro no verificado directamente — §8)*
- Liu, B., Long, J., Deng, M., Tang, J., Huang, J. (2022). *Revealing spatiotemporal correlation of
  urban roads via traffic perturbation simulation.* **Sustainable Cities and Society** 77:103545.
  <https://www.sciencedirect.com/science/article/abs/pii/S2210670721008118> *(existencia y tema
  verificados; los valores 0.83/0.78/0.73 usados en el código NO verificados — §8)*

**Datos de Chengdu y distribuciones de velocidad**

- Guo, F., Zhang, D., Dong, Y., Guo, Z. (2019). *Urban link travel speed dataset from a megacity road
  network.* **Scientific Data** 6:61. DOI
  [10.1038/s41597-019-0060-3](https://doi.org/10.1038/s41597-019-0060-3) ·
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6522518/) *(leído: 1.902 nodos, 5.943 links, 45 días,
  cinco ventanas de 2 h, 97,81 % normalidad, 49,75 % de links con correlación significativa)*
- Maghrour Zefreh, M., Török, Á. (2020). *Distribution of traffic speed in different traffic conditions:
  an empirical study in Budapest.* **Transport** 35(1):68-86. DOI
  [10.3846/transport.2019.11725](https://doi.org/10.3846/transport.2019.11725) *(página del editor
  leída: normal en flujo libre, gamma en congestión, lognormal en flujo subsaturado)*
- DiDi Chuxing GAIA Open Dataset (trayectorias y Travel Time Index de Chengdu).
  <https://outreach.didichuxing.com/> *(no verificado el estado actual de acceso)*

**TDVRP / STDVRP y caminos en redes estocásticas**

- Ichoua, S., Gendreau, M., Potvin, J.-Y. (2003). *Vehicle dispatching with time-dependent travel
  times.* **EJOR** 144(2):379-396. DOI
  [10.1016/S0377-2217(02)00147-9](https://doi.org/10.1016/S0377-2217(02)00147-9) ·
  [PDF](https://www.iro.umontreal.ca/~marcotte/PLU6000/PLU6000_H04/Ichoua2.pdf)
- Miller-Hooks, E. D., Mahmassani, H. S. (2000). *Least expected time paths in stochastic, time-varying
  transportation networks.* **Transportation Science** 34(2):198-215.
  <https://www.scholars.northwestern.edu/en/publications/least-expected-time-paths-in-stochastic-time-varying-transportati>
- Gao, S., Chabini, I. (2006). *Optimal routing policy problems in stochastic time-dependent networks.*
  **Transportation Research Part B** 40(2):93-122.
  <https://www.sciencedirect.com/science/article/abs/pii/S0191261505000391> ·
  [PDF de autor](https://people.umass.edu/sgao/orp.pdf)
- Blauth, J., Held, S., Müller, D., Schlomberg, N., Traub, V., Tröbst, T., Vygen, J. (2022/2024).
  *Vehicle Routing with Time-Dependent Travel Times: Theory, Practice, and Benchmarks.*
  [arXiv:2205.00889](https://arxiv.org/abs/2205.00889)
- Kamphuis, R., Levering, N., Mandjes, M. (2025). *Optimal departure-time advice in road networks with
  stochastic disruptions.* **Computers & Operations Research** 183:107148. DOI
  [10.1016/j.cor.2025.107148](https://doi.org/10.1016/j.cor.2025.107148) ·
  [arXiv:2208.14516](https://arxiv.org/pdf/2208.14516)

**RL para VRP (revisiones y los trabajos más cercanos)**

- Hildebrandt, F. D., Thomas, B. W., Ulmer, M. W. (2023). *Opportunities for reinforcement learning in
  stochastic dynamic vehicle routing.* **Computers & Operations Research** 150:106071.
  <https://www.sciencedirect.com/science/article/abs/pii/S030505482200301X> · <https://trid.trb.org/view/2060446>
- Bogyrbayeva, A., Meraliyev, M., Mustakhov, T., Dauletbayev, B. (2024). *Machine Learning to Solve
  Vehicle Routing Problems: A Survey.* **IEEE T-ITS** 25(6). DOI
  [10.1109/TITS.2023.3334976](https://doi.org/10.1109/TITS.2023.3334976) ·
  [arXiv:2205.02453](https://arxiv.org/pdf/2205.02453)
- Bai, R. et al. (2023). *Analytics and machine learning in vehicle routing research.* **International
  Journal of Production Research** 61(1):4-30. DOI
  [10.1080/00207543.2021.2013566](https://doi.org/10.1080/00207543.2021.2013566) ·
  [arXiv:2102.10012](https://arxiv.org/abs/2102.10012)
- Zhou, F., Lischka, A., Kulcsár, B., Wu, J., Haghir Chehreghani, M., Laporte, G. (2025). *Learning for
  routing: A guided review of recent developments and future directions.* **Transportation Research
  Part E** 202:104278. [arXiv:2507.00218](https://arxiv.org/abs/2507.00218)
- Chen, D., Imdahl, C., Lai, D., Van Woensel, T. (2025). *The Dynamic Traveling Salesman Problem with
  Time-Dependent and Stochastic travel times: A deep reinforcement learning approach.* **Transportation
  Research Part C** 172:105022. DOI
  [10.1016/j.trc.2025.105022](https://doi.org/10.1016/j.trc.2025.105022) ·
  [SSRN 4809480](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4809480) *(403; abstract vía
  búsqueda — §8)*
- Chen, D., Imdahl, C., Lai, D., Van Woensel, T. (2026). *A reinforcement learning approach for the
  dynamic vehicle routing and scheduling problem with stochastic request times and time-dependent,
  stochastic travel times.* **Transportation Research Part C** 182:105387. DOI
  [10.1016/j.trc.2025.105387](https://doi.org/10.1016/j.trc.2025.105387) *(metadatos vía Crossref;
  ScienceDirect 403 — §8)*
- Mozhdehi, A., Wang, Y., Sun, S., Wang, X. (2025). *SED2AM: Solving Multi-Trip Time-Dependent Vehicle
  Routing Problem using Deep Reinforcement Learning.* [arXiv:2503.04085](https://arxiv.org/abs/2503.04085)
- Arani, M., Rezvani, M. M., Davarikia, H., Chan, Y. (2020). *Routing of Electric Vehicles in a
  Stochastic Network with Non-recurrent Incidents.* [arXiv:2001.08587](https://arxiv.org/abs/2001.08587)
- SVRPBench (2025). [arXiv:2505.21887](https://arxiv.org/abs/2505.21887) *(discutido en
  `rl-methodology-for-stdvrp.md` §3.3; aquí solo se referencia)*

---

## 8. Limitaciones de esta revisión / qué no se pudo verificar

1. **El archivo legado fue borrado durante la sesión.** `Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`
   aparece como `D` (borrado, en índice) en `git status`; lo recuperé con
   `git show HEAD:…` desde el commit `636d28c`. Las líneas citadas corresponden a esa versión. Si el
   borrado forma parte del refactor en curso, cite en la tesis el tag `legacy-monolith` (ADR-0001).
2. **No pude ejecutar el código ni medir nada.** Los datos de instancia no están versionados
   (`data/README.md`). En particular **no verificado**: el número esperado de eventos por época
   `Λ = Σ_ij P_ij`, el número esperado de eventos por episodio, la fracción de arcos afectados
   simultáneamente, y por tanto **si la tasa de eventos calibrada es plausible o excesiva**. Es el primer
   número que hay que instrumentar y reportar.
3. **Los valores 0.83 / 0.78 / 0.73 no están verificados contra Liu et al. (2022).** El paper existe, el
   tema coincide (simulación de perturbaciones para revelar correlación espacio-temporal), pero
   ScienceDirect devolvió acceso restringido y no pude confirmar que esos tres números aparezcan en él ni
   con qué significado. **Verifíquelo antes de defender la tesis**: es el único anclaje bibliográfico de
   la propagación, que es la novedad declarada.
4. **La ecuación general de retardo de Morales se leyó con OCR degradado.** El PDF de TRB Circular 344 se
   extrajo correctamente en texto narrativo y en la Tabla 1, pero la ecuación general aparece con los
   subíndices corrompidos. Lo que **sí** está verificado es la estructura: es una suma de términos
   `T_i²·(diferencias de flujo)` y productos cruzados `2·T_i·T_j·(…)`, dividida por `2(S₁−S₃)`. La
   afirmación "el retardo es cuadrático en la duración" se apoya en esa estructura, no en una lectura
   literal de la fórmula. Para citarla textualmente en la tesis, consiga el PDF original de Morales
   (1987), *ITE Journal* 57(1):45-49.
5. **Textos completos no obtenidos** (uso solo registro bibliográfico, ninguna cifra sale de ellos):
   Giuliano (1989); Nam & Mannering (2000); Golob et al. (1987); Lord & Mannering (2010); Cassidy &
   Bertini (1999); Chen et al. TR-C 172 y TR-C 182 (ScienceDirect y SSRN devolvieron 403); Mouhous et al.
   (2025) (MDPI devolvió 403 al fetch, aunque el abstract se recuperó vía búsqueda); las ecuaciones
   "data-poor"/"data-rich" de SHRP 2 L03. **Por eso esta nota no propone valores numéricos de μ_ln, σ_ln
   ni de tasas de incidentes por VMT tomados de la literatura, y en su lugar propone calibrarlos contra
   los datos de Chengdu.**
6. **Número de carriles por arco:** `link.csv` de Guo et al. (2019) tiene `Length`, coordenadas y nodos,
   pero **no** número de carriles ni clase funcional. La mejora M4 (capacidad residual por carriles
   bloqueados) requiere imputarlos (por ejemplo desde OpenStreetMap por *map matching*). No verifiqué la
   viabilidad de ese *matching*.
7. **Datos climáticos de Chengdu jun-jul 2015** para calibrar `p_w` en M8: no buscados.
8. **Velocidad de propagación de ondas de choque** (el "1-2 km/h" mencionado en M5) es un orden de
   magnitud de memoria, **no verificado** contra fuente primaria en esta revisión. Calíbrelo o cite
   Newell / el HCM antes de usarlo.
9. **La afirmación de que "nadie combina las seis columnas"** (§5.2) se basa en búsquedas en arXiv,
   Crossref, Semantic Scholar y buscador web sobre TR-C, TR-E, EJOR, C&OR, *Transportation Science* e
   *INFORMS J. on Computing* durante 2020-2026. **No es una revisión sistemática** y no puedo excluir un
   trabajo en una revista menor o en actas. Antes de imprimir la declaración de brecha, corra una
   búsqueda estructurada en Scopus/WoS con las cadenas `("vehicle routing") AND ("reinforcement
   learning" OR "approximate dynamic programming") AND ("incident" OR "non-recurrent" OR "disruption")`
   y `… AND ("time-dependent") AND ("stochastic travel time")`.

---

## Archivos del repositorio referenciados

- `src/stdvrp/congestion/generator.py` — `CongestionGenerator` (seam) y `ArcProbabilityCongestionGenerator`
- `src/stdvrp/traffic/travel_time_model.py` — `_compute_event_probabilities` (`:294-335`)
- `src/stdvrp/simulation/model.py` — cadencia del sorteo (`:267-271`), expiración (`:499-517`),
  velocidad bajo congestión (`:520-543`)
- `src/stdvrp/simulation/episode.py` — orden de sembrado de RNG (`:86-87`, `:182-183`, `:221-224`)
- `src/stdvrp/config.py` — parámetros de congestión (`:49-51`, `:107-111`)
- `Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py` **en `git show 636d28c:…`** — legado
- `docs/research/rl-methodology-for-stdvrp.md` — nota hermana sobre el estimador de RL
- `CONTEXT.md` — glosario del dominio (`CongestionGenerator`, `TravelTimeModel`, `Model`)
