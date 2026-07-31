# Revisión del simulador STDVRP

**Fecha:** 2026-07-28 · **Rama:** `FEATURE_DEV_total_code_refactorization` · **Commit base:** `5a456c6`

Revisión de correctitud de la simulación: física del movimiento, contabilidad de
distancias y costes, orden de eventos del reloj, congestión y la lectura del State que
hace la Policy. No es una revisión de estilo, tipos ni rendimiento.

---

## Veredicto

**La sospecha original no se confirma: la actualización de velocidades y posiciones
*dentro de un arco* es correcta.** La integración distancia = velocidad × tiempo es
consistente hasta la precisión de coma flotante, el reloj nunca retrocede, no hay doble
cobro y no se cobra distancia a vehículos parados. Esto se verificó con instrumentación
sobre cientos de episodios, no por lectura.

**Pero el simulador sí tiene defectos, y dos afectan directamente a lo que se mide.** El
primero retira vehículos de la carretera a mitad de arco; el segundo hace que la demanda
abandonada antes de que venza su ventana no cueste nada, lo que vuelve el coste de un
episodio **no comparable entre episodios que terminan en momentos distintos** — y es
exactamente el número por el que se elige `best_w`.

| # | Hallazgo | Severidad | Efecto medido |
|---|----------|-----------|---------------|
| B1a | **Cadena del depósito** — *capa 1 (política)*: la regla depot-idle deja fuera de servicio a vehículos que solo *pasan* por el depósito | **Alto** | Hasta media flota inactiva con clientes pendientes |
| B1b | *capa 2 (modelo)*: ese vehículo es teletransportado al depósito; arco restante y vuelta nunca se cobran | Medio | +1.24 % coste medio al corregir; 0.19 km/episodio gratis |
| B3 | La fórmula de retraso es correcta en vivo, pero se reutiliza sin cambios al terminar: demanda abandonada dentro de ventana cuesta 0 | **Alto** | Mismos 19 clientes: 0.00 vs 11 245.00 según cuándo pare la flota |
| B4 | El único canal de tráfico ya recorrido (`mean_velocities`) respeta la observabilidad parcial pero se calcula y se descarta | Medio | Dato calculado en cada decisión, nunca leído |
| B5 | Crash `min()` sobre secuencia vacía en el endgame | Medio | Reproducido en tau ∈ (310, 350] |
| B6 | Las features miden desde el nodo del que el vehículo salió | Medio | ETA optimista en 78.5 % de los pares, media +4.03 min |
| B7 | Un evento de congestión posterior trunca la expiración del anterior | Medio | ~1400 entradas/época terminan hasta 90 min antes |
| B8 | La saturación del multiplicador anula el decaimiento por distancia | Medio | ~3/4 de los arcos con el mismo 0.4 sin importar la distancia |
| B9 | `_reachable_nodes` es DFS, no BFS: profundidades no mínimas | Medio | ~15 % de los nodos dentro de `max_depth` nunca se alcanzan |
| B10 | Feature 10 idénticamente cero; ningún bin cuenta ventanas ≥ 600 | Medio | 18 pesos entrenables de 19; ~29 % de la demanda sin contar |
| B11 | Asignación duplicada del mismo cliente a dos vehículos | Medio | 734 transiciones en 60 episodios |
| B12 | `horizon_end_minute` no acota el episodio; manda el 1150 hardcodeado | Medio | `horizon_end=1400` da episodios idénticos a 780 |
| B13 | El cap de 60 km/h censura la distribución sobre un átomo | Bajo | Hasta 49 % de la masa en un punto; media realizada −4.3 % |
| B14 | `delay_clients = 0` con `delay_cost = 4205` | Bajo | Reporte contradictorio |
| B15 | Horas extra **negativas** si `horizon_end_minute > 1148` | Latente | −43.33 por vehículo con horizonte 1200 |
| B16 | La cadencia de congestión usa módulo en coma flotante | Latente | Trampa de configuración, no afecta a lo ya ejecutado |
| B17 | El libro de congestión nunca purga los eventos expirados | Bajo | 116/116 arcos expirados en el fixture |
| B18 | El nombre sugiere "N arcos"; el diseño real es una ventana de las últimas N velocidades observadas (correcto, solo mal nombrado) | Bajo | 68 % son remuestreos del mismo arco — dato descriptivo, no un fallo |
| B19 | El endpoint derecho de la ventana de std no es la primera observación | Bajo | Mediana 10 % de diferencia relativa en la std |
| B20 | Una acción debe ser ejecutable: `path_between(n, n)` no lanza, y el crash aguas abajo depende del action set de la Policy — no descartado, ver *Hipótesis descartadas* de la versión anterior de este informe | **Alto** (crash) | 3 crashes / 80 episodios (transformer, ε-greedy, sin entrenar); 0/600 con `MonteCarloPolicy` lineal |

---

## Método

- Lectura completa de `src/stdvrp/simulation/`, `traffic/`, `congestion/`, `policies/`,
  `demand/`, `network/`, más `docs/adr/0001` y la suite de invariantes.
- Auditoría en paralelo por siete dimensiones, **cada hallazgo sometido después a una
  pasada adversarial de refutación** que reimplementaba el arreglo propuesto y lo medía.
  De 23 hallazgos, 22 sobrevivieron y 1 fue refutado; varios cambiaron de severidad o de
  atribución en esa pasada (se indica abajo dónde).
- **Instrumentación de episodios reales** sobre `tests/fixtures/chengdu_mini` con la
  configuración real del experimento (`experiments/chengdu/config.yaml`: horizonte
  300–780, `max_congestion_duration: 120`, `n_observed_arcs: 3`), y comprobaciones de
  datos contra el archivo completo de Chengdu.

Todo lo que se afirma abajo está medido. Donde una hipótesis no se sostuvo, se dice
explícitamente (ver *Hipótesis descartadas*).

---

## Lo que está bien (verificado, no asumido)

Conviene decirlo primero, porque es el núcleo de lo que preocupaba:

1. **Conservación de distancia, arco por arco.** La distancia que `advance_fleet_to` cobra
   entre un `begin_arc` y el siguiente coincide con la longitud del arco en `travel_data`
   con error relativo < 1e-7, en **675 episodios** (25 semillas × 3 probabilidades de
   evento × 3 duraciones × 3 tamaños de flota), con y sin W aleatorio. Cero violaciones.

2. **No hay doble cobro.** `resample_arc` solo alimenta `arc_distance_travelled` (que sirve
   para acortar el arco restante) y nunca toca el ledger; `advance_fleet_to` es el único
   que cobra.

3. **El reloj nunca retrocede** y `arc_distance_travelled` nunca supera la longitud del
   arco, así que nunca se programa una llegada en el pasado. `advance_fleet_to` nunca se
   llama con un objetivo posterior a la primera llegada pendiente: los seis sitios de
   llamada pasan o `earliest_arrival()`, o `next_decision_tau`, o una expiración de
   congestión, y cada uno es el mínimo de los tres eventos candidatos.

4. **Un vehículo en servicio no se cobra**: `_hold_for_service` pone la velocidad observada
   a 0.

5. **Las rutas están bien formadas.** El invariante `route[v][0] == vehicle_position[v]` se
   cumple en todos los puntos de re-ruteo (0 violaciones en 380 episodios); no hay rutas
   degeneradas; y como el último nodo de una ruta es siempre el `destination` actual, las
   comparaciones `nodes_left == 2` / `<= 2` son exactas, no off-by-one.

6. **La longitud de un arco no cambia a mitad de recorrido**: ningún par
   (`Node_Start`, `Node_End`) aparece en más de un `Link` en el `link.csv` real de 5943
   filas, así que todas las entradas por minuto de un arco llevan la misma longitud.

7. **No hay huecos de datos en el horizonte** en el archivo real: la caída documentada de
   `_interpolated_speed` a `None` (velocidad NaN) nunca se dispara.

8. **La captura de entrenamiento es correcta.** `TrainingSnapshot` cubre exactamente los
   campos que el replay de `update_W` lee; no hay aliasing con el State vivo, y el retorno
   Monte Carlo está bien alineado (`U_t += rewards[t+1]` con `rewards[0] = 0`).

---

## La causa raíz común: `vehicle_position` no significa "dónde está el vehículo"

Tres de los hallazgos graves salen del mismo malentendido, así que conviene aislarlo.

`State.vehicle_position` guarda **el último nodo que el vehículo alcanzó**, no dónde está.
`vehicle_reaches_node` (`model.py:517`) escribe ese campo para un nodo por el que el
vehículo simplemente *pasa*, y acto seguido `begin_arc` lo lanza al siguiente arco. Nada
distingue "parado en el nodo" de "cruzando el nodo".

Eso importa porque el depósito **no es solo el punto de partida**: 138 de las 2025 rutas
más cortas cacheadas (6.8 %) lo usan como nodo interior. Se midieron **75 cruces del
depósito en 60 episodios**. Cada uno deja al vehículo con `vehicle_position == depot`
mientras rueda.

De ahí sale la **cadena del depósito** (B1), el defecto más importante de este informe.

---

## Bugs

### B1 — La cadena del depósito

**Es un solo defecto en dos capas, no dos bugs independientes.** Comparten disparador y en
las mediciones nunca aparecen por separado: de los 71 disparos que cazaron a un vehículo en
marcha, **los 71** venían de la regla de la capa 1. Se documentan por separado porque viven
en módulos distintos, **hacen falta dos arreglos**, y **arreglar uno no cura el daño del
otro**:

| | Dónde | Qué rompe | Si se arregla solo esta |
|---|---|---|---|
| **Capa 1** | `policies/monte_carlo.py:297` | La flota pierde capacidad | El vehículo sigue trabajando y la capa 2 deja de alcanzarse por esta vía |
| **Capa 2** | `simulation/model.py:376` | Física y contabilidad | El vehículo termina el arco, vuelve y aparca — paga el trayecto, pero **sigue saliendo de servicio** |

---

#### B1a — Capa 1: la regla depot-idle deja fuera de servicio a vehículos en marcha

**Severidad: alta.** `src/stdvrp/policies/monte_carlo.py:297-300`.

```python
if (
    self.state.vehicle_position[vehicle] == self.depot and self.state.tau_episode > 350
) or len(self.state.clients_not_visited) == 0:
    possible_actions.append(self.depot)
```

Cuando esta condición se cumple, el depósito es la **única** acción candidata. Para un
vehículo que acaba de cruzar el depósito camino de su cliente, eso significa que la
política lo manda a casa por accidente: no está parado en el depósito, está pasando por él.

El literal 350 sí está documentado como quirk heredado (`monte_carlo.py:62-64`, «Depot-idle
cutoffs are inconsistent literals»), pero **la consecuencia no está documentada en ninguna
parte**.

**Medido:** el efecto sobre la flota es lo llamativo. En el fixture, un vehículo 1–3
minutos pasado el depósito, con 5–12 clientes pendientes y ~2.5 h de horizonte por delante,
deja de moverse. En la semilla 42 eso deja **media flota** inactiva. El tamaño de flota
reportado sobreestima la flota que la política realmente opera.

**Arreglo propuesto:** la condición necesita distinguir "parado en el depósito" de "el
depósito fue el último nodo alcanzado" — p. ej. exigiendo además que el vehículo no esté
viajando (`fleet.departure_tau[v] >= tau`), o introduciendo un campo explícito de presencia
en nodo en lugar de inferirlo. Cambia decisiones → re-baseline del golden master.

---

#### B1b — Capa 2: el vehículo retirado es teletransportado al depósito

**Severidad: media (física y contabilidad).** `src/stdvrp/simulation/model.py:376-382`.

```python
elif (
    action[vehicle] == self.depot and self.state.vehicle_position[vehicle] == self.depot
):
    # Not ``FleetRoutes.park``: the legacy leaves ``departure_tau``
    # alone here, unlike the vehicle that arrives at the depot.
    fleet.arrival_tau[vehicle] = PARKED
    fleet.horizon_change_tau[vehicle] = self.state.tau_episode
```

La rama es **correcta y frecuente** en su caso normal: un vehículo genuinamente parado en el
depósito (o ya retirado) al que se le dice "quédate ahí". Eso es un no-op sin coste.

**Cuándo se convierte en bug.** Hacen falta las cuatro condiciones a la vez:

1. `tau > 350` — el literal depot-idle de B1a, que es lo que hace que la política emita
   `action = depot` sin alternativa. *Medido: el disparo más temprano fue en tau = 351.0;
   ninguno por debajo de 350.*
2. **El camino más corto del vehículo hacia su cliente cruza el nodo depósito** como punto
   intermedio (6.8 % de las rutas cacheadas lo hacen). Al cruzarlo, `vehicle_reaches_node`
   escribe `vehicle_position = depot` y `begin_arc` lo lanza de inmediato al arco siguiente.
3. **La época de decisión cae mientras aún recorre ese arco de salida del depósito** — una
   ventana de ~2–4 minutos en el fixture.
4. No está dentro de la ventana de servicio de un cliente (si lo estuviera ganaría la rama 1
   de `_reroute_for`).

**El caso "acaba de salir del depósito por primera vez" no ocurre en la práctica**, y la razón
es instructiva: después del minuto 350 un vehículo **no puede iniciar un viaje desde el
depósito**, porque estando parado allí la política solo le ofrece el depósito y lo retira en el
acto (eso es B1a). Y las salidas del arranque del episodio (tau = 300) ya han llegado a su
primer nodo mucho antes de 351. Así que el único origen posible es el cruce del punto 2.

*Medido sobre 180 episodios (60 semillas × flotas 1/3/6):* de todos los disparos de esta rama,
**71 cazaron a un vehículo en marcha** — y los 71 venían de cruzar el depósito como nodo
interior, ninguno de una primera salida. Los 71 provenían de la regla depot-idle; ninguno de
las otras condiciones que también pueden emitir `action = depot` (no quedan clientes, endgame,
retorno por horizonte). Reloj de los disparos: entre 351.0 y 1094.0.

Cuando esas condiciones se cumplen, la rama declara al vehículo aparcado **en el depósito**
mientras está físicamente en mitad de un arco. Consecuencias, todas verificadas:

- el resto del arco **y el viaje de vuelta nunca se recorren ni se cobran**
  (`advance_fleet_to` salta los `PARKED`, `model.py:539`);
- **nunca se le pueden cobrar horas extra**: su posición registrada es el depósito, así que
  el conteo `position != depot` de `terminate_state_passing_horizon` (línea 554) lo excluye,
  y `_vehicle_parks_at_depot` no llega a ejecutarse;
- **no se puede revivir**: `is_travelling` es `False`, así que la rama de re-ruteo de la
  línea 384 queda muerta para él para siempre;
- cuenta como "en casa" para `all_parked()` y `_every_vehicle_home_and_no_clients_left()`,
  así que puede terminar el episodio en el sitio.

**Frecuencia medida:** ~0.4 disparos con vehículo en marcha por episodio; **24 de 60
episodios** afectados (hasta 3 por episodio), y 100 disparos a mitad de arco en otra tanda de
200 episodios. Reproducción propia
independiente: 8 de 20 episodios con un solo vehículo, con 0.03–1.01 km ya recorridos; en la
semilla 14 el vehículo fue retirado **0.03 minutos antes de llegar** a su cliente.

**Magnitud honesta del coste:** al implementar el guardia y volver a medir sobre 180
episodios, el coste medio pasa de **1060.27 a 1073.39 (+1.24 %)** y se recuperan 34.14 km
(0.19 km/episodio) que hoy el simulador conduce gratis. Es decir: lo grave es la violación
de la física y la contabilidad, no la cifra.

**Aviso importante sobre el arreglo.** Poner el guardia **no** devuelve el vehículo al
servicio: con él, el vehículo cae en la rama de re-ruteo, termina el arco, conduce de vuelta
al depósito y aparca. Sale de servicio igual — pero ahora pagando el trayecto. Para que el
vehículo *siga trabajando* hay que corregir además B1a.

**¿Por qué documentarlo aparte si nunca se observó sin B1a?** Porque la capa 2 tiene
disparadores que no pasan por el literal 350. La primera rama de
`_select_vehicle_possible_actions` es
`(position == depot and tau > 350) or len(clients_not_visited) == 0`: **ese segundo
disyuntivo no depende del reloj**. Cuando se sirve el último cliente, todos los vehículos
reciben `action = depot`, y cualquiera que esté cruzando el depósito en ese instante se
teletransporta igual — perdiendo el cobro del trayecto de vuelta, aunque sin pérdida de
capacidad porque ya iba a casa. Lo mismo puede ocurrir por la rama de endgame o cuando el
`argmin` elige el depósito por el guardia de retorno por horizonte. En 180 episodios medidos
esas vías salieron **0 veces**, así que son alcanzables estructuralmente pero no observadas
en este fixture.

---

### B3 — La fórmula de retraso es correcta en vivo, pero se reutiliza mal al terminar

**Severidad: alta (definición del objetivo).** `src/stdvrp/simulation/model.py:576-584`.

```python
self.costs.charge_unserved_delays(
    tau_episode - time_windows[client][1]
    for client in self.state.clients_not_visited
    if tau_episode > time_windows[client][1]
)
```

**Primero lo que está bien.** `cost(t) = max(0, t - due)` es la forma estándar y correcta de
penalizar un retraso: continua, cero mientras la ventana no ha vencido, creciente después. Y
es exactamente lo que *debe* pasar mientras el episodio sigue corriendo — un cliente aún no
visitado con ventana abierta no debería costar nada todavía, porque en teoría un vehículo
todavía podría llegar a tiempo. Eso no es un bug: es el comportamiento correcto de un hinge
cost mientras el proceso sigue vivo.

**El problema es aplicar esa misma fórmula, en el mismo instante, a un evento distinto.**
`_charge_unserved_delays` solo se invoca una vez, al **terminar** el episodio
(`terminate_state_passing_horizon` o `terminate_state_if_all_vehicles_come_back`). En ese
instante, "la ventana no ha vencido todavía" ya no significa "todavía hay tiempo de
servirlo" — significa que el cliente **nunca va a ser servido**, y además cae antes de su
fecha límite. Son dos eventos semánticamente distintos:

- **durante el episodio**, un cliente pendiente con ventana abierta: correcto que cueste 0,
  porque el desenlace todavía está abierto;
- **en la terminación**, ese mismo cliente: el desenlace ya está decidido (abandono
  definitivo), pero el código sigue evaluando la fórmula "en vivo" como si aún hubiera
  margen.

El simulador no distingue ambos casos: usa la misma expresión para "todavía puede llegar a
tiempo" y para "ya nunca llegará". La continuidad de la fórmula no es el defecto; el defecto
es no tener una fórmula *distinta* para el evento de terminación.

Esto **no está documentado en ningún sitio**: `docs/adr/0001` menciona la ruta all-back solo
en el arreglo 5 de fase 2 (la fórmula `1150 - due` vs `tau - due`), y ni el docstring de
`Model` ni la lista de quirks de `cost_ledger` mencionan a los clientes no servidos dentro de
ventana. Es comportamiento heredado del monolito (línea 5609) y hay un test que lo fija, pero
nada le dice al investigador que el objetivo **no contiene ningún término por demanda no
atendida** que aún no estuviera vencida.

**Evidencia dentro de la misma semilla** (la comparación que aísla el defecto de la fórmula,
sin depender de si el reloj de parada "debería" haber sido otro): en la semilla 14, los
mismos 19 clientes no servidos —la misma demanda real abandonada, ni un cliente más ni
menos— valen **0.00** si se evalúan al reloj en que la flota efectivamente paró, y
**11 245.00** si se evalúan al reloj del horizonte de emergencia (1150). Mismo abandono de
demanda, dos precios que difieren en cuatro órdenes de magnitud, únicamente en función de
*cuándo* se evaluó la fórmula — no de cuánta demanda real se perdió.

**Frecuencia:** 15 de los 60 episodios de un vehículo dejaron al menos un cliente en ventana
sin pagar (3 de ellos solo clientes en ventana), 116 clientes en total. Con W aleatorio no
nulo, 13 de 200 episodios multi-vehículo también.

**Por qué importa para la investigación:**

- `Trainer.train` fija `best_w` por el **mínimo coste medio de evaluación**
  (`trainer.py:266-277`), y `run_training_episode` devuelve los mismos cargos por transición
  como retorno Monte Carlo. Cualquier W que acorte los episodios se puntúa mejor.
- Una media que mezcle episodios terminados por horizonte (donde todo cliente no servido se
  cobra) con episodios all-back (donde los que están en ventana son gratis) **está
  promediando dos funciones de coste distintas**.

**Matiz de honestidad:** en la semilla 14 la flota no *eligió* rendirse — fue eliminada por
la cadena del depósito (B1a/B1b). La política no tiene agencia directa para "decidir
abandonar": solo se le ofrece el depósito bajo las condiciones de
`_select_vehicle_possible_actions`. El canal explotable es indirecto (qué clientes se visitan
y cuándo un vehículo acaba ocioso en el depósito). El defecto de tarificación es real e
independiente de B1a/B1b — la comparación 0.00 vs 11 245.00 dentro de la misma semilla lo
demuestra sin depender de si ese episodio en particular terminó por el bug o por una decisión
legítima de la política.

**Arreglo propuesto:** separar las dos fórmulas explícitamente. La penalización "en vivo"
(cero mientras la ventana no vence) puede quedarse tal cual — es correcta. Lo que hace falta
es una fórmula **distinta para el evento de terminación**, que no dependa del reloj en que la
flota casualmente se detuvo: p. ej. tarificar todo cliente no servido al final del episodio
como si el reloj hubiera llegado al fin del horizonte configurado
(`max(horizon_end, tau) - due`, con piso en 0 solo si el resultado da negativo) más,
opcionalmente, un cargo fijo por no-servicio que capture el hecho de que esa demanda se pierde
para siempre y no es lo mismo que "llegar tarde". Es un cambio de **definición del
objetivo**: merece un ADR antes que código.

---

### B4 — El único canal de tráfico realizado que respeta la observabilidad parcial (`mean_velocities`) se calcula y se descarta

**Severidad: media (acotado a partir de lo que primero parecía un defecto estructural).**
`src/stdvrp/policies/feature_extraction.py:175-176`.

**Primero, lo que es diseño correcto y no un defecto.** Todas las features de ETA, distancia
y coste esperado hacia clientes candidatos —incluidos los que un vehículo aún no ha
visitado— salen de `EpisodeGeometry`, que copia los promedios históricos precalculados del
`ShortestPathCache`. Eso es lo correcto por diseño: un vehículo no puede legítimamente conocer
la velocidad real de un arco por el que **no ha pasado todavía** — solo observa velocidades
en los arcos que efectivamente recorre (`state.observed_velocity`), nunca en arcos distintos.
Usar el promedio histórico para tasar arcos aún no recorridos respeta esa restricción de
observabilidad parcial; no es una omisión, es el comportamiento esperado de un agente que no
tiene información privilegiada sobre el futuro.

**Lo que sí queda como hallazgo, dentro de esa misma restricción.** El vector
`state.observed_velocity` — que por construcción solo contiene velocidades de arcos que el
vehículo **ya recorrió**, exactamente la información que el diseño permite ver — se resume en
`mean_velocities`:

```python
mean_velocities=tuple(
    sum(velocities) / len(velocities) for velocities in state.observed_velocity
),
```

y el propio docstring del módulo reconoce que `mean_velocities` «nunca se añade como
feature. Nada lo lee». Un `grep` sobre `src/` lo confirma: no hay otro lector. Es decir: el
único canal que respeta al pie de la letra la restricción "solo arcos ya recorridos, nunca
arcos distintos" se calcula en cada decisión y se tira — no porque el diseño exija
descartarlo, sino porque nadie lo conectó al vector de 19 features.

**Matiz relacionado (ver B18):** incluso si se conectara, lo que hoy contiene ese vector no es
exactamente "la velocidad media de los últimos arcos distintos recorridos" — el 68 % de sus
entradas son remuestreos del mismo arco en curso, no arcos diferentes. Así que el contenido
informativo real de `mean_velocities`, tal como está hoy, es más estrecho de lo que su nombre
sugiere: es una señal de "cómo ha ido el tráfico en los últimos 4-6 minutos", no un resumen
por arco.

**Matiz preciso sobre el efecto indirecto:** decir que dos episodios idénticos salvo por la
congestión producen features idénticas solo es cierto en la primera época afectada. Después,
la congestión sí cambia las features — pero **indirectamente**, moviendo dónde acaban los
vehículos y qué hora es, nunca como una lectura directa de "el tráfico ha estado más lento de
lo normal". La W aprendida no tiene ningún término que le permita aprender esa asociación,
aunque el dato para hacerlo ya se calcula en cada decisión.

**Por qué es más una oportunidad perdida que un bug de corrección:** decidir si vale la pena
conectar `mean_velocities` tal cual —una ventana temporal reciente, no un resumen por arco
distinto, ver B18— es una decisión de diseño experimental, no una corrección obligatoria — el
simulador en sí no está mal, solo desaprovecha una señal que su propia restricción de
observabilidad
parcial permitiría usar.

---

### B5 — Crash: `min()` sobre secuencia vacía

**Severidad: media.** `src/stdvrp/policies/monte_carlo.py:425-437`.

Dos umbrales descoordinados dejan una ventana de 40 minutos sin cubrir:
`_select_vehicle_possible_actions` desvía al depósito con `tau > 350` (línea 298), pero
`_classify_shortest_distance_clients` descarta vehículos con `tau > 310` (línea 429). Con
exactamente 1 cliente pendiente, todos los vehículos leyendo `vehicle_position == depot` y
`310 < tau <= 350`, la lista `distances` queda vacía y `min(distances)` lanza `ValueError`.

**Reproducido:**

```
tau= 305.0 -> action=[7, 0]
tau= 320.0 -> ValueError: min() arg is an empty sequence   <-- CRASH
tau= 340.0 -> ValueError: min() arg is an empty sequence   <-- CRASH
tau= 351.0 -> action=[0, 0]
```

Recuerda que `vehicle_position == depot` también es cierto para vehículos que salieron o
cruzaron el depósito (causa raíz común), así que la condición es más fácil de cumplir de lo
que parece. Más probable en instancias pequeñas o con `min_number_clients` bajo.

**Arreglo propuesto:** unificar el umbral y añadir fallback al depósito cuando `distances`
quede vacía, como ya hace la rama de 2 clientes vía `heapq.nsmallest`.

---

### B6 — Las features miden desde el nodo del que el vehículo salió

**Severidad: media.** `src/stdvrp/policies/feature_extraction.py:169,182`.

`vehicle_minutes` y `vehicle_length` se calculan con
`geometry.average_minutes_rows(state.vehicle_position)`. Para un vehículo a mitad de arco eso
parte de un origen que el simulador **no va a honrar**: `Model._reroute_for` lo obliga a
terminar el arco actual y re-rutear desde `fleet.next_node(vehicle)` (`model.py:395-403`).

**Cuantificado** sobre 14 077 decisiones con el vehículo genuinamente a mitad de arco
(semillas 0–59):

- la ETA que la feature cree **subestima** la llegada realizable en el **78.5 %** de los
  pares vehículo-cliente, con media **+4.03 min**;
- el 21.5 % restante es *pesimista* (hasta −8.3 min), porque cuando el camino más corto al
  cliente pasa por el nodo al que el vehículo ya se dirige, el origen obsoleto sobreestima;
- en mi propia medición complementaria, la fracción del arco actual ya recorrida pero
  contabilizada como pendiente tiene media **50.8 %** (mediana 51.3 %, p90 89.9 %), lo que
  equivale a **1.03 km de media** (máx. 5.99 km) contados como "por delante".

Afecta a la selección de candidatos y a todas las features de earliness, retraso y horas
extra sobre las que se ajusta la Q lineal.

---

### B7 — Un evento de congestión posterior trunca la expiración del anterior

**Severidad: media.** `src/stdvrp/congestion/generator.py:88,131`.

`congested_arcs` guarda un único par `[multiplicador, expiración]` por arco y **cada
escritura reemplaza los dos campos**. Dentro de una misma llamada a `generate()`, un evento
posterior puede dejar un arco *menos* congestionado y, sobre todo, terminar su congestión
*antes* que el evento aún activo que ya estaba anotado.

**Medido:** el defecto dominante es el campo de expiración, no el multiplicador. Por época,
unos **25–30 arcos** se vuelven silenciosamente más rápidos al añadir un evento, y
**~1400 entradas** (≈5 % de los ~3300 arcos congestionados por época) ven su congestión
terminar **hasta 90 minutos antes** de lo sorteado.

El alcance es *dentro de una llamada* a `generate()`, no entre épocas.

**Arreglo propuesto:** al escribir sobre un arco ya congestionado y aún activo, componer en
lugar de reemplazar: quedarse con el multiplicador más severo y con la expiración más tardía.

---

### B8 — La saturación del multiplicador anula el decaimiento por distancia

**Severidad: media.** `src/stdvrp/congestion/generator.py:120-123`.

El arreglo 7 de la fase 2 (ADR-0001) hizo que los multiplicadores de propagación saturaran en
`congestion_upper_bound`. Con los límites reales (`0.3` / `0.4` en
`experiments/chengdu/config.yaml`) eso deja **la tabla de amortiguación inerte**: un
multiplicador sorteado `p ~ U(0.3, 0.4)` satura en profundidad 1 para `p >= 0.332` (68 % de
los sorteos), en profundidad 2 para `p >= 0.312` (88 %) y en profundidad 3 **siempre**.

**Consecuencia medida:** por época de congestión, **más de la mitad de la red** queda a ≤ 40 %
de la velocidad libre, y **unos tres cuartos de esos arcos llevan el multiplicador idéntico
0.4** sin importar si están a 1, 2 o 3 saltos del epicentro. El factor 0.73 que el arreglo 7
resucitó no puede observarse nunca con esta configuración.

Es decir: la congestión no decae con la distancia. Es una decisión de modelado documentada,
pero su efecto bajo la configuración real seguramente no es el que se pretendía.

---

### B9 — `_reachable_nodes` es DFS, no BFS

**Severidad: media.** `src/stdvrp/congestion/generator.py:136-164`.

El docstring lo llama «BFS-by-recursion», pero la recursión es **en profundidad con un único
conjunto `visited`**: un nodo descubierto primero por una rama profunda conserva esa
profundidad, recibe el factor de amortiguación equivocado y, si su profundidad registrada
alcanza `max_depth`, deja de expandirse.

**Consecuencia medida:** **~15 % de los nodos que están genuinamente dentro de `max_depth`**
de un epicentro nunca se congestionan, y qué arcos alcanza la propagación depende en parte del
**orden de la tabla de arcos** en `successors`, no de la topología de la red.

Se ejecuta dos veces por evento disparado (una por extremo), ~64 eventos por época, ~8 épocas
por episodio.

**Matiz:** el efecto sobre el *factor de amortiguación* es casi invisible bajo la
configuración real, precisamente por la saturación de B8. Lo que sí queda mal es **qué arcos**
se congestionan.

---

### B10 — Feature 10 idénticamente cero; ningún bin cuenta ventanas ≥ 600

**Severidad: media.** `src/stdvrp/policies/feature_extraction.py:208-214,230`.

```python
counts_earliness = [0, 0, 0, 0]
if tau < 400: counts_earliness[0] = ...(earliness < 400)
if tau < 500: counts_earliness[1] = ...(400 <= earliness < 500)
if tau < 600: counts_earliness[2] = ...(500 <= earliness < 600)
# counts_earliness[3] nunca se asigna
general[7:11] = [count / self._number_clients for count in counts_earliness]
```

Tres consecuencias verificadas:

1. **`general[10]` es idénticamente cero** en todo instante y con toda ventana. Como el
   gradiente es `lr * err * X`, el peso `W[10]` nunca se actualiza: **el modelo efectivo es
   de 18 dimensiones, no de 19.** Los docstrings documentan otro peso muerto (el relleno
   `X[:,13]`) pero no este.
2. **Ningún bin de conteo cuenta a un cliente cuya ventana abre en el minuto 600 o más
   tarde** — con la configuración real, `Uniform[300, 720]`, eso es el **28.7 %** de la
   demanda.
3. Para `tau >= 600` todas las `general[7:11]` valen cero, y `general[11]` vale cero para
   `tau >= 580`: **cinco de las doce features generales no aportan información durante
   aproximadamente medio horizonte**, justo donde la presión del fin de jornada y las
   penalizaciones de retraso empiezan a morder. Dos estados con demanda pendiente
   radicalmente distinta colisionan exactamente en `tau = 610`.

**Corrección respecto a una versión previa de este informe:** la demanda posterior al minuto
600 **no** es invisible para todo el bloque de earliness — la feature 11
(`mean_earliness_diff`) sí la registra mientras `tau < 580`. Solo es invisible para los cuatro
bins de conteo.

---

### B11 — Asignación duplicada del mismo cliente

**Severidad: media.** `src/stdvrp/policies/monte_carlo.py:302-308`.

La rama de endgame (`len(clients_not_visited) < 3`) no filtra `forbidden_actions`, a
diferencia de la rama normal (línea 311). Dos vehículos reciben el mismo cliente.

**Reproducido:** `action=[7, 7]` con dos vehículos; `action=[7, 0, 7]` con tres. Sobre 60
episodios pequeños se contaron **734 transiciones** con una asignación duplicada.

El coste directo es capacidad de flota desperdiciada en el tramo final. Además abre la única
ruta de fin de transición que **no** hace `commit_transition()` (`model.py:505-514`): el
vehículo perdedor llega a un cliente ya servido y su coste queda dentro de `distance_cost`
pero fuera de `total_cost`.

**Matiz honesto:** esa segunda parte es *latente*. En 84 episodios instrumentados la rama sin
commit **nunca se alcanzó** (0 casos), porque el perdedor se re-rutea antes de llegar. La
identidad "total = suma de componentes" se mantuvo en todos los episodios medidos.

---

### B12 — `horizon_end_minute` no acota el episodio

**Severidad: media.** `src/stdvrp/simulation/model.py:76,334`.

El corte real es `EMERGENCY_HORIZON = 1150`, hardcodeado e independiente de la configuración.
`horizon_end_minute` solo actúa como umbral de horas extra (`shift_end_minute`).

**Reproducido:** con `horizon_end = 1400` los episodios terminan exactamente igual que con
`horizon_end = 780` — mismos tau finales, mismos costes, mismo número de decisiones.

Está documentado como quirk heredado en ADR-0001, pero es un desajuste entre lo que la
configuración promete y lo que el simulador hace, y es la causa raíz de B15. El normalizador
`time_left = (1150 - tau) / 850` (`feature_extraction.py:197`) repite el mismo literal, así
que las features tampoco respetan el horizonte configurado.

---

### B13 — El cap de 60 km/h censura la distribución sobre un átomo

**Severidad: baja-media.** `src/stdvrp/simulation/episode_velocities.py:143-145`.

```python
if speed < 1 and velocity > 1:
    velocity = 1
```

En arcos cuya media está justo por debajo de 1 km/min, el truncamiento no recorta la cola:
**la colapsa sobre un único átomo en 1.000 km/min**. En los arcos-minuto donde la media queda
a menos de ~1 desviación del cap, hasta el **49 %** de la distribución de velocidad se
concentra en ese punto y la media realizada cae **~4.3 %** por debajo de la pretendida.

**Medido:** 24 418 sorteos en 200 episodios; el **1.28 %** de los sorteos sub-1 resultan
censurados de media. El efecto agregado es pequeño; el problema es que la distribución
muestreada no es la que el modelo declara.

Hallazgo hermano (**severidad baja**): el suelo `if velocity <= 0: velocity = 0.001` produce un
cuasi-parón (0.002 km en 2 minutos en lugar de ~0.34 km). En el fixture nunca se dispara (0 de
24 418 sorteos); en la instancia real de Chengdu se espera en el **0.173 %** de los sorteos,
subiendo a ~18 % en el peor arco-minuto.

---

### B14 — `delay_clients = 0` con `delay_cost = 4205`

**Severidad: baja (reporte).** `src/stdvrp/simulation/cost_ledger.py:121-135`.

`charge_unserved_delays` y `charge_fleet_overtime` suman dinero sin incrementar
`late_clients` / `overtime_vehicles`. Está documentado como quirk heredado, pero significa que
los números publicados se contradicen.

**Reproducido:**

```
seed  tau_end  delay_cost  delay_clients
   4  1148.00     4205.00              0
   9  1148.00     1839.00              0
```

**Precisión:** el subconteo es exactamente el número de clientes abandonados que estaban
**retrasados** en el momento de terminar (los que el filtro de `model.py:583` llega a cobrar),
no todos los abandonados. Ocurrió en 9 de 180 episodios de evaluación.

---

### B15 — Horas extra negativas si el horizonte configurado supera 1148

**Severidad: latente (trampa de configuración).** `src/stdvrp/simulation/model.py:554-557`.

`terminate_state_passing_horizon` cobra `tau - shift_end_minute` **sin** el guardia
`tau > shift_end` que sí tiene `_vehicle_parks_at_depot` (línea 306). Como todo episodio se
termina a la fuerza en el reloj 1148, un `horizon_end_minute` mayor produce un *ingreso* por
horas extra.

**Reproducido** (semilla 4, un vehículo):

| `horizon_end_minute` | Fin | `overtime_cost` | Total |
|---|---|---|---|
| 780 | 1148 | 306.67 | 4997.76 |
| 1200 | 1148 | **−43.33** | 3551.59 |
| 1400 | 1148 | **−210.00** | 3049.82 |

**Ninguna configuración del repositorio está afectada** (todas usan 780), así que no hay
resultados publicados en riesgo. Pero `ExperimentConfig` solo valida
`0 <= horizon_start < horizon_end`, y los datos de velocidad cubren hasta el minuto 1198, así
que un horizonte de jornada completa es una elección plausible. Combinado con B3 (los clientes
cuya ventana vence después de 1148 no se cobran), **empujar el horizonte hacia fuera hace que
el objetivo parezca mejor por razones puramente contables**.

La suite de invariantes afirma `component_total >= 0`, pero solo prueba con
`horizon_end = 780`, así que no lo detecta.

---

### B16 — La cadencia de congestión usa módulo en coma flotante

**Severidad: latente (trampa de configuración).** `src/stdvrp/simulation/model.py:355`.

```python
return (self.state.tau_episode + 180 - 2) / 60 % self.hours_max_duration == 0
```

Es una comparación de igualdad en coma flotante. Para duraciones cuyo cociente `/60` no es
exactamente representable en binario, el sorteo se reduce a una o dos veces por episodio.

**Bajo toda configuración enviada, testeada o capturada en el repositorio** (30/45/60/90/120/
180/240) la puerta dispara a la cadencia pretendida, así que **nada de lo ya ejecutado está
mal**. Pero si se barre `max_congestion_duration` de 60 a 50 o 70, la congestión pasa de
12–17 sorteos por episodio a **uno**, mientras `_compute_event_probabilities` sigue calibrando
las probabilidades como si nada.

**Nota:** que subir la duración reduzca el número de sorteos **no** es un bug — la cadencia
*es* la duración por diseño (un sorteo cada `max_congestion_duration` minutos). Lo que está mal
es solo la aritmética flotante para valores no diádicos.

**Arreglo propuesto:** aritmética entera en minutos
(`(tau + 178) % max_congestion_duration == 0`), y considerar separar "cada cuánto se sortea" de
"cuánto dura un evento", que hoy son el mismo parámetro.

---

### B17 — El libro de congestión nunca purga

**Severidad: baja.** `src/stdvrp/simulation/episode_velocities.py:71`.

`congested_arcs` solo se limpia al terminar el episodio. Los eventos expirados se quedan
dentro, así que `any_congestion` queda permanentemente en `True` tras el primer sorteo y el
escaneo de expiraciones recorre la flota entera en cada iteración del bucle.

**Medido:** en el fixture el libro llega a 116 arcos, **los 116 ya expirados**. La semántica es
correcta (`sample()` comprueba `tau >= event[1]`), así que es coste de cálculo y ruido, no un
error físico.

---

### B18 — El docstring y el nombre del parámetro dicen "N arcos"; el diseño real es una ventana de las últimas N velocidades observadas

**Severidad: baja (imprecisión de nomenclatura, no defecto de comportamiento).**
`src/stdvrp/simulation/state.py:38`.

**El código hace lo correcto.** `begin_arc` añade una entrada por arco nuevo y `resample_arc`
añade una entrada **por época de decisión**, incluso sobre el mismo arco. Eso no es un error:
es una ventana deslizante de las últimas *N* velocidades observadas, sea cual sea su origen —
exactamente el diseño previsto, y no requiere ningún cambio.

**Lo único que desalinea es la redacción.** El docstring («velocidades observadas en los
últimos `n_arcs` arcos») y el propio nombre del parámetro (`n_arcs` en `State`,
`n_observed_arcs` en la configuración) sugieren que la ventana captura *N arcos distintos*,
cuando en realidad captura *N observaciones de velocidad en el tiempo*, que suelen ser
remuestreos repetidos del mismo arco.

**Medido, como dato descriptivo (no como evidencia de un fallo):** de 4928 inserciones, 1586
fueron por arco nuevo y 3342 por remuestreo del mismo arco — con `n_observed_arcs: 3` la
ventana cubre en la práctica unos 4–6 minutos recientes, no necesariamente tres arcos
distintos. Vale la pena saberlo si algún día se conecta `mean_velocities` (B4): lo que ese
vector resumiría es "cómo ha ido el tráfico en los últimos minutos", una señal temporal de
recencia — razonable e incluso preferible como proxy de congestión local, ya que un evento de
congestión dura decenas de minutos y una ventana temporal lo captura mejor que una ventana por
arcos distintos lo haría. El único ajuste pendiente, si acaso, es de documentación: precisar
el docstring y considerar renombrar el parámetro para que no sugiera "arcos" donde el diseño
es "observaciones recientes".

---

### B19 — El endpoint derecho de la ventana de std no es la primera observación

**Severidad: baja.** `src/stdvrp/traffic/travel_time_model.py:285,292,299`.

El docstring justifica los endpoints desplazados (418/542, 658/842, 958/1082) como «los
últimos/primeros minutos *observados* alrededor de cada hueco de datos». En el archivo real,
los izquierdos (418/658/958) sí lo son, pero los primeros minutos observados tras el hueco son
**540/840/1080**, no 542/842/1082 — el ancla derecha está una observación más allá del borde.

**Medido:** la std almacenada para un minuto dentro de la ventana 420–540 difiere en una
**mediana del 10 %** (p90 ~25 %) según qué ancla derecha se use, sobre una std media de 0.134
km/min. Es pequeño y sin sesgo direccional; lo que falla es la justificación documentada, no la
magnitud. El desplazamiento izquierdo es además inerte, porque el hueco de datos lo hace
equivalente.

---

### B20 — Una acción debe ser ejecutable

**Severidad: alta (crash).** `src/stdvrp/simulation/model.py`, rama at-a-node de
`_reroute_for`.

**Corrección a esta misma revisión.** La primera versión de este informe clasificó la búsqueda de
arcos auto-lazo (`path_between(n, n)`) como hipótesis descartada — *"0 búsquedas de arcos
auto-lazo en 120 episodios"* — bajo la premisa de que `_reroute_for` reemplaza la ruta inicial
antes del primer `begin_arc` y por tanto nunca la consulta. **La medición era correcta; la
inferencia, no.** No es que la búsqueda sea inalcanzable — es que alcanzarla depende del *action
set* de la Policy evaluada, y `MonteCarloPolicy` (la única evaluada entonces) construye el suyo de
forma que nunca nombra el nodo en el que el vehículo ya está. Encontrado por el ticket 08 de
`neural-policy` entrenando el transformer sobre Chengdu, cuyo action set (ADR-0007: "todo cliente
pendiente no reclamado, más el depósito") sí puede nombrarlo. La cifra 6.8 % de rutas cacheadas que
usan el depósito como nodo interior (la causa raíz de B1, más arriba) es un hallazgo relacionado
pero distinto: `path(0→13)` en la reproducción de abajo pasa por `0→4→16→13`, así que el depósito
se vuelve interior porque el vehículo *ya iba hacia allí* y fue redirigido a mitad de arco, no
porque una ruta cacheada lo contenga.

```python
elif (
    action[vehicle] == self.depot and self.state.vehicle_position[vehicle] == self.depot
):
    ...
elif fleet.destination[vehicle] != action[vehicle] and fleet.is_travelling(vehicle):
    ...
    if fleet.departure_tau[vehicle] == self.state.tau_episode:
        fleet.route[vehicle] = list(
            self.shortest_path_cache.path_between(last_node_reached, vehicle_destination).nodes
        )
        self.begin_arc(vehicle)
```

**El hueco.** ADR-0005 define `vehicle_standing = False` en el instante exacto en que `begin_arc`
lanza al vehículo — el mismo instante en que `departure_tau == tau` (cero progreso de arco). Ese
instante el simulador dice a la vez "en el nodo" (por lo que corre la rama at-a-node) y "no
parado" (por lo que el guardia `is_parked_at_depot`, que exige `standing`, no se dispara). Si la
decisión de ese instante nombra el nodo en el que el vehículo ya está, la rama cae al re-ruteo de
más abajo y pide `path_between(n, n)`. `all_shortest_paths.csv` contiene las 45 filas de
autobucle (`0,0,0,0.0,0.0`), así que la búsqueda **no lanza excepción** — devuelve una ruta
bien formada de un nodo y longitud cero. `FleetRoutes.current_arc` lee entonces `route[1]` sobre
esa ruta de un elemento y muere con `IndexError`.

Hoy solo el **depósito** es alcanzable así, porque es el único nodo que está siempre en el action
set (ADR-0007). El caso **cliente** — un vehículo parado sobre un Client pendiente que cruzó sin
servir — es alcanzable por construcción y falla de forma idéntica.

**Disparador.** Las cuatro condiciones a la vez:

1. el vehículo está *en* un nodo con progreso de arco cero (`departure_tau == tau`);
2. `vehicle_standing` es `False` — `begin_arc` ya lo lanzó, así que el guardia
   `is_parked_at_depot` del ticket 04 no se dispara;
3. la decisión nombra ese mismo nodo;
4. `fleet.destination != action` (si no, no hay re-ruteo).

**Reproducción.** `tests/fixtures/chengdu_mini`, semilla **1131**, red sin entrenar, vehículo 0:

| τ | evento |
|---|---|
| 302.0 | rumbo al Client 10, a mitad del arco `0→4`. La decisión cambia a **depósito**. Empalme a mitad de arco `[último] + path(4→0)` → ruta `[0, 4, 0]` |
| 303.28 | cruza el nodo 4 → ruta `[4, 0]` |
| 308.0 | la decisión cambia a **Client 13**. Empalme `[4] + path(0→13)` → ruta `[4, 0, 4, 16, 13]` — **el depósito queda ahora como waypoint interior** |
| 308.7466 | llega al depósito como waypoint; `begin_arc` fija `departure_tau = τ`, `standing = False`. **En el mismo instante**, el vehículo 2 aparca en el depósito → termina la transición |
| 308.7466 | `_reroute_for`: rama at-a-node, `last_node_reached = 0`, acción = `0` → `path_between(0,0)` → `[0.0]` → 💥 |

La coincidencia del último paso no es azar: los vehículos 0 y 2 iban en **lockstep** —
lanzados juntos, con rutas más cortas que comparten prefijo, y `EpisodeVelocities` memoiza la
velocidad por (arco, minuto) — así que viajan de forma idéntica y llegan juntos. La llegada de un
vehículo termina rutinariamente la transición en el instante exacto en que otro está cruzando un
nodo.

**Medido:**

| medición | muestra | resultado |
|---|---|---|
| Crashes, transformer (sin entrenar, episodios de entrenamiento ε-greedy) | 80 episodios | **3** (semillas 1116, 1131, 1134) — todas en τ = 308.7466 |
| Crash / precondición, `MonteCarloPolicy` lineal | 600 episodios (200 semillas × flotas 1/3/6) | **0 / 0** — la hipótesis descartada original, remedida a 5× la muestra |
| Rutas degeneradas, política lineal | 600 episodios | **0** |
| "Depósito registrado, no parado" (el estado que un guardia Policy-side tendría que excluir) | 2229 decisiones del transformer | 15 (0.67 %) — **todas** genuinamente a mitad de arco, ninguna con progreso cero |
| Crashes, política uniforme-aleatoria de action set completo | 300 episodios | **0** |
| Crashes, política adversarial hand-built "lockstep flip-flop" | 120 episodios | **0** |

**Las dos últimas filas son el hallazgo importante.** El disparador necesita *comportamiento
correlacionado de la flota* — vehículos lanzados juntos, en rutas que comparten prefijo,
sincronizados por el memo per-(arco, minuto). La aleatoriedad uniforme **destruye** esa
correlación; una Policy greedy real la produce gratis. Por eso este defecto es estructuralmente
invisible al fuzzing, y ampliar la suite de invariantes de la forma obvia tampoco lo habría
cazado — el catch real es un test unitario hand-built y una regresión end-to-end sobre la semilla
observada, no un barrido de política aleatoria.

**El arreglo.** ADR-0008: `FleetRoutes.is_at_node(vehicle, tau)` nombra la presencia
posicional (`departure_tau >= tau`) que la rama de aparcar de `_reroute_for` necesita en
lugar de `is_parked_at_depot`; `vehicle_standing` se fija `True` al aparcar por esta vía, para no
romper los otros siete sitios que leen `is_parked_at_depot`. El caso cliente se cierra del lado de
la Policy: `TransformerMonteCarloPolicy._sweep` excluye un Client pendiente igual a
`last_node_reached[v]`, en la rama greedy y en la de exploración ε — el depósito nunca se filtra,
porque conserva sus dos significados (aparcar / viajar). `monte_carlo.py` queda intacto: su propio
candidate set ya excluye por construcción el nodo actual del vehículo. Consecuencia para
`neural-policy`: el action set del transformer cambia, así que toda cifra de Gate A recogida antes
de este arreglo queda invalidada.

---

## Hipótesis descartadas

Se probaron y **no** se sostuvieron. Se listan para que no se reinvestiguen:

- **Terminación prematura con un vehículo en marcha.** `_every_vehicle_home_and_no_clients_left`
  usa `vehicle_position`, así que en teoría podría declarar completo un episodio con un vehículo
  en la carretera. **0 casos en 75 episodios.**
- **Subconteo de horas extra en la terminación por horizonte.** **0 discrepancias en 75
  episodios.**
- **`arc_distance_travelled` superando la longitud del arco** (que daría llegadas en el pasado).
  **0 casos en 675 episodios.**
- **Doble cobro de distancia** entre `resample_arc` y `advance_fleet_to`. No existe.
- **Rotura de la identidad "total = suma de componentes"** por la transición sin commit.
  Latente pero no alcanzada (ver B11).
- **Velocidades o std NaN** por el `dropna()` heredado. Los datos reales no tienen huecos en el
  horizonte; la rama nunca se dispara.
- **Tabla de flotas del test final "desacoplada de la demanda"** — **refutado como bug**. El
  test final corre deliberadamente una flota fija por semilla (la tabla heredada) mientras los
  bloques de evaluación durante el entrenamiento usan la flota dimensionada por la demanda. No
  es un error de simulación, pero **sí es una advertencia de interpretación**: las dos cifras
  principales de una ejecución no son directamente comparables, porque agregan sobre un rango de
  3.2× en carga de trabajo por vehículo y usan dos regímenes distintos de dimensionado de flota.

---

## Recomendaciones, por orden

1. **La cadena del depósito (B1a + B1b), las dos capas.** Empezar por **B1a**: es la que
   vacía la flota, y arreglarla también cierra la vía por la que hoy se alcanza B1b. Arreglar
   B1b además, porque tiene disparadores propios que no dependen del literal 350 y porque es
   lo que hace que la contabilidad cuadre. Arreglar solo B1b hace que el vehículo pague su
   vuelta a casa pero no lo devuelve al servicio. Añadir un test de invariante: *ningún
   vehículo pasa a `PARKED` mientras `departure_tau < tau < arrival_tau`*.
2. **B3 como decisión de modelado, no como parche.** Hasta que se decida cómo penalizar la
   demanda abandonada, cualquier media de coste que mezcle episodios terminados por horizonte con
   episodios all-back está promediando dos funciones de coste distintas. Documentar en un ADR.
3. **B5** es un arreglo de dos líneas que elimina un crash.
4. **B6** es un defecto real por sí solo: la ETA de los candidatos se mide desde un origen que
   el simulador no va a honrar (el nodo del que el vehículo salió, no por dónde va). Merece su
   propio ADR, independiente de B4.
5. **B7, B8, B9** cambian qué arcos se congestionan y por cuánto tiempo. Afectan a la
   interpretación de cualquier estudio sobre congestión ya ejecutado.
6. **B10** implica que uno de los 19 pesos nunca se entrena y que medio horizonte se ve con
   cinco features menos.
7. **B4, B11, B12, B13, B14, B17, B18, B19** son deuda o mejoras opcionales, sin urgencia: B4 en
   particular no exige ningún cambio para que el simulador sea correcto — es una señal ya
   calculada (`mean_velocities`) que se podría conectar si se quiere que la política reaccione
   al tráfico recién experimentado, sabiendo que lo que aportaría es una ventana temporal
   reciente (B18), no un resumen por arco distinto. B18 en sí no requiere ningún cambio de
   comportamiento, solo — si acaso — precisar el docstring.
8. **B15 y B16** son trampas de configuración: no afectan a nada ya ejecutado, pero conviene
   añadir validación (`horizon_end_minute <= EMERGENCY_HORIZON`) y aritmética entera antes del
   próximo barrido de parámetros.

Cualquier arreglo de B1a, B1b, B3, B6, B7, B8, B9 o B10 cambia los resultados de episodio y exige
re-baseline del golden master (`scripts/rebaseline_golden_master.py`), igual que hicieron los
arreglos de la fase 2 del ticket 12.

---

## Reproducción

Los sondeos instrumentan `Model` por subclase y se ejecutan con `uv run python <script>` desde
la raíz del repo, sobre `tests/fixtures/chengdu_mini`. El sondeo clave (la cadena del depósito) es:

```python
class Probe(Model):
    def _reroute_for(self, action):
        watch = [v for v in range(len(action))
                 if action[v] == self.depot
                 and self.state.vehicle_position[v] == self.depot
                 and self.fleet.departure_tau[v] < self.state.tau_episode
                 and self.fleet.arrival_tau[v] > self.state.tau_episode]
        super()._reroute_for(action)
        for v in watch:
            if self.fleet.arrival_tau[v] == PARKED:
                print(f"tau={self.state.tau_episode}: vehículo {v} retirado a mitad de arco")
```

Sustituyendo `stdvrp.simulation.episode.Model` por esa subclase y corriendo
`run_evaluation_episode` con `vehicle_count=1` sobre las semillas 0–19 se reproducen 8 casos.
