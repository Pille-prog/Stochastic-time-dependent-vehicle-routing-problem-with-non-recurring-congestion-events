# Ruta de lectura: redes neuronales y transformers, para leer `network.py`

> **Estado:** guía de lectura personal, no una nota de investigación ni una decisión.
> No genera ADR ni afecta a ninguna decisión del effort `neural-policy`.
> **Alcance:** los fundamentos de deep learning que `src/stdvrp/policies/network.py` y
> `src/stdvrp/policies/transformer_policy.py` dan por sabidos. **Asume Reinforcement
> Learning conocido** — retornos Monte Carlo, `Q(s,a)`, ε-greedy, aproximación de
> función. Nada de eso se cubre aquí.
> **Fecha:** 2026-07-31, sobre la rama `FEATURE_DEV_total_code_refactorization`
> (tickets 01-07 del effort resueltos, 08 en curso).
> **Anclas de línea:** verificadas sobre el árbol de trabajo del 2026-07-31. Van a
> derivar. Trate los nombres de símbolo como autoritativos y los números como pistas.
> **Complementa** a [`rl-methodology-for-stdvrp.md`](./rl-methodology-for-stdvrp.md),
> que cubre el *estimador* (por qué MC, por qué VFA neuronal, por qué no AM/POMO).
> Esta guía cubre el *aproximador*: qué es exactamente la red que ese documento
> recomienda construir.

---

## Qué hueco cubre esta guía, y cuál no

`transformer_policy.py` es RL casi puro: `_backward_returns` (línea 392) es every-visit
Monte Carlo literal, `_sweep` (línea 214) es ε-greedy sobre un `argmin`, y
`_already_acquired_cost` (línea 407) es una resta de coste hundido. Si RL ya se entiende,
ese archivo se lee sin ayuda.

El hueco está entero en **`network.py`**, y en particular en su docstring de módulo
(líneas 1-154), que argumenta cuatro cosas no triviales:

| Afirmación del docstring | Líneas | Requiere entender |
|---|---|---|
| Por qué el par "arco" tiene su propia vía y no entra al transformer | 23-48 | Embeddings, atención, invarianza a `m` |
| El warm start miope: `Q == minutes_from_vehicle` **exacto** en init | 51-106 | Pre-LN, conexiones residuales, composición de capas |
| Por qué las filas hermanas de la 0 llevan pesos aleatorios y no cero | 108-132 | **Regla de la cadena, backprop** |
| Por qué basta `init_rng` para bit-identidad | 134-153 | Xavier, de dónde salen los pesos |

La tercera es el argumento más sutil del repo y es cálculo puro, no arquitectura.

---

## El libro base

**Simon J. D. Prince, *Understanding Deep Learning*, MIT Press.**
PDF gratuito y oficial: <https://udlbook.github.io/udlbook/>

De los 21 capítulos, sólo seis tocan este código:

| Cap. | Contenido | Uso |
|---|---|---|
| 3-4 | Redes shallow y deep | Qué es realmente `nn.Linear` + `torch.relu` |
| 5 | Funciones de pérdida | Ojear — `functional.huber_loss` (`transformer_policy.py:319`) |
| 6 | Ajuste: SGD, momentum, **Adam** | El `optimizer` inyectado en `TransformerMonteCarloPolicy` |
| **7** | **Gradientes e inicialización** | **El capítulo central de esta guía** |
| 11 | Residuales y normalización | *Es* `norm_first=True` (`network.py:269`) |
| 12 | Transformers | `nn.TransformerEncoderLayer` |

Saltar 8-10 (performance, regularización, convnets) y 13-21. No aparecen en el código.

---

## Las cuatro fases

Cada fase cierra con un **checkpoint contra el código de este repo**. El criterio es
simple: si el checkpoint no se puede responder sin abrir el docstring, la fase no está
cerrada. Presupuesto realista: 4-5 semanas a ~1 h/día.

### Fase 0 — Autograd (~1 semana)

Va primero porque el argumento más importante del repo es regla de la cadena.

- **Karpathy, *Neural Networks: Zero to Hero*, vídeo 1 — micrograd**
  <https://karpathy.ai/zero-to-hero.html> (2 h 15). Construye un motor de
  diferenciación automática en ~100 líneas. **Escribirlo, no verlo.**
- Prince cap. 7.

> **Checkpoint.** Derivar a mano por qué
> `∂loss/∂layer2.weight[0, row] = hidden[row] · ∂loss/∂Q`,
> y por qué si `hidden[row] ≡ 0` para *toda* entrada ese peso queda congelado
> **permanentemente**, no sólo un paso. Es la "deadlock note" de `network.py:108-132`,
> el motivo por el que `QHead._init_weights` (línea 366) deja las filas 1..hidden-1 con
> Xavier en vez de ponerlas a cero. El test que lo pina es
> `TestQHeadBackgroundUnitsAreTrainable::test_layer2_background_columns_receive_gradient`
> (`tests/unit/test_network.py:276`).

### Fase 1 — MLP, no-linealidad, inicialización (~1 semana)

- Prince cap. 3, 4, 6.
- **Glorot & Bengio (2010), *Understanding the difficulty of training deep feedforward
  neural networks***
  <https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf> — el origen de
  `_xavier_uniform_` (`network.py:198`).

> **Checkpoint.** (a) ¿De dónde sale `sqrt(6.0 / (fan_in + fan_out))` en
> `network.py:210`? (b) ¿Por qué inicializar **todos** los bias a cero "no cuesta nada"
> (docstring, líneas 139-143) mientras que inicializar todos los *pesos* a cero sí lo
> costaría? Son la misma pregunta vista desde dos lados. (c) El docstring afirma que el
> `ReLU` de `QHead.forward` (línea 399) "nunca recorta" el valor del warm start —
> ¿por qué?

### Fase 2 — Atención (~1 semana)

- **Karpathy, mismo curso, vídeo 7 — *Let's build GPT*** (1 h 56). Construye
  self-attention desde cero.
- Prince cap. 12.
- **Peter Bloem, *Transformers from scratch*** — <https://peterbloem.nl/blog/transformers>
  (el mejor texto en prosa sobre el tema; matemáticamente honesto).

> **Checkpoint.** `TokenEncoder` **no tiene positional encoding** — sólo
> `type_embedding` (`network.py:261`, aplicado en 322-324). (a) ¿Por qué eso es correcto
> y no un bug? Pista: qué debería pasar si se permuta el orden de `clients_not_visited`.
> (b) Entonces, ¿qué hace `type_embedding` que un positional encoding no haría?
> (c) ¿Qué se rompería si se añadiera un positional encoding sinusoidal estándar?

### Fase 3 — El bloque completo y las decisiones de init (~1 semana)

- **Phuong & Hutter, *Formal Algorithms for Transformers*** —
  <https://arxiv.org/pdf/2207.09238> (16 páginas, pseudocódigo, cero metáforas).
- **The Annotated Transformer** (Harvard NLP) —
  <https://nlp.seas.harvard.edu/annotated-transformer/> (PyTorch línea por línea).
- **Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*** —
  <https://arxiv.org/abs/2002.04745>. Pre-LN vs post-LN: por qué `norm_first=True`.
- **Goyal et al. (2017), §5.1 (*zero-γ*)** — <https://arxiv.org/abs/1706.02677>.
  El precedente estándar de anular la proyección final de un bloque residual, que el
  docstring cita como analogía (líneas 120-124).
- Opcional, mismo mecanismo llevado al extremo: **Fixup Initialization**,
  <https://arxiv.org/abs/1901.09321>.

> **Checkpoint (el importante).** Partiendo de `norm_first=True`, escribir las dos líneas
> que computa una `TransformerEncoderLayer`. Después: (a) ¿por qué poner a cero
> `out_proj.weight` **y** `out_proj.bias` (`network.py:301-302`) da identidad **exacta**,
> pero poner a cero sólo el weight **no**? (b) ¿Por qué no hace falta tocar `in_proj`
> ni `linear1`? (c) ¿Por qué `dropout=0.0` (línea 267) es un requisito de determinismo y
> no una decisión de regularización? El test es
> `TestIdentityAtInit::test_transformer_is_identity_for_arbitrary_input`
> (`tests/unit/test_network.py:259`).

### Fase 4 — PyTorch como herramienta (~3 días, en paralelo a las demás)

No es teoría, es artesanía: shapes, broadcasting, `unsqueeze`/`expand`/`cat`,
`nn.Module`, `torch.no_grad`, y el ciclo `zero_grad → backward → step`
(`transformer_policy.py:313-322`).

> **Checkpoint.** Anotar con su shape **cada línea** de `TokenEncoder.forward`
> (`network.py:310-338`). Las tres que cuestan:
> (a) `arc_pairs = torch.stack([minutes, lengths], dim=-1)` (línea 320) — ¿por qué
> `dim=-1` y no `dim=0`?
> (b) `client_context.unsqueeze(1).expand(-1, number_vehicles, -1)` (línea 335) — ¿qué
> hace `expand` que `repeat` no, y por qué importa aquí?
> (c) ¿Por qué `Embeddings.clients` acaba siendo `[n_pending, m, 2*d_model]` y no
> `[n_pending, 2*d_model]`? La respuesta está en el docstring (líneas 23-48) y es la
> razón de que el conteo de parámetros sea independiente de `m`.
> El test que pina las shapes es `TestShapes::test_embeddings_shapes`
> (`tests/unit/test_network.py:241`).

---

## El ejercicio final: romper el warm start a propósito

Después de la Fase 3. Es la forma más rápida de convertir el docstring de
`network.py:51-132` en conocimiento propio en vez de texto aceptado.

Para cada peso hand-set de abajo: **predecir primero qué test falla, después
comentar la línea y correr `pytest tests/unit/test_network.py`.**

| Línea a romper | Qué hace | Predicción |
|---|---|---|
| `network.py:288` — `arc_embed.weight[0,:] = [1.0, 0.0]` | Reconstruye `minutes` en la dim 0 | ? |
| `network.py:376` — `layer1.weight[0, _arc_dim0_index] = 1.0` | La unidad limpia de `QHead` | ? |
| `network.py:375` — `layer1.weight[0,:] = 0.0` (dejarla Xavier) | Aísla la fila 0 | ? |
| `network.py:382` — `layer2.weight[0,0] = 1.0` | `layer2` lee sólo esa unidad | ? |
| `network.py:301-302` — el zero de `out_proj` | Identidad en init | ? |

La última fila es la interesante y conviene pensarla antes de ejecutarla: el warm start
lee `arc_embed`, y `arc` **no pasa por el transformer** (línea 333, fuera de la llamada
de la 327). ¿Se sigue de ahí que `TestWarmStart` sobrevive a romper la identidad? Si la
predicción y el resultado coinciden, la Fase 3 está cerrada de verdad — y de paso queda
claro que *identidad-en-init* y *warm start miope* son dos propiedades independientes que
el docstring presenta juntas.

---

## Bibliografía

**Libro base**

- Prince, S. J. D. (2023/2024). *Understanding Deep Learning*. MIT Press.
  PDF libre: <https://udlbook.github.io/udlbook/> · Editor:
  <https://mitpress.mit.edu/9780262048644/understanding-deep-learning/>

**Cursos**

- Karpathy, A. *Neural Networks: Zero to Hero*. <https://karpathy.ai/zero-to-hero.html>
  · Notebooks: <https://github.com/karpathy/nn-zero-to-hero>

**Transformers**

- Phuong, M. & Hutter, M. (2022). *Formal Algorithms for Transformers*.
  arXiv:2207.09238 — <https://arxiv.org/abs/2207.09238>
- Vaswani et al. (2017). *Attention Is All You Need*. arXiv:1706.03762 —
  <https://arxiv.org/abs/1706.03762> *(histórico; peor como explicación que los dos
  anteriores)*
- Rush, A. et al. *The Annotated Transformer*, Harvard NLP —
  <https://nlp.seas.harvard.edu/annotated-transformer/>
- Bloem, P. *Transformers from scratch* — <https://peterbloem.nl/blog/transformers>
- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture*.
  arXiv:2002.04745 — <https://arxiv.org/abs/2002.04745>
- Lee et al. (2019). *Set Transformer*. arXiv:1810.00825 —
  <https://arxiv.org/abs/1810.00825> *(por qué un conjunto de clientes no necesita
  positional encoding)*

**Inicialización y optimización**

- Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep
  feedforward neural networks*. AISTATS —
  <https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>
- Goyal et al. (2017). *Accurate, Large Minibatch SGD*, §5.1. arXiv:1706.02677 —
  <https://arxiv.org/abs/1706.02677>
- Zhang, Dauphin & Ma (2019). *Fixup Initialization*. arXiv:1901.09321 —
  <https://arxiv.org/abs/1901.09321>
- Kingma, D. & Ba, J. (2014). *Adam*. arXiv:1412.6980 —
  <https://arxiv.org/abs/1412.6980>

**Contexto del dominio** *(ya cubierto en la nota hermana, se listan por cercanía)*

- Kool, van Hoof & Welling (2019). *Attention, Learn to Solve Routing Problems!*
  arXiv:1803.08475 — <https://arxiv.org/abs/1803.08475>
- Chen, Ulmer & Thomas. arXiv:1910.11901 — <https://arxiv.org/abs/1910.11901>

---

## Notas cruzadas

- [`rl-methodology-for-stdvrp.md`](./rl-methodology-for-stdvrp.md) — el *estimador*:
  por qué Monte Carlo, por qué VFA neuronal (recomendación #5), por qué **no** AM/POMO.
  Esta guía es el prerrequisito técnico para leer su §7 con criterio propio.
- [`.scratch/neural-policy/spec.md`](../../.scratch/neural-policy/spec.md) — las 14
  decisiones del effort. Las decisiones 1, 5, 6 y 9 se leen mejor después de la Fase 3.
- [`docs/adr/0006-what-the-policy-is-allowed-to-see.md`](../adr/0006-what-the-policy-is-allowed-to-see.md)
  — la regla de observabilidad, que es *ortogonal* a todo lo de esta guía: restringe qué
  entra al tokenizer, no cómo la red lo procesa.
