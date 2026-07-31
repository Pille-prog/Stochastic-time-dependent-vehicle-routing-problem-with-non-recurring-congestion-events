"""network: the transformer encoder over ticket-04 tokens, the Q head that scores
(vehicle, Client) pairs, and the myopic warm start (ticket 05, neural-policy).

torch is an optional extra (ticket 03): this module imports it at module scope
because its whole reason to exist is the network, unlike ``torch_support.py``'s
deferred-import boundary. It must therefore never be imported from
``stdvrp.policies.__init__`` or any module reachable from package-import time —
the same discipline :mod:`~stdvrp.policies.tokenizer` follows by *not* needing
torch at all. Callers import it explicitly:
``from stdvrp.policies.network import TokenEncoder, QHead``.

## Shape (spec.md, ticket 05)

``TokenEncoder`` runs once per decision epoch over the concatenated
``{client, vehicle, global}`` token set (a learned type embedding per kind),
producing per-token embeddings. ``QHead`` then runs once per vehicle (the
"sweep") over the client embeddings, scoring every pending Client — cheap,
because the encoder pass is shared. Both classes take an **injected**
``init_rng: np.random.Generator`` (never a global — same discipline ticket 13
of ``generic-stdvrp-refactor`` established for the Episode streams) and use it
for every learned weight, so a fixed seed reproduces bit-identical parameters.

## Why the per-(client, vehicle) "arc" fact gets its own pathway

Ticket 04's ``client_tokens`` row is ``3 + 2*m`` wide (``m`` = vehicle count):
three base facts (``tw_start``, ``tw_end``, ``tw_end - tau``) plus **two
raw facts per vehicle** (``minutes_from_vehicle[v]``, ``path_length_from_vehicle[v]``).
``m`` varies across configs and even across ``test_seeds`` within one trained
network's evaluation sweep (``ExperimentConfig.test_vehicle_counts``), so a
single ``nn.Linear(3 + 2*m, d_model)`` over the whole row is not viable — its
weight shape would have to change with ``m``. Instead, the per-vehicle "arc"
pair ``[minutes_from_vehicle[v], path_length_from_vehicle[v]]`` is embedded by
a small ``nn.Linear(2, d_model)`` applied identically to every ``(client,
vehicle)`` pair (weight-shared across ``v``, so parameter count is independent
of ``m``) — an ``m``-agnostic embedding by construction, not a workaround
adopted only for the warm start below. The client's three base facts go
through a separate, ordinary fixed-width embedding and the shared
self-attention transformer; the arc embedding bypasses the transformer
entirely (it is cheap — ``O(n_pending * m)`` linear ops, not attention — and
keeping it un-mixed by cross-token attention is exactly what makes the warm
start below exact rather than approximate).

``Embeddings.clients`` therefore has shape ``[n_pending, m, 2*d_model]``: for
each client, one row per vehicle, formed by concatenating that client's
(vehicle-independent) transformer-refined context embedding with the
(client, vehicle)-specific arc embedding. ``QHead`` takes one vehicle's slice,
``embeddings.clients[:, v, :]``, alongside ``embeddings.vehicles[v]`` — the
per-vehicle sweep (ticket 06) does the slicing, not the head.

## The warm start (this module's load-bearing part)

F2 (``docs/research/rl-methodology-for-stdvrp.md``) recommends a myopic warm
start: initialise so ``Q(i, j) ≈ minutes_from_vehicle_i_to_j`` rather than
leaving a ~150-Client argmin to an untrained network with no prior (the
farthest Client would be as likely a greedy choice as the nearest). This is an
**initialization, not a feature**: ``minutes_from_vehicle`` is not passed to
``QHead`` as an extra argument (its signature stays exactly
``(vehicle_embedding, client_embeddings, claimed)``); instead, specific weights
of the *ordinary, otherwise-trainable* layers below are hand-set so that the
network's output happens to equal that value at initialisation. Every one of
those weights remains a normal ``nn.Parameter`` — nothing here is a permanent,
non-trainable skip connection, and gradient descent is free to move every one
of them once training starts.

The construction, precisely:

1. **Every ``nn.TransformerEncoderLayer`` is an exact identity map at
   init.** With ``norm_first=True``, a layer computes
   ``x = x + attn(norm1(x))`` then ``x = x + ffn(norm2(x))``. Zero-initialising
   ``self_attn.out_proj`` and the FFN's ``linear2`` (weight **and** bias) makes
   ``attn(...)`` and ``ffn(...)`` evaluate to an exact-zero tensor *regardless*
   of what ``norm1``/``norm2``/``in_proj``/``linear1`` computed upstream (a
   ``Linear`` with zero weight and zero bias is a constant-zero function of any
   finite input) — so each residual sum degenerates to ``x + 0 = x``, bit for
   bit. Verified empirically before writing this module (a
   ``TransformerEncoder(num_layers=3)`` built this way returns its input
   unchanged, exactly). ``dropout=0.0`` throughout — see "Determinism" below.
   Consequently ``client_context`` (a client token's post-encoder embedding)
   equals its **pre-encoder linear embedding** at init: a plain, transformer-
   untouched function of the client's three base facts.
2. **The arc embedding's dimension 0 reconstructs ``minutes_from_vehicle``
   exactly.** ``arc_embed: nn.Linear(2, d_model)``'s row 0 is hand-set to
   ``[1.0, 0.0]`` with bias ``0.0`` — reads the ``minutes`` input, ignores
   ``path_length`` — so ``arc_embed(pair)[0] == minutes_from_vehicle`` exactly
   (already ``>= 0``, a travel time, so the head's ``ReLU`` below never clips
   it). Every other output dimension of ``arc_embed`` (and every other
   weight in this module) is Xavier-uniform from ``init_rng``, giving the
   network real capacity to learn beyond the warm start.
3. **``QHead``'s first layer has one "clean" row.** Row 0 of
   ``layer1: nn.Linear(3*d_model + 1, hidden)`` is zeroed except for a single
   ``1.0`` at the column reading ``arc_embed``'s dimension 0 (global index
   ``2*d_model`` within the concatenated ``[vehicle | client_context | arc |
   claimed]`` input — the layout ``TokenEncoder`` builds ``Embeddings.clients``
   in). Bias 0. So ``hidden[0] == ReLU(minutes_from_vehicle) ==
   minutes_from_vehicle`` exactly, uncontaminated by ``vehicle_embedding``,
   ``client_context`` or ``claimed`` (all zero-weighted on that row). ``claimed``
   is therefore init-inert (by design — the warm start must not depend on it),
   but not a dead argument: rows 1..hidden-1 are ordinary Xavier-random and do
   read it, so it starts affecting ``Q`` as soon as training moves ``layer2``'s
   matching background columns off zero (see the deadlock note below for why
   row 0's siblings must *not* also be zeroed).
4. **``QHead``'s second layer reads only that one hidden unit.**
   ``layer2: nn.Linear(hidden, 1)`` has column 0 set to ``1.0``, every other
   column and the bias at ``0.0``. So ``Q == hidden[0] ==
   minutes_from_vehicle_i_to_j`` exactly at construction, for every vehicle,
   every client, every state — not a statistical approximation.

**Why row 0's siblings (1..hidden-1) get real random weights, not zero:** an
earlier draft zeroed the *entire* first layer except row 0, reasoning that
those "background" units contribute nothing at init and can stay dormant. That
is a genuine dead end, not a transient one: with ``layer1.weight[row, :] == 0``
identically, ``hidden[row]`` is exactly ``0`` for *every* input, so
``d(loss)/d(layer2.weight[0, row]) = hidden[row] * d(loss)/d(Q) == 0``
*always* — ``layer2``'s background columns never move, so
``d(loss)/d(hidden[row])`` (which routes back through ``layer2.weight[0,
row]``) never moves either, and ``layer1``'s background rows are frozen at
zero forever. Giving the background rows of ``layer1`` ordinary Xavier-random
weights breaks this: ``hidden[row]`` is then a real, input-dependent, generally
nonzero value, so ``layer2``'s corresponding column *does* receive a nonzero
gradient from the first backward pass (its gradient formula never depended on
its own current value being nonzero — only on ``hidden[row]`` and the
downstream loss gradient, both already nonzero) and moves off zero
immediately; ``layer1``'s row then unlocks starting the following step, once
``layer2.weight[0, row]`` is no longer exactly zero. (This is the same
mechanism as zero-gamma residual-block initialisation in ResNet-style
networks: zeroing a block's *final* projection is safe and standard; zeroing
*everything feeding into it too* is not.) The exactly-zero pieces in this
module (``out_proj``/``linear2`` for identity-at-init, and ``QHead``'s
designated background columns/rows) are all of the "safe" kind — a real,
nonzero, input-dependent quantity feeds them from upstream, so their own
gradient is nonzero from step one even though their forward output starts at
zero.

## Reproducibility and determinism

Every learned weight is drawn from ``init_rng: np.random.Generator`` via
:func:`_xavier_uniform_` (computed with plain numpy, then copied into the
``nn.Parameter``) — never from torch's global default generator, which
``nn.Linear.reset_parameters()`` would otherwise silently consume. Biases are
always zero (a deliberate simplification, not a numerical requirement: a
bias's own gradient never depends on its current value, so zero-initialising
every bias costs nothing and needs no extra draws). Two ``TokenEncoder``s (or
``QHead``s) built from freshly-seeded generators with the same seed are
therefore bit-identical.

``dropout=0.0`` everywhere (no config knob for it exists yet, and none of
tickets 04-06 ask for one): a ``TransformerEncoderLayer`` with no dropout has
no stochastic op in its forward pass at all, so two forward calls on the same
instance — training mode or eval — agree bit for bit on CPU without needing
``torch.use_deterministic_algorithms``. CUDA's kernel-selection nondeterminism
(e.g. attention-backend choice) is a separate, real concern the ticket calls
out; :func:`~stdvrp.policies.torch_support.resolve_device` handles it once, for
every torch caller, rather than here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from stdvrp.policies.tokenizer import (
    CLIENT_TOKEN_BASE_WIDTH,
    GLOBAL_TOKEN_WIDTH,
    VEHICLE_TOKEN_BASE_WIDTH,
    Tokens,
)

__all__ = ["Embeddings", "QHead", "TokenEncoder"]

# Tokens.client_tokens' per-vehicle arc block is [minutes_from_vehicle, path_length_from_vehicle].
ARC_WIDTH = 2

# Type-embedding indices (Embeddings' three token kinds).
_TYPE_CLIENT = 0
_TYPE_VEHICLE = 1
_TYPE_GLOBAL = 2
_N_TOKEN_TYPES = 3


def _arc_dim0_index(d_model: int) -> int:
    """Global index of ``arc_embed``'s reconstruction dimension within a QHead
    input row (``[vehicle | client_context | arc | claimed]``).

    The single source of truth for the layout ``TokenEncoder.forward`` builds
    (``Embeddings.clients`` = ``concat([client_context, arc], dim=-1)``, each
    ``d_model`` wide) and ``QHead`` depends on for the warm start — both sides
    call this instead of re-deriving the offset independently, and
    ``TestWarmStart`` (``tests/unit/test_network.py``) is the end-to-end check
    that would catch the two going out of sync (e.g. a future reordering of
    ``TokenEncoder``'s ``torch.cat``).
    """
    return d_model + d_model


def _xavier_uniform_(tensor: torch.Tensor, rng: np.random.Generator) -> None:
    """Xavier/Glorot-uniform init, drawn from ``rng`` (never torch's global generator).

    ``tensor`` must be 2D (``[fan_out, fan_in]``, the ``nn.Linear``/attention
    projection convention) or 1D (embedding-style, no fan-in/out — bounded like
    a fan_in-only Xavier row of width ``tensor.shape[0]``).
    """
    shape = tuple(tensor.shape)
    if tensor.dim() == 2:
        fan_out, fan_in = shape
    else:
        fan_in = fan_out = shape[0]
    bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
    values = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    with torch.no_grad():
        tensor.copy_(torch.from_numpy(values))


def _zero_(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.zero_()


@dataclass(frozen=True, slots=True)
class Embeddings:
    """One decision epoch's post-encoder embeddings — everything ``QHead`` needs."""

    clients: torch.Tensor
    """``[n_pending, m, 2*d_model]``: dims ``[:d_model]`` are the client's
    vehicle-independent context (transformer-refined), dims ``[d_model:]`` are
    the ``(client, vehicle)``-specific arc embedding. Sliced per vehicle by the
    caller: ``embeddings.clients[:, v, :]``."""

    vehicles: torch.Tensor
    """``[m, d_model]``, ``last_node_reached`` order (matches ``vehicle_tokens``
    and the vehicle axis of ``clients`` above)."""


class TokenEncoder(nn.Module):
    """Runs once per decision epoch: tokens -> :class:`Embeddings`.

    See the module docstring for the full shape rationale and the warm-start
    construction this class is half of (the other half is :class:`QHead`).
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_observed_velocities: int,
        init_rng: np.random.Generator,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * d_model

        self.client_base_embed = nn.Linear(CLIENT_TOKEN_BASE_WIDTH, d_model)
        self.vehicle_embed = nn.Linear(VEHICLE_TOKEN_BASE_WIDTH + n_observed_velocities, d_model)
        self.global_embed = nn.Linear(GLOBAL_TOKEN_WIDTH, d_model)
        self.arc_embed = nn.Linear(ARC_WIDTH, d_model)
        self.type_embedding = nn.Parameter(torch.empty(_N_TOKEN_TYPES, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self._init_weights(init_rng)

    def _init_weights(self, rng: np.random.Generator) -> None:
        for embed in (self.client_base_embed, self.vehicle_embed, self.global_embed):
            _xavier_uniform_(embed.weight, rng)
            _zero_(embed.bias)

        # arc_embed: row 0 reconstructs minutes_from_vehicle exactly (the warm
        # start's other load-bearing half lives in QHead._init_weights below);
        # every other output dimension is ordinary Xavier-random capacity.
        _xavier_uniform_(self.arc_embed.weight, rng)
        _zero_(self.arc_embed.bias)
        with torch.no_grad():
            self.arc_embed.weight[0, :] = torch.tensor([1.0, 0.0])
            self.arc_embed.bias[0] = 0.0

        _xavier_uniform_(self.type_embedding, rng)

        for encoder_layer in self.transformer.layers:
            attn = encoder_layer.self_attn
            _xavier_uniform_(attn.in_proj_weight, rng)
            _zero_(attn.in_proj_bias)
            # Identity-at-init (see module docstring): a Linear with zero weight
            # AND zero bias is a constant-zero function of any finite input, so
            # zeroing out_proj/linear2 makes each residual branch contribute
            # exactly zero regardless of what in_proj/linear1 computed.
            _zero_(attn.out_proj.weight)
            _zero_(attn.out_proj.bias)
            _xavier_uniform_(encoder_layer.linear1.weight, rng)
            _zero_(encoder_layer.linear1.bias)
            _zero_(encoder_layer.linear2.weight)
            _zero_(encoder_layer.linear2.bias)
            # norm1/norm2: PyTorch's own default (weight=1, bias=0) is already
            # deterministic, not random -- nothing to draw from init_rng here.

    def forward(self, tokens: Tokens) -> Embeddings:
        client_tokens = torch.from_numpy(tokens.client_tokens).float()
        vehicle_tokens = torch.from_numpy(tokens.vehicle_tokens).float()
        global_token = torch.from_numpy(tokens.global_token).float()

        number_vehicles = vehicle_tokens.shape[0]
        base = client_tokens[:, :CLIENT_TOKEN_BASE_WIDTH]
        minutes_end = CLIENT_TOKEN_BASE_WIDTH + number_vehicles
        minutes = client_tokens[:, CLIENT_TOKEN_BASE_WIDTH:minutes_end]
        lengths = client_tokens[:, minutes_end:]
        arc_pairs = torch.stack([minutes, lengths], dim=-1)  # [n_pending, number_vehicles, 2]

        client_in = self.client_base_embed(base) + self.type_embedding[_TYPE_CLIENT]
        vehicle_in = self.vehicle_embed(vehicle_tokens) + self.type_embedding[_TYPE_VEHICLE]
        global_in = self.global_embed(global_token) + self.type_embedding[_TYPE_GLOBAL]

        sequence = torch.cat([client_in, vehicle_in, global_in.unsqueeze(0)], dim=0)
        encoded = self.transformer(sequence.unsqueeze(0)).squeeze(0)

        n_pending = client_in.shape[0]
        client_context = encoded[:n_pending]
        vehicle_context = encoded[n_pending : n_pending + number_vehicles]

        arc = self.arc_embed(arc_pairs)  # [n_pending, number_vehicles, d_model]
        clients = torch.cat(
            [client_context.unsqueeze(1).expand(-1, number_vehicles, -1), arc], dim=-1
        )  # [n_pending, number_vehicles, 2*d_model]

        return Embeddings(clients=clients, vehicles=vehicle_context)


class QHead(nn.Module):
    """Scores one vehicle against every pending Client: a **cost to minimize**.

    Matches the baseline's ``argmin`` convention (spec.md) — the caller (ticket
    06) takes ``argmin`` over feasible clients, exactly as ``MonteCarloPolicy``
    does. Never flip this sign.
    """

    def __init__(
        self,
        *,
        d_model: int,
        init_rng: np.random.Generator,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else d_model
        client_dim = 2 * d_model  # Embeddings.clients' per-vehicle slice width
        input_dim = d_model + client_dim + 1  # + claimed
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self._arc_dim0_index = _arc_dim0_index(d_model)

        self._init_weights(init_rng)

    def _init_weights(self, rng: np.random.Generator) -> None:
        _xavier_uniform_(self.layer1.weight, rng)
        _zero_(self.layer1.bias)
        with torch.no_grad():
            # Row 0: the warm start's reconstruction unit -- reads only
            # arc_embed's dimension 0 (== minutes_from_vehicle, see module
            # docstring). Rows 1..hidden-1 keep the Xavier-random weights just
            # drawn above -- zeroing them too would freeze layer2's matching
            # background columns forever (see module docstring's deadlock note).
            self.layer1.weight[0, :] = 0.0
            self.layer1.weight[0, self._arc_dim0_index] = 1.0
            self.layer1.bias[0] = 0.0

        _zero_(self.layer2.weight)
        _zero_(self.layer2.bias)
        with torch.no_grad():
            self.layer2.weight[0, 0] = 1.0

    def forward(
        self,
        vehicle_embedding: torch.Tensor,
        client_embeddings: torch.Tensor,
        claimed: torch.Tensor,
    ) -> torch.Tensor:
        """``vehicle_embedding``: ``[d_model]``. ``client_embeddings``:
        ``[n_pending, 2*d_model]`` (the caller's ``Embeddings.clients[:, v, :]``
        slice for this vehicle). ``claimed``: ``[n_pending]`` (nonzero where
        another vehicle already claimed that Client this decision). Returns
        ``[n_pending]``.
        """
        n_pending = client_embeddings.shape[0]
        vehicle = vehicle_embedding.unsqueeze(0).expand(n_pending, -1)
        x = torch.cat([vehicle, client_embeddings, claimed.unsqueeze(-1)], dim=-1)
        hidden = torch.relu(self.layer1(x))
        output: torch.Tensor = self.layer2(hidden).squeeze(-1)
        return output
