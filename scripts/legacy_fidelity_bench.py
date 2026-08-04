"""Legacy-fidelity bench: train the linear Policy with candidate legacy reverts toggled on.

Built for ``.scratch/linear-policy-learning/`` — the effort investigating why the
linear ``MonteCarloPolicy`` does not converge while the reference monolith
(``Main_Chengdu_Sirve_Con_Clip.py``) is stable from its 50th episode.

Every candidate revert is applied by **monkeypatching**, never by editing
``src/``. That is deliberate: each one undoes a deliberate, ADR-documented fix
(ADR-0004, ADR-0005, ADR-0008), so they must be measurable *before* anyone
decides whether to adopt them. Nothing here belongs in production code.

Usage::

    .venv/Scripts/python.exe -u scripts/legacy_fidelity_bench.py \\
        experiments/chengdu/config_linear_congestion_10k.yaml 500 50 clip,evalpool15 50

Positional arguments: config, training episodes, evaluation cadence, a
comma-separated toggle list (``none`` for the unmodified baseline), and the
number of evaluation seeds.

Toggles
-------
``clip``
    Clip features at 3 inside ``learn`` — the legacy's ``actualize_W`` line
    4374. **Already shipped** in ``MonteCarloPolicy`` as
    ``UPDATE_FEATURE_CEILING``; kept here only so a run can be compared against
    a no-clip baseline without reverting ``src/``. Harmless to pass (it
    re-applies the same arithmetic).
``bin3``
    Hold ``general[13]`` at 0, as both legacy sources structurally do —
    ``client_counts_earliness[3]`` is never assigned (legacy 2938-2947). Undoes
    B10. Measured *worse*; see the spec.
``legacydepot``
    Depot-idle predicates by position alone, ignoring ``vehicle_standing``.
    Undoes ADR-0005 / B1a / B1b at every site at once.
``unserved``
    The legacy's two termination pricing rules: charge only Clients with
    ``tau > due``, at the live ``tau`` on the horizon path and at
    ``episode_end_minute`` on the all-back path. Undoes ADR-0004 / B3.
``nofifo``
    Drop the congestion-expiry branch, which is commented out in the legacy
    (5723-5736; its own plot filename says ``_SinFIFO_``).
``evalpool15``
    Evaluate at ``vehicles + 15``, which is what ``training_model`` line 6983
    uses for its evaluation block. **Pass this whenever comparing against
    legacy numbers** — the Trainer's blocks use ``vehicles + 2``, a candidate
    set 13 narrower, and the two are not comparable.
``step6``
    Advance the simulation clock every 6 minutes instead of 2, landing on
    exactly the same decision instants (see the inline comment — starting
    ``next_decision_tau`` at 302 rather than 306 is load-bearing; at 306 the
    decision gate never fires at all and the episode takes no decisions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ``stdvrp.simulation`` must resolve before ``stdvrp.policies``: importing
# policies first goes monte_carlo -> action_set -> simulation.state, which pulls
# in simulation/__init__, which imports episode, which imports monte_carlo while
# it is still half-initialised (ImportError). A plain ``import`` sorts ahead of
# every ``from`` import in this block, so this line both breaks the cycle and
# keeps ruff's isort happy. The cycle is pre-existing in the package, not
# introduced here — it also breaks ``pytest tests/unit`` collection.
import stdvrp.simulation  # noqa: F401
from stdvrp.config import ExperimentConfig
from stdvrp.policies import action_set as action_set_module
from stdvrp.policies import feature_extraction as feature_extraction_module
from stdvrp.policies import monte_carlo as monte_carlo_module
from stdvrp.policies.feature_extraction import FeatureExtractor
from stdvrp.policies.monte_carlo import UPDATE_FEATURE_CEILING, MonteCarloPolicy
from stdvrp.simulation import episode as episode_module
from stdvrp.simulation import model as model_module
from stdvrp.simulation import run_training_episode
from stdvrp.traffic import world_cache
from stdvrp.training.episode_pool import EpisodeWorld

CONFIG = Path(sys.argv[1])
EPISODES = int(sys.argv[2])
EVAL_EVERY = int(sys.argv[3])
TOGGLES = set(sys.argv[4].split(",")) - {"none", ""}
EVAL_SEED_COUNT = int(sys.argv[5]) if len(sys.argv) > 5 else 10

config = ExperimentConfig.from_yaml(CONFIG)

if "clip" in TOGGLES:

    def clipped_learn(self, snapshots, actions, rewards):  # type: ignore[no-untyped-def]
        T = len(actions)
        U_t = 0.0
        lr = self.learning_rate
        for t in range(T - 1, -1, -1):
            U_t += rewards[t + 1]
            snapshot = snapshots[t]
            acquired = self._already_acquired_cost(snapshot)
            X = self.feature_extractor.action_features(
                self.feature_extractor.state_features(snapshot), actions[t]
            )
            X = np.clip(X, a_min=None, a_max=UPDATE_FEATURE_CEILING)
            assert self.W is not None
            Q_pred = np.dot(X, self.W)
            self.W = self.W + lr * ((U_t - acquired - Q_pred) * X)

    MonteCarloPolicy.learn = clipped_learn  # type: ignore[method-assign]

if "bin3" in TOGGLES:
    _original_general = FeatureExtractor._general_features

    def legacy_general(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        general = _original_general(self, *args, **kwargs)
        general[13] = 0.0
        return general

    FeatureExtractor._general_features = legacy_general  # type: ignore[method-assign]

if "legacydepot" in TOGGLES:

    def legacy_is_parked_at_depot(last_node_reached, vehicle_standing, depot):  # type: ignore[no-untyped-def]
        return last_node_reached == depot

    for module in (
        feature_extraction_module,
        action_set_module,
        monte_carlo_module,
        model_module,
    ):
        module.is_parked_at_depot = legacy_is_parked_at_depot  # type: ignore[attr-defined]

if "unserved" in TOGGLES:

    def legacy_charge(self, clock):  # type: ignore[no-untyped-def]
        tau = self.state.tau_episode
        windows = self.time_windows
        self.costs.charge_unserved_delays(
            clock - windows[client][1]
            for client in self.state.clients_not_visited
            if tau > windows[client][1]
        )

    def legacy_passing_horizon(self):  # type: ignore[no-untyped-def]
        self.state.terminal = True
        self._transition_ended = True
        legacy_charge(self, self.state.tau_episode)
        if self.state.tau_episode > self.shift_end_minute:
            state = self.state
            vehicles_out = sum(
                not model_module.is_parked_at_depot(node, standing, self.depot)
                for node, standing in zip(
                    state.last_node_reached, state.vehicle_standing, strict=True
                )
            )
            self.costs.charge_fleet_overtime(
                self.state.tau_episode - self.shift_end_minute, vehicles=vehicles_out
            )
        self.costs.commit_transition()

    def legacy_comeback(self):  # type: ignore[no-untyped-def]
        self.state.terminal = True
        self._transition_ended = True
        legacy_charge(self, self.episode_end_minute)
        self.costs.commit_transition()

    model_module.Model.terminate_state_passing_horizon = legacy_passing_horizon
    model_module.Model.terminate_state_if_all_vehicles_come_back = legacy_comeback

if "nofifo" in TOGGLES:
    model_module.Model._next_congestion_expiry = lambda self: float("inf")  # type: ignore[assignment]

if "step6" in TOGGLES:
    # Decisions fire when ``(tau + 178) % 6 == 0``: tau = 302, 308, 314, ...
    # Starting ``next_decision_tau`` at horizon_start + 6 = 306 and stepping by
    # 6 leaves that remainder at 4 forever, so NO decision is ever taken.
    # Starting at 302 lands on every decision instant of the 2-minute clock and
    # none of the intermediate ones. The congestion gate fires at 302, 422,
    # 542, ... — 120 apart, divisible by 6, so those are all still hit too.
    model_module.DECISION_EPOCH_MINUTES = 6
    _original_model_init = model_module.Model.__init__

    def six_minute_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _original_model_init(self, *args, **kwargs)
        self.next_decision_tau = self.state.horizon_start_minute + 2

    model_module.Model.__init__ = six_minute_init  # type: ignore[method-assign]


world = EpisodeWorld.load(config, cache_dir=world_cache.default_cache_dir())
eval_seeds = list(config.evaluation_seeds)[:EVAL_SEED_COUNT]
extra_actions = 15 if "evalpool15" in TOGGLES else 2
print(
    f"config {CONFIG.name}  lr={config.learning_rate}  eps={config.epsilon}  "
    f"toggles={sorted(TOGGLES) or ['none']}  "
    f"eval={len(eval_seeds)} seeds @ vehicles+{extra_actions}"
)

eval_vehicles = {seed: world.client_generator.generate(seed).vehicle_count for seed in eval_seeds}
episode_kwargs = world.episode_kwargs()
eval_kwargs = {name: value for name, value in episode_kwargs.items() if name != "learning_rate"}


def evaluate(w) -> float:  # type: ignore[no-untyped-def]
    costs = [
        episode_module.run_evaluation_episode(
            seed=seed,
            W=w,
            number_actions_test=eval_vehicles[seed] + extra_actions,
            **eval_kwargs,
        ).total_cost
        for seed in eval_seeds
    ]
    return float(np.mean(costs))


w = None
learning_rate = (
    config.warmup_learning_rate if config.warmup_learning_rate is not None else config.learning_rate
)
for index in range(EPISODES):
    seed = config.first_train_seed + index
    result = run_training_episode(seed=seed, W=w, learning_rate=learning_rate, **episode_kwargs)
    learning_rate = config.learning_rate
    w = result.w
    completed = index + 1
    if completed % EVAL_EVERY == 0:
        print(
            f"  after {completed:4d} episodes: eval mean cost {evaluate(w):9.2f}   "
            f"|W| {np.linalg.norm(w):9.2f}  max|W| {np.abs(w).max():8.2f}",
            flush=True,
        )
print("W = [" + ", ".join(f"{value:.4g}" for value in w) + "]")
