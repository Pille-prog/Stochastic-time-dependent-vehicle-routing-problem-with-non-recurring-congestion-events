"""Frozen, validated experiment configuration.

Replaces the legacy comma-separated ``sys.argv`` string plus the values that were
hardcoded across ``main()``, ``training_and_testing`` and ``model`` (horizon 300-780,
``n_arcs=3``, warm-up learning rate 1e-6, evaluation seeds 100000-100049, data file
paths). One YAML file per experiment, versioned next to the experiment.

The legacy's hardcoded ``mean_static_policy`` plot baseline lived here as
``static_policy_mean_cost`` through simulator-correctness; ticket 01
(neural-policy) retired it in favour of ``ReferenceCard``
(``stdvrp.training.reference_card``) — a frozen per-seed cost vector, which
supports the paired comparison a single scalar cannot.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Accepted values of ``neural_warm_start``, **duplicated** from
#: ``stdvrp.policies.tokenizer.WARM_START_WEIGHTS`` rather than imported, and
#: pinned in sync by ``test_experiment_config.py``.
#:
#: Importing it would make this module — which everything imports, early —
#: depend on the ``stdvrp.policies`` package, whose ``__init__`` pulls in
#: ``monte_carlo`` and through it ``stdvrp.simulation``: a circular import that
#: surfaces only when ``config`` happens to be imported first. That is the same
#: reason ``tokenizer.py`` duplicates the simulator's cost rates instead of
#: importing them. Two names is a cheap price for a validation that cannot
#: depend on import order.
NEURAL_WARM_STARTS = ("minutes", "cost")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Every knob of one experiment; immutable once loaded."""

    # Data (formerly hardcoded relative file names resolved against the CWD).
    data_dir: Path
    links_file: str
    shortest_paths_file: str
    instance_day: int
    traffic_days: tuple[int, ...]

    # Horizon in minutes since 03:00 (formerly hardcoded 300 and 780). Two
    # distinct clocks (ticket 02, simulator-correctness, B12/B15): the shift's
    # end, past which overtime accrues (formerly the lying ``horizon_end_minute``
    # name — the code already called it ``shift_end_minute`` internally), and
    # the episode's hard stop, formerly the model-internal hardcoded
    # ``EMERGENCY_HORIZON`` and independent of any config.
    horizon_start_minute: int
    shift_end_minute: int
    episode_end_minute: int

    # Demand (former argv: mean_number_clients, diff_TW; former ClientGenerator
    # hardcodes: gauss stddev 30, 60-client floor, the {150: 28, 250: 29}
    # vehicle-ratio table, range(1, 1900); former main() hardcodes:
    # random.seed(0); random.sample(range(1, 1900), 150)).
    mean_number_clients: int
    client_count_stddev: float
    min_number_clients: int
    clients_per_vehicle: int
    time_window_spread: int
    client_universe_seed: int
    client_universe_size: int
    client_universe_node_range: tuple[int, int]

    # Congestion (former argv).
    congestion_lower_bound: float
    congestion_upper_bound: float
    max_congestion_duration: int

    # Policy and training (former argv plus hardcoded values). The legacy always
    # trained the first Episode with a hardcoded 1e-6 warm-up rate; null disables
    # the warm-up so learning_rate applies from episode 1 (ticket 12, ADR-0001
    # phase-2 change log).
    total_train_iterations: int
    test_frequency: int
    learning_rate: float
    warmup_learning_rate: float | None
    epsilon: float
    n_observed_velocities: int
    first_train_seed: int
    evaluation_seed_start: int
    evaluation_seed_count: int
    # Inert (ticket 02, simulation-performance): the legacy test_model repeated
    # every (action count, seed) episode this many times and averaged, but
    # per-seed Generators (ticket 13) made every repeat bit-identical, so
    # Trainer.final_test now runs each pair once and never reads this field.
    # Kept for YAML config-file compatibility (still required, still validated
    # positive) rather than removed.
    test_episodes: int

    # Final test (former test_model hardcodes: the action-count list [2,10,20,30,40,50]
    # and the per-seed fleet-size tables keyed by mean_number_clients).
    test_action_counts: tuple[int, ...]
    test_seeds: tuple[int, ...]
    test_vehicle_counts: tuple[int, ...]

    # Neural policy (ticket 03, neural-policy): the transformer approximator's
    # architecture, tunable on the evaluation seeds only (spec.md's "Starting
    # architecture" row: d=128, 3 layers, 4 heads — measured 594,945 params at
    # dim_feedforward=4*d_model, see spec.md's "Compute budget" section). torch
    # is an optional extra (`stdvrp.policies` stays importable without it), so
    # these fields are plain ints/strings, never a torch type.
    neural_d_model: int
    neural_n_layers: int
    neural_n_heads: int
    # Ticket 06: the Policy's own learning-rule knobs, separate from the linear
    # baseline's ``learning_rate`` (a constant-step SGD rate; this one seeds
    # Adam, an unrelated scale — spec.md's live-report example shows ``3.0e-4``,
    # nothing like the baseline's ``1e-5``). ``neural_learning_rate`` is what
    # ticket 07's convergence stopping multiplies by 0.3 on a patience trigger.
    # ``neural_learn_passes`` is ``learn``'s ``K`` (shuffled minibatch passes
    # per episode; spec.md's compute-budget measurement used 4).
    # ``neural_batch_size`` is the minibatch size within one episode's ~400
    # (epoch, vehicle) decision samples.
    neural_learning_rate: float
    neural_learn_passes: int
    neural_batch_size: int
    # Ticket 08 (Gate A stability): optional max L2 norm for the gradient of
    # one minibatch step of ``TransformerMonteCarloPolicy.learn`` (clipped over
    # encoder+head jointly, torch.nn.utils.clip_grad_norm_). ``null`` (the
    # default) disables clipping -- the exact pre-knob behavior. The lever the
    # ticket's lr/gradient-clipping stability sweep varies on
    # ``evaluation_seeds``; not a frozen acceptance parameter.
    neural_grad_clip_norm: float | None = None
    # Ticket 08; DEAD since ticket 15 (kept for YAML/historical-run compat,
    # still validated, no longer wired to anything). Used to select which
    # myopic warm start ``TokenEncoder`` initialised ``arc_embed`` row 0 with
    # (``network.WARM_START_WEIGHTS``). Ticket 15 took the myopic base ``c``
    # out of the network entirely (``network.py``, "The myopic base"): ``c``
    # is now always ``WARM_START_WEIGHTS["cost"]``, computed by
    # ``TokenEncoder.forward`` and never touched by this field. The value
    # below (and every existing config file's) is inert; the historical
    # "minutes" vs "cost" measurements it once selected between stay citeable
    # under the old architecture (network.py, "The myopic base").
    neural_warm_start: str = "minutes"
    # Ticket 08; DEAD-ish since ticket 15 (kept wired, but the problem it
    # addresses is far smaller now -- see below). ``delta`` for ``learn``'s
    # Huber loss. torch's own default of 1.0 is not a neutral choice here --
    # the standardized target and the network's output both live around
    # 1e-2, so every residual falls in the quadratic branch and the loss is
    # exactly ``0.5 * MSE``, robustness never engaging. One truncated
    # training episode (the 40000-minus-200-per-visit terminal penalty,
    # research note F10) then lands on *every* decision epoch's target in
    # that episode, squared. A delta near the residual scale is what makes
    # those episodes contribute a bounded gradient instead of a dominating
    # one. Kept at 1.0 by default so the knob alone changes nothing. Ticket
    # 16 is what actually retires this field, by replacing ``learn``'s
    # per-episode Huber-loss SGD with a closed-form least-squares solve.
    neural_huber_delta: float = 1.0
    # Ticket 08; MOOT since ticket 15 (kept wired, at its measured
    # bit-identical-to-absent default). How much faster ``QHead``'s level
    # term (``linear``'s bias -- the one weight added identically to every
    # candidate, so the only one the argmin cannot see) moves per optimizer
    # step. Under the pre-ticket-15 architecture, at init ``Q_joint`` was
    # 0.3-0.9 while the standardized return was ~0.03, and closing that gap
    # at the shared learning rate took ~100 episodes of same-signed steps
    # that dragged every ranking weight along with them; a gain closed it
    # inside the first episode instead. Ticket 15 removes the mismatch at
    # its source (``network.py``, "The level term" -- the section's "why
    # this is moot now" note): ``Q_joint`` at init is now ``Σ_v c(s, v,
    # a_v)``, already on the scale of the return it approximates, so there
    # is no longer a large, same-signed gap for this gain to close. 1.0 (the
    # default) is bit-identical to the term not existing.
    neural_level_gain: float = 1.0
    # "cpu", "cuda", or "auto" (ticket 12; amends spec.md decision 7). "auto"
    # is the default: it resolves once per run (torch_support.resolve_device)
    # to "cuda" if available, else "cpu". The old "cpu by default" choice
    # rested on EpisodePool workers each wanting their own CUDA context on an
    # 8 GB laptop GPU (ticket 03) -- a risk that never became live, because
    # the neural training path is single-process (ticket 07 runs its
    # evaluation blocks serially; EpisodePool appears nowhere in trainer.py's
    # neural path). "auto" is kept as the default despite ticket 12's own
    # measurement finding CUDA is *not* faster than CPU on the real network on
    # the reference hardware (contrary to ticket 03's stub-based estimate --
    # see spec.md's compute-budget section) -- an "auto" that resolves to the
    # genuinely better local device costs nothing, and the choice is
    # machine-specific, not something to hardcode. Because CPU and CUDA do not
    # agree bit for bit, "auto" moves the pinning of a result out of the
    # config and into the run's own record (the resolved device is printed,
    # checkpointed, and written into the results); an explicit "cuda" with no
    # GPU available fails loudly rather than silently downgrading
    # (resolve_device).
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.traffic_days:
            raise ValueError("traffic_days must not be empty")
        if self.instance_day not in self.traffic_days:
            raise ValueError(f"instance_day {self.instance_day} must be one of traffic_days")
        if not 0 <= self.horizon_start_minute < self.shift_end_minute:
            raise ValueError("horizon must satisfy 0 <= horizon_start_minute < shift_end_minute")
        if self.shift_end_minute > self.episode_end_minute:
            # B15: an unguarded shift end past the episode's hard stop is how the
            # legacy priced negative overtime — reject the config outright rather
            # than merely leave it unreached (spec.md decision 5).
            raise ValueError("shift_end_minute must be <= episode_end_minute")
        if self.mean_number_clients <= 0:
            raise ValueError("mean_number_clients must be positive")
        if self.client_count_stddev < 0:
            raise ValueError("client_count_stddev must be >= 0")
        if not 0 <= self.time_window_spread <= self.shift_end_minute - self.horizon_start_minute:
            raise ValueError("time_window_spread must fit within the horizon")
        lo, hi = self.client_universe_node_range
        if lo >= hi:
            raise ValueError("client_universe_node_range must be (low, high) with low < high")
        # The sample of client nodes must at least fit the floor; a gauss draw above
        # hi - lo still fails at generation time, exactly as the legacy would.
        if not 0 < self.min_number_clients <= hi - lo:
            raise ValueError(
                f"min_number_clients must be in 1..{hi - lo} for node range ({lo}, {hi})"
            )
        if not 0 < self.client_universe_size <= hi - lo:
            raise ValueError(
                f"client_universe_size must be in 1..{hi - lo} for node range ({lo}, {hi})"
            )
        if not 0 <= self.congestion_lower_bound <= self.congestion_upper_bound:
            raise ValueError("congestion bounds must satisfy 0 <= lower <= upper")
        # The legacy draws congestion durations with random.randint(30, max_duration).
        if self.max_congestion_duration < 30:
            raise ValueError("max_congestion_duration must be >= 30 minutes")
        for name in (
            "clients_per_vehicle",
            "total_train_iterations",
            "test_frequency",
            "test_episodes",
            "evaluation_seed_count",
            "n_observed_velocities",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_learning_rate is not None and self.warmup_learning_rate <= 0:
            raise ValueError("warmup_learning_rate must be positive or null")
        if not self.test_action_counts or any(count <= 0 for count in self.test_action_counts):
            raise ValueError("test_action_counts must be a non-empty list of positive integers")
        if not self.test_seeds:
            raise ValueError("test_seeds must not be empty")
        if len(self.test_vehicle_counts) != len(self.test_seeds) or any(
            count <= 0 for count in self.test_vehicle_counts
        ):
            raise ValueError(
                "test_vehicle_counts must pair a positive fleet size with every test seed"
            )
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        for name in (
            "neural_d_model",
            "neural_n_layers",
            "neural_n_heads",
            "neural_learn_passes",
            "neural_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.neural_d_model % self.neural_n_heads != 0:
            raise ValueError("neural_d_model must be divisible by neural_n_heads")
        if self.neural_learning_rate <= 0:
            raise ValueError("neural_learning_rate must be positive")
        if self.neural_grad_clip_norm is not None and self.neural_grad_clip_norm <= 0:
            raise ValueError("neural_grad_clip_norm must be positive or null")
        if self.neural_huber_delta <= 0:
            raise ValueError("neural_huber_delta must be positive")
        if self.neural_level_gain <= 0:
            raise ValueError("neural_level_gain must be positive")
        if self.neural_warm_start not in NEURAL_WARM_STARTS:
            raise ValueError(
                f"neural_warm_start must be one of {sorted(NEURAL_WARM_STARTS)}, "
                f"got {self.neural_warm_start!r}"
            )
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {self.device!r}")

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        """The fixed seeds used for every evaluation pass (legacy range(100000, 100050))."""
        return tuple(
            range(
                self.evaluation_seed_start, self.evaluation_seed_start + self.evaluation_seed_count
            )
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load and validate a config; relative data_dir resolves against the YAML's folder."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: config must be a YAML mapping")
        field_names = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(set(raw) - set(field_names))
        if unknown:
            raise ValueError(f"{path}: unknown config keys {unknown}")
        # A field with a dataclass default may be omitted from the YAML (it
        # takes the default, exactly as direct construction would); every
        # defaultless field stays required.
        required = [
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        ]
        missing = sorted(set(required) - set(raw))
        if missing:
            raise ValueError(f"{path}: missing config keys {missing}")

        values: dict[str, Any] = dict(raw)
        data_dir = Path(str(values["data_dir"]))
        if not data_dir.is_absolute():
            data_dir = path.parent / data_dir
        values["data_dir"] = data_dir
        for name in ("traffic_days", "test_action_counts", "test_seeds", "test_vehicle_counts"):
            values[name] = tuple(_require_int_list(path, name, values[name]))
        node_range = _require_int_list(
            path, "client_universe_node_range", values["client_universe_node_range"]
        )
        if len(node_range) != 2:
            raise ValueError(f"{path}: client_universe_node_range must have exactly 2 entries")
        values["client_universe_node_range"] = (node_range[0], node_range[1])
        for name in (
            "client_count_stddev",
            "congestion_lower_bound",
            "congestion_upper_bound",
            "learning_rate",
            "epsilon",
            "neural_learning_rate",
        ):
            values[name] = _require_float(path, name, values[name])
        if values["warmup_learning_rate"] is not None:
            values["warmup_learning_rate"] = _require_float(
                path, "warmup_learning_rate", values["warmup_learning_rate"]
            )
        if values.get("neural_grad_clip_norm") is not None:
            values["neural_grad_clip_norm"] = _require_float(
                path, "neural_grad_clip_norm", values["neural_grad_clip_norm"]
            )
        for name in ("links_file", "shortest_paths_file", "device"):
            if name in values and (not isinstance(values[name], str) or not values[name]):
                raise ValueError(f"{path}: {name} must be a non-empty string")
        for name in (
            "instance_day",
            "horizon_start_minute",
            "shift_end_minute",
            "episode_end_minute",
            "mean_number_clients",
            "min_number_clients",
            "clients_per_vehicle",
            "time_window_spread",
            "client_universe_seed",
            "client_universe_size",
            "max_congestion_duration",
            "total_train_iterations",
            "test_frequency",
            "n_observed_velocities",
            "first_train_seed",
            "evaluation_seed_start",
            "evaluation_seed_count",
            "test_episodes",
            "neural_d_model",
            "neural_n_layers",
            "neural_n_heads",
            "neural_learn_passes",
            "neural_batch_size",
        ):
            values[name] = _require_int(path, name, values[name])
        return cls(**values)


def _require_int(path: Path, name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: {name} must be an integer, got {value!r}")
    return value


def _require_int_list(path: Path, name: str, value: Any) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{path}: {name} must be a list of integers, got {value!r}")
    return list(value)


def _require_float(path: Path, name: str, value: Any) -> float:
    # PyYAML parses "1e-6" (no dot) as a string; accept it as a float for ergonomics.
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{path}: {name} must be a number, got {value!r}")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{path}: {name} must be a number, got {value!r}") from error
