"""Capture the golden master from the untouched legacy script on the full Chengdu data.

This is ticket 04's shim + capture driver (ADR-0001 and its addendum). The legacy
monolith is imported as-is — zero edits — and its hardcoded relative data paths
(``link.csv``, ``speed[601]_[0].csv``, ``speed[601]_[1].csv``, plus the ~88 day
files and ``all_shortest_paths.csv`` discovered in ticket 03) are redirected by
running with the working directory set to the full dataset folder.

The capture protocol mirrors ``training_and_testing.training_model`` /
``test_model`` line by line, minus plotting and report files, with the loop
bounds made explicit:

* a small training run (seeds 1000, 1001, ...; warm-up learning rate 1e-6 on the
  first episode exactly as the legacy does), capturing W after every episode;
* one evaluation block with the newest W (the legacy evaluates seeds
  100000..100049 every ``test_frequency`` episodes; we run a leading subset —
  each episode fully re-seeds ``random`` and ``np.random``, so a subset's
  per-seed values are identical to the full run's);
* per-seed test episodes with the legacy's hardcoded seed/vehicle tables
  (leading subset, same argument), capturing the total cost and its four
  components at full float precision.

Because a single evaluation block always improves on the legacy's initial
``Q_pred = 1e11``, ``Best_W`` (which drives the test episodes) equals the
post-training W regardless of how many evaluation seeds run.

Usage (writes tests/fixtures/golden_master/chengdu_full.json):

    uv run python scripts/capture_golden_master.py

The golden pytest test (tests/test_golden_master.py, marker ``golden``) re-runs
this protocol from the stored file and requires exact equality.

Loading the data through the legacy classes costs ~15 minutes of pure CPU
(88 speed files re-read plus a pure-Python parse of the 907 MB
``all_shortest_paths.csv``), so the built world objects are pickled to a local
cache (outside the repo/OneDrive; see ``default_cache_path``) keyed by the
legacy sha256, the data-file signature and the world-shaping parameters. The
cache stores the pristine post-``__init__`` state — episodes mutate
``DataCalculations`` — so a cache hit is state-identical to a fresh build.
Delete the cache file (or pass ``--no-cache``) to force a cold rebuild.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = REPO_ROOT / "Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "golden_master" / "chengdu_full.json"

# The legacy's hardcoded per-seed vehicle table for mean_number_clients == 150
# (test_model, lines 6396-6398 of the monolith).
TEST_SEEDS_150 = [
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
]
TEST_VEHICLES_150 = [
    6,
    5,
    5,
    7,
    7,
    6,
    6,
    5,
    6,
    5,
    6,
    4,
    7,
    5,
    6,
    6,
    5,
    5,
    6,
    5,
    6,
    4,
    7,
    6,
    5,
    7,
    8,
    4,
    4,
    4,
    4,
    4,
    5,
    4,
    5,
    6,
    8,
    5,
    5,
    7,
    5,
    7,
    6,
    6,
    4,
    6,
    7,
    5,
    6,
    6,
]


def consumed_data_files() -> list[str]:
    """Every file the legacy load path reads, per ticket 03's findings."""
    days = [*range(601, 631), *range(701, 715)]
    speed = [f"speed[{day}]_[{half}].csv" for day in days for half in (0, 1)]
    return ["link.csv", "all_shortest_paths.csv", *speed]


def missing_data_files(data_dir: Path) -> list[str]:
    return [name for name in consumed_data_files() if not (data_dir / name).is_file()]


def data_signature(data_dir: Path) -> dict[str, int]:
    """Cheap drift detector: byte size of every consumed data file."""
    return {name: (data_dir / name).stat().st_size for name in consumed_data_files()}


def default_data_dir() -> Path:
    env = os.environ.get("STDVRP_DATA_DIR")
    return Path(env) if env else REPO_ROOT.parent


def load_legacy() -> ModuleType:
    """Import the monolith unchanged. Headless matplotlib; no other side effects."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location("legacy_monolith", LEGACY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # When run as `python script.py`, __builtins__ is the builtins *module*; when
    # imported, CPython makes it a dict — which breaks the legacy's literal
    # `__builtins__.min(...)` (line 5717). Restore the __main__-style environment.
    module.__dict__["__builtins__"] = builtins
    # Pickle stores legacy instances by "legacy_monolith.<class>" reference; the
    # module must be importable by that name for the world cache to round-trip.
    sys.modules["legacy_monolith"] = module
    return module


# --- World cache: skip the ~15-minute legacy data load on repeat runs ----------

CACHE_FORMAT = 1


def default_cache_path() -> Path:
    """A local, non-synced location (the pickle can reach gigabytes)."""
    env = os.environ.get("STDVRP_GOLDEN_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "stdvrp" / "golden_world_cache.pkl"


def world_cache_key(data_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Everything the built world depends on; any change invalidates the cache."""
    import numpy
    import pandas

    return {
        "cache_format": CACHE_FORMAT,
        "legacy_sha256": hashlib.sha256(LEGACY_SCRIPT.read_bytes()).hexdigest(),
        "data_signature": data_signature(data_dir),
        "world_params": {
            "horizon_start_time": protocol["horizon_start_time"],
            "horizon_end_time": protocol["horizon_end_time"],
            "max_congestion_duration": protocol["max_congestion_duration"],
        },
        # Pickles are not guaranteed portable across these versions.
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        },
    }


def read_world_cache(path: Path, key: dict[str, Any]) -> Any | None:
    """Return the cached world on an exact key match; None on miss or corruption."""
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    return payload.get("world")


def write_world_cache(path: Path, key: dict[str, Any], world: Any) -> None:
    """Atomic write (tmp + replace) so an interrupted run never leaves a torn cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"key": key, "world": world}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_world(
    legacy: ModuleType, data_dir: Path, protocol: dict[str, Any], cache_path: Path | None
) -> tuple[Any, Any]:
    """Build (data_calculations, spm) exactly as legacy main() does, or reuse the cache.

    Must run with the working directory already set to ``data_dir`` (the legacy
    reads relative paths). The setup RNG calls run on both branches so the global
    ``random`` stream ends in the same state either way (every episode re-seeds,
    so this is belt and braces, not a behavioral requirement).

    The cache is written immediately after construction, before any episode runs:
    episodes mutate ``DataCalculations`` (``total_congestions``,
    ``unexpected_event_velocity``, ...), and the cache must hold the pristine
    state a fresh build would produce.
    """
    t0 = time.perf_counter()
    # --- Setup, verbatim from legacy main() ---------------------------------
    file_path = "link.csv"
    file_path_velocities_morning = "speed[601]_[0].csv"
    file_path_velocities_afternoon = "speed[601]_[1].csv"
    random.seed(0)
    clients = random.sample(range(1, 1900), 150)
    clients.insert(0, 0)

    key = world_cache_key(data_dir, protocol)
    if cache_path is not None:
        cached = read_world_cache(cache_path, key)
        if cached is not None:
            data_calculations, spm = cached
            log(f"world loaded from cache in {time.perf_counter() - t0:.1f}s ({cache_path})")
            return data_calculations, spm

    g = legacy.environment(
        file_path,
        file_path_velocities_morning,
        file_path_velocities_afternoon,
        clients,
        protocol["horizon_start_time"],
        protocol["horizon_end_time"],
    )
    g.preprocess_data_average()
    data_calculations = legacy.DataCalculations(g, protocol["max_congestion_duration"])
    spm = legacy.shortest_path_memory(g)
    del g
    log(f"data loaded in {time.perf_counter() - t0:.1f}s")

    if cache_path is not None:
        t1 = time.perf_counter()
        write_world_cache(cache_path, key, (data_calculations, spm))
        log(f"world cached in {time.perf_counter() - t1:.1f}s ({cache_path})")
    return data_calculations, spm


def default_protocol() -> dict[str, Any]:
    return {
        # Experiment parameters (the sys.argv fields of the legacy main()).
        "learning_rate": 0.001,
        "epsilon": 0.05,
        "congestion_lower_bound": 0.1,
        "congestion_upper_bound": 0.3,
        "max_congestion_duration": 60,
        "mean_number_clients": 150,
        "diff_TW": 150,
        # Constants the legacy hardcodes inside its loops.
        "horizon_start_time": 300,
        "horizon_end_time": 780,
        "n_arcs": 3,
        "warmup_learning_rate": 1e-6,
        # The legacy policy's exploration RNGs (local_rng, local_rng_2) are
        # unseeded random.Random() instances — training would differ on every
        # run. The capture seeds them per episode (offset + train seed) right
        # after constructing the policy, making the stored W trajectory exactly
        # re-runnable. Purely driver-level: the legacy script is not modified.
        "train_exploration_seed_offset": 10_000_000,
        "train_repair_seed_offset": 20_000_000,
        # Loop bounds (legacy: unbounded train count, eval 100000..100049,
        # test over the full 50-entry seed table and actions [2,10,20,30,40,50]).
        "train_seeds": [1000, 1001, 1002, 1003],
        "eval_seeds": list(range(100000, 100010)),
        "test_actions": [2],
        "test_seeds": TEST_SEEDS_150[:10],
        "test_vehicles": TEST_VEHICLES_150[:10],
    }


def run_capture(
    legacy: ModuleType,
    data_dir: Path,
    protocol: dict[str, Any],
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Run the mirrored protocol inside ``data_dir`` and return the captured values."""
    np = legacy.np  # the same module object as our numpy; named for fidelity below

    learning_rate = protocol["learning_rate"]
    epsilon = protocol["epsilon"]
    congestion_lower_bound = protocol["congestion_lower_bound"]
    congestion_upper_bound = protocol["congestion_upper_bound"]
    max_congestion_duration = protocol["max_congestion_duration"]
    mean_number_clients = protocol["mean_number_clients"]
    diff_TW = protocol["diff_TW"]
    horizon_start_time = protocol["horizon_start_time"]
    horizon_end_time = protocol["horizon_end_time"]
    n_arcs = protocol["n_arcs"]

    with contextlib.chdir(data_dir):
        data_calculations, spm = load_world(legacy, data_dir, protocol, cache_path)

        # --- Training, mirroring training_model ---------------------------------
        w = None
        lr = protocol["warmup_learning_rate"]
        w_trajectory = []
        for random_seed in protocol["train_seeds"]:
            t_ep = time.perf_counter()
            rutas_multiarmed_150 = [[]]
            random_depot = 0
            cg = legacy.ClientGenerator(random_depot)
            cg.client_generator_function(
                random_seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time
            )
            np.random.seed(random_seed)
            number_vehicles = int(cg.number_vehicles)
            clients = cg.client_list
            number_clients = len(clients)
            number_actions_train = number_vehicles + 2
            number_actions_test = number_vehicles + 2

            s = legacy.state(number_vehicles, clients, n_arcs, horizon_start_time, random_depot)
            p = legacy.policy(
                number_vehicles,
                rutas_multiarmed_150,
                spm,
                cg,
                data_calculations,
                s,
                number_clients,
                epsilon,
                random_depot,
                congestion_lower_bound,
                congestion_upper_bound,
                number_actions_train,
                number_actions_test,
                lr,
                w,
            )
            m = legacy.model(
                s,
                p,
                data_calculations,
                spm,
                cg,
                number_vehicles,
                horizon_start_time,
                horizon_end_time,
                random_depot,
                congestion_lower_bound,
                congestion_upper_bound,
                max_congestion_duration,
            )
            # Make the unseeded exploration RNGs reproducible (see default_protocol).
            p.local_rng.seed(protocol["train_exploration_seed_offset"] + random_seed)
            p.local_rng_2.seed(protocol["train_repair_seed_offset"] + random_seed)
            lr = learning_rate
            m.create_monte_carlo_episode_train()
            w = m.policy.W
            w_trajectory.append(copy.deepcopy(w))
            log(
                f"train seed {random_seed}: cost={m.total_cost!r} "
                f"({time.perf_counter() - t_ep:.1f}s)"
            )

        # --- One evaluation block, mirroring the cont % test_frequency branch ---
        Newest_W = copy.deepcopy(w)
        eval_costs = []
        for seed in protocol["eval_seeds"]:
            t_ep = time.perf_counter()
            random_depot = 0
            cg = legacy.ClientGenerator(random_depot)
            cg.client_generator_function(
                seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time
            )
            np.random.seed(seed)
            number_vehicles = int(cg.number_vehicles)
            clients = cg.client_list
            number_clients = len(clients)
            number_actions_train = number_vehicles + 2
            number_actions_test = number_vehicles + 2

            s = legacy.state(number_vehicles, clients, n_arcs, horizon_start_time, random_depot)
            p = legacy.policy(
                number_vehicles,
                [[]],
                spm,
                cg,
                data_calculations,
                s,
                number_clients,
                epsilon,
                random_depot,
                congestion_lower_bound,
                congestion_upper_bound,
                number_actions_train,
                number_actions_test,
                learning_rate,
                Newest_W,
            )
            m = legacy.model(
                s,
                p,
                data_calculations,
                spm,
                cg,
                number_vehicles,
                horizon_start_time,
                horizon_end_time,
                random_depot,
                congestion_lower_bound,
                congestion_upper_bound,
                max_congestion_duration,
            )
            m.create_monte_carlo_episode_test()
            eval_costs.append(m.total_cost)
            log(f"eval seed {seed}: cost={m.total_cost!r} ({time.perf_counter() - t_ep:.1f}s)")
        # First (and only) evaluation always beats Q_pred = 1e11, so Best_W = Newest_W.
        Best_W = copy.deepcopy(Newest_W)

        # --- Test episodes, mirroring test_model (num_iteraciones_test = 1) -----
        test_results: dict[str, list[dict[str, Any]]] = {}
        for actions in protocol["test_actions"]:
            per_seed = []
            for seed, number_vehicles in zip(
                protocol["test_seeds"], protocol["test_vehicles"], strict=True
            ):
                t_ep = time.perf_counter()
                rutas_multiarmed_150 = [[]]
                random_depot = 0
                cg = legacy.ClientGenerator(random_depot)
                cg.client_generator_function(
                    seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time
                )
                np.random.seed(seed)
                clients = cg.client_list
                number_clients = len(clients)
                number_actions_train = number_vehicles + 2
                number_actions_test = number_vehicles + actions

                s = legacy.state(
                    number_vehicles, clients, n_arcs, horizon_start_time, cg.random_depot
                )
                p = legacy.policy(
                    number_vehicles,
                    rutas_multiarmed_150,
                    spm,
                    cg,
                    data_calculations,
                    s,
                    number_clients,
                    epsilon,
                    cg.random_depot,
                    congestion_lower_bound,
                    congestion_upper_bound,
                    number_actions_train,
                    number_actions_test,
                    learning_rate,
                    Best_W,
                )
                m = legacy.model(
                    s,
                    p,
                    data_calculations,
                    spm,
                    cg,
                    number_vehicles,
                    horizon_start_time,
                    horizon_end_time,
                    cg.random_depot,
                    congestion_lower_bound,
                    congestion_upper_bound,
                    max_congestion_duration,
                )
                m.create_monte_carlo_episode_test()
                per_seed.append(
                    {
                        "seed": seed,
                        "vehicles": number_vehicles,
                        "total_cost": m.total_cost,
                        "distance_cost": m.total_distance_cost,
                        "delay_cost": m.total_delay_cost,
                        "earliness_cost": m.total_earliness_cost,
                        "overtime_cost": m.total_overtime_cost,
                        "tau": m.state.tau_episode,
                        "state_count": m.total_state_counter,
                        "delay_clients": m.total_delay_clients,
                        "earliness_clients": m.total_earliness_clients,
                    }
                )
                log(
                    f"test actions={actions} seed {seed}: cost={m.total_cost!r} "
                    f"({time.perf_counter() - t_ep:.1f}s)"
                )
            test_results[str(actions)] = per_seed

    return {
        "training": {"w_trajectory": w_trajectory, "eval_costs": eval_costs},
        "test": test_results,
    }


def capture_document(
    data_dir: Path, protocol: dict[str, Any], cache_path: Path | None = None
) -> dict[str, Any]:
    legacy = load_legacy()
    results = run_capture(legacy, data_dir, protocol, cache_path)
    meta = {
        "legacy_script": LEGACY_SCRIPT.name,
        "legacy_sha256": hashlib.sha256(LEGACY_SCRIPT.read_bytes()).hexdigest(),
        "data_signature": data_signature(data_dir),
        "versions": {
            "python": platform.python_version(),
            "numpy": legacy.np.__version__,
            "pandas": legacy.pd.__version__,
            "networkx": legacy.nx.__version__,
        },
    }
    return to_jsonable({"meta": meta, "protocol": protocol, **results})


# --- Pure helpers (unit-tested in tests/unit/test_golden_capture_helpers.py) ---


def to_jsonable(value: Any) -> Any:
    """Convert captured values to plain JSON types, rejecting non-finite floats.

    json round-trips float64 exactly (shortest-repr serialization), which is what
    makes exact golden equality through a text file sound.
    """
    import numpy as np

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"non-finite value in golden master: {value!r}")
        return result
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported type in golden master: {type(value).__name__}")


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(document), indent=1)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def compare_results(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Exact structural comparison; returns human-readable difference descriptions."""
    label = path or "<root>"
    if isinstance(expected, dict) and isinstance(actual, dict):
        diffs = []
        for key in expected.keys() | actual.keys():
            child = f"{path}.{key}" if path else str(key)
            if key not in actual:
                diffs.append(f"{child}: missing from actual")
            elif key not in expected:
                diffs.append(f"{child}: unexpected key in actual")
            else:
                diffs.extend(compare_results(expected[key], actual[key], child))
        return sorted(diffs)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{label}: length {len(expected)} != {len(actual)}"]
        return [
            diff
            for i, (e, a) in enumerate(zip(expected, actual, strict=True))
            for diff in compare_results(e, a, f"{path}[{i}]")
        ]
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float)) and isinstance(actual, (int, float))
    ):
        return [f"{label}: type {type(expected).__name__} != {type(actual).__name__}"]
    if expected != actual:
        return [f"{label}: {expected!r} != {actual!r}"]
    return []


def log(message: str) -> None:
    print(f"[capture] {message}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="folder holding the full Chengdu CSVs (default: repo parent, or STDVRP_DATA_DIR)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=None,
        help="override number of training episodes (seeds 1000..)",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        default=None,
        help="override number of evaluation seeds (100000..)",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        default=None,
        help="override number of test seeds (legacy table prefix)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=default_cache_path(),
        help="pickle cache of the loaded world (default: %(default)s, or STDVRP_GOLDEN_CACHE)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="always rebuild the world from the CSVs and do not write the cache",
    )
    args = parser.parse_args()

    missing = missing_data_files(args.data_dir)
    if missing:
        log(
            f"cannot capture: {len(missing)} data files missing under {args.data_dir} "
            f"(first: {missing[0]})"
        )
        return 1

    protocol = default_protocol()
    if args.train_episodes is not None:
        protocol["train_seeds"] = list(range(1000, 1000 + args.train_episodes))
    if args.eval_seeds is not None:
        protocol["eval_seeds"] = list(range(100000, 100000 + args.eval_seeds))
    if args.test_seeds is not None:
        protocol["test_seeds"] = TEST_SEEDS_150[: args.test_seeds]
        protocol["test_vehicles"] = TEST_VEHICLES_150[: args.test_seeds]

    t0 = time.perf_counter()
    cache_path = None if args.no_cache else args.cache_path
    document = capture_document(args.data_dir, protocol, cache_path)
    write_json(args.out, document)
    log(f"golden master written to {args.out} ({time.perf_counter() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
