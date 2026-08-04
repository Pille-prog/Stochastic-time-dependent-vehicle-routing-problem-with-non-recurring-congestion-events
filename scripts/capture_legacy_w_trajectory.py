"""Capture the reference monolith's W after every training episode.

Built for ``.scratch/linear-policy-learning/issues/01-legacy-w-trajectory-diff.md``.
Every comparison in that effort so far has been between two black-box cost
totals; this is the first that looks *inside* W. It writes one 24-component
vector per training episode to JSON, for
:mod:`scripts.compare_w_trajectories` to diff against ours.

**The reference here is not the golden master's monolith.**
``scripts/capture_golden_master.py`` loads
``Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py`` from the
``legacy-monolith`` tag (ADR-0001). The script this effort is chasing is
``Main_Chengdu_Sirve_Con_Clip.py`` — the one that *does* learn, named for the
update-time feature clip at its ``actualize_W`` line 4374. It is not in the
repo and not in any tag; it lives beside the Chengdu CSVs, so ``--legacy-script``
defaults to ``<data-dir>/Main_Chengdu_Sirve_Con_Clip.py``. Both carry the same
24-component vector (16 general + 8 state-action, verified append-by-append
against ``feature_extraction``'s layout), so the per-component diff is index-wise
meaningful.

The shim is ticket 04's, reused rather than re-implemented: ``load_legacy`` and
``load_world`` are ``capture_golden_master``'s own, both now taking the script
path and its sha as arguments so the ``__builtins__`` restore, the
``legacy_monolith`` registration the world pickle depends on, and the
transcription of ``main()``'s data load have one definition between the two
drivers. This module only supplies the cwd-into-``data_dir`` redirection (the
legacy reads relative CSV paths) and its own cache location, keyed by the
Con_Clip sha so the two drivers never evict each other. The cold load is ~13
minutes and happens once; a warm run loads in ~15s.

What is mirrored is ``training_and_testing.training_model``'s **training loop
only**: seeds from 1000, the 1e-6 warm-up learning rate on the first episode,
``number_actions_train = vehicles + 2``. Its periodic evaluation block is
deliberately *not* run — it costs 50 test episodes per block and this ticket
asks about W, not cost. The legacy's evaluation costs at these parameters are
already recorded in ``spec.md``.

Skipping it costs no fidelity, which is worth stating rather than assuming:
every training episode re-seeds *both* global streams before it draws anything
(``client_generator_function`` opens with ``random.seed(random_seed)``, the loop
body follows with ``np.random.seed(random_seed)``), so whatever the evaluation
block would have consumed cannot reach the next training episode. The trajectory
below is the one the full ``training_model`` produces.

Usage::

    .venv/Scripts/python.exe -u scripts/capture_legacy_w_trajectory.py \\
        --episodes 200 --out .scratch/linear-policy-learning/legacy_w.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import platform
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_golden_master as golden  # noqa: E402
from compare_w_trajectories import TrajectoryLog  # noqa: E402

DEFAULT_LEGACY_NAME = "Main_Chengdu_Sirve_Con_Clip.py"

# ``experiments/chengdu/config_linear_congestion_10k.yaml``, expressed as the
# legacy's own argv fields. The equivalent legacy invocation is
# ``"<episodes>,50,0.001,0.1,0.1,0.4,120,150,150,1"`` -- the two 150s are
# ``mean_number_clients`` and ``diff_TW``, which are unrelated and easy to
# misread as one.
DEFAULT_PROTOCOL: dict[str, Any] = {
    "learning_rate": 0.001,
    "epsilon": 0.1,
    "congestion_lower_bound": 0.1,
    "congestion_upper_bound": 0.4,
    "max_congestion_duration": 120,
    "mean_number_clients": 150,
    # The reference runs' time-window width: ``capture_golden_master.py``'s
    # protocol and the ``Tw150`` in the legacy's own result filenames. The repo's
    # Chengdu configs said 60 until this ticket found the mismatch; the two are
    # not comparable, and the legacy's own stored costs only line up at 150.
    "diff_TW": 150,
    "horizon_start_time": 300,
    "horizon_end_time": 780,
    "n_arcs": 3,
    "warmup_learning_rate": 1e-6,
    # The legacy policy's two exploration RNGs are unseeded ``random.Random()``
    # instances (Con_Clip lines 1677-78), so training differs on every run.
    # Ticket 04's convention: seed them per episode right after constructing the
    # policy, at driver level, without touching the legacy file.
    "train_exploration_seed_offset": 10_000_000,
    "train_repair_seed_offset": 20_000_000,
    "first_train_seed": 1000,
}


def script_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_cache_path(legacy_sha: str) -> Path:
    """Per-script cache, kept away from the golden master's own.

    The golden capture's cache holds a *different* monolith's objects at a
    different ``max_congestion_duration``; sharing one path would make the two
    evict each other on every alternating run.
    """
    env = os.environ.get("STDVRP_LEGACY_W_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or "/tmp"
    return Path(base) / "stdvrp" / f"legacy_w_world_{legacy_sha[:12]}.pkl"


def capture_trajectory(
    legacy: ModuleType,
    data_dir: Path,
    legacy_sha: str,
    protocol: dict[str, Any],
    episodes: int,
    cache: Path | None,
    checkpoint: Path | None = None,
) -> TrajectoryLog:
    """Run ``episodes`` training episodes, returning W and the cost after each.

    ``checkpoint`` is written after every episode: a 200-episode run is hours
    long, and a partial trajectory still answers the ticket.
    """
    np = legacy.np

    n_arcs = protocol["n_arcs"]
    horizon_start_time = protocol["horizon_start_time"]
    horizon_end_time = protocol["horizon_end_time"]
    epsilon = protocol["epsilon"]
    lower = protocol["congestion_lower_bound"]
    upper = protocol["congestion_upper_bound"]
    duration = protocol["max_congestion_duration"]
    mean_number_clients = protocol["mean_number_clients"]
    diff_TW = protocol["diff_TW"]

    with contextlib.chdir(data_dir):
        data_calculations, spm = golden.load_world(legacy, data_dir, protocol, cache, legacy_sha)

        w = None
        lr = protocol["warmup_learning_rate"]
        trajectory = TrajectoryLog()

        for index in range(episodes):
            random_seed = protocol["first_train_seed"] + index
            t_ep = time.perf_counter()
            random_depot = 0

            cg = legacy.ClientGenerator(random_depot)
            cg.client_generator_function(
                random_seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time
            )
            np.random.seed(random_seed)
            number_vehicles = int(cg.number_vehicles)
            clients = cg.client_list
            number_clients = len(clients)

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
                lower,
                upper,
                number_vehicles + 2,
                number_vehicles + 2,
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
                lower,
                upper,
                duration,
            )
            # Make the unseeded exploration RNGs reproducible (see DEFAULT_PROTOCOL).
            p.local_rng.seed(protocol["train_exploration_seed_offset"] + random_seed)
            p.local_rng_2.seed(protocol["train_repair_seed_offset"] + random_seed)
            lr = protocol["learning_rate"]

            m.create_monte_carlo_episode_train()
            w = m.policy.W
            trajectory.record(w, m.total_cost, number_vehicles, number_clients)
            golden.log(
                f"episode {index + 1:4d} (seed {random_seed}): cost={m.total_cost:.2f} "
                f"|W|={float(np.linalg.norm(w)):.2f} ({time.perf_counter() - t_ep:.1f}s)"
            )
            if checkpoint is not None:
                trajectory.checkpoint(checkpoint)

    return trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, default=golden.default_data_dir())
    parser.add_argument(
        "--legacy-script",
        type=Path,
        default=None,
        help=f"the reference monolith (default: <data-dir>/{DEFAULT_LEGACY_NAME})",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / ".scratch" / "linear-policy-learning" / "legacy_w_trajectory.json",
    )
    parser.add_argument(
        "--diff-tw",
        type=int,
        default=None,
        help=(
            f"the legacy's time-window width (default {DEFAULT_PROTOCOL['diff_TW']}). "
            "Pass 60 to reproduce the superseded measurements this repo's configs "
            "used to produce."
        ),
    )
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # The capture runs inside ``contextlib.chdir(data_dir)`` -- the legacy reads
    # relative CSV paths -- so a relative --out would put the per-episode
    # checkpoints next to the Chengdu data instead of in the repo. Resolve every
    # path against the real cwd first, while it is still the real cwd.
    out = args.out.resolve()
    data_dir = args.data_dir.resolve()

    legacy_path = args.legacy_script or (data_dir / DEFAULT_LEGACY_NAME)
    if not legacy_path.is_file():
        golden.log(f"cannot capture: no legacy script at {legacy_path}")
        return 1
    missing = golden.missing_data_files(data_dir)
    if missing:
        golden.log(
            f"cannot capture: {len(missing)} data files missing under {data_dir} "
            f"(first: {missing[0]})"
        )
        return 1

    legacy_sha = script_sha256(legacy_path)
    cache = None
    if not args.no_cache:
        cache = (args.cache_path or default_cache_path(legacy_sha)).resolve()

    protocol = dict(DEFAULT_PROTOCOL)
    if args.diff_tw is not None:
        protocol["diff_TW"] = args.diff_tw
    t0 = time.perf_counter()
    legacy = golden.load_legacy(legacy_path)
    trajectory = capture_trajectory(
        legacy,
        data_dir,
        legacy_sha,
        protocol,
        args.episodes,
        cache,
        checkpoint=out.with_suffix(".partial.json"),
    )
    trajectory.write(
        out,
        meta={
            "side": "legacy",
            "legacy_script": legacy_path.name,
            "legacy_sha256": legacy_sha,
            "data_signature": golden.data_signature(data_dir),
            "versions": {
                "python": platform.python_version(),
                "numpy": legacy.np.__version__,
                "pandas": legacy.pd.__version__,
            },
        },
        protocol=protocol,
    )
    golden.log(f"W trajectory written to {out} ({time.perf_counter() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
