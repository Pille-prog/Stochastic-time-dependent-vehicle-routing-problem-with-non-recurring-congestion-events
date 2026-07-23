"""Shared venue for the legacy-monolith characterization tests.

The unmodified legacy classes run on a temporary world of 44 fixture-day copies
(the legacy hardcodes 44 traffic days; single-day speed std is NaN, 44 identical
copies make it exactly 0.0). Constants and builders live here; the module-scoped
fixtures wiring them together live in ``conftest.py``. Tests with a different
world shape (e.g. the ticket 05 travel-time characterization) keep their own
fixtures.

The Episode-level bit-exact comparisons (tickets 07/08) retired with ticket 13
(RNG modernization, ADR-0001 phase 2): exact equality with the legacy cannot
survive the switch from its shared global RNG streams to injected per-concern
``np.random.Generator`` instances. ``FIXTURE_DIR``/``LEGACY_DAYS``/
``build_legacy_world``/``load_legacy_module`` remain — they still back the
data-spine characterization tests and the Trainer smoke test's fixture world.
"""

import builtins
import importlib.util
import os
import shutil
from pathlib import Path
from types import ModuleType

from legacy_source import legacy_script_path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"
LEGACY_DAYS = tuple(range(601, 631)) + tuple(range(701, 715))


def build_legacy_world(world: Path) -> Path:
    """Populate ``world`` with link.csv plus the 44 fixture-day speed copies."""
    shutil.copyfile(FIXTURE_DIR / "link.csv", world / "link.csv")
    for day in LEGACY_DAYS:
        for half in (0, 1):
            shutil.copyfile(
                FIXTURE_DIR / f"speed[601]_[{half}].csv",
                world / f"speed[{day}]_[{half}].csv",
            )
    return world


def load_legacy_module() -> ModuleType:
    """Import the monolith unchanged (read-only reference, ADR-0001)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location("legacy_monolith", legacy_script_path())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Imported modules get a dict __builtins__; the legacy transition_function
    # calls the literal ``__builtins__.min`` (script-style), so restore the module.
    module.__dict__["__builtins__"] = builtins
    return module
