"""Shared fixtures for the test suite."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import characterization_world

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def capture() -> ModuleType:
    """The golden-master capture module (scripts/ is not an importable package)."""
    spec = importlib.util.spec_from_file_location(
        "capture_golden_master", REPO_ROOT / "scripts" / "capture_golden_master.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Episode characterization venue (tickets 07/08; see characterization_world) ---


@pytest.fixture(scope="module")
def legacy_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return characterization_world.build_legacy_world(tmp_path_factory.mktemp("legacy_world"))


@pytest.fixture(scope="module")
def legacy_module() -> ModuleType:
    return characterization_world.load_legacy_module()


@pytest.fixture(scope="module")
def legacy_calc(legacy_module: ModuleType, legacy_world: Path) -> Any:
    return characterization_world.build_legacy_calc(legacy_module, legacy_world)


@pytest.fixture(scope="module")
def legacy_spm(legacy_module: ModuleType) -> Any:
    return characterization_world.build_legacy_spm(legacy_module)


@pytest.fixture(scope="module")
def ported_world(legacy_world: Path) -> dict[str, Any]:
    return characterization_world.build_ported_world(legacy_world)
