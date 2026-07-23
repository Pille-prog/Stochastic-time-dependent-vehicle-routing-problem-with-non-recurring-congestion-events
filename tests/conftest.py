"""Shared fixtures for the test suite."""

import importlib.util
from pathlib import Path
from types import ModuleType

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


# --- Legacy-monolith characterization venue (see characterization_world) ---


@pytest.fixture(scope="module")
def legacy_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return characterization_world.build_legacy_world(tmp_path_factory.mktemp("legacy_world"))


@pytest.fixture(scope="module")
def legacy_module() -> ModuleType:
    return characterization_world.load_legacy_module()
