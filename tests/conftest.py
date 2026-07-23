"""Shared fixtures for the test suite."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import characterization_world

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str) -> ModuleType:
    """Load ``scripts/<name>.py`` by file path (scripts/ is not an importable package).

    Registered in ``sys.modules`` before exec so dataclasses under
    ``from __future__ import annotations`` can resolve their own module.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def capture() -> ModuleType:
    """The golden-master capture module (scripts/ is not an importable package)."""
    return _load_script_module("capture_golden_master")


@pytest.fixture(scope="session")
def benchmark_module() -> ModuleType:
    """scripts/benchmark_episodes.py — the episode benchmark and projection (ticket 01)."""
    return _load_script_module("benchmark_episodes")


# --- Legacy-monolith characterization venue (see characterization_world) ---


@pytest.fixture(scope="module")
def legacy_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return characterization_world.build_legacy_world(tmp_path_factory.mktemp("legacy_world"))


@pytest.fixture(scope="module")
def legacy_module() -> ModuleType:
    return characterization_world.load_legacy_module()
