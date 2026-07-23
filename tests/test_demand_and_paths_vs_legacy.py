"""Exact-equality characterization: ShortestPathCache vs the unmodified legacy
class (read-only import, ADR-0001).

The path cache comparison loads the same fixture CSV through both loaders and
requires the resulting mappings to be bit-identical, float node ids included.

The former ``ClientGenerator`` bit-exact comparison retired with ticket 13 (RNG
modernization, ADR-0001 phase 2): ``ClientGenerator.generate`` now draws from an
``np.random.Generator`` instead of the legacy's global ``random`` stream, so
exact-value equality with the legacy is no longer expected or asserted.
"""

import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

from stdvrp.network import ShortestPathCache

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chengdu_mini"


def test_shortest_path_cache_is_bit_identical(legacy_module: ModuleType) -> None:
    previous_cwd = os.getcwd()
    os.chdir(FIXTURE_DIR)  # the legacy loads "all_shortest_paths.csv" from the CWD
    try:
        legacy = legacy_module.shortest_path_memory(SimpleNamespace(node_list=[]))
    finally:
        os.chdir(previous_cwd)

    cache = ShortestPathCache.from_csv(FIXTURE_DIR / "all_shortest_paths.csv")

    assert list(cache.as_dict()) == list(legacy.shortest_paths)
    assert cache.as_dict() == legacy.shortest_paths
