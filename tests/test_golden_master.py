"""Golden-master equality test: the legacy monolith, re-run, must match exactly.

Excluded from default runs (marker ``golden``, deselected via addopts) because it
needs the full local Chengdu dataset and takes minutes. Run it with:

    uv run pytest -m golden

It re-executes the capture protocol stored in the golden-master file against the
untouched legacy script and requires bit-exact float equality (ADR-0001). Skips
when the local dataset is absent (e.g. CI).
"""

import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_master" / "chengdu_full.json"

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip("golden master not captured yet (scripts/capture_golden_master.py)")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data_dir(capture: ModuleType) -> Path:
    directory = capture.default_data_dir()
    missing = capture.missing_data_files(directory)
    if missing:
        pytest.skip(
            f"full Chengdu dataset not available under {directory} "
            f"({len(missing)} files missing, first: {missing[0]})"
        )
    return directory


def test_dataset_matches_captured_signature(
    capture: ModuleType, golden: dict, data_dir: Path
) -> None:
    """If this fails, the data changed — value mismatches below would be explained."""
    assert capture.data_signature(data_dir) == golden["meta"]["data_signature"]


def test_legacy_script_unchanged_since_capture(capture: ModuleType, golden: dict) -> None:
    current = capture.legacy_sha256()
    assert current == golden["meta"]["legacy_sha256"], (
        "the legacy-monolith tag no longer matches the golden-master capture; "
        "the tag must stay frozen forever (ADR-0001)"
    )


def test_rerun_reproduces_golden_master_exactly(
    capture: ModuleType, golden: dict, data_dir: Path
) -> None:
    legacy = capture.load_legacy()
    # The world cache only skips re-parsing the CSVs (state-identical objects);
    # delete the cache file to force a fully cold verification.
    fresh = capture.to_jsonable(
        capture.run_capture(
            legacy, data_dir, golden["protocol"], cache_path=capture.default_cache_path()
        )
    )
    expected = {"training": golden["training"], "test": golden["test"]}
    diffs = capture.compare_results(expected, fresh)
    assert not diffs, "golden master mismatch:\n" + "\n".join(diffs[:50])
