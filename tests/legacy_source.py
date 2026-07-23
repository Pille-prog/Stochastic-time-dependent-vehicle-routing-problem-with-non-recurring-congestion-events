"""Single source of truth for reading the legacy monolith from the git tag.

Ticket 14 removed the script from the working tree; the ``legacy-monolith`` tag is
its permanent home (ADR-0001). Both the characterization tests (via
``characterization_world``) and ``scripts/capture_golden_master.py`` (which loads
this module by file path — ``scripts/`` is not importable from here, nor tests/
from a standalone script run) extract the source through these helpers.

Stdlib-only on purpose: the capture script must stay independent of the ported
package, so nothing here may import ``stdvrp``.
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_TAG = "legacy-monolith"
LEGACY_FILENAME = "Main_Chengdu_Sirve_2_Acciones_Sin_Algunas_Variables.py"

_legacy_source_cache: bytes | None = None


def read_legacy_source() -> bytes:
    """The monolith's bytes, read from the tag once per process."""
    global _legacy_source_cache
    if _legacy_source_cache is None:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{LEGACY_TAG}:{LEGACY_FILENAME}"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cannot read {LEGACY_FILENAME} from the {LEGACY_TAG} tag — "
                f"shallow clone without tags? Run `git fetch --tags`. "
                f"git said: {result.stderr.decode(errors='replace').strip()}"
            )
        _legacy_source_cache = result.stdout
    return _legacy_source_cache


def legacy_sha256() -> str:
    return hashlib.sha256(read_legacy_source()).hexdigest()


def legacy_script_path() -> Path:
    """A real on-disk copy of the monolith (importlib/inspect need a file path).

    Content-addressed under the system temp dir, so repeat runs reuse one copy
    instead of leaking a directory per process; the tag is frozen, so a hit can
    never be stale.
    """
    path = Path(tempfile.gettempdir()) / f"stdvrp-legacy-{legacy_sha256()[:12]}" / LEGACY_FILENAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(read_legacy_source())
    return path
