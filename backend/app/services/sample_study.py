"""The bundled sample study, for demonstration.

"Load Sample Study" exists so a demonstration does not depend on finding a file.
It is not a mock: the volume is the same one the end-to-end smoke test validates,
it goes through the identical pipeline, and the analysis it produces is
indistinguishable from an upload except for an ``is_sample`` flag used to label it
honestly in the interface.

No result is ever fabricated. If the volume is not present on this machine the API
says so and the UI hides the button, rather than falling back to canned output.
"""

from __future__ import annotations

from pathlib import Path

from backend.app.config import Settings

#: Candidate locations, in preference order. The first is the volume the smoke
#: test uses; the others let a checkout with a differently-extracted dataset still
#: offer the demo.
CANDIDATES = (
    Path("data/extracted/images/33_t2.mha"),
    Path("data/extracted/images/33_t1.mha"),
    Path("data/extracted/images/1_t2.mha"),
)

SAMPLE_LABEL = "Sample research study"

SAMPLE_NOTE = (
    "This is the bundled sample research study from the SPIDER dataset, provided "
    "so the pipeline can be demonstrated without locating a file. It is processed "
    "by exactly the same pipeline as an upload; no result is pre-computed."
)


def sample_study_path(settings: Settings) -> Path | None:
    """Return the sample volume, or None when it is not available here."""
    for relative in CANDIDATES:
        candidate = settings.project_root / relative
        if candidate.exists() and candidate.stat().st_size > 2048:
            return candidate
    return None
