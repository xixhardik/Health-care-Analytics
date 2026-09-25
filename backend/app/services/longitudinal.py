"""Longitudinal demonstration cases.

The SPIDER dataset this project is built on is **cross-sectional**: one timepoint
per patient, no postoperative imaging, no surgical record, no follow-up
assessment. A recovery tracker therefore cannot be validated on it, and nothing
here claims otherwise.

What this module does is load *demonstration* cases from ``demo/longitudinal_cases/``.
A case arranges several real SPIDER studies - **different studies from different
patients** - into an ordered timeline so the workflow can be shown end to end.
The imaging is real. The timeline is not.

Three rules this module exists to enforce:

1. **Provenance survives.** Every stage carries the study it actually came from,
   its split, and an explicit ``is_true_followup: false``. A consumer cannot
   render a stage without also having the fact that it is not a follow-up.
2. **Nothing is substituted.** If a stage's referenced volume is missing on this
   machine, the stage is reported unavailable with a reason. It never silently
   falls back to another scan, and there is no synthetic imaging anywhere.
3. **Paths stay inside the dataset directory.** A manifest is data, so its
   ``volume_path`` is validated against the project root before use rather than
   trusted.

No manifest may describe itself as real longitudinal data: ``load_case`` rejects
a manifest whose ``type`` is not ``SIMULATED_LONGITUDINAL_DEMO`` or which sets
``is_true_followup`` true.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("lumbar.api")

#: The only case type this loader will serve.
REQUIRED_CASE_TYPE = "SIMULATED_LONGITUDINAL_DEMO"

#: Shown by the API and rendered verbatim by the interface.
UNAVAILABLE_NOTICE = "Longitudinal follow-up unavailable for this study."

SINGLE_STUDY_NOTICE = "Single-study analysis"

#: Research-oriented status vocabulary. Deliberately descriptive rather than
#: diagnostic: none of these is a clinical assessment of a patient.
RESEARCH_STATUSES = (
    "Baseline",
    "Early postoperative",
    "Recovery monitoring",
    "Final follow-up demonstration",
    "Requires review",
    "Follow-up data unavailable",
)

#: Used where imaging findings are present. Phrased as a prompt for expert
#: review, never as a recommendation to a patient.
SURGICAL_EVALUATION_WORDING = (
    "Surgical evaluation may be appropriate based on detected imaging findings."
)


class ManifestError(ValueError):
    """A manifest is malformed, or claims to be something it must not."""


def cases_root(project_root: Path) -> Path:
    return project_root / "demo" / "longitudinal_cases"


def list_case_ids(project_root: Path) -> list[str]:
    """Case ids that have a readable manifest, in stable order."""
    root = cases_root(project_root)
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "manifest.json").is_file():
            found.append(child.name)
    return found


def _read_manifest(project_root: Path, case_id: str) -> dict[str, Any]:
    # `case_id` arrives from the URL, so it is checked before it is used to build
    # a path rather than after.
    if not case_id or not all(c.isalnum() or c in "-_" for c in case_id):
        raise ManifestError(f"Invalid case id {case_id!r}.")
    path = cases_root(project_root) / case_id / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No manifest for case {case_id!r}.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest for {case_id!r} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"Manifest for {case_id!r} is not an object.")
    return manifest


def resolve_volume(project_root: Path, volume_path: str) -> Path | None:
    """Resolve a manifest volume reference, or ``None`` if it is unusable.

    The path must stay inside the project's dataset directory. A manifest is
    data, and data that escapes its directory is a path-traversal attempt
    whatever the intent behind it.
    """
    if not volume_path:
        return None
    candidate = (project_root / volume_path).resolve()
    allowed = (project_root / "data").resolve()
    try:
        candidate.relative_to(allowed)
    except ValueError:
        logger.warning("Rejected demo volume reference outside data/: %s", volume_path)
        return None
    if not candidate.is_file() or candidate.stat().st_size < 2048:
        return None
    return candidate


def _stage_view(project_root: Path, stage: dict[str, Any]) -> dict[str, Any]:
    """One stage, shaped for the API, with provenance attached inseparably."""
    volume_path = str(stage.get("volume_path") or "")
    resolved = resolve_volume(project_root, volume_path)
    available = resolved is not None

    return {
        "order": int(stage.get("order", 0)),
        "stage_id": str(stage.get("stage_id") or ""),
        "label": str(stage.get("label") or ""),
        "research_status": str(stage.get("research_status") or "Baseline"),
        # Non-identifying label the interface shows by default.
        "display_reference": str(stage.get("display_reference") or ""),
        "stage_note": stage.get("stage_note"),
        "scanner": stage.get("scanner"),
        # Genuine acquisition time is not available for these studies, so this is
        # null rather than invented.
        "acquisition_timestamp": stage.get("acquisition_timestamp"),
        "available": available,
        "unavailable_reason": (
            None if available else
            f"The source study for this stage is not present on this machine "
            f"({volume_path}). Extract the dataset with scripts/01_extract.py. "
            f"No substitute scan is used."
        ),
        "provenance": {
            "source_study_id": str(stage.get("source_study_id") or ""),
            "source_patient_id": stage.get("source_patient_id"),
            "source_dataset": str(stage.get("source_dataset") or "SPIDER"),
            "source_split": stage.get("source_split"),
            "is_true_followup": False,
            "measurement_source": str(
                stage.get("measurement_source") or "real_pipeline"
            ),
        },
        "expected": stage.get("expected") or {},
    }


def load_case(project_root: Path, case_id: str) -> dict[str, Any]:
    """Load one demonstration case, with its stages ordered.

    Raises ``ManifestError`` if the manifest does not declare itself a simulated
    demonstration. A loader that would serve a manifest claiming real
    longitudinal follow-up is a loader that can misrepresent the dataset, so this
    refuses rather than warns.
    """
    manifest = _read_manifest(project_root, case_id)

    declared = str(manifest.get("type") or "")
    if declared != REQUIRED_CASE_TYPE:
        raise ManifestError(
            f"Case {case_id!r} declares type {declared!r}. Only "
            f"{REQUIRED_CASE_TYPE!r} is served, because this dataset contains no "
            f"true longitudinal follow-up."
        )
    if manifest.get("is_true_followup") is True or manifest.get("is_same_patient") is True:
        raise ManifestError(
            f"Case {case_id!r} claims to be real follow-up data for one patient. "
            f"The SPIDER dataset contains no such data and this is refused."
        )

    raw_stages = manifest.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ManifestError(f"Case {case_id!r} defines no stages.")

    stages = [_stage_view(project_root, s) for s in raw_stages if isinstance(s, dict)]
    if not stages:
        raise ManifestError(f"Case {case_id!r} defines no usable stages.")
    # Ordered by the manifest's own `order`, so timeline position never depends on
    # filesystem or JSON key ordering.
    stages.sort(key=lambda s: s["order"])

    return {
        "case_id": str(manifest.get("case_id") or case_id),
        "type": REQUIRED_CASE_TYPE,
        "title": str(manifest.get("title") or case_id),
        "is_true_followup": False,
        "is_same_patient": False,
        "source_dataset": str(manifest.get("source_dataset") or "SPIDER"),
        "disclaimer": str(manifest.get("disclaimer") or ""),
        "ui_notice": str(manifest.get("ui_notice") or ""),
        "interpretation": str(manifest.get("interpretation") or ""),
        "measurement_policy": manifest.get("measurement_policy") or {},
        "selection": manifest.get("selection") or {},
        "stage_count": len(stages),
        "available_stage_count": sum(1 for s in stages if s["available"]),
        "stages": stages,
    }


def load_cases(project_root: Path) -> list[dict[str, Any]]:
    """Every readable demonstration case. A broken manifest is skipped, not fatal."""
    cases = []
    for case_id in list_case_ids(project_root):
        try:
            cases.append(load_case(project_root, case_id))
        except (ManifestError, FileNotFoundError) as exc:
            logger.warning("Skipping demo case %s: %s", case_id, exc)
    return cases


def get_stage(project_root: Path, case_id: str, stage_id: str) -> dict[str, Any]:
    """One stage of one case.

    Raises ``KeyError`` when the stage is not part of the case, so the router can
    answer with a list of the valid ids rather than a bare 404.
    """
    case = load_case(project_root, case_id)
    for stage in case["stages"]:
        if stage["stage_id"] == stage_id:
            return stage
    raise KeyError(stage_id)


def stage_ids(case: dict[str, Any]) -> list[str]:
    return [s["stage_id"] for s in case["stages"]]
