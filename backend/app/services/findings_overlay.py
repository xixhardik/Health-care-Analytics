"""Which disc regions the findings overlay may mark, and on what grounds.

This module is the single place that decides whether a disc qualifies for the red
"finding-associated" overlay. It exists so the rule cannot drift between the
renderer, the API response and the interface: the browser is never asked to
re-derive it.

What the overlay is, precisely
------------------------------
The red region is the **existing validated disc segmentation** for a disc that
carries a supported, positive, model-estimated binary finding. It marks a *disc
region associated with a model-estimated finding*.

What it is not
--------------
There is no pixel-level pathology model anywhere in this system. No pixel inside
a red region has been classified, and the shape of the red region carries no
information about where within the disc anything is. It is the disc outline,
coloured. Any reading of it as tissue-level localisation would be wrong.
"""

from __future__ import annotations

#: Binary disc-level targets that may trigger the overlay.
#:
#: Each is served by a persisted estimator with a published held-out test PR-AUC
#: (see ``outputs/models/findings/manifest.json``). Deliberate exclusions:
#:
#: * ``pfirrmann_grade`` - an ordinal grade, not a positive/negative finding.
#:   Grade 2 is not "positive" anything, so colouring a disc by grade would mean
#:   inventing a severity threshold that was never validated.
#: * ``spondylolisthesis`` - not served; 5 positive discs in the test split.
#: * ``modic_type`` - not served; the nominal type was never modelled.
OVERLAY_ELIGIBLE_FINDINGS: tuple[str, ...] = (
    "narrowing",
    "bulging",
    "herniation",
    "any_modic",
    "upper_endplate",
    "lower_endplate",
)

#: Rendered verbatim by the UI and the API so the wording cannot drift.
OVERLAY_NOTE = (
    "The red overlay marks disc regions associated with a model-estimated "
    "finding. It is drawn from the validated disc segmentation for that disc, "
    "not from a pixel-level pathology model, and it is not a diagnosis of "
    "damaged tissue."
)

OVERLAY_LABEL = "Model-estimated finding"


def _has_segmentation_region(disc: dict) -> bool:
    """True when this disc actually has a segmented region to colour.

    A finding with no region behind it must not produce an overlay entry, because
    there would be nothing honest to draw.
    """
    if (disc.get("slices_present") or 0) > 0:
        return True
    area = (disc.get("measurements") or {}).get("area_mm2")
    return area is not None and float(area) > 0.0


def qualifies(finding: dict) -> bool:
    """Whether one finding estimate may turn its disc red.

    All four conditions must hold. Each rules out a specific way of overstating
    what the system knows:

    1. the target is one the served estimators actually cover
    2. it is not unavailable - an absent model is not a negative finding
    3. it came from a model prediction, not an unsupported placeholder
    4. the value is boolean ``True`` - never a grade, a probability or a truthy
       number
    """
    if finding.get("name") not in OVERLAY_ELIGIBLE_FINDINGS:
        return False
    if finding.get("unavailable_reason"):
        return False
    if finding.get("source") != "model_prediction":
        return False
    value = finding.get("value")
    # `isinstance(value, bool)` matters: an ordinal grade of 1 must never pass as
    # a positive finding.
    return isinstance(value, bool) and value is True


def disc_finding_labels(disc: dict) -> list[str]:
    """Labels of the supported positive findings on one disc, in served order."""
    labels: list[str] = []
    for finding in disc.get("findings") or []:
        if not qualifies(finding):
            continue
        labels.append(str(finding.get("label") or finding.get("name")))
    return labels


def overlay_discs(result: dict) -> dict[int, list[str]]:
    """Map disc index -> positive finding labels, for discs the overlay may mark.

    A disc is absent from the result when it has no qualifying finding or no
    segmented region. Nothing is inferred beyond the served findings.
    """
    marked: dict[int, list[str]] = {}
    for disc in result.get("discs") or []:
        index = disc.get("index")
        if not isinstance(index, int):
            continue
        if not _has_segmentation_region(disc):
            continue
        labels = disc_finding_labels(disc)
        if labels:
            marked[index] = labels
    return marked


def overlay_summary(result: dict) -> dict:
    """The overlay's decision, shaped for the API response.

    Served so the interface can mark the same discs the renderer colours without
    re-implementing the rule client-side.
    """
    marked = overlay_discs(result)
    return {
        "disc_indices": sorted(marked),
        "discs": [
            {"index": index, "findings": marked[index]}
            for index in sorted(marked)
        ],
        "eligible_findings": list(OVERLAY_ELIGIBLE_FINDINGS),
        "label": OVERLAY_LABEL,
        "note": OVERLAY_NOTE,
    }
