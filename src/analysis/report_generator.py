"""Stage F/G - structured per-disc finding reports.

Naming
------
Called a **"Radiological Finding Assessment" / "Automated MRI Finding Report"**,
never a diagnosis. Every line carries its provenance, one of:

``dataset_annotation``
    The value recorded in ``radiological_gradings.csv`` by the human graders.
``model_prediction``
    Estimated by the Stage E baseline, with its probability.
``measured_image_feature``
    Computed geometrically from the segmentation mask in millimetres.

No composite severity or "damage percentage" is produced. The dataset grades
eight findings independently and combining them into one number would be a
clinical judgement this project has no basis for.

A note on level naming
----------------------
The dataset numbers discs from the bottom up: disc 1 is the most inferior. It
does **not** state which vertebra is L5 - the published description says the
lowest annotated vertebra is usually L5 but can also be L4 or L6. Conventional
names such as "L5-S1" are therefore emitted as
:data:`PROVISIONAL_LEVEL_NAMES` and always flagged
``level_name_is_provisional: true``. The authoritative identifier in every
record is the integer ``ivd_label``.

Longitudinal readiness (Stage G)
--------------------------------
Every record carries the keys a future comparison would join on:
``patient_id``, ``series_id``, ``ivd_label``, plus ``acquisition_timestamp`` and
``timepoint_label``. For this dataset both time fields are **explicitly null** -
it contains no acquisition date and one study per patient. They are present as
schema, not as invented values.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

#: Schema version, so a later sprint can detect format changes.
RECORD_SCHEMA_VERSION = "1.0"

#: Conventional level names under the assumption that the lowest annotated
#: vertebra is L5. PROVISIONAL - see the module docstring.
PROVISIONAL_LEVEL_NAMES: dict[int, str] = {
    1: "L5-S1",
    2: "L4-L5",
    3: "L3-L4",
    4: "L2-L3",
    5: "L1-L2",
    6: "T12-L1",
    7: "T11-T12",
    8: "T10-T11",
    9: "T9-T10",
}

#: Human-readable labels for the binary findings.
BINARY_FINDING_LABELS: dict[str, str] = {
    "bulging": "Disc bulging",
    "narrowing": "Disc narrowing",
    "herniation": "Disc herniation",
    "spondylolisthesis": "Spondylolisthesis",
    "any_modic": "Modic change (any type)",
    "upper_endplate": "Upper endplate defect / Schmorl's node",
    "lower_endplate": "Lower endplate defect / Schmorl's node",
}

MODIC_TYPE_LABELS: dict[int, str] = {
    0: "none",
    1: "type I",
    2: "type II",
    3: "type III",
}

#: Measurements included in the report, with units and display names.
REPORTED_MEASUREMENTS: list[tuple[str, str, str]] = [
    ("height_mm_central", "Disc height (central)", "mm"),
    ("height_mm_anterior", "Disc height (anterior)", "mm"),
    ("height_mm_posterior", "Disc height (posterior)", "mm"),
    ("area_mm2", "Disc cross-sectional area", "mm2"),
    ("ap_extent_mm", "Antero-posterior extent", "mm"),
    ("height_ratio_to_series_median", "Height relative to this spine's median", "ratio"),
    ("disc_to_vertebra_height_ratio", "Disc-to-vertebra height ratio", "ratio"),
    ("height_ap_asymmetry", "Anterior-posterior height asymmetry", "ratio"),
    ("vertebral_ap_offset_mm", "Vertebral antero-posterior offset", "mm"),
    ("canal_width_at_disc_mm", "Spinal canal width at this level", "mm"),
]

DISCLAIMER = (
    "This is an automated radiological finding assessment produced for research "
    "and review. It is NOT a medical diagnosis, and it must not be used as one. "
    "Every value is traceable to its source: a dataset annotation made by human "
    "graders, a measurement computed from the segmentation mask, or a baseline "
    "model estimate with its probability. No composite severity score or "
    "'damage percentage' is produced."
)


def _clean(value):
    """Convert numpy/NaN values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 5)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_disc_record(
    row: pd.Series,
    *,
    predictions: dict | None = None,
) -> dict:
    """Build one disc's record.

    Parameters
    ----------
    row:
        A row of the Stage D ``disc_analysis`` table.
    predictions:
        Optional baseline model output for this disc, mapping finding name ->
        ``{"value": ..., "probability": ...}``. When absent, the record contains
        annotations and measurements only.
    """
    ivd_label = int(row["ivd_label"])

    record: dict = {
        # --- Stage G longitudinal keys -----------------------------------
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "patient_id": int(row["patient_id"]),
        "series_id": row["series_id"],
        "study_id": None,          # this dataset has no study identifier
        "modality": row.get("modality"),
        "ivd_label": ivd_label,
        "level_name_provisional": PROVISIONAL_LEVEL_NAMES.get(ivd_label),
        "level_name_is_provisional": True,
        "level_naming_note": (
            "Disc index is authoritative. The conventional name assumes the "
            "lowest annotated vertebra is L5; the dataset documents that it is "
            "usually L5 but can also be L4 or L6."
        ),
        # Explicitly null: this dataset carries no acquisition date and holds
        # one study per patient. Not invented.
        "acquisition_timestamp": None,
        "timepoint_label": None,
        "timepoint_available": False,

        "segmentation_source": row.get("mask_source"),
        "representative_slice_id": row.get("representative_slice_id"),
        "n_slices_measured": _clean(row.get("n_slices_present")),
        "mm_per_pixel": _clean(row.get("mm_per_pixel")),

        "findings": {},
        "measurements": {},
    }

    # --- findings from the dataset annotation ---------------------------
    if bool(row.get("has_grading", False)):
        pfirrmann = _clean(row.get("pfirrmann_grade"))
        record["findings"]["pfirrmann_grade"] = {
            "label": "Pfirrmann grade",
            "value": int(pfirrmann) if pfirrmann is not None else None,
            "scale": "1 (normal) to 5 (most degenerate), ordinal",
            "source": "dataset_annotation",
        }
        modic_type = _clean(row.get("modic"))
        record["findings"]["modic"] = {
            "label": "Modic change",
            "value": int(modic_type) if modic_type is not None else None,
            "value_label": MODIC_TYPE_LABELS.get(
                int(modic_type) if modic_type is not None else -1, "unknown"
            ),
            "scale": "0 none, 1/2/3 = type I/II/III (nominal, not ordered)",
            "source": "dataset_annotation",
        }
        for key, label in BINARY_FINDING_LABELS.items():
            if key == "any_modic" or key not in row.index:
                continue
            value = _clean(row.get(key))
            record["findings"][key] = {
                "label": label,
                "value": int(value) if value is not None else None,
                "value_label": (
                    "present" if value == 1 else "absent" if value == 0 else "unknown"
                ),
                "source": "dataset_annotation",
            }

    # --- findings from the baseline model -------------------------------
    if predictions:
        for key, payload in predictions.items():
            label = BINARY_FINDING_LABELS.get(key, key)
            record["findings"][f"{key}_predicted"] = {
                "label": f"{label} (model estimate)",
                "value": _clean(payload.get("value")),
                "probability": _clean(payload.get("probability")),
                "source": "model_prediction",
                "model": payload.get("model"),
            }

    # --- measured image features ----------------------------------------
    for column, label, unit in REPORTED_MEASUREMENTS:
        if column not in row.index:
            continue
        record["measurements"][column] = {
            "label": label,
            "value": _clean(row.get(column)),
            "unit": unit,
            "source": "measured_image_feature",
        }

    return record


def build_patient_report(
    patient_rows: pd.DataFrame, *, predictions: dict | None = None
) -> dict:
    """Assemble one patient's report, ordered from the lowest disc upward."""
    patient_id = int(patient_rows["patient_id"].iloc[0])
    ordered = patient_rows.sort_values("ivd_label")

    discs = []
    for _, row in ordered.iterrows():
        disc_predictions = None
        if predictions:
            disc_predictions = predictions.get((patient_id, int(row["ivd_label"])))
        discs.append(build_disc_record(row, predictions=disc_predictions))

    return {
        "report_type": "Radiological Finding Assessment",
        "report_subtype": "Automated MRI Finding Report",
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "patient_id": patient_id,
        "series_ids": sorted(ordered["series_id"].unique().tolist()),
        "split": ordered["split"].iloc[0] if "split" in ordered.columns else None,
        "n_discs": len(discs),
        "acquisition_timestamp": None,
        "timepoint_label": None,
        "timepoint_available": False,
        "longitudinal_note": (
            "This dataset is cross-sectional: one study per patient, with no "
            "acquisition date recorded. The timestamp and timepoint fields are "
            "present as schema for a future longitudinal extension and are "
            "deliberately null rather than populated with invented values."
        ),
        "discs": discs,
    }


def render_patient_markdown(report: dict) -> str:
    """Render a patient report as readable Markdown.

    Follows the requested layout: patient, then one block per disc level with
    its findings, each tagged with where the value came from.
    """
    lines: list[str] = [
        f"# Radiological Finding Assessment - Patient {report['patient_id']}",
        "",
        "**Automated MRI Finding Report**",
        "",
        f"> {report['disclaimer']}",
        "",
        "| | |",
        "| --- | --- |",
        f"| Patient ID | `{report['patient_id']}` |",
        f"| MRI series | {', '.join(f'`{s}`' for s in report['series_ids'])} |",
        f"| Dataset split | {report.get('split') or '-'} |",
        f"| Discs assessed | {report['n_discs']} |",
        f"| Acquisition timestamp | *not available in this dataset* |",
        f"| Timepoint | *not applicable - single timepoint dataset* |",
        f"| Generated | {report['generated']} |",
        f"| Schema version | {report['record_schema_version']} |",
        "",
        "Levels are listed from the **most inferior disc upward**, which is the "
        "dataset's own numbering. Conventional level names are *provisional*: they "
        "assume the lowest annotated vertebra is L5, which the dataset documents as "
        "usual but not guaranteed. The integer disc index is the authoritative "
        "identifier.",
        "",
        "---",
        "",
    ]

    for disc in report["discs"]:
        name = disc.get("level_name_provisional") or f"disc {disc['ivd_label']}"
        lines += [
            f"## Disc {disc['ivd_label']}  ({name}, provisional)",
            "",
        ]

        findings = disc["findings"]
        if findings:
            lines += [
                "### Findings",
                "",
                "| Finding | Value | Source |",
                "| --- | --- | --- |",
            ]
            for entry in findings.values():
                value = entry.get("value_label") or entry.get("value")
                if value is None:
                    value = "-"
                if "probability" in entry and entry["probability"] is not None:
                    value = f"{value} (p={entry['probability']:.3f})"
                lines.append(
                    f"| {entry['label']} | {value} | `{entry['source']}` |"
                )
            lines.append("")

        measurements = disc["measurements"]
        if measurements:
            lines += [
                "### Measurements",
                "",
                "| Measurement | Value | Unit | Source |",
                "| --- | --- | --- | --- |",
            ]
            for entry in measurements.values():
                value = entry["value"]
                shown = "-" if value is None else f"{value:g}"
                lines.append(
                    f"| {entry['label']} | {shown} | {entry['unit']} | "
                    f"`{entry['source']}` |"
                )
            lines.append("")

        lines += [
            f"_Measured from the {disc['segmentation_source']} segmentation on "
            f"{disc['n_slices_measured']} sagittal slice(s) at "
            f"{disc['mm_per_pixel']} mm/pixel; representative slice "
            f"`{disc['representative_slice_id']}`._",
            "",
            "---",
            "",
        ]

    lines += [
        "## Provenance legend",
        "",
        "| Tag | Meaning |",
        "| --- | --- |",
        "| `dataset_annotation` | Recorded by the human graders in "
        "`radiological_gradings.csv` |",
        "| `measured_image_feature` | Computed from the segmentation mask in "
        "millimetres |",
        "| `model_prediction` | Estimated by the Stage E baseline, with probability |",
        "",
        "No composite severity score or damage percentage is produced: the eight "
        "findings are graded independently in the source data, and combining them "
        "would be a clinical judgement, not a data-processing step.",
        "",
    ]
    return "\n".join(lines)


def render_patient_text(report: dict) -> str:
    """Render the compact indented tree layout requested in the Sprint 2 brief."""
    lines = [
        f"Patient {report['patient_id']}   "
        f"[series: {', '.join(report['series_ids'])}]",
    ]
    for disc in report["discs"]:
        name = disc.get("level_name_provisional") or ""
        lines.append(f"    Disc {disc['ivd_label']}  ({name}, provisional)")
        for entry in disc["findings"].values():
            value = entry.get("value_label") or entry.get("value")
            if value is None:
                value = "-"
            if "probability" in entry and entry["probability"] is not None:
                value = f"{value} (p={entry['probability']:.3f})"
            lines.append(f"        {entry['label']}: {value}  [{entry['source']}]")

        height = disc["measurements"].get("height_mm_central", {}).get("value")
        area = disc["measurements"].get("area_mm2", {}).get("value")
        if height is not None or area is not None:
            lines.append(
                f"        Measured: central height="
                f"{'-' if height is None else f'{height:g} mm'}, "
                f"area={'-' if area is None else f'{area:g} mm2'}  "
                f"[measured_image_feature]"
            )
    return "\n".join(lines)


def save_patient_report(
    report: dict, output_dir: Path, *, formats: tuple[str, ...] = ("json", "md")
) -> list[Path]:
    """Write a patient report in the requested formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    stem = f"patient_{report['patient_id']:03d}"
    if "json" in formats:
        path = output_dir / f"{stem}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        written.append(path)
    if "md" in formats:
        path = output_dir / f"{stem}.md"
        path.write_text(render_patient_markdown(report), encoding="utf-8")
        written.append(path)
    if "txt" in formats:
        path = output_dir / f"{stem}.txt"
        path.write_text(render_patient_text(report), encoding="utf-8")
        written.append(path)
    return written


def build_longitudinal_ready_table(analysis: pd.DataFrame) -> pd.DataFrame:
    """Flat, longitudinal-ready table: one row per (patient, series, disc).

    Deliberately includes the null time columns. A future sprint that ingests
    follow-up studies populates ``acquisition_timestamp`` and
    ``timepoint_label`` and the schema does not otherwise change, so a
    comparison can be written as a join on
    ``(patient_id, ivd_label)`` across ``timepoint_label``.
    """
    columns_in_order = [
        # identity / longitudinal keys
        "patient_id", "series_id", "ivd_label", "modality", "split",
        # findings (dataset annotation)
        "pfirrmann_grade", "modic", "any_modic", "bulging", "narrowing",
        "herniation", "spondylolisthesis", "upper_endplate", "lower_endplate",
        # measurements
        "height_mm_central", "height_mm_anterior", "height_mm_posterior",
        "area_mm2", "ap_extent_mm", "height_ratio_to_series_median",
        "disc_to_vertebra_height_ratio", "height_ap_asymmetry",
        "vertebral_ap_offset_mm", "canal_width_at_disc_mm",
        # intensity
        "intensity_mean", "intensity_disc_vertebra_ratio",
        "intensity_nucleus_annulus_ratio", "intensity_cv",
        # provenance
        "mask_source", "representative_slice_id", "n_slices_present",
        "mm_per_pixel", "grading_level", "is_primary_series",
    ]
    available = [c for c in columns_in_order if c in analysis.columns]
    table = analysis[available].copy()

    table.insert(0, "record_schema_version", RECORD_SCHEMA_VERSION)
    table["study_id"] = pd.NA
    table["acquisition_timestamp"] = pd.NA
    table["timepoint_label"] = pd.NA
    table["timepoint_available"] = False
    table["level_name_provisional"] = table["ivd_label"].map(PROVISIONAL_LEVEL_NAMES)
    table["level_name_is_provisional"] = True

    return table.sort_values(["patient_id", "series_id", "ivd_label"])
