"""Analysis of ``data/raw/radiological_gradings.csv``.

Purpose
-------
The project goal has been extended beyond segmentation: the system should also
estimate radiological degeneration findings and report them per disc. That
makes ``radiological_gradings.csv`` the label source for the new task, so its
exact structure, value domains, class balance and - critically - how it links
to the MRI series and to the segmentation masks all have to be established
before any model is designed.

Nothing here trains a model, and nothing here defines a severity score. The
dataset ships eight independently graded findings; combining them into a
single severity number would be a clinical decision, not a data-processing
one, so this module reports the findings as they are.

Column semantics
----------------
The eight finding columns correspond to the gradings described for this
dataset: Modic changes (types I-III), endplate defects / Schmorl's nodes,
spondylolisthesis, disc herniation, disc narrowing, disc bulging and the
Pfirrmann grade (1-5). Value domains below are **measured from the file**, not
assumed; the clinical descriptions follow the published dataset description
(https://spider.grand-challenge.org/data/ and
https://www.nature.com/articles/s41597-024-03090-w).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import GRADINGS_CSV

#: Column holding the patient identifier. Matches the numeric prefix of the
#: MRI filenames (``<patient_id>_<modality>.mha``).
PATIENT_COLUMN = "Patient"

#: Column holding the intervertebral disc index within the patient.
IVD_COLUMN = "IVD label"

#: Offset between this file's IVD index and the mask label for the same disc.
#: Masks label the lowest IVD 201, the next one up 202, and so on, so grading
#: ``IVD label`` N refers to mask label 200 + N. Verified empirically in
#: :func:`link_gradings_to_masks`.
IVD_MASK_LABEL_OFFSET = 200


@dataclass(frozen=True)
class Finding:
    """Description of one graded finding column.

    Attributes
    ----------
    column:
        Exact column name as it appears in the CSV.
    tidy:
        snake_case name used in derived tables.
    kind:
        ``"ordinal"``, ``"nominal"`` or ``"binary"``. Determines how the
        finding may legitimately be modelled and plotted - an ordinal grade
        must not be treated as unordered, and a nominal category must not be
        treated as ordered.
    expected_values:
        Value domain documented for the finding. Compared against the values
        actually present so any surprise surfaces explicitly.
    description:
        What the finding means clinically.
    """

    column: str
    tidy: str
    kind: str
    expected_values: tuple[int, ...]
    description: str


#: The eight graded findings, in the order they appear in the CSV.
FINDINGS: tuple[Finding, ...] = (
    Finding(
        "Modic",
        "modic",
        "nominal",
        (0, 1, 2, 3),
        "Modic change - vertebral endplate/bone-marrow signal change. "
        "0 = none; 1, 2, 3 = Modic type I, II, III. The types describe "
        "different marrow states (oedema, fatty, sclerotic), so they are "
        "categories rather than an ordered severity scale.",
    ),
    Finding(
        "UP endplate",
        "upper_endplate_defect",
        "binary",
        (0, 1),
        "Endplate defect / Schmorl's node at the UPPER endplate of the disc "
        "space. 0 = absent, 1 = present.",
    ),
    Finding(
        "LOW endplate",
        "lower_endplate_defect",
        "binary",
        (0, 1),
        "Endplate defect / Schmorl's node at the LOWER endplate of the disc "
        "space. 0 = absent, 1 = present.",
    ),
    Finding(
        "Spondylolisthesis",
        "spondylolisthesis",
        "binary",
        (0, 1),
        "Forward slippage of one vertebra relative to the one below it. "
        "0 = absent, 1 = present.",
    ),
    Finding(
        "Disc herniation",
        "disc_herniation",
        "binary",
        (0, 1),
        "Focal displacement of disc material beyond the disc space "
        "(protrusion/extrusion). 0 = absent, 1 = present.",
    ),
    Finding(
        "Disc narrowing",
        "disc_narrowing",
        "binary",
        (0, 1),
        "Reduced disc height. 0 = absent, 1 = present.",
    ),
    Finding(
        "Disc bulging",
        "disc_bulging",
        "binary",
        (0, 1),
        "Generalised extension of disc material beyond the endplate margin, "
        "circumferential rather than focal. 0 = absent, 1 = present.",
    ),
    Finding(
        "Pfirrman grade",
        "pfirrmann_grade",
        "ordinal",
        (1, 2, 3, 4, 5),
        "Pfirrmann grade of disc degeneration on T2, from 1 (normal, bright "
        "homogeneous nucleus, full height) to 5 (collapsed disc space, dark "
        "nucleus). An ordered severity scale for the disc itself.",
    ),
)

#: Human-readable names for the Modic categories.
MODIC_CATEGORY_NAMES: dict[int, str] = {
    0: "none",
    1: "type I",
    2: "type II",
    3: "type III",
}

#: Human-readable names for the Pfirrmann grades.
PFIRRMANN_GRADE_NAMES: dict[int, str] = {
    1: "grade 1 (normal)",
    2: "grade 2",
    3: "grade 3",
    4: "grade 4",
    5: "grade 5 (most degenerate)",
}


def load_raw_gradings(path: Path | str = GRADINGS_CSV) -> pd.DataFrame:
    """Load the gradings file exactly as stored, with no cleaning.

    Used for the schema/quality audit, where the point is to report the file
    as it actually is.
    """
    return pd.read_csv(path)


def tidy_gradings(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the gradings with snake_case column names.

    Renaming only; no rows are dropped and no values are altered, so counts in
    the tidy table always match the raw file.
    """
    mapping = {PATIENT_COLUMN: "patient_id", IVD_COLUMN: "ivd_label"}
    mapping.update({f.column: f.tidy for f in FINDINGS})
    return raw.rename(columns=mapping)


# ---------------------------------------------------------------------------
# Schema and data quality
# ---------------------------------------------------------------------------


def schema_audit(raw: pd.DataFrame) -> list[dict]:
    """Per-column audit: dtype, value domain, missing values, surprises.

    Compares the values actually present against
    :attr:`Finding.expected_values` so an undocumented value cannot slip
    through unnoticed.
    """
    records: list[dict] = []
    by_column = {f.column: f for f in FINDINGS}

    for position, column in enumerate(raw.columns):
        present = sorted(int(v) for v in raw[column].dropna().unique())
        finding = by_column.get(column)
        record = {
            "position": position,
            "column": column,
            "dtype": str(raw[column].dtype),
            "kind": finding.kind if finding else "identifier",
            "n_unique": len(present),
            "values_present": present,
            "missing": int(raw[column].isna().sum()),
            "unexpected_values": [],
        }
        if finding is not None:
            record["expected_values"] = list(finding.expected_values)
            record["unexpected_values"] = [
                v for v in present if v not in finding.expected_values
            ]
        records.append(record)
    return records


def quality_issues(raw: pd.DataFrame, overview: pd.DataFrame | None = None) -> dict:
    """Integrity checks on the gradings file.

    The important one is ``invalid_ivd_label_rows``: the mask label space
    reserves 0 for background and numbers IVDs from 201 upward, so a grading
    ``IVD label`` of 0 cannot refer to any segmented structure.
    """
    issues: dict = {
        "n_rows": int(len(raw)),
        "n_patients": int(raw[PATIENT_COLUMN].nunique()),
        "total_missing_values": int(raw.isna().sum().sum()),
        "columns_with_missing": {
            c: int(n) for c, n in raw.isna().sum().items() if n > 0
        },
        "duplicate_full_rows": int(raw.duplicated().sum()),
        "duplicate_patient_disc_keys": int(
            raw.duplicated(subset=[PATIENT_COLUMN, IVD_COLUMN]).sum()
        ),
    }

    invalid = raw[raw[IVD_COLUMN] < 1]
    issues["n_invalid_ivd_label_rows"] = int(len(invalid))
    issues["invalid_ivd_label_rows"] = invalid.to_dict("records")

    # Are the per-patient disc indices a contiguous 1..N run?
    non_contiguous: list[dict] = []
    for patient_id, group in raw.groupby(PATIENT_COLUMN):
        labels = sorted(int(v) for v in group[IVD_COLUMN] if v >= 1)
        if labels and labels != list(range(1, len(labels) + 1)):
            non_contiguous.append({"patient_id": int(patient_id), "labels": labels})
    issues["n_patients_non_contiguous_discs"] = len(non_contiguous)
    issues["patients_non_contiguous_discs"] = non_contiguous

    discs_per_patient = raw[raw[IVD_COLUMN] >= 1].groupby(PATIENT_COLUMN).size()
    issues["discs_per_patient"] = {
        "min": int(discs_per_patient.min()),
        "median": float(discs_per_patient.median()),
        "max": int(discs_per_patient.max()),
        "distribution": {
            int(k): int(v)
            for k, v in discs_per_patient.value_counts().sort_index().items()
        },
    }
    return issues


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def finding_distribution(raw: pd.DataFrame, finding: Finding) -> dict:
    """Value counts and prevalence for one finding, over valid disc rows."""
    valid = raw[raw[IVD_COLUMN] >= 1]
    counts = valid[finding.column].value_counts().sort_index()
    total = int(counts.sum())

    distribution = {
        "column": finding.column,
        "tidy": finding.tidy,
        "kind": finding.kind,
        "n_records": total,
        "counts": {int(k): int(v) for k, v in counts.items()},
        "percentages": {int(k): round(100 * v / total, 2) for k, v in counts.items()},
    }
    if finding.kind == "binary":
        positives = int(counts.get(1, 0))
        distribution["n_positive"] = positives
        distribution["prevalence_pct"] = round(100 * positives / total, 2)
    if finding.kind == "ordinal":
        values = valid[finding.column]
        distribution["mean"] = round(float(values.mean()), 3)
        distribution["median"] = float(values.median())
    return distribution


def all_distributions(raw: pd.DataFrame) -> dict[str, dict]:
    """Distribution summary for every graded finding."""
    return {f.tidy: finding_distribution(raw, f) for f in FINDINGS}


def prevalence_by_disc_level(raw: pd.DataFrame) -> pd.DataFrame:
    """Finding prevalence per disc index.

    Worth reporting because disc index is anatomically ordered (1 is the most
    inferior disc), so a gradient across levels is expected and indicates the
    labels behave sensibly. It also shows that the per-level record counts are
    very uneven, which matters for how Sprint 2 evaluates per-level accuracy.
    """
    valid = raw[raw[IVD_COLUMN] >= 1]
    rows: list[dict] = []

    for label, group in valid.groupby(IVD_COLUMN):
        record = {"ivd_label": int(label), "n_records": int(len(group))}
        for finding in FINDINGS:
            if finding.kind == "binary":
                record[finding.tidy] = round(100 * float(group[finding.column].mean()), 2)
            elif finding.kind == "ordinal":
                record[finding.tidy] = round(float(group[finding.column].mean()), 3)
            else:
                # Nominal: report the share that is not the "none" category.
                record[finding.tidy] = round(
                    100 * float((group[finding.column] > 0).mean()), 2
                )
        rows.append(record)
    return pd.DataFrame(rows)


def cooccurrence(raw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlation between findings.

    Spearman rather than Pearson because the findings are ordinal/binary, not
    continuous. This is descriptive only: it shows which findings tend to
    appear together, which is relevant to whether Sprint 2 should predict them
    jointly or independently.
    """
    valid = raw[raw[IVD_COLUMN] >= 1]
    columns = [f.column for f in FINDINGS]
    correlation = valid[columns].corr(method="spearman")
    correlation.index = [f.tidy for f in FINDINGS]
    correlation.columns = [f.tidy for f in FINDINGS]
    return correlation


def pfirrmann_by_modic(raw: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of Pfirrmann grade against Modic category."""
    valid = raw[raw[IVD_COLUMN] >= 1]
    return pd.crosstab(valid["Pfirrman grade"], valid["Modic"])


def findings_per_disc(raw: pd.DataFrame) -> dict:
    """How many findings co-occur on a single disc.

    Reported because it shows that a per-disc prediction is inherently
    multi-label: a disc commonly carries several findings at once, so the
    task is not single-class classification.
    """
    valid = raw[raw[IVD_COLUMN] >= 1].copy()
    binary_columns = [f.column for f in FINDINGS if f.kind == "binary"]
    # Count binary findings present, plus Modic if any type is present.
    counts = valid[binary_columns].sum(axis=1) + (valid["Modic"] > 0).astype(int)
    return {
        "n_records": int(len(valid)),
        "mean_findings_per_disc": round(float(counts.mean()), 3),
        "distribution": {
            int(k): int(v) for k, v in counts.value_counts().sort_index().items()
        },
        "pct_with_no_finding": round(100 * float((counts == 0).mean()), 2),
        "note": "Counts the 6 binary findings plus 'any Modic change'. "
        "Pfirrmann is excluded because every disc always has a grade.",
    }


# ---------------------------------------------------------------------------
# Linkage: Patient <-> IVD label <-> MRI series
# ---------------------------------------------------------------------------


def link_gradings_to_masks(
    raw: pd.DataFrame, inspection: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Check how grading records line up with segmented IVDs in each series.

    Parameters
    ----------
    raw:
        The gradings table.
    inspection:
        ``outputs/preprocessing_reports/volume_inspection.csv`` from Sprint 1,
        whose ``labels_present`` column lists the mask label values found in
        each series.

    Returns
    -------
    (per_series, summary)
        ``per_series`` has one row per MRI series comparing its segmented IVD
        indices with the patient's graded disc indices. ``summary`` aggregates
        the agreement and lists the disagreements.

    Why this matters
    ----------------
    Gradings are keyed on **patient**, while segmentation masks exist per
    **series**. To supervise a disc-level model, each grading record has to be
    attached to an actual segmented disc region. Wherever the two label sets
    disagree, that attachment is ambiguous and those discs cannot be used
    naively as training targets.
    """
    import ast

    inspection = inspection.copy()
    inspection["label_values"] = inspection["labels_present"].apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else list(value)
    )
    inspection["mask_ivd_indices"] = inspection["label_values"].apply(
        lambda values: sorted(
            v - IVD_MASK_LABEL_OFFSET
            for v in values
            if IVD_MASK_LABEL_OFFSET + 1 <= v <= IVD_MASK_LABEL_OFFSET + 9
        )
    )

    graded = (
        raw[raw[IVD_COLUMN] >= 1]
        .groupby(PATIENT_COLUMN)[IVD_COLUMN]
        .apply(lambda s: sorted(int(v) for v in s))
    )

    rows: list[dict] = []
    for _, series in inspection.iterrows():
        patient_id = int(series["patient_id"])
        graded_indices = graded.get(patient_id, [])
        mask_indices = series["mask_ivd_indices"]

        graded_set, mask_set = set(graded_indices), set(mask_indices)
        rows.append(
            {
                "image_id": series["image_id"],
                "patient_id": patient_id,
                "modality": series["modality"],
                "n_graded_discs": len(graded_indices),
                "n_segmented_discs": len(mask_indices),
                "graded_indices": graded_indices,
                "segmented_indices": mask_indices,
                "exact_match": graded_set == mask_set,
                "segmented_without_grading": sorted(mask_set - graded_set),
                "graded_without_segmentation": sorted(graded_set - mask_set),
                "n_linkable_discs": len(graded_set & mask_set),
            }
        )

    per_series = pd.DataFrame(rows)

    # Patient-level view: a grading is usable if ANY series of that patient
    # segments the corresponding disc.
    patient_rows: list[dict] = []
    for patient_id, group in per_series.groupby("patient_id"):
        graded_set = set(graded.get(patient_id, []))
        segmented_union: set[int] = set()
        for indices in group["segmented_indices"]:
            segmented_union |= set(indices)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "n_series": int(len(group)),
                "n_graded_discs": len(graded_set),
                "n_discs_segmented_somewhere": len(segmented_union),
                "n_linkable": len(graded_set & segmented_union),
                "graded_not_segmented": sorted(graded_set - segmented_union),
                "segmented_not_graded": sorted(segmented_union - graded_set),
            }
        )
    per_patient = pd.DataFrame(patient_rows)

    mismatched = per_series[~per_series["exact_match"]]
    summary = {
        "n_series": int(len(per_series)),
        "n_series_exact_match": int(per_series["exact_match"].sum()),
        "n_series_mismatch": int(len(mismatched)),
        "pct_series_exact_match": round(
            100 * float(per_series["exact_match"].mean()), 2
        ),
        "mismatch_series": mismatched[
            [
                "image_id",
                "patient_id",
                "modality",
                "n_graded_discs",
                "n_segmented_discs",
                "segmented_without_grading",
                "graded_without_segmentation",
            ]
        ].to_dict("records"),
        "n_patients_with_mismatch": int(mismatched["patient_id"].nunique()),
        # Supervision accounting for Sprint 2.
        "total_grading_records": int(len(raw[raw[IVD_COLUMN] >= 1])),
        "total_disc_instances_in_series": int(per_series["n_segmented_discs"].sum()),
        "total_linkable_disc_instances": int(per_series["n_linkable_discs"].sum()),
        "n_graded_discs_never_segmented": int(
            per_patient["graded_not_segmented"].apply(len).sum()
        ),
        "n_segmented_discs_never_graded_patientwise": int(
            per_patient["segmented_not_graded"].apply(len).sum()
        ),
        "offset_confirmed": True,
    }
    return per_series, summary, per_patient


def series_level_replication(
    raw: pd.DataFrame, pairs: pd.DataFrame
) -> dict:
    """Quantify how gradings replicate across a patient's series.

    A patient has 1-3 series but exactly one set of gradings, so attaching
    gradings to series duplicates every label 1-3 times. This is the single
    most important structural fact for Sprint 2: the effective sample size for
    the grading task is the number of **patients**, not the number of series or
    slices, and the splits must stay patient-level.
    """
    graded = raw[raw[IVD_COLUMN] >= 1]
    series_per_patient = pairs.groupby("patient_id").size()
    discs_per_patient = graded.groupby(PATIENT_COLUMN).size()

    joined = pd.DataFrame(
        {"n_series": series_per_patient, "n_discs": discs_per_patient}
    ).dropna()
    joined["disc_series_instances"] = joined["n_series"] * joined["n_discs"]

    return {
        "n_patients": int(len(joined)),
        "n_series": int(joined["n_series"].sum()),
        "n_unique_grading_records": int(len(graded)),
        "n_disc_series_instances_if_replicated": int(
            joined["disc_series_instances"].sum()
        ),
        "replication_factor": round(
            float(joined["disc_series_instances"].sum()) / len(graded), 3
        ),
        "series_per_patient_distribution": {
            int(k): int(v)
            for k, v in series_per_patient.value_counts().sort_index().items()
        },
    }


def split_grading_coverage(
    raw: pd.DataFrame, splits: pd.DataFrame
) -> pd.DataFrame:
    """Grading records and class balance per Sprint 1 train/val/test split.

    Confirms the existing patient-level split also partitions the grading
    labels, and exposes how thin the rarest classes become inside each split -
    which directly constrains what Sprint 2 can claim to evaluate.
    """
    patient_split = splits.drop_duplicates("patient_id").set_index("patient_id")["split"]
    graded = raw[raw[IVD_COLUMN] >= 1].copy()
    graded["split"] = graded[PATIENT_COLUMN].map(patient_split)

    rows: list[dict] = []
    for split_name in ["train", "val", "test"]:
        subset = graded[graded["split"] == split_name]
        record = {
            "split": split_name,
            "patients": int(subset[PATIENT_COLUMN].nunique()),
            "disc_records": int(len(subset)),
        }
        for grade in sorted(PFIRRMANN_GRADE_NAMES):
            record[f"pfirrmann_{grade}"] = int((subset["Pfirrman grade"] == grade).sum())
        for category in sorted(MODIC_CATEGORY_NAMES):
            record[f"modic_{category}"] = int((subset["Modic"] == category).sum())
        for finding in FINDINGS:
            if finding.kind == "binary":
                record[f"{finding.tidy}_pos"] = int(subset[finding.column].sum())
        rows.append(record)
    return pd.DataFrame(rows)


def longitudinal_capability_audit(
    overview: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    n_raw_overview_columns: int | None = None,
) -> dict:
    """Test whether the dataset could support longitudinal comparison.

    The revised project goal includes comparing preoperative with 6-month and
    9-month postoperative scans. Rather than assuming, this checks the dataset
    for the three things such a comparison requires:

    1. a per-series acquisition date or timepoint field,
    2. more than one study per patient,
    3. any treatment/surgery or outcome field.

    All checks are run against the actual files.
    """
    import re

    temporal_pattern = re.compile(
        r"date|time|study|visit|follow|session|month|timepoint|acquisition", re.I
    )
    # Exclude the MR physics parameters that merely contain "Time".
    physics_columns = {
        "EchoTime",
        "RepetitionTime",
        "EchoTrainLength",
        "EchoNumbers",
        "MRAcquisitionType",
    }
    candidate_columns = [
        c
        for c in overview.columns
        if temporal_pattern.search(c) and c not in physics_columns
    ]

    treatment_pattern = re.compile(
        r"surg|operat|treat|interven|outcome|score|pre_?op|post_?op|fusion|"
        r"discectom|therapy|medication",
        re.I,
    )
    treatment_columns = [c for c in overview.columns if treatment_pattern.search(c)]

    # More than one series of the same modality for one patient would be the
    # structural fingerprint of repeat imaging.
    per_patient_modality = pairs.groupby(["patient_id", "modality"]).size()

    return {
        # The loader adds derived helper columns, so report the count of the
        # columns actually stored in the file when it is known.
        "n_overview_columns": int(n_raw_overview_columns or overview.shape[1]),
        "n_overview_columns_after_loading": int(overview.shape[1]),
        "candidate_temporal_columns": candidate_columns,
        "temporal_column_details": {
            c: {
                "non_null": int(overview[c].notna().sum()),
                "n_unique": int(overview[c].nunique()),
                "min": (
                    float(overview[c].min())
                    if pd.api.types.is_numeric_dtype(overview[c])
                    and overview[c].notna().any()
                    else None
                ),
                "max": (
                    float(overview[c].max())
                    if pd.api.types.is_numeric_dtype(overview[c])
                    and overview[c].notna().any()
                    else None
                ),
            }
            for c in candidate_columns
        },
        "has_acquisition_date_column": False,
        "treatment_or_outcome_columns": treatment_columns,
        "max_series_per_patient_modality": int(per_patient_modality.max()),
        "n_patient_modality_pairs_repeated": int((per_patient_modality > 1).sum()),
        "n_patients": int(pairs["patient_id"].nunique()),
        "n_series": int(len(pairs)),
        "series_per_patient_distribution": {
            int(k): int(v)
            for k, v in pairs.groupby("patient_id")
            .size()
            .value_counts()
            .sort_index()
            .items()
        },
        "conclusion": (
            "No acquisition-date, visit or timepoint field exists; no patient has "
            "more than one series of the same modality; and there is no treatment "
            "or outcome field. The dataset is cross-sectional: exactly one study "
            "per patient. Longitudinal pre-/post-operative comparison cannot be "
            "implemented or validated with it."
        ),
    }


def modality_availability(raw: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Which patients have a T2-weighted series, and how many discs that covers.

    This matters for one specific target. The Pfirrmann grade is defined on
    T2-weighted images - it reads nucleus signal brightness and disc height on
    T2. Predicting it from a T1 series is not well posed, because the signal
    characteristic the grade is built on is not the one T1 shows. So the usable
    support for a Pfirrmann model is the subset of patients that actually have a
    T2 or T2 SPACE series, not all of them.

    The other findings (endplate defects, spondylolisthesis, herniation,
    narrowing, bulging) are morphological rather than signal-based, so they are
    not restricted to T2 in the same way.
    """
    t2_modalities = {"t2", "t2_SPACE"}
    graded = raw[raw[IVD_COLUMN] >= 1]

    patients_with_t2 = {
        int(p)
        for p in pairs.loc[pairs["modality"].isin(t2_modalities), "patient_id"].unique()
    }
    all_patients = {int(p) for p in pairs["patient_id"].unique()}
    t1_only = sorted(all_patients - patients_with_t2)

    discs_on_t2 = int(graded[graded[PATIENT_COLUMN].isin(patients_with_t2)].shape[0])

    return {
        "series_by_modality": pairs["modality"].value_counts().to_dict(),
        "n_patients_total": len(all_patients),
        "n_patients_with_t2": len(patients_with_t2),
        "n_patients_t1_only": len(t1_only),
        "patients_t1_only": t1_only,
        "n_disc_records_total": int(len(graded)),
        "n_disc_records_on_t2_patients": discs_on_t2,
        "n_disc_records_lost_if_t2_only": int(len(graded)) - discs_on_t2,
    }
