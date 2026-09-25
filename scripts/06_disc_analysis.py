"""Sprint 2 / Stages C+D - extract disc instances and link radiological gradings.

Stage C walks every preprocessed slice, isolates each intervertebral disc from
the Sprint 1 **instance** mask, and measures it in millimetres.
Stage D collapses those measurements to one row per (patient, series, disc) and
joins ``radiological_gradings.csv`` on ``patient_id + ivd_label``.

Writes:

    outputs/reports/disc_analysis.csv          the Stage D analysis table
    outputs/reports/disc_slice_measurements.csv per-slice measurements (Stage C)
    outputs/reports/disc_analysis_summary.json  counts and linkage audit
    outputs/visualizations/disc_measurements.png

Sprint 1 data is read-only. No model is involved in this stage: measurements
come from the ground-truth masks, findings come from the dataset annotation.

Usage
-----
    python scripts/06_disc_analysis.py
    python scripts/06_disc_analysis.py --limit 200       # quick check
    python scripts/06_disc_analysis.py --mask-source prediction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.disc_features import (  # noqa: E402
    DEFAULT_MM_PER_PIXEL,
    add_relative_height_features,
    aggregate_disc_records,
    extract_disc_features_from_slice,
)
from src.analysis.gradings import (  # noqa: E402
    FINDINGS,
    IVD_COLUMN,
    PATIENT_COLUMN,
    load_raw_gradings,
    tidy_gradings,
)
from src.models.data import load_slice_index  # noqa: E402
from src.utils.paths import (  # noqa: E402
    ANALYSIS_REPORTS_DIR,
    PROJECT_ROOT,
    VISUALIZATIONS_DIR,
    ensure_dirs,
)
from src.utils.reporting import save_json  # noqa: E402

#: Preference order when picking one representative series per patient.
#: T2 first because the Pfirrmann grade is defined on T2 signal; T2 SPACE next
#: (also T2-weighted); T1 last.
MODALITY_PREFERENCE = {"t2": 0, "t2_SPACE": 1, "t1": 2}

#: Grading columns, renamed to the names the Sprint 2 spec asks for.
GRADING_RENAME = {
    "pfirrmann_grade": "pfirrmann_grade",
    "modic": "modic",
    "disc_bulging": "bulging",
    "disc_narrowing": "narrowing",
    "disc_herniation": "herniation",
    "spondylolisthesis": "spondylolisthesis",
    "upper_endplate_defect": "upper_endplate",
    "lower_endplate_defect": "lower_endplate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N slices (smoke check).")
    parser.add_argument("--mask-source", default="ground_truth",
                        choices=["ground_truth", "prediction"],
                        help="ground_truth uses the Sprint 1 instance masks; "
                             "prediction uses data/processed/predictions/ written "
                             "by evaluate_unet.py --save-predictions.")
    parser.add_argument("--min-area-px", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Directory of predicted masks (default "
                             "data/processed/predictions).")
    parser.add_argument("--suffix", default=None,
                        help="Override the output filename suffix. Use this to "
                             "keep a second predicted-mask run from overwriting "
                             "the first (e.g. --suffix _sprint2_extended).")
    return parser.parse_args()


def load_slice_arrays(
    row: pd.Series, mask_source: str, predictions_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load ``(image, instance_mask)`` for one slice from the requested source.

    The image always comes from the Sprint 1 ``.npz``; only the mask source
    changes, so intensity features are computed on identical pixels whether the
    geometry came from ground truth or from a prediction.
    """
    with np.load(PROJECT_ROOT / row["npz_path"]) as bundle:
        image = bundle["image"].astype(np.float32)
        ground_truth_instance = bundle["mask_instance"]

    if mask_source == "ground_truth":
        return image, ground_truth_instance

    base = predictions_dir or (PROJECT_ROOT / "data" / "processed" / "predictions")
    path = Path(base) / f"{row['slice_id']}.npz"
    if not path.exists():
        return None
    with np.load(path) as bundle:
        return image, bundle["instance"]


def main() -> None:
    args = parse_args()
    ensure_dirs()

    index = load_slice_index()
    if args.limit:
        index = index.head(args.limit)

    print(f"Stage C: extracting disc instances from {len(index):,} slices "
          f"(source={args.mask_source}) ...")

    records: list[dict] = []
    missing = 0
    for position, (_, row) in enumerate(index.iterrows(), start=1):
        loaded = load_slice_arrays(row, args.mask_source, args.predictions_dir)
        if loaded is None:
            missing += 1
            continue
        image, instance_mask = loaded

        records.extend(
            extract_disc_features_from_slice(
                image,
                instance_mask,
                patient_id=int(row["patient_id"]),
                series_id=row["image_id"],
                slice_id=row["slice_id"],
                slice_index=int(row["slice_index"]),
                mask_source=args.mask_source,
                mm_per_pixel=DEFAULT_MM_PER_PIXEL,
                min_area_px=args.min_area_px,
            )
        )

        if args.progress_every and position % args.progress_every == 0:
            print(f"  {position:,}/{len(index):,} slices, "
                  f"{len(records):,} disc-slice records", flush=True)

    if missing:
        print(f"  WARNING: {missing} slices had no mask available and were skipped")

    slice_frame = pd.DataFrame(records)
    if slice_frame.empty:
        raise SystemExit("No disc instances were extracted - check the mask source.")

    # Predicted-mask runs write to separate files so the ground-truth table,
    # which Stages E and F consume, is never overwritten by a comparison run.
    if args.suffix is not None:
        suffix = args.suffix
    else:
        suffix = "" if args.mask_source == "ground_truth" else "_predicted"

    slice_path = ANALYSIS_REPORTS_DIR / f"disc_slice_measurements{suffix}.csv"
    slice_frame.to_csv(slice_path, index=False)
    print(f"\n  {len(slice_frame):,} disc-slice measurement records")
    print(f"  -> {slice_path}")

    # --- Stage C aggregation: one row per (patient, series, disc) --------
    print("\nStage C: aggregating to one row per (patient, series, disc) ...")
    disc_frame = aggregate_disc_records(slice_frame)
    disc_frame = add_relative_height_features(disc_frame)
    print(f"  {len(disc_frame):,} disc records "
          f"({disc_frame['patient_id'].nunique()} patients, "
          f"{disc_frame['series_id'].nunique()} series)")

    # --- Stage D: link the gradings --------------------------------------
    print("\nStage D: linking radiological_gradings.csv on patient_id + ivd_label ...")
    analysis, linkage = link_gradings(disc_frame, index)

    analysis_path = ANALYSIS_REPORTS_DIR / f"disc_analysis{suffix}.csv"
    analysis.to_csv(analysis_path, index=False)
    print(f"  {len(analysis):,} rows written")
    print(f"  -> {analysis_path}")

    for key, value in linkage.items():
        if not isinstance(value, (list, dict)):
            print(f"  {key:46s} {value}")

    # --- figures and summary --------------------------------------------
    if args.mask_source == "ground_truth":
        render_figures(analysis)

    summary = {
        "mask_source": args.mask_source,
        "mm_per_pixel": DEFAULT_MM_PER_PIXEL,
        "n_slices_processed": int(len(index)),
        "n_slices_missing_mask": missing,
        "n_disc_slice_records": int(len(slice_frame)),
        "n_disc_records": int(len(disc_frame)),
        "n_patients": int(disc_frame["patient_id"].nunique()),
        "n_series": int(disc_frame["series_id"].nunique()),
        "linkage": linkage,
        "measurement_summary": measurement_summary(analysis),
        "columns": list(analysis.columns),
    }
    save_json(summary, ANALYSIS_REPORTS_DIR / f"disc_analysis_summary{suffix}.json")
    print(f"\n  -> {ANALYSIS_REPORTS_DIR / f'disc_analysis_summary{suffix}.json'}")
    print("\nDone. Measurements are from masks; findings are dataset annotations. "
          "No model prediction and no severity score in this table.")


def link_gradings(
    disc_frame: pd.DataFrame, index: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Join the per-disc gradings onto the measured discs.

    Handling the patient-vs-series granularity correctly
    ---------------------------------------------------
    Gradings are keyed on **patient**, while measurements are per **series**. A
    patient with three series therefore legitimately produces three measurement
    rows that share one grading. That is recorded explicitly rather than hidden:

    * ``grading_level`` states that the finding is a patient-level annotation.
    * ``n_series_for_patient`` gives the replication factor for that row.
    * ``is_primary_series`` marks exactly one series per patient, so any
      modelling step can deduplicate to one row per (patient, disc) with a
      single filter instead of accidentally training on duplicated labels.
    """
    raw = load_raw_gradings()
    tidy = tidy_gradings(raw)
    valid = tidy[tidy["ivd_label"] >= 1].copy()

    grading_columns = ["patient_id", "ivd_label"] + [f.tidy for f in FINDINGS]
    gradings = valid[grading_columns].rename(columns=GRADING_RENAME)
    # A derived convenience target: the Sprint 2 scoping analysis showed Modic
    # types I and III are too rare to model separately (4 and 7 records).
    gradings["any_modic"] = (gradings["modic"] > 0).astype(int)

    merged = disc_frame.merge(
        gradings, on=["patient_id", "ivd_label"], how="left", validate="many_to_one"
    )

    # Attach the Sprint 1 split (patient-level) and modality.
    series_meta = (
        index[["image_id", "patient_id", "modality", "split"]]
        .drop_duplicates("image_id")
        .rename(columns={"image_id": "series_id"})
    )
    merged = merged.merge(
        series_meta.drop(columns=["patient_id"]), on="series_id", how="left"
    )

    # Replication bookkeeping.
    merged["grading_level"] = "patient_level_annotation"
    series_per_patient = (
        merged.groupby("patient_id")["series_id"].nunique().rename("n_series_for_patient")
    )
    merged = merged.merge(series_per_patient, on="patient_id", how="left")

    merged["modality_rank"] = merged["modality"].map(MODALITY_PREFERENCE).fillna(99)
    primary = (
        merged.sort_values(["patient_id", "modality_rank", "series_id"])
        .drop_duplicates("patient_id")["series_id"]
        .tolist()
    )
    merged["is_primary_series"] = merged["series_id"].isin(primary)
    merged = merged.drop(columns=["modality_rank"])

    merged["has_grading"] = merged["pfirrmann_grade"].notna()

    # --- audit -----------------------------------------------------------
    unmatched = merged[~merged["has_grading"]]
    graded_keys = set(zip(gradings["patient_id"], gradings["ivd_label"]))
    measured_keys = set(zip(merged["patient_id"], merged["ivd_label"]))

    linkage = {
        "n_measured_disc_rows": int(len(merged)),
        "n_rows_with_grading": int(merged["has_grading"].sum()),
        "n_rows_without_grading": int(len(unmatched)),
        "n_unique_patient_disc_measured": len(measured_keys),
        "n_unique_patient_disc_graded": len(graded_keys),
        "n_graded_not_measured": len(graded_keys - measured_keys),
        "n_measured_not_graded": len(measured_keys - graded_keys),
        "graded_not_measured_examples": sorted(graded_keys - measured_keys)[:20],
        "measured_not_graded_examples": sorted(measured_keys - graded_keys)[:20],
        "n_primary_series_rows": int(merged["is_primary_series"].sum()),
        "rows_per_split": merged["split"].value_counts().to_dict(),
        "primary_rows_per_split": merged[merged["is_primary_series"]]["split"]
        .value_counts()
        .to_dict(),
    }

    ordered = order_columns(merged)
    return ordered, linkage


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the identity and grading columns first, measurements after."""
    identity = [
        "patient_id", "ivd_label", "series_id", "modality", "split",
        "representative_slice_id", "n_slices_present", "mask_source",
        "instance_label", "raw_mask_label", "mm_per_pixel",
        "grading_level", "n_series_for_patient", "is_primary_series", "has_grading",
    ]
    gradings = [
        "pfirrmann_grade", "modic", "any_modic", "bulging", "narrowing",
        "herniation", "spondylolisthesis", "upper_endplate", "lower_endplate",
    ]
    lead = [c for c in identity + gradings if c in frame.columns]
    rest = [c for c in frame.columns if c not in lead]
    return frame[lead + rest]


def measurement_summary(analysis: pd.DataFrame) -> dict:
    """Summary statistics of the key measurements, for the report."""
    columns = [
        "area_mm2", "height_mm_central", "height_mm_anterior", "height_mm_posterior",
        "height_mm_mean", "height_mm_min", "ap_extent_mm",
        "height_ratio_to_series_median", "height_ratio_to_neighbours",
        "disc_to_vertebra_height_ratio", "height_ap_asymmetry",
        "vertebral_ap_offset_mm", "canal_width_at_disc_mm",
    ]
    out: dict = {}
    for column in columns:
        if column not in analysis.columns:
            continue
        values = analysis[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        out[column] = {
            "n": int(len(values)),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "p25": round(float(values.quantile(0.25)), 4),
            "median": round(float(values.median()), 4),
            "p75": round(float(values.quantile(0.75)), 4),
            "max": round(float(values.max()), 4),
        }
    return out


def render_figures(analysis: pd.DataFrame) -> None:
    """Measurement distributions and their relationship to Pfirrmann grade."""
    import matplotlib.pyplot as plt

    graded = analysis[analysis["has_grading"]]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    # Row 1: measurement distributions.
    for axis, column, label in zip(
        axes[0],
        ["height_mm_central", "area_mm2", "ap_extent_mm"],
        ["central disc height (mm)", "disc area (mm2)", "AP extent (mm)"],
    ):
        axis.hist(analysis[column].dropna(), bins=50, color="#4c72b0",
                  edgecolor="black", linewidth=0.4)
        axis.set_xlabel(label)
        axis.set_ylabel("disc records")
        axis.set_title(f"Distribution of {label}", fontsize=10)

    # Row 2: measurements against Pfirrmann grade. Descriptive only - this is
    # measured association, not a model and not a severity rule.
    for axis, column, label in zip(
        axes[1],
        ["height_mm_central", "height_ratio_to_series_median", "area_mm2"],
        ["central disc height (mm)", "height / series median", "disc area (mm2)"],
    ):
        grades = sorted(graded["pfirrmann_grade"].dropna().unique())
        data = [graded.loc[graded["pfirrmann_grade"] == g, column].dropna()
                for g in grades]
        axis.boxplot(data, tick_labels=[f"{int(g)}" for g in grades], showfliers=False)
        axis.set_xlabel("Pfirrmann grade (dataset annotation)")
        axis.set_ylabel(label)
        axis.set_title(f"{label} by Pfirrmann grade", fontsize=10)
        axis.grid(alpha=0.25, axis="y")

    figure.suptitle(
        "Stage C disc measurements (from ground-truth masks, 1.0 mm/px)\n"
        "Bottom row: measured association with the annotated Pfirrmann grade - "
        "descriptive, not a model output",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = VISUALIZATIONS_DIR / "disc_measurements.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
