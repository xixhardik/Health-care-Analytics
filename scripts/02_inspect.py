"""Sprint 1 / Step 2 + 4 + 7 - inspect the dataset and build the pairing table.

Reads every extracted volume once and writes:

    outputs/preprocessing_reports/dataset_inspection.md    human-readable report
    outputs/preprocessing_reports/dataset_inspection.json   machine-readable summary
    outputs/preprocessing_reports/volume_inspection.csv     one row per series
    outputs/preprocessing_reports/image_mask_pairs.csv      the pairing table

Nothing in ``data/raw/`` is modified.

Usage
-----
    python scripts/02_inspect.py
    python scripts/02_inspect.py --limit 20      # quick smoke test
    python scripts/02_inspect.py --skip-hashes   # skip duplicate detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.labels import describe_raw_label  # noqa: E402
from src.preprocessing.pairing import (  # noqa: E402
    describe_pairing,
    load_gradings,
    load_overview,
    pair_images_and_masks,
)
from src.preprocessing.validate import (  # noqa: E402
    check_raw_files_present,
    find_duplicate_files,
    summarise_duplicates,
    summarise_inspection,
    validate_dataset,
)
from src.utils.paths import (  # noqa: E402
    EXTRACTED_IMAGES_DIR,
    EXTRACTED_MASKS_DIR,
    PAIRS_CSV,
    REPORTS_DIR,
    VOLUME_INSPECTION_CSV,
    ensure_dirs,
)
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Inspect only the first N series (smoke test).")
    parser.add_argument("--skip-hashes", action="store_true",
                        help="Skip content hashing (duplicate detection).")
    args = parser.parse_args()

    ensure_dirs()

    # --- Step 2a: raw files present? -------------------------------------
    print("[1/6] Checking raw files ...")
    raw_status = check_raw_files_present()
    for name, info in raw_status.items():
        print(f"  {name:28s} present={info['present']} size={info['size_mb']} MB")

    # --- Step 2b: metadata ------------------------------------------------
    print("[2/6] Loading metadata CSVs ...")
    overview = load_overview()
    gradings = load_gradings()
    print(f"  overview.csv  : {overview.shape[0]} rows x {overview.shape[1]} cols")
    print(f"  gradings.csv  : {gradings.shape[0]} rows x {gradings.shape[1]} cols")

    # --- Step 4: pairing --------------------------------------------------
    print("[3/6] Pairing images and masks ...")
    pairs, pairing_issues = pair_images_and_masks(overview=overview, gradings=gradings)
    print(describe_pairing(pairs, pairing_issues, n_examples=5))
    pairs.to_csv(PAIRS_CSV, index=False)
    print(f"  -> {PAIRS_CSV}")

    # --- Step 2c: duplicates ----------------------------------------------
    if args.skip_hashes:
        print("[4/6] Duplicate detection skipped (--skip-hashes).")
        image_dupes = mask_dupes = {}
    else:
        print("[4/6] Hashing files for duplicate detection (reads ~3.8 GB) ...")
        image_dupes = find_duplicate_files(EXTRACTED_IMAGES_DIR)
        mask_dupes = find_duplicate_files(EXTRACTED_MASKS_DIR)
        print(f"  duplicate image groups: {len(image_dupes)}")
        print(f"  duplicate mask groups : {len(mask_dupes)}")
    image_dupe_summary = summarise_duplicates(image_dupes)
    mask_dupe_summary = summarise_duplicates(mask_dupes)

    # --- Step 2d: per-volume inspection -----------------------------------
    print("[5/6] Inspecting every volume pair (this reads the full dataset) ...")
    inspection = validate_dataset(pairs, limit=args.limit)
    inspection.to_csv(VOLUME_INSPECTION_CSV, index=False)
    print(f"  -> {VOLUME_INSPECTION_CSV}")

    summary = summarise_inspection(inspection, pairs)
    summary["raw_files"] = raw_status
    summary["pairing_issues"] = pairing_issues
    summary["duplicates"] = {"images": image_dupe_summary, "masks": mask_dupe_summary}
    summary["metadata"] = {
        "overview_rows": int(len(overview)),
        "overview_columns": int(overview.shape[1]),
        "gradings_rows": int(len(gradings)),
        "gradings_patients": int(gradings["patient_id"].nunique()),
        "official_subset_counts": overview["subset"].value_counts().to_dict(),
        "sex_counts": overview["sex"].value_counts(dropna=False).to_dict(),
        "columns_with_missing_values": {
            k: int(v) for k, v in overview.isna().sum().items() if v > 0
        },
    }

    # --- Step 7: write the report ----------------------------------------
    print("[6/6] Writing reports ...")
    write_report(summary, inspection, pairs, overview, gradings, pairing_issues)
    save_json(summary, REPORTS_DIR / "dataset_inspection.json")
    print(f"  -> {REPORTS_DIR / 'dataset_inspection.md'}")
    print(f"  -> {REPORTS_DIR / 'dataset_inspection.json'}")
    print("Done.")


def write_report(
    summary: dict,
    inspection: pd.DataFrame,
    pairs: pd.DataFrame,
    overview: pd.DataFrame,
    gradings: pd.DataFrame,
    pairing_issues: dict,
) -> Path:
    """Compose the Markdown dataset inspection report."""
    ok = inspection[inspection["read_error"].isna()]
    report = MarkdownReport(
        "Dataset Inspection Report",
        "Automated Segmentation of Vertebrae and Intervertebral Discs in "
        "Lumbar Spine MRI Images - Sprint 1",
    )

    # ---------------- headline numbers ----------------
    report.heading("1. Headline figures")
    report.key_values(
        {
            "Total image volumes": summary["n_series"],
            "Total mask volumes": summary["n_series"],
            "Readable volumes": summary["n_series_readable"],
            "Read errors": summary["n_read_errors"],
            "Distinct patients": summary["n_patients"],
            "Series per modality": summary["modality_counts"],
            "Image format": summary["image_formats"],
            "Mask format": summary["mask_formats"],
            "Image voxel dtype": summary["image_dtypes"],
            "Mask voxel dtype": summary["mask_dtypes"],
            "Total sagittal slices": summary["slices"]["total"],
            "Slices containing any label": summary["slices"]["with_any_label"],
            "Slices with >=50 labelled px": summary["slices"]["usable"],
        }
    )

    # ---------------- formats & geometry ----------------
    report.heading("2. Format, dimensions and geometry")
    report.text(
        "Volumes are stored as **MetaImage** (`.mha`) 3-D sagittal series, not 2-D "
        "image files. Every series has one mask volume of the same name."
    )
    report.key_values(
        {
            "In-plane rows (sup-inf)": f"{summary['in_plane_rows']['min']} - "
            f"{summary['in_plane_rows']['max']} px "
            f"({summary['in_plane_rows']['unique_count']} distinct values)",
            "In-plane cols (ant-post)": f"{summary['in_plane_cols']['min']} - "
            f"{summary['in_plane_cols']['max']} px "
            f"({summary['in_plane_cols']['unique_count']} distinct values)",
            "Slices per volume": f"{summary['slices']['per_volume_min']} - "
            f"{summary['slices']['per_volume_max']} "
            f"(median {summary['slices']['per_volume_median']:.0f})",
            "Row spacing": f"{summary['spacing_mm']['row_min']:.3f} - "
            f"{summary['spacing_mm']['row_max']:.3f} mm",
            "Col spacing": f"{summary['spacing_mm']['col_min']:.3f} - "
            f"{summary['spacing_mm']['col_max']:.3f} mm",
            "Slice spacing": f"{summary['spacing_mm']['slice_min']:.3f} - "
            f"{summary['spacing_mm']['slice_max']:.3f} mm",
            "Native orientations": summary["native_orientations"],
        }
    )
    report.text(
        "**Finding - image dimensions are not constant.** Both the matrix size and the "
        "physical voxel size vary between series, so a fixed resize in pixels would "
        "place the same vertebra at different physical scales. Preprocessing therefore "
        "resamples to a common mm/pixel scale before cropping to a fixed matrix."
    )
    report.text(
        "**Finding - storage orientation is not constant.** The 2-D TSE series are "
        "stored `LPS` while the 3-D `t2_SPACE` series are stored `PIR`, meaning the "
        "array axis that steps through sagittal slices differs between files. Every "
        "volume is reoriented to a canonical `RAS` frame on load; this is a pure axis "
        "permutation plus flips, so it is loss-less and safe for label masks."
    )

    report.heading("Per-series geometry (first 15 series)", level=3)
    geom_cols = [
        "image_id", "modality", "img_native_orientation", "img_rows", "img_cols",
        "img_n_slices", "img_row_spacing_mm", "img_col_spacing_mm",
        "img_slice_spacing_mm", "img_fov_rows_mm", "img_fov_cols_mm",
    ]
    report.dataframe(ok[[c for c in geom_cols if c in ok.columns]], max_rows=15)

    # ---------------- labels ----------------
    report.heading("3. Mask labels and classes")
    report.text(
        "Masks are integer label volumes (not binary, not one-hot). The complete "
        "label vocabulary observed across the dataset is:"
    )
    report.table(
        [
            {"raw value": v, "meaning": describe_raw_label(v)}
            for v in summary["label_vocabulary"]
        ],
        ["raw value", "meaning"],
    )
    report.key_values(
        {
            "Distinct raw label values": len(summary["label_vocabulary"]),
            "Undocumented label values found": summary["unknown_labels"] or "none",
            "Vertebra labels per volume": f"{int(ok['n_vertebra_labels'].min())} - "
            f"{int(ok['n_vertebra_labels'].max())} "
            f"(median {ok['n_vertebra_labels'].median():.0f})",
            "IVD labels per volume": f"{int(ok['n_ivd_labels'].min())} - "
            f"{int(ok['n_ivd_labels'].max())} "
            f"(median {ok['n_ivd_labels'].median():.0f})",
            "Volumes containing spinal canal": f"{int(ok['has_canal'].sum())} / {len(ok)}",
            "Labelled voxel fraction": f"{ok['labelled_voxel_fraction'].mean()*100:.2f}% "
            "of voxels on average",
        }
    )
    report.text(
        "**Class imbalance is severe**: on average only a few percent of voxels carry "
        "any label at all, and the discs are far smaller than the vertebrae. This is "
        "recorded now because it will drive the loss function choice in a later sprint."
    )
    total_v = float(ok["voxels_vertebrae"].sum())
    total_i = float(ok["voxels_ivd"].sum())
    total_c = float(ok["voxels_canal"].sum())
    total_labelled = total_v + total_i + total_c
    report.heading("Semantic class volume distribution (all volumes pooled)", level=3)
    report.table(
        [
            {"class": "vertebra", "voxels": int(total_v),
             "share of labelled": f"{100*total_v/total_labelled:.1f}%"},
            {"class": "intervertebral_disc", "voxels": int(total_i),
             "share of labelled": f"{100*total_i/total_labelled:.1f}%"},
            {"class": "spinal_canal", "voxels": int(total_c),
             "share of labelled": f"{100*total_c/total_labelled:.1f}%"},
        ],
        ["class", "voxels", "share of labelled"],
    )

    # ---------------- intensity ----------------
    report.heading("4. Intensity characteristics")
    report.text(
        "MRI intensities have no absolute physical meaning, and this dataset makes "
        "that concrete: **two incompatible intensity conventions coexist.**"
    )
    report.key_values(
        {
            "Series with a constant padding floor": summary["intensity_conventions"][
                "with_padding_value"
            ],
            "Series without a padding floor": summary["intensity_conventions"][
                "without_padding_value"
            ],
            "Distinct raw minimum values": summary["intensity_conventions"]["raw_min_values"],
            "Distinct raw maximum values": summary["intensity_conventions"]["raw_max_values"],
        }
    )
    report.text(
        "In the padded group a single extreme value (typically -1000) fills the empty "
        "region of the resampled grid and can account for a large share of all voxels, "
        "while a further share sits pinned at the maximum. Any mean/std or min-max "
        "normalisation would be dominated by those two spikes rather than by tissue. "
        "Preprocessing therefore computes statistics over **foreground voxels only** "
        "and clips at robust percentiles."
    )
    intensity_cols = [
        "image_id", "raw_min", "raw_max", "padding_value", "foreground_fraction",
        "saturated_fraction", "fg_p1", "fg_p50", "fg_p99",
    ]
    report.heading("Per-series intensity (first 15 series)", level=3)
    report.dataframe(ok[[c for c in intensity_cols if c in ok.columns]], max_rows=15)
    report.key_values(
        {
            "Mean foreground fraction": f"{ok['foreground_fraction'].mean()*100:.1f}%",
            "Mean saturated fraction": f"{ok['saturated_fraction'].mean()*100:.2f}%",
            "Max saturated fraction": f"{ok['saturated_fraction'].max()*100:.2f}%",
        }
    )

    # ---------------- pairing ----------------
    report.heading("5. Image-mask correspondence")
    report.text(
        "Filenames follow `<patient_id>_<modality>.mha`, and `images/X.mha` pairs with "
        "`masks/X.mha`. Pairing uses that parsed identifier, never directory order, so "
        "a missing file surfaces as an explicit mismatch instead of silently shifting "
        "every subsequent pair."
    )
    report.key_values(
        {
            "Matched pairs": len(pairs),
            "Images without a mask": pairing_issues["images_without_mask"] or "none",
            "Masks without an image": pairing_issues["masks_without_image"] or "none",
            "Filenames not matching the convention": pairing_issues["unparsable_stems"] or "none",
            "Series missing an overview.csv row": pairing_issues["missing_from_overview"] or "none",
            "Image/mask shape disagreements": summary["geometry_mismatches"]["shape"],
            "Image/mask spacing disagreements": summary["geometry_mismatches"]["spacing"],
            "Image/mask orientation disagreements": summary["geometry_mismatches"]["orientation"],
        }
    )
    report.heading("Example verified pairs", level=3)
    report.code(describe_pairing(pairs, pairing_issues, n_examples=5))

    # ---------------- anatomical checks ----------------
    report.heading("6. Anatomical sanity checks")
    report.text(
        "These check that the reorientation actually placed the axes where the code "
        "assumes they are. They are computed from the mask geometry alone."
    )
    checks = summary["anatomical_checks"]
    report.table(
        [
            {
                "check": "spinal canal lies posterior to the vertebral bodies",
                "passed": checks["canal_posterior_pass"],
                "failed": checks["canal_posterior_fail"],
            },
            {
                "check": "vertebra label 1 lies inferior to the highest vertebra label",
                "passed": checks["label1_inferior_pass"],
                "failed": checks["label1_inferior_fail"],
            },
        ],
        ["check", "passed", "failed"],
    )

    # ---------------- duplicates ----------------
    report.heading("7. Duplicate detection")
    report.text("Grouping by SHA-256 of the file contents.")
    for name, info in summary["duplicates"].items():
        report.heading(name, level=3)
        report.key_values(
            {
                "Duplicate groups": info["n_duplicate_groups"],
                "Groups within a single patient": info["n_same_patient_groups"],
                "Groups spanning different patients": info["n_cross_patient_groups"],
                "Examples (same patient)": info["same_patient_examples"] or "none",
                "Cross-patient groups (would be a problem)": info["cross_patient_groups"]
                or "none",
            }
        )
    report.text(
        "**Interpretation.** No image volume is duplicated. A large number of *mask* "
        "volumes are byte-identical, but always within one patient: the T1 and T2 "
        "series of a patient were acquired on the same grid and share a single "
        "annotation. This is a property of the annotation process, not corruption. It "
        "does, however, mean the T1/T2 series of a patient are highly correlated, "
        "which is exactly why the train/val/test split must be made **per patient**."
    )

    # ---------------- metadata ----------------
    report.heading("8. Metadata files")
    meta = summary["metadata"]
    report.key_values(
        {
            "overview.csv": f"{meta['overview_rows']} rows x {meta['overview_columns']} columns "
            "(one row per series)",
            "radiological_gradings.csv": f"{meta['gradings_rows']} rows, "
            f"{meta['gradings_patients']} patients (one row per patient x disc)",
            "Identifier in overview.csv": "`new_file_name` matches the volume filename stem exactly",
            "Identifier in gradings": "`Patient` matches the numeric prefix of the filename",
            "Official subset column": meta["official_subset_counts"],
        }
    )
    report.heading("Relevant metadata columns", level=3)
    report.bullets(
        [
            "`new_file_name` - series identifier, joins to the volume files",
            "`num_vertebrae`, `num_discs` - annotated structure counts per series",
            "`sex`, `subset` - patient/series level attributes",
            "`Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength` - scanner",
            "`MRAcquisitionType`, `ScanningSequence`, `SeriesDescription` - sequence",
            "`PixelSpacing`, `SliceThickness`, `SpacingBetweenSlices` - geometry",
            "`EchoTime`, `RepetitionTime` - contrast weighting",
            "gradings: `IVD label`, `Pfirrman grade`, `Modic`, `Disc herniation`, "
            "`Disc narrowing`, `Disc bulging`, `Spondylolisthesis`, `UP/LOW endplate` "
            "- per-disc radiological gradings (classification targets for a later "
            "sprint, not segmentation targets)",
        ]
    )
    report.heading("Missing values in overview.csv", level=3)
    report.key_values(meta["columns_with_missing_values"] or {"(none)": "-"})
    report.text(
        "**Finding - dirty categorical values.** `sex` is stored with inconsistent "
        f"trailing whitespace ({meta['sex_counts']}), i.e. `'F'` and `'F '` appear as "
        "two distinct values. The loader strips whitespace from all text columns. "
        "The heavily-missing columns are optional DICOM acquisition fields and are not "
        "used by preprocessing."
    )

    # ---------------- problems ----------------
    report.heading("9. Problems found")
    problems = build_problem_list(summary, ok, overview)
    report.bullets(problems)

    return report.save(REPORTS_DIR / "dataset_inspection.md")


def build_problem_list(summary: dict, ok: pd.DataFrame, overview: pd.DataFrame) -> list[str]:
    """Collect the concrete dataset problems worth flagging."""
    problems: list[str] = []

    if summary["n_read_errors"]:
        problems.append(f"{summary['n_read_errors']} volume(s) could not be read.")
    else:
        problems.append("No unreadable volumes: all series loaded successfully.")

    if summary["pairing_issues"]["images_without_mask"] or summary["pairing_issues"]["masks_without_image"]:
        problems.append("Unmatched image/mask files exist (see section 5).")
    else:
        problems.append("No missing or unmatched files: every image has exactly one mask.")

    if summary["unknown_labels"]:
        problems.append(f"Undocumented label values present: {summary['unknown_labels']}.")
    else:
        problems.append("No undocumented label values in any mask.")

    mism = summary["geometry_mismatches"]
    if any(mism.values()):
        problems.append(f"Image/mask geometry disagreements: {mism}.")
    else:
        problems.append("Image and mask geometry agree for every pair.")

    problems.append(
        "Heterogeneous acquisition: matrix size, pixel spacing, slice thickness and "
        "storage orientation all vary between series - handled by resampling to a "
        "common physical scale and reorienting to RAS."
    )
    problems.append(
        "Two incompatible intensity conventions coexist (padded -1000..3096 vs "
        "plain 0..~700) - handled by foreground-restricted percentile normalisation."
    )
    problems.append(
        f"{summary['duplicates']['masks']['n_duplicate_groups']} groups of byte-identical "
        "mask volumes, all within a single patient (shared T1/T2 annotation). Not "
        "corruption, but it makes patient-level splitting mandatory."
    )
    problems.append(
        "`sex` contains trailing-whitespace duplicates in the raw CSV; stripped on load."
    )

    # Series whose annotated structure count is unusually low.
    low = overview[(overview["num_vertebrae"] < 6) | (overview["num_discs"] < 6)]
    if len(low):
        problems.append(
            f"{len(low)} series annotate fewer than 6 vertebrae/discs "
            f"({low['new_file_name'].tolist()}), i.e. a much smaller field of view than "
            "the rest. Kept, but flagged as outliers."
        )

    empty = ok[ok["n_slices_usable"] == 0]
    if len(empty):
        problems.append(
            f"{len(empty)} series have no slice with >=50 labelled pixels: "
            f"{empty['image_id'].tolist()}."
        )
    else:
        problems.append("Every series contains at least one usably-annotated slice.")

    problems.append(
        "The official `subset` column only splits training/validation (no test set) "
        "and is series-level, so Sprint 1 derives its own patient-level 3-way split."
    )
    return problems


if __name__ == "__main__":
    main()
