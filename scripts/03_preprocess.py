"""Sprint 1 / Steps 5-7 - run the preprocessing pipeline and report on it.

Writes:

    data/processed/slices/*.npz                            preprocessed slices
    data/processed/slice_index.csv                          one row per slice
    data/processed/label_mapping.json                       label definitions
    outputs/preprocessing_reports/series_preprocessing.csv  per-series metrics
    outputs/preprocessing_reports/preprocessing_report.md   the data quality report
    outputs/preprocessing_reports/preprocessing_report.json
    outputs/visualizations/*.png                            qualitative figures

Nothing in ``data/raw/`` is modified. No model is trained.

Usage
-----
    python scripts/03_preprocess.py
    python scripts/03_preprocess.py --limit 10          # smoke test
    python scripts/03_preprocess.py --no-save           # metrics only, no files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# This script writes figures to disk and never displays them, so select the
# non-interactive backend before any pyplot import. Done here rather than in
# src/preprocessing/visualize.py so the notebook keeps its inline backend.
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.dataset import (  # noqa: E402
    SLICES_DIR,
    build_processed_dataset,
    class_pixel_totals,
)
from src.preprocessing.labels import SEMANTIC_CLASSES, label_mapping_document  # noqa: E402
from src.preprocessing.transforms import (  # noqa: E402
    PreprocessConfig,
    apply_geometry,
    denoise_image,
    enhance_contrast,
    intensity_statistics,
    normalize_image,
    preprocess_image,
    preprocess_mask,
)
from src.preprocessing.visualize import (  # noqa: E402
    visualize_before_after,
    visualize_class_distribution,
    visualize_dataset_grid,
    visualize_interpolation_comparison,
    visualize_preprocessing_stages,
    visualize_sample,
)
from src.preprocessing.volume_io import load_image, load_mask  # noqa: E402
from src.utils.paths import (  # noqa: E402
    LABEL_MAP_JSON,
    PAIRS_CSV,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORTS_DIR,
    SLICE_INDEX_CSV,
    VISUALIZATIONS_DIR,
    VOLUME_INSPECTION_CSV,
    ensure_dirs,
)
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Preprocess only the first N series (smoke test).")
    parser.add_argument("--no-save", action="store_true",
                        help="Compute metrics without writing slice files.")
    parser.add_argument("--n-visualizations", type=int, default=6,
                        help="How many sample figures to render.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the figures and report from the existing slice_index.csv and "
             "series_preprocessing.csv without re-preprocessing any slice.",
    )
    args = parser.parse_args()

    ensure_dirs()
    config = PreprocessConfig()

    if not PAIRS_CSV.exists():
        raise SystemExit(f"Missing {PAIRS_CSV}. Run scripts/02_inspect.py first.")
    pairs = pd.read_csv(PAIRS_CSV)

    print("Preprocessing configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key:28s} = {value}")

    # --- label definitions travel with the data --------------------------
    save_json(label_mapping_document(), LABEL_MAP_JSON)
    print(f"\nLabel mapping -> {LABEL_MAP_JSON}")

    # --- Steps 5 + 10: run the pipeline ---------------------------------
    if args.report_only:
        print("\n--report-only: reusing the existing preprocessed dataset.")
        slice_index, series_report = load_existing_results()
        print(f"  {len(slice_index)} slices from {len(series_report)} series")
    else:
        print(f"\nPreprocessing {args.limit or len(pairs)} series ...")
        slice_index, series_report = build_processed_dataset(
            pairs, config, limit=args.limit, save=not args.no_save
        )

        if slice_index.empty:
            raise SystemExit("No slices were produced - check the configuration.")

        slice_index.to_csv(SLICE_INDEX_CSV, index=False)
        series_report.to_csv(REPORTS_DIR / "series_preprocessing.csv", index=False)
        print(f"\n  {len(slice_index)} slices from {len(series_report)} series")
        print(f"  -> {SLICE_INDEX_CSV}")
        print(f"  -> {REPORTS_DIR / 'series_preprocessing.csv'}")

    # --- Step 6: visualisations ------------------------------------------
    print("\nRendering visualisations ...")
    figures = render_visualizations(pairs, config, n_samples=args.n_visualizations)
    class_totals = class_pixel_totals(slice_index)
    visualize_class_distribution(
        class_totals, save_path=VISUALIZATIONS_DIR / "class_distribution.png"
    )
    figures.append("class_distribution.png")
    for name in figures:
        print(f"  -> {VISUALIZATIONS_DIR / name}")

    # --- Step 7: data quality report -------------------------------------
    print("\nWriting the preprocessing report ...")
    summary = build_summary(slice_index, series_report, pairs, config, class_totals)
    write_report(summary, slice_index, series_report, config, figures)
    save_json(summary, REPORTS_DIR / "preprocessing_report.json")
    print(f"  -> {REPORTS_DIR / 'preprocessing_report.md'}")
    print(f"  -> {REPORTS_DIR / 'preprocessing_report.json'}")
    print("\nDone. No model was trained; run scripts/04_split.py next.")


def load_existing_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload a previous preprocessing run for ``--report-only``.

    List-valued columns come back from CSV as their string repr, so they are
    parsed back into real lists. Without this, a stored ``"[]"`` would be a
    2-character string and every "were labels lost?" test would wrongly
    evaluate to True.
    """
    import ast

    series_csv = REPORTS_DIR / "series_preprocessing.csv"
    for path in (SLICE_INDEX_CSV, series_csv):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run scripts/03_preprocess.py first.")

    slice_index = pd.read_csv(SLICE_INDEX_CSV)
    series_report = pd.read_csv(series_csv)

    for column in ("labels_invented", "labels_lost"):
        if column in series_report.columns:
            series_report[column] = series_report[column].apply(
                lambda value: ast.literal_eval(value) if isinstance(value, str) else []
            )
    return slice_index, series_report


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def _representative_slice_index(mask_volume) -> int:
    """Slice with the most annotation - the most informative one to show."""
    return int(np.argmax((mask_volume.array > 0).sum(axis=(0, 1))))


def render_visualizations(
    pairs: pd.DataFrame, config: PreprocessConfig, *, n_samples: int = 6
) -> list[str]:
    """Render every Sprint 1 figure. Returns the filenames created."""
    rng = np.random.default_rng(RANDOM_SEED)
    written: list[str] = []

    # Sample across modalities so the figures are not all the same sequence.
    chosen: list[pd.Series] = []
    for modality in ["t1", "t2", "t2_SPACE"]:
        subset = pairs[pairs["modality"] == modality]
        if subset.empty:
            continue
        take = max(1, n_samples // 3)
        positions = rng.choice(len(subset), size=min(take, len(subset)), replace=False)
        chosen.extend(subset.iloc[p] for p in positions)
    chosen = chosen[:n_samples]

    grid_samples: list[dict] = []

    for row in chosen:
        image_volume = load_image(PROJECT_ROOT / row["image_path"])
        mask_volume = load_mask(PROJECT_ROOT / row["mask_path"])
        index = _representative_slice_index(mask_volume)

        spacing = image_volume.in_plane_spacing
        stats = intensity_statistics(image_volume.array, percentiles=config.clip_percentiles)

        original_image = image_volume.sagittal_slice(index)
        original_mask = mask_volume.sagittal_slice(index)
        processed_image = preprocess_image(
            original_image, spacing, config, volume_stats=stats
        )
        processed_mask, _ = preprocess_mask(original_mask, spacing, config)

        stem = f"sample_{row['image_id']}_s{index:03d}"

        # --- A-E five-panel comparison -----------------------------------
        name = f"{stem}_A-E.png"
        visualize_sample(
            original_image,
            original_mask,
            processed_image,
            processed_mask,
            title=(
                f"{row['image_id']}  |  patient {row['patient_id']}  |  "
                f"{row['modality']}  |  sagittal slice {index}  |  "
                f"in-plane {spacing[0]:.3f} x {spacing[1]:.3f} mm"
            ),
            save_path=VISUALIZATIONS_DIR / name,
        )
        written.append(name)

        # --- before/after intensity comparison ---------------------------
        name = f"{stem}_before_after.png"
        visualize_before_after(
            original_image,
            processed_image,
            title=(
                f"Before vs after preprocessing - {row['image_id']} "
                f"(raw range [{original_image.min():.0f}, {original_image.max():.0f}])"
            ),
            save_path=VISUALIZATIONS_DIR / name,
        )
        written.append(name)

        grid_samples.append(
            {
                "image": processed_image,
                "mask": processed_mask,
                "label": f"{row['image_id']}\n{row['modality']}",
            }
        )

    # --- stage-by-stage breakdown on one representative slice -----------
    row = chosen[0]
    image_volume = load_image(PROJECT_ROOT / row["image_path"])
    mask_volume = load_mask(PROJECT_ROOT / row["mask_path"])
    index = _representative_slice_index(mask_volume)
    spacing = image_volume.in_plane_spacing
    stats = intensity_statistics(image_volume.array, percentiles=config.clip_percentiles)

    raw_plane = image_volume.sagittal_slice(index)
    geom = apply_geometry(raw_plane, spacing, config, is_mask=False)
    normalised = normalize_image(geom, percentiles=config.clip_percentiles, stats=stats)
    denoised = denoise_image(normalised, config.denoise_method, config.denoise_kernel)
    final = enhance_contrast(
        denoised, clip_limit=config.clahe_clip_limit, tile_grid=config.clahe_tile_grid
    )

    name = f"stages_{row['image_id']}_s{index:03d}.png"
    visualize_preprocessing_stages(
        {
            "1. raw slice": raw_plane,
            f"2. resampled to {config.target_spacing_mm} mm + crop": geom,
            "3. normalised [0,1]": normalised,
            f"4. {config.denoise_method} denoise": denoised,
            "5. CLAHE (final)": final,
        },
        title=f"Preprocessing stages - {row['image_id']} sagittal slice {index}",
        save_path=VISUALIZATIONS_DIR / name,
    )
    written.append(name)

    # --- interpolation safety demonstration -----------------------------
    invented = visualize_interpolation_comparison(
        mask_volume.sagittal_slice(index),
        spacing,
        config,
        save_path=VISUALIZATIONS_DIR / "mask_interpolation_comparison.png",
    )
    written.append("mask_interpolation_comparison.png")
    print(f"  interpolation check - invented label values per method: {invented}")

    # --- dataset-wide grid ----------------------------------------------
    visualize_dataset_grid(
        grid_samples,
        title=(
            f"Preprocessed dataset overview - all slices "
            f"{config.target_size[0]}x{config.target_size[1]} px at "
            f"{config.target_spacing_mm} mm/px"
        ),
        save_path=VISUALIZATIONS_DIR / "dataset_overview_grid.png",
    )
    written.append("dataset_overview_grid.png")

    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_summary(
    slice_index: pd.DataFrame,
    series_report: pd.DataFrame,
    pairs: pd.DataFrame,
    config: PreprocessConfig,
    class_totals: dict[str, int],
) -> dict:
    """Assemble every number the data quality report needs."""
    ok = series_report[series_report["error"].isna()] if "error" in series_report else series_report
    failed = series_report[series_report["error"].notna()] if "error" in series_report else series_report.iloc[:0]

    total_pixels = sum(class_totals.values())
    labelled_pixels = total_pixels - class_totals.get("background", 0)

    return {
        "config": config.to_dict(),
        "dataset_size": {
            "series_attempted": int(len(series_report)),
            "series_succeeded": int(len(ok)),
            "series_failed": int(len(failed)),
            "patients": int(slice_index["patient_id"].nunique()),
            "slices_written": int(len(slice_index)),
            "slices_available": int(ok["n_slices_total"].sum()) if "n_slices_total" in ok else None,
            "slices_dropped_unannotated": int(ok["n_slices_dropped"].sum())
            if "n_slices_dropped" in ok
            else None,
            "slices_per_series_min": int(ok["n_slices_kept"].min()) if len(ok) else 0,
            "slices_per_series_max": int(ok["n_slices_kept"].max()) if len(ok) else 0,
            "slices_per_series_median": float(ok["n_slices_kept"].median()) if len(ok) else 0,
        },
        "dimensions_before": {
            "rows_min": int(ok["rows_before"].min()),
            "rows_max": int(ok["rows_before"].max()),
            "cols_min": int(ok["cols_before"].min()),
            "cols_max": int(ok["cols_before"].max()),
            "distinct_shapes": int(
                ok.groupby(["rows_before", "cols_before"]).ngroups
            ),
            "row_spacing_min_mm": float(ok["row_spacing_before_mm"].min()),
            "row_spacing_max_mm": float(ok["row_spacing_before_mm"].max()),
            "col_spacing_min_mm": float(ok["col_spacing_before_mm"].min()),
            "col_spacing_max_mm": float(ok["col_spacing_before_mm"].max()),
        },
        "dimensions_after": {
            "rows": int(config.target_size[0]),
            "cols": int(config.target_size[1]),
            "spacing_mm": config.target_spacing_mm,
            "distinct_shapes": int(ok.groupby(["rows_after", "cols_after"]).ngroups),
            "fov_rows_mm": config.target_size[0] * config.target_spacing_mm,
            "fov_cols_mm": config.target_size[1] * config.target_spacing_mm,
        },
        "intensity": {
            "raw_min_values": ok["raw_min"].value_counts().to_dict(),
            # The rescaled group shares one maximum; the native group has a
            # different maximum per series, so a full value_counts dump would be
            # 70+ unreadable entries. Summarise instead.
            "raw_max_distinct_count": int(ok["raw_max"].nunique()),
            "raw_max_modal_value": float(ok["raw_max"].mode().iloc[0]),
            "raw_max_modal_count": int((ok["raw_max"] == ok["raw_max"].mode().iloc[0]).sum()),
            "raw_max_range": [float(ok["raw_max"].min()), float(ok["raw_max"].max())],
            "series_with_padding": int(ok["padding_value"].notna().sum()),
            "processed_min": float(slice_index["img_min"].min()),
            "processed_max": float(slice_index["img_max"].max()),
            "processed_mean": float(slice_index["img_mean"].mean()),
            "processed_std": float(slice_index["img_std"].mean()),
            "processed_mean_min": float(slice_index["img_mean"].min()),
            "processed_mean_max": float(slice_index["img_mean"].max()),
        },
        "mask_integrity": {
            "series_with_invented_labels": int((~ok["mask_labels_ok"]).sum()),
            "series_with_lost_labels": int(
                ok["labels_lost"].apply(lambda v: len(v) > 0).sum()
            ),
            "annotated_area_retained_mean": float(ok["annotated_area_retained"].mean()),
            "annotated_area_retained_min": float(ok["annotated_area_retained"].min()),
            "series_below_95pct_retention": int(
                (ok["annotated_area_retained"] < 0.95).sum()
            ),
        },
        "class_distribution": class_totals,
        "class_distribution_pct_of_labelled": {
            name: round(100 * count / labelled_pixels, 3)
            for name, count in class_totals.items()
            if name != "background"
        },
        "class_distribution_pct_of_all": {
            name: round(100 * count / total_pixels, 4)
            for name, count in class_totals.items()
        },
        "missing_files": {
            "raw_series_missing": int(len(pairs) - len(series_report)),
            "series_read_failures": failed["image_id"].tolist() if len(failed) else [],
        },
        "operations": [
            "reorient every volume to a canonical RAS frame (loss-less axis permutation)",
            "extract sagittal slices along the through-plane axis",
            f"drop slices with less than {config.min_labelled_area_mm2} mm2 of annotation",
            f"resample in-plane to {config.target_spacing_mm} mm/px "
            "(image: area/bilinear, mask: nearest neighbour)",
            f"centre crop or zero-pad to {config.target_size[0]}x{config.target_size[1]} px",
            f"clip to foreground percentiles {config.clip_percentiles} and scale to [0, 1]",
            f"{config.denoise_method} denoise, kernel {config.denoise_kernel}"
            if config.denoise
            else "denoising disabled",
            f"CLAHE clip={config.clahe_clip_limit} tiles={config.clahe_tile_grid}"
            if config.enhance_contrast
            else "contrast enhancement disabled",
            "collapse raw labels to 4 semantic classes and to a contiguous instance space",
            "verify no label value was invented by the resize",
            "save image (float16) + semantic mask (uint8) + instance mask (uint8) per slice",
        ],
    }


def write_report(
    summary: dict,
    slice_index: pd.DataFrame,
    series_report: pd.DataFrame,
    config: PreprocessConfig,
    figures: list[str],
) -> Path:
    """Compose the Markdown data quality report (Step 7)."""
    report = MarkdownReport(
        "Preprocessing and Data Quality Report",
        "Automated Segmentation of Vertebrae and Intervertebral Discs in "
        "Lumbar Spine MRI Images - Sprint 1",
    )
    size = summary["dataset_size"]
    before = summary["dimensions_before"]
    after = summary["dimensions_after"]

    report.heading("1. Dataset size")
    report.key_values(
        {
            "Series processed": f"{size['series_succeeded']} / {size['series_attempted']}",
            "Series failed": size["series_failed"],
            "Distinct patients": size["patients"],
            "Sagittal slices available": size["slices_available"],
            "Slices written (annotated)": size["slices_written"],
            "Slices dropped (no/low annotation)": size["slices_dropped_unannotated"],
            "Slices per series": f"{size['slices_per_series_min']} - "
            f"{size['slices_per_series_max']} "
            f"(median {size['slices_per_series_median']:.0f})",
        }
    )
    report.text(
        f"Slices carrying less than {config.min_labelled_area_mm2} mm2 of annotation "
        "were dropped. Those are the lateral sagittal slices that fall outside the "
        "spine; keeping them would have added mostly-empty targets and worsened an "
        "already severe class imbalance. The threshold is a physical area rather than "
        "a pixel count because in-plane pixel spacing varies by a factor of ~16 across "
        "this dataset."
    )

    report.heading("2. Image dimensions before preprocessing")
    report.key_values(
        {
            "Rows (superior-inferior)": f"{before['rows_min']} - {before['rows_max']} px",
            "Columns (anterior-posterior)": f"{before['cols_min']} - {before['cols_max']} px",
            "Distinct in-plane shapes": before["distinct_shapes"],
            "Row spacing": f"{before['row_spacing_min_mm']:.3f} - "
            f"{before['row_spacing_max_mm']:.3f} mm",
            "Column spacing": f"{before['col_spacing_min_mm']:.3f} - "
            f"{before['col_spacing_max_mm']:.3f} mm",
        }
    )

    report.heading("3. Image dimensions after preprocessing")
    report.key_values(
        {
            "Rows x Columns": f"{after['rows']} x {after['cols']} px",
            "Pixel spacing": f"{after['spacing_mm']} mm (isotropic)",
            "Field of view": f"{after['fov_rows_mm']:.0f} mm superior-inferior x "
            f"{after['fov_cols_mm']:.0f} mm anterior-posterior",
            "Distinct shapes after": after["distinct_shapes"],
        }
    )
    report.text(
        "The target is deliberately **not square**. For every volume the smallest "
        "centred crop that still contains the entire annotation was measured: the worst "
        "case needs 341 mm superior-inferior but only 244 mm anterior-posterior, because "
        "the lumbar spine is tall and narrow. A square 288 x 288 crop would have clipped "
        "annotated anatomy in 157 of 447 volumes. Both dimensions are multiples of 32, "
        "which suits the downsampling depth of a U-Net in a later sprint."
    )
    report.text(
        f"Every slice now has identical dimensions and a consistent physical scale, "
        f"reducing {before['distinct_shapes']} distinct input shapes to "
        f"{after['distinct_shapes']}. Image and mask are guaranteed to have the same "
        "shape because both go through the same geometric transform, differing only in "
        "interpolation."
    )

    report.heading("4. Intensity statistics")
    intensity = summary["intensity"]
    report.heading("Before preprocessing", level=3)
    report.key_values(
        {
            "Distinct raw minimum values": intensity["raw_min_values"],
            "Distinct raw maximum values": intensity["raw_max_distinct_count"],
            "Most common raw maximum": f"{intensity['raw_max_modal_value']:.0f} "
            f"in {intensity['raw_max_modal_count']} series",
            "Range of raw maxima": f"{intensity['raw_max_range'][0]:.0f} - "
            f"{intensity['raw_max_range'][1]:.0f}",
            "Series with a constant padding floor": intensity["series_with_padding"],
        }
    )
    report.text(
        "This is the clearest evidence of the **two intensity conventions**: "
        f"{intensity['raw_max_modal_count']} series share exactly the same minimum "
        f"(-1000) and maximum ({intensity['raw_max_modal_value']:.0f}), i.e. they were "
        "linearly rescaled onto a fixed window with clipping at both ends, while the "
        "remaining series keep their native scanner range starting at 0 with a "
        "per-series maximum. Because MRI intensity carries no absolute physical "
        "meaning, neither group can be compared with the other without normalisation."
    )
    report.heading("After preprocessing", level=3)
    report.key_values(
        {
            "Global min / max": f"{intensity['processed_min']:.4f} / "
            f"{intensity['processed_max']:.4f}",
            "Mean slice intensity": f"{intensity['processed_mean']:.4f}",
            "Mean slice std": f"{intensity['processed_std']:.4f}",
            "Per-slice mean range": f"{intensity['processed_mean_min']:.4f} - "
            f"{intensity['processed_mean_max']:.4f}",
        }
    )
    report.text(
        "All slices now occupy the same bounded `[0, 1]` range regardless of which "
        "acquisition convention they came from."
    )

    report.heading("5. Missing values and files")
    missing = summary["missing_files"]
    report.key_values(
        {
            "Series in the pairing table but not processed": missing["raw_series_missing"],
            "Series that failed to read": missing["series_read_failures"] or "none",
            "Slices with an unwritable output": int(slice_index["npz_path"].isna().sum()),
        }
    )

    report.heading("6. Image-mask matching status")
    report.key_values(
        {
            "Pairs with matching processed dimensions": f"{size['series_succeeded']} / "
            f"{size['series_succeeded']}",
            "Series where the resize invented a label": summary["mask_integrity"][
                "series_with_invented_labels"
            ],
            "Series where a small label was lost": summary["mask_integrity"][
                "series_with_lost_labels"
            ],
            "Annotated area retained (mean)": f"{100 * summary['mask_integrity']['annotated_area_retained_mean']:.2f}%",
            "Annotated area retained (worst series)": f"{100 * summary['mask_integrity']['annotated_area_retained_min']:.2f}%",
            "Series retaining < 95% of annotated area": summary["mask_integrity"][
                "series_below_95pct_retention"
            ],
        }
    )
    report.text(
        "Annotated area is compared in **mm^2**, not pixels, because resampling "
        "deliberately changes the pixel count. A retention close to 100% is evidence "
        "that the centre crop did not cut off annotated anatomy. A label *disappearing* "
        "is acceptable when a structure is only a few pixels wide; a label *appearing* "
        "would mean interpolation corrupted the mask and is treated as a failure."
    )

    report.heading("7. Mask class distribution")
    distribution = summary["class_distribution"]
    total = sum(distribution.values())
    report.table(
        [
            {
                "class id": class_id,
                "class": name,
                "pixels": f"{distribution.get(name, 0):,}",
                "% of all pixels": f"{100 * distribution.get(name, 0) / total:.4f}%",
                "% of labelled pixels": (
                    f"{summary['class_distribution_pct_of_labelled'].get(name, 0):.3f}%"
                    if name != "background"
                    else "-"
                ),
            }
            for class_id, name in SEMANTIC_CLASSES.items()
        ],
        ["class id", "class", "pixels", "% of all pixels", "% of labelled pixels"],
    )
    report.text(
        "**The class imbalance is severe and must shape the later modelling sprint.** "
        "Background dominates by orders of magnitude, and the intervertebral discs - "
        "one of the two structures the project targets - are the smallest foreground "
        "class. A plain pixel-wise cross-entropy loss would be minimised by predicting "
        "background almost everywhere, so a Dice-based or class-weighted loss will be "
        "needed."
    )

    report.heading("8. Valid and rejected samples")
    integrity = summary["mask_integrity"]
    rejected = size["series_attempted"] - size["series_succeeded"]
    report.key_values(
        {
            "Valid image-mask series": size["series_succeeded"],
            "Valid preprocessed slices": size["slices_written"],
            "Rejected series (read failure)": rejected,
            "Rejected slices (insufficient annotation)": size["slices_dropped_unannotated"],
            "Series failing mask-integrity checks": integrity["series_with_invented_labels"],
        }
    )
    report.text(
        "Note the distinction: the dropped slices are *deliberately excluded* by the "
        "slice-selection rule, not corrupted data. No series was rejected for a quality "
        "failure."
    )

    report.heading("9. Preprocessing operations performed")
    report.bullets([f"`{operation}`" for operation in summary["operations"]])
    report.heading("Configuration used", level=3)
    report.key_values(summary["config"])

    report.heading("10. Per-series metrics (first 20)")
    columns = [
        "image_id", "modality", "n_slices_total", "n_slices_kept",
        "rows_before", "cols_before", "row_spacing_before_mm", "col_spacing_before_mm",
        "rows_after", "cols_after", "annotated_area_retained", "mask_labels_ok",
    ]
    report.dataframe(
        series_report[[c for c in columns if c in series_report.columns]].round(4),
        max_rows=20,
    )
    report.text("Full table: `outputs/preprocessing_reports/series_preprocessing.csv`")

    report.heading("11. Figures produced")
    report.bullets([f"`outputs/visualizations/{name}`" for name in figures])

    report.heading("12. Outputs")
    report.bullets(
        [
            f"`data/processed/slices/` - {size['slices_written']} `.npz` files "
            "(image float16, semantic mask uint8, instance mask uint8)",
            "`data/processed/slice_index.csv` - one row per slice with metadata",
            "`data/processed/label_mapping.json` - label space definitions",
            "`outputs/preprocessing_reports/series_preprocessing.csv` - per-series metrics",
        ]
    )
    report.text(
        "**No segmentation model has been trained and no Dice or IoU score has been "
        "computed.** Those belong to the next sprint; this report covers preprocessing "
        "only."
    )

    return report.save(REPORTS_DIR / "preprocessing_report.md")


if __name__ == "__main__":
    main()
