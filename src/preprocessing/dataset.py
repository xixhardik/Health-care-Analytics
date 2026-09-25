"""Building the preprocessed 2-D slice dataset from the raw 3-D volumes.

The raw data is 3-D sagittal volumes; the stated project target is 2-D
segmentation of vertebrae and intervertebral discs. This module is the bridge:
it walks every series, extracts sagittal slices, preprocesses image and mask
through the *same* geometric transform, validates the result, and writes it to
``data/processed/slices/``.

Storage layout
--------------
One compressed ``.npz`` per slice::

    data/processed/slices/<image_id>_s<NNN>.npz
        image          float16, shape = config.target_size, range [0, 1]
        mask           uint8,   semantic classes 0..3
        mask_instance  uint8,   contiguous instance ids 0..19

Slices are stored **flat**, not inside per-split folders, and the split is
recorded in ``slice_index.csv`` instead. That way the split can be changed or
re-seeded without re-running preprocessing.

``float16`` is used for the image because the normalised range is ``[0, 1]``
and the raw data carries fewer than ~1000 distinct intensity levels, so
half precision is loss-less in practice while halving the dataset size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.labels import to_instance, to_semantic
from src.preprocessing.transforms import (
    PreprocessConfig,
    intensity_statistics,
    preprocess_image,
    preprocess_mask,
)
from src.preprocessing.volume_io import load_image, load_mask
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, relative

SLICES_DIR = PROCESSED_DIR / "slices"

#: Semantic class ids that the per-slice statistics are tracked for.
SEMANTIC_IDS = (0, 1, 2, 3)


def _physical_area_mm2(mask: np.ndarray, row_mm: float, col_mm: float) -> float:
    """Physical area of all labelled pixels, in mm^2.

    Used instead of a raw pixel count so that "did the crop lose anatomy?" can
    be answered across a resampling step that deliberately changes the pixel
    count.
    """
    return float((mask > 0).sum()) * row_mm * col_mm


def preprocess_series(
    image_path: Path | str,
    mask_path: Path | str,
    config: PreprocessConfig,
    *,
    image_id: str,
    output_dir: Path = SLICES_DIR,
    save: bool = True,
) -> tuple[list[dict], dict]:
    """Preprocess every usable sagittal slice of one series.

    Returns
    -------
    slice_records : list[dict]
        One record per written slice (path, class pixel counts, intensity).
    series_record : dict
        Series-level quality metrics, including how much annotated area
        survived the geometric pipeline and whether any label was invented.
    """
    image_volume = load_image(image_path)
    mask_volume = load_mask(mask_path)

    # Volume-level intensity statistics: computed once and reused for every
    # slice so all slices of a series share one intensity scale.
    volume_stats = intensity_statistics(
        image_volume.array, percentiles=config.clip_percentiles
    )

    row_mm, col_mm = image_volume.in_plane_spacing
    target_mm = config.target_spacing_mm
    labelled_per_slice = (mask_volume.array > 0).sum(axis=(0, 1))
    # Physical area, so the selection threshold means the same amount of
    # anatomy regardless of the scanner's pixel spacing.
    labelled_area_per_slice = labelled_per_slice * row_mm * col_mm

    slice_records: list[dict] = []
    area_before = 0.0
    area_after = 0.0
    labels_before: set[int] = set()
    labels_after: set[int] = set()
    n_considered = 0

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(mask_volume.n_sagittal_slices):
        labelled_pixels = int(labelled_per_slice[index])
        labelled_area = float(labelled_area_per_slice[index])

        # --- slice selection ---------------------------------------------
        # Lateral slices sit outside the spine and carry no annotation at all.
        # Keeping them would flood the dataset with empty targets.
        if (
            config.keep_only_annotated_slices
            and labelled_area < config.min_labelled_area_mm2
        ):
            continue
        n_considered += 1

        image_plane = image_volume.sagittal_slice(index)
        mask_plane = mask_volume.sagittal_slice(index)

        processed_image = preprocess_image(
            image_plane, (row_mm, col_mm), config, volume_stats=volume_stats
        )
        semantic, instance = preprocess_mask(mask_plane, (row_mm, col_mm), config)

        # --- quality accounting ------------------------------------------
        area_before += _physical_area_mm2(mask_plane, row_mm, col_mm)
        area_after += _physical_area_mm2(semantic, target_mm, target_mm)
        labels_before.update(int(v) for v in np.unique(to_instance(mask_plane)))
        labels_after.update(int(v) for v in np.unique(instance))

        relative_path = None
        if save:
            out_path = output_dir / f"{image_id}_s{index:03d}.npz"
            np.savez_compressed(
                out_path,
                image=processed_image.astype(np.float16),
                mask=semantic,
                mask_instance=instance,
            )
            relative_path = relative(out_path)

        record = {
            "image_id": image_id,
            "slice_index": index,
            "slice_id": f"{image_id}_s{index:03d}",
            "npz_path": relative_path,
            "labelled_pixels_raw": labelled_pixels,
            "labelled_area_raw_mm2": round(labelled_area, 2),
            "labelled_pixels_processed": int((semantic > 0).sum()),
            "img_mean": float(processed_image.mean()),
            "img_std": float(processed_image.std()),
            "img_min": float(processed_image.min()),
            "img_max": float(processed_image.max()),
        }
        for class_id in SEMANTIC_IDS:
            record[f"px_class_{class_id}"] = int((semantic == class_id).sum())
        slice_records.append(record)

    # Labels vanishing is legitimate (a tiny structure can fall below the
    # resample grid); a label *appearing* would mean interpolation corrupted
    # the mask and is a hard failure.
    invented = sorted(labels_after - labels_before)
    lost = sorted(labels_before - labels_after)

    series_record = {
        "image_id": image_id,
        "n_slices_total": int(mask_volume.n_sagittal_slices),
        "n_slices_kept": len(slice_records),
        "n_slices_dropped": int(mask_volume.n_sagittal_slices) - len(slice_records),
        "rows_before": image_volume.in_plane_shape[0],
        "cols_before": image_volume.in_plane_shape[1],
        "row_spacing_before_mm": round(row_mm, 4),
        "col_spacing_before_mm": round(col_mm, 4),
        "rows_after": config.target_size[0],
        "cols_after": config.target_size[1],
        "spacing_after_mm": target_mm,
        "annotated_area_before_mm2": round(area_before, 1),
        "annotated_area_after_mm2": round(area_after, 1),
        "annotated_area_retained": (area_after / area_before) if area_before else np.nan,
        "labels_invented": invented,
        "labels_lost": lost,
        "mask_labels_ok": not invented,
        "raw_min": volume_stats["raw_min"],
        "raw_max": volume_stats["raw_max"],
        "padding_value": volume_stats["padding_value"],
        "fg_p1": volume_stats.get(f"fg_p{config.clip_percentiles[0]:g}"),
        "fg_p99": volume_stats.get(f"fg_p{config.clip_percentiles[1]:g}"),
        "error": None,
    }
    return slice_records, series_record


def build_processed_dataset(
    pairs: pd.DataFrame,
    config: PreprocessConfig,
    *,
    output_dir: Path = SLICES_DIR,
    limit: int | None = None,
    save: bool = True,
    progress_every: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess every series in ``pairs``.

    A failure on one series is recorded and the run continues, so one bad file
    cannot abort a multi-hundred-volume job.

    Returns
    -------
    (slice_index, series_report)
        ``slice_index`` has one row per written slice; ``series_report`` has
        one row per series with the quality metrics.
    """
    rows = pairs if limit is None else pairs.head(limit)
    total = len(rows)

    all_slices: list[dict] = []
    all_series: list[dict] = []

    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        image_id = row["image_id"]
        try:
            slice_records, series_record = preprocess_series(
                PROJECT_ROOT / row["image_path"],
                PROJECT_ROOT / row["mask_path"],
                config,
                image_id=image_id,
                output_dir=output_dir,
                save=save,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            slice_records = []
            series_record = {
                "image_id": image_id,
                "n_slices_kept": 0,
                "mask_labels_ok": False,
                "error": repr(exc),
            }

        all_slices.extend(slice_records)
        all_series.append(series_record)

        if progress_every and (position % progress_every == 0 or position == total):
            print(
                f"  preprocessed {position}/{total} series "
                f"({len(all_slices)} slices so far)",
                flush=True,
            )

    slice_index = pd.DataFrame.from_records(all_slices)
    series_report = pd.DataFrame.from_records(all_series)

    # Attach series-level metadata so the slice index is self-contained.
    metadata_columns = [
        c
        for c in ["image_id", "patient_id", "modality", "subset", "sex",
                  "num_vertebrae", "num_discs", "Manufacturer", "MagneticFieldStrength"]
        if c in pairs.columns
    ]
    if not slice_index.empty:
        slice_index = slice_index.merge(
            pairs[metadata_columns], on="image_id", how="left", validate="many_to_one"
        )
    if not series_report.empty:
        series_report = series_report.merge(
            pairs[metadata_columns], on="image_id", how="left", validate="one_to_one"
        )

    return slice_index, series_report


def load_processed_slice(npz_path: Path | str) -> dict[str, np.ndarray]:
    """Load one preprocessed slice back into arrays.

    The image is promoted back to ``float32`` because most downstream code
    (and matplotlib) is happier with it than with ``float16``.
    """
    path = Path(npz_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with np.load(path) as bundle:
        return {
            "image": bundle["image"].astype(np.float32),
            "mask": bundle["mask"],
            "mask_instance": bundle["mask_instance"],
        }


def class_pixel_totals(slice_index: pd.DataFrame) -> dict[str, int]:
    """Total pixels per semantic class across the preprocessed dataset."""
    from src.preprocessing.labels import SEMANTIC_CLASSES

    return {
        SEMANTIC_CLASSES[class_id]: int(slice_index[f"px_class_{class_id}"].sum())
        for class_id in SEMANTIC_IDS
        if f"px_class_{class_id}" in slice_index.columns
    }
