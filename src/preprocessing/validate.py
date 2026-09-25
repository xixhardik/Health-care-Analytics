"""Dataset-level inspection and integrity checking.

This module answers the Sprint 1 questions with measurements instead of
assumptions: real formats, real dimensions, real label vocabulary, real
intensity ranges, and whether image and mask geometry actually agree.

It also runs two *anatomical* sanity checks. These are cheap but powerful:
they confirm that the reorientation in :mod:`src.preprocessing.volume_io` put
the axes where the code thinks they are. If the axis handling were wrong, the
checks would fail on most volumes.

  * the spinal canal must lie **posterior** to the vertebral bodies
  * vertebra label 1 (most inferior by convention) must lie **inferior** to
    the highest vertebra label present
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.labels import (
    CANAL_RAW,
    IVD_RAW_RANGE,
    VERTEBRA_RAW_RANGE,
    unknown_labels,
)
from src.preprocessing.transforms import intensity_statistics
from src.preprocessing.volume_io import (
    SAGITTAL_AXIS,
    Volume,
    load_image,
    load_mask,
    volume_geometry,
)
from src.utils.paths import PROJECT_ROOT


def _centroid(mask: np.ndarray, selector: np.ndarray) -> tuple[float, float, float] | None:
    """Centre of mass (in voxel index space, ``[z, y, x]``) of a boolean selector."""
    if not selector.any():
        return None
    coords = np.argwhere(selector)
    return tuple(float(v) for v in coords.mean(axis=0))


def anatomical_checks(mask_volume: Volume) -> dict:
    """Verify the reoriented axes against known lumbar anatomy.

    Runs on the RAS-reoriented mask array, where ``z`` increases superiorly
    and ``y`` increases anteriorly.
    """
    mask = mask_volume.array
    vertebrae = (mask >= VERTEBRA_RAW_RANGE[0]) & (mask <= VERTEBRA_RAW_RANGE[1])
    canal = mask == CANAL_RAW

    result: dict = {
        "canal_posterior_to_vertebrae": None,
        "label1_inferior_to_top_vertebra": None,
    }

    # --- check 1: canal sits behind (posterior to) the vertebral bodies ---
    v_centroid = _centroid(mask, vertebrae)
    c_centroid = _centroid(mask, canal)
    if v_centroid is not None and c_centroid is not None:
        # y is the anterior axis, so "posterior" means a smaller y.
        result["canal_posterior_to_vertebrae"] = bool(c_centroid[1] < v_centroid[1])

    # --- check 2: label 1 is the most inferior vertebra -------------------
    present = [
        int(v)
        for v in np.unique(mask)
        if VERTEBRA_RAW_RANGE[0] <= int(v) <= VERTEBRA_RAW_RANGE[1]
    ]
    if len(present) >= 2:
        low = _centroid(mask, mask == min(present))
        high = _centroid(mask, mask == max(present))
        if low is not None and high is not None:
            # z is the superior axis, so "inferior" means a smaller z.
            result["label1_inferior_to_top_vertebra"] = bool(low[0] < high[0])

    return result


def annotation_extent_mm(mask_volume: Volume) -> dict:
    """Physical bounding box of all annotated voxels, in millimetres.

    Drives the choice of crop size in
    :class:`~src.preprocessing.transforms.PreprocessConfig`: the target field
    of view has to be at least as large as this box for every volume,
    otherwise centre-cropping would cut off annotated anatomy.
    """
    labelled = mask_volume.array > 0
    if not labelled.any():
        return {
            "ann_rows_mm": 0.0,
            "ann_cols_mm": 0.0,
            "ann_slices": 0,
            "ann_row_center_frac": np.nan,
            "ann_col_center_frac": np.nan,
        }

    coords = np.argwhere(labelled)
    dz, dy, dx = mask_volume.spacing_zyx
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)

    rows, cols = mask_volume.in_plane_shape
    return {
        # +1 because the bounding box is inclusive of both end voxels.
        "ann_rows_mm": float((z1 - z0 + 1) * dz),
        "ann_cols_mm": float((y1 - y0 + 1) * dy),
        "ann_slices": int(x1 - x0 + 1),
        # Where the annotation sits relative to the image centre (0.5 = centred).
        # Tells us whether a centre crop is a safe assumption.
        "ann_row_center_frac": float(((z0 + z1) / 2) / max(rows - 1, 1)),
        "ann_col_center_frac": float(((y0 + y1) / 2) / max(cols - 1, 1)),
    }


def label_inventory(mask_volume: Volume) -> dict:
    """Per-volume label vocabulary and structure counts."""
    mask = mask_volume.array
    values, counts = np.unique(mask, return_counts=True)
    present = {int(v): int(c) for v, c in zip(values, counts)}

    vertebra_labels = [
        v for v in present if VERTEBRA_RAW_RANGE[0] <= v <= VERTEBRA_RAW_RANGE[1]
    ]
    ivd_labels = [v for v in present if IVD_RAW_RANGE[0] <= v <= IVD_RAW_RANGE[1]]

    total = int(mask.size)
    labelled = total - present.get(0, 0)
    return {
        "labels_present": sorted(present),
        "n_vertebra_labels": len(vertebra_labels),
        "n_ivd_labels": len(ivd_labels),
        "has_canal": CANAL_RAW in present,
        "unknown_labels": unknown_labels(mask),
        "labelled_voxel_fraction": labelled / total if total else 0.0,
        "voxels_vertebrae": sum(present[v] for v in vertebra_labels),
        "voxels_ivd": sum(present[v] for v in ivd_labels),
        "voxels_canal": present.get(CANAL_RAW, 0),
    }


def annotated_slice_count(mask_volume: Volume, min_area_mm2: float = 10.0) -> dict:
    """How many sagittal slices carry a usable amount of annotation.

    Lateral sagittal slices sit outside the spine and contain no labels at
    all. ``min_area_mm2`` filters out slices that only clip the very edge of
    a structure, which would otherwise contribute near-empty training targets.

    The threshold is a **physical area**, not a pixel count: in-plane pixel
    spacing varies by roughly a factor of 16 across this dataset, so a fixed
    pixel count would mean wildly different amounts of anatomy depending on
    which scanner produced the series.
    """
    per_slice = (mask_volume.array > 0).sum(axis=(0, 1))  # reduce over z, y
    row_mm, col_mm = mask_volume.in_plane_spacing
    area_per_slice = per_slice * row_mm * col_mm
    return {
        "n_slices_total": int(mask_volume.n_sagittal_slices),
        "n_slices_any_label": int((per_slice > 0).sum()),
        "n_slices_usable": int((area_per_slice >= min_area_mm2).sum()),
        "max_labelled_area_mm2_per_slice": float(area_per_slice.max()),
    }


def inspect_volume_pair(
    image_path: Path | str,
    mask_path: Path | str,
    *,
    min_labelled_area_mm2: float = 10.0,
) -> dict:
    """Inspect a single image/mask pair and return one flat record.

    Everything a Sprint 1 report needs about one series: format, geometry,
    intensity, labels, anatomy checks and geometry agreement between the
    image and its mask.
    """
    image_path, mask_path = Path(image_path), Path(mask_path)
    image = load_image(image_path)
    mask = load_mask(mask_path)

    record: dict = {
        "image_id": image_path.stem,
        "image_format": image_path.suffix.lstrip("."),
        "mask_format": mask_path.suffix.lstrip("."),
        "image_file_mb": round(image_path.stat().st_size / 1e6, 2),
        "mask_file_mb": round(mask_path.stat().st_size / 1e6, 2),
    }

    # --- geometry --------------------------------------------------------
    img_geom = volume_geometry(image)
    record.update({f"img_{k}": v for k, v in img_geom.items()})
    record["mask_dtype"] = str(mask.array.dtype)

    # Image and mask must describe the same physical grid, otherwise they
    # cannot be overlaid and the pair is unusable.
    record["geometry_match_shape"] = bool(image.array.shape == mask.array.shape)
    record["geometry_match_spacing"] = bool(
        np.allclose(image.spacing_zyx, mask.spacing_zyx, atol=1e-4)
    )
    record["geometry_match_orientation"] = bool(
        image.native_orientation == mask.native_orientation
    )

    # --- intensity -------------------------------------------------------
    record.update(intensity_statistics(image.array))

    # --- labels, slices, anatomy ----------------------------------------
    record.update(label_inventory(mask))
    record.update(annotated_slice_count(mask, min_area_mm2=min_labelled_area_mm2))
    record.update(annotation_extent_mm(mask))
    record.update(anatomical_checks(mask))

    return record


def validate_dataset(
    pairs: pd.DataFrame,
    *,
    min_labelled_area_mm2: float = 10.0,
    progress_every: int = 25,
    limit: int | None = None,
) -> pd.DataFrame:
    """Inspect every pair in ``pairs`` and return one row per series.

    Parameters
    ----------
    pairs:
        Output of :func:`src.preprocessing.pairing.pair_images_and_masks`.
    limit:
        Inspect only the first ``limit`` rows. Intended for quick smoke tests;
        the reported numbers are only dataset-wide when it is ``None``.
    """
    rows = pairs if limit is None else pairs.head(limit)
    total = len(rows)
    records = []

    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]
        try:
            record = inspect_volume_pair(
                image_path, mask_path, min_labelled_area_mm2=min_labelled_area_mm2
            )
            record["read_error"] = None
        except Exception as exc:  # keep scanning; record the failure
            record = {"image_id": Path(image_path).stem, "read_error": repr(exc)}
        record["patient_id"] = row["patient_id"]
        record["modality"] = row["modality"]
        records.append(record)

        if progress_every and (position % progress_every == 0 or position == total):
            print(f"  inspected {position}/{total} series", flush=True)

    return pd.DataFrame.from_records(records)


def summarise_inspection(inspection: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Condense the per-series inspection table into report-level findings."""
    ok = inspection[inspection["read_error"].isna()]

    def counter(column: str) -> dict:
        return ok[column].value_counts().sort_index().to_dict()

    # Union of every label value seen anywhere in the dataset.
    all_labels: set[int] = set()
    for labels in ok["labels_present"].dropna():
        all_labels.update(int(v) for v in labels)

    unknown: set[int] = set()
    for labels in ok["unknown_labels"].dropna():
        unknown.update(int(v) for v in labels)

    return {
        "n_series": int(len(inspection)),
        "n_series_readable": int(len(ok)),
        "n_read_errors": int(inspection["read_error"].notna().sum()),
        "n_patients": int(pairs["patient_id"].nunique()),
        "modality_counts": pairs["modality"].value_counts().to_dict(),
        "image_formats": counter("image_format"),
        "mask_formats": counter("mask_format"),
        "image_dtypes": counter("img_dtype"),
        "mask_dtypes": counter("mask_dtype"),
        "native_orientations": counter("img_native_orientation"),
        "label_vocabulary": sorted(all_labels),
        "unknown_labels": sorted(unknown),
        "n_classes_semantic": 4,
        "slices": {
            "total": int(ok["n_slices_total"].sum()),
            "with_any_label": int(ok["n_slices_any_label"].sum()),
            "usable": int(ok["n_slices_usable"].sum()),
            "per_volume_min": int(ok["n_slices_total"].min()),
            "per_volume_max": int(ok["n_slices_total"].max()),
            "per_volume_median": float(ok["n_slices_total"].median()),
        },
        "in_plane_rows": {
            "min": int(ok["img_rows"].min()),
            "max": int(ok["img_rows"].max()),
            "unique_count": int(ok["img_rows"].nunique()),
        },
        "in_plane_cols": {
            "min": int(ok["img_cols"].min()),
            "max": int(ok["img_cols"].max()),
            "unique_count": int(ok["img_cols"].nunique()),
        },
        "spacing_mm": {
            "row_min": float(ok["img_row_spacing_mm"].min()),
            "row_max": float(ok["img_row_spacing_mm"].max()),
            "col_min": float(ok["img_col_spacing_mm"].min()),
            "col_max": float(ok["img_col_spacing_mm"].max()),
            "slice_min": float(ok["img_slice_spacing_mm"].min()),
            "slice_max": float(ok["img_slice_spacing_mm"].max()),
        },
        "annotation_extent_mm": {
            "rows_max": float(ok["ann_rows_mm"].max()),
            "cols_max": float(ok["ann_cols_mm"].max()),
            "rows_p99": float(ok["ann_rows_mm"].quantile(0.99)),
            "cols_p99": float(ok["ann_cols_mm"].quantile(0.99)),
        },
        "geometry_mismatches": {
            "shape": int((~ok["geometry_match_shape"]).sum()),
            "spacing": int((~ok["geometry_match_spacing"]).sum()),
            "orientation": int((~ok["geometry_match_orientation"]).sum()),
        },
        "anatomical_checks": {
            "canal_posterior_pass": int(
                (ok["canal_posterior_to_vertebrae"] == True).sum()  # noqa: E712
            ),
            "canal_posterior_fail": int(
                (ok["canal_posterior_to_vertebrae"] == False).sum()  # noqa: E712
            ),
            "label1_inferior_pass": int(
                (ok["label1_inferior_to_top_vertebra"] == True).sum()  # noqa: E712
            ),
            "label1_inferior_fail": int(
                (ok["label1_inferior_to_top_vertebra"] == False).sum()  # noqa: E712
            ),
        },
        "intensity_conventions": {
            "with_padding_value": int(ok["padding_value"].notna().sum()),
            "without_padding_value": int(ok["padding_value"].isna().sum()),
            "raw_min_values": counter("raw_min"),
            "raw_max_values": counter("raw_max"),
        },
    }


# ---------------------------------------------------------------------------
# File-level integrity: duplicates and missing files
# ---------------------------------------------------------------------------


def file_digest(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so large volumes do not load into memory."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_files(directory: Path | str, pattern: str = "*.mha") -> dict[str, list[str]]:
    """Group byte-identical files in ``directory`` by content hash.

    Only groups with more than one member are returned. Used to answer the
    "are there duplicate samples?" question with evidence rather than a guess.
    """
    from collections import defaultdict

    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(Path(directory).glob(pattern)):
        groups[file_digest(path)].append(path.name)
    return {h: names for h, names in groups.items() if len(names) > 1}


def summarise_duplicates(duplicate_groups: dict[str, list[str]]) -> dict:
    """Interpret duplicate groups, separating expected from unexpected cases.

    In this dataset a patient's T1 and T2 series are acquired on the same
    grid and share a single annotation, so ``<id>_t1.mha`` and
    ``<id>_t2.mha`` masks are legitimately byte-identical. That is a property
    of the annotation process, not a corrupted dataset, so it is reported
    separately from duplicates that span *different* patients - which would
    be a genuine problem.
    """
    same_patient, cross_patient = [], []
    for names in duplicate_groups.values():
        patients = {name.split("_", 1)[0] for name in names}
        (same_patient if len(patients) == 1 else cross_patient).append(sorted(names))
    return {
        "n_duplicate_groups": len(duplicate_groups),
        "n_same_patient_groups": len(same_patient),
        "n_cross_patient_groups": len(cross_patient),
        "same_patient_examples": same_patient[:5],
        "cross_patient_groups": cross_patient,
    }


def check_raw_files_present() -> dict:
    """Confirm the four raw dataset files exist and report their sizes."""
    from src.utils.paths import GRADINGS_CSV, IMAGES_ZIP, MASKS_ZIP, OVERVIEW_CSV

    status = {}
    for path in (IMAGES_ZIP, MASKS_ZIP, OVERVIEW_CSV, GRADINGS_CSV):
        status[path.name] = {
            "present": path.exists(),
            "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else None,
        }
    return status
