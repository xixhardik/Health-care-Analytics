"""Stage C - deterministic per-disc instance extraction and measurement.

What this does
--------------
Takes an **instance** mask (the label space Sprint 1 saved alongside each
preprocessed slice) and isolates each intervertebral disc as a separate record
with geometric measurements in millimetres.

Identity, not ordering
----------------------
Disc identity comes from the label *value*, never from position or filename
order. The mapping established in the Sprint 2 scoping analysis is:

    radiological_gradings.csv 'IVD label' N   <->   raw mask label 200 + N
                                              <->   Sprint 1 instance label 10 + N

Sprint 1 remapped the sparse raw vocabulary (``0``, ``1-9``, ``100``,
``201-209``) onto a contiguous instance space (``0``, ``1-9`` vertebrae, ``10``
canal, ``11-19`` IVDs) so it fits in ``uint8``. This module converts back to the
grading-file index, so every record carries the ``ivd_label`` that joins
directly to ``radiological_gradings.csv``.

Millimetres, not pixels
-----------------------
Sprint 1 resampled every slice to a known isotropic **1.0 mm/pixel**, which is
precisely what makes physical measurement possible. All distances and areas here
are therefore reported in mm and mm^2 as well as pixels. The scale is passed in
explicitly rather than assumed, so a future run at a different spacing stays
correct.

Scope
-----
Measurements only. No severity score, no grade, no diagnosis - those come from
either the dataset annotation or a model prediction, and are joined on later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.preprocessing.labels import (
    INSTANCE_CANAL,
    INSTANCE_IVD_OFFSET,
    IVD_RAW_RANGE,
)

#: Instance-space label values that correspond to intervertebral discs.
INSTANCE_IVD_VALUES = tuple(
    range(INSTANCE_IVD_OFFSET + 1, INSTANCE_IVD_OFFSET + 10)
)  # 11..19

#: Instance-space label values that correspond to vertebrae.
INSTANCE_VERTEBRA_VALUES = tuple(range(1, 10))  # 1..9

#: Sprint 1's preprocessed pixel size. Passed explicitly everywhere; this is
#: only the default.
DEFAULT_MM_PER_PIXEL = 1.0


def instance_to_ivd_label(instance_value: int) -> int:
    """Convert a Sprint 1 instance label to the grading file's ``IVD label``.

    ``11 -> 1``, ``12 -> 2``, ... Raises for values that are not discs, so a
    vertebra or canal label cannot silently be treated as a disc.
    """
    if instance_value not in INSTANCE_IVD_VALUES:
        raise ValueError(
            f"Instance label {instance_value} is not an IVD "
            f"(discs are {INSTANCE_IVD_VALUES[0]}..{INSTANCE_IVD_VALUES[-1]})"
        )
    return instance_value - INSTANCE_IVD_OFFSET


def ivd_label_to_raw_mask_label(ivd_label: int) -> int:
    """Convert a grading ``IVD label`` to the original ``.mha`` mask value."""
    return IVD_RAW_RANGE[0] - 1 + ivd_label  # 1 -> 201


@dataclass
class DiscMeasurements:
    """Geometric measurements for one disc instance on one slice.

    Every field is either a pixel count, a pixel index, or a millimetre value
    derived from it using the known pixel spacing. Nothing here is inferred or
    modelled.
    """

    # --- identity (Stage G longitudinal keys) ---
    patient_id: int
    series_id: str
    slice_id: str
    slice_index: int
    ivd_label: int
    instance_label: int
    raw_mask_label: int
    mask_source: str  # 'ground_truth' or 'prediction'

    # --- bounding box, pixel index space (row0, col0, row1, col1) inclusive ---
    bbox_row_min: int
    bbox_col_min: int
    bbox_row_max: int
    bbox_col_max: int

    # --- area ---
    area_px: int
    area_mm2: float

    # --- centroid, pixel index space ---
    centroid_row: float
    centroid_col: float

    # --- extents. "height" is superior-inferior (rows), "AP extent" is
    #     anterior-posterior (cols). The Sprint 1 display convention is
    #     superior at the top, anterior on the left. ---
    height_px_bbox: int
    height_mm_bbox: float
    height_mm_mean: float          # over ALL occupied columns, incl. tapered tips
    height_mm_median: float
    height_mm_min: float
    height_mm_max: float
    # Regional heights. 'central' is the one intended for downstream use - it
    # excludes the tapering anterior/posterior tips that drag the all-column
    # mean below a clinically sensible disc height.
    height_mm_anterior: float
    height_mm_central: float
    height_mm_posterior: float
    height_mm_mid_column: float
    ap_extent_px: int
    ap_extent_mm: float

    # --- shape descriptors ---
    aspect_ratio: float          # AP extent / height
    fill_ratio: float            # area / bbox area, 1.0 = perfectly rectangular

    # --- neighbouring structures ---
    vertebra_above_label: int | None
    vertebra_below_label: int | None
    vertebra_above_area_px: int
    vertebra_below_area_px: int
    vertebra_above_height_mm: float
    vertebra_below_height_mm: float
    # Antero-posterior offset between the two bounding vertebral bodies.
    # Relevant to spondylolisthesis, which is a slip between vertebrae.
    vertebral_ap_offset_mm: float
    canal_width_at_disc_mm: float

    # --- provenance ---
    mm_per_pixel: float
    n_components: int  # >1 means the mask for this disc is fragmented


def _largest_component(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep only the largest connected component of a boolean mask.

    A clean disc annotation is one blob. A *predicted* mask can be fragmented,
    and measuring the bounding box across scattered fragments would inflate
    every extent. The component count is retained as a quality flag.
    """
    from scipy import ndimage

    labelled, n = ndimage.label(binary)
    if n <= 1:
        return binary, int(n)
    sizes = ndimage.sum(binary, labelled, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labelled == keep, int(n)


def _row_wise_widths(binary: np.ndarray) -> np.ndarray:
    """Per-row pixel counts, for rows that contain any pixel."""
    counts = binary.sum(axis=1)
    return counts[counts > 0]


def _column_wise_heights(binary: np.ndarray) -> np.ndarray:
    """Per-column pixel counts, for columns that contain any pixel.

    For a disc this is the more meaningful notion of "height": each column is
    an anterior-posterior position, and its pixel count is the disc height
    there. Disc narrowing reduces exactly this.
    """
    counts = binary.sum(axis=0)
    return counts[counts > 0]


def _regional_heights(binary: np.ndarray) -> dict[str, float]:
    """Disc height measured at anterior / central / posterior positions.

    Why this exists
    ---------------
    A lumbar disc is lens-shaped: at its extreme anterior and posterior tips the
    height tapers to one or two pixels. Averaging the height over *every*
    occupied column therefore underestimates the disc height substantially -
    measured on this dataset, the all-column mean lands around 5-6 mm where a
    lumbar disc is nearer 8-12 mm.

    Disc height is conventionally read at defined anterior-posterior positions
    rather than averaged over the whole profile, so three regional heights are
    reported plus a robust central height. Sprint 1 stores slices with
    **anterior on the left**, so low column indices are anterior.

    ``central`` (the middle 50% of the AP width) is the value intended for
    downstream use: it excludes both tapering tips and is the most stable of
    the four.
    """
    heights = binary.sum(axis=0).astype(float)
    occupied = np.flatnonzero(heights > 0)
    if occupied.size == 0:
        return {k: float("nan") for k in ("anterior", "central", "posterior", "mid_column")}

    first, last = int(occupied[0]), int(occupied[-1])
    width = last - first + 1
    profile = heights[first : last + 1]

    third = max(1, width // 3)
    quarter = max(1, width // 4)

    return {
        "anterior": float(profile[:third].mean()),
        "posterior": float(profile[-third:].mean()),
        # Middle 50%: drop a quarter from each end.
        "central": float(profile[quarter : width - quarter].mean())
        if width > 2 * quarter
        else float(profile.mean()),
        "mid_column": float(profile[width // 2]),
    }


def measure_disc(
    instance_mask: np.ndarray,
    instance_label: int,
    *,
    patient_id: int,
    series_id: str,
    slice_id: str,
    slice_index: int,
    mask_source: str,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
) -> DiscMeasurements | None:
    """Measure one disc instance on one slice.

    Returns ``None`` when the label is absent from this slice, so callers can
    simply skip it.
    """
    disc = instance_mask == instance_label
    if not disc.any():
        return None

    disc, n_components = _largest_component(disc)
    rows, cols = np.where(disc)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())

    area_px = int(disc.sum())
    column_heights = _column_wise_heights(disc)
    regional = _regional_heights(disc)

    height_bbox = row_max - row_min + 1
    ap_extent = col_max - col_min + 1
    bbox_area = height_bbox * ap_extent

    ivd_label = instance_to_ivd_label(instance_label)
    above, below = _neighbouring_vertebrae(instance_mask, ivd_label)

    return DiscMeasurements(
        patient_id=patient_id,
        series_id=series_id,
        slice_id=slice_id,
        slice_index=slice_index,
        ivd_label=ivd_label,
        instance_label=int(instance_label),
        raw_mask_label=ivd_label_to_raw_mask_label(ivd_label),
        mask_source=mask_source,
        bbox_row_min=row_min,
        bbox_col_min=col_min,
        bbox_row_max=row_max,
        bbox_col_max=col_max,
        area_px=area_px,
        area_mm2=round(area_px * mm_per_pixel**2, 3),
        centroid_row=round(float(rows.mean()), 3),
        centroid_col=round(float(cols.mean()), 3),
        height_px_bbox=height_bbox,
        height_mm_bbox=round(height_bbox * mm_per_pixel, 3),
        height_mm_mean=round(float(column_heights.mean()) * mm_per_pixel, 3),
        height_mm_median=round(float(np.median(column_heights)) * mm_per_pixel, 3),
        height_mm_min=round(float(column_heights.min()) * mm_per_pixel, 3),
        height_mm_max=round(float(column_heights.max()) * mm_per_pixel, 3),
        height_mm_anterior=round(regional["anterior"] * mm_per_pixel, 3),
        height_mm_central=round(regional["central"] * mm_per_pixel, 3),
        height_mm_posterior=round(regional["posterior"] * mm_per_pixel, 3),
        height_mm_mid_column=round(regional["mid_column"] * mm_per_pixel, 3),
        ap_extent_px=ap_extent,
        ap_extent_mm=round(ap_extent * mm_per_pixel, 3),
        aspect_ratio=round(ap_extent / height_bbox, 4) if height_bbox else float("nan"),
        fill_ratio=round(area_px / bbox_area, 4) if bbox_area else float("nan"),
        vertebra_above_label=above["label"],
        vertebra_below_label=below["label"],
        vertebra_above_area_px=above["area_px"],
        vertebra_below_area_px=below["area_px"],
        vertebra_above_height_mm=round(above["height_px"] * mm_per_pixel, 3),
        vertebra_below_height_mm=round(below["height_px"] * mm_per_pixel, 3),
        vertebral_ap_offset_mm=_vertebral_offset_mm(above, below, mm_per_pixel),
        canal_width_at_disc_mm=_canal_width_mm(
            instance_mask, row_min, row_max, mm_per_pixel
        ),
        mm_per_pixel=mm_per_pixel,
        n_components=n_components,
    )


def _neighbouring_vertebrae(instance_mask: np.ndarray, ivd_label: int) -> tuple[dict, dict]:
    """Describe the vertebrae bounding a disc.

    The dataset's numbering makes this deterministic rather than geometric:
    IVD N is named after the vertebra immediately above it, so IVD ``N`` sits
    between vertebra ``N`` (above) and vertebra ``N - 1`` (below). For the
    lowest disc there is no annotated vertebra below it (the sacrum is excluded
    from this dataset), which is reported as ``None`` rather than guessed.
    """
    def describe(label: int | None) -> dict:
        if label is None or label not in INSTANCE_VERTEBRA_VALUES:
            return {"label": None, "area_px": 0, "height_px": 0.0,
                    "centroid_col": float("nan")}
        selector = instance_mask == label
        if not selector.any():
            return {"label": None, "area_px": 0, "height_px": 0.0,
                    "centroid_col": float("nan")}
        rows, cols = np.where(selector)
        return {
            "label": int(label),
            "area_px": int(selector.sum()),
            "height_px": float(rows.max() - rows.min() + 1),
            "centroid_col": float(cols.mean()),
        }

    above_label = ivd_label if ivd_label in INSTANCE_VERTEBRA_VALUES else None
    below_label = ivd_label - 1 if (ivd_label - 1) in INSTANCE_VERTEBRA_VALUES else None
    return describe(above_label), describe(below_label)


def _vertebral_offset_mm(above: dict, below: dict, mm_per_pixel: float) -> float:
    """Antero-posterior offset between the bounding vertebral centroids, in mm.

    A proxy measurement for vertebral slip. Reported as a raw signed distance,
    **not** interpreted as a spondylolisthesis grade - that judgement needs the
    posterior vertebral body margins and a clinical grading rule, neither of
    which this dataset supplies.
    """
    if np.isnan(above["centroid_col"]) or np.isnan(below["centroid_col"]):
        return float("nan")
    return round((above["centroid_col"] - below["centroid_col"]) * mm_per_pixel, 3)


def _canal_width_mm(
    instance_mask: np.ndarray, row_min: int, row_max: int, mm_per_pixel: float
) -> float:
    """Mean spinal canal width over the disc's row span, in mm.

    Measured only across the rows the disc occupies, so it describes the canal
    at that level rather than over the whole slice.
    """
    band = instance_mask[row_min : row_max + 1] == INSTANCE_CANAL
    if not band.any():
        return float("nan")
    widths = band.sum(axis=1)
    widths = widths[widths > 0]
    return round(float(widths.mean()) * mm_per_pixel, 3)


def extract_discs_from_slice(
    instance_mask: np.ndarray,
    *,
    patient_id: int,
    series_id: str,
    slice_id: str,
    slice_index: int,
    mask_source: str,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
    min_area_px: int = 20,
) -> list[DiscMeasurements]:
    """Measure every disc present on one slice.

    ``min_area_px`` drops slivers where a slice only clips the very edge of a
    disc; such fragments give meaningless height and aspect measurements.
    """
    records: list[DiscMeasurements] = []
    present = set(int(v) for v in np.unique(instance_mask))

    for instance_label in INSTANCE_IVD_VALUES:
        if instance_label not in present:
            continue
        if int((instance_mask == instance_label).sum()) < min_area_px:
            continue
        measurement = measure_disc(
            instance_mask,
            instance_label,
            patient_id=patient_id,
            series_id=series_id,
            slice_id=slice_id,
            slice_index=slice_index,
            mask_source=mask_source,
            mm_per_pixel=mm_per_pixel,
        )
        if measurement is not None:
            records.append(measurement)
    return records


def discs_to_frame(records: list[DiscMeasurements]) -> pd.DataFrame:
    """Convert measurement records to a DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([asdict(r) for r in records])


# ---------------------------------------------------------------------------
# Aggregation: slice level -> disc level
# ---------------------------------------------------------------------------

#: Measurements aggregated by taking the value from the slice where the disc is
#: largest. A disc is a 3-D structure seen on several sagittal slices; its
#: mid-sagittal appearance is the clinically read one, and the slice where the
#: disc has the greatest cross-sectional area is the closest deterministic
#: proxy for that. Averaging across slices instead would mix mid-sagittal
#: anatomy with lateral edge slices.
REPRESENTATIVE_COLUMNS = [
    "slice_id", "slice_index", "area_px", "area_mm2",
    "height_mm_bbox", "height_mm_mean", "height_mm_median",
    "height_mm_min", "height_mm_max",
    "height_mm_anterior", "height_mm_central", "height_mm_posterior",
    "height_mm_mid_column",
    "ap_extent_px", "ap_extent_mm", "aspect_ratio", "fill_ratio",
    "centroid_row", "centroid_col",
    "bbox_row_min", "bbox_col_min", "bbox_row_max", "bbox_col_max",
    "vertebra_above_label", "vertebra_below_label",
    "vertebra_above_area_px", "vertebra_below_area_px",
    "n_components",
]

#: Measurements describing structures *around* the disc rather than the disc
#: itself. These are aggregated as the median over the slices where they are
#: defined (see :func:`aggregate_disc_records`), because the canal and the
#: bounding vertebrae are not visible on every slice a disc appears on.
CONTEXT_COLUMNS = [
    "vertebra_above_height_mm",
    "vertebra_below_height_mm",
    "vertebral_ap_offset_mm",
    "canal_width_at_disc_mm",
]


def aggregate_disc_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-slice disc measurements to one row per (series, disc).

    Two kinds of column are produced:

    * ``*`` - the value on the **representative slice**, the one where the disc
      has the largest area (see :data:`REPRESENTATIVE_COLUMNS`).
    * ``*_across_slices`` - summary statistics over all slices the disc appears
      on, which capture how consistent the structure is.

    Both are kept because they answer different questions, and mixing them
    silently would be misleading.
    """
    if frame.empty:
        return pd.DataFrame()

    aggregated: list[dict] = []
    group_keys = ["patient_id", "series_id", "ivd_label", "mask_source"]

    for keys, group in frame.groupby(group_keys, sort=True):
        best = group.loc[group["area_mm2"].idxmax()]
        record = dict(zip(group_keys, keys))

        record["n_slices_present"] = int(len(group))
        record["representative_slice_id"] = best["slice_id"]
        record["mm_per_pixel"] = float(best["mm_per_pixel"])
        record["instance_label"] = int(best["instance_label"])
        record["raw_mask_label"] = int(best["raw_mask_label"])

        for column in REPRESENTATIVE_COLUMNS:
            if column in group.columns:
                record[column] = best[column]

        # Cross-slice stability of the key measurements.
        for column in ["area_mm2", "height_mm_central", "ap_extent_mm"]:
            record[f"{column}_across_slices_mean"] = round(float(group[column].mean()), 3)
            record[f"{column}_across_slices_std"] = round(float(group[column].std(ddof=0)), 3)
            record[f"{column}_across_slices_max"] = round(float(group[column].max()), 3)

        # Context measurements are taken as the median over the slices where
        # they are actually defined, not from the representative slice alone.
        # The spinal canal is a midline structure: it is absent from ~25% of
        # slices (measured) because discs extend further laterally than the
        # canal does, so reading it only off the largest-disc slice would
        # leave it missing for about a quarter of discs for no good reason.
        # Likewise a bounding vertebra may not appear on every slice.
        for column in CONTEXT_COLUMNS:
            if column not in group.columns:
                continue
            defined = group[column].replace([np.inf, -np.inf], np.nan).dropna()
            record[column] = round(float(defined.median()), 3) if len(defined) else np.nan
            record[f"{column}_n_slices_defined"] = int(len(defined))

        # Intensity features get the same treatment, and for the same reason:
        # the vertebral and canal references are not visible on every slice.
        # The median over defined slices is also more robust than the single
        # representative slice for a signal measurement.
        for column in INTENSITY_COLUMNS:
            if column not in group.columns:
                continue
            defined = group[column].replace([np.inf, -np.inf], np.nan).dropna()
            record[column] = round(float(defined.median()), 5) if len(defined) else np.nan

        record["any_fragmented_slice"] = bool((group["n_components"] > 1).any())
        aggregated.append(record)

    return pd.DataFrame(aggregated)


def add_relative_height_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add each disc's height relative to the other discs of the same series.

    Disc narrowing is judged comparatively - a disc is narrowed *relative to*
    its neighbours and to the patient's own spine - so an absolute millimetre
    height alone is weak evidence. These ratios normalise away patient size and
    are the single most informative geometric feature for the narrowing target.
    """
    if frame.empty:
        return frame

    frame = frame.sort_values(["patient_id", "series_id", "ivd_label"]).copy()
    group = frame.groupby(["patient_id", "series_id", "mask_source"])

    # Central height is the reference measurement (see _regional_heights).
    frame["series_median_height_mm"] = group["height_mm_central"].transform("median")
    frame["height_ratio_to_series_median"] = (
        frame["height_mm_central"] / frame["series_median_height_mm"]
    ).round(4)

    frame["series_median_area_mm2"] = group["area_mm2"].transform("median")
    frame["area_ratio_to_series_median"] = (
        frame["area_mm2"] / frame["series_median_area_mm2"]
    ).round(4)

    # Neighbour comparison: ivd_label is anatomically ordered, so shifting
    # within a series compares adjacent levels.
    frame["height_mm_disc_below"] = group["height_mm_central"].shift(1)
    frame["height_mm_disc_above"] = group["height_mm_central"].shift(-1)
    neighbour_mean = frame[["height_mm_disc_below", "height_mm_disc_above"]].mean(axis=1)
    frame["height_ratio_to_neighbours"] = (
        frame["height_mm_central"] / neighbour_mean
    ).round(4)

    # Disc height normalised by the adjacent vertebral body height - removes
    # patient scale without relying on other discs.
    vertebra_height = frame[
        ["vertebra_above_height_mm", "vertebra_below_height_mm"]
    ].replace(0, np.nan).mean(axis=1)
    frame["disc_to_vertebra_height_ratio"] = (
        frame["height_mm_central"] / vertebra_height
    ).round(4)

    # Anterior/posterior height asymmetry - a wedged disc has a different
    # profile from a uniformly narrowed one.
    frame["height_ap_asymmetry"] = (
        (frame["height_mm_anterior"] - frame["height_mm_posterior"])
        / frame["height_mm_central"]
    ).round(4)

    return frame


# ---------------------------------------------------------------------------
# Deriving instances from a semantic prediction
# ---------------------------------------------------------------------------
#
# The U-Net predicts four SEMANTIC classes (background / vertebra / IVD /
# canal). It does not predict which disc is which. Ground-truth instance masks
# carry that identity directly, but a prediction has to be separated into
# instances afterwards.
#
# This is done deterministically by connected components ordered along the
# superior-inferior axis, matching the dataset's own convention (the most
# inferior disc is number 1, counting upward). It is NOT a reliable substitute
# for ground-truth identity: if the model merges two adjacent discs or misses
# one, every index above the error shifts. That failure mode is exactly why
# `evaluate_disc_identification()` below measures how often the derived
# identities agree with the ground truth, instead of assuming they do.


def instances_from_semantic(
    semantic_mask: np.ndarray,
    *,
    class_id: int,
    first_instance_label: int,
    min_area_px: int = 20,
    max_instances: int = 9,
) -> tuple[np.ndarray, int]:
    """Split one semantic class into ordered instances.

    Components are ordered by centroid row **descending**. Sprint 1 stores
    slices with superior anatomy at the top, so a larger row index is more
    inferior; ordering descending therefore assigns index 1 to the most
    inferior structure, matching the dataset's numbering.

    Returns
    -------
    (instance_mask, n_found)
        ``instance_mask`` holds ``first_instance_label + k`` for the k-th
        structure counting upward from the bottom, and 0 elsewhere.
    """
    from scipy import ndimage

    binary = semantic_mask == class_id
    out = np.zeros(semantic_mask.shape, dtype=np.uint8)
    if not binary.any():
        return out, 0

    labelled, n = ndimage.label(binary)
    components: list[tuple[float, int, int]] = []
    for component in range(1, n + 1):
        selector = labelled == component
        area = int(selector.sum())
        if area < min_area_px:
            continue
        rows, _ = np.where(selector)
        components.append((float(rows.mean()), area, component))

    # Most inferior (largest mean row) first.
    components.sort(key=lambda item: -item[0])
    components = components[:max_instances]

    for position, (_, _, component) in enumerate(components):
        out[labelled == component] = first_instance_label + position

    return out, len(components)


def instance_mask_from_semantic(
    semantic_mask: np.ndarray, *, min_area_px: int = 20
) -> dict:
    """Build a full instance mask from a 4-class semantic prediction.

    Reconstructs the Sprint 1 instance label space so the same measurement code
    runs unchanged on predictions: vertebrae ``1..9``, canal ``10``, discs
    ``11..19``.
    """
    from src.preprocessing.labels import SEM_CANAL, SEM_IVD, SEM_VERTEBRA

    discs, n_discs = instances_from_semantic(
        semantic_mask,
        class_id=SEM_IVD,
        first_instance_label=INSTANCE_IVD_OFFSET + 1,
        min_area_px=min_area_px,
    )
    vertebrae, n_vertebrae = instances_from_semantic(
        semantic_mask,
        class_id=SEM_VERTEBRA,
        first_instance_label=1,
        min_area_px=min_area_px,
    )

    combined = np.zeros(semantic_mask.shape, dtype=np.uint8)
    combined[vertebrae > 0] = vertebrae[vertebrae > 0]
    combined[semantic_mask == SEM_CANAL] = INSTANCE_CANAL
    combined[discs > 0] = discs[discs > 0]

    return {
        "instance_mask": combined,
        "n_discs_found": n_discs,
        "n_vertebrae_found": n_vertebrae,
    }


def evaluate_disc_identification(
    truth_instance: np.ndarray,
    predicted_instance: np.ndarray,
    *,
    min_area_px: int = 20,
) -> dict:
    """Compare derived disc identities against ground truth on one slice.

    For each ground-truth disc, finds the predicted disc with the largest
    overlap and records whether the assigned index matches. This separates two
    different failures that a Dice score alone conflates:

    * the disc region was found but given the **wrong index** (numbering shift)
    * the disc was **not found** at all

    Returns counts plus the per-disc detail.
    """
    truth_labels = [
        v for v in INSTANCE_IVD_VALUES
        if int((truth_instance == v).sum()) >= min_area_px
    ]
    predicted_labels = [
        v for v in INSTANCE_IVD_VALUES
        if int((predicted_instance == v).sum()) >= min_area_px
    ]

    matches: list[dict] = []
    for truth_label in truth_labels:
        truth_region = truth_instance == truth_label
        best_label, best_overlap = None, 0
        for predicted_label in predicted_labels:
            overlap = int((truth_region & (predicted_instance == predicted_label)).sum())
            if overlap > best_overlap:
                best_label, best_overlap = predicted_label, overlap

        matches.append(
            {
                "truth_ivd_label": instance_to_ivd_label(truth_label),
                "matched_pred_ivd_label": (
                    instance_to_ivd_label(best_label) if best_label else None
                ),
                "overlap_px": best_overlap,
                "index_correct": bool(best_label == truth_label),
                "region_found": bool(best_label is not None),
            }
        )

    return {
        "n_truth_discs": len(truth_labels),
        "n_predicted_discs": len(predicted_labels),
        "count_matches": len(truth_labels) == len(predicted_labels),
        "n_region_found": sum(1 for m in matches if m["region_found"]),
        "n_index_correct": sum(1 for m in matches if m["index_correct"]),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Intensity features
# ---------------------------------------------------------------------------
#
# Geometry alone cannot express the Pfirrmann grade. The grade is read from the
# T2 signal of the nucleus pulposus - a bright, homogeneous nucleus is grade 1,
# a dark collapsed one is grade 5 - together with disc height. So an intensity
# description of the disc interior is the single most relevant image feature for
# that target, and the geometric features cover the height component.
#
# Two normalisations are provided because absolute MRI intensity is not
# comparable between scans even after Sprint 1's per-volume normalisation:
#   * disc intensity relative to the adjacent VERTEBRAL BODY (a within-slice
#     reference that is always present)
#   * disc intensity relative to the SPINAL CANAL (CSF is the conventional
#     bright reference on T2, though it is only visible on midline slices)


def measure_disc_intensity(
    image: np.ndarray,
    instance_mask: np.ndarray,
    instance_label: int,
    *,
    erode_iterations: int = 1,
) -> dict:
    """Intensity description of one disc and its local references.

    Parameters
    ----------
    image:
        The preprocessed slice, already normalised to ``[0, 1]`` by Sprint 1.
    instance_mask:
        Sprint 1 instance labels for the same slice.
    instance_label:
        The disc to describe (11..19).
    erode_iterations:
        Erosion applied before sampling the "nucleus" region, to pull the
        sample away from the disc boundary where partial-volume effects mix in
        bone and CSF signal.

    Returns
    -------
    dict
        Intensity statistics, plus the two normalised ratios. Missing
        references give NaN rather than a substituted value.
    """
    from scipy import ndimage

    disc = instance_mask == instance_label
    if not disc.any():
        return {}

    values = image[disc].astype(np.float64)
    ivd_label = instance_to_ivd_label(instance_label)

    out: dict = {
        "intensity_mean": round(float(values.mean()), 5),
        "intensity_std": round(float(values.std()), 5),
        "intensity_p10": round(float(np.percentile(values, 10)), 5),
        "intensity_median": round(float(np.median(values)), 5),
        "intensity_p90": round(float(np.percentile(values, 90)), 5),
        # Coefficient of variation: a degenerate disc is not just darker, it is
        # also less homogeneous, which this captures independently of brightness.
        "intensity_cv": round(
            float(values.std() / values.mean()) if values.mean() > 0 else np.nan, 5
        ),
    }

    # --- nucleus vs annulus ---------------------------------------------
    # The eroded core approximates the nucleus; the remaining rim approximates
    # the annulus. Crude, but it is the distinction the Pfirrmann scale rests
    # on and it needs no extra annotation.
    core = ndimage.binary_erosion(disc, iterations=erode_iterations)
    rim = disc & ~core
    out["intensity_nucleus_mean"] = (
        round(float(image[core].mean()), 5) if core.any() else np.nan
    )
    out["intensity_annulus_mean"] = (
        round(float(image[rim].mean()), 5) if rim.any() else np.nan
    )
    if core.any() and rim.any() and image[rim].mean() > 0:
        out["intensity_nucleus_annulus_ratio"] = round(
            float(image[core].mean() / image[rim].mean()), 5
        )
    else:
        out["intensity_nucleus_annulus_ratio"] = np.nan

    # --- reference: adjacent vertebral bodies ----------------------------
    vertebra_labels = [
        label
        for label in (ivd_label, ivd_label - 1)
        if label in INSTANCE_VERTEBRA_VALUES
    ]
    vertebra_selector = np.zeros_like(disc)
    for label in vertebra_labels:
        vertebra_selector |= instance_mask == label

    if vertebra_selector.any():
        vertebra_mean = float(image[vertebra_selector].mean())
        out["intensity_vertebra_mean"] = round(vertebra_mean, 5)
        out["intensity_disc_vertebra_ratio"] = (
            round(float(values.mean() / vertebra_mean), 5)
            if vertebra_mean > 0
            else np.nan
        )
    else:
        out["intensity_vertebra_mean"] = np.nan
        out["intensity_disc_vertebra_ratio"] = np.nan

    # --- reference: spinal canal (CSF) -----------------------------------
    canal = instance_mask == INSTANCE_CANAL
    if canal.any():
        canal_mean = float(image[canal].mean())
        out["intensity_canal_mean"] = round(canal_mean, 5)
        out["intensity_disc_canal_ratio"] = (
            round(float(values.mean() / canal_mean), 5) if canal_mean > 0 else np.nan
        )
    else:
        out["intensity_canal_mean"] = np.nan
        out["intensity_disc_canal_ratio"] = np.nan

    return out


#: Intensity columns aggregated as the median over slices where defined.
INTENSITY_COLUMNS = [
    "intensity_mean", "intensity_std", "intensity_p10", "intensity_median",
    "intensity_p90", "intensity_cv", "intensity_nucleus_mean",
    "intensity_annulus_mean", "intensity_nucleus_annulus_ratio",
    "intensity_vertebra_mean", "intensity_disc_vertebra_ratio",
    "intensity_canal_mean", "intensity_disc_canal_ratio",
]


def extract_disc_features_from_slice(
    image: np.ndarray,
    instance_mask: np.ndarray,
    *,
    patient_id: int,
    series_id: str,
    slice_id: str,
    slice_index: int,
    mask_source: str,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
    min_area_px: int = 20,
) -> list[dict]:
    """Geometry **and** intensity features for every disc on one slice.

    Convenience wrapper combining :func:`extract_discs_from_slice` with
    :func:`measure_disc_intensity`, returning plain dicts ready for a DataFrame.
    """
    geometric = extract_discs_from_slice(
        instance_mask,
        patient_id=patient_id,
        series_id=series_id,
        slice_id=slice_id,
        slice_index=slice_index,
        mask_source=mask_source,
        mm_per_pixel=mm_per_pixel,
        min_area_px=min_area_px,
    )

    records: list[dict] = []
    for measurement in geometric:
        record = asdict(measurement)
        record.update(
            measure_disc_intensity(image, instance_mask, measurement.instance_label)
        )
        records.append(record)
    return records
