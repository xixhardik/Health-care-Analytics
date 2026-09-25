"""Image and mask preprocessing operations.

Design decisions, and the evidence behind them
----------------------------------------------
Every choice below follows from the dataset inspection rather than from a
generic recipe. The measurements that justify them are produced by
``scripts/02_inspect.py`` and ``scripts/03_preprocess.py`` and written into
``outputs/preprocessing_reports/``.

**Geometry - resample to a common physical scale, then crop/pad.**
In-plane pixel spacing ranges from 0.077 mm to 1.233 mm across the dataset,
and the matrix size from 216 to 3682 rows.
Resizing purely by pixel count would make the same vertebra appear at
different physical sizes in different patients. Each slice is therefore
resampled to a fixed millimetres-per-pixel scale, which preserves the
anatomical aspect ratio by construction, and then centre-cropped or
zero-padded to a fixed matrix size. No anatomy is stretched.

**Masks - nearest neighbour only.**
Labels are categorical (vertebra #3 is not "between" #2 and #4), so any
averaging interpolation would invent labels that do not exist. Every mask
operation in this module uses ``cv2.INTER_NEAREST``, and
:func:`validate_mask_labels` re-checks afterwards that no new label appeared.

**Intensity - robust per-volume normalisation.**
Two incompatible intensity conventions coexist in this dataset: roughly 80%
of series are linearly rescaled into ``[-1000, 3096]`` with a large spike of
background voxels sitting exactly at -1000, and the rest are plain
``[0, ~700]`` ranges. MRI intensity has no absolute physical meaning, so
normalisation is mandatory. Plain min-max or mean/std would be dominated by
the background spike (up to 70% of voxels), so statistics are computed over
*foreground* voxels only and the result is clipped at robust percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from src.preprocessing.labels import to_instance, to_semantic, unknown_labels

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """All preprocessing parameters in one auditable place.

    Defaults were chosen from the measured dataset statistics:

    target_spacing_mm
        1.0 mm/pixel. The in-plane field of view in this dataset spans
        240-343 mm, and vertebral bodies are ~30-40 mm tall, so 1 mm keeps
        every structure comfortably resolved while making the matrix size
        manageable.
    target_size
        352 rows x 256 cols, i.e. 352 mm (superior-inferior) x 256 mm
        (anterior-posterior). **This is measured, not guessed.** For every
        volume the smallest centred crop that still contains the whole
        annotation was computed from the inspection results; the worst case
        needs 341 mm vertically but only 244 mm horizontally, because the
        lumbar spine is tall and narrow. A square 288 x 288 crop would have
        clipped annotated anatomy in 157 of 447 volumes. At 352 x 256 no
        volume loses any annotation, and both sides are multiples of 32,
        which suits the encoder/decoder depth of a U-Net in a later sprint.
    clip_percentiles
        (1, 99) over foreground voxels. Discards the saturated ceiling
        (some series have ~5% of voxels pinned at the maximum value) without
        touching real tissue contrast.
    min_labelled_area_mm2
        Slice selection threshold expressed as a physical area rather than a
        pixel count, because pixel spacing varies by a factor of ~16 across
        the dataset and a fixed pixel count would mean very different
        amounts of anatomy in different series.
    """

    # --- geometry ---
    target_spacing_mm: float = 1.0
    target_size: tuple[int, int] = (352, 256)  # (rows=sup-inf, cols=ant-post)

    # --- intensity ---
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    normalize: bool = True

    # --- optional enhancement (enabled only where inspection justified it) ---
    denoise: bool = True
    denoise_method: str = "median"  # 'median' | 'gaussian' | 'bilateral' | 'none'
    denoise_kernel: int = 3

    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    bias_field_correction: bool = False  # see notes in scripts/02_inspect.py

    # --- slice selection ---
    keep_only_annotated_slices: bool = True
    min_labelled_area_mm2: float = 10.0

    def to_dict(self) -> dict:
        """Serialisable form, embedded in the preprocessing report."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Intensity analysis
# ---------------------------------------------------------------------------


def detect_padding_value(array: np.ndarray, min_fraction: float = 0.01) -> float | None:
    """Return the constant background/padding value, if the volume has one.

    Many series in this dataset were resampled onto a larger grid and the
    empty region filled with a single extreme value (commonly -1000), which
    can account for up to ~70% of all voxels. That spike must be excluded
    before computing intensity statistics, otherwise it dominates them.

    The minimum is reported as padding only when it occupies at least
    ``min_fraction`` of the volume, so genuinely dark tissue is not mistaken
    for padding.
    """
    minimum = float(array.min())
    fraction = float((array == minimum).mean())
    return minimum if fraction >= min_fraction else None


def foreground_mask(array: np.ndarray) -> np.ndarray:
    """Boolean mask of voxels that are not constant background padding."""
    padding = detect_padding_value(array)
    if padding is None:
        return np.ones(array.shape, dtype=bool)
    return array > padding


def intensity_statistics(
    array: np.ndarray, percentiles: tuple[float, ...] = (1, 25, 50, 75, 99)
) -> dict:
    """Intensity summary of a volume, computed over foreground voxels.

    Both the raw full-volume range and the foreground-only percentiles are
    reported so the report can show *why* foreground-restricted statistics
    were necessary.
    """
    array = np.asarray(array, dtype=np.float32)
    fg = foreground_mask(array)
    fg_values = array[fg]
    if fg_values.size == 0:  # pathological volume; fall back to everything
        fg_values = array.ravel()

    stats = {
        "raw_min": float(array.min()),
        "raw_max": float(array.max()),
        "padding_value": detect_padding_value(array),
        "foreground_fraction": float(fg.mean()),
        "fg_mean": float(fg_values.mean()),
        "fg_std": float(fg_values.std()),
    }
    for p in percentiles:
        stats[f"fg_p{p:g}"] = float(np.percentile(fg_values, p))
    # Fraction of voxels pinned at the maximum -> saturation/clipping check.
    stats["saturated_fraction"] = float((array == array.max()).mean())
    return stats


def normalize_image(
    array: np.ndarray,
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
    stats: dict | None = None,
) -> np.ndarray:
    """Scale intensities to ``[0, 1]`` using robust foreground percentiles.

    Parameters
    ----------
    array:
        Image data (a single slice or a whole volume).
    percentiles:
        Lower/upper percentile, computed over foreground voxels, that map to
        0 and 1. Values outside are clipped.
    stats:
        Pre-computed :func:`intensity_statistics` for the *parent volume*.
        Passing this in is the intended usage when normalising individual
        slices: it keeps every slice of a volume on one common scale, so
        relative brightness between slices is preserved. Computing
        percentiles per slice would make near-empty lateral slices explode
        in contrast.

    Returns
    -------
    numpy.ndarray
        ``float32`` array in ``[0, 1]``.
    """
    array = np.asarray(array, dtype=np.float32)
    low_p, high_p = percentiles

    if stats is not None:
        low = stats.get(f"fg_p{low_p:g}")
        high = stats.get(f"fg_p{high_p:g}")
        padding = stats.get("padding_value")
    else:
        local = intensity_statistics(array, percentiles=(low_p, high_p))
        low, high = local[f"fg_p{low_p:g}"], local[f"fg_p{high_p:g}"]
        padding = local["padding_value"]

    if high is None or low is None or high <= low:
        # Degenerate (constant) slice - return all zeros rather than divide by 0.
        return np.zeros(array.shape, dtype=np.float32)

    out = (np.clip(array, low, high) - low) / (high - low)

    # Padding voxels carry no signal; pin them to 0 so the background is a
    # single consistent value across the whole dataset.
    if padding is not None:
        out[array <= padding] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------


def denoise_image(
    image: np.ndarray, method: str = "median", kernel: int = 3
) -> np.ndarray:
    """Apply mild noise reduction to a normalised 2D slice.

    A 3x3 median filter is the default: MRI magnitude images carry
    Rician noise with occasional speckle, and a median kernel suppresses it
    while preserving the vertebra/disc boundaries that the segmentation
    depends on. Gaussian smoothing blurs exactly those boundaries, so it is
    available but not the default.

    ``kernel`` must be odd. Larger kernels start erasing the thin cortical
    bone rim, so 3 is used.
    """
    if method == "none":
        return image
    if kernel % 2 == 0:
        raise ValueError(f"denoise kernel must be odd, got {kernel}")

    image = np.asarray(image, dtype=np.float32)
    if method == "median":
        # OpenCV's medianBlur only supports float32 with kernel size 3.
        return cv2.medianBlur(image, kernel)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if method == "bilateral":
        # Edge-preserving but ~10x slower; sigmas tuned for [0, 1] data.
        return cv2.bilateralFilter(image, d=kernel, sigmaColor=0.1, sigmaSpace=kernel)
    raise ValueError(f"Unknown denoise method: {method!r}")


def enhance_contrast(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a normalised ``[0, 1]`` slice, returning ``[0, 1]``.

    CLAHE equalises contrast in local tiles, which is the useful property
    here: lumbar MRI slices contain a bright subcutaneous-fat band and a
    much lower-contrast vertebra/disc region, and a single global window
    leaves the spine itself flat.

    The image is promoted to 16-bit for the OpenCV call rather than 8-bit so
    the ~900 distinct intensity levels present in the raw data are not
    quantised away.
    """
    image = np.asarray(image, dtype=np.float32)
    as_uint16 = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    equalised = clahe.apply(as_uint16)
    return (equalised.astype(np.float32) / 65535.0).astype(np.float32)


def correct_bias_field(volume_array: np.ndarray, *, shrink_factor: int = 4) -> np.ndarray:
    """N4 bias field correction for MRI intensity inhomogeneity.

    Implemented and available, but **off by default** (see
    :class:`PreprocessConfig`). The reason is measured, not assumed: with
    ``shrink_factor=4`` N4 costs only ~0.2-1 s per volume here, so cost is not
    the issue. What the measurement showed is that its effect on this dataset
    is negligible and inconsistent - the low-frequency brightness drift across
    vertebral bone changed by well under 1%, and on one test volume it got
    slightly worse. CLAHE, which is applied, reduces the same inhomogeneity
    metric by roughly half as a side effect of being a *local* equalisation.
    See ``outputs/preprocessing_reports/preprocessing_choices.md``.

    ``shrink_factor`` estimates the field on a downsampled grid, then applies
    it at full resolution.
    """
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(volume_array, dtype=np.float32))
    shrunk = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    mask = sitk.OtsuThreshold(shrunk, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30] * 3)
    corrector.Execute(shrunk, mask)

    log_field = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_field)
    return sitk.GetArrayFromImage(corrected)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def resample_to_spacing(
    plane: np.ndarray,
    src_spacing: tuple[float, float],
    target_spacing: float,
    *,
    is_mask: bool,
) -> np.ndarray:
    """Resample a 2D slice to isotropic ``target_spacing`` mm/pixel.

    The physical size of the slice is preserved, so the anatomical aspect
    ratio is preserved exactly; only the sampling grid changes.

    ``is_mask`` selects the interpolation and is the single switch that keeps
    label data safe: masks always use nearest neighbour, images use area
    averaging when shrinking (which avoids aliasing) and bilinear when
    enlarging.
    """
    row_mm, col_mm = src_spacing
    rows, cols = plane.shape[:2]

    new_rows = max(1, int(round(rows * row_mm / target_spacing)))
    new_cols = max(1, int(round(cols * col_mm / target_spacing)))
    if (new_rows, new_cols) == (rows, cols):
        return plane

    if is_mask:
        interpolation = cv2.INTER_NEAREST
    else:
        shrinking = new_rows * new_cols < rows * cols
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR

    # cv2 takes (width, height) = (cols, rows).
    return cv2.resize(plane, (new_cols, new_rows), interpolation=interpolation)


def center_crop_or_pad(
    plane: np.ndarray, target_shape: tuple[int, int], pad_value: float = 0.0
) -> np.ndarray:
    """Centre a slice inside ``target_shape``, cropping and/or padding.

    Applied identically to images and masks (it moves data, never blends it),
    which guarantees the two stay pixel-aligned. Masks are padded with 0,
    i.e. background, which is semantically correct.
    """
    target_rows, target_cols = target_shape
    rows, cols = plane.shape[:2]

    # --- crop the axes that are too large -------------------------------
    row_start = max(0, (rows - target_rows) // 2)
    col_start = max(0, (cols - target_cols) // 2)
    cropped = plane[
        row_start : row_start + min(rows, target_rows),
        col_start : col_start + min(cols, target_cols),
    ]

    # --- pad the axes that are too small --------------------------------
    out = np.full(target_shape, pad_value, dtype=plane.dtype)
    rows, cols = cropped.shape[:2]
    row_offset = (target_rows - rows) // 2
    col_offset = (target_cols - cols) // 2
    out[row_offset : row_offset + rows, col_offset : col_offset + cols] = cropped
    return out


def apply_geometry(
    plane: np.ndarray,
    src_spacing: tuple[float, float],
    config: PreprocessConfig,
    *,
    is_mask: bool,
) -> np.ndarray:
    """Run the full geometric pipeline (resample then crop/pad) on one slice."""
    resampled = resample_to_spacing(
        plane, src_spacing, config.target_spacing_mm, is_mask=is_mask
    )
    return center_crop_or_pad(resampled, config.target_size, pad_value=0)


# ---------------------------------------------------------------------------
# Slice-level entry points
# ---------------------------------------------------------------------------


def preprocess_image(
    plane: np.ndarray,
    src_spacing: tuple[float, float],
    config: PreprocessConfig,
    *,
    volume_stats: dict | None = None,
) -> np.ndarray:
    """Preprocess one MRI slice.

    Order of operations, and why:

    1. **Geometry** (resample + crop/pad) first, so all later filters act on
       a consistent physical scale - a 3x3 kernel then means the same
       ~3 mm everywhere in the dataset.
    2. **Normalisation** to ``[0, 1]`` using the parent volume's foreground
       statistics, putting both intensity conventions on one scale.
    3. **Denoising** before contrast enhancement, because CLAHE amplifies
       whatever noise is present.
    4. **CLAHE** last, operating on clean, normalised data.

    Returns a ``float32`` array of shape ``config.target_size`` in ``[0, 1]``.
    """
    plane = np.asarray(plane, dtype=np.float32)
    out = apply_geometry(plane, src_spacing, config, is_mask=False)

    if config.normalize:
        out = normalize_image(
            out, percentiles=config.clip_percentiles, stats=volume_stats
        )

    if config.denoise:
        out = denoise_image(out, method=config.denoise_method, kernel=config.denoise_kernel)

    if config.enhance_contrast:
        out = enhance_contrast(
            out, clip_limit=config.clahe_clip_limit, tile_grid=config.clahe_tile_grid
        )

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preprocess_mask(
    plane: np.ndarray, src_spacing: tuple[float, float], config: PreprocessConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess one mask slice into both label spaces.

    Geometry is applied to the **raw** label values with nearest-neighbour
    interpolation, and the semantic/instance remaps happen afterwards. Doing
    it in this order means resizing never sees a collapsed label space and
    the identity of each vertebra/disc is preserved through the resize.

    Returns
    -------
    (semantic, instance)
        ``uint8`` arrays of shape ``config.target_size``. See
        :mod:`src.preprocessing.labels` for the class definitions.
    """
    plane = np.asarray(plane)
    resized_raw = apply_geometry(plane, src_spacing, config, is_mask=True)
    return to_semantic(resized_raw), to_instance(resized_raw)


def validate_mask_labels(
    original: np.ndarray, processed: np.ndarray, *, mapping
) -> dict:
    """Check that resizing did not invent or silently drop labels.

    Compares the label set of the mask before and after the geometric
    pipeline. Labels *disappearing* is legitimate (a structure only a few
    pixels wide can fall outside the crop or below the resample grid), but a
    **new** label appearing would mean the interpolation corrupted the data
    and is treated as a hard failure by the caller.
    """
    expected = {int(v) for v in np.unique(mapping(original))}
    actual = {int(v) for v in np.unique(processed)}
    return {
        "labels_before": sorted(expected),
        "labels_after": sorted(actual),
        "labels_lost": sorted(expected - actual),
        "labels_invented": sorted(actual - expected),
        "ok": not (actual - expected),
    }
