"""Objective image-quality metrics used to *justify* preprocessing choices.

The point of this module is to keep the pipeline honest: instead of asserting
that denoising or CLAHE helps, each optional step is scored on a sample of
real slices and the decision is recorded with numbers behind it.

Three complementary metrics are used, because optimising any single one alone
is misleading:

``estimate_noise_sigma``
    How much high-frequency noise remains. Denoising should lower this.
``boundary_sharpness``
    Gradient magnitude measured *on the true vertebra/disc boundary* taken
    from the mask. Denoising must not lower this much - a filter that blurs
    the boundary is useless for segmentation even if it removes noise.
``class_contrast``
    Contrast-to-noise ratio between the vertebra and disc classes, i.e. how
    separable the two target structures actually are. Contrast enhancement
    should raise this.

All metrics are implemented with numpy/scipy/OpenCV only, to avoid pulling in
a dependency that would be used for a single function.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.preprocessing.labels import SEM_IVD, SEM_VERTEBRA

# Immerkaer's 2-D noise-estimation kernel. Its response cancels smooth image
# content, so what survives is dominated by noise.
_IMMERKAER_KERNEL = np.array(
    [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64
)


def estimate_noise_sigma(image: np.ndarray, roi: np.ndarray | None = None) -> float:
    """Estimate the additive noise standard deviation of a 2-D image.

    Uses Immerkaer's fast estimator: convolve with a kernel whose response to
    smooth structure is zero, then scale the mean absolute response.

    Parameters
    ----------
    image:
        2-D array.
    roi:
        Optional boolean mask restricting the estimate to a region. Passing
        the tissue region avoids measuring the constant padded background,
        which would bias the estimate towards zero.
    """
    image = np.asarray(image, dtype=np.float64)
    response = ndimage.convolve(image, _IMMERKAER_KERNEL, mode="reflect")

    if roi is not None:
        # Erode so the ROI border (a genuine edge) does not pollute the estimate.
        inner = ndimage.binary_erosion(roi, iterations=2)
        values = np.abs(response[inner]) if inner.any() else np.abs(response[roi])
    else:
        values = np.abs(response[1:-1, 1:-1])

    if values.size == 0:
        return float("nan")
    # sqrt(pi/2) / 6 converts mean|response| into a sigma estimate.
    return float(np.sqrt(np.pi / 2) * values.mean() / 6.0)


def boundary_sharpness(image: np.ndarray, mask: np.ndarray, width: int = 1) -> float:
    """Mean gradient magnitude on the boundary of the labelled structures.

    The boundary is taken from the *ground-truth mask*, not detected from the
    image, so the measurement is not biased by the filtering being evaluated:
    every variant is scored on exactly the same set of pixels.

    A higher value means the intensity transition across the true anatomical
    boundary is steeper, i.e. better preserved.
    """
    image = np.asarray(image, dtype=np.float64)
    foreground = mask > 0
    if not foreground.any():
        return float("nan")

    dilated = ndimage.binary_dilation(foreground, iterations=width)
    eroded = ndimage.binary_erosion(foreground, iterations=width)
    band = dilated & ~eroded
    if not band.any():
        return float("nan")

    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    return float(magnitude[band].mean())


def class_contrast(image: np.ndarray, mask: np.ndarray) -> float:
    """Contrast-to-noise ratio between the vertebra and IVD classes.

    ``CNR = |mean_vertebra - mean_disc| / sqrt((var_vertebra + var_disc) / 2)``

    This is the quantity that matters for the project goal: the model has to
    tell vertebrae and discs apart, so a preprocessing step is only useful for
    contrast if it increases their separability.
    """
    image = np.asarray(image, dtype=np.float64)
    vertebra = image[mask == SEM_VERTEBRA]
    disc = image[mask == SEM_IVD]
    if vertebra.size < 10 or disc.size < 10:
        return float("nan")

    spread = np.sqrt((vertebra.var() + disc.var()) / 2.0)
    if spread == 0:
        return float("nan")
    return float(abs(vertebra.mean() - disc.mean()) / spread)


def dynamic_range_usage(image: np.ndarray, roi: np.ndarray | None = None) -> float:
    """Standard deviation over the ROI - how much of [0, 1] is actually used.

    A very low value means the normalised image is flat and most of the
    available range is wasted, which is what contrast enhancement addresses.
    """
    image = np.asarray(image, dtype=np.float64)
    values = image[roi] if roi is not None else image.ravel()
    return float(values.std()) if values.size else float("nan")


def bias_field_inhomogeneity(image: np.ndarray, mask: np.ndarray, sigma: float = 25.0) -> float:
    """Quantify low-frequency intensity drift across one structure class.

    MRI intensity inhomogeneity shows up as a smooth multiplicative field: the
    *same* tissue appears brighter in one part of the image than another. This
    estimates it by heavily blurring the image, sampling that smooth field
    inside the vertebra mask, and returning its coefficient of variation.

    A value near 0 means vertebral bone has a consistent brightness everywhere
    in the slice; a large value means it drifts, which is what N4 bias field
    correction would address.
    """
    image = np.asarray(image, dtype=np.float64)
    vertebra = mask == SEM_VERTEBRA
    if vertebra.sum() < 50:
        return float("nan")

    smooth = ndimage.gaussian_filter(image, sigma=sigma)
    values = smooth[vertebra]
    mean = values.mean()
    if mean <= 0:
        return float("nan")
    return float(values.std() / mean)


def score_variant(image: np.ndarray, mask: np.ndarray, roi: np.ndarray | None = None) -> dict:
    """Score one preprocessing variant on a single slice."""
    if roi is None:
        roi = image > 0
    return {
        "noise_sigma": estimate_noise_sigma(image, roi=roi),
        "boundary_sharpness": boundary_sharpness(image, mask),
        "class_contrast_cnr": class_contrast(image, mask),
        "dynamic_range_std": dynamic_range_usage(image, roi=roi),
        "bias_inhomogeneity": bias_field_inhomogeneity(image, mask),
    }
