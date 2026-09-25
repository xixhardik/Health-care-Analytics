"""Reading the raw ``.mha`` volumes and exposing consistent geometry.

Why this module exists
----------------------
Dataset inspection showed the volumes are **not** stored in a single
consistent way:

* Most 2D TSE series are stored ``LPS``, while the 3D ``t2_SPACE`` series
  are stored ``PIR``. The array axis that steps through sagittal slices is
  therefore *not* the same axis in every file.
* In-plane pixel spacing varies from 0.077 mm to 1.233 mm, and the spacing
  between sagittal slices from 0.86 mm to 9.63 mm (measured across all 447
  series).

Both facts would silently corrupt a pipeline that assumed a fixed axis
order. Every volume is therefore reoriented to a canonical ``RAS`` frame on
load. Reorientation is a pure axis permutation plus flips - no interpolation
- so it is exactly loss-less and safe to apply to label masks.

After reorientation the convention is fixed:

* numpy array is indexed ``[z, y, x]``
* ``x`` steps right-ward  -> the **sagittal / through-plane** axis (axis 2)
* ``y`` steps anterior-ward -> in-plane columns
* ``z`` steps superior-ward -> in-plane rows
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Canonical anatomical frame that every volume is mapped into.
CANONICAL_ORIENTATION = "RAS"

# Index of the through-plane (sagittal) axis in the numpy array once the
# volume has been reoriented to CANONICAL_ORIENTATION.
SAGITTAL_AXIS = 2


@dataclass(frozen=True)
class Volume:
    """A loaded 3D volume plus the geometry needed to preprocess it.

    Attributes
    ----------
    array:
        Voxel data indexed ``[z, y, x]`` (superior, anterior, right).
    spacing_zyx:
        Physical voxel size in millimetres, aligned with ``array`` axes, i.e.
        ``(dz, dy, dx)``.
    native_orientation:
        Orientation code the file was stored in, before reorientation.
    path:
        Source file.
    """

    array: np.ndarray
    spacing_zyx: tuple[float, float, float]
    native_orientation: str
    path: Path

    @property
    def n_sagittal_slices(self) -> int:
        """Number of sagittal slices available in this volume."""
        return self.array.shape[SAGITTAL_AXIS]

    @property
    def in_plane_shape(self) -> tuple[int, int]:
        """(rows, cols) of one sagittal slice = (superior-inferior, anterior-posterior)."""
        return (self.array.shape[0], self.array.shape[1])

    @property
    def in_plane_spacing(self) -> tuple[float, float]:
        """(row_spacing_mm, col_spacing_mm) of one sagittal slice."""
        return (self.spacing_zyx[0], self.spacing_zyx[1])

    @property
    def slice_spacing(self) -> float:
        """Distance between adjacent sagittal slices, in millimetres."""
        return self.spacing_zyx[2]

    def sagittal_slice(self, index: int) -> np.ndarray:
        """Return sagittal slice ``index`` as a 2D array of shape (rows, cols).

        Both in-plane axes are reversed so the result follows the standard
        radiological sagittal convention:

        * rows: *superior* anatomy at the top (RAS ``z`` increases upward, but
          array row 0 is drawn at the top, so the axis is reversed)
        * cols: *anterior* anatomy on the left (RAS ``y`` increases
          anteriorly, so the axis is reversed)

        Both operations are pure index reversals - no interpolation and no
        value changes - so the identical call is safe for label masks and
        keeps image and mask perfectly aligned.
        """
        plane = np.take(self.array, index, axis=SAGITTAL_AXIS)
        return np.flip(plane, axis=(0, 1))


def _read_reoriented(path: Path) -> tuple[sitk.Image, str]:
    """Read ``path`` and reorient it to :data:`CANONICAL_ORIENTATION`.

    Returns the reoriented image together with the *native* orientation code
    so the original storage convention can still be reported.
    """
    image = sitk.ReadImage(str(path))
    native = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )
    return sitk.DICOMOrient(image, CANONICAL_ORIENTATION), native


def _to_volume(path: Path, dtype: np.dtype | None) -> Volume:
    """Shared body of :func:`load_image` and :func:`load_mask`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")

    reoriented, native = _read_reoriented(path)
    array = sitk.GetArrayFromImage(reoriented)  # -> [z, y, x]
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    # SimpleITK reports spacing as (x, y, z); numpy axes are (z, y, x).
    sx, sy, sz = reoriented.GetSpacing()
    return Volume(
        array=array,
        spacing_zyx=(float(sz), float(sy), float(sx)),
        native_orientation=native,
        path=path,
    )


def load_image(path: Path | str) -> Volume:
    """Load an MRI volume, reoriented to RAS, as ``float32``.

    ``float32`` is used because the intensity normalisation that follows is
    a floating-point operation and the raw ``int16`` range differs between
    acquisition conventions in this dataset.
    """
    return _to_volume(Path(path), np.float32)


def load_mask(path: Path | str) -> Volume:
    """Load a segmentation mask volume, reoriented to RAS.

    The dtype is kept **integer** (``int16``, matching the raw files) so that
    label values are never altered by a float round-trip. Raw labels reach
    209, so an 8-bit type would overflow and is deliberately not used here.
    """
    return _to_volume(Path(path), np.int16)


def volume_geometry(volume: Volume) -> dict:
    """Summarise a volume's geometry for the inspection report."""
    rows, cols = volume.in_plane_shape
    row_mm, col_mm = volume.in_plane_spacing
    return {
        "native_orientation": volume.native_orientation,
        "n_slices": volume.n_sagittal_slices,
        "rows": rows,
        "cols": cols,
        "row_spacing_mm": round(row_mm, 4),
        "col_spacing_mm": round(col_mm, 4),
        "slice_spacing_mm": round(volume.slice_spacing, 4),
        "fov_rows_mm": round(rows * row_mm, 1),
        "fov_cols_mm": round(cols * col_mm, 1),
        "dtype": str(volume.array.dtype),
    }
