"""Upload inspection and validation for a single MRI volume.

Validation runs *before* the pipeline so a bad upload produces a useful message
instead of a stack trace half way through inference. Everything here is derived
from the file itself - no dataset-level lookup is involved, so an unseen study is
handled the same way as a dataset volume.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Formats the validated pipeline can actually read. SimpleITK reads more than
#: this, but these are the ones Sprint 1 was built and verified against, so the
#: list is deliberately narrow rather than optimistic.
SUPPORTED_SUFFIXES = (".mha", ".mhd", ".nii", ".nii.gz")

#: Guard rails chosen from the dataset's own range (0.04 MB - 42 MB observed).
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MIN_UPLOAD_BYTES = 2 * 1024

#: A lumbar sagittal study outside these bounds is almost certainly not what the
#: model was validated on. Dataset range: 15-120 sagittal slices, in-plane
#: 216-3682 px.
MIN_SLICES = 3
MAX_SLICES = 512
MIN_IN_PLANE = 64
MAX_IN_PLANE = 4096


class ValidationError(Exception):
    """Raised when an upload cannot be accepted.

    Carries a machine-readable ``code`` so the API can map it to a stable error
    contract, and a message written for a human rather than for a log.
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class VolumeInfo:
    """What can be said about an uploaded volume without running inference."""

    filename: str
    size_bytes: int
    suffix: str
    slice_count: int
    in_plane_shape: tuple[int, int]
    in_plane_spacing_mm: tuple[float, float]
    slice_spacing_mm: float
    native_orientation: str
    dtype: str
    intensity_min: float
    intensity_max: float
    modality: str | None
    modality_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalised_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return Path(lowered).suffix


def sanitise_filename(name: str) -> str:
    """Strip directory components and anything not safe for a filesystem.

    Upload filenames are attacker-controlled. Only the basename is kept, and it
    is reduced to a conservative character set so it can never traverse
    directories or collide with shell metacharacters.
    """
    base = Path(name).name
    cleaned = "".join(c if (c.isalnum() or c in "._- ") else "_" for c in base)
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned[:128] or "upload"


def infer_modality(filename: str) -> tuple[str | None, str]:
    """Guess the modality from the filename, honestly reporting the source.

    The SPIDER convention is ``<patient>_<modality>.mha`` - for example
    ``33_t1.mha`` or ``45_t2_SPACE.mha``. A DICOM series description is not
    available in a converted ``.mha``, so this is a filename heuristic and is
    labelled as one rather than presented as metadata.
    """
    stem = Path(filename).name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "t2_space" in stem or "t2space" in stem:
        return "t2_SPACE", "filename_heuristic"
    if "t2" in stem:
        return "t2", "filename_heuristic"
    if "t1" in stem:
        return "t1", "filename_heuristic"
    return None, "not_determinable"


def validate_upload(path: Path, original_filename: str) -> VolumeInfo:
    """Validate an uploaded file and describe it. Raises :class:`ValidationError`.

    Checks, in the order a user would hit them: extension, size, readability,
    then geometry. Each failure names the actual problem.
    """
    suffix = _normalised_suffix(original_filename)
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValidationError(
            "UNSUPPORTED_FORMAT",
            f"Unsupported MRI format '{suffix or 'unknown'}'. "
            f"Supported formats: {', '.join(SUPPORTED_SUFFIXES)}.",
            {"supported_formats": list(SUPPORTED_SUFFIXES)},
        )

    if not path.exists():
        raise ValidationError("UPLOAD_MISSING", "The uploaded file could not be found.")

    size = path.stat().st_size
    if size < MIN_UPLOAD_BYTES:
        raise ValidationError(
            "FILE_TOO_SMALL",
            f"The file is {size:,} bytes, which is too small to be an MRI volume.",
            {"size_bytes": size, "minimum_bytes": MIN_UPLOAD_BYTES},
        )
    if size > MAX_UPLOAD_BYTES:
        raise ValidationError(
            "FILE_TOO_LARGE",
            f"The file is {size / 1e6:.1f} MB, above the "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit.",
            {"size_bytes": size, "maximum_bytes": MAX_UPLOAD_BYTES},
        )

    return inspect_volume(path, original_filename, size, suffix)


def inspect_volume(
    path: Path, original_filename: str, size: int, suffix: str
) -> VolumeInfo:
    """Read the volume header/array and check the geometry is plausible."""
    from src.preprocessing.volume_io import load_image

    try:
        volume = load_image(path)
    except Exception as exc:  # noqa: BLE001 - any reader failure is a bad upload
        raise ValidationError(
            "UNREADABLE_VOLUME",
            "Unable to read the MRI volume. The file may be corrupt, truncated, "
            "or not a supported image volume.",
            {"reader_error": type(exc).__name__},
        ) from exc

    array = volume.array
    if array.ndim != 3:
        raise ValidationError(
            "UNEXPECTED_DIMENSIONS",
            f"Expected a 3-D volume, found {array.ndim} dimensions.",
            {"shape": list(array.shape)},
        )

    slices = volume.n_sagittal_slices
    rows, cols = volume.in_plane_shape
    if not (MIN_SLICES <= slices <= MAX_SLICES):
        raise ValidationError(
            "UNEXPECTED_SLICE_COUNT",
            f"The volume has {slices} sagittal slices, outside the supported "
            f"range {MIN_SLICES}-{MAX_SLICES}.",
            {"slice_count": slices},
        )
    if not (MIN_IN_PLANE <= rows <= MAX_IN_PLANE
            and MIN_IN_PLANE <= cols <= MAX_IN_PLANE):
        raise ValidationError(
            "UNEXPECTED_DIMENSIONS",
            f"In-plane size {rows}x{cols} is outside the supported range "
            f"{MIN_IN_PLANE}-{MAX_IN_PLANE} pixels.",
            {"in_plane_shape": [rows, cols]},
        )

    row_mm, col_mm = volume.in_plane_spacing
    if not (0.0 < row_mm < 20.0 and 0.0 < col_mm < 20.0):
        raise ValidationError(
            "IMPLAUSIBLE_SPACING",
            f"In-plane pixel spacing {row_mm:.3f} x {col_mm:.3f} mm is not "
            f"plausible for a lumbar MRI.",
            {"in_plane_spacing_mm": [row_mm, col_mm]},
        )

    import numpy as np

    if not np.isfinite(array).any():
        raise ValidationError(
            "EMPTY_VOLUME", "The volume contains no finite voxel values."
        )

    modality, modality_source = infer_modality(original_filename)
    return VolumeInfo(
        filename=original_filename,
        size_bytes=size,
        suffix=suffix,
        slice_count=int(slices),
        in_plane_shape=(int(rows), int(cols)),
        in_plane_spacing_mm=(round(float(row_mm), 4), round(float(col_mm), 4)),
        slice_spacing_mm=round(float(volume.slice_spacing), 4),
        native_orientation=volume.native_orientation,
        dtype=str(array.dtype),
        intensity_min=round(float(np.nanmin(array)), 2),
        intensity_max=round(float(np.nanmax(array)), 2),
        modality=modality,
        modality_source=modality_source,
    )
