"""Image <-> mask pairing based on the dataset's real identifiers.

Observed naming convention (verified against all 447 files, no exceptions):

    <patient_id>_<modality>.mha        e.g. 1_t1.mha, 10_t2.mha, 98_t2_SPACE.mha

* ``patient_id`` is an integer, **not** contiguous (range 1..257 over 218
  distinct patients), and is shared by every series of that patient.
* ``modality`` is one of ``t1``, ``t2``, ``t2_SPACE``.
* ``images/<stem>.mha`` always corresponds to ``masks/<stem>.mha``.
* ``overview.csv`` keys rows on exactly this stem via ``new_file_name``.
* ``radiological_gradings.csv`` keys on ``Patient`` = ``patient_id``.

Pairing is done on the parsed stem, never on directory listing order, so a
missing or extra file surfaces as an explicit mismatch instead of silently
shifting every pair by one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils.paths import (
    EXTRACTED_IMAGES_DIR,
    EXTRACTED_MASKS_DIR,
    GRADINGS_CSV,
    OVERVIEW_CSV,
    relative,
)

# <digits>_<rest>, where <rest> may itself contain underscores (t2_SPACE).
SERIES_STEM_RE = re.compile(r"^(?P<patient_id>\d+)_(?P<modality>.+)$")

#: Metadata columns from overview.csv that are genuinely useful downstream.
#: Restricted on purpose - overview.csv has 39 columns, most of which are
#: DICOM acquisition fields that are not needed for preprocessing.
USEFUL_OVERVIEW_COLUMNS: list[str] = [
    "new_file_name",
    "num_vertebrae",
    "num_discs",
    "sex",
    "subset",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "MRAcquisitionType",
    "ScanningSequence",
    "SeriesDescription",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "EchoTime",
    "RepetitionTime",
]


def parse_series_stem(stem: str) -> tuple[int, str]:
    """Split a file stem into ``(patient_id, modality)``.

    Raises
    ------
    ValueError
        If ``stem`` does not follow the dataset convention. Raising rather
        than guessing keeps unexpected files from being paired incorrectly.
    """
    match = SERIES_STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"Stem does not match '<patient_id>_<modality>': {stem!r}")
    return int(match.group("patient_id")), match.group("modality")


def load_overview(path: Path | str = OVERVIEW_CSV) -> pd.DataFrame:
    """Load ``overview.csv`` and clean the issues found during inspection.

    Cleaning applied (and why):

    * ``sex`` contains trailing-whitespace variants (``'F'`` and ``'F '``
      are stored as different values), so string columns are stripped.
    * ``patient_id`` / ``modality`` are derived from ``new_file_name`` so the
      table can be joined to the file system and to the gradings table.
    """
    overview = pd.read_csv(path)

    # Strip whitespace from every object column; 'F ' vs 'F' would otherwise
    # be treated as two distinct categories.
    for column in overview.select_dtypes(include="object").columns:
        overview[column] = overview[column].str.strip()

    parsed = overview["new_file_name"].astype(str).apply(parse_series_stem)
    overview["patient_id"] = [p for p, _ in parsed]
    overview["modality"] = [m for _, m in parsed]
    return overview


def load_gradings(path: Path | str = GRADINGS_CSV) -> pd.DataFrame:
    """Load ``radiological_gradings.csv`` with tidy column names.

    One row per (patient, intervertebral disc). These are *radiological
    gradings* (Pfirrmann grade, herniation, ...), i.e. classification labels
    for a possible later sprint - they are not segmentation targets, so the
    preprocessing pipeline only summarises them.
    """
    gradings = pd.read_csv(path)
    gradings = gradings.rename(
        columns={c: c.strip().lower().replace(" ", "_") for c in gradings.columns}
    )
    return gradings.rename(columns={"patient": "patient_id"})


def summarise_gradings(gradings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-disc gradings to one row per patient.

    Only a compact summary is joined onto the series table: the number of
    graded discs and the worst (maximum) Pfirrmann grade. Keeping the full
    per-disc table separate avoids exploding the series table.
    """
    grading_columns = [
        c for c in gradings.columns if c not in {"patient_id", "ivd_label"}
    ]
    aggregated = gradings.groupby("patient_id").agg(
        n_graded_discs=("ivd_label", "count"),
        max_pfirrman_grade=(
            "pfirrman_grade" if "pfirrman_grade" in grading_columns else "ivd_label",
            "max",
        ),
    )
    return aggregated.reset_index()


def pair_images_and_masks(
    images_dir: Path | str = EXTRACTED_IMAGES_DIR,
    masks_dir: Path | str = EXTRACTED_MASKS_DIR,
    *,
    overview: pd.DataFrame | None = None,
    gradings: pd.DataFrame | None = None,
    extension: str = ".mha",
) -> tuple[pd.DataFrame, dict]:
    """Build the image/mask pairing table for the whole dataset.

    Pairing key is the file stem, so ``images/42_t2.mha`` can only ever pair
    with ``masks/42_t2.mha``.

    Returns
    -------
    pairs : pandas.DataFrame
        One row per matched series with at least ``image_id``, ``patient_id``,
        ``modality``, ``image_path``, ``mask_path`` plus joined metadata.
    issues : dict
        Pairing problems: images without masks, masks without images, stems
        that could not be parsed, and rows missing from ``overview.csv``.
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)

    image_files = {p.stem: p for p in sorted(images_dir.glob(f"*{extension}"))}
    mask_files = {p.stem: p for p in sorted(masks_dir.glob(f"*{extension}"))}

    image_stems, mask_stems = set(image_files), set(mask_files)
    matched = sorted(image_stems & mask_stems)

    issues: dict = {
        "images_without_mask": sorted(image_stems - mask_stems),
        "masks_without_image": sorted(mask_stems - image_stems),
        "unparsable_stems": [],
        "missing_from_overview": [],
    }

    records = []
    for stem in matched:
        try:
            patient_id, modality = parse_series_stem(stem)
        except ValueError:
            issues["unparsable_stems"].append(stem)
            continue
        records.append(
            {
                "image_id": stem,
                "patient_id": patient_id,
                "modality": modality,
                "image_path": relative(image_files[stem]),
                "mask_path": relative(mask_files[stem]),
            }
        )

    pairs = pd.DataFrame.from_records(records)
    if pairs.empty:
        return pairs, issues

    # --- join series-level metadata from overview.csv ---------------------
    if overview is None:
        overview = load_overview()
    available = [c for c in USEFUL_OVERVIEW_COLUMNS if c in overview.columns]
    meta = overview[available].rename(columns={"new_file_name": "image_id"})
    pairs = pairs.merge(meta, on="image_id", how="left", validate="one_to_one")
    issues["missing_from_overview"] = sorted(
        pairs.loc[pairs["subset"].isna(), "image_id"].tolist()
        if "subset" in pairs.columns
        else []
    )

    # --- join a compact patient-level grading summary ---------------------
    if gradings is None:
        gradings = load_gradings()
    pairs = pairs.merge(
        summarise_gradings(gradings), on="patient_id", how="left", validate="many_to_one"
    )

    return pairs.sort_values(["patient_id", "modality"]).reset_index(drop=True), issues


def describe_pairing(pairs: pd.DataFrame, issues: dict, n_examples: int = 5) -> str:
    """Render a short human-readable pairing summary for reports/notebooks."""
    lines = [
        f"Matched image/mask pairs : {len(pairs)}",
        f"Distinct patients        : {pairs['patient_id'].nunique()}",
        f"Modalities               : {pairs['modality'].value_counts().to_dict()}",
        f"Images without a mask    : {len(issues['images_without_mask'])}",
        f"Masks without an image   : {len(issues['masks_without_image'])}",
        f"Unparsable filenames     : {len(issues['unparsable_stems'])}",
        f"Missing overview.csv row : {len(issues['missing_from_overview'])}",
        "",
        f"Example pairs (first {n_examples}):",
    ]
    for _, row in pairs.head(n_examples).iterrows():
        lines.append(
            f"  [{row['image_id']}] patient={row['patient_id']} "
            f"modality={row['modality']}"
        )
        lines.append(f"      image: {row['image_path']}")
        lines.append(f"      mask : {row['mask_path']}")
    return "\n".join(lines)
