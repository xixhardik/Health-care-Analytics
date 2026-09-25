"""Central definition of project paths.

Every script and notebook imports from here so that there is exactly one
place where the directory layout is declared. All paths are absolute and
derived from the location of this file, which makes the project portable:
moving or renaming the project folder does not break anything.
"""

from __future__ import annotations

from pathlib import Path

# paths.py lives in <project>/src/utils/, so the project root is three
# levels up (utils -> src -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# --- data -----------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
EXTRACTED_DIR: Path = DATA_DIR / "extracted"
PROCESSED_DIR: Path = DATA_DIR / "processed"

IMAGES_ZIP: Path = RAW_DIR / "images.zip"
MASKS_ZIP: Path = RAW_DIR / "masks.zip"
OVERVIEW_CSV: Path = RAW_DIR / "overview.csv"
GRADINGS_CSV: Path = RAW_DIR / "radiological_gradings.csv"

EXTRACTED_IMAGES_DIR: Path = EXTRACTED_DIR / "images"
EXTRACTED_MASKS_DIR: Path = EXTRACTED_DIR / "masks"

# --- outputs --------------------------------------------------------------
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
VISUALIZATIONS_DIR: Path = OUTPUTS_DIR / "visualizations"
REPORTS_DIR: Path = OUTPUTS_DIR / "preprocessing_reports"

# General analysis reports that are not preprocessing-specific (e.g. the
# radiological grading analysis that scopes Sprint 2).
ANALYSIS_REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# Derived artefacts that later sprints will consume.
PAIRS_CSV: Path = REPORTS_DIR / "image_mask_pairs.csv"
VOLUME_INSPECTION_CSV: Path = REPORTS_DIR / "volume_inspection.csv"
SLICE_INDEX_CSV: Path = PROCESSED_DIR / "slice_index.csv"
SPLIT_CSV: Path = PROCESSED_DIR / "splits.csv"
LABEL_MAP_JSON: Path = PROCESSED_DIR / "label_mapping.json"

# Reproducibility: a single seed used by every stochastic step.
RANDOM_SEED: int = 42


def ensure_dirs() -> None:
    """Create every output/data directory the pipeline writes into.

    Safe to call repeatedly; existing directories are left untouched.
    Raw data directories are deliberately NOT created here because the raw
    dataset must already exist and is never written to.
    """
    for directory in (
        EXTRACTED_DIR,
        EXTRACTED_IMAGES_DIR,
        EXTRACTED_MASKS_DIR,
        PROCESSED_DIR,
        OUTPUTS_DIR,
        VISUALIZATIONS_DIR,
        REPORTS_DIR,
        ANALYSIS_REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path | str) -> str:
    """Return ``path`` relative to the project root as a POSIX-style string.

    Used in reports and CSVs so that stored paths stay portable instead of
    baking in an absolute Windows path.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Path lives outside the project; fall back to the absolute form.
        return path.as_posix()
