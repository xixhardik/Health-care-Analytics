"""Sprint 1 / Step 3 - extract the raw archives into data/extracted/.

Usage
-----
    python scripts/01_extract.py            # resume-safe extraction
    python scripts/01_extract.py --overwrite  # force re-extraction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly (python scripts/01_extract.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.extract import extract_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract files even if they already exist with the right size.",
    )
    args = parser.parse_args()

    counts = extract_dataset(overwrite=args.overwrite)
    print(f"images: {counts['images']} files, masks: {counts['masks']} files")


if __name__ == "__main__":
    main()
