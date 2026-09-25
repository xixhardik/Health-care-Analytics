"""Reproducible extraction of the raw dataset archives.

The raw ZIP files in ``data/raw/`` are treated as read-only. This module
only ever reads from them and writes into ``data/extracted/``, so the
extraction can be re-run at any time to rebuild the working copy of the
dataset from scratch.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.utils.paths import (
    EXTRACTED_DIR,
    EXTRACTED_IMAGES_DIR,
    EXTRACTED_MASKS_DIR,
    IMAGES_ZIP,
    MASKS_ZIP,
    ensure_dirs,
)


def list_archive(zip_path: Path) -> list[str]:
    """Return the member names stored inside ``zip_path`` without extracting."""
    with zipfile.ZipFile(zip_path) as archive:
        return archive.namelist()


def extract_archive(
    zip_path: Path,
    destination: Path,
    *,
    strip_top_level: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Extract every file from ``zip_path`` into ``destination``.

    Parameters
    ----------
    zip_path:
        Archive to read. Never modified.
    destination:
        Directory that receives the extracted files. Created if missing.
    strip_top_level:
        The dataset archives wrap everything in a single top-level folder
        (``images/`` and ``masks/``). When True that wrapper is removed so
        the files land directly in ``destination`` instead of
        ``destination/images/``.
    overwrite:
        When False (default) a member whose target file already exists with
        the expected size is skipped. This makes re-runs cheap and makes the
        step safe to interrupt and resume.
    verbose:
        Print progress every 50 files.

    Returns
    -------
    list[Path]
        Absolute paths of all files that now exist in ``destination``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        total = len(members)

        for index, member in enumerate(members, start=1):
            # Normalise the member name and optionally drop the wrapper dir.
            parts = Path(member.filename).parts
            if strip_top_level and len(parts) > 1:
                parts = parts[1:]
            target = destination.joinpath(*parts)

            # Guard against path traversal ("zip slip") in untrusted archives.
            if not str(target.resolve()).startswith(str(destination.resolve())):
                raise ValueError(f"Unsafe path in archive: {member.filename!r}")

            target.parent.mkdir(parents=True, exist_ok=True)

            already_ok = (
                target.exists()
                and target.stat().st_size == member.file_size
                and not overwrite
            )
            if not already_ok:
                with archive.open(member) as src, open(target, "wb") as dst:
                    # Stream in chunks: some volumes are >100 MB.
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)

            written.append(target)
            if verbose and (index % 50 == 0 or index == total):
                print(f"  {zip_path.name}: {index}/{total} files", flush=True)

    return written


def extract_dataset(*, overwrite: bool = False, verbose: bool = True) -> dict[str, int]:
    """Extract both dataset archives into ``data/extracted/``.

    Returns a mapping of dataset part -> number of extracted files.
    """
    ensure_dirs()
    counts: dict[str, int] = {}

    for name, zip_path, destination in (
        ("images", IMAGES_ZIP, EXTRACTED_IMAGES_DIR),
        ("masks", MASKS_ZIP, EXTRACTED_MASKS_DIR),
    ):
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing raw archive: {zip_path}")
        if verbose:
            size_mb = zip_path.stat().st_size / 1e6
            print(f"Extracting {zip_path.name} ({size_mb:,.0f} MB) -> {destination}")
        files = extract_archive(
            zip_path, destination, overwrite=overwrite, verbose=verbose
        )
        counts[name] = len(files)

    if verbose:
        print(f"Extraction complete: {counts} (root: {EXTRACTED_DIR})")
    return counts


if __name__ == "__main__":  # pragma: no cover - manual entry point
    extract_dataset()
