"""Local filesystem storage for analyses.

Deliberately plain: one directory per analysis holding the upload, a small JSON
record, and one compressed array bundle for the viewer. That is enough for a
locally run research prototype and avoids standing up a database for data that is
already file-shaped.

Privacy notes:
* the analysis id is a random token, never derived from the filename or any
  identifier, so nothing sensitive appears in a URL
* filenames are sanitised before they touch the filesystem
* voxel data is never logged
"""

from __future__ import annotations

import json
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

RECORD_NAME = "record.json"
RESULT_NAME = "result.json"
ARRAYS_NAME = "arrays.npz"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_analysis_id() -> str:
    """An unguessable, non-identifying id."""
    return secrets.token_hex(8)


def _replace(temporary: Path, path: Path, *, attempts: int = 12) -> None:
    """Atomically move ``temporary`` onto ``path``, tolerating a transient lock.

    On Windows ``os.replace`` raises ``PermissionError`` when anything else has
    the target open, even for reading. A status poll arriving while the worker
    mirrors its progress is enough, and the write is perfectly valid - so failing
    the whole analysis over it loses real work for no reason.

    The replace itself stays atomic; only the attempt is repeated, with a short
    backoff, and a genuinely stuck file still raises.
    """
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


def _default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


class AnalysisStore:
    """CRUD over the per-analysis directories."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------

    def directory(self, analysis_id: str) -> Path:
        # Guard against traversal: ids are hex tokens, so anything else is refused
        # before it is used to build a path.
        if not analysis_id or not all(c in "0123456789abcdef" for c in analysis_id):
            raise ValueError("Invalid analysis id")
        return self.root / analysis_id

    def exists(self, analysis_id: str) -> bool:
        try:
            return (self.directory(analysis_id) / RECORD_NAME).exists()
        except ValueError:
            return False

    # -- records ----------------------------------------------------------

    def create(self, analysis_id: str, record: dict) -> Path:
        directory = self.directory(analysis_id)
        directory.mkdir(parents=True, exist_ok=True)
        self.write_record(analysis_id, record)
        return directory

    def write_record(self, analysis_id: str, record: dict) -> None:
        path = self.directory(analysis_id) / RECORD_NAME
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=_default)
        _replace(temporary, path)

    def read_record(self, analysis_id: str) -> dict | None:
        path = self.directory(analysis_id) / RECORD_NAME
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def update_record(self, analysis_id: str, **changes: Any) -> dict:
        record = self.read_record(analysis_id) or {}
        record.update(changes)
        self.write_record(analysis_id, record)
        return record

    # -- results ----------------------------------------------------------

    def write_result(self, analysis_id: str, result: dict) -> None:
        path = self.directory(analysis_id) / RESULT_NAME
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=_default)
        _replace(temporary, path)

    def read_result(self, analysis_id: str) -> dict | None:
        path = self.directory(analysis_id) / RESULT_NAME
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    # -- arrays for the viewer -------------------------------------------

    def write_arrays(
        self,
        analysis_id: str,
        *,
        images: np.ndarray,
        semantic: np.ndarray,
        instance: np.ndarray,
        slice_ids: list[str],
    ) -> None:
        """Persist what the slice endpoint needs, and nothing more.

        The display image is stored as ``uint8`` rather than float32: it exists
        only to be rendered as a PNG, so a third of the size costs nothing in
        visible quality.
        """
        path = self.directory(analysis_id) / ARRAYS_NAME
        np.savez_compressed(
            path,
            images=(np.clip(images, 0.0, 1.0) * 255).astype(np.uint8),
            semantic=semantic.astype(np.uint8),
            instance=instance.astype(np.uint8),
            slice_ids=np.array(slice_ids, dtype=object),
        )

    def read_arrays(self, analysis_id: str) -> dict | None:
        path = self.directory(analysis_id) / ARRAYS_NAME
        if not path.exists():
            return None
        with np.load(path, allow_pickle=True) as bundle:
            return {
                "images": bundle["images"],
                "semantic": bundle["semantic"],
                "instance": bundle["instance"],
                "slice_ids": [str(v) for v in bundle["slice_ids"]],
            }

    # -- listing / deletion ----------------------------------------------

    def list_records(self) -> list[dict]:
        records: list[dict] = []
        for directory in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not directory.is_dir():
                continue
            path = directory / RECORD_NAME
            if not path.exists():
                continue
            try:
                with path.open(encoding="utf-8") as handle:
                    records.append(json.load(handle))
            except (OSError, json.JSONDecodeError):
                continue
        records.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return records

    def delete(self, analysis_id: str) -> bool:
        try:
            directory = self.directory(analysis_id)
        except ValueError:
            return False
        if not directory.exists():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return not directory.exists()
