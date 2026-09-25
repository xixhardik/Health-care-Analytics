"""Storage writes survive a transient Windows file lock.

Found by the Sprint 7.1 live end-to-end run: an analysis failed mid-flight with
``PermissionError: [WinError 5]`` while the worker mirrored its progress into
``record.json``. Nothing was wrong with the write. On Windows ``os.replace``
raises when anything else merely has the target open, and a status poll arriving
at that instant is enough - so a valid analysis was being thrown away over a
millisecond of contention.

These tests pin the retry so the regression cannot come back silently.
"""

from __future__ import annotations

import json

import pytest

from backend.app.services.storage import AnalysisStore


@pytest.fixture
def store(tmp_path) -> AnalysisStore:
    return AnalysisStore(tmp_path / "app_data")


def test_a_record_round_trips(store):
    store.create("abc123abc123abcd", {"status": "queued", "progress": 0})
    assert store.read_record("abc123abc123abcd")["status"] == "queued"


def test_a_write_retries_through_a_transient_lock(store, monkeypatch):
    """A lock that clears must cost a delay, not the analysis."""
    analysis_id = "abc123abc123abcd"
    store.create(analysis_id, {"status": "queued"})

    from pathlib import Path

    real_replace = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    store.write_record(analysis_id, {"status": "processing", "progress": 40})

    assert calls["n"] == 4, "the write should have been retried, not abandoned"
    assert store.read_record(analysis_id)["progress"] == 40


def test_a_persistent_lock_still_raises(store, monkeypatch):
    """The retry must not turn a genuinely stuck file into silent data loss."""
    analysis_id = "abc123abc123abcd"
    store.create(analysis_id, {"status": "queued"})

    from pathlib import Path

    def always_locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_locked)
    with pytest.raises(PermissionError):
        store.write_record(analysis_id, {"status": "processing"})


def test_a_result_write_retries_too(store, monkeypatch):
    analysis_id = "abc123abc123abcd"
    store.create(analysis_id, {"status": "queued"})

    from pathlib import Path

    real_replace = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    store.write_result(analysis_id, {"analysis_id": analysis_id, "discs": []})

    assert store.read_result(analysis_id)["analysis_id"] == analysis_id


def test_the_written_file_is_never_left_truncated(store):
    """The replace stays atomic: a reader sees either the old or the new file."""
    analysis_id = "abc123abc123abcd"
    store.create(analysis_id, {"status": "queued", "progress": 0})
    for progress in range(0, 101, 10):
        store.write_record(analysis_id, {"status": "processing",
                                         "progress": progress})
        # Parsing proves the file is complete, not half-written.
        path = store.directory(analysis_id) / "record.json"
        assert json.loads(path.read_text(encoding="utf-8"))["progress"] == progress
    # No stray temporary file is left behind.
    assert not list(store.directory(analysis_id).glob("*.tmp"))
