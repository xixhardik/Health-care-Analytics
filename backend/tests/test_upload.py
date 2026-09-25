"""Upload validation: every rejection must be specific and traceback-free."""

from __future__ import annotations

import pytest


def _error(response) -> dict:
    body = response.json()
    assert "error" in body, body
    return body["error"]


def test_rejects_unsupported_extension(app_client):
    response = app_client.post(
        "/api/analysis/upload",
        files={"file": ("clinical-notes.txt", b"x" * 5_000, "text/plain")},
    )
    assert response.status_code == 400
    error = _error(response)
    assert error["code"] == "UNSUPPORTED_FORMAT"
    assert ".mha" in error["message"]


def test_rejects_file_that_is_too_small(app_client):
    response = app_client.post(
        "/api/analysis/upload",
        files={"file": ("truncated.mha", b"x" * 64, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert _error(response)["code"] == "FILE_TOO_SMALL"


def test_rejects_unreadable_volume(app_client):
    """A correct extension with junk content must fail at the reader, cleanly."""
    response = app_client.post(
        "/api/analysis/upload",
        files={"file": ("corrupt.mha", b"\x00\xff" * 30_000, "application/octet-stream")},
    )
    assert response.status_code == 400
    error = _error(response)
    assert error["code"] == "UNREADABLE_VOLUME"
    # No Python internals may leak to the client.
    assert "Traceback" not in error["message"]
    assert "File \"" not in error["message"]


def test_accepts_a_real_volume_and_describes_it(app_client, sample_volume):
    with sample_volume.open("rb") as handle:
        response = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        )
    assert response.status_code == 201
    body = response.json()

    assert body["status"] == "queued"
    # The id must not be derived from the filename: nothing identifying in a URL.
    assert body["analysis_id"] not in sample_volume.name
    assert len(body["analysis_id"]) == 16

    study = body["study"]
    assert study["slice_count"] == 24
    assert study["dimensions"] == [242, 305]
    assert study["format"] == ".mha"
    assert study["modality"] == "t2"
    # The modality came from the filename, and must say so rather than implying
    # it was read from DICOM metadata.
    assert study["modality_source"] == "filename_heuristic"
    assert study["native_orientation"] == "LPS"
    assert len(study["in_plane_spacing_mm"]) == 2


def test_sanitises_a_hostile_filename(app_client, sample_volume):
    """Directory traversal in the filename must not escape the storage root."""
    payload = sample_volume.read_bytes()
    response = app_client.post(
        "/api/analysis/upload",
        files={
            "file": (
                "../../../../etc/passwd.mha",
                payload,
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 201
    stored = response.json()["study"]["filename"]
    assert "/" not in stored and "\\" not in stored
    assert ".." not in stored


def test_unknown_analysis_returns_a_clean_404(app_client):
    response = app_client.get("/api/analysis/deadbeefdeadbeef/status")
    assert response.status_code == 404
    assert _error(response)["code"] == "ANALYSIS_NOT_FOUND"


def test_malformed_analysis_id_is_not_found_not_a_crash(app_client):
    """A non-hex id must be refused before it is used to build a path."""
    response = app_client.get("/api/analysis/..%2F..%2Fsecret/status")
    assert response.status_code in (404, 400)
    assert "error" in response.json()
