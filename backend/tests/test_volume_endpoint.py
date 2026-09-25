"""Sprint 8 - the volume endpoint for browser-side 3D rendering.

This endpoint deliberately reverses the previous "the volume never crosses the
network" property, so these tests pin what it does and does not do: it serves only
completed analyses, it round-trips the exact voxels the 2D viewer renders from, it
carries the server's finding-overlay decision so the 3D view cannot disagree with
the 2D one, and it never touches raw dataset files.
"""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from backend.app.services.imaging import DISC_INSTANCE_OFFSET, disc_region
from backend.app.services.volume_export import (
    CHANNELS,
    VOLUME_FORMAT_VERSION,
    build_volume_payload,
    parse_volume_payload,
)

PIPELINE_VERSION = "SPIDER-Lumbar-v1"


def _wait(client, analysis_id: str, timeout: float = 420.0) -> dict:
    deadline = time.time() + timeout
    state: dict = {}
    while time.time() < deadline:
        state = client.get(f"/api/analysis/{analysis_id}/status").json()
        if state["status"] in ("completed", "failed"):
            break
        time.sleep(0.4)
    return state


@pytest.fixture(scope="module")
def analysis(app_client):
    response = app_client.post("/api/analysis/sample")
    if response.status_code == 404:
        pytest.skip("Sample volume not extracted on this machine")
    assert response.status_code == 201, response.text
    analysis_id = response.json()["analysis_id"]
    app_client.post(f"/api/analysis/{analysis_id}/run")
    state = _wait(app_client, analysis_id)
    assert state["status"] == "completed", state
    return {
        "id": analysis_id,
        "result": app_client.get(f"/api/analysis/{analysis_id}/result").json(),
    }


# ---------------------------------------------------------------------------
# The serialiser, in isolation
# ---------------------------------------------------------------------------


def _synthetic_arrays(slices=4, rows=8, cols=6):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(slices, rows, cols), dtype=np.uint8)
    semantic = np.zeros((slices, rows, cols), dtype=np.uint8)
    instance = np.zeros((slices, rows, cols), dtype=np.uint8)
    semantic[:, 2:4, 1:5] = 2
    instance[:, 2:4, 1:5] = DISC_INSTANCE_OFFSET + 1
    return {
        "images": image, "semantic": semantic, "instance": instance,
        "slice_ids": [f"s{i}" for i in range(slices)],
    }


def test_the_payload_round_trips_every_voxel():
    arrays = _synthetic_arrays()
    payload, header = build_volume_payload(arrays, analysis_id="abc123abc123abcd")
    parsed_header, channels = parse_volume_payload(payload)

    assert parsed_header["format_version"] == VOLUME_FORMAT_VERSION
    assert set(channels) == set(CHANNELS)
    for name, key in (("image", "images"), ("semantic", "semantic"),
                      ("instance", "instance")):
        assert np.array_equal(channels[name], arrays[key]), name


def test_the_header_declares_geometry_and_axis_order():
    arrays = _synthetic_arrays(slices=4, rows=8, cols=6)
    _, header = build_volume_payload(arrays, analysis_id="abc123abc123abcd")
    assert header["axis_order"] == ["slice", "row", "col"]
    assert header["dimensions"] == {"slices": 4, "rows": 8, "cols": 6}
    # Preprocessing resampled in-plane to 1.0 mm/px; that is what makes the 3D
    # view's millimetre scale meaningful.
    assert header["spacing_mm"]["row"] == 1.0
    assert header["spacing_mm"]["col"] == 1.0


def test_through_plane_spacing_is_null_rather_than_guessed():
    arrays = _synthetic_arrays()
    _, header = build_volume_payload(arrays, analysis_id="a" * 16, study={})
    assert header["spacing_mm"]["slice"] is None

    _, header = build_volume_payload(
        arrays, analysis_id="a" * 16, study={"slice_spacing_mm": 4.0}
    )
    assert header["spacing_mm"]["slice"] == 4.0


def test_the_header_length_prefix_matches_the_json():
    arrays = _synthetic_arrays()
    payload, _ = build_volume_payload(arrays, analysis_id="a" * 16)
    (declared,) = struct.unpack("<I", payload[:4])
    assert payload[4:4 + declared].decode("utf-8").startswith("{")


def test_channel_offsets_are_contiguous_and_sized():
    arrays = _synthetic_arrays()
    payload, header = build_volume_payload(arrays, analysis_id="a" * 16)
    expected = arrays["images"].size
    offset = 0
    for spec in header["channels"]:
        assert spec["offset"] == offset
        assert spec["length"] == expected
        assert spec["dtype"] == "uint8"
        offset += spec["length"]
    assert header["total_bytes"] == offset
    assert len(payload) == 4 + len(
        __import__("json").dumps(header, separators=(",", ":")).encode()
    ) + offset


def test_the_disc_instance_convention_travels_with_the_volume():
    """The 3D view must not have to hardcode `10 + N`."""
    arrays = _synthetic_arrays()
    _, header = build_volume_payload(arrays, analysis_id="a" * 16)
    assert header["disc_instance_offset"] == DISC_INSTANCE_OFFSET


def test_the_finding_overlay_decision_travels_with_the_volume():
    arrays = _synthetic_arrays()
    _, header = build_volume_payload(
        arrays, analysis_id="a" * 16, finding_discs=(3, 1, 2)
    )
    assert header["finding_discs"] == [1, 2, 3]


def test_mismatched_channel_shapes_are_refused():
    arrays = _synthetic_arrays()
    arrays["semantic"] = arrays["semantic"][:2]
    with pytest.raises(ValueError, match="disagree on shape"):
        build_volume_payload(arrays, analysis_id="a" * 16)


def test_a_non_3d_volume_is_refused():
    arrays = _synthetic_arrays()
    arrays = {k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    with pytest.raises(ValueError):
        build_volume_payload(arrays, analysis_id="a" * 16)


def test_serialising_does_not_mutate_the_stored_arrays():
    arrays = _synthetic_arrays()
    before = {k: v.copy() for k, v in arrays.items() if isinstance(v, np.ndarray)}
    build_volume_payload(arrays, analysis_id="a" * 16)
    for key, original in before.items():
        assert np.array_equal(arrays[key], original), key


def test_a_truncated_payload_is_detected():
    arrays = _synthetic_arrays()
    payload, _ = build_volume_payload(arrays, analysis_id="a" * 16)
    with pytest.raises(ValueError, match="truncated"):
        parse_volume_payload(payload[:-32])


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


def test_the_volume_endpoint_serves_a_completed_analysis(app_client, analysis):
    response = app_client.get(f"/api/analysis/{analysis['id']}/volume")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["X-Volume-Format-Version"] == str(VOLUME_FORMAT_VERSION)


def test_the_served_volume_matches_the_stored_arrays(app_client, analysis):
    """The 3D view must render the same voxels the 2D view renders."""
    response = app_client.get(f"/api/analysis/{analysis['id']}/volume")
    _, channels = parse_volume_payload(response.content)
    stored = app_client.app.state.store.read_arrays(analysis["id"])

    assert np.array_equal(channels["image"], stored["images"])
    assert np.array_equal(channels["semantic"], stored["semantic"])
    assert np.array_equal(channels["instance"], stored["instance"])


def test_the_volume_geometry_matches_the_reported_study(app_client, analysis):
    response = app_client.get(f"/api/analysis/{analysis['id']}/volume")
    header, _ = parse_volume_payload(response.content)
    assert header["dimensions"]["slices"] == analysis["result"]["study"]["slice_count"]
    assert header["analysis_id"] == analysis["id"]
    assert header["pipeline_version"] == PIPELINE_VERSION


def test_the_volume_header_agrees_with_the_finding_overlay(app_client, analysis):
    """3D and 2D must mark the same discs, from one server-side decision."""
    response = app_client.get(f"/api/analysis/{analysis['id']}/volume")
    header, _ = parse_volume_payload(response.content)
    assert header["finding_discs"] == analysis["result"]["finding_overlay"]["disc_indices"]

    from_header = [int(v) for v in
                   (response.headers.get("X-Finding-Discs") or "").split(",") if v]
    assert from_header == header["finding_discs"]


def test_the_instance_channel_locates_the_same_discs_as_the_2d_renderer(
    app_client, analysis
):
    response = app_client.get(f"/api/analysis/{analysis['id']}/volume")
    _, channels = parse_volume_payload(response.content)
    stored = app_client.app.state.store.read_arrays(analysis["id"])

    for disc in analysis["result"]["discs"][:3]:
        index = disc["index"]
        served = channels["instance"] == (DISC_INSTANCE_OFFSET + index)
        # disc_region is what draw_finding_overlay and the bracket already use.
        expected = np.stack([
            disc_region(stored["instance"][i], index)
            for i in range(stored["instance"].shape[0])
        ])
        assert np.array_equal(served, expected), index


def test_the_volume_is_cacheable_as_immutable(app_client, analysis):
    response = app_client.get(f"/api/analysis/{analysis['id']}/volume")
    assert "immutable" in response.headers.get("cache-control", "")


def test_one_request_delivers_the_whole_volume(app_client, analysis):
    """§J: no repeated large transfers."""
    response = app_client.get(f"/api/analysis/{analysis['id']}/volume")
    header, channels = parse_volume_payload(response.content)
    assert len(channels) == len(CHANNELS)
    assert all(c.size > 0 for c in channels.values())
    assert len(response.content) == 4 + len(
        __import__("json").dumps(header, separators=(",", ":")).encode()
    ) + header["total_bytes"]


def test_an_unknown_analysis_is_a_clean_404(app_client):
    response = app_client.get("/api/analysis/deadbeefdeadbeef/volume")
    assert response.status_code == 404
    assert "Traceback" not in response.text


def test_an_invalid_analysis_id_is_refused(app_client):
    response = app_client.get("/api/analysis/not-hex-at-all/volume")
    assert response.status_code == 404


def test_an_incomplete_analysis_has_no_volume(app_client):
    """Restricted to completed analyses: before that the arrays do not exist."""
    created = app_client.post("/api/analysis/sample")
    if created.status_code == 404:
        pytest.skip("Sample volume not extracted on this machine")
    analysis_id = created.json()["analysis_id"]
    try:
        response = app_client.get(f"/api/analysis/{analysis_id}/volume")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "VOLUME_NOT_READY"
    finally:
        app_client.delete(f"/api/analysis/{analysis_id}")


def test_the_volume_endpoint_does_not_disturb_the_slice_endpoint(app_client, analysis):
    """The 2D viewer's path must be untouched by the new one."""
    analysis_id = analysis["id"]
    mid = analysis["result"]["study"]["slice_count"] // 2
    before = app_client.get(f"/api/analysis/{analysis_id}/slice/{mid}",
                            params={"mode": "overlay"})
    app_client.get(f"/api/analysis/{analysis_id}/volume")
    after = app_client.get(f"/api/analysis/{analysis_id}/slice/{mid}",
                           params={"mode": "overlay"})
    assert before.status_code == after.status_code == 200
    assert before.content == after.content


def test_serving_the_volume_leaves_the_stored_arrays_untouched(app_client, analysis):
    store = app_client.app.state.store
    before = store.read_arrays(analysis["id"])
    snapshot = {k: before[k].copy() for k in ("images", "semantic", "instance")}
    app_client.get(f"/api/analysis/{analysis['id']}/volume")
    after = store.read_arrays(analysis["id"])
    for key, original in snapshot.items():
        assert np.array_equal(after[key], original), key
