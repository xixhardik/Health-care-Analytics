"""Sprint 7 additions: pipeline identity, API-served metrics, demo mode,
disc highlighting, provenance labelling, and the remaining error paths.
"""

from __future__ import annotations

import time

import pytest

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


# ---------------------------------------------------------------------------
# 1. Pipeline identity and frozen composition
# ---------------------------------------------------------------------------


def test_health_declares_the_pipeline_version(app_client):
    pipeline = app_client.get("/api/health").json()["pipeline"]
    assert pipeline["version"] == PIPELINE_VERSION
    assert pipeline["preprocessing"] == "Sprint 1"
    assert pipeline["segmentation"] == "Sprint 3"
    assert pipeline["postprocessing"] == "Sprint 5"


def test_sprint4_is_documented_as_excluded(app_client):
    """The rejected experiment must be named and explained, not hidden."""
    excluded = app_client.get("/api/health").json()["pipeline"]["excluded"]
    assert "Sprint 4" in excluded
    assert "not selected" in excluded["Sprint 4"]


def test_health_serves_the_metrics_so_the_ui_need_not_hardcode_them(app_client):
    body = app_client.get("/api/health").json()

    metrics = body["validated_metrics"]
    assert metrics["segmentation_macro_foreground_dice"] == 0.90001
    assert metrics["disc_indexing_percent"] == 93.84
    assert metrics["disc_indexing_baseline_percent"] == 84.08
    assert metrics["disc_height_mae_mm"] == 0.6492
    assert metrics["disc_area_mae_mm2"] == 21.4831
    assert metrics["intensity_ratio_pearson_r"] == 0.9726
    assert metrics["pfirrmann_qwk_end_to_end"] == 0.6543

    table = body["metric_table"]
    assert len(table) >= 10
    kinds = {row["kind"] for row in table}
    # Segmentation and post-processing figures must stay distinguishable.
    assert kinds == {"segmentation", "postprocessing"}
    for row in table:
        assert row["label"] and row["value"] and row["sprint"]


def test_health_serves_the_research_progress_including_the_rejection(app_client):
    stages = app_client.get("/api/health").json()["research_progress"]
    sprints = {stage["sprint"]: stage for stage in stages}
    assert {"Sprint 2", "Sprint 3", "Sprint 4", "Sprint 5"} <= set(sprints)

    sprint4 = sprints["Sprint 4"]
    assert sprint4["outcome"] == "rejected"
    assert "did not improve" in sprint4["summary"]
    assert sprints["Sprint 5"]["outcome"] == "selected"


def test_health_serves_provenance_and_evidence_vocabulary(app_client):
    body = app_client.get("/api/health").json()

    provenance = body["provenance_labels"]
    assert provenance["segmentation_derived"] == "Segmentation-derived"
    assert provenance["model_prediction"] == "Model-estimated"
    assert provenance["unsupported"] == "Not modelled"

    evidence = body["evidence_labels"]
    assert set(evidence) == {"strong", "moderate", "modest", "weak"}
    # Qualitative only: a numeric confidence was never validated.
    for label in evidence.values():
        assert "%" not in label


def test_result_carries_the_pipeline_version(app_client, sample_analysis):
    result = sample_analysis["result"]
    assert result["pipeline"]["version"] == PIPELINE_VERSION
    assert result["provenance_labels"]["model_prediction"] == "Model-estimated"


# ---------------------------------------------------------------------------
# 2. Demo mode
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_analysis(app_client):
    """Create and run an analysis through the sample-study endpoint."""
    response = app_client.post("/api/analysis/sample")
    if response.status_code == 404:
        pytest.skip("Sample volume not extracted on this machine")
    assert response.status_code == 201, response.text
    created = response.json()
    analysis_id = created["analysis_id"]

    app_client.post(f"/api/analysis/{analysis_id}/run")
    state = _wait(app_client, analysis_id)
    assert state["status"] == "completed", state
    result = app_client.get(f"/api/analysis/{analysis_id}/result").json()
    return {"id": analysis_id, "created": created, "result": result}


def test_sample_endpoint_creates_a_real_flagged_analysis(sample_analysis):
    created = sample_analysis["created"]
    assert created["is_sample"] is True
    assert created["status"] == "queued"
    # A real volume was read, not a canned payload.
    assert created["study"]["slice_count"] > 0
    assert "Sample research study" in created["message"]


def test_sample_analysis_runs_the_real_pipeline(sample_analysis):
    result = sample_analysis["result"]
    assert result["is_sample"] is True
    assert len(result["discs"]) > 0
    assert result["processing"]["duration_seconds"] > 0
    # Real measurements, not placeholders.
    for disc in result["discs"]:
        assert disc["measurements"]["height_mm_central"] is not None


def test_sample_is_labelled_in_history(app_client, sample_analysis):
    items = app_client.get("/api/analysis").json()["items"]
    entry = next(i for i in items if i["analysis_id"] == sample_analysis["id"])
    assert entry["is_sample"] is True


# ---------------------------------------------------------------------------
# 3. Disc highlight / viewer synchronisation
# ---------------------------------------------------------------------------


def test_highlight_changes_the_rendered_slice(app_client, sample_analysis):
    """The highlight must actually be drawn, so the viewer can track selection."""
    analysis_id = sample_analysis["id"]
    disc = sample_analysis["result"]["discs"][0]
    slice_index = disc["representative_slice_index"]

    plain = app_client.get(
        f"/api/analysis/{analysis_id}/slice/{slice_index}", params={"mode": "overlay"}
    )
    highlighted = app_client.get(
        f"/api/analysis/{analysis_id}/slice/{slice_index}",
        params={"mode": "overlay", "highlight_disc": disc["index"]},
    )
    assert plain.status_code == highlighted.status_code == 200
    assert plain.content != highlighted.content


def test_highlight_index_is_validated(app_client, sample_analysis):
    response = app_client.get(
        f"/api/analysis/{sample_analysis['id']}/slice/5",
        params={"highlight_disc": 99},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_highlight_of_an_absent_disc_still_renders(app_client, sample_analysis):
    """A disc not present on this slice must not break the image."""
    response = app_client.get(
        f"/api/analysis/{sample_analysis['id']}/slice/0",
        params={"mode": "overlay", "highlight_disc": 9},
    )
    assert response.status_code == 200
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# 4. Structured, provenance-tagged presentation
# ---------------------------------------------------------------------------


def test_every_disc_has_structured_provenance_tagged_lines(sample_analysis):
    for disc in sample_analysis["result"]["discs"]:
        lines = disc["structured"]
        assert lines, "structured summary lines are required for display"

        by_label = {line["label"]: line for line in lines}
        assert "Pfirrmann grade" in by_label
        assert "Disc height (central)" in by_label
        assert "Modic type" in by_label
        assert "Spondylolisthesis" in by_label

        # Measurements are segmentation-derived, estimates are model-derived.
        assert by_label["Disc height (central)"]["provenance"] == "segmentation_derived"
        assert by_label["Pfirrmann grade"]["provenance"] == "model_prediction"
        assert by_label["Modic type"]["provenance"] == "unsupported"
        assert by_label["Spondylolisthesis"]["provenance"] == "unsupported"


def test_unavailable_values_never_render_as_a_negative(sample_analysis):
    """Null means unavailable. It must not read as 0, No or Normal."""
    for disc in sample_analysis["result"]["discs"]:
        for line in disc["structured"]:
            if line["available"]:
                continue
            assert line["value"] == "unavailable", line
            assert line["value"].lower() not in ("0", "no", "normal", "not detected")
            # An unavailable value must explain itself.
            if line["provenance"] == "unsupported":
                assert line["unavailable_reason"]


def test_binary_findings_distinguish_detected_from_unavailable(sample_analysis):
    for disc in sample_analysis["result"]["discs"]:
        by_label = {line["label"]: line for line in disc["structured"]}
        bulging = by_label["Bulging"]
        assert bulging["value"] in ("detected", "not detected")
        assert bulging["available"] is True
        # Modic type has no model, so it is never given a detected/not state.
        assert by_label["Modic type"]["value"] == "unavailable"


def test_served_findings_carry_a_qualitative_strength(sample_analysis):
    for disc in sample_analysis["result"]["discs"]:
        for line in disc["structured"]:
            if line["provenance"] != "model_prediction":
                continue
            if not line["available"]:
                continue
            assert line["strength"] in ("strong", "moderate", "modest", "weak")


# ---------------------------------------------------------------------------
# 5. Report content
# ---------------------------------------------------------------------------


def test_report_states_the_pipeline_version_and_timestamp(app_client, sample_analysis):
    body = app_client.get(
        f"/api/analysis/{sample_analysis['id']}/download", params={"fmt": "md"}
    ).text

    assert PIPELINE_VERSION in body
    assert "| Analysis id |" in body
    assert "| Processed |" in body
    assert "Sample research study" in body
    # The rejected experiment is disclosed in the report too.
    assert "Sprint 4 is deliberately not served" in body


def test_report_identifies_unsupported_findings(app_client, sample_analysis):
    body = app_client.get(
        f"/api/analysis/{sample_analysis['id']}/download", params={"fmt": "md"}
    ).text
    assert "Targets deliberately not reported" in body
    assert "spondylolisthesis" in body
    assert "modic_type" in body


# ---------------------------------------------------------------------------
# 6. Remaining error paths
# ---------------------------------------------------------------------------


def test_missing_output_is_reported_not_crashed(app_client, sample_volume):
    """Slice images requested before completion must give a clear 409."""
    with sample_volume.open("rb") as handle:
        created = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        ).json()
    response = app_client.get(f"/api/analysis/{created['analysis_id']}/slice/0")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SLICES_NOT_READY"


def test_download_before_completion_is_a_conflict(app_client, sample_volume):
    with sample_volume.open("rb") as handle:
        created = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        ).json()
    response = app_client.get(f"/api/analysis/{created['analysis_id']}/download")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "RESULT_NOT_READY"


def test_interrupted_job_is_reported_as_such(app_client, sample_volume):
    """A record left at 'processing' with no live job is stale, not running."""
    with sample_volume.open("rb") as handle:
        created = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        ).json()
    analysis_id = created["analysis_id"]

    store = app_client.app.state.store
    store.update_record(analysis_id, status="processing", progress=40,
                        stage="Segmentation")
    # Drop the live job entry to simulate a restart mid-analysis.
    app_client.app.state.jobs._jobs.pop(analysis_id, None)

    state = app_client.get(f"/api/analysis/{analysis_id}/status").json()
    assert state["status"] == "failed"
    assert state["error"]["code"] == "ANALYSIS_INTERRUPTED"
    # And it must be restartable rather than wedged.
    assert app_client.post(f"/api/analysis/{analysis_id}/run").status_code == 202
    _wait(app_client, analysis_id)


def test_malformed_query_parameters_are_rejected_cleanly(app_client, sample_analysis):
    response = app_client.get(
        f"/api/analysis/{sample_analysis['id']}/slice/0",
        params={"mode": "hologram"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert "Traceback" not in response.text


def test_unknown_route_returns_the_error_envelope(app_client):
    response = app_client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.json()
