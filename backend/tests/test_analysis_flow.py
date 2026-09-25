"""The analysis job: state transitions, real progress, and the result schema.

This is the end-to-end smoke test: upload, create, process, retrieve result,
retrieve a slice. It runs the real pipeline on one small volume.
"""

from __future__ import annotations

import time

import pytest

TERMINAL = ("completed", "failed")


def _wait(client, analysis_id: str, timeout: float = 420.0) -> dict:
    """Poll until terminal, collecting the progress trace."""
    trace: list[tuple[int, str]] = []
    deadline = time.time() + timeout
    state: dict = {}
    while time.time() < deadline:
        state = client.get(f"/api/analysis/{analysis_id}/status").json()
        point = (state["progress"], state["stage"])
        if not trace or trace[-1] != point:
            trace.append(point)
        if state["status"] in TERMINAL:
            break
        time.sleep(0.4)
    state["_trace"] = trace
    return state


@pytest.fixture(scope="module")
def completed(app_client, sample_volume):
    """Run one analysis to completion and share it across assertions."""
    with sample_volume.open("rb") as handle:
        created = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        ).json()
    analysis_id = created["analysis_id"]

    run = app_client.post(f"/api/analysis/{analysis_id}/run")
    assert run.status_code == 202
    assert run.json()["status"] in ("queued", "processing")

    state = _wait(app_client, analysis_id)
    assert state["status"] == "completed", state
    result = app_client.get(f"/api/analysis/{analysis_id}/result").json()
    return {"id": analysis_id, "state": state, "result": result}


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def test_run_returns_immediately_without_blocking(app_client, uploaded):
    """POST /run must return promptly; the work happens in the background."""
    analysis_id = uploaded["analysis_id"]
    started = time.perf_counter()
    response = app_client.post(f"/api/analysis/{analysis_id}/run")
    elapsed = time.perf_counter() - started

    assert response.status_code == 202
    assert response.json() == {
        "analysis_id": analysis_id,
        "status": response.json()["status"],
    }
    # The pipeline takes seconds; the call must not wait for it.
    assert elapsed < 2.0, f"run blocked for {elapsed:.2f}s"
    _wait(app_client, analysis_id)


def test_result_before_completion_is_a_conflict_not_an_empty_object(
    app_client, sample_volume
):
    with sample_volume.open("rb") as handle:
        created = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        ).json()
    response = app_client.get(f"/api/analysis/{created['analysis_id']}/result")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "RESULT_NOT_READY"


def test_status_transitions_and_progress_is_monotonic(completed):
    state = completed["state"]
    trace = state["_trace"]

    assert state["status"] == "completed"
    assert state["progress"] == 100
    assert state["elapsed_seconds"] > 0

    values = [progress for progress, _ in trace]
    assert values == sorted(values), f"progress went backwards: {values}"
    assert values[-1] == 100

    # Real pipeline stages must be reported, not invented ones.
    stages = set(state["stages_completed"])
    assert {"Upload validated", "Segmentation", "Complete"} <= stages


def test_slice_count_matches_the_uploaded_volume(completed):
    assert completed["result"]["study"]["slice_count"] == 24


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


def test_result_schema_is_complete(completed):
    result = completed["result"]
    for key in (
        "analysis_id", "status", "created_at", "completed_at", "study",
        "timepoint", "segmentation", "processing", "discs", "summary",
        "validated_metrics", "disclaimer",
    ):
        assert key in result, f"missing {key}"

    assert result["status"] == "completed"
    assert result["segmentation"]["slice_count"] == 24
    assert [c["name"] for c in result["segmentation"]["classes"]] == [
        "background", "vertebra", "intervertebral_disc", "spinal_canal",
    ]

    processing = result["processing"]
    assert processing["segmentation_source"] == "Sprint 3 Coverage (best validated model)"
    assert processing["postprocessing_method"] == "Sprint 5 5A+5D"
    assert processing["duration_seconds"] > 0
    assert processing["stage_durations"]

    # Single-timepoint seam for a future longitudinal module.
    assert result["timepoint"]["is_baseline"] is True
    assert result["timepoint"]["id"] == "baseline"


def test_discs_carry_real_measurements(completed):
    discs = completed["result"]["discs"]
    assert len(discs) > 0, "the pipeline found no disc in a known-good study"

    # Indices must be a contiguous run starting at 1 (most inferior upward).
    indices = [d["index"] for d in discs]
    assert indices == sorted(indices)
    assert indices[0] == 1

    for disc in discs:
        m = disc["measurements"]
        assert m["source"] == "segmentation_derived_measurement"
        # A real measurement, in a plausible physical range for a lumbar disc.
        assert m["height_mm_central"] is not None
        assert 1.0 < m["height_mm_central"] < 25.0
        assert m["area_mm2"] is not None and m["area_mm2"] > 0
        assert disc["slices_present"] and disc["slices_present"] > 0
        # Identity support comes from Sprint 5 track voting.
        assert disc["identity_confidence"] is None or 0.0 <= disc["identity_confidence"] <= 1.0


def test_unsupported_findings_are_null_with_a_reason_not_fabricated(completed):
    result = completed["result"]

    for disc in result["discs"]:
        # These two must never carry a value.
        assert disc["modic_type"] is None
        assert disc["spondylolisthesis"] is None

        by_name = {f["name"]: f for f in disc["findings"]}
        for target in ("spondylolisthesis", "modic_type"):
            assert target in by_name, f"{target} must be declared, not omitted"
            entry = by_name[target]
            assert entry["value"] is None
            assert entry["unavailable_reason"]
            assert entry["source"] == "unsupported"

    assert "spondylolisthesis" in result["summary"]["unsupported_targets"]
    assert "modic_type" in result["summary"]["unsupported_targets"]


def test_served_findings_carry_probabilities_and_provenance(completed):
    for disc in completed["result"]["discs"]:
        by_name = {f["name"]: f for f in disc["findings"]}
        for target in ("narrowing", "bulging", "any_modic"):
            entry = by_name[target]
            assert isinstance(entry["value"], bool)
            assert 0.0 <= entry["probability"] <= 1.0
            assert entry["source"] == "model_prediction"
            # Every estimate states how well it validated.
            assert entry["strength"] in ("strong", "moderate", "modest", "weak")
            assert entry["validated_value"] is not None


def test_pfirrmann_is_reported_for_a_t2_study(completed):
    """The sample is T2, so the grade must be present and in range."""
    for disc in completed["result"]["discs"]:
        grade = disc["pfirrmann_grade"]
        assert grade is not None, "T2 study should receive a grade"
        assert 1 <= grade <= 5
        entry = next(f for f in disc["findings"] if f["name"] == "pfirrmann_grade")
        assert entry["expected"] is not None
        assert entry["probabilities"]
        assert abs(sum(entry["probabilities"].values()) - 1.0) < 0.05


def test_summary_is_template_generated_and_scoped(completed):
    summary = completed["result"]["summary"]
    assert summary["disc_count"] == len(completed["result"]["discs"])
    assert summary["findings"]

    categories = {f["category"] for f in summary["findings"]}
    assert "Detection" in categories
    # The scope statement must always be present.
    assert "Scope" in categories

    text = " ".join(f["text"] for f in summary["findings"]).lower()
    assert "postoperative" in text
    # No composite damage score.
    assert "damage" not in text

    # The detection statement must explicitly disclaim anatomical level naming.
    detection = next(f for f in summary["findings"] if f["category"] == "Detection")
    assert "no anatomical level name" in detection["text"].lower()

    # No level name may be asserted as an actual finding. The disclaimer above
    # mentions one as an example, so only the substantive statements are checked.
    substantive = [
        f["text"].lower() for f in summary["findings"]
        if f["category"] != "Detection"
    ]
    for statement in substantive:
        for banned in ("l1-l2", "l2-l3", "l3-l4", "l4-l5", "l5-s1"):
            assert banned not in statement, f"level name asserted: {statement}"


def test_validated_metrics_are_the_published_figures(completed):
    metrics = completed["result"]["validated_metrics"]
    assert metrics["segmentation_macro_foreground_dice"] == 0.90001
    assert metrics["disc_indexing_percent"] == 93.84
    assert metrics["disc_indexing_baseline_percent"] == 84.08
    assert metrics["disc_height_mae_mm"] == 0.6492
    assert metrics["pfirrmann_qwk_end_to_end"] == 0.6543
    assert metrics["test_patients"] == 33
    # The note must distinguish segmentation from post-processing metrics.
    assert "Sprint 3" in metrics["note"] and "Sprint 5" in metrics["note"]


# ---------------------------------------------------------------------------
# Slice imaging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["original", "mask", "overlay"])
def test_slice_endpoint_returns_png(app_client, completed, mode):
    response = app_client.get(
        f"/api/analysis/{completed['id']}/slice/12", params={"mode": mode}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert response.headers["X-Slice-Count"] == "24"
    # Completed analyses are immutable, so the browser should cache hard.
    assert "immutable" in response.headers.get("cache-control", "")


def test_slice_out_of_range_is_rejected(app_client, completed):
    response = app_client.get(f"/api/analysis/{completed['id']}/slice/999")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "SLICE_OUT_OF_RANGE"
    assert error["details"]["slice_count"] == 24


def test_hidden_classes_are_not_rendered(app_client, completed):
    """Filtering a class must change the image, proving it happens server-side."""
    base = app_client.get(
        f"/api/analysis/{completed['id']}/slice/12",
        params={"mode": "mask", "classes": "1,2,3"},
    ).content
    discs_only = app_client.get(
        f"/api/analysis/{completed['id']}/slice/12",
        params={"mode": "mask", "classes": "2"},
    ).content
    assert base != discs_only


# ---------------------------------------------------------------------------
# Report and history
# ---------------------------------------------------------------------------


def test_report_matches_the_result(app_client, completed):
    report = app_client.get(f"/api/analysis/{completed['id']}/report").json()
    assert report["analysis_id"] == completed["result"]["analysis_id"]
    assert len(report["discs"]) == len(completed["result"]["discs"])


@pytest.mark.parametrize(
    "fmt,fragment", [("md", "# Lumbar Spine MRI Analysis Report"), ("json", "\"discs\"")]
)
def test_download_produces_an_attachment(app_client, completed, fmt, fragment):
    response = app_client.get(
        f"/api/analysis/{completed['id']}/download", params={"fmt": fmt}
    )
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    body = response.content.decode("utf-8")
    assert fragment in body
    if fmt == "md":
        # The report must carry the disclaimer and the limitations section.
        assert "require review by a qualified radiologist" in body
        assert "## 7. Limitations" in body
        assert "postoperative" in body.lower()


def test_history_lists_the_analysis(app_client, completed):
    body = app_client.get("/api/analysis").json()
    ids = {item["analysis_id"] for item in body["items"]}
    assert completed["id"] in ids
    assert body["counts"].get("completed", 0) >= 1

    entry = next(i for i in body["items"] if i["analysis_id"] == completed["id"])
    assert entry["slice_count"] == 24
    assert entry["disc_count"] == len(completed["result"]["discs"])


def test_delete_removes_the_analysis(app_client, sample_volume):
    with sample_volume.open("rb") as handle:
        created = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        ).json()
    analysis_id = created["analysis_id"]

    response = app_client.delete(f"/api/analysis/{analysis_id}")
    assert response.status_code == 200
    assert response.json() == {"analysis_id": analysis_id, "deleted": True}
    assert app_client.get(f"/api/analysis/{analysis_id}/status").status_code == 404
