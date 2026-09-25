"""Sprint 8 - longitudinal demonstration cases.

The thing these tests defend is not a feature, it is a claim: that this system
never represents the SPIDER dataset as containing postoperative follow-up. So
alongside the ordinary discovery/ordering/404 coverage, several tests assert the
*refusals* - a manifest that declares itself real follow-up must not load, a
missing source volume must not be substituted, and provenance must be
inseparable from any stage a client can obtain.
"""

from __future__ import annotations

import json

import pytest

from backend.app.services.longitudinal import (
    REQUIRED_CASE_TYPE,
    RESEARCH_STATUSES,
    SURGICAL_EVALUATION_WORDING,
    UNAVAILABLE_NOTICE,
    ManifestError,
    get_stage,
    list_case_ids,
    load_case,
    load_cases,
    resolve_volume,
    stage_ids,
)

CASE_ID = "LS-DEMO-001"
EXPECTED_STAGES = ["pre_surgery", "post_surgery", "month_3", "month_6"]


@pytest.fixture
def project_root(app_client):
    return app_client.app.state.settings.project_root


# ---------------------------------------------------------------------------
# Discovery and ordering
# ---------------------------------------------------------------------------


def test_the_shipped_demo_case_is_discovered(project_root):
    assert CASE_ID in list_case_ids(project_root)


def test_case_ids_are_stable_across_calls(project_root):
    assert list_case_ids(project_root) == list_case_ids(project_root)


def test_stages_are_ordered_by_the_manifests_own_order(project_root):
    case = load_case(project_root, CASE_ID)
    assert stage_ids(case) == EXPECTED_STAGES
    assert [s["order"] for s in case["stages"]] == [1, 2, 3, 4]


def test_stage_order_does_not_depend_on_json_key_order(project_root, tmp_path):
    """Shuffle the manifest's stage list; the timeline must not move."""
    original = json.loads(
        (project_root / "demo/longitudinal_cases" / CASE_ID / "manifest.json")
        .read_text(encoding="utf-8")
    )
    shuffled = dict(original)
    shuffled["stages"] = list(reversed(original["stages"]))

    case_dir = tmp_path / "demo" / "longitudinal_cases" / "SHUFFLED-01"
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(json.dumps(shuffled), encoding="utf-8")

    case = load_case(tmp_path, "SHUFFLED-01")
    assert [s["order"] for s in case["stages"]] == [1, 2, 3, 4]


def test_the_case_declares_four_stages(project_root):
    case = load_case(project_root, CASE_ID)
    assert case["stage_count"] == 4


# ---------------------------------------------------------------------------
# The disclaimer and the refusals
# ---------------------------------------------------------------------------


def test_the_case_is_typed_as_a_simulation(project_root):
    case = load_case(project_root, CASE_ID)
    assert case["type"] == REQUIRED_CASE_TYPE
    assert case["is_true_followup"] is False
    assert case["is_same_patient"] is False


def test_the_disclaimer_states_the_dataset_has_no_followup(project_root):
    case = load_case(project_root, CASE_ID)
    disclaimer = case["disclaimer"].lower()
    assert "simulated longitudinal demonstration" in disclaimer
    assert "does not contain true postoperative longitudinal follow-up" in disclaimer


def test_the_ui_notice_says_the_stages_are_different_studies(project_root):
    notice = load_case(project_root, CASE_ID)["ui_notice"].lower()
    assert "different spider studies" in notice
    assert "not postoperative follow-up scans from the same patient" in notice


def test_a_manifest_claiming_real_followup_is_refused(tmp_path):
    """The central refusal. Serving this would misrepresent the dataset."""
    case_dir = tmp_path / "demo" / "longitudinal_cases" / "FAKE-01"
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(json.dumps({
        "case_id": "FAKE-01",
        "type": "REAL_LONGITUDINAL",
        "stages": [{"order": 1, "stage_id": "a", "label": "A"}],
    }), encoding="utf-8")

    with pytest.raises(ManifestError, match="Only"):
        load_case(tmp_path, "FAKE-01")


def test_a_manifest_claiming_one_patient_is_refused(tmp_path):
    case_dir = tmp_path / "demo" / "longitudinal_cases" / "SAME-01"
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(json.dumps({
        "case_id": "SAME-01",
        "type": REQUIRED_CASE_TYPE,
        "is_same_patient": True,
        "stages": [{"order": 1, "stage_id": "a", "label": "A"}],
    }), encoding="utf-8")

    with pytest.raises(ManifestError, match="claims to be real follow-up"):
        load_case(tmp_path, "SAME-01")


def test_a_broken_manifest_is_skipped_not_fatal(project_root, tmp_path):
    """One bad case must not take the whole listing down."""
    root = tmp_path
    good = root / "demo" / "longitudinal_cases" / "GOOD-01"
    good.mkdir(parents=True)
    (good / "manifest.json").write_text(json.dumps({
        "case_id": "GOOD-01", "type": REQUIRED_CASE_TYPE,
        "disclaimer": "d", "ui_notice": "n",
        "stages": [{"order": 1, "stage_id": "a", "label": "A"}],
    }), encoding="utf-8")
    bad = root / "demo" / "longitudinal_cases" / "BAD-01"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{not json", encoding="utf-8")

    cases = load_cases(root)
    assert [c["case_id"] for c in cases] == ["GOOD-01"]


def test_a_manifest_with_no_stages_is_refused(tmp_path):
    case_dir = tmp_path / "demo" / "longitudinal_cases" / "EMPTY-01"
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(json.dumps({
        "case_id": "EMPTY-01", "type": REQUIRED_CASE_TYPE, "stages": [],
    }), encoding="utf-8")

    with pytest.raises(ManifestError, match="no stages"):
        load_case(tmp_path, "EMPTY-01")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_every_stage_carries_its_real_source_study(project_root):
    case = load_case(project_root, CASE_ID)
    sources = [s["provenance"]["source_study_id"] for s in case["stages"]]
    assert sources == ["177_t2", "106_t2", "16_t2", "6_t2"]


def test_the_four_stages_are_four_different_patients(project_root):
    """The whole point: this is not one patient over time."""
    case = load_case(project_root, CASE_ID)
    patients = [s["provenance"]["source_patient_id"] for s in case["stages"]]
    assert len(set(patients)) == len(patients) == 4


def test_every_stage_is_flagged_as_not_a_followup(project_root):
    case = load_case(project_root, CASE_ID)
    assert all(s["provenance"]["is_true_followup"] is False for s in case["stages"])


def test_every_stage_declares_whether_its_numbers_are_real(project_root):
    case = load_case(project_root, CASE_ID)
    for stage in case["stages"]:
        assert stage["provenance"]["measurement_source"] in (
            "real_pipeline", "simulated_demo_value"
        )


def test_the_shipped_case_uses_real_pipeline_measurements(project_root):
    """No fabricated clinical numbers anywhere in this case."""
    case = load_case(project_root, CASE_ID)
    assert all(
        s["provenance"]["measurement_source"] == "real_pipeline"
        for s in case["stages"]
    )
    assert case["measurement_policy"]["source"] == "real_pipeline"


def test_the_measurement_policy_warns_the_trend_is_a_selection_artefact(project_root):
    caveat = load_case(project_root, CASE_ID)["measurement_policy"]["trend_caveat"]
    assert "not an observed recovery trajectory" in caveat.lower()


def test_stages_come_from_the_held_out_test_split(project_root):
    case = load_case(project_root, CASE_ID)
    assert all(s["provenance"]["source_split"] == "test" for s in case["stages"])


def test_stages_expose_a_non_identifying_display_reference(project_root):
    case = load_case(project_root, CASE_ID)
    for stage in case["stages"]:
        assert stage["display_reference"].startswith("Demonstration study")
        # The identifier exists for provenance but is not what the UI shows.
        assert str(stage["provenance"]["source_patient_id"]) not in stage["display_reference"]


def test_no_stage_invents_an_acquisition_time(project_root):
    case = load_case(project_root, CASE_ID)
    assert all(s["acquisition_timestamp"] is None for s in case["stages"])


def test_research_statuses_are_descriptive_not_diagnostic(project_root):
    case = load_case(project_root, CASE_ID)
    for stage in case["stages"]:
        assert stage["research_status"] in RESEARCH_STATUSES
    assert "Baseline" in RESEARCH_STATUSES
    assert "Follow-up data unavailable" in RESEARCH_STATUSES


def test_the_surgical_wording_is_a_prompt_for_review_not_a_recommendation():
    wording = SURGICAL_EVALUATION_WORDING.lower()
    assert wording.startswith("surgical evaluation may be appropriate")
    assert "based on detected imaging findings" in wording
    # It must not tell a patient anything.
    for forbidden in ("you need", "you should", "we recommend", "requires surgery"):
        assert forbidden not in wording


def test_the_single_study_notice_is_available():
    assert UNAVAILABLE_NOTICE == "Longitudinal follow-up unavailable for this study."


# ---------------------------------------------------------------------------
# Volume resolution and path safety
# ---------------------------------------------------------------------------


def test_stage_volumes_resolve_when_the_dataset_is_present(project_root):
    case = load_case(project_root, CASE_ID)
    resolved = [
        resolve_volume(project_root, f"data/extracted/images/"
                                     f"{s['provenance']['source_study_id']}.mha")
        for s in case["stages"]
    ]
    if all(r is None for r in resolved):
        pytest.skip("Dataset not extracted on this machine")
    assert all(r is not None for r in resolved), (
        "some stage volumes are missing; extract with scripts/01_extract.py"
    )


def test_a_missing_volume_marks_the_stage_unavailable_with_a_reason(tmp_path):
    """No substitution, ever. An absent scan is reported, not replaced."""
    case_dir = tmp_path / "demo" / "longitudinal_cases" / "GAP-01"
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(json.dumps({
        "case_id": "GAP-01", "type": REQUIRED_CASE_TYPE,
        "disclaimer": "d", "ui_notice": "n",
        "stages": [{
            "order": 1, "stage_id": "pre_surgery", "label": "Pre-Surgery",
            "source_study_id": "999_t2",
            "volume_path": "data/extracted/images/999_t2.mha",
        }],
    }), encoding="utf-8")

    stage = load_case(tmp_path, "GAP-01")["stages"][0]
    assert stage["available"] is False
    assert "not present on this machine" in stage["unavailable_reason"]
    assert "No substitute scan is used." in stage["unavailable_reason"]


def test_a_volume_path_escaping_the_data_directory_is_rejected(project_root):
    for hostile in (
        "../../../../Windows/System32/drivers/etc/hosts",
        "outputs/checkpoints/sprint3_coverage/best_val_dice.pt",
        "/etc/passwd",
    ):
        assert resolve_volume(project_root, hostile) is None


def test_an_empty_volume_path_resolves_to_nothing(project_root):
    assert resolve_volume(project_root, "") is None


def test_an_invalid_case_id_is_rejected_before_touching_the_filesystem(project_root):
    for hostile in ("../secrets", "a/b", "case id", ""):
        with pytest.raises((ManifestError, FileNotFoundError)):
            load_case(project_root, hostile)


def test_get_stage_raises_keyerror_for_an_unknown_stage(project_root):
    with pytest.raises(KeyError):
        get_stage(project_root, CASE_ID, "month_12")


def test_get_stage_returns_the_requested_stage(project_root):
    stage = get_stage(project_root, CASE_ID, "month_3")
    assert stage["label"] == "3-Month Recovery"
    assert stage["provenance"]["source_study_id"] == "16_t2"


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------


def test_the_case_list_endpoint_carries_the_notice(app_client):
    body = app_client.get("/api/longitudinal/demo-cases").json()
    assert body["total"] >= 1
    notice = body["notice"].lower()
    assert "no postoperative longitudinal follow-up" in notice
    assert "no recovery outcome is claimed" in notice
    assert "Baseline" in body["research_statuses"]


def test_the_listed_case_is_the_shipped_simulation(app_client):
    body = app_client.get("/api/longitudinal/demo-cases").json()
    case = next(c for c in body["cases"] if c["case_id"] == CASE_ID)
    assert case["type"] == REQUIRED_CASE_TYPE
    assert case["is_true_followup"] is False
    assert len(case["stages"]) == 4


def test_the_case_endpoint_returns_ordered_stages(app_client):
    case = app_client.get(f"/api/longitudinal/demo-cases/{CASE_ID}").json()
    assert [s["stage_id"] for s in case["stages"]] == EXPECTED_STAGES


def test_an_unknown_case_returns_the_error_envelope(app_client):
    response = app_client.get("/api/longitudinal/demo-cases/NOPE-99")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DEMO_CASE_NOT_FOUND"


@pytest.mark.parametrize("stage_id", EXPECTED_STAGES)
def test_each_stage_can_be_fetched_individually(app_client, stage_id):
    stage = app_client.get(
        f"/api/longitudinal/demo-cases/{CASE_ID}/stage/{stage_id}"
    ).json()
    assert stage["stage_id"] == stage_id
    assert stage["provenance"]["is_true_followup"] is False
    assert stage["provenance"]["source_dataset"] == "SPIDER"


def test_an_invalid_stage_returns_the_valid_ones(app_client):
    response = app_client.get(
        f"/api/longitudinal/demo-cases/{CASE_ID}/stage/month_99"
    )
    assert response.status_code == 404
    body = response.json()["error"]
    assert body["code"] == "DEMO_STAGE_NOT_FOUND"
    assert body["details"]["valid_stages"] == EXPECTED_STAGES


def test_the_longitudinal_endpoints_do_not_disturb_the_existing_api(app_client):
    """Adding a router must not change what was already there."""
    assert app_client.get("/api/health").status_code == 200
    assert app_client.get("/api/analysis").status_code == 200


def test_no_traceback_leaks_from_a_bad_case_request(app_client):
    response = app_client.get("/api/longitudinal/demo-cases/%2E%2E%2Fsecrets")
    assert response.status_code in (404, 422)
    assert "Traceback" not in response.text
