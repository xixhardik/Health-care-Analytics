"""Health endpoint: the model must be loaded once, at startup, and reported."""

from __future__ import annotations


def test_health_reports_ok_and_loaded_model(app_client):
    response = app_client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["api_version"]

    model = body["segmentation_model"]
    assert model["loaded"] is True
    # The served model must be the Sprint 3 checkpoint, not Sprint 4, which
    # measured worse on every class and on disc indexing.
    assert model["source_sprint"] == "Sprint 3 Coverage"
    assert model["parameters"] == 1_963_860
    assert "base=16" in model["architecture"]
    assert model["device"] in ("cpu", "cuda")
    assert model["selection"] == "best validation foreground Dice"
    assert model["validated_macro_foreground_dice"] == 0.90001


def test_health_declares_supported_and_unsupported_findings(app_client):
    body = app_client.get("/api/health").json()
    findings = body["finding_models"]

    assert findings["loaded"] is True
    assert "pfirrmann_grade" in findings["supported_targets"]
    for target in ("narrowing", "bulging", "any_modic", "upper_endplate",
                   "lower_endplate", "herniation"):
        assert target in findings["supported_targets"]

    # Unsupported targets must be declared with a reason, not silently missing.
    unsupported = findings["unsupported_targets"]
    assert "spondylolisthesis" in unsupported
    assert "modic_type" in unsupported
    for reason in unsupported.values():
        assert len(reason) > 20


def test_health_uses_the_validated_postprocessing_method(app_client):
    body = app_client.get("/api/health").json()
    assert body["postprocessing_method"] == "Sprint 5 5A+5D"


def test_health_carries_a_research_disclaimer(app_client):
    disclaimer = app_client.get("/api/health").json()["disclaimer"].lower()
    assert "research" in disclaimer
    assert "not" in disclaimer and "diagnosis" in disclaimer
