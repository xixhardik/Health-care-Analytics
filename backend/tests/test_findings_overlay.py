"""Sprint 7.1 - the finding-aware disc overlay.

Two layers are tested separately, on purpose:

* the **rule** (`services/findings_overlay.py`) decides which discs may be
  marked. It is pure, so it is tested exhaustively against the finding shapes the
  API actually returns, including every way a finding can fail to qualify.
* the **renderer** (`services/imaging.py`) draws what it is told. Its tests prove
  the red is confined to the disc instance region and that nothing it is handed
  is mutated.

Then a handful of endpoint tests prove the wiring, the parameter validation and
the header contract.

The overarching claim these tests defend: the red overlay is the *validated disc
segmentation* of a disc carrying a supported positive finding. It is not a
pathology mask, and no pixel inside it was classified.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest

from backend.app.services.findings_overlay import (
    OVERLAY_ELIGIBLE_FINDINGS,
    disc_finding_labels,
    overlay_discs,
    overlay_summary,
    qualifies,
)
from backend.app.services.imaging import (
    FINDING_COLOUR,
    disc_region,
    draw_finding_overlay,
    render_slice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def positive(name: str = "narrowing", **overrides) -> dict:
    """A supported, positive, model-estimated binary finding, as served."""
    finding = {
        "name": name,
        "label": "Disc narrowing",
        "value": True,
        "probability": 0.997,
        "strength": "strong",
        "validated_metric": "test_pr_auc",
        "validated_value": 0.8779,
        "prevalence_baseline": 0.323,
        "source": "model_prediction",
        "unavailable_reason": None,
    }
    finding.update(overrides)
    return finding


def synthetic_slice():
    """A flat grey slice with two disc regions, discs 1 and 2."""
    height, width = 24, 32
    image = np.full((height, width), 120, dtype=np.uint8)
    semantic = np.zeros((height, width), dtype=np.uint8)
    instance = np.zeros((height, width), dtype=np.uint8)
    # Instance labels encode disc N as 10 + N.
    semantic[4:8, 6:14] = 2
    instance[4:8, 6:14] = 11
    semantic[14:18, 6:14] = 2
    instance[14:18, 6:14] = 12
    return image, semantic, instance


# ---------------------------------------------------------------------------
# 1. The rule: what may turn a disc red
# ---------------------------------------------------------------------------


def test_a_supported_positive_finding_qualifies():
    assert qualifies(positive()) is True


@pytest.mark.parametrize("name", OVERLAY_ELIGIBLE_FINDINGS)
def test_every_eligible_binary_target_can_qualify(name):
    assert qualifies(positive(name)) is True


def test_a_negative_finding_does_not_qualify():
    """`False` is a real negative result, not something to colour."""
    assert qualifies(positive(value=False)) is False


def test_an_unavailable_finding_never_qualifies():
    """The absence of a model is not a positive finding."""
    unavailable = positive(
        value=None,
        probability=None,
        source="unsupported",
        unavailable_reason="Not served. Too few positive cases to validate.",
    )
    assert qualifies(unavailable) is False
    # Even if a reason were attached to an otherwise positive-looking payload.
    assert qualifies(positive(unavailable_reason="anything at all")) is False


def test_a_null_or_missing_value_never_qualifies():
    assert qualifies(positive(value=None)) is False
    finding = positive()
    del finding["value"]
    assert qualifies(finding) is False
    assert qualifies({}) is False


def test_an_unsupported_source_never_qualifies():
    assert qualifies(positive(source="unsupported")) is False
    assert qualifies(positive(source="segmentation_derived")) is False


def test_the_unsupported_targets_can_never_qualify():
    """Spondylolisthesis and the nominal Modic type are not served at all."""
    for name in ("spondylolisthesis", "modic_type"):
        assert name not in OVERLAY_ELIGIBLE_FINDINGS
        assert qualifies(positive(name)) is False


def test_the_pfirrmann_grade_never_qualifies():
    """An ordinal grade is not a positive finding.

    Colouring a disc by grade would mean inventing a severity threshold that was
    never validated, so the grade is excluded by name *and* by the boolean check.
    """
    assert "pfirrmann_grade" not in OVERLAY_ELIGIBLE_FINDINGS
    for grade in (1, 2, 3, 4, 5):
        assert qualifies(
            {
                "name": "pfirrmann_grade",
                "value": grade,
                "source": "model_prediction",
                "unavailable_reason": None,
            }
        ) is False


def test_a_truthy_number_is_not_accepted_as_a_positive_finding():
    """`1` is not `True`. A numeric value must never pass the boolean gate."""
    assert qualifies(positive(value=1)) is False
    assert qualifies(positive(value=0.997)) is False
    assert qualifies(positive(value="true")) is False


def test_disc_labels_list_only_the_qualifying_findings():
    disc = {
        "index": 1,
        "slices_present": 11,
        "findings": [
            positive("narrowing", label="Disc narrowing"),
            positive("bulging", label="Disc bulging", value=False),
            positive(
                "herniation", label="Disc herniation",
                value=None, source="unsupported",
                unavailable_reason="not served",
            ),
            positive("any_modic", label="Modic-type change present"),
        ],
    }
    assert disc_finding_labels(disc) == [
        "Disc narrowing",
        "Modic-type change present",
    ]


# ---------------------------------------------------------------------------
# 2. The rule applied to a whole result
# ---------------------------------------------------------------------------


def test_only_discs_with_a_positive_supported_finding_are_marked():
    result = {
        "discs": [
            {"index": 1, "slices_present": 11, "findings": [positive()]},
            {"index": 2, "slices_present": 11,
             "findings": [positive(value=False)]},
            {"index": 3, "slices_present": 11, "findings": []},
        ]
    }
    assert sorted(overlay_discs(result)) == [1]


def test_a_disc_without_a_segmentation_region_is_not_marked():
    """A finding with nothing segmented behind it has nothing honest to draw."""
    result = {
        "discs": [
            {"index": 1, "slices_present": 0,
             "measurements": {"area_mm2": None}, "findings": [positive()]},
            {"index": 2, "slices_present": 0,
             "measurements": {"area_mm2": 0.0}, "findings": [positive()]},
            {"index": 3, "slices_present": 0,
             "measurements": {"area_mm2": 231.0}, "findings": [positive()]},
        ]
    }
    # Only disc 3 has a region, by way of a measured area.
    assert sorted(overlay_discs(result)) == [3]


def test_an_empty_result_marks_nothing():
    assert overlay_discs({}) == {}
    assert overlay_discs({"discs": []}) == {}


def test_overlay_summary_states_the_rule_and_the_caveat():
    result = {"discs": [{"index": 1, "slices_present": 5,
                         "findings": [positive()]}]}
    summary = overlay_summary(result)

    assert summary["disc_indices"] == [1]
    assert summary["discs"] == [{"index": 1, "findings": ["Disc narrowing"]}]
    assert "pfirrmann_grade" not in summary["eligible_findings"]
    # The wording the UI renders verbatim must refuse the tissue-level reading.
    note = summary["note"].lower()
    assert "not a diagnosis" in note
    assert "damaged tissue" in note
    assert "pixel-level pathology model" in note
    assert "disc regions associated with" in note


# ---------------------------------------------------------------------------
# 3. The renderer: the red is confined to the disc region
# ---------------------------------------------------------------------------


def test_the_overlay_is_confined_to_the_disc_instance_region():
    image, semantic, instance = synthetic_slice()
    plain = render_slice(image, semantic, mode="original", instance=instance)
    tinted = render_slice(
        image, semantic, mode="original", instance=instance,
        finding_discs=(1,), finding_opacity=0.5,
    )

    changed = np.any(plain != tinted, axis=2)
    region = disc_region(instance, 1)
    # Every changed pixel is a disc-1 pixel, and every disc-1 pixel changed.
    assert np.array_equal(changed, region)


def test_an_unmarked_disc_is_left_untouched():
    image, semantic, instance = synthetic_slice()
    plain = render_slice(image, semantic, mode="original", instance=instance)
    tinted = render_slice(
        image, semantic, mode="original", instance=instance, finding_discs=(1,),
    )

    other = disc_region(instance, 2)
    assert np.array_equal(plain[other], tinted[other])


def test_a_disc_absent_from_this_slice_draws_nothing():
    """No fallback, no approximation: if the disc is not here, nothing is drawn."""
    image, semantic, instance = synthetic_slice()
    plain = render_slice(image, semantic, mode="original", instance=instance)
    tinted = render_slice(
        image, semantic, mode="original", instance=instance, finding_discs=(7,),
    )
    assert np.array_equal(plain, tinted)


def test_no_finding_discs_reproduces_the_existing_rendering_exactly():
    image, semantic, instance = synthetic_slice()
    for mode in ("original", "mask", "overlay"):
        before = render_slice(image, semantic, mode=mode, instance=instance)
        after = render_slice(
            image, semantic, mode=mode, instance=instance, finding_discs=(),
        )
        assert np.array_equal(before, after), mode


def test_the_overlay_works_in_every_render_mode():
    image, semantic, instance = synthetic_slice()
    region = disc_region(instance, 1)
    for mode in ("original", "mask", "overlay"):
        plain = render_slice(image, semantic, mode=mode, instance=instance)
        tinted = render_slice(
            image, semantic, mode=mode, instance=instance,
            finding_discs=(1,), finding_opacity=0.6,
        )
        assert not np.array_equal(plain[region], tinted[region]), mode
        outside = ~region
        assert np.array_equal(plain[outside], tinted[outside]), mode


def test_the_selected_disc_is_rendered_more_strongly():
    """Emphasis changes opacity only - both discs use the same mask."""
    image, semantic, instance = synthetic_slice()
    rendered = render_slice(
        image, semantic, mode="original", instance=instance,
        finding_discs=(1, 2), finding_opacity=0.4, highlight_disc=2,
    )
    # Sample an interior pixel of each disc.
    plain_disc, selected_disc = tuple(rendered[5, 9]), tuple(rendered[15, 9])
    assert selected_disc != plain_disc
    # Closer to pure red means more strongly tinted.
    assert selected_disc[0] > plain_disc[0]
    assert selected_disc[1] < plain_disc[1]


def test_different_discs_produce_different_renderings():
    image, semantic, instance = synthetic_slice()
    first = render_slice(image, semantic, mode="original", instance=instance,
                         finding_discs=(1,))
    second = render_slice(image, semantic, mode="original", instance=instance,
                          finding_discs=(2,))
    both = render_slice(image, semantic, mode="original", instance=instance,
                        finding_discs=(1, 2))
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, both)
    assert not np.array_equal(second, both)


def test_zero_opacity_leaves_the_image_unchanged():
    image, semantic, instance = synthetic_slice()
    plain = render_slice(image, semantic, mode="original", instance=instance)
    tinted = render_slice(image, semantic, mode="original", instance=instance,
                          finding_discs=(1,), finding_opacity=0.0)
    assert np.array_equal(plain, tinted)


def test_full_opacity_paints_the_disc_region_solid_red():
    image, semantic, instance = synthetic_slice()
    tinted = render_slice(image, semantic, mode="original", instance=instance,
                          finding_discs=(1,), finding_opacity=1.0)
    region = disc_region(instance, 1)
    assert (tinted[region] == np.asarray(FINDING_COLOUR)).all()


def test_the_renderer_never_mutates_the_arrays_it_is_given():
    """The stored MRI, semantic and Sprint 5 instance maps are read-only here."""
    image, semantic, instance = synthetic_slice()
    image_before = image.copy()
    semantic_before = semantic.copy()
    instance_before = instance.copy()

    render_slice(
        image, semantic, mode="overlay", instance=instance,
        finding_discs=(1, 2), finding_opacity=0.8, highlight_disc=1,
    )

    assert np.array_equal(image, image_before)
    assert np.array_equal(semantic, semantic_before)
    assert np.array_equal(instance, instance_before)


def test_draw_finding_overlay_returns_a_new_array():
    image, _, instance = synthetic_slice()
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    rgb_before = rgb.copy()
    out = draw_finding_overlay(rgb, instance, (1,), opacity=0.5)
    assert out is not rgb
    assert np.array_equal(rgb, rgb_before)


def test_the_overlay_is_deterministic():
    image, semantic, instance = synthetic_slice()
    kwargs = dict(mode="overlay", instance=instance, finding_discs=(1, 2),
                  finding_opacity=0.55, highlight_disc=1)
    first = render_slice(image, semantic, **kwargs)
    second = render_slice(image, semantic, **kwargs)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# 4. The endpoint
# ---------------------------------------------------------------------------


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
    """One completed analysis from the bundled sample study."""
    response = app_client.post("/api/analysis/sample")
    if response.status_code == 404:
        pytest.skip("Sample volume not extracted on this machine")
    assert response.status_code == 201, response.text
    analysis_id = response.json()["analysis_id"]
    app_client.post(f"/api/analysis/{analysis_id}/run")
    state = _wait(app_client, analysis_id)
    assert state["status"] == "completed", state
    result = app_client.get(f"/api/analysis/{analysis_id}/result").json()
    return {"id": analysis_id, "result": result}


def _slice(app_client, analysis_id, **params):
    return app_client.get(
        f"/api/analysis/{analysis_id}/slice/{params.pop('at')}", params=params
    )


def test_the_result_serves_the_overlay_decision(analysis):
    overlay = analysis["result"]["finding_overlay"]
    assert isinstance(overlay["disc_indices"], list)
    assert "pfirrmann_grade" not in overlay["eligible_findings"]
    assert "damaged tissue" in overlay["note"].lower()
    # The served decision agrees with the rule applied to the same payload.
    assert overlay["disc_indices"] == sorted(overlay_discs(analysis["result"]))


def test_the_served_decision_matches_the_positive_findings(analysis):
    """Cross-check the API against the discs' own finding lists."""
    result = analysis["result"]
    expected = {
        disc["index"]
        for disc in result["discs"]
        if any(qualifies(f) for f in disc["findings"])
    }
    assert set(result["finding_overlay"]["disc_indices"]) <= expected


def test_overlay_off_is_byte_identical_to_the_previous_behaviour(
    app_client, analysis
):
    """The default rendering must not shift because the feature was added."""
    analysis_id, middle = analysis["id"], 12
    default = _slice(app_client, analysis_id, at=middle, mode="overlay")
    explicit = _slice(app_client, analysis_id, at=middle, mode="overlay",
                      highlight="disc")
    assert default.status_code == 200
    assert default.content == explicit.content


def test_overlay_on_changes_the_image(app_client, analysis):
    marked = analysis["result"]["finding_overlay"]["disc_indices"]
    if not marked:
        pytest.skip("This study has no positive supported finding to mark")
    analysis_id, middle = analysis["id"], 12
    off = _slice(app_client, analysis_id, at=middle, mode="overlay")
    on = _slice(app_client, analysis_id, at=middle, mode="overlay",
                highlight="findings")
    assert on.status_code == 200
    assert on.headers["content-type"] == "image/png"
    assert on.content != off.content


def test_the_response_reports_which_discs_it_marked(app_client, analysis):
    analysis_id = analysis["id"]
    marked = analysis["result"]["finding_overlay"]["disc_indices"]

    on = _slice(app_client, analysis_id, at=12, mode="overlay",
                highlight="findings")
    header = [int(v) for v in on.headers["X-Finding-Discs"].split(",") if v]
    assert header == marked

    off = _slice(app_client, analysis_id, at=12, mode="overlay")
    assert off.headers["X-Finding-Discs"] == ""


def test_the_rendered_red_stays_inside_the_disc_segmentation(app_client, analysis):
    """Decode the PNG and check containment against the stored instance map."""
    from PIL import Image

    marked = analysis["result"]["finding_overlay"]["disc_indices"]
    if not marked:
        pytest.skip("This study has no positive supported finding to mark")

    analysis_id = analysis["id"]
    arrays = app_client.app.state.store.read_arrays(analysis_id)
    # A slice where at least one marked disc is actually present.
    at = next(
        (
            index
            for index in range(len(arrays["instance"]))
            if any(disc_region(arrays["instance"][index], d).any() for d in marked)
        ),
        None,
    )
    assert at is not None, "no slice contains a marked disc"

    off = _slice(app_client, analysis_id, at=at, mode="original")
    on = _slice(app_client, analysis_id, at=at, mode="original",
                highlight="findings")
    before = np.asarray(Image.open(io.BytesIO(off.content)).convert("RGB"))
    after = np.asarray(Image.open(io.BytesIO(on.content)).convert("RGB"))

    changed = np.any(before != after, axis=2)
    allowed = np.zeros_like(changed)
    for disc_index in marked:
        allowed |= disc_region(arrays["instance"][at], disc_index)

    assert changed.any(), "the overlay changed nothing on a slice that has a disc"
    # Nothing outside the marked discs' own segmentation was touched.
    assert not (changed & ~allowed).any()


def test_the_underlying_mri_is_unchanged_beneath_the_overlay(app_client, analysis):
    """Turning the overlay off returns exactly the original pixels again."""
    analysis_id = analysis["id"]
    first = _slice(app_client, analysis_id, at=12, mode="original")
    _slice(app_client, analysis_id, at=12, mode="original", highlight="findings")
    again = _slice(app_client, analysis_id, at=12, mode="original")
    assert first.content == again.content


def test_the_stored_arrays_are_not_modified_by_rendering(app_client, analysis):
    analysis_id = analysis["id"]
    store = app_client.app.state.store
    before = store.read_arrays(analysis_id)
    semantic_before = before["semantic"].copy()
    instance_before = before["instance"].copy()
    images_before = before["images"].copy()

    for highlight in ("none", "disc", "findings"):
        _slice(app_client, analysis_id, at=12, mode="overlay",
               highlight=highlight, highlight_disc=1)

    after = store.read_arrays(analysis_id)
    assert np.array_equal(after["semantic"], semantic_before)
    assert np.array_equal(after["instance"], instance_before)
    assert np.array_equal(after["images"], images_before)


def test_an_invalid_highlight_mode_is_refused_with_the_error_envelope(
    app_client, analysis
):
    response = _slice(app_client, analysis["id"], at=12, highlight="tumour")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    # No image is produced for a request the server did not understand.
    assert response.headers["content-type"].startswith("application/json")


def test_an_out_of_range_finding_opacity_is_refused(app_client, analysis):
    for value in (-0.1, 1.5):
        response = _slice(app_client, analysis["id"], at=12,
                          highlight="findings", finding_opacity=value)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_highlight_none_suppresses_the_selected_disc_marker(app_client, analysis):
    analysis_id = analysis["id"]
    bare = _slice(app_client, analysis_id, at=12, mode="overlay",
                  highlight="none")
    suppressed = _slice(app_client, analysis_id, at=12, mode="overlay",
                        highlight="none", highlight_disc=1)
    assert bare.content == suppressed.content


def test_disc_selection_still_marks_the_selected_disc(app_client, analysis):
    """The Sprint 7 bracket behaviour is unchanged by the new parameter."""
    analysis_id = analysis["id"]
    plain = _slice(app_client, analysis_id, at=12, mode="overlay")
    marked = _slice(app_client, analysis_id, at=12, mode="overlay",
                    highlight_disc=1)
    assert marked.status_code == 200
    assert marked.content != plain.content


def test_selection_and_the_findings_overlay_compose(app_client, analysis):
    """Selecting a disc while the overlay is on renders differently again."""
    marked = analysis["result"]["finding_overlay"]["disc_indices"]
    if not marked:
        pytest.skip("This study has no positive supported finding to mark")
    analysis_id = analysis["id"]
    overlay_only = _slice(app_client, analysis_id, at=12, mode="overlay",
                          highlight="findings")
    with_selection = _slice(app_client, analysis_id, at=12, mode="overlay",
                            highlight="findings", highlight_disc=marked[0])
    assert with_selection.content != overlay_only.content


def test_class_toggles_still_filter_server_side_with_the_overlay_on(
    app_client, analysis
):
    analysis_id = analysis["id"]
    all_classes = _slice(app_client, analysis_id, at=12, mode="mask",
                         highlight="findings", classes="1,2,3")
    discs_only = _slice(app_client, analysis_id, at=12, mode="mask",
                        highlight="findings", classes="2")
    assert all_classes.status_code == 200
    assert discs_only.content != all_classes.content


def test_the_overlay_renders_across_slices_and_modes(app_client, analysis):
    analysis_id = analysis["id"]
    slice_count = analysis["result"]["study"]["slice_count"]
    for at in (0, slice_count // 2, slice_count - 1):
        for mode in ("original", "mask", "overlay"):
            response = _slice(app_client, analysis_id, at=at, mode=mode,
                              highlight="findings")
            assert response.status_code == 200, (at, mode)
            assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
