"""The single source of truth for pipeline identity and validated performance.

Two rules this module exists to enforce:

1. **One pipeline version.** The served combination of preprocessing,
   segmentation and post-processing is named once, here, and echoed by the API.
   Nothing else may invent a version string.
2. **No metrics hard-coded in the UI.** Every research figure the interface shows
   is served from here through ``/api/health`` and the analysis result, so the
   frontend cannot drift from what was actually measured.

All figures below were measured on the 33-patient held-out test split during
Sprints 2-5. They describe the pipeline, not any uploaded study, and every
consumer is expected to label them as research evaluation results.
"""

from __future__ import annotations

#: Name of the frozen served pipeline. Bump only when the served combination of
#: preprocessing, segmentation or post-processing actually changes.
PIPELINE_VERSION = "SPIDER-Lumbar-v1"

PIPELINE = {
    "version": PIPELINE_VERSION,
    "preprocessing": "Sprint 1",
    "segmentation": "Sprint 3",
    "postprocessing": "Sprint 5",
    "segmentation_detail": "16-channel U-Net, depth 4, bilinear, 1,963,860 parameters",
    "postprocessing_detail": "Series-level disc ordering (5A) + vertebral-body separation (5D)",
    "preprocessing_detail": (
        "Resample to 1.0 mm/px, centre crop or pad to 352x256, per-volume "
        "foreground percentile normalisation, median denoise, CLAHE"
    ),
    "excluded": {
        "Sprint 4": (
            "32-channel capacity experiment did not improve the primary "
            "segmentation metrics and was therefore not selected."
        )
    },
    "dataset": "SPIDER lumbar spine MRI (research dataset)",
}

#: Published Sprint 3 segmentation Dice, per class and macro.
SEGMENTATION_DICE = {
    "vertebra": 0.91663,
    "intervertebral_disc": 0.87827,
    "spinal_canal": 0.90514,
    "macro_foreground": 0.90001,
}

#: The headline research evaluation figures, grouped by which question they answer.
#: Segmentation and post-processing metrics are kept apart because they measure
#: different things and must not be presented as one score.
VALIDATED_METRICS = {
    "segmentation_macro_foreground_dice": SEGMENTATION_DICE["macro_foreground"],
    "segmentation_vertebra_dice": SEGMENTATION_DICE["vertebra"],
    "segmentation_disc_dice": SEGMENTATION_DICE["intervertebral_disc"],
    "segmentation_canal_dice": SEGMENTATION_DICE["spinal_canal"],
    "disc_indexing_percent": 93.84,
    "disc_indexing_baseline_percent": 84.08,
    "disc_height_mae_mm": 0.6492,
    "disc_area_mae_mm2": 21.4831,
    "intensity_ratio_pearson_r": 0.9726,
    "pfirrmann_qwk_end_to_end": 0.6543,
    "test_patients": 33,
    "note": (
        "Research evaluation results measured on a 33-patient held-out test "
        "split. Segmentation figures are Sprint 3; disc indexing, measurement "
        "and Pfirrmann figures are Sprint 5 post-processing applied on top of "
        "it. They describe the pipeline, not this study."
    ),
}

#: Rows for the methodology page's metric table. ``kind`` lets the UI keep
#: segmentation and post-processing visually separate without deciding which is
#: which itself.
METRIC_TABLE = [
    {"label": "Macro foreground Dice", "value": "0.90001",
     "kind": "segmentation", "sprint": "Sprint 3"},
    {"label": "Vertebra Dice", "value": "0.91663",
     "kind": "segmentation", "sprint": "Sprint 3"},
    {"label": "Intervertebral disc Dice", "value": "0.87827",
     "kind": "segmentation", "sprint": "Sprint 3"},
    {"label": "Spinal canal Dice", "value": "0.90514",
     "kind": "segmentation", "sprint": "Sprint 3"},
    {"label": "Disc indexing, per-slice ordering", "value": "84.08%",
     "kind": "postprocessing", "sprint": "Sprint 3"},
    {"label": "Disc indexing, series-level ordering", "value": "93.84%",
     "kind": "postprocessing", "sprint": "Sprint 5"},
    {"label": "Disc height mean absolute error", "value": "0.6492 mm",
     "kind": "postprocessing", "sprint": "Sprint 5"},
    {"label": "Disc area mean absolute error", "value": "21.4831 mm²",
     "kind": "postprocessing", "sprint": "Sprint 5"},
    {"label": "Disc signal ratio correlation", "value": "r = 0.9726",
     "kind": "postprocessing", "sprint": "Sprint 5"},
    {"label": "End-to-end Pfirrmann agreement", "value": "QWK 0.6543",
     "kind": "postprocessing", "sprint": "Sprint 5"},
]

#: The experimental record, including the experiment that was rejected. Negative
#: results are served alongside positive ones deliberately.
RESEARCH_PROGRESS = [
    {
        "sprint": "Sprint 2",
        "title": "Baseline U-Net",
        "outcome": "selected",
        "summary": (
            "First working 16-channel U-Net plus the disc-level measurement and "
            "finding estimators. Established the baseline the later experiments "
            "were measured against."
        ),
        "headline": "Macro foreground Dice 0.8979, disc indexing 82.6%",
    },
    {
        "sprint": "Sprint 3",
        "title": "100% training-data coverage",
        "outcome": "selected",
        "summary": (
            "An audit found the sampler had only ever shown the model 28% of the "
            "training slices. A rotating sampler raised that to 100%. Validation "
            "Dice did not move, so coverage was not the binding constraint - but "
            "this run produced the best segmentation and is the one served."
        ),
        "headline": "Macro foreground Dice 0.90001, disc indexing 84.08%",
    },
    {
        "sprint": "Sprint 4",
        "title": "32-channel capacity experiment",
        "outcome": "rejected",
        "summary": (
            "32-channel capacity experiment did not improve the primary "
            "segmentation metrics and was therefore not selected. Every "
            "foreground Dice fell and disc indexing dropped to 82.70%, at 2.8x "
            "the training cost. Reported rather than discarded, because it is "
            "what rules out further width scaling."
        ),
        "headline": "Macro foreground Dice 0.89393, disc indexing 82.70% — worse",
    },
    {
        "sprint": "Sprint 5",
        "title": "Series-level post-processing",
        "outcome": "selected",
        "summary": (
            "Disc identity is assigned once per series from row-aligned tracks "
            "instead of independently per slice, and vertebral bodies are "
            "separated from posterior elements. Segmentation output is byte "
            "identical; only disc identity changed."
        ),
        "headline": "Disc indexing 84.08% → 93.84%, height MAE 0.6492 mm",
    },
]

#: Shown wherever a result is displayed.
DISCLAIMER = (
    "Research and educational prototype. Results are model-derived research "
    "estimates and require review by a qualified radiologist. This system does "
    "not provide a clinical diagnosis and is not a medical device."
)

RESEARCH_NOTICE = (
    "Research / educational prototype — results require expert radiological "
    "review."
)

#: How each reported quantity was obtained. The UI renders these verbatim so a
#: measurement can never be mistaken for a model estimate, or a missing value for
#: a negative finding.
PROVENANCE = {
    "segmentation_derived": "Segmentation-derived",
    "model_prediction": "Model-estimated",
    "unsupported": "Not modelled",
}

#: Evidence wording per validated strength category. Deliberately qualitative:
#: converting a model probability into a numeric medical confidence would invent
#: a quantity that was never validated.
EVIDENCE_LABELS = {
    "strong": "Strong evidence",
    "moderate": "Moderate evidence",
    "modest": "Limited evidence",
    "weak": "Weak evidence",
}
