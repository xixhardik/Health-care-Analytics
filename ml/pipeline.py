"""The frozen inference pipeline, as five composable steps.

Every step delegates to the validated Sprint 1-5 code in ``src/``. Nothing is
retrained and no parameter is re-chosen: the Sprint 5 post-processing
configuration is read from the published
``outputs/reports/sprint5_indexing/test_results.json`` so the served pipeline is
provably the one that was measured at 93.84% disc indexing.

Validated performance of this exact chain, on the 33-patient held-out test split:

======================================  =========
segmentation macro foreground Dice       0.90001
vertebra / disc / canal Dice             0.91663 / 0.87827 / 0.90514
disc indexing (corrected taxonomy)       93.84%
disc height MAE                          0.6492 mm
disc area MAE                            21.4831 mm^2
disc intensity ratio Pearson r           0.9726
end-to-end Pfirrmann QWK                 0.6543
======================================  =========

The segmentation figures come from Sprint 3 and the indexing/measurement figures
from Sprint 5 post-processing on top of it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROGRESS_CALLBACK = Callable[[int, str], None]


@dataclass(frozen=True)
class PipelineConfig:
    """Serving configuration. All paths are relative to the project root."""

    project_root: Path
    checkpoint: Path
    finding_model_dir: Path
    sprint5_params_file: Path
    batch_size: int = 4
    min_area_px: int = 20
    #: Slices whose predicted foreground is below this are treated as carrying no
    #: usable anatomy. Training dropped slices with under 10 mm2 of annotation, so
    #: the model has never seen purely lateral slices; this is the inference-time
    #: analogue, applied only to *reporting*, never to the segmentation itself.
    min_foreground_px: int = 50

    @property
    def mm_per_pixel(self) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# 1. Preprocessing (Sprint 1)
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedStudy:
    """The output of Sprint 1 preprocessing for one uploaded volume."""

    slice_ids: list[str]
    slice_indices: list[int]
    images: np.ndarray           # (n, 352, 256) float32 in [0, 1]
    series_id: str
    source_slice_count: int
    config: dict
    native_in_plane_spacing: tuple[float, float]
    native_in_plane_shape: tuple[int, int]


def preprocess_volume(
    path: Path, *, series_id: str, progress: PROGRESS_CALLBACK | None = None
) -> PreprocessedStudy:
    """Run Sprint 1 preprocessing over every sagittal slice of one volume.

    This reproduces the image branch of ``src.preprocessing.dataset``
    exactly - resample to 1.0 mm/px, centre crop or pad to 352x256, normalise on
    per-volume foreground percentiles, median denoise, CLAHE - but without the
    batch driver's mask dependency, because an uploaded study has no annotation.

    Every slice is kept. Training discarded slices with under 10 mm2 of
    annotation, but that test needs a mask; deciding which slices carry anatomy
    is left to the prediction itself.
    """
    from src.preprocessing.transforms import PreprocessConfig, intensity_statistics, preprocess_image
    from src.preprocessing.volume_io import load_image

    volume = load_image(path)
    config = PreprocessConfig()
    stats = intensity_statistics(volume.array, percentiles=config.clip_percentiles)
    spacing = volume.in_plane_spacing

    images: list[np.ndarray] = []
    slice_ids: list[str] = []
    slice_indices: list[int] = []
    total = volume.n_sagittal_slices
    for index in range(total):
        plane = volume.sagittal_slice(index)
        processed = preprocess_image(plane, spacing, config, volume_stats=stats)
        images.append(processed)
        slice_ids.append(f"{series_id}_s{index:03d}")
        slice_indices.append(index)
        if progress and total > 1 and index % max(1, total // 8) == 0:
            progress(20 + int(15 * index / total), "Preprocessing")

    return PreprocessedStudy(
        slice_ids=slice_ids,
        slice_indices=slice_indices,
        images=np.stack(images).astype(np.float32),
        series_id=series_id,
        source_slice_count=total,
        config=config.to_dict(),
        native_in_plane_spacing=(float(spacing[0]), float(spacing[1])),
        native_in_plane_shape=tuple(int(v) for v in volume.in_plane_shape),
    )


# ---------------------------------------------------------------------------
# 2. Segmentation (Sprint 3)
# ---------------------------------------------------------------------------


def segment_slices(
    model,
    images: np.ndarray,
    *,
    batch_size: int = 4,
    device: str = "cpu",
    progress: PROGRESS_CALLBACK | None = None,
) -> np.ndarray:
    """Run the frozen Sprint 3 U-Net over preprocessed slices.

    Returns the semantic label map per slice: 0 background, 1 vertebra,
    2 intervertebral disc, 3 spinal canal. The model is supplied already loaded -
    this function never reads a checkpoint, because loading per request is
    exactly the cost the model service exists to avoid.
    """
    import torch

    model.eval()
    outputs: list[np.ndarray] = []
    total = len(images)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            chunk = images[start : start + batch_size]
            tensor = torch.from_numpy(chunk).unsqueeze(1).to(device)
            tensor = tensor.to(memory_format=torch.channels_last)
            logits = model(tensor)
            predicted = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            outputs.append(predicted)
            if progress:
                done = min(total, start + batch_size)
                progress(40 + int(20 * done / total), "Segmentation")
    return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# 3. Disc indexing (Sprint 5)
# ---------------------------------------------------------------------------


def load_sprint5_params(config: PipelineConfig):
    """Load the published Sprint 5 post-processing configuration.

    Read from the artefact rather than hardcoded, so the served pipeline cannot
    silently drift from the configuration that was validated.
    """
    from src.analysis.disc_postprocess import PostProcessParams

    if config.sprint5_params_file.exists():
        with config.sprint5_params_file.open(encoding="utf-8") as handle:
            published = json.load(handle)
        params = published.get("selected_params")
        if params:
            return PostProcessParams(**params), published.get("selected_method", "5A+5D")
    # Falling back to the dataclass defaults would change behaviour silently, so
    # this is an error rather than a quiet default.
    raise FileNotFoundError(
        f"Sprint 5 parameters not found at {config.sprint5_params_file}. "
        f"The served pipeline must use the published configuration."
    )


@dataclass
class IndexedStudy:
    """Sprint 5 output: corrected instance maps plus series-level diagnostics."""

    instance_maps: np.ndarray           # (n, 352, 256) uint8
    n_tracks: int
    tracks: list[dict]
    informative: dict[str, bool]
    method: str
    params: dict
    n_components_rejected: int
    n_components_unassigned: int


def index_discs(
    semantic: np.ndarray,
    slice_ids: list[str],
    *,
    config: PipelineConfig,
    progress: PROGRESS_CALLBACK | None = None,
) -> IndexedStudy:
    """Assign disc and vertebral-body identity with the Sprint 5 method.

    Series-level ordering (5A) plus vertebral-body separation (5D). The whole
    uploaded study is one series, which is exactly the unit the method was
    designed and validated on.
    """
    from src.analysis import disc_postprocess as pp

    params, method = load_sprint5_params(config)
    if progress:
        progress(62, "Disc indexing")

    semantic_by_slice = {
        slice_id: semantic[position] for position, slice_id in enumerate(slice_ids)
    }
    result = pp.process_series(semantic_by_slice, params=params)
    if progress:
        progress(72, "Disc indexing")

    instance_maps = np.stack(
        [result["instance_maps"][slice_id] for slice_id in slice_ids]
    ).astype(np.uint8)

    return IndexedStudy(
        instance_maps=instance_maps,
        n_tracks=int(result["n_tracks"]),
        tracks=list(result["tracks"]),
        informative=dict(result["informative"]),
        method=method,
        params=params.to_dict(),
        n_components_rejected=int(result["n_components_rejected"]),
        n_components_unassigned=int(result["n_components_unassigned"]),
    )


# ---------------------------------------------------------------------------
# 4. Measurements
# ---------------------------------------------------------------------------


def measure_discs(
    images: np.ndarray,
    instance_maps: np.ndarray,
    slice_ids: list[str],
    slice_indices: list[int],
    *,
    series_id: str,
    config: PipelineConfig,
    progress: PROGRESS_CALLBACK | None = None,
):
    """Measure every identified disc, then aggregate to one row per disc.

    Uses the same measurement code as the research pipeline, so the numbers are
    comparable with the published MAEs. Aggregation picks the largest-area slice
    as each disc's representative and adds the series-relative features the
    finding estimators expect.
    """
    import pandas as pd
    from src.analysis.disc_features import (
        add_relative_height_features,
        aggregate_disc_records,
        extract_disc_features_from_slice,
    )

    records: list[dict] = []
    total = len(slice_ids)
    for position, slice_id in enumerate(slice_ids):
        rows = extract_disc_features_from_slice(
            images[position].astype(np.float32),
            instance_maps[position],
            # An uploaded study has no patient identity in the pipeline. A fixed
            # sentinel keeps the schema intact without inventing one, and no
            # patient identifier is ever derived from the upload.
            patient_id=0,
            series_id=series_id,
            slice_id=slice_id,
            slice_index=slice_indices[position],
            mask_source="prediction",
            mm_per_pixel=config.mm_per_pixel,
            min_area_px=config.min_area_px,
        )
        records.extend(rows)
        if progress and total and position % max(1, total // 5) == 0:
            progress(75 + int(8 * position / total), "Feature extraction")

    per_slice = pd.DataFrame(records)
    if per_slice.empty:
        return per_slice, per_slice

    aggregated = aggregate_disc_records(per_slice)
    aggregated = add_relative_height_features(aggregated)
    return per_slice, aggregated


# ---------------------------------------------------------------------------
# 5. Findings
# ---------------------------------------------------------------------------


@dataclass
class FindingEstimators:
    """The persisted finding estimators, loaded once."""

    binary: dict = field(default_factory=dict)
    ordinal: dict | None = None
    manifest: dict = field(default_factory=dict)

    @property
    def supported_targets(self) -> list[str]:
        targets = sorted(self.binary)
        if self.ordinal:
            targets.append(self.ordinal["target"])
        return targets


def load_finding_estimators(model_dir: Path) -> FindingEstimators:
    """Load the estimators written by ``scripts/18_export_finding_models.py``."""
    import joblib

    manifest_path = model_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

    estimators = FindingEstimators(manifest=manifest)
    for artifact in sorted(model_dir.glob("*.joblib")):
        payload = joblib.load(artifact)
        if payload.get("kind") == "ordinal":
            estimators.ordinal = payload
        else:
            estimators.binary[payload["target"]] = payload
    return estimators


def estimate_findings(
    aggregated,
    estimators: FindingEstimators,
    *,
    modality: str | None,
    progress: PROGRESS_CALLBACK | None = None,
) -> dict:
    """Apply the persisted estimators to the measured discs.

    Returns a mapping of ``ivd_label -> {target: {...}}``. A target that is not
    served, or whose preconditions are not met, yields ``None`` with a reason
    rather than a guess.

    The Pfirrmann estimator was validated on T2 and T2-SPACE series only, because
    the grade is defined on T2 signal. On a T1 study it is withheld rather than
    extrapolated.
    """
    import numpy as np
    import pandas as pd

    if aggregated is None or len(aggregated) == 0:
        return {}
    if progress:
        progress(90, "Radiological analysis")

    def feature_matrix(features: list[str]) -> np.ndarray:
        frame = pd.DataFrame(index=aggregated.index)
        for name in features:
            frame[name] = (
                aggregated[name] if name in aggregated.columns else np.nan
            )
        return frame.to_numpy(dtype=np.float64)

    out: dict[int, dict] = {
        int(row.ivd_label): {} for row in aggregated.itertuples()
    }

    # ---- binary findings ----
    for target, payload in sorted(estimators.binary.items()):
        matrix = feature_matrix(payload["features"])
        pipeline = payload["pipeline"]
        try:
            probability = pipeline.predict_proba(matrix)[:, 1]
            predicted = pipeline.predict(matrix)
        except Exception:  # noqa: BLE001 - a broken estimator must not 500
            for label in out:
                out[label][target] = {
                    "value": None, "probability": None,
                    "unavailable_reason": "The estimator could not be applied "
                                          "to this study's features.",
                }
            continue
        validated = payload.get("validated", {})
        for position, row in enumerate(aggregated.itertuples()):
            out[int(row.ivd_label)][target] = {
                "value": bool(int(predicted[position])),
                "probability": round(float(probability[position]), 4),
                "label": payload.get("label", target),
                "strength": validated.get("strength"),
                "validated_metric": validated.get("metric"),
                "validated_value": validated.get("published"),
                "prevalence_baseline": validated.get("prevalence_baseline"),
                "source": "model_prediction",
                "unavailable_reason": None,
            }

    # ---- Pfirrmann ----
    ordinal = estimators.ordinal
    if ordinal is not None:
        allowed = ordinal.get("modalities") or []
        if modality is not None and modality not in allowed:
            reason = (
                f"The Pfirrmann estimator was validated on {', '.join(allowed)} "
                f"series only, because the grade is defined on T2 signal. This "
                f"study was identified as '{modality}', so no grade is reported."
            )
            for label in out:
                out[label]["pfirrmann_grade"] = {
                    "value": None, "expected": None,
                    "unavailable_reason": reason, "source": "model_prediction",
                }
        else:
            matrix = feature_matrix(ordinal["features"])
            model = ordinal["model"]
            validated = ordinal.get("validated", {})
            note = None
            if modality is None:
                note = (
                    "The modality could not be determined from the upload, so "
                    "the grade is reported on the assumption that this is a T2 "
                    "series. Treat it with extra caution."
                )
            try:
                grades = model.predict(matrix)
                expected = model.predict_expected(matrix)
                probabilities = model.predict_proba(matrix)
            except Exception:  # noqa: BLE001
                for label in out:
                    out[label]["pfirrmann_grade"] = {
                        "value": None, "expected": None,
                        "unavailable_reason": "The grade estimator could not be "
                                              "applied to this study's features.",
                        "source": "model_prediction",
                    }
            else:
                classes = ordinal.get("classes", [1, 2, 3, 4, 5])
                for position, row in enumerate(aggregated.itertuples()):
                    out[int(row.ivd_label)]["pfirrmann_grade"] = {
                        "value": int(grades[position]),
                        "expected": round(float(expected[position]), 3),
                        "probabilities": {
                            str(int(cls)): round(float(probabilities[position][i]), 4)
                            for i, cls in enumerate(classes)
                        },
                        "label": ordinal.get("label", "Pfirrmann grade"),
                        "strength": validated.get("strength"),
                        "validated_metric": "end_to_end_quadratic_weighted_kappa",
                        "validated_value": validated.get("end_to_end_qwk_sprint5"),
                        "within_one_grade": validated.get(
                            "published_within_one_grade"),
                        "source": "model_prediction",
                        "caveat": note,
                        "unavailable_reason": None,
                    }

    # ---- explicitly unsupported targets ----
    unsupported = (estimators.manifest or {}).get("unsupported", {})
    for target, reason in unsupported.items():
        for label in out:
            out[label][target] = {
                "value": None, "probability": None,
                "unavailable_reason": reason, "source": "unsupported",
            }

    return out
