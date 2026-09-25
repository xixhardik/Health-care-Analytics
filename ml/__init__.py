"""Inference adapter over the validated research code in ``src/``.

Why this package exists
-----------------------
``src/`` holds the Sprint 1-5 research code. Every published metric in this
project was produced by it, and eighteen scripts import it, so it is treated as
frozen: nothing here modifies it.

The backend, however, should not reach into six different research submodules and
re-derive the orchestration each time. ``ml/`` is the serving seam: a small set of
stable entry points that take one uploaded volume through exactly the validated
pipeline and return plain Python data structures. The FastAPI layer depends only
on this package, never on ``src.*`` directly.

Pipeline, in order:

1. :func:`ml.pipeline.preprocess_volume`  - Sprint 1 preprocessing
2. :func:`ml.pipeline.segment_slices`     - Sprint 3 16-channel U-Net
3. :func:`ml.pipeline.index_discs`        - Sprint 5 series-level indexing + 5D
4. :func:`ml.pipeline.measure_discs`      - disc-level measurements
5. :func:`ml.pipeline.estimate_findings`  - the persisted finding estimators

Nothing in this package trains anything.
"""

from ml.pipeline import (
    PipelineConfig,
    estimate_findings,
    index_discs,
    measure_discs,
    preprocess_volume,
    segment_slices,
)
from ml.volume import VolumeInfo, ValidationError, inspect_volume, validate_upload

__all__ = [
    "PipelineConfig",
    "VolumeInfo",
    "ValidationError",
    "estimate_findings",
    "index_discs",
    "inspect_volume",
    "measure_discs",
    "preprocess_volume",
    "segment_slices",
    "validate_upload",
]
