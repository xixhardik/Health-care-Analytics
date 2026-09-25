"""Model service: load once, reuse for every request.

The U-Net checkpoint is ~8 MB and takes noticeable time to deserialise and warm
up on CPU; the finding estimators carry fitted imputation statistics. Loading
either per request would dominate the response time, so both are loaded during
FastAPI startup and held in a process-wide singleton.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.app.pipeline_info import (  # noqa: E402
    DISCLAIMER, PIPELINE, PIPELINE_VERSION, SEGMENTATION_DICE,
)
from ml.pipeline import FindingEstimators, load_finding_estimators  # noqa: E402

logger = logging.getLogger("lumbar.api")

__all__ = [
    "CLASS_TABLE", "DISCLAIMER", "LoadedModel", "ModelService",
    "PIPELINE", "PIPELINE_VERSION", "SEGMENTATION_DICE",
    "get_model_service", "init_model_service",
]

CLASS_TABLE = [
    (0, "background", "Background", None),
    (1, "vertebra", "Vertebrae", SEGMENTATION_DICE["vertebra"]),
    (2, "intervertebral_disc", "Intervertebral discs",
     SEGMENTATION_DICE["intervertebral_disc"]),
    (3, "spinal_canal", "Spinal canal", SEGMENTATION_DICE["spinal_canal"]),
]


@dataclass
class LoadedModel:
    model: object
    device: str
    architecture: str
    parameters: int
    checkpoint_name: str
    metadata: dict


class ModelService:
    """Process-wide holder for the segmentation model and finding estimators."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._segmentation: LoadedModel | None = None
        self._findings: FindingEstimators | None = None
        self._postprocessing_method: str | None = None

    # -- loading ----------------------------------------------------------

    def load(self) -> None:
        """Load everything. Called once from the application lifespan."""
        with self._lock:
            self._load_segmentation()
            self._load_findings()
            self._load_postprocessing()

    def _select_device(self) -> str:
        import torch

        # CUDA is used automatically when present; the development machine here
        # is CPU-only, which the pipeline fully supports.
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_segmentation(self) -> None:
        import torch
        from src.models.unet import build_unet

        path = self._settings.checkpoint
        if not path.exists():
            logger.error("Segmentation checkpoint missing at %s", path)
            return

        device = self._select_device()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        architecture = checkpoint["architecture"]
        model = build_unet(
            n_classes=architecture["n_classes"],
            base_channels=architecture["base_channels"],
            depth=architecture["depth"],
            in_channels=architecture["in_channels"],
            bilinear=architecture["bilinear"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device=device, memory_format=torch.channels_last)
        model.eval()

        counts = model.count_parameters()
        self._segmentation = LoadedModel(
            model=model,
            device=device,
            architecture=model.describe(),
            parameters=int(counts["total"]),
            checkpoint_name=path.name,
            metadata=dict(checkpoint.get("metadata") or {}),
        )
        logger.info("Segmentation model loaded on %s: %s", device, model.describe())

    def _load_findings(self) -> None:
        directory = self._settings.finding_model_dir
        if not directory.exists():
            logger.warning("Finding model directory missing at %s", directory)
            self._findings = FindingEstimators()
            return
        self._findings = load_finding_estimators(directory)
        logger.info(
            "Finding estimators loaded: %s", self._findings.supported_targets
        )

    def _load_postprocessing(self) -> None:
        import json

        path = self._settings.sprint5_params_file
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                self._postprocessing_method = json.load(handle).get(
                    "selected_method", "5A+5D"
                )

    # -- access -----------------------------------------------------------

    @property
    def segmentation(self) -> LoadedModel | None:
        return self._segmentation

    @property
    def findings(self) -> FindingEstimators:
        return self._findings or FindingEstimators()

    @property
    def postprocessing_method(self) -> str | None:
        return self._postprocessing_method

    @property
    def ready(self) -> bool:
        return self._segmentation is not None

    def require_segmentation(self) -> LoadedModel:
        if self._segmentation is None:
            from backend.app.errors import ApiError

            raise ApiError(
                "MODEL_UNAVAILABLE",
                "The segmentation model is not loaded, so analysis cannot run. "
                "Check the MODEL_CHECKPOINT setting.",
                status_code=503,
            )
        return self._segmentation


_service: ModelService | None = None


def init_model_service(settings: Settings) -> ModelService:
    global _service
    _service = ModelService(settings)
    _service.load()
    return _service


def get_model_service() -> ModelService:
    if _service is None:
        raise RuntimeError("Model service accessed before startup")
    return _service
