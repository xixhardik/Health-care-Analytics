"""Backend configuration.

Everything that could differ between machines comes from the environment, with
project-relative defaults so a fresh clone runs without editing anything. No
absolute path is hardcoded; all paths are built with :mod:`pathlib` from the
repository root, which is located relative to this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    project_root: Path = PROJECT_ROOT
    app_name: str = "Lumbar Spine MRI Analysis API"
    api_version: str = "1.0.0"

    #: The validated Sprint 3 segmentation checkpoint. Sprint 4 (width 32) is
    #: deliberately not the default: it measured worse on every Dice class and on
    #: disc indexing, so serving it would regress the product.
    checkpoint: Path = field(
        default_factory=lambda: _env_path(
            "MODEL_CHECKPOINT",
            PROJECT_ROOT / "outputs/checkpoints/sprint3_coverage/best_val_dice.pt",
        )
    )
    finding_model_dir: Path = field(
        default_factory=lambda: _env_path(
            "FINDING_MODEL_DIR", PROJECT_ROOT / "outputs/models/findings"
        )
    )
    sprint5_params_file: Path = field(
        default_factory=lambda: _env_path(
            "SPRINT5_PARAMS",
            PROJECT_ROOT / "outputs/reports/sprint5_indexing/test_results.json",
        )
    )
    storage_dir: Path = field(
        default_factory=lambda: _env_path(
            "STORAGE_DIR", PROJECT_ROOT / "outputs/app_data"
        )
    )

    batch_size: int = field(default_factory=lambda: _env_int("INFERENCE_BATCH_SIZE", 4))
    max_workers: int = field(default_factory=lambda: _env_int("JOB_WORKERS", 1))
    job_timeout_seconds: int = field(
        default_factory=lambda: _env_int("JOB_TIMEOUT_SECONDS", 1800)
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS", ["http://localhost:3000", "http://127.0.0.1:3000"]
        )
    )

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def analyses_dir(self) -> Path:
        return self.storage_dir / "analyses"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.analyses_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings, so the environment is read once per process."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
