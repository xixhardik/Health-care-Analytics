"""Resumable training loop with checkpointing and early stopping.

Written for the Sprint 2 extended run. The baseline training script
(``scripts/train_unet.py``) is left untouched so the 12-epoch result stays
exactly reproducible; this module is additive.

What it adds over the baseline loop
-----------------------------------
* a full checkpoint every epoch containing model **and** optimiser **and**
  scheduler state, the epoch counter, the history so far, the early-stopping
  counter and the RNG states - so an interrupted run resumes without
  re-treading epochs or replaying the same augmentations
* two separate "best" checkpoints, selected on different criteria (validation
  foreground Dice and validation loss), because they do not always agree
* early stopping on validation foreground Dice
* history persisted to CSV **and** JSON after every epoch, so an interruption
  never loses the log

Test-set safety
---------------
This module never receives a test loader. :class:`Trainer` accepts only
``train`` and ``val`` loaders and raises if a ``test`` key is present, so the
held-out set cannot influence training or model selection even by accident.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.metrics import ConfusionAccumulator
from src.preprocessing.labels import SEMANTIC_CLASSES

N_CLASSES = len(SEMANTIC_CLASSES)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    """Everything needed to resume a run exactly where it stopped."""

    epoch: int = 0                       # epochs COMPLETED so far
    best_val_dice: float = -1.0
    best_val_dice_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val_loss_epoch: int = -1
    epochs_without_improvement: int = 0
    history: list[dict] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""
    # Cumulative count of distinct training slices seen. Only meaningful when a
    # per-epoch loader factory rotates the subset; with a fixed loader it stays
    # equal to the subset size.
    unique_slices_seen: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingState":
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def architecture_of(model: torch.nn.Module) -> dict:
    """Architecture arguments needed to rebuild the model."""
    return {
        "in_channels": model.in_channels,
        "n_classes": model.n_classes,
        "base_channels": model.base_channels,
        "depth": model.depth,
        "bilinear": model.bilinear,
    }


def save_full_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    state: TrainingState,
    config: dict,
    augment_rng_state: dict | None = None,
) -> None:
    """Write a fully resumable checkpoint.

    Saved atomically via a temporary file and replace, so an interruption
    *during* the write cannot leave a truncated checkpoint that would then fail
    to load on resume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "architecture": architecture_of(model),
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "training_state": state.to_dict(),
        "config": config,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "augment": augment_rng_state,
        },
        "metadata": {
            "epoch": state.epoch,
            "best_val_dice": state.best_val_dice,
            "best_val_dice_epoch": state.best_val_dice_epoch,
            "best_val_loss": state.best_val_loss,
            "best_val_loss_epoch": state.best_val_loss_epoch,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_weights_checkpoint(
    path: Path, *, model: torch.nn.Module, **metadata
) -> None:
    """Write a weights-only checkpoint, matching the Sprint 2 baseline format.

    Kept format-compatible with ``scripts/evaluate_unet.py`` so the evaluation
    script needs no special case for extended-run checkpoints.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "architecture": architecture_of(model),
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint, tolerating both the full and weights-only formats."""
    return torch.load(path, map_location="cpu", weights_only=False)


def is_resumable(checkpoint: dict) -> bool:
    """True if the checkpoint carries optimiser state for a genuine resume."""
    return (
        checkpoint.get("optimiser_state") is not None
        and checkpoint.get("training_state") is not None
    )


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop when the monitored metric has not improved for ``patience`` epochs.

    Monitors validation foreground Dice (higher is better). ``min_delta``
    guards against counting numerical noise as improvement.
    """

    def __init__(self, patience: int = 6, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta

    def update(self, state: TrainingState, value: float) -> bool:
        """Record ``value``; return True if training should stop.

        Mutates ``state.epochs_without_improvement`` so the counter survives a
        checkpoint/resume cycle.
        """
        if value > state.best_val_dice + self.min_delta:
            state.epochs_without_improvement = 0
            return False

        state.epochs_without_improvement += 1
        if state.epochs_without_improvement >= self.patience:
            state.stopped_early = True
            state.stop_reason = (
                f"validation foreground Dice did not improve for "
                f"{self.patience} consecutive epochs "
                f"(best {state.best_val_dice:.4f} at epoch "
                f"{state.best_val_dice_epoch})"
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimiser=None,
    *,
    log_every: int = 50,
    log_prefix: str = "",
) -> dict:
    """One pass over ``loader``. Trains when ``optimiser`` is supplied.

    Identical in behaviour to the baseline loop: same channels_last memory
    format, same loss call, same argmax for metrics.
    """
    is_train = optimiser is not None
    model.train(is_train)

    accumulator = ConfusionAccumulator(N_CLASSES)
    totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0}
    n_batches = 0
    start = time.perf_counter()

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(memory_format=torch.channels_last)
            mask = batch["mask"]

            logits = model(image)
            parts = criterion(logits, mask)

            if is_train:
                optimiser.zero_grad()
                parts["loss"].backward()
                optimiser.step()

            totals["loss"] += float(parts["loss"].detach())
            totals["ce"] += float(parts["ce"])
            totals["dice"] += float(parts["dice"])
            n_batches += 1
            accumulator.update(logits.argmax(dim=1), mask)

            if log_every and step % log_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"{log_prefix}  batch {step}/{len(loader)} "
                    f"loss={totals['loss'] / n_batches:.4f} "
                    f"({elapsed / step:.2f}s/batch)",
                    flush=True,
                )

    divisor = max(n_batches, 1)
    return {
        "loss": totals["loss"] / divisor,
        "ce": totals["ce"] / divisor,
        "dice_loss": totals["dice"] / divisor,
        "seconds": time.perf_counter() - start,
        "metrics": accumulator.summary(),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Resumable trainer with per-epoch checkpointing and early stopping."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        loaders: dict,
        criterion,
        optimiser: torch.optim.Optimizer,
        scheduler,
        output_dir: Path,
        config: dict,
        early_stopping: EarlyStopping | None = None,
        state: TrainingState | None = None,
        train_loader_factory=None,
    ) -> None:
        # Hard guarantee that the test set cannot leak into training or
        # model selection.
        if "test" in loaders:
            raise ValueError(
                "Trainer received a 'test' loader. The held-out test set must not "
                "be used during training or model selection."
            )
        if "val" not in loaders:
            raise ValueError("Trainer needs a 'val' loader")
        if train_loader_factory is None and "train" not in loaders:
            raise ValueError(
                "Trainer needs either a 'train' loader or a train_loader_factory"
            )

        # Optional per-epoch training loader. When supplied, the training subset
        # is rebuilt at the start of every epoch by calling
        # ``train_loader_factory(epoch)``, which is what lets a rotating sampler
        # expose different slices each epoch. The validation loader is always
        # fixed, so the validation metric stays comparable across epochs.
        self.train_loader_factory = train_loader_factory
        self._slices_seen: set[str] = set()

        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.config = config
        self.early_stopping = early_stopping or EarlyStopping(patience=6)
        self.state = state or TrainingState()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------

    @property
    def last_path(self) -> Path:
        return self.output_dir / "last.pt"

    @property
    def best_dice_path(self) -> Path:
        return self.output_dir / "best_val_dice.pt"

    @property
    def best_loss_path(self) -> Path:
        return self.output_dir / "best_val_loss.pt"

    def epoch_path(self, epoch: int) -> Path:
        return self.output_dir / "epochs" / f"epoch_{epoch:03d}.pt"

    @property
    def history_csv(self) -> Path:
        return self.output_dir / "history.csv"

    @property
    def history_json(self) -> Path:
        return self.output_dir / "history.json"

    # -- helpers ----------------------------------------------------------

    def _augment_rng_state(self) -> dict | None:
        # With a per-epoch factory the dataset is rebuilt from a deterministic
        # (seed, epoch) pair, so its RNG state does not need checkpointing -
        # resuming epoch N reconstructs the same dataset.
        if self.train_loader_factory is not None:
            return None
        dataset = self.loaders["train"].dataset
        getter = getattr(dataset, "get_rng_state", None)
        return getter() if callable(getter) else None

    def _restore_augment_rng(self, state: dict | None) -> None:
        if state is None or self.train_loader_factory is not None:
            return
        dataset = self.loaders["train"].dataset
        setter = getattr(dataset, "set_rng_state", None)
        if callable(setter):
            setter(state)

    def _train_loader_for(self, epoch: int):
        """Return the training loader for ``epoch`` and update coverage."""
        if self.train_loader_factory is not None:
            loader = self.train_loader_factory(epoch)
            self.loaders["train"] = loader
        else:
            loader = self.loaders["train"]

        # Track distinct slices seen across the run. The index frame is carried
        # on the dataset, so this needs no extra bookkeeping in the caller.
        index = getattr(loader.dataset, "index", None)
        if index is not None and "slice_id" in index.columns:
            self._slices_seen.update(index["slice_id"].tolist())
            self.state.unique_slices_seen = len(self._slices_seen)
        return loader

    def recompute_coverage(self, up_to_epoch: int) -> int:
        """Rebuild the seen-slice set for epochs 1..N after a resume.

        The rotating sampler is a pure function of (seed, epoch), so the set can
        be reconstructed exactly without loading any image data.
        """
        if self.train_loader_factory is None:
            return self.state.unique_slices_seen
        self._slices_seen = set()
        for epoch in range(1, up_to_epoch + 1):
            loader = self.train_loader_factory(epoch)
            index = getattr(loader.dataset, "index", None)
            if index is not None:
                self._slices_seen.update(index["slice_id"].tolist())
        self.state.unique_slices_seen = len(self._slices_seen)
        return self.state.unique_slices_seen

    def _persist_history(self) -> None:
        """Write the history after every epoch so nothing is lost on a crash."""
        frame = pd.DataFrame(self.state.history)
        frame.to_csv(self.history_csv, index=False)
        with open(self.history_json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": self.config,
                    "state": {
                        k: v for k, v in self.state.to_dict().items() if k != "history"
                    },
                    "history": self.state.history,
                },
                handle,
                indent=2,
            )

    # -- resume -----------------------------------------------------------

    def resume_from(self, checkpoint: dict) -> str:
        """Restore what the checkpoint provides. Returns a description.

        Two cases are handled and reported differently, because they are not
        equivalent:

        ``full``
            optimiser, scheduler, epoch counter, history and RNG states are all
            present - the run continues exactly.
        ``weights_only``
            only the model weights are present. The optimiser must be rebuilt,
            so Adam's first/second moment estimates start from zero. This is a
            genuine difference from an uninterrupted run and is reported as
            such rather than glossed over.
        """
        self.model.load_state_dict(checkpoint["state_dict"])

        if not is_resumable(checkpoint):
            epoch = int(checkpoint.get("metadata", {}).get("epoch", 0))
            self.state.epoch = epoch
            return "weights_only"

        self.optimiser.load_state_dict(checkpoint["optimiser_state"])
        if self.scheduler is not None and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.state = TrainingState.from_dict(checkpoint["training_state"])

        rng = checkpoint.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        self._restore_augment_rng(rng.get("augment"))
        return "full"

    # -- fit --------------------------------------------------------------

    def fit(self, max_epochs: int) -> TrainingState:
        """Train until ``max_epochs`` total, or until early stopping fires."""
        if self.state.epoch >= max_epochs:
            print(
                f"Nothing to do: {self.state.epoch} epochs already completed and "
                f"max_epochs={max_epochs}."
            )
            return self.state

        print(
            f"Training epochs {self.state.epoch + 1}..{max_epochs} "
            f"(early stopping: monitor=val foreground Dice, "
            f"patience={self.early_stopping.patience})"
        )

        for epoch in range(self.state.epoch + 1, max_epochs + 1):
            print(f"\nEpoch {epoch}/{max_epochs}")
            learning_rate = self.optimiser.param_groups[0]["lr"]

            train_loader = self._train_loader_for(epoch)
            n_epoch_slices = len(train_loader.dataset)
            if self.train_loader_factory is not None:
                print(f"  epoch subset: {n_epoch_slices:,} slices | "
                      f"unique seen so far: {self.state.unique_slices_seen:,}")

            train_result = run_epoch(
                self.model, train_loader, self.criterion, self.optimiser
            )
            val_result = run_epoch(
                self.model, self.loaders["val"], self.criterion, None, log_every=0
            )

            val_aggregate = val_result["metrics"]["aggregate"]
            val_foreground = val_aggregate["macro_foreground"]
            per_class = val_aggregate["per_class"]

            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "n_train_slices_this_epoch": n_epoch_slices,
                "unique_train_slices_seen": self.state.unique_slices_seen,
                "train_loss": train_result["loss"],
                "train_ce": train_result["ce"],
                "train_dice_loss": train_result["dice_loss"],
                "train_fg_dice":
                    train_result["metrics"]["aggregate"]["macro_foreground"]["dice"],
                "val_loss": val_result["loss"],
                "val_fg_dice": val_foreground["dice"],
                "val_fg_iou": val_foreground["iou"],
                "val_fg_precision": val_foreground["precision"],
                "val_fg_recall": val_foreground["recall"],
                "val_dice_vertebra": per_class["vertebra"]["dice"],
                "val_dice_intervertebral_disc": per_class["intervertebral_disc"]["dice"],
                "val_dice_spinal_canal": per_class["spinal_canal"]["dice"],
                "val_dice_background": per_class["background"]["dice"],
                "train_seconds": round(train_result["seconds"], 1),
                "val_seconds": round(val_result["seconds"], 1),
                "epoch_seconds": round(
                    train_result["seconds"] + val_result["seconds"], 1
                ),
            }

            # --- model selection (validation only, never test) ------------
            improved_dice = row["val_fg_dice"] > self.state.best_val_dice
            improved_loss = row["val_loss"] < self.state.best_val_loss

            # Early stopping is evaluated BEFORE best_val_dice is updated, so
            # the comparison is against the previous best.
            should_stop = self.early_stopping.update(self.state, row["val_fg_dice"])

            if improved_dice:
                self.state.best_val_dice = row["val_fg_dice"]
                self.state.best_val_dice_epoch = epoch
                save_weights_checkpoint(
                    self.best_dice_path,
                    model=self.model,
                    epoch=epoch,
                    val_fg_dice=row["val_fg_dice"],
                    val_loss=row["val_loss"],
                    selection="best validation foreground Dice",
                )
            if improved_loss:
                self.state.best_val_loss = row["val_loss"]
                self.state.best_val_loss_epoch = epoch
                save_weights_checkpoint(
                    self.best_loss_path,
                    model=self.model,
                    epoch=epoch,
                    val_fg_dice=row["val_fg_dice"],
                    val_loss=row["val_loss"],
                    selection="best validation loss",
                )

            row["is_best_val_dice"] = bool(improved_dice)
            row["is_best_val_loss"] = bool(improved_loss)
            row["epochs_without_improvement"] = self.state.epochs_without_improvement

            if self.scheduler is not None:
                self.scheduler.step()

            self.state.epoch = epoch
            self.state.history.append(row)
            self._persist_history()

            # Per-epoch archive plus the resumable 'last'.
            save_full_checkpoint(
                self.epoch_path(epoch),
                model=self.model,
                optimiser=self.optimiser,
                scheduler=self.scheduler,
                state=self.state,
                config=self.config,
                augment_rng_state=self._augment_rng_state(),
            )
            save_full_checkpoint(
                self.last_path,
                model=self.model,
                optimiser=self.optimiser,
                scheduler=self.scheduler,
                state=self.state,
                config=self.config,
                augment_rng_state=self._augment_rng_state(),
            )

            print(
                f"  lr={learning_rate:.2e}  "
                f"train_loss={row['train_loss']:.4f}  "
                f"val_loss={row['val_loss']:.4f}  "
                f"val_fg_dice={row['val_fg_dice']:.4f}  "
                f"(vert={row['val_dice_vertebra']:.4f} "
                f"ivd={row['val_dice_intervertebral_disc']:.4f} "
                f"canal={row['val_dice_spinal_canal']:.4f})  "
                f"seen={row['unique_train_slices_seen']:,}  "
                f"{row['epoch_seconds']:.0f}s"
                + ("  [best dice]" if improved_dice else "")
                + ("  [best loss]" if improved_loss else ""),
                flush=True,
            )

            if should_stop:
                print(f"\nEarly stopping: {self.state.stop_reason}")
                break

        if not self.state.stopped_early:
            self.state.stop_reason = (
                f"reached the maximum of {max_epochs} epochs without triggering "
                f"early stopping"
            )

        self._persist_history()
        return self.state

    # -- finalise ---------------------------------------------------------

    def restore_best(self, criterion: str = "val_dice") -> dict:
        """Load the selected best checkpoint back into the model.

        Called at the end so the model left in memory - and used for the final
        evaluation - is the selected one rather than whatever the last epoch
        happened to produce.
        """
        path = self.best_dice_path if criterion == "val_dice" else self.best_loss_path
        if not path.exists():
            raise FileNotFoundError(f"No best checkpoint at {path}")
        checkpoint = load_checkpoint(path)
        self.model.load_state_dict(checkpoint["state_dict"])
        return checkpoint.get("metadata", {})
