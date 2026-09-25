"""Sprint 2 extended run - continue training the SAME 16-channel U-Net.

A controlled experiment. Exactly one thing changes relative to the Sprint 2
baseline: the number of epochs (12 -> up to 30). Everything else is held fixed:

* architecture      16-channel U-Net, depth 4, bilinear  (1,963,860 params)
* split             the Sprint 1 patient-level split, unchanged
* seed              42
* preprocessing     Sprint 1 slices, 352x256 at 1.0 mm/px, untouched
* classes           4 (background / vertebra / IVD / spinal canal)
* loss              CrossEntropy + soft Dice, equal weight
* optimiser         Adam, lr 1e-3
* augmentation      the same translate + intensity-jitter policy
* slices per epoch  2,560 train / 800 val - the SAME subset, since the seed and
                    cap are unchanged
* CPU settings      channels_last, fp32. No bf16 (measured 21x slower here).

Writes everything to a NEW directory so the baseline is never overwritten:

    outputs/checkpoints/sprint2_extended/
        last.pt                 fully resumable (model+optimiser+scheduler+RNG)
        best_val_dice.pt        best validation foreground Dice
        best_val_loss.pt        best validation loss
        epochs/epoch_NNN.pt     per-epoch archive
        history.csv / .json     per-epoch log, rewritten every epoch

The test set is never loaded here.

Usage
-----
    python scripts/train_unet_extended.py                 # resume, up to 30
    python scripts/train_unet_extended.py --max-epochs 30
    python scripts/train_unet_extended.py --fresh         # ignore prior weights
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.data import N_CLASSES, build_dataloaders, describe_loaders  # noqa: E402
from src.models.losses import DiceCrossEntropyLoss  # noqa: E402
from src.models.training import (  # noqa: E402
    EarlyStopping,
    Trainer,
    TrainingState,
    is_resumable,
    load_checkpoint,
)
from src.models.unet import build_unet  # noqa: E402
from src.utils.paths import OUTPUTS_DIR, RANDOM_SEED, ensure_dirs  # noqa: E402
from src.utils.reporting import save_json  # noqa: E402

EXPERIMENT = "sprint2_extended"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / EXPERIMENT
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT

#: The baseline run's final checkpoint, used to bootstrap when this experiment
#: has no checkpoint of its own yet.
BASELINE_LAST = OUTPUTS_DIR / "models" / "unet_last.pt"

# --- frozen hyperparameters (identical to the baseline) --------------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
BASE_CHANNELS = 16
DEPTH = 4
DOWNSAMPLE = 1
MAX_TRAIN_SLICES = 2560
MAX_VAL_SLICES = 800
NUM_WORKERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Total epochs including those already completed.")
    parser.add_argument("--patience", type=int, default=6,
                        help="Early-stopping patience on validation foreground Dice.")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--fresh", action="store_true",
                        help="Start from random weights instead of resuming.")
    parser.add_argument("--restore-criterion", default="val_dice",
                        choices=["val_dice", "val_loss"],
                        help="Which best checkpoint to load back at the end.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"cuda={torch.cuda.is_available()}")
    print(f"experiment: {EXPERIMENT}")
    print(f"checkpoints -> {CHECKPOINT_DIR}")

    # --- data: train + val ONLY ------------------------------------------
    # The test split is deliberately not built here.
    loaders, frames = build_dataloaders(
        batch_size=BATCH_SIZE,
        downsample=DOWNSAMPLE,
        augment_train=True,
        num_workers=NUM_WORKERS,
        max_train_slices=MAX_TRAIN_SLICES,
        max_eval_slices=MAX_VAL_SLICES,
        seed=args.seed,
    )
    loaders.pop("test", None)
    frames.pop("test", None)
    print("\nData (test split intentionally excluded):")
    print(describe_loaders(loaders, frames))

    # --- model, loss, optimiser: identical to the baseline ----------------
    model = build_unet(
        n_classes=N_CLASSES, base_channels=BASE_CHANNELS, depth=DEPTH
    ).to(memory_format=torch.channels_last)
    criterion = DiceCrossEntropyLoss(n_classes=N_CLASSES)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.max_epochs
    )
    print(f"\nModel: {model.describe()}")

    config = {
        "experiment": EXPERIMENT,
        "max_epochs": args.max_epochs,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "base_channels": BASE_CHANNELS,
        "depth": DEPTH,
        "downsample": DOWNSAMPLE,
        "input_size": [352, 256],
        "augment": True,
        "seed": args.seed,
        "loss": "CrossEntropy + soft Dice (equal weight)",
        "optimiser": "Adam",
        "scheduler": f"CosineAnnealingLR(T_max={args.max_epochs})",
        "early_stopping": {
            "monitor": "val_fg_dice",
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "model": model.describe(),
        "n_parameters": model.count_parameters(),
        "device": "cpu",
        "memory_format": "channels_last",
        "bf16_autocast": False,
        "slices_used": {k: int(len(v)) for k, v in frames.items()},
        "patients_used": {k: int(v["patient_id"].nunique()) for k, v in frames.items()},
        "test_set_used_during_training": False,
        "selection_metric": "validation macro foreground Dice",
        "controlled_variable": "number of epochs (12 -> up to 30); all else fixed",
    }

    trainer = Trainer(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimiser=optimiser,
        scheduler=scheduler,
        output_dir=CHECKPOINT_DIR,
        config=config,
        early_stopping=EarlyStopping(patience=args.patience, min_delta=args.min_delta),
        state=TrainingState(),
    )

    # --- resume ----------------------------------------------------------
    resume_info = resume(trainer, args, scheduler)
    config["resume"] = resume_info
    save_json(config, REPORT_DIR / "training_config.json")

    # --- train -----------------------------------------------------------
    started = time.perf_counter()
    state = trainer.fit(max_epochs=args.max_epochs)
    wall_seconds = time.perf_counter() - started

    # --- restore the selected best checkpoint ----------------------------
    # Skipped when no epoch ran in this session, since no best checkpoint of
    # this experiment exists yet.
    if state.history:
        print(f"\nRestoring best checkpoint (criterion={args.restore_criterion}) ...")
        restored = trainer.restore_best(args.restore_criterion)
        print(f"  restored: {restored}")
    else:
        restored = {}
        print("\nNo epoch ran in this session; nothing to restore.")

    summary = {
        "experiment": EXPERIMENT,
        "resume": resume_info,
        "epochs_completed_total": state.epoch,
        "epochs_run_this_session": state.epoch - resume_info["resumed_at_epoch"],
        "max_epochs": args.max_epochs,
        "stopped_early": state.stopped_early,
        "stop_reason": state.stop_reason,
        "best_val_dice": state.best_val_dice,
        "best_val_dice_epoch": state.best_val_dice_epoch,
        "best_val_loss": state.best_val_loss,
        "best_val_loss_epoch": state.best_val_loss_epoch,
        "epochs_without_improvement_at_end": state.epochs_without_improvement,
        "wall_seconds_this_session": round(wall_seconds, 1),
        "wall_minutes_this_session": round(wall_seconds / 60, 1),
        "mean_epoch_seconds": round(
            float(np.mean([r["epoch_seconds"] for r in state.history])), 1
        ) if state.history else None,
        "restored_checkpoint": {
            "criterion": args.restore_criterion,
            "path": str(
                trainer.best_dice_path
                if args.restore_criterion == "val_dice"
                else trainer.best_loss_path
            ),
            "metadata": restored,
            "restored": bool(restored),
        },
        "test_set_used_during_training": False,
        "checkpoints": {
            "last": str(trainer.last_path),
            "best_val_dice": str(trainer.best_dice_path),
            "best_val_loss": str(trainer.best_loss_path),
            "per_epoch_dir": str(CHECKPOINT_DIR / "epochs"),
        },
    }
    save_json(summary, REPORT_DIR / "training_summary.json")

    print("\n" + "=" * 78)
    print("EXTENDED TRAINING COMPLETE")
    print("=" * 78)
    print(f"  epochs completed (total)  : {state.epoch}")
    print(f"  epochs this session       : {summary['epochs_run_this_session']}")
    print(f"  wall time this session    : {summary['wall_minutes_this_session']} min")
    print(f"  stopped early             : {state.stopped_early}")
    print(f"  reason                    : {state.stop_reason}")
    print(f"  best val foreground Dice  : {state.best_val_dice:.4f} "
          f"(epoch {state.best_val_dice_epoch})")
    print(f"  best val loss             : {state.best_val_loss:.4f} "
          f"(epoch {state.best_val_loss_epoch})")
    print(f"\n  -> {REPORT_DIR / 'training_summary.json'}")
    print(f"  -> {trainer.history_csv}")
    print("\nNext: evaluate ONLY the selected checkpoint on the held-out test set:")
    print(f"  python scripts/evaluate_unet.py --split test \\")
    print(f"      --checkpoint {trainer.best_dice_path} \\")
    print(f"      --tag {EXPERIMENT} --save-predictions")


def resume(trainer: Trainer, args: argparse.Namespace, scheduler) -> dict:
    """Decide where to continue from, preferring a genuine resume.

    Priority:
      1. this experiment's own ``last.pt``  -> full resume (optimiser + RNG)
      2. the baseline's ``unet_last.pt``    -> weights-only bootstrap
      3. nothing                            -> fresh random initialisation
    """
    if args.fresh:
        print("\n--fresh: starting from random initialisation.")
        return {
            "mode": "fresh",
            "resumed_at_epoch": 0,
            "source": None,
            "optimiser_state_restored": False,
            "note": "Requested explicitly; no weights reused.",
        }

    if trainer.last_path.exists():
        checkpoint = load_checkpoint(trainer.last_path)
        mode = trainer.resume_from(checkpoint)
        print(f"\nResuming from {trainer.last_path} ({mode}) at epoch "
              f"{trainer.state.epoch}")
        return {
            "mode": f"{mode}_from_experiment_checkpoint",
            "resumed_at_epoch": trainer.state.epoch,
            "source": str(trainer.last_path),
            "optimiser_state_restored": mode == "full",
            "note": "Exact continuation: optimiser, scheduler and RNG restored.",
        }

    if BASELINE_LAST.exists():
        checkpoint = load_checkpoint(BASELINE_LAST)
        mode = trainer.resume_from(checkpoint)
        completed = trainer.state.epoch
        print(f"\nBootstrapping from the Sprint 2 baseline {BASELINE_LAST.name} "
              f"({mode}), which completed {completed} epochs.")

        # The baseline saved no optimiser or scheduler state, so both are
        # rebuilt. Advance the cosine schedule to the resume point so the
        # learning rate continues the 30-epoch curve instead of restarting at
        # its peak - a restart at 1e-3 would be a large perturbation to an
        # already-converging model.
        for _ in range(completed):
            scheduler.step()
        resumed_lr = trainer.optimiser.param_groups[0]["lr"]
        expected_lr = LEARNING_RATE * (
            1 + math.cos(math.pi * completed / args.max_epochs)
        ) / 2
        print(f"  scheduler fast-forwarded {completed} steps: "
              f"lr now {resumed_lr:.3e} (cosine at epoch {completed}/"
              f"{args.max_epochs}, expected {expected_lr:.3e})")

        return {
            "mode": "weights_only_from_baseline",
            "resumed_at_epoch": completed,
            "source": str(BASELINE_LAST),
            "optimiser_state_restored": False,
            "scheduler_fast_forwarded_steps": completed,
            "learning_rate_at_resume": round(float(resumed_lr), 8),
            "note": (
                "The baseline checkpoint stored weights only - no optimiser or "
                "scheduler state - so Adam's moment estimates restart from zero "
                "and the LR schedule was re-derived. This run is therefore a "
                "warm-restarted continuation, NOT bit-identical to an "
                "uninterrupted 30-epoch run. The baseline also used "
                "CosineAnnealingLR(T_max=12), so its LR had annealed to ~0 by "
                "epoch 12; here the schedule is re-cast over 30 epochs."
            ),
        }

    print("\nNo checkpoint found; starting from random initialisation.")
    return {
        "mode": "fresh",
        "resumed_at_epoch": 0,
        "source": None,
        "optimiser_state_restored": False,
        "note": "No prior checkpoint existed.",
    }


if __name__ == "__main__":
    main()
