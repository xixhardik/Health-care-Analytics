"""Sprint 3 - training-data coverage experiment.

Controlled single-variable experiment. **Exactly one thing changes** relative to
Sprint 2 Extended: the training subset rotates each epoch instead of being fixed.

CHANGED
    sampler   rotating shard sampler, ~2,284 slices/epoch, 100% coverage by
              epoch 4 (verified in the Sprint 3 audit)

UNCHANGED
    architecture      16-channel U-Net, depth 4, bilinear (1,963,860 params)
    batch size        8
    optimiser         Adam, lr 1e-3
    schedule          CosineAnnealingLR(T_max=30)
    loss              CrossEntropy + soft Dice, equal weight
    augmentation      translate +/-5% and intensity jitter, train split only
    preprocessing     Sprint 1 slices, 352x256 at 1.0 mm/px
    seed              42
    split             Sprint 1 patient-level split
    validation        the same 800-slice validation subset as Sprint 2, so the
                      validation Dice is directly comparable to 0.8985
    classes           4 (background / vertebra / IVD / spinal canal)

DIFFERENT FROM SPRINT 2 BY DESIGN
    initialisation    random, NOT resumed and NOT warm-started - warm-starting
                      from the converged Sprint 2 weights would confound the
                      coverage variable
    early stopping    min_delta 0.001 (Sprint 2 used 0, which made early
                      stopping ineffective - any +0.00004 reset the counter)

The test set is never loaded here.

Outputs
-------
    outputs/checkpoints/sprint3_coverage/   last.pt, best_val_dice.pt,
                                            best_val_loss.pt, epochs/, history.*
    outputs/reports/sprint3_coverage/       config + summary

Usage
-----
    python scripts/12_train_sprint3_coverage.py
    python scripts/12_train_sprint3_coverage.py --max-epochs 30 --patience 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.data import (  # noqa: E402
    N_CLASSES,
    LumbarSliceDataset,
    assert_no_patient_leakage,
    load_slice_index,
    _sample_per_patient,
)
from src.models.losses import DiceCrossEntropyLoss  # noqa: E402
from src.models.samplers import (  # noqa: E402
    measure_coverage,
    rotating_epoch_sample,
    verify_no_leakage_in_schedule,
)
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

EXPERIMENT = "sprint3_coverage"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / EXPERIMENT
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT

# --- frozen hyperparameters, identical to Sprint 2 Extended ----------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
BASE_CHANNELS = 16
DEPTH = 4
DOWNSAMPLE = 1
SLICES_PER_EPOCH = 2284      # the CHANGED variable's per-epoch budget
MAX_VAL_SLICES = 800         # same validation subset as Sprint 2
NUM_WORKERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--restore-criterion", default="val_dice",
                        choices=["val_dice", "val_loss"])
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
    print(f"experiment: {EXPERIMENT}  (controlled variable: training-slice coverage)")
    print(f"checkpoints -> {CHECKPOINT_DIR}")

    index = load_slice_index()
    assert_no_patient_leakage(index)

    # --- verify the sampler before using it ------------------------------
    print("\nVerifying the rotating sampler (no leakage, full coverage) ...")
    leakage = verify_no_leakage_in_schedule(
        index, strategy="rotating", n_epochs=args.max_epochs,
        slices_per_epoch=SLICES_PER_EPOCH, seed=args.seed,
    )
    if not leakage["leakage_free"]:
        raise SystemExit(f"Sampler leaks: {leakage['violations'][:3]}")
    coverage = measure_coverage(
        index, strategy="rotating", n_epochs=args.max_epochs,
        slices_per_epoch=SLICES_PER_EPOCH, seed=args.seed,
    )
    print(f"  leakage-free over {args.max_epochs} epochs: "
          f"{leakage['leakage_free']} (0 violations)")
    print(f"  coverage: {coverage['coverage_pct']}% of "
          f"{coverage['train_slices_total']:,} training slices, "
          f"100% reached at epoch {coverage['epochs_to_full_coverage']}")
    print(f"  slices per epoch: {coverage['slices_per_epoch_actual']:,} | "
          f"each slice seen {coverage['times_seen']['min']}-"
          f"{coverage['times_seen']['max']} times")

    # --- fixed validation loader -----------------------------------------
    # Capped and seeded exactly as Sprint 2 did, so the validation metric is
    # computed on the same 800 slices and is directly comparable to 0.8985.
    validation_rows = _sample_per_patient(
        index[index["split"] == "val"],
        MAX_VAL_SLICES,
        np.random.default_rng(args.seed),
    )
    val_loader = DataLoader(
        LumbarSliceDataset(validation_rows, augment=False, downsample=DOWNSAMPLE,
                           seed=args.seed),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    print(f"\nValidation: {len(validation_rows):,} slices, "
          f"{validation_rows['patient_id'].nunique()} patients (fixed, no augment)")

    # --- per-epoch training loader factory --------------------------------
    def train_loader_factory(epoch: int) -> DataLoader:
        """Build the training loader for one epoch from the rotating sampler.

        The dataset is seeded with (seed, epoch) so augmentation varies between
        epochs - matching Sprint 2, where a single long-lived generator advanced
        across epochs - while staying reproducible for any given epoch.
        """
        rows = rotating_epoch_sample(
            index, epoch=epoch, slices_per_epoch=SLICES_PER_EPOCH, seed=args.seed
        )
        # Defence in depth: the sampler is already verified, but assert here too
        # so a future change cannot silently introduce non-train slices.
        splits = set(rows["split"].unique())
        if splits != {"train"}:
            raise RuntimeError(f"Epoch {epoch} subset contains splits {splits}")

        dataset = LumbarSliceDataset(
            rows, augment=True, downsample=DOWNSAMPLE,
            seed=args.seed * 1000 + epoch,
        )
        return DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True,
        )

    # --- model: RANDOM initialisation, no warm start ---------------------
    model = build_unet(
        n_classes=N_CLASSES, base_channels=BASE_CHANNELS, depth=DEPTH
    ).to(memory_format=torch.channels_last)
    criterion = DiceCrossEntropyLoss(n_classes=N_CLASSES)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.max_epochs
    )
    print(f"\nModel: {model.describe()}  [random initialisation]")

    config = {
        "experiment": EXPERIMENT,
        "controlled_variable": "training-slice coverage (fixed subset -> rotating)",
        "changed": {
            "sampler": "rotating shard sampler",
            "slices_per_epoch": SLICES_PER_EPOCH,
            "coverage_pct_over_run": coverage["coverage_pct"],
            "epochs_to_full_coverage": coverage["epochs_to_full_coverage"],
        },
        "unchanged_from_sprint2_extended": {
            "architecture": model.describe(),
            "base_channels": BASE_CHANNELS,
            "depth": DEPTH,
            "batch_size": BATCH_SIZE,
            "optimiser": "Adam",
            "lr": LEARNING_RATE,
            "scheduler": f"CosineAnnealingLR(T_max={args.max_epochs})",
            "loss": "CrossEntropy + soft Dice (equal weight)",
            "augmentation": "translate +/-5% + intensity jitter, train only",
            "preprocessing": "Sprint 1 slices, 352x256 at 1.0 mm/px",
            "seed": args.seed,
            "split": "Sprint 1 patient-level",
            "validation_slices": int(len(validation_rows)),
            "n_classes": N_CLASSES,
            "input_size": [352, 256],
            "memory_format": "channels_last",
            "bf16_autocast": False,
        },
        "different_by_design": {
            "initialisation": "random (NOT resumed, NOT warm-started)",
            "early_stopping_min_delta": args.min_delta,
            "rationale": (
                "Warm-starting from the converged Sprint 2 weights would confound "
                "the coverage variable. min_delta was 0 in Sprint 2, which made "
                "early stopping ineffective."
            ),
        },
        "early_stopping": {
            "monitor": "val_fg_dice",
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "sampler_verification": {
            "leakage_free": leakage["leakage_free"],
            "violations": len(leakage["violations"]),
            "val_test_patients_excluded": leakage["val_test_patients_excluded"],
            "coverage": {
                k: v for k, v in coverage.items()
                if k != "cumulative_coverage_pct_by_epoch"
            },
        },
        "baselines_to_beat": {
            "sprint2_extended_val_fg_dice": 0.8985,
            "sprint2_extended_test_macro_fg_dice": 0.8979,
            "sprint2_extended_test_vertebra_dice": 0.9133,
            "sprint2_extended_test_ivd_dice": 0.8779,
            "sprint2_extended_test_canal_dice": 0.9026,
            "sprint2_extended_disc_indexing_pct": 82.6,
        },
        "test_set_used_during_training": False,
        "device": "cpu",
    }

    trainer = Trainer(
        model=model,
        loaders={"val": val_loader},
        criterion=criterion,
        optimiser=optimiser,
        scheduler=scheduler,
        output_dir=CHECKPOINT_DIR,
        config=config,
        early_stopping=EarlyStopping(patience=args.patience, min_delta=args.min_delta),
        state=TrainingState(),
        train_loader_factory=train_loader_factory,
    )

    # --- resume only from THIS experiment's own checkpoint ----------------
    resume_info = {"mode": "fresh_random_init", "resumed_at_epoch": 0}
    if trainer.last_path.exists():
        checkpoint = load_checkpoint(trainer.last_path)
        mode = trainer.resume_from(checkpoint)
        seen = trainer.recompute_coverage(trainer.state.epoch)
        resume_info = {
            "mode": f"{mode}_from_experiment_checkpoint",
            "resumed_at_epoch": trainer.state.epoch,
            "unique_slices_seen_recomputed": seen,
        }
        print(f"\nResuming this experiment from epoch {trainer.state.epoch} "
              f"({mode}); {seen:,} unique slices already seen")
    else:
        print("\nStarting from random initialisation (no prior checkpoint).")
    config["resume"] = resume_info
    save_json(config, REPORT_DIR / "training_config.json")

    # --- train ------------------------------------------------------------
    started = time.perf_counter()
    state = trainer.fit(max_epochs=args.max_epochs)
    wall_seconds = time.perf_counter() - started

    if state.history:
        print(f"\nRestoring best checkpoint (criterion={args.restore_criterion}) ...")
        restored = trainer.restore_best(args.restore_criterion)
        print(f"  restored: {restored}")
    else:
        restored = {}

    summary = {
        "experiment": EXPERIMENT,
        "controlled_variable": config["controlled_variable"],
        "resume": resume_info,
        "epochs_completed": state.epoch,
        "max_epochs": args.max_epochs,
        "stopped_early": state.stopped_early,
        "stop_reason": state.stop_reason,
        "epochs_without_improvement_at_end": state.epochs_without_improvement,
        "best_val_dice": state.best_val_dice,
        "best_val_dice_epoch": state.best_val_dice_epoch,
        "best_val_loss": state.best_val_loss,
        "best_val_loss_epoch": state.best_val_loss_epoch,
        "unique_train_slices_seen": state.unique_slices_seen,
        "train_slices_available": int((index["split"] == "train").sum()),
        "coverage_pct_achieved": round(
            100 * state.unique_slices_seen / int((index["split"] == "train").sum()), 2
        ),
        "wall_seconds": round(wall_seconds, 1),
        "wall_minutes": round(wall_seconds / 60, 1),
        "wall_hours": round(wall_seconds / 3600, 2),
        "mean_epoch_seconds": round(
            float(np.mean([r["epoch_seconds"] for r in state.history])), 1
        ) if state.history else None,
        "restored_checkpoint": {
            "criterion": args.restore_criterion,
            "path": str(trainer.best_dice_path),
            "metadata": restored,
        },
        "comparison_vs_sprint2_extended": {
            "sprint2_extended_best_val_fg_dice": 0.8985,
            "sprint3_best_val_fg_dice": state.best_val_dice,
            "delta": round(state.best_val_dice - 0.8985, 5),
            "exceeded": bool(state.best_val_dice > 0.8985),
        },
        "test_set_used_during_training": False,
    }
    save_json(summary, REPORT_DIR / "training_summary.json")

    print("\n" + "=" * 78)
    print("SPRINT 3 COVERAGE TRAINING COMPLETE")
    print("=" * 78)
    print(f"  epochs completed          : {state.epoch} / {args.max_epochs}")
    print(f"  stopped early             : {state.stopped_early}")
    print(f"  reason                    : {state.stop_reason}")
    print(f"  unique train slices seen  : {state.unique_slices_seen:,} / "
          f"{summary['train_slices_available']:,} "
          f"({summary['coverage_pct_achieved']}%)")
    print(f"  best val foreground Dice  : {state.best_val_dice:.4f} "
          f"(epoch {state.best_val_dice_epoch})")
    print(f"  best val loss             : {state.best_val_loss:.4f} "
          f"(epoch {state.best_val_loss_epoch})")
    print(f"  vs Sprint 2 Extended      : 0.8985 -> {state.best_val_dice:.4f} "
          f"({summary['comparison_vs_sprint2_extended']['delta']:+.5f}, "
          f"exceeded={summary['comparison_vs_sprint2_extended']['exceeded']})")
    print(f"  wall time                 : {summary['wall_hours']} h")
    print(f"\n  -> {REPORT_DIR / 'training_summary.json'}")
    print(f"  -> {trainer.history_csv}")
    print("\nNext: evaluate ONLY the selected checkpoint on the held-out test set.")


if __name__ == "__main__":
    main()
