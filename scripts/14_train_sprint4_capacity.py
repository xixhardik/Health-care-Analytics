"""Sprint 4 - model capacity experiment (U-Net width 16 -> 32).

Controlled experiment. The variable under test is **model capacity**: the
encoder width doubles from 16 to 32 base channels (1,963,860 -> 7,848,964
parameters). Everything about the data pipeline, loss, optimiser, schedule,
augmentation, split, seed and validation protocol is held at the Sprint 3
Coverage settings.

CHANGED
    base_channels     16 -> 32          (the variable under test)
    batch_size        8 -> 2            (dependency, not an independent choice)

    The batch size is not a free parameter here. The Sprint 3 audit measured
    width 32 at batch 8 needing ~3,834 MB peak commit charge - 1.5x the
    memory-proven level for this machine - and 80 h for 30 full-coverage
    epochs. At batch 2 the same model needs ~1,823 MB, which is *below* the
    proven width-16/batch-8 level (~2,534 MB), and 30 rotating epochs cost
    ~16.9 h. Batch 2 is therefore the setting that makes width 32 runnable at
    all on this hardware. It is reported as a co-varying dependency of the
    capacity change, not hidden as "unchanged".

UNCHANGED FROM SPRINT 3 COVERAGE
    topology          plain U-Net, depth 4, bilinear upsampling, BatchNorm
    sampler           rotating shard sampler, ~2,284 slices/epoch
    coverage          100% of the 9,128 training slices, full by epoch 4
    optimiser         Adam, lr 1e-3
    schedule          CosineAnnealingLR(T_max=30)
    loss              CrossEntropy + soft Dice, equal weight
    augmentation      translate +/-5% and intensity jitter, train split only
    preprocessing     Sprint 1 slices, 352x256 at 1.0 mm/px
    seed              42
    split             Sprint 1 patient-level split
    validation        the same 800-slice validation subset as Sprint 2 and 3
    classes           4 (background / vertebra / IVD / spinal canal)
    early stopping    monitor val foreground Dice, patience 6, min_delta 0.001

DIFFERENT BY DESIGN
    initialisation    random. NOT resumed from Sprint 3, NOT warm-started from
                      Sprint 2 - and it could not be anyway, since the layer
                      shapes differ. Stated explicitly because the experimental
                      control depends on it.

The test set is never loaded here. :class:`Trainer` rejects a 'test' loader.

Outputs
-------
    outputs/checkpoints/sprint4_capacity/   last.pt, best_val_dice.pt,
                                            best_val_loss.pt, epochs/, history.*
    outputs/reports/sprint4_capacity/       training_config.json,
                                            training_summary.json,
                                            smoke_test.json

Usage
-----
    python scripts/14_train_sprint4_capacity.py --smoke
    python scripts/14_train_sprint4_capacity.py --max-epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
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
    load_checkpoint,
    is_resumable,
)
from src.models.unet import build_unet  # noqa: E402
from src.utils.paths import OUTPUTS_DIR, RANDOM_SEED, ensure_dirs  # noqa: E402
from src.utils.reporting import save_json  # noqa: E402

EXPERIMENT = "sprint4_capacity"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / EXPERIMENT
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT
SMOKE_CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / f"{EXPERIMENT}_smoke"

# --- the variable under test -----------------------------------------------
BASE_CHANNELS = 32           # Sprint 3 used 16
BATCH_SIZE = 2               # dependency of the capacity change (see docstring)

# --- held at Sprint 3 Coverage values --------------------------------------
LEARNING_RATE = 1e-3
DEPTH = 4
DOWNSAMPLE = 1
SLICES_PER_EPOCH = 2284
MAX_VAL_SLICES = 800
NUM_WORKERS = 0

#: Peak commit charge of the width-16 / batch-8 configuration, which completed
#: three multi-hour runs on this machine. Used as the memory reference point,
#: exactly as in the Sprint 3 audit.
PROVEN_COMMIT_MB = 2550.0

#: Reference points carried forward for the report. Measured, not assumed.
SPRINT3 = {
    "val_fg_dice": 0.89831,
    "test_macro_fg_dice": 0.90001,
    "test_vertebra_dice": 0.91663,
    "test_ivd_dice": 0.87827,
    "test_canal_dice": 0.90514,
    "disc_indexing_pct_evaluate_unet": 84.42,
    "disc_indexing_pct_taxonomy": 84.08,
    "pfirrmann_qwk": 0.6317,
    "base_channels": 16,
    "batch_size": 8,
    "params": 1963860,
    "epochs_completed": 29,
    "wall_hours": 5.88,
    "mean_epoch_seconds": 730.2,
}
SPRINT2_EXTENDED = {
    "val_fg_dice": 0.8985,
    "test_macro_fg_dice": 0.8979,
    "test_vertebra_dice": 0.9133,
    "test_ivd_dice": 0.8779,
    "test_canal_dice": 0.9026,
    "disc_indexing_pct_evaluate_unet": 82.6,
    "pfirrmann_qwk": 0.6375,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--restore-criterion", default="val_dice",
                        choices=["val_dice", "val_loss"])
    parser.add_argument("--smoke", action="store_true",
                        help="Run the short feasibility check instead of training.")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-train-slices", type=int, default=64)
    parser.add_argument("--smoke-val-slices", type=int, default=32)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Memory measurement (commit charge, matching the Sprint 3 audit)
# ---------------------------------------------------------------------------


def memory_snapshot() -> dict:
    """Snapshot this process's memory.

    Commit charge ('pagefile' on Windows) is the primary metric. Resident-set
    size only counts pages currently in physical RAM, so under memory pressure
    it understates what was actually allocated - during the Sprint 3 audit a
    width-64 model appeared to use *less* RSS than width 32 because it was
    being paged out.
    """
    import psutil

    process = psutil.Process(os.getpid())
    info = process.memory_info()
    system = psutil.virtual_memory()
    return {
        "rss_mb": round(info.rss / 1e6, 1),
        "commit_mb": round(getattr(info, "pagefile", info.rss) / 1e6, 1),
        "peak_rss_mb": round(getattr(info, "peak_wset", info.rss) / 1e6, 1),
        "peak_commit_mb": round(getattr(info, "peak_pagefile", info.rss) / 1e6, 1),
        "system_available_mb": round(system.available / 1e6, 1),
        "system_percent_used": system.percent,
    }


# ---------------------------------------------------------------------------
# Trainer that also logs cumulative coverage
# ---------------------------------------------------------------------------


class CoverageLoggingTrainer(Trainer):
    """``Trainer`` that adds a cumulative-coverage column to the epoch log.

    Sprint 4 is required to log cumulative training coverage per epoch. The base
    ``Trainer`` already records ``unique_train_slices_seen``; this subclass
    derives the percentage from it at persist time. Subclassing keeps
    ``src/models/training.py`` byte-for-byte as Sprint 3 left it, so the Sprint 3
    run stays exactly reproducible.
    """

    def __init__(self, *args, train_slices_available: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_slices_available = max(1, int(train_slices_available))

    def _persist_history(self) -> None:
        for row in self.state.history:
            row["cumulative_coverage_pct"] = round(
                100 * row["unique_train_slices_seen"] / self.train_slices_available, 2
            )
        super()._persist_history()


# ---------------------------------------------------------------------------
# Shared construction
# ---------------------------------------------------------------------------


def build_validation_loader(index, seed: int, max_slices: int, batch_size: int):
    """The fixed validation loader.

    Capped and seeded exactly as Sprint 2 and Sprint 3 did, so the validation
    Dice is computed on the same 800 slices and stays directly comparable to
    Sprint 3's 0.89831.
    """
    rows = _sample_per_patient(
        index[index["split"] == "val"], max_slices, np.random.default_rng(seed)
    )
    loader = DataLoader(
        LumbarSliceDataset(rows, augment=False, downsample=DOWNSAMPLE, seed=seed),
        batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
    )
    return rows, loader


def make_train_loader_factory(index, seed: int, slices_per_epoch: int,
                              batch_size: int):
    """Return a callable building the training loader for a given epoch."""

    def factory(epoch: int) -> DataLoader:
        rows = rotating_epoch_sample(
            index, epoch=epoch, slices_per_epoch=slices_per_epoch, seed=seed
        )
        # Defence in depth: the schedule is verified up front, but assert here
        # too so a future change cannot silently admit non-train slices.
        splits = set(rows["split"].unique())
        if splits != {"train"}:
            raise RuntimeError(f"Epoch {epoch} subset contains splits {splits}")
        dataset = LumbarSliceDataset(
            rows, augment=True, downsample=DOWNSAMPLE, seed=seed * 1000 + epoch
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True,
        )

    return factory


def build_model() -> torch.nn.Module:
    """Width-32 U-Net, random initialisation, channels_last."""
    model = build_unet(
        n_classes=N_CLASSES, base_channels=BASE_CHANNELS, depth=DEPTH
    ).to(memory_format=torch.channels_last)
    return model


# ---------------------------------------------------------------------------
# Phase: smoke / throughput test
# ---------------------------------------------------------------------------


def run_smoke_test(args: argparse.Namespace) -> dict:
    """Short feasibility check. Never touches the test split."""
    print("=" * 78)
    print("SPRINT 4 SMOKE / THROUGHPUT TEST (width 32, batch 2)")
    print("=" * 78)

    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # --- environment ------------------------------------------------------
    print("\n[1] Environment")
    cuda_available = torch.cuda.is_available()
    print(f"  torch {torch.__version__}  threads={torch.get_num_threads()}")
    record("no CUDA assumed (CPU-only run)", not cuda_available,
           f"cuda.is_available()={cuda_available}, device=cpu")

    baseline_memory = memory_snapshot()
    print(f"  memory baseline: commit {baseline_memory['commit_mb']} MB, "
          f"system available {baseline_memory['system_available_mb']} MB")

    # --- model ------------------------------------------------------------
    print("\n[2] Model initialisation and parameter count")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model()
    counts = model.count_parameters()
    print(f"  {model.describe()}")
    record("model initialises at width 32",
           model.base_channels == BASE_CHANNELS and model.depth == DEPTH,
           f"base_channels={model.base_channels}, depth={model.depth}, "
           f"bilinear={model.bilinear}")
    record("parameter count is 4x Sprint 3 width-16 model",
           counts["total"] > 4 * SPRINT3["params"] * 0.95,
           f"{counts['total']:,} params vs width-16 {SPRINT3['params']:,} "
           f"({counts['total'] / SPRINT3['params']:.2f}x)")
    record("all parameters trainable and finite",
           counts["total"] == counts["trainable"]
           and all(torch.isfinite(p).all().item() for p in model.parameters()),
           f"trainable {counts['trainable']:,}/{counts['total']:,}, no NaN/Inf")

    # --- data -------------------------------------------------------------
    print("\n[3] Data loaders (train + val only, test never loaded)")
    index = load_slice_index()
    assert_no_patient_leakage(index)
    train_available = int((index["split"] == "train").sum())

    smoke_factory = make_train_loader_factory(
        index, args.seed, args.smoke_train_slices, BATCH_SIZE
    )
    smoke_train = smoke_factory(1)
    val_rows, smoke_val = build_validation_loader(
        index, args.seed, args.smoke_val_slices, BATCH_SIZE
    )
    print(f"  smoke train subset: {len(smoke_train.dataset):,} slices, "
          f"{len(smoke_train)} batches of {BATCH_SIZE}")
    print(f"  smoke val subset  : {len(val_rows):,} slices, "
          f"{len(smoke_val)} batches of {BATCH_SIZE}")
    record("batch size 2 produces the expected batch shape",
           len(smoke_train) > 0,
           f"{len(smoke_train)} train batches, {len(smoke_val)} val batches")

    loaded_splits = set(smoke_train.dataset.index["split"].unique()) | set(
        val_rows["split"].unique()
    )
    record("test split never loaded", "test" not in loaded_splits,
           f"splits loaded: {sorted(loaded_splits)}")

    # --- one manual step: loss finite, gradients valid ---------------------
    print("\n[4] Single training step: loss and gradients")
    criterion = DiceCrossEntropyLoss(n_classes=N_CLASSES)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    batch = next(iter(smoke_train))
    image = batch["image"].to(memory_format=torch.channels_last)
    mask = batch["mask"]
    print(f"  batch tensors: image {tuple(image.shape)} {image.dtype}, "
          f"mask {tuple(mask.shape)} {mask.dtype}")
    record("input batch is (2, 1, 352, 256)",
           tuple(image.shape) == (BATCH_SIZE, 1, 352, 256),
           str(tuple(image.shape)))

    model.train()
    logits = model(image)
    parts = criterion(logits, mask)
    loss_value = float(parts["loss"].detach())
    record("forward pass output shape correct",
           tuple(logits.shape) == (BATCH_SIZE, N_CLASSES, 352, 256),
           str(tuple(logits.shape)))
    record("loss is finite", np.isfinite(loss_value),
           f"loss={loss_value:.5f} (ce={float(parts['ce']):.5f}, "
           f"dice={float(parts['dice']):.5f})")

    optimiser.zero_grad()
    parts["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)))
    n_without_grad = sum(1 for p in model.parameters() if p.grad is None)
    all_finite = all(torch.isfinite(g).all().item() for g in grads)
    record("every parameter received a gradient", n_without_grad == 0,
           f"{len(grads)} tensors with grads, {n_without_grad} without")
    record("gradients finite and non-zero",
           all_finite and grad_norm > 0,
           f"global grad L2 norm = {grad_norm:.5f}")
    optimiser.step()
    record("optimiser step leaves parameters finite",
           all(torch.isfinite(p).all().item() for p in model.parameters()),
           "no NaN/Inf after one Adam step")

    # --- throughput -------------------------------------------------------
    print("\n[5] Throughput (after warm-up)")
    n_steps = 6
    start = time.perf_counter()
    for step, batch in enumerate(smoke_train, start=1):
        image = batch["image"].to(memory_format=torch.channels_last)
        out = model(image)
        loss = criterion(out, batch["mask"])["loss"]
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step >= n_steps:
            break
    train_s_per_step = (time.perf_counter() - start) / n_steps
    train_img_per_s = BATCH_SIZE / train_s_per_step
    after_train_memory = memory_snapshot()

    model.eval()
    n_infer = 6
    with torch.inference_mode():
        start = time.perf_counter()
        for step, batch in enumerate(smoke_val, start=1):
            model(batch["image"].to(memory_format=torch.channels_last))
            if step >= n_infer:
                break
        infer_s_per_step = (time.perf_counter() - start) / n_infer
    infer_img_per_s = BATCH_SIZE / infer_s_per_step

    train_batches = SLICES_PER_EPOCH // BATCH_SIZE
    val_batches = MAX_VAL_SLICES // BATCH_SIZE
    projected_epoch_s = train_batches * train_s_per_step + val_batches * infer_s_per_step
    projected_30ep_h = 30 * projected_epoch_s / 3600
    print(f"  train: {train_s_per_step:.4f} s/step, {train_img_per_s:.2f} img/s")
    print(f"  infer: {infer_s_per_step:.4f} s/step, {infer_img_per_s:.2f} img/s")
    print(f"  projected epoch ({train_batches} train + {val_batches} val batches): "
          f"{projected_epoch_s / 60:.1f} min")
    print(f"  projected 30 epochs: {projected_30ep_h:.1f} h "
          f"(audit estimate 16.9 h)")
    record("throughput measured and finite",
           np.isfinite(train_s_per_step) and train_s_per_step > 0,
           f"{train_img_per_s:.2f} img/s train, {infer_img_per_s:.2f} img/s infer")

    # --- memory -----------------------------------------------------------
    print("\n[6] CPU memory and paging")
    peak_commit = after_train_memory["peak_commit_mb"]
    commit_minus_rss = round(
        after_train_memory["commit_mb"] - after_train_memory["rss_mb"], 1
    )
    print(f"  peak commit charge : {peak_commit} MB "
          f"({peak_commit / PROVEN_COMMIT_MB:.2f}x the proven "
          f"{PROVEN_COMMIT_MB:.0f} MB level)")
    print(f"  commit - rss       : {commit_minus_rss} MB (paging indicator)")
    print(f"  system available   : {after_train_memory['system_available_mb']} MB "
          f"(was {baseline_memory['system_available_mb']} MB)")
    record("peak commit within the memory-proven envelope",
           peak_commit <= PROVEN_COMMIT_MB,
           f"{peak_commit} MB <= {PROVEN_COMMIT_MB:.0f} MB proven")
    record("system did not run out of memory",
           after_train_memory["system_available_mb"] > 150,
           f"{after_train_memory['system_available_mb']} MB available")

    # --- validation + checkpoint writing through the real Trainer ---------
    print(f"\n[7] {args.smoke_epochs} short epochs through the real Trainer "
          f"(validation + checkpointing)")
    if SMOKE_CHECKPOINT_DIR.exists():
        for stale in SMOKE_CHECKPOINT_DIR.rglob("*.pt"):
            stale.unlink()
    smoke_model = build_model()
    smoke_criterion = DiceCrossEntropyLoss(n_classes=N_CLASSES)
    smoke_optimiser = torch.optim.Adam(smoke_model.parameters(), lr=LEARNING_RATE)
    smoke_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        smoke_optimiser, T_max=args.max_epochs
    )
    smoke_trainer = CoverageLoggingTrainer(
        model=smoke_model,
        loaders={"val": smoke_val},
        criterion=smoke_criterion,
        optimiser=smoke_optimiser,
        scheduler=smoke_scheduler,
        output_dir=SMOKE_CHECKPOINT_DIR,
        config={"experiment": f"{EXPERIMENT}_smoke", "purpose": "feasibility only"},
        early_stopping=EarlyStopping(patience=args.patience,
                                     min_delta=args.min_delta),
        state=TrainingState(),
        train_loader_factory=smoke_factory,
        train_slices_available=train_available,
    )
    smoke_state = smoke_trainer.fit(max_epochs=args.smoke_epochs)

    record("validation ran and produced a finite Dice",
           bool(smoke_state.history)
           and np.isfinite(smoke_state.history[-1]["val_fg_dice"]),
           f"{len(smoke_state.history)} epochs, last val_fg_dice="
           f"{smoke_state.history[-1]['val_fg_dice']:.5f}")
    record("all logged epoch losses finite",
           all(np.isfinite(r["train_loss"]) and np.isfinite(r["val_loss"])
               for r in smoke_state.history),
           ", ".join(f"ep{r['epoch']} train={r['train_loss']:.4f} "
                     f"val={r['val_loss']:.4f}" for r in smoke_state.history))
    required_columns = {
        "epoch", "train_loss", "val_loss", "val_fg_dice", "val_dice_vertebra",
        "val_dice_intervertebral_disc", "val_dice_spinal_canal",
        "learning_rate", "epoch_seconds", "unique_train_slices_seen",
        "cumulative_coverage_pct",
    }
    logged = set(smoke_state.history[-1]) if smoke_state.history else set()
    record("every required per-epoch field is logged",
           required_columns <= logged,
           f"missing: {sorted(required_columns - logged) or 'none'}")

    # --- checkpoint integrity --------------------------------------------
    print("\n[8] Checkpoint contents and round-trip load")
    record("history.csv and history.json written",
           smoke_trainer.history_csv.exists() and smoke_trainer.history_json.exists(),
           f"{smoke_trainer.history_csv.name}, {smoke_trainer.history_json.name}")
    record("last.pt / best_val_dice.pt written",
           smoke_trainer.last_path.exists() and smoke_trainer.best_dice_path.exists(),
           f"last.pt {smoke_trainer.last_path.stat().st_size / 1e6:.1f} MB")
    record("per-epoch checkpoint archive written",
           smoke_trainer.epoch_path(1).exists(),
           f"{len(list((SMOKE_CHECKPOINT_DIR / 'epochs').glob('*.pt')))} files")

    checkpoint = load_checkpoint(smoke_trainer.last_path)
    present = {
        "optimiser_state": checkpoint.get("optimiser_state") is not None,
        "scheduler_state": checkpoint.get("scheduler_state") is not None,
        "rng.torch": (checkpoint.get("rng") or {}).get("torch") is not None,
        "rng.numpy": (checkpoint.get("rng") or {}).get("numpy") is not None,
        "epoch": checkpoint.get("training_state", {}).get("epoch") is not None,
        "best_val_dice": checkpoint.get("metadata", {}).get("best_val_dice")
        is not None,
    }
    record("checkpoint carries optimiser, scheduler, RNG, epoch, best metric",
           all(present.values()) and is_resumable(checkpoint),
           ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in present.items()))

    reloaded = build_model()
    reloaded.load_state_dict(checkpoint["state_dict"])
    max_delta = max(
        float((a - b).abs().max())
        for a, b in zip(reloaded.state_dict().values(),
                        smoke_model.state_dict().values())
        if a.dtype.is_floating_point
    )
    record("weights round-trip into a freshly built width-32 model",
           max_delta == 0.0, f"max |delta| = {max_delta:.2e}")
    record("checkpoint architecture records width 32",
           checkpoint["architecture"]["base_channels"] == BASE_CHANNELS,
           json.dumps(checkpoint["architecture"]))

    # --- validation metric is invariant to batch size ---------------------
    # Justifies validating at batch 2 rather than Sprint 3's batch 8: the
    # metric must stay comparable to Sprint 3's 0.89831.
    print("\n[9] Validation metric invariance to batch size")
    from src.models.training import run_epoch

    smoke_model.eval()
    _, val_b8 = build_validation_loader(index, args.seed, args.smoke_val_slices, 8)
    result_b2 = run_epoch(smoke_model, smoke_val, smoke_criterion, None, log_every=0)
    result_b8 = run_epoch(smoke_model, val_b8, smoke_criterion, None, log_every=0)
    dice_b2 = result_b2["metrics"]["aggregate"]["macro_foreground"]["dice"]
    dice_b8 = result_b8["metrics"]["aggregate"]["macro_foreground"]["dice"]
    record("validation Dice identical at batch 2 and batch 8",
           abs(dice_b2 - dice_b8) < 1e-9,
           f"batch2={dice_b2:.8f}, batch8={dice_b8:.8f}, "
           f"delta={abs(dice_b2 - dice_b8):.2e}")

    # --- summary ----------------------------------------------------------
    n_pass = sum(1 for c in checks.values() if c["passed"])
    n_fail = len(checks) - n_pass
    payload = {
        "experiment": f"{EXPERIMENT}_smoke",
        "purpose": "feasibility and throughput check before the full run",
        "test_set_used": False,
        "device": "cpu",
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": cuda_available,
        "architecture": model.describe(),
        "base_channels": BASE_CHANNELS,
        "depth": DEPTH,
        "parameters": counts,
        "params_vs_sprint3": round(counts["total"] / SPRINT3["params"], 3),
        "batch_size": BATCH_SIZE,
        "single_step": {
            "loss": round(loss_value, 6),
            "global_grad_l2_norm": round(grad_norm, 6),
        },
        "throughput": {
            "train_s_per_step": round(train_s_per_step, 4),
            "train_img_per_s": round(train_img_per_s, 3),
            "infer_s_per_step": round(infer_s_per_step, 4),
            "infer_img_per_s": round(infer_img_per_s, 3),
            "train_batches_per_epoch": train_batches,
            "val_batches_per_epoch": val_batches,
            "projected_epoch_minutes": round(projected_epoch_s / 60, 1),
            "projected_30_epoch_hours": round(projected_30ep_h, 1),
            "audit_estimate_30_epoch_hours": 16.9,
        },
        "memory": {
            "baseline": baseline_memory,
            "after_training_steps": after_train_memory,
            "peak_commit_mb": peak_commit,
            "commit_minus_rss_mb": commit_minus_rss,
            "proven_commit_mb": PROVEN_COMMIT_MB,
            "peak_commit_vs_proven": round(peak_commit / PROVEN_COMMIT_MB, 3),
            "audit_predicted_peak_commit_mb": 1822.9,
        },
        "smoke_epochs": [
            {k: r[k] for k in (
                "epoch", "train_loss", "val_loss", "val_fg_dice",
                "unique_train_slices_seen", "cumulative_coverage_pct",
                "epoch_seconds")}
            for r in smoke_state.history
        ],
        "validation_batch_size_invariance": {
            "dice_batch_2": dice_b2,
            "dice_batch_8": dice_b8,
            "absolute_difference": abs(dice_b2 - dice_b8),
        },
        "checks": checks,
        "n_passed": n_pass,
        "n_failed": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(payload, REPORT_DIR / "smoke_test.json")

    print("\n" + "=" * 78)
    print(f"SMOKE TEST {'PASSED' if n_fail == 0 else 'FAILED'}  "
          f"({n_pass} passed, {n_fail} failed)")
    print("=" * 78)
    for name, result in checks.items():
        if not result["passed"]:
            print(f"  FAIL {name}: {result['detail']}")
    print(f"  measured projection: {projected_30ep_h:.1f} h for 30 epochs "
          f"({projected_epoch_s / 60:.1f} min/epoch)")
    print(f"  peak commit: {peak_commit} MB "
          f"({peak_commit / PROVEN_COMMIT_MB:.2f}x proven)")
    print(f"  -> {REPORT_DIR / 'smoke_test.json'}")
    if n_fail == 0:
        print("\nSafe to launch the full run:")
        print("  python scripts/14_train_sprint4_capacity.py --max-epochs 30")
    return payload


# ---------------------------------------------------------------------------
# Phase: full training
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"cuda={torch.cuda.is_available()}")
    print(f"experiment: {EXPERIMENT}  (controlled variable: model capacity, "
          f"width {SPRINT3['base_channels']} -> {BASE_CHANNELS})")
    print(f"checkpoints -> {CHECKPOINT_DIR}")

    index = load_slice_index()
    assert_no_patient_leakage(index)
    train_available = int((index["split"] == "train").sum())

    # --- verify the sampler before committing to a long run ---------------
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

    # --- loaders ----------------------------------------------------------
    validation_rows, val_loader = build_validation_loader(
        index, args.seed, MAX_VAL_SLICES, BATCH_SIZE
    )
    print(f"\nValidation: {len(validation_rows):,} slices, "
          f"{validation_rows['patient_id'].nunique()} patients (fixed, no augment)")
    train_loader_factory = make_train_loader_factory(
        index, args.seed, SLICES_PER_EPOCH, BATCH_SIZE
    )

    # --- model: RANDOM initialisation ------------------------------------
    model = build_model()
    counts = model.count_parameters()
    criterion = DiceCrossEntropyLoss(n_classes=N_CLASSES)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.max_epochs
    )
    print(f"\nModel: {model.describe()}  [random initialisation]")
    print(f"  capacity vs Sprint 3: {counts['total']:,} params vs "
          f"{SPRINT3['params']:,} "
          f"({counts['total'] / SPRINT3['params']:.2f}x)")

    smoke = None
    smoke_path = REPORT_DIR / "smoke_test.json"
    if smoke_path.exists():
        with smoke_path.open(encoding="utf-8") as handle:
            smoke = json.load(handle)
        print(f"  smoke test on record: {smoke.get('verdict')} "
              f"({smoke.get('n_passed')} checks passed), projected "
              f"{smoke.get('throughput', {}).get('projected_30_epoch_hours')} h")

    config = {
        "experiment": EXPERIMENT,
        "controlled_variable": (
            f"model capacity (U-Net base channels "
            f"{SPRINT3['base_channels']} -> {BASE_CHANNELS})"
        ),
        "changed": {
            "base_channels": {"from": SPRINT3["base_channels"], "to": BASE_CHANNELS},
            "parameters": {"from": SPRINT3["params"], "to": counts["total"],
                           "ratio": round(counts["total"] / SPRINT3["params"], 3)},
            "batch_size": {
                "from": SPRINT3["batch_size"], "to": BATCH_SIZE,
                "status": "co-varying dependency, not an independent choice",
                "rationale": (
                    "Width 32 at batch 8 was measured in the Sprint 3 audit at "
                    "~3,834 MB peak commit (1.5x the memory-proven level) and "
                    "80 h for 30 full-coverage epochs. At batch 2 it needs "
                    "~1,823 MB, below the proven width-16/batch-8 level of "
                    "~2,534 MB, and ~16.9 h for 30 rotating epochs. Batch 2 is "
                    "what makes width 32 runnable on this hardware."
                ),
            },
        },
        "unchanged_from_sprint3_coverage": {
            "topology": "plain U-Net, BatchNorm, bilinear upsampling",
            "depth": DEPTH,
            "bilinear": True,
            "sampler": "rotating shard sampler",
            "slices_per_epoch": SLICES_PER_EPOCH,
            "coverage_pct_over_run": coverage["coverage_pct"],
            "epochs_to_full_coverage": coverage["epochs_to_full_coverage"],
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
            "early_stopping": {
                "monitor": "val_fg_dice",
                "patience": args.patience,
                "min_delta": args.min_delta,
            },
        },
        "different_by_design": {
            "initialisation": (
                "random. NOT resumed from Sprint 3, NOT warm-started from "
                "Sprint 2 - the layer shapes differ, and warm-starting would "
                "confound the capacity variable."
            ),
            "validation_batch_size": (
                f"{BATCH_SIZE} rather than Sprint 3's {SPRINT3['batch_size']}. "
                f"Verified in the smoke test to leave the validation Dice "
                f"bit-identical: BatchNorm runs in eval mode and the confusion "
                f"matrix accumulates over the full 800 slices, so the metric "
                f"stays directly comparable to Sprint 3's 0.89831."
            ),
        },
        "architecture": model.describe(),
        "parameters": counts,
        "sampler_verification": {
            "leakage_free": leakage["leakage_free"],
            "violations": len(leakage["violations"]),
            "val_test_patients_excluded": leakage["val_test_patients_excluded"],
            "coverage": {
                k: v for k, v in coverage.items()
                if k != "cumulative_coverage_pct_by_epoch"
            },
        },
        "reference_results": {
            "sprint3_coverage": SPRINT3,
            "sprint2_extended": SPRINT2_EXTENDED,
        },
        "smoke_test": {
            "verdict": (smoke or {}).get("verdict"),
            "n_passed": (smoke or {}).get("n_passed"),
            "n_failed": (smoke or {}).get("n_failed"),
            "projected_30_epoch_hours":
                (smoke or {}).get("throughput", {}).get("projected_30_epoch_hours"),
            "peak_commit_mb": (smoke or {}).get("memory", {}).get("peak_commit_mb"),
        },
        "test_set_used_during_training": False,
        "device": "cpu",
    }

    trainer = CoverageLoggingTrainer(
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
        train_slices_available=train_available,
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

    epoch_times = [r["epoch_seconds"] for r in state.history] or [0.0]
    summary = {
        "experiment": EXPERIMENT,
        "controlled_variable": config["controlled_variable"],
        "resume": resume_info,
        "architecture": model.describe(),
        "base_channels": BASE_CHANNELS,
        "parameters": counts,
        "batch_size": BATCH_SIZE,
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
        "train_slices_available": train_available,
        "coverage_pct_achieved": round(
            100 * state.unique_slices_seen / train_available, 2
        ),
        "wall_seconds": round(wall_seconds, 1),
        "wall_minutes": round(wall_seconds / 60, 1),
        "wall_hours": round(wall_seconds / 3600, 2),
        "mean_epoch_seconds": round(float(np.mean(epoch_times)), 1),
        "min_epoch_seconds": round(float(np.min(epoch_times)), 1),
        "max_epoch_seconds": round(float(np.max(epoch_times)), 1),
        "restored_checkpoint": {
            "criterion": args.restore_criterion,
            "path": str(trainer.best_dice_path),
            "metadata": restored,
        },
        "comparison_vs_sprint3_coverage": {
            "sprint3_best_val_fg_dice": SPRINT3["val_fg_dice"],
            "sprint4_best_val_fg_dice": state.best_val_dice,
            "delta": round(state.best_val_dice - SPRINT3["val_fg_dice"], 5),
            "exceeded": bool(state.best_val_dice > SPRINT3["val_fg_dice"]),
            "exceeded_by_min_delta": bool(
                state.best_val_dice > SPRINT3["val_fg_dice"] + args.min_delta
            ),
        },
        "comparison_vs_sprint2_extended": {
            "sprint2_extended_best_val_fg_dice": SPRINT2_EXTENDED["val_fg_dice"],
            "delta": round(state.best_val_dice - SPRINT2_EXTENDED["val_fg_dice"], 5),
        },
        "cost_vs_sprint3": {
            "sprint3_mean_epoch_seconds": SPRINT3["mean_epoch_seconds"],
            "sprint4_mean_epoch_seconds": round(float(np.mean(epoch_times)), 1),
            "slowdown_factor": round(
                float(np.mean(epoch_times)) / SPRINT3["mean_epoch_seconds"], 2
            ),
            "sprint3_wall_hours": SPRINT3["wall_hours"],
            "sprint4_wall_hours": round(wall_seconds / 3600, 2),
        },
        "test_set_used_during_training": False,
    }
    save_json(summary, REPORT_DIR / "training_summary.json")

    print("\n" + "=" * 78)
    print("SPRINT 4 CAPACITY TRAINING COMPLETE")
    print("=" * 78)
    print(f"  architecture              : {model.describe()}")
    print(f"  epochs completed          : {state.epoch} / {args.max_epochs}")
    print(f"  stopped early             : {state.stopped_early}")
    print(f"  reason                    : {state.stop_reason}")
    print(f"  unique train slices seen  : {state.unique_slices_seen:,} / "
          f"{train_available:,} ({summary['coverage_pct_achieved']}%)")
    print(f"  best val foreground Dice  : {state.best_val_dice:.5f} "
          f"(epoch {state.best_val_dice_epoch})")
    print(f"  best val loss             : {state.best_val_loss:.5f} "
          f"(epoch {state.best_val_loss_epoch})")
    print(f"  vs Sprint 3 Coverage      : {SPRINT3['val_fg_dice']} -> "
          f"{state.best_val_dice:.5f} "
          f"({summary['comparison_vs_sprint3_coverage']['delta']:+.5f}, "
          f"exceeded={summary['comparison_vs_sprint3_coverage']['exceeded']})")
    print(f"  mean epoch                : "
          f"{summary['mean_epoch_seconds']:.0f}s "
          f"({summary['cost_vs_sprint3']['slowdown_factor']}x Sprint 3)")
    print(f"  wall time                 : {summary['wall_hours']} h")
    print(f"\n  -> {REPORT_DIR / 'training_summary.json'}")
    print(f"  -> {trainer.history_csv}")
    print("\nNext: evaluate ONLY the selected checkpoint on the held-out test set.")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.smoke:
        payload = run_smoke_test(args)
        raise SystemExit(0 if payload["verdict"] == "PASS" else 1)
    run_training(args)


if __name__ == "__main__":
    main()
