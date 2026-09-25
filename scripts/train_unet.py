"""Sprint 2 / Stage A - train the baseline 2-D U-Net.

Reads the Sprint 1 preprocessed slices and the Sprint 1 patient-level split.
**Neither is modified.**

Writes:

    outputs/models/unet_baseline.pt            best checkpoint (by val Dice)
    outputs/models/unet_last.pt                final-epoch checkpoint
    outputs/metrics/training_history.csv       per-epoch losses and metrics
    outputs/metrics/smoke_test.json            smoke-test evidence
    outputs/metrics/training_config.json       exact configuration used

Usage
-----
    python scripts/train_unet.py --smoke-test          # verify plumbing only
    python scripts/train_unet.py                       # real training run
    python scripts/train_unet.py --epochs 20 --base-channels 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.data import (  # noqa: E402
    N_CLASSES,
    build_dataloaders,
    describe_loaders,
    load_slice_index,
)
from src.models.losses import DiceCrossEntropyLoss  # noqa: E402
from src.models.metrics import (  # noqa: E402
    ConfusionAccumulator,
    format_metrics_table,
)
from src.models.unet import build_unet  # noqa: E402
from src.preprocessing.labels import SEMANTIC_CLASSES  # noqa: E402
from src.utils.paths import OUTPUTS_DIR, RANDOM_SEED, ensure_dirs  # noqa: E402
from src.utils.reporting import save_json  # noqa: E402

MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16,
                        help="U-Net width. 16 for CPU; 64 reproduces the paper.")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--downsample", type=int, default=1,
                        help="Integer shrink factor. 1 keeps Sprint 1's 352x256 "
                             "at 1.0 mm/px, which the mm measurements depend on.")
    parser.add_argument("--max-train-slices", type=int, default=2560,
                        help="Slices per epoch, sampled per patient. CPU budget "
                             "lever; use 0 for all of them.")
    parser.add_argument("--max-val-slices", type=int, default=800)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run the Stage A verification checks and exit.")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    """Make the run reproducible as far as CPU training allows."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Smoke test (Stage A requirement)
# ---------------------------------------------------------------------------


def run_smoke_test(args: argparse.Namespace) -> dict:
    """Verify every link in the chain before committing to a long run.

    Checks, in order: tensor shapes, dtypes, label validity, forward pass,
    loss computation, backpropagation (that gradients actually reach the first
    layer and that parameters change), and prediction generation.
    """
    print("=" * 78)
    print("STAGE A SMOKE TEST")
    print("=" * 78)
    results: dict = {"checks": {}, "passed": True}

    def record(name: str, passed: bool, detail) -> None:
        results["checks"][name] = {"passed": bool(passed), "detail": detail}
        results["passed"] = results["passed"] and bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # --- data ------------------------------------------------------------
    index = load_slice_index()
    loaders, frames = build_dataloaders(
        batch_size=args.batch_size,
        downsample=args.downsample,
        augment_train=not args.no_augment,
        num_workers=args.num_workers,
        max_train_slices=min(args.batch_size * 4, 64),
        max_eval_slices=min(args.batch_size * 2, 32),
        seed=args.seed,
        index=index,
    )
    record("slice index loaded", len(index) > 0,
           f"{len(index):,} slices, splits={index['split'].value_counts().to_dict()}")
    record("patient-level split is disjoint", True,
           "verified by assert_no_patient_leakage() inside build_dataloaders")

    batch = next(iter(loaders["train"]))
    image, mask = batch["image"], batch["mask"]

    # --- 1. image tensor shape ------------------------------------------
    expected_hw = (352 // args.downsample, 256 // args.downsample)
    ok = (
        image.ndim == 4
        and image.shape[1] == 1
        and tuple(image.shape[2:]) == expected_hw
    )
    record("image tensor shape", ok,
           f"{tuple(image.shape)} (N,C,H,W), dtype={image.dtype}, "
           f"range=[{image.min():.4f}, {image.max():.4f}]")

    # --- 2. mask tensor shape -------------------------------------------
    ok = mask.ndim == 3 and tuple(mask.shape[1:]) == expected_hw
    record("mask tensor shape", ok,
           f"{tuple(mask.shape)} (N,H,W), dtype={mask.dtype}")

    ok = image.dtype == torch.float32 and mask.dtype == torch.int64
    record("tensor dtypes", ok,
           f"image={image.dtype} (need float32), mask={mask.dtype} (need int64)")

    # --- 3. valid class IDs ---------------------------------------------
    present = sorted(int(v) for v in torch.unique(mask))
    ok = all(0 <= v < N_CLASSES for v in present)
    record("valid class IDs", ok,
           f"present={present}, allowed=0..{N_CLASSES - 1} "
           f"({ {i: SEMANTIC_CLASSES[i] for i in present} })")

    # --- model -----------------------------------------------------------
    model = build_unet(n_classes=N_CLASSES, base_channels=args.base_channels,
                       depth=args.depth)
    model = model.to(memory_format=torch.channels_last)
    record("model built", True, model.describe())

    # --- 4. forward pass -------------------------------------------------
    model.train()
    logits = model(image.to(memory_format=torch.channels_last))
    ok = tuple(logits.shape) == (image.shape[0], N_CLASSES, *expected_hw)
    record("forward pass", ok,
           f"logits {tuple(logits.shape)}, finite={bool(torch.isfinite(logits).all())}")

    # --- 5. loss calculation --------------------------------------------
    criterion = DiceCrossEntropyLoss(n_classes=N_CLASSES)
    loss_parts = criterion(logits, mask)
    loss = loss_parts["loss"]
    ok = torch.isfinite(loss) and loss.item() > 0
    record("loss calculation", ok,
           f"total={loss.item():.4f} (ce={loss_parts['ce'].item():.4f}, "
           f"dice={loss_parts['dice'].item():.4f})")

    # --- 6. backpropagation ---------------------------------------------
    before = model.stem.block[0].weight.detach().clone()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimiser.zero_grad()
    loss.backward()

    first_grad = model.stem.block[0].weight.grad
    grad_ok = first_grad is not None and torch.isfinite(first_grad).all() \
        and first_grad.abs().sum() > 0
    record("backpropagation reaches first layer", bool(grad_ok),
           f"grad norm={first_grad.norm().item():.6f}" if first_grad is not None
           else "no gradient")

    n_without_grad = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is None
    )
    record("all trainable params have gradients", n_without_grad == 0,
           f"{n_without_grad} parameter tensors without a gradient")

    optimiser.step()
    changed = not torch.allclose(before, model.stem.block[0].weight.detach())
    record("optimiser step changes weights", bool(changed),
           f"max abs delta={(model.stem.block[0].weight.detach() - before).abs().max():.8f}")

    # --- 7. prediction generation ---------------------------------------
    model.eval()
    with torch.inference_mode():
        eval_logits = model(image.to(memory_format=torch.channels_last))
        prediction = eval_logits.argmax(dim=1)
    pred_values = sorted(int(v) for v in torch.unique(prediction))
    ok = tuple(prediction.shape) == tuple(mask.shape) and all(
        0 <= v < N_CLASSES for v in pred_values
    )
    record("prediction generation", ok,
           f"prediction {tuple(prediction.shape)}, values={pred_values}")

    # --- metrics plumbing ------------------------------------------------
    accumulator = ConfusionAccumulator(N_CLASSES)
    accumulator.update(prediction, mask)
    summary = accumulator.summary()
    record("metrics computable", np.isfinite(summary["pixel_accuracy"]),
           f"pixel_accuracy={summary['pixel_accuracy']:.4f} on an untrained model "
           f"(expected to be poor - this only checks the metric code runs)")

    # --- short training loop --------------------------------------------
    print(f"\n  Running {args.smoke_epochs} mini-epochs to confirm the loop runs "
          f"and the loss moves ...")
    history = train_loop(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimiser=optimiser,
        epochs=args.smoke_epochs,
        checkpoint_dir=None,
        log_prefix="    ",
    )
    losses = [row["train_loss"] for row in history]
    record("training loop runs", len(history) == args.smoke_epochs,
           f"train_loss per epoch: {[round(v, 4) for v in losses]}")

    results["history"] = history
    results["config"] = {
        "batch_size": args.batch_size,
        "downsample": args.downsample,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "smoke_epochs": args.smoke_epochs,
        "n_train_slices_used": len(frames["train"]),
        "n_val_slices_used": len(frames.get("val", [])),
    }
    results["model"] = model.describe()

    print("\n" + "=" * 78)
    print(f"SMOKE TEST {'PASSED' if results['passed'] else 'FAILED'}")
    print("=" * 78)
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimiser=None,
    *,
    log_prefix: str = "",
    log_every: int = 50,
) -> dict:
    """One pass over ``loader``. Trains if ``optimiser`` is given, else evaluates."""
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
                print(f"{log_prefix}  batch {step}/{len(loader)} "
                      f"loss={totals['loss'] / n_batches:.4f} "
                      f"({elapsed / step:.2f}s/batch)", flush=True)

    summary = accumulator.summary()
    return {
        "loss": totals["loss"] / max(n_batches, 1),
        "ce": totals["ce"] / max(n_batches, 1),
        "dice_loss": totals["dice"] / max(n_batches, 1),
        "seconds": time.perf_counter() - start,
        "metrics": summary,
    }


def train_loop(
    *,
    model: torch.nn.Module,
    loaders: dict,
    criterion,
    optimiser,
    epochs: int,
    checkpoint_dir: Path | None,
    scheduler=None,
    log_prefix: str = "",
) -> list[dict]:
    """Train for ``epochs``, tracking the best validation foreground Dice."""
    history: list[dict] = []
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        print(f"{log_prefix}Epoch {epoch}/{epochs}")
        train_result = run_epoch(model, loaders["train"], criterion, optimiser,
                                 log_prefix=log_prefix)

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_ce": train_result["ce"],
            "train_dice_loss": train_result["dice_loss"],
            "train_seconds": round(train_result["seconds"], 1),
            "train_fg_dice":
                train_result["metrics"]["aggregate"]["macro_foreground"]["dice"],
        }

        if "val" in loaders:
            val_result = run_epoch(model, loaders["val"], criterion, None,
                                   log_prefix=log_prefix, log_every=0)
            val_fg = val_result["metrics"]["aggregate"]["macro_foreground"]
            row.update({
                "val_loss": val_result["loss"],
                "val_seconds": round(val_result["seconds"], 1),
                "val_fg_dice": val_fg["dice"],
                "val_fg_iou": val_fg["iou"],
                "val_fg_precision": val_fg["precision"],
                "val_fg_recall": val_fg["recall"],
            })
            for name, values in val_result["metrics"]["aggregate"]["per_class"].items():
                row[f"val_dice_{name}"] = values["dice"]

            # Selection metric: foreground macro Dice. Background is excluded
            # because it is 94.8% of pixels and would mask real changes.
            if checkpoint_dir is not None and val_fg["dice"] > best_dice:
                best_dice = val_fg["dice"]
                save_checkpoint(model, checkpoint_dir / "unet_baseline.pt",
                                epoch=epoch, val_fg_dice=best_dice)
                row["checkpoint_saved"] = True

        if scheduler is not None:
            scheduler.step()
            row["lr"] = optimiser.param_groups[0]["lr"]

        history.append(row)
        print(f"{log_prefix}  train_loss={row['train_loss']:.4f}"
              + (f" val_loss={row['val_loss']:.4f} "
                 f"val_fg_dice={row['val_fg_dice']:.4f}" if "val_loss" in row else "")
              + f"  ({row['train_seconds']:.0f}s train)", flush=True)

    return history


def save_checkpoint(model, path: Path, **metadata) -> None:
    """Save weights plus the architecture needed to rebuild them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {
                "in_channels": model.in_channels,
                "n_classes": model.n_classes,
                "base_channels": model.base_channels,
                "depth": model.depth,
                "bilinear": model.bilinear,
            },
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"cuda={torch.cuda.is_available()}")

    if args.smoke_test:
        results = run_smoke_test(args)
        save_json(results, METRICS_DIR / "smoke_test.json")
        print(f"\n-> {METRICS_DIR / 'smoke_test.json'}")
        raise SystemExit(0 if results["passed"] else 1)

    # --- real run --------------------------------------------------------
    cap_train = args.max_train_slices or None
    cap_val = args.max_val_slices or None

    loaders, frames = build_dataloaders(
        batch_size=args.batch_size,
        downsample=args.downsample,
        augment_train=not args.no_augment,
        num_workers=args.num_workers,
        max_train_slices=cap_train,
        max_eval_slices=cap_val,
        seed=args.seed,
    )
    print("\nData:")
    print(describe_loaders(loaders, frames))

    model = build_unet(n_classes=N_CLASSES, base_channels=args.base_channels,
                       depth=args.depth).to(memory_format=torch.channels_last)
    print(f"\nModel: {model.describe()}")

    criterion = DiceCrossEntropyLoss(n_classes=N_CLASSES)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "downsample": args.downsample,
        "input_size": [352 // args.downsample, 256 // args.downsample],
        "augment": not args.no_augment,
        "seed": args.seed,
        "loss": "CrossEntropy + soft Dice (equal weight)",
        "optimiser": "Adam",
        "scheduler": "CosineAnnealingLR",
        "model": model.describe(),
        "n_parameters": model.count_parameters(),
        "device": "cpu",
        "memory_format": "channels_last",
        "slices_used": {k: int(len(v)) for k, v in frames.items()},
        "patients_used": {k: int(v["patient_id"].nunique()) for k, v in frames.items()},
        "selection_metric": "validation macro foreground Dice",
        "note": "Slice caps and width 16 are CPU-budget decisions, measured: "
                "3.99 img/s train, 12.93 img/s inference at 352x256 with "
                "channels_last. Sprint 1 data and split are unmodified.",
    }
    save_json(config, METRICS_DIR / "training_config.json")

    print(f"\nTraining {args.epochs} epochs ...")
    started = time.perf_counter()
    history = train_loop(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimiser=optimiser,
        epochs=args.epochs,
        checkpoint_dir=MODELS_DIR,
        scheduler=scheduler,
    )
    total_seconds = time.perf_counter() - started

    save_checkpoint(model, MODELS_DIR / "unet_last.pt", epoch=args.epochs)
    frame = pd.DataFrame(history)
    frame.to_csv(METRICS_DIR / "training_history.csv", index=False)

    print(f"\nFinished in {total_seconds / 60:.1f} min")
    if "val_fg_dice" in frame.columns:
        best = frame.loc[frame["val_fg_dice"].idxmax()]
        print(f"Best epoch: {int(best['epoch'])}  "
              f"val foreground Dice={best['val_fg_dice']:.4f}")
    print(f"  -> {MODELS_DIR / 'unet_baseline.pt'}")
    print(f"  -> {METRICS_DIR / 'training_history.csv'}")
    print("\nRun scripts/evaluate_unet.py next for test-set metrics.")


if __name__ == "__main__":
    main()
