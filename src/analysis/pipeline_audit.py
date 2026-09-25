"""Pipeline audit for Sprint 3 (sections A, C, D).

Reads the current implementation and the artefacts it produced, and reports what
the pipeline *actually* does - architecture, parameter distribution, sampling,
loss, optimiser, schedule, augmentation, throughput and class balance.

Nothing here trains or modifies anything.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.data import LumbarSliceDataset, _sample_per_patient, build_dataloaders
from src.models.losses import DiceCrossEntropyLoss, soft_dice_loss
from src.models.unet import build_unet
from src.preprocessing.labels import SEMANTIC_CLASSES


# ---------------------------------------------------------------------------
# A. Architecture
# ---------------------------------------------------------------------------


def audit_architecture(base_channels: int = 16, depth: int = 4) -> dict:
    """Parameter breakdown of the current U-Net, stage by stage."""
    model = build_unet(n_classes=4, base_channels=base_channels, depth=depth)
    counts = model.count_parameters()

    # Group parameters by top-level module so it is visible where capacity sits.
    by_stage: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        stage = name.split(".")[0]
        by_stage[stage] = by_stage.get(stage, 0) + parameter.numel()

    widths = [base_channels * 2**i for i in range(depth + 1)]
    rows, cols = 352, 256
    spatial = [(rows // 2**i, cols // 2**i) for i in range(depth + 1)]

    return {
        "class": type(model).__name__,
        "in_channels": model.in_channels,
        "n_classes": model.n_classes,
        "base_channels": base_channels,
        "depth": depth,
        "bilinear_upsampling": model.bilinear,
        "size_divisor": model.size_divisor,
        "params_total": counts["total"],
        "params_trainable": counts["trainable"],
        "params_millions": round(counts["total"] / 1e6, 3),
        "params_by_stage": {
            k: {"count": v, "pct": round(100 * v / counts["total"], 1)}
            for k, v in sorted(by_stage.items(), key=lambda kv: -kv[1])
        },
        "encoder_channel_widths": widths,
        "spatial_sizes_per_level": [f"{r}x{c}" for r, c in spatial],
        "bottleneck_spatial": f"{spatial[-1][0]}x{spatial[-1][1]}",
        "normalisation": "BatchNorm2d",
        "activation": "ReLU",
        "notes": [
            "Plain U-Net (Ronneberger topology) with BatchNorm added.",
            "No attention, no residual blocks, no deep supervision, no 3-D convs.",
            "padding=1 on every 3x3 conv, so skip connections align without cropping.",
        ],
    }


def audit_loss() -> dict:
    """Describe the loss actually in use, read from the implementation."""
    loss = DiceCrossEntropyLoss(n_classes=4)
    return {
        "class": type(loss).__name__,
        "formula": "ce_weight * CrossEntropy + dice_weight * (1 - mean soft Dice)",
        "ce_weight": loss.ce_weight,
        "dice_weight": loss.dice_weight,
        "class_weights_used": loss.cross_entropy.weight is not None,
        "dice_ignores_background": loss.ignore_background_in_dice,
        "dice_smooth": inspect.signature(soft_dice_loss).parameters["smooth"].default,
        "dice_averaging": (
            "soft Dice computed per class over the whole batch (sums over N,H,W), "
            "then averaged across classes - so each class contributes equally "
            "regardless of pixel count"
        ),
        "notes": [
            "Cross-entropy is unweighted: every pixel contributes equally, so the "
            "94.8% background majority dominates that term.",
            "The Dice term is what counteracts the imbalance, because it is "
            "normalised per class.",
            "Background is INCLUDED in the Dice average by default, so one of the "
            "four averaged terms is a class that is trivially easy.",
        ],
    }


def audit_optimiser_and_schedule() -> dict:
    """Optimiser and schedule as configured in the two completed runs."""
    return {
        "optimiser": "Adam",
        "learning_rate": 1e-3,
        "betas": "(0.9, 0.999) - PyTorch defaults",
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR",
        "baseline_T_max": 12,
        "extended_T_max": 30,
        "eta_min": 0.0,
        "gradient_clipping": None,
        "notes": [
            "No weight decay and no gradient clipping are applied.",
            "The baseline annealed the LR to ~0 by epoch 12; the extended run "
            "re-cast the cosine over 30 epochs and fast-forwarded it, which is why "
            "epoch 13 saw a warm-restart dip.",
        ],
    }


def audit_augmentation() -> dict:
    """Augmentation policy, read from the dataset implementation."""
    return {
        "applied_to": "training split only",
        "operations": [
            {
                "name": "integer translation",
                "detail": "np.roll by up to +/-5% of each axis, applied identically "
                          "to image and mask",
                "interpolation": "none - pure index shift, so labels are exact",
            },
            {
                "name": "intensity scale and offset",
                "detail": "image * U(0.9, 1.1) + U(-0.05, 0.05), clipped to [0, 1]",
                "interpolation": "n/a",
            },
        ],
        "deliberately_excluded": [
            "horizontal / vertical flips - a sagittal slice has fixed anatomical "
            "orientation (superior up, anterior left); flipping would create "
            "images that cannot occur and would teach the model that "
            "anterior/posterior is irrelevant, when the spinal canal is defined "
            "by being posterior to the vertebral bodies",
            "rotation, elastic deformation, scaling - would all require "
            "interpolating the mask",
        ],
        "notes": [
            "np.roll wraps around rather than padding, so a large shift would move "
            "anatomy from one edge to the opposite one. At <=5% the shifted band is "
            "background in practice, but this is a latent sharp edge in the policy.",
            "No geometric augmentation that changes scale is applied, which is "
            "consistent with Sprint 1's decision to normalise physical scale to "
            "1.0 mm/px.",
        ],
    }


# ---------------------------------------------------------------------------
# C. Data utilisation
# ---------------------------------------------------------------------------


def audit_data_utilisation(
    index: pd.DataFrame, *, max_train_slices: int = 2560, seed: int = 42
) -> dict:
    """Explain exactly which slices are used and why the rest are not."""
    train = index[index["split"] == "train"]
    rng = np.random.default_rng(seed)
    sampled = _sample_per_patient(train, max_train_slices, rng)

    per_patient_available = train.groupby("patient_id").size()
    per_patient_sampled = sampled.groupby("patient_id").size()

    return {
        "total_slices": int(len(index)),
        "split_sizes": index["split"].value_counts().to_dict(),
        "train_slices_available": int(len(train)),
        "train_patients": int(train["patient_id"].nunique()),
        "train_series": int(train["image_id"].nunique()),
        "max_train_slices_configured": max_train_slices,
        "train_slices_used_per_epoch": int(len(sampled)),
        "fraction_of_train_used_per_epoch": round(len(sampled) / len(train), 4),
        "slices_never_seen": int(len(train) - len(sampled)),
        "per_patient_available": {
            "min": int(per_patient_available.min()),
            "median": float(per_patient_available.median()),
            "max": int(per_patient_available.max()),
        },
        "per_patient_sampled": {
            "min": int(per_patient_sampled.min()),
            "median": float(per_patient_sampled.median()),
            "max": int(per_patient_sampled.max()),
        },
        "sampling_function": "src.models.data._sample_per_patient",
        "sampling_strategy": (
            "cap // n_patients slices are drawn without replacement from each "
            "training patient (2560 // 152 = 16), then the remainder is topped up "
            "at random from whatever is left"
        ),
        "why_capped": [
            "The cap is a CPU wall-clock decision, not a data or method decision. "
            "Measured training throughput is ~3.9 img/s at 352x256 width 16, so a "
            "full 9,128-slice epoch costs ~39 min of training plus validation.",
            "At 2,560 slices an epoch costs ~11-13 min, which made a 12-epoch run "
            "fit in ~2.6 h and a 30-epoch run in ~3.9 h.",
            "The cap is passed as max_train_slices; nothing in the architecture, "
            "split or preprocessing requires it.",
        ],
        "critical_flaw": (
            "The sampling RNG is seeded once per run, not per epoch, and the "
            "DataLoader is constructed once before training. The SAME 2,560 slices "
            "are therefore reused for every epoch - shuffling only reorders them. "
            "The remaining 6,568 training slices (72% of the available training "
            "data) were never seen by either completed run."
        ),
        "leakage_status": (
            "No leakage. Sampling draws exclusively from rows where split == "
            "'train', and build_dataloaders asserts the splits are patient-disjoint "
            "on every construction."
        ),
    }


def verify_sampling_determinism(
    index: pd.DataFrame, *, max_train_slices: int = 2560, seed: int = 42
) -> dict:
    """Confirm the current sampler returns an identical subset on every call."""
    train = index[index["split"] == "train"]
    first = _sample_per_patient(train, max_train_slices, np.random.default_rng(seed))
    second = _sample_per_patient(train, max_train_slices, np.random.default_rng(seed))
    return {
        "identical_across_calls": bool(
            set(first["slice_id"]) == set(second["slice_id"])
        ),
        "n_slices": int(len(first)),
        "implication": (
            "Because the subset is a deterministic function of the seed alone, "
            "re-running or resuming re-selects exactly the same 2,560 slices. This "
            "is good for reproducibility and bad for coverage."
        ),
    }


# ---------------------------------------------------------------------------
# D. Class balance for the loss analysis
# ---------------------------------------------------------------------------


def measure_class_balance(
    index: pd.DataFrame, *, split: str = "train", sample_size: int | None = 400,
    seed: int = 42,
) -> dict:
    """Measure per-class pixel frequency directly from the preprocessed slices.

    The Sprint 1 report gives dataset-wide figures; this recomputes them on the
    split that the loss actually sees, and separately on the capped subset, so
    the loss analysis rests on the same pixels the optimiser saw.
    """
    from src.utils.paths import PROJECT_ROOT

    subset = index[index["split"] == split]
    if sample_size is not None and len(subset) > sample_size:
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(subset), size=sample_size, replace=False)
        subset = subset.iloc[sorted(positions)]

    counts = np.zeros(len(SEMANTIC_CLASSES), dtype=np.int64)
    n_slices_with_class = np.zeros(len(SEMANTIC_CLASSES), dtype=np.int64)

    for _, row in subset.iterrows():
        with np.load(PROJECT_ROOT / row["npz_path"]) as bundle:
            mask = bundle["mask"]
        values, class_counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, class_counts):
            counts[int(value)] += int(count)
            n_slices_with_class[int(value)] += 1

    total = int(counts.sum())
    return {
        "split": split,
        "n_slices_measured": int(len(subset)),
        "total_pixels": total,
        "per_class": {
            SEMANTIC_CLASSES[i]: {
                "pixels": int(counts[i]),
                "pct_of_all_pixels": round(100 * counts[i] / total, 4),
                "slices_containing_class": int(n_slices_with_class[i]),
                "pct_slices_containing": round(
                    100 * n_slices_with_class[i] / len(subset), 1
                ),
                "inverse_frequency_weight": round(float(total / max(counts[i], 1)), 1),
            }
            for i in range(len(SEMANTIC_CLASSES))
        },
        "foreground_pct": round(100 * (total - counts[0]) / total, 3),
        "imbalance_ratio_background_to_disc": round(
            float(counts[0] / max(counts[2], 1)), 1
        ),
    }


def probe_loss_term_magnitudes(
    index: pd.DataFrame, *, n_batches: int = 6, batch_size: int = 8, seed: int = 42
) -> dict:
    """Measure the relative size of the CE and Dice terms on real data.

    Two regimes are probed, because the balance between the terms changes
    completely as the model learns:

    * an **untrained** model - what the optimiser sees at initialisation
    * the **trained extended** checkpoint - what it sees at convergence

    If one term becomes negligible at convergence, it has stopped contributing
    gradient and the weighting deserves revisiting.
    """
    from src.utils.paths import OUTPUTS_DIR

    subset = index[index["split"] == "train"]
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(subset), size=n_batches * batch_size, replace=False)
    dataset = LumbarSliceDataset(subset.iloc[sorted(positions)], augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = DiceCrossEntropyLoss(n_classes=4)
    results: dict = {}

    checkpoints = {
        "untrained": None,
        "trained_extended": OUTPUTS_DIR / "checkpoints" / "sprint2_extended"
                            / "best_val_dice.pt",
    }

    for label, path in checkpoints.items():
        model = build_unet(n_classes=4, base_channels=16, depth=4)
        if path is not None:
            if not Path(path).exists():
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
        model = model.to(memory_format=torch.channels_last).eval()

        ce_values, dice_values = [], []
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(memory_format=torch.channels_last)
                parts = criterion(model(image), batch["mask"])
                ce_values.append(float(parts["ce"]))
                dice_values.append(float(parts["dice"]))

        ce_mean = float(np.mean(ce_values))
        dice_mean = float(np.mean(dice_values))
        results[label] = {
            "ce_mean": round(ce_mean, 5),
            "dice_term_mean": round(dice_mean, 5),
            "total_mean": round(ce_mean + dice_mean, 5),
            "dice_share_of_total": round(dice_mean / (ce_mean + dice_mean), 4),
            "ce_share_of_total": round(ce_mean / (ce_mean + dice_mean), 4),
        }

    return {
        "n_batches": n_batches,
        "batch_size": batch_size,
        "regimes": results,
        "interpretation": (
            "At convergence the cross-entropy term shrinks far faster than the "
            "Dice term, because CE is dominated by the 94.8% background pixels "
            "that become easy. The Dice term therefore supplies most of the "
            "remaining gradient - which is the intended behaviour, but it also "
            "means the equal 1:1 weighting is effectively Dice-dominated late in "
            "training."
        ),
    }


# ---------------------------------------------------------------------------
# Throughput / resource measurement
# ---------------------------------------------------------------------------


def measure_runtime_resources(index: pd.DataFrame, *, n_batches: int = 6) -> dict:
    """Measure loader throughput, CPU utilisation and memory during real steps."""
    import os
    import time

    import psutil

    process = psutil.Process(os.getpid())
    virtual = psutil.virtual_memory()

    loaders, _ = build_dataloaders(
        batch_size=8, num_workers=0, max_train_slices=8 * (n_batches + 2),
        max_eval_slices=8, seed=42, index=index,
    )
    loader = loaders["train"]

    # Loader-only throughput.
    iterator = iter(loader)
    next(iterator)
    start = time.perf_counter()
    count = 0
    for _ in range(n_batches):
        try:
            next(iterator)
        except StopIteration:
            break
        count += 1
    loader_seconds = (time.perf_counter() - start) / max(count, 1)

    # Full training step with CPU sampling.
    model = build_unet(n_classes=4, base_channels=16, depth=4).to(
        memory_format=torch.channels_last
    )
    criterion = DiceCrossEntropyLoss(n_classes=4)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    batch = next(iter(loader))
    image = batch["image"].to(memory_format=torch.channels_last)
    mask = batch["mask"]

    def step():
        parts = criterion(model(image), mask)
        optimiser.zero_grad()
        parts["loss"].backward()
        optimiser.step()

    step()  # warm-up
    rss_before = process.memory_info().rss
    process.cpu_percent(interval=None)
    start = time.perf_counter()
    for _ in range(n_batches):
        step()
    step_seconds = (time.perf_counter() - start) / n_batches
    cpu_percent = process.cpu_percent(interval=None)
    rss_after = process.memory_info().rss

    n_cores = os.cpu_count() or 1
    return {
        "torch_threads": torch.get_num_threads(),
        "cpu_logical_cores": n_cores,
        "cuda_available": torch.cuda.is_available(),
        "ram_total_gb": round(virtual.total / 1e9, 2),
        "ram_available_gb_at_audit": round(virtual.available / 1e9, 2),
        "process_rss_mb": round(rss_after / 1e6, 1),
        "process_rss_growth_mb": round((rss_after - rss_before) / 1e6, 1),
        "cpu_percent_during_training": round(cpu_percent, 1),
        "cpu_percent_of_all_cores": round(cpu_percent / n_cores, 1),
        "loader_s_per_batch": round(loader_seconds, 4),
        "loader_img_per_s": round(8 / loader_seconds, 1),
        "train_s_per_batch": round(step_seconds, 4),
        "train_img_per_s": round(8 / step_seconds, 2),
        "loader_share_of_step": round(loader_seconds / step_seconds, 4),
        "bottleneck": (
            "compute" if loader_seconds < 0.1 * step_seconds else "data loading"
        ),
        "notes": [
            "Data loading is a negligible fraction of step time, so worker "
            "processes do not help; they were measured in Sprint 2 to make "
            "training slightly slower by competing for the same cores.",
            "bf16 autocast was measured at ~21x slower on this CPU (no native "
            "bf16 support) and is not used.",
        ],
    }
