"""Sprint 2 / Stage B - evaluate the trained U-Net and visualise predictions.

Evaluates on the **held-out test patients** from the Sprint 1 patient-level
split, using every test slice (no cap).

Writes:

    outputs/metrics/segmentation_metrics_<split>.json   Dice / IoU / precision / recall
    outputs/metrics/segmentation_per_class_<split>.csv  per-class table
    outputs/metrics/segmentation_per_patient_<split>.csv per-patient Dice
    outputs/metrics/disc_identification_<split>.json    Stage C sanity: do derived
                                                        disc indices match truth?
    outputs/visualizations/predictions/*.png            5/6-panel figures
    outputs/visualizations/training_history.png
    outputs/visualizations/segmentation_metrics.png

Usage
-----
    python scripts/evaluate_unet.py
    python scripts/evaluate_unet.py --split val --n-visualizations 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.disc_features import (  # noqa: E402
    evaluate_disc_identification,
    instance_mask_from_semantic,
)
from src.models.data import (  # noqa: E402
    N_CLASSES,
    LumbarSliceDataset,
    assert_no_patient_leakage,
    load_slice_index,
)
from src.models.metrics import (  # noqa: E402
    ConfusionAccumulator,
    format_metrics_table,
    per_patient_dice,
)
from src.models.predict_viz import (  # noqa: E402
    plot_metric_bars,
    plot_training_history,
    visualize_prediction,
)
from src.models.unet import build_unet  # noqa: E402
from src.preprocessing.labels import SEMANTIC_CLASSES  # noqa: E402
from src.utils.paths import (  # noqa: E402
    OUTPUTS_DIR,
    RANDOM_SEED,
    VISUALIZATIONS_DIR,
    ensure_dirs,
)
from src.utils.reporting import save_json  # noqa: E402

MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = VISUALIZATIONS_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=MODELS_DIR / "unet_baseline.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slices", type=int, default=0,
                        help="0 = use every slice in the split.")
    parser.add_argument("--n-visualizations", type=int, default=8)
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also write predicted masks to data/processed/predictions/ "
                             "for the disc-extraction stage.")
    parser.add_argument("--tag", default=None,
                        help="Experiment tag. When given, metrics go to "
                             "outputs/reports/<tag>/ and figures to "
                             "outputs/visualizations/<tag>/ instead of the "
                             "top-level directories, so a previous run's results "
                             "are never overwritten.")
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Where to write predicted masks (default "
                             "data/processed/predictions).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def resolve_output_dirs(tag: str | None) -> tuple[Path, Path, Path]:
    """Return (metrics_dir, visualizations_dir, predictions_figure_dir).

    Without a tag the script writes exactly where it always did, so the Sprint 2
    baseline outputs stay reproducible. With a tag everything is namespaced.
    """
    if tag is None:
        return METRICS_DIR, VISUALIZATIONS_DIR, PREDICTIONS_DIR
    metrics = OUTPUTS_DIR / "reports" / tag / "metrics"
    visualizations = VISUALIZATIONS_DIR / tag
    return metrics, visualizations, visualizations / "predictions"


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from the checkpoint's stored architecture."""
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run scripts/train_unet.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = build_unet(
        n_classes=architecture["n_classes"],
        base_channels=architecture["base_channels"],
        depth=architecture["depth"],
        in_channels=architecture["in_channels"],
        bilinear=architecture["bilinear"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(memory_format=torch.channels_last)
    return model, checkpoint.get("metadata", {})


def main() -> None:
    global METRICS_DIR, VISUALIZATIONS_DIR, PREDICTIONS_DIR

    args = parse_args()
    ensure_dirs()
    METRICS_DIR, VISUALIZATIONS_DIR, PREDICTIONS_DIR = resolve_output_dirs(args.tag)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if args.tag:
        print(f"experiment tag: {args.tag}")
        print(f"  metrics   -> {METRICS_DIR}")
        print(f"  figures   -> {VISUALIZATIONS_DIR}")

    model, metadata = load_model(args.checkpoint)
    print(f"Loaded {args.checkpoint.name}: {model.describe()}")
    print(f"  checkpoint metadata: {metadata}")

    index = load_slice_index()
    assert_no_patient_leakage(index)
    subset = index[index["split"] == args.split].reset_index(drop=True)
    if args.max_slices:
        subset = subset.head(args.max_slices)

    print(f"\nEvaluating on '{args.split}': {len(subset):,} slices, "
          f"{subset['patient_id'].nunique()} patients")

    dataset = LumbarSliceDataset(subset, augment=False, return_instance=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    accumulator = ConfusionAccumulator(N_CLASSES)
    patient_records: list[dict] = []
    identification: list[dict] = []
    per_slice_rows: list[dict] = []

    prediction_dir = None
    if args.save_predictions:
        prediction_dir = args.predictions_dir or Path("data/processed/predictions")
        prediction_dir.mkdir(parents=True, exist_ok=True)
        print(f"  predicted masks -> {prediction_dir}")

    print("Running inference ...")
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(memory_format=torch.channels_last)
            truth = batch["mask"]
            truth_instance = batch["mask_instance"]

            logits = model(image)
            prediction = logits.argmax(dim=1)

            accumulator.update(prediction, truth)
            patient_records.append(
                {
                    "patient_id": int(batch["patient_id"][0]),
                    "prediction": prediction,
                    "target": truth,
                }
            )

            # Per-slice Dice, and disc-identity agreement (Stage C sanity check).
            for i in range(prediction.shape[0]):
                slice_accumulator = ConfusionAccumulator(N_CLASSES)
                slice_accumulator.update(prediction[i : i + 1], truth[i : i + 1])
                slice_metrics = slice_accumulator.summary()
                per_slice_rows.append(
                    {
                        "slice_id": batch["slice_id"][i],
                        "patient_id": int(batch["patient_id"][i]),
                        "image_id": batch["image_id"][i],
                        "modality": batch["modality"][i],
                        "fg_dice": slice_metrics["aggregate"]["macro_foreground"]["dice"],
                        **{
                            f"dice_{name}": values["dice"]
                            for name, values in
                            slice_metrics["aggregate"]["per_class"].items()
                        },
                    }
                )

                derived = instance_mask_from_semantic(prediction[i].numpy())
                agreement = evaluate_disc_identification(
                    truth_instance[i].numpy(), derived["instance_mask"]
                )
                identification.append(
                    {
                        "slice_id": batch["slice_id"][i],
                        "patient_id": int(batch["patient_id"][i]),
                        **{k: v for k, v in agreement.items() if k != "matches"},
                    }
                )

                if prediction_dir is not None:
                    np.savez_compressed(
                        prediction_dir / f"{batch['slice_id'][i]}.npz",
                        semantic=prediction[i].numpy().astype(np.uint8),
                        instance=derived["instance_mask"],
                    )

            if step % 25 == 0 or step == len(loader):
                print(f"  {step}/{len(loader)} batches", flush=True)

    # --- metrics ---------------------------------------------------------
    summary = accumulator.summary()
    summary["split"] = args.split
    summary["n_patients"] = int(subset["patient_id"].nunique())
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_metadata"] = metadata
    summary["model"] = model.describe()

    print(f"\n=== {args.split.upper()} SET METRICS - aggregate (dataset-level) ===")
    print(format_metrics_table(summary, style="aggregate"))
    print(f"\n=== {args.split.upper()} SET METRICS - per-slice mean ===")
    print(format_metrics_table(summary, style="per_slice_mean"))
    print(f"\n  pixel accuracy: {summary['pixel_accuracy']:.4f}")

    patient_dice = per_patient_dice(patient_records, N_CLASSES)
    summary["per_patient_dice"] = patient_dice
    print("\n=== per-patient Dice (mean +/- std over patients) ===")
    for name, values in patient_dice.items():
        print(f"  {name:22s} {values['mean_dice']:.4f} +/- {values['std_dice']:.4f}  "
              f"(n={values['n_patients']} patients)")

    # --- disc identification --------------------------------------------
    identification_frame = pd.DataFrame(identification)
    id_summary = {
        "n_slices": int(len(identification_frame)),
        "n_truth_discs_total": int(identification_frame["n_truth_discs"].sum()),
        "n_predicted_discs_total": int(identification_frame["n_predicted_discs"].sum()),
        "n_regions_found": int(identification_frame["n_region_found"].sum()),
        "n_index_correct": int(identification_frame["n_index_correct"].sum()),
        "pct_slices_with_matching_disc_count": round(
            100 * float(identification_frame["count_matches"].mean()), 2
        ),
        "pct_discs_region_found": round(
            100 * identification_frame["n_region_found"].sum()
            / max(identification_frame["n_truth_discs"].sum(), 1), 2
        ),
        "pct_discs_index_correct": round(
            100 * identification_frame["n_index_correct"].sum()
            / max(identification_frame["n_truth_discs"].sum(), 1), 2
        ),
        "note": "Disc identity derived from the SEMANTIC prediction by ordered "
                "connected components. Measures how much segmentation error "
                "propagates into disc numbering. Ground-truth instance masks are "
                "used for the Stage D feature table precisely because of this.",
    }
    print("\n=== disc identification from predicted masks ===")
    for key in ["pct_slices_with_matching_disc_count", "pct_discs_region_found",
                "pct_discs_index_correct"]:
        print(f"  {key:42s} {id_summary[key]}%")
    save_json(id_summary, METRICS_DIR / f"disc_identification_{args.split}.json")

    # --- save ------------------------------------------------------------
    save_json(summary, METRICS_DIR / f"segmentation_metrics_{args.split}.json")

    per_class_rows = []
    for name, values in summary["aggregate"]["per_class"].items():
        per_slice = summary["per_slice_mean"]["per_class"][name]
        per_class_rows.append(
            {
                "class": name,
                "class_id": values["class_id"],
                "dice_aggregate": round(values["dice"], 5),
                "iou_aggregate": round(values["iou"], 5),
                "precision_aggregate": round(values["precision"], 5),
                "recall_aggregate": round(values["recall"], 5),
                "dice_per_slice_mean": round(per_slice["dice"], 5),
                "iou_per_slice_mean": round(per_slice["iou"], 5),
                "precision_per_slice_mean": round(per_slice["precision"], 5),
                "recall_per_slice_mean": round(per_slice["recall"], 5),
                "dice_per_patient_mean": round(patient_dice[name]["mean_dice"], 5),
                "dice_per_patient_std": round(patient_dice[name]["std_dice"], 5),
                "support_pixels_true": values["support_pixels_true"],
                "n_slices_counted": per_slice["n_slices_counted"],
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        METRICS_DIR / f"segmentation_per_class_{args.split}.csv", index=False
    )

    slice_frame = pd.DataFrame(per_slice_rows)
    slice_frame.to_csv(
        METRICS_DIR / f"segmentation_per_slice_{args.split}.csv", index=False
    )
    patient_frame = (
        slice_frame.groupby("patient_id")
        .agg(n_slices=("slice_id", "count"), mean_fg_dice=("fg_dice", "mean"))
        .reset_index()
        .sort_values("mean_fg_dice")
    )
    patient_frame.to_csv(
        METRICS_DIR / f"segmentation_per_patient_{args.split}.csv", index=False
    )

    # --- figures ---------------------------------------------------------
    print("\nRendering figures ...")
    plot_metric_bars(
        summary["aggregate"]["per_class"],
        title=f"{args.split} set segmentation metrics per class (aggregate)",
        save_path=VISUALIZATIONS_DIR / "segmentation_metrics.png",
    )
    print(f"  -> {VISUALIZATIONS_DIR / 'segmentation_metrics.png'}")

    history_path = METRICS_DIR / "training_history.csv"
    if history_path.exists():
        plot_training_history(
            pd.read_csv(history_path),
            save_path=VISUALIZATIONS_DIR / "training_history.png",
        )
        print(f"  -> {VISUALIZATIONS_DIR / 'training_history.png'}")

    render_prediction_figures(
        model, subset, slice_frame, args.n_visualizations, args.split, args.seed
    )

    print(f"\n  -> {METRICS_DIR / f'segmentation_metrics_{args.split}.json'}")
    print(f"  -> {METRICS_DIR / f'segmentation_per_class_{args.split}.csv'}")
    print("\nDone. These are baseline CPU-budget results, not final performance.")


def render_prediction_figures(
    model, subset: pd.DataFrame, slice_frame: pd.DataFrame,
    n: int, split: str, seed: int,
) -> None:
    """Render prediction figures spanning the quality range.

    Samples are chosen deliberately rather than randomly: the best, median and
    worst slices by foreground Dice are all included. Showing only random
    samples would hide the failure modes, which is what a reviewer most needs
    to see.
    """
    # Bring in the annotated-area column so mid-sagittal slices can be found.
    if "labelled_pixels_processed" in subset.columns:
        slice_frame = slice_frame.merge(
            subset[["slice_id", "labelled_pixels_processed"]], on="slice_id", how="left"
        )

    ranked = slice_frame.sort_values("fg_dice").reset_index(drop=True)
    if ranked.empty:
        return

    picks: list[tuple[str, pd.Series]] = []
    n_each = max(1, n // 4)

    # Mid-sagittal slices first. Ranking by Dice alone is misleading here: a
    # far-lateral slice containing a 100-pixel vertebra tip can score 0.96
    # because the structure is tiny and isolated, while a mid-sagittal slice
    # with the whole lumbar spine scores lower on a much harder problem. The
    # mid-sagittal slices are the clinically informative ones, so they are shown
    # explicitly rather than left to chance.
    if "labelled_pixels_processed" in ranked.columns:
        mid_sagittal = ranked.nlargest(n_each, "labelled_pixels_processed")
        for _, row in mid_sagittal.iterrows():
            picks.append(("midsagittal", row))

    for label, rows in [
        ("worst", ranked.head(n_each)),
        ("median", ranked.iloc[
            max(0, len(ranked) // 2 - n_each // 2) : len(ranked) // 2 + n_each // 2 + 1
        ]),
        ("best", ranked.tail(n_each)),
    ]:
        for _, row in rows.iterrows():
            picks.append((label, row))
    picks = picks[:n]

    lookup = subset.set_index("slice_id")
    dataset = LumbarSliceDataset(subset, augment=False)
    positions = {slice_id: i for i, slice_id in enumerate(subset["slice_id"])}

    with torch.inference_mode():
        for label, row in picks:
            slice_id = row["slice_id"]
            sample = dataset[positions[slice_id]]
            image = sample["image"][None].to(memory_format=torch.channels_last)
            prediction = model(image).argmax(dim=1)[0].numpy()

            meta = lookup.loc[slice_id]
            labelled = row.get("labelled_pixels_processed")
            dice_text = (
                f"foreground Dice = {row['fg_dice']:.3f}   |   "
                + "   ".join(
                    f"{name}={row[f'dice_{name}']:.3f}"
                    for name in SEMANTIC_CLASSES.values()
                    if name != "background" and np.isfinite(row[f"dice_{name}"])
                )
                + (
                    f"   |   annotated area = {int(labelled):,} px"
                    if labelled is not None and np.isfinite(labelled)
                    else ""
                )
            )
            name = f"pred_{split}_{label}_{slice_id}.png"
            visualize_prediction(
                sample["image"][0].numpy(),
                sample["mask"].numpy(),
                prediction,
                title=(
                    f"[{label}] {slice_id}  |  patient {meta['patient_id']}  |  "
                    f"{meta['modality']}  |  split={split}"
                ),
                dice_text=dice_text,
                save_path=PREDICTIONS_DIR / name,
            )
            print(f"  -> {PREDICTIONS_DIR / name}")


if __name__ == "__main__":
    main()
