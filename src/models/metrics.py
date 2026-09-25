"""Segmentation metrics: Dice, IoU, precision and recall.

Two ways of aggregating, because they answer different questions and are
routinely conflated in papers:

**Aggregate ("dataset-level")** - accumulate TP/FP/FN over every pixel of every
slice, then compute the metric once. Large structures dominate. This is the
primary number reported here because it is unambiguous and has no undefined
cases.

**Per-slice mean ("macro over slices")** - compute the metric on each slice and
average. Every slice counts equally, so a slice clipping the edge of a disc
weighs as much as a mid-sagittal one. This has an undefined case: a class
absent from both the prediction and the ground truth of a slice gives 0/0. Such
slices are **excluded** from that class's average and the count of contributing
slices is reported alongside, rather than silently scoring them 1.0 (which
inflates results) or 0.0 (which deflates them).

Both are reported so the numbers can be compared against other work either way.
"""

from __future__ import annotations

import numpy as np
import torch

from src.preprocessing.labels import SEMANTIC_CLASSES


class ConfusionAccumulator:
    """Accumulates per-class TP/FP/FN/TN across batches.

    Kept as integer counts so the aggregate metrics are exact rather than a
    running average of averages.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.tp = np.zeros(n_classes, dtype=np.int64)
        self.fp = np.zeros(n_classes, dtype=np.int64)
        self.fn = np.zeros(n_classes, dtype=np.int64)
        self.tn = np.zeros(n_classes, dtype=np.int64)
        self.n_pixels = 0
        self.n_slices = 0

        # Per-slice metric accumulation, per class. Each metric keeps its own
        # count because they become undefined under different conditions: on a
        # slice where the class is in the ground truth but nothing is predicted,
        # Dice and IoU are 0 (defined) while precision is 0/0 (undefined).
        # Sharing one counter would let a single undefined slice turn an entire
        # average into NaN.
        self._slice_sums = {
            name: np.zeros(n_classes, dtype=np.float64)
            for name in ("dice", "iou", "precision", "recall")
        }
        self._slice_metric_counts = {
            name: np.zeros(n_classes, dtype=np.int64)
            for name in ("dice", "iou", "precision", "recall")
        }
        self._slice_counts = np.zeros(n_classes, dtype=np.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Parameters
        ----------
        prediction:
            ``(N, H, W)`` int - predicted class indices (already arg-maxed).
        target:
            ``(N, H, W)`` int - ground-truth class indices.
        """
        prediction = prediction.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
            )

        self.n_pixels += int(prediction.size)
        self.n_slices += int(prediction.shape[0])

        for class_id in range(self.n_classes):
            pred_c = prediction == class_id
            true_c = target == class_id

            self.tp[class_id] += int(np.sum(pred_c & true_c))
            self.fp[class_id] += int(np.sum(pred_c & ~true_c))
            self.fn[class_id] += int(np.sum(~pred_c & true_c))
            self.tn[class_id] += int(np.sum(~pred_c & ~true_c))

            # Per-slice metrics for this class.
            for i in range(prediction.shape[0]):
                p = pred_c[i]
                t = true_c[i]
                tp = int(np.sum(p & t))
                fp = int(np.sum(p & ~t))
                fn = int(np.sum(~p & t))

                # Class absent from both -> metric undefined, skip it.
                if tp + fp + fn == 0:
                    continue

                self._slice_counts[class_id] += 1
                for name, value in (
                    ("dice", _safe_div(2 * tp, 2 * tp + fp + fn)),
                    ("iou", _safe_div(tp, tp + fp + fn)),
                    ("precision", _safe_div(tp, tp + fp)),
                    ("recall", _safe_div(tp, tp + fn)),
                ):
                    # Skip undefined values rather than letting one NaN
                    # propagate through the whole average.
                    if not np.isnan(value):
                        self._slice_sums[name][class_id] += value
                        self._slice_metric_counts[name][class_id] += 1

    # -- results ----------------------------------------------------------

    def aggregate_per_class(self) -> dict[str, dict[str, float]]:
        """Dataset-level metrics for each class."""
        out: dict[str, dict[str, float]] = {}
        for class_id in range(self.n_classes):
            tp = int(self.tp[class_id])
            fp = int(self.fp[class_id])
            fn = int(self.fn[class_id])
            out[SEMANTIC_CLASSES[class_id]] = {
                "class_id": class_id,
                "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
                "iou": _safe_div(tp, tp + fp + fn),
                "precision": _safe_div(tp, tp + fp),
                "recall": _safe_div(tp, tp + fn),
                "support_pixels_true": tp + fn,
                "support_pixels_pred": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return out

    def per_slice_per_class(self) -> dict[str, dict[str, float]]:
        """Per-slice-averaged metrics for each class, with contributing counts."""
        out: dict[str, dict[str, float]] = {}
        for class_id in range(self.n_classes):
            entry = {
                "class_id": class_id,
                "n_slices_counted": int(self._slice_counts[class_id]),
            }
            for name, sums in self._slice_sums.items():
                count = int(self._slice_metric_counts[name][class_id])
                entry[name] = float(sums[class_id] / count) if count else float("nan")
                # How many slices each metric is actually averaged over, so a
                # reader can see when a figure rests on fewer slices than the
                # class appears in.
                entry[f"{name}_n_defined"] = count
            out[SEMANTIC_CLASSES[class_id]] = entry
        return out

    def summary(self) -> dict:
        """Everything: per-class and overall, both aggregation styles."""
        aggregate = self.aggregate_per_class()
        per_slice = self.per_slice_per_class()

        foreground = [SEMANTIC_CLASSES[i] for i in range(1, self.n_classes)]
        all_classes = [SEMANTIC_CLASSES[i] for i in range(self.n_classes)]

        def mean_over(source: dict, names: list[str], metric: str) -> float:
            values = [source[n][metric] for n in names if not np.isnan(source[n][metric])]
            return float(np.mean(values)) if values else float("nan")

        pixel_accuracy = _safe_div(int(self.tp.sum()), self.n_pixels)

        return {
            "n_slices": self.n_slices,
            "n_pixels": self.n_pixels,
            "pixel_accuracy": pixel_accuracy,
            "aggregate": {
                "per_class": aggregate,
                "macro_all_classes": {
                    m: mean_over(aggregate, all_classes, m)
                    for m in ("dice", "iou", "precision", "recall")
                },
                "macro_foreground": {
                    m: mean_over(aggregate, foreground, m)
                    for m in ("dice", "iou", "precision", "recall")
                },
            },
            "per_slice_mean": {
                "per_class": per_slice,
                "macro_all_classes": {
                    m: mean_over(per_slice, all_classes, m)
                    for m in ("dice", "iou", "precision", "recall")
                },
                "macro_foreground": {
                    m: mean_over(per_slice, foreground, m)
                    for m in ("dice", "iou", "precision", "recall")
                },
            },
        }


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns NaN instead of raising on a zero denominator."""
    return float(numerator) / float(denominator) if denominator else float("nan")


def format_metrics_table(summary: dict, *, style: str = "aggregate") -> str:
    """Render a metrics summary as a fixed-width table for console/logs."""
    block = summary[style]
    lines = [
        f"  {'class':22s} {'dice':>8s} {'iou':>8s} {'precision':>10s} {'recall':>8s} "
        f"{'true px':>12s}",
        "  " + "-" * 72,
    ]
    for name, values in block["per_class"].items():
        support = values.get("support_pixels_true", values.get("n_slices_counted", 0))
        lines.append(
            f"  {name:22s} {values['dice']:8.4f} {values['iou']:8.4f} "
            f"{values['precision']:10.4f} {values['recall']:8.4f} {support:12,d}"
        )
    lines.append("  " + "-" * 72)
    macro_all = block["macro_all_classes"]
    macro_fg = block["macro_foreground"]
    lines.append(
        f"  {'macro (all classes)':22s} {macro_all['dice']:8.4f} {macro_all['iou']:8.4f} "
        f"{macro_all['precision']:10.4f} {macro_all['recall']:8.4f}"
    )
    lines.append(
        f"  {'macro (foreground)':22s} {macro_fg['dice']:8.4f} {macro_fg['iou']:8.4f} "
        f"{macro_fg['precision']:10.4f} {macro_fg['recall']:8.4f}"
    )
    return "\n".join(lines)


def per_patient_dice(
    records: list[dict], n_classes: int = 4
) -> dict[str, dict[str, float]]:
    """Aggregate Dice per patient, then average across patients.

    Slice counts differ per patient (8-120 sagittal slices), so a plain
    slice-level mean over-weights patients with thicker stacks. Averaging
    within a patient first gives every patient equal weight, which matches how
    clinical performance would be reported.
    """
    by_patient: dict[int, ConfusionAccumulator] = {}
    for record in records:
        patient_id = record["patient_id"]
        accumulator = by_patient.setdefault(patient_id, ConfusionAccumulator(n_classes))
        accumulator.update(record["prediction"], record["target"])

    per_class_values: dict[str, list[float]] = {
        SEMANTIC_CLASSES[i]: [] for i in range(n_classes)
    }
    for accumulator in by_patient.values():
        for name, values in accumulator.aggregate_per_class().items():
            if not np.isnan(values["dice"]):
                per_class_values[name].append(values["dice"])

    return {
        name: {
            "mean_dice": float(np.mean(values)) if values else float("nan"),
            "std_dice": float(np.std(values)) if values else float("nan"),
            "n_patients": len(values),
        }
        for name, values in per_class_values.items()
    }
