"""Loss functions for multi-class segmentation.

Why not plain cross-entropy
---------------------------
Sprint 1 measured the class balance of the preprocessed dataset:

    background 94.77%, vertebra 3.76%, IVD 0.79%, spinal canal 0.68%

A pixel-wise cross-entropy is minimised very effectively by predicting
background almost everywhere - that alone scores ~95% pixel accuracy while
segmenting nothing. Dice loss is computed per class over the region union, so a
class contributes equally regardless of how few pixels it occupies, which is
exactly the property needed here.

The default is the sum of both. Cross-entropy gives well-behaved per-pixel
gradients early in training; Dice supplies the gradient signal for the small
classes. This combination is the standard baseline for imbalanced medical
segmentation and is deliberately not something more elaborate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    n_classes: int,
    smooth: float = 1.0,
    ignore_background: bool = False,
) -> torch.Tensor:
    """Soft (differentiable) multi-class Dice loss.

    Parameters
    ----------
    logits:
        ``(N, C, H, W)`` raw network outputs.
    target:
        ``(N, H, W)`` int64 class indices.
    smooth:
        Added to numerator and denominator. Keeps the loss finite when a class
        is absent from both prediction and target, which happens constantly
        here - most slices contain no spinal canal at their lateral edges.
    ignore_background:
        Exclude class 0 from the average. Left False by default so background
        still receives gradient; the metric reporting handles the
        foreground-only view separately.

    Returns
    -------
    torch.Tensor
        Scalar loss in ``[0, 1]``.
    """
    # Softmax gives a differentiable stand-in for the one-hot prediction.
    probabilities = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=n_classes)
    # (N, H, W, C) -> (N, C, H, W)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    start = 1 if ignore_background else 0
    probabilities = probabilities[:, start:]
    target_onehot = target_onehot[:, start:]

    # Sum over batch and spatial dims, keeping classes separate.
    dims = (0, 2, 3)
    intersection = (probabilities * target_onehot).sum(dims)
    cardinality = probabilities.sum(dims) + target_onehot.sum(dims)

    dice_per_class = (2 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice_per_class.mean()


class DiceCrossEntropyLoss(nn.Module):
    """Sum of cross-entropy and soft Dice loss.

    Parameters
    ----------
    n_classes:
        Number of segmentation classes.
    ce_weight, dice_weight:
        Relative weights of the two terms. Equal by default.
    class_weights:
        Optional per-class weights for the cross-entropy term. Available but
        not used by default: the Dice term already counteracts the imbalance,
        and stacking both tends to make training unstable at this batch size.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
        ignore_background_in_dice: bool = False,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_background_in_dice = ignore_background_in_dice
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        """Return the total loss plus each component, for logging."""
        ce = self.cross_entropy(logits, target)
        dice = soft_dice_loss(
            logits,
            target,
            n_classes=self.n_classes,
            ignore_background=self.ignore_background_in_dice,
        )
        total = self.ce_weight * ce + self.dice_weight * dice
        return {"loss": total, "ce": ce.detach(), "dice": dice.detach()}


def inverse_frequency_class_weights(
    class_pixel_counts: dict[int, int], *, normalise: bool = True
) -> torch.Tensor:
    """Class weights inversely proportional to pixel frequency.

    Provided for experimentation. Not enabled by default - see
    :class:`DiceCrossEntropyLoss`.
    """
    counts = torch.tensor(
        [max(class_pixel_counts.get(i, 0), 1) for i in sorted(class_pixel_counts)],
        dtype=torch.float32,
    )
    weights = counts.sum() / counts
    if normalise:
        weights = weights / weights.mean()
    return weights
