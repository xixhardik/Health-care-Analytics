"""Baseline 2-D multi-class U-Net.

Deliberately plain. Sprint 2 asks for a *baseline* that works end to end, so
this is the original U-Net topology (Ronneberger et al., 2015) with two
pragmatic changes and nothing else:

* **Batch normalisation** in each conv block. The original paper predates it;
  without it a from-scratch run on this dataset converges noticeably slower.
* **Configurable width and depth**, because this project trains on CPU only
  (verified: no CUDA device on this machine). A width-64 U-Net has ~31 M
  parameters and is not trainable in reasonable time here, so the default is
  width 16 (~1.9 M parameters). The architecture is otherwise unchanged, and
  the width is a single constructor argument, so a GPU run can use 64 without
  touching this file.

No attention, no transformer blocks, no ensembling, no 3-D convolutions - those
are explicitly out of scope for this sprint.

Input/output contract
---------------------
* input  ``(N, in_channels, H, W)`` float32, intensities already in ``[0, 1]``
  by Sprint 1 preprocessing
* output ``(N, n_classes, H, W)`` raw logits - **no softmax applied**, because
  ``torch.nn.CrossEntropyLoss`` expects logits

``H`` and ``W`` must both be divisible by ``2 ** depth``. Sprint 1 produces
352 x 256 slices, and 352 = 2**5 x 11, 256 = 2**8, so depths up to 5 are safe.
:meth:`UNet.check_spatial_size` enforces this instead of letting it fail as a
shape mismatch deep in the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2, the standard U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # padding=1 keeps the spatial size, so skip connections line up
            # exactly and no cropping is needed.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling step: max-pool by 2, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling step: upsample, concatenate the skip connection, DoubleConv.

    ``bilinear=True`` upsamples with interpolation and halves the channels with
    a 1x1 convolution; ``False`` uses a learned transposed convolution.
    Bilinear is the default because it has fewer parameters and avoids the
    checkerboard artefacts transposed convolutions can produce - both useful
    when the training budget is small.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            upsampled_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2,
                                         stride=2)
            upsampled_channels = in_channels // 2
        self.conv = DoubleConv(upsampled_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against odd input sizes: pad rather than silently mis-align.
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y or diff_x:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """2-D multi-class U-Net.

    Parameters
    ----------
    in_channels:
        Input channels. 1 for the grayscale MRI slices Sprint 1 produces.
    n_classes:
        Number of output classes. 4 for this project: background, vertebra,
        intervertebral disc, spinal canal.
    base_channels:
        Width of the first encoder stage; each stage doubles it. 16 by default
        for CPU training; 64 reproduces the original paper's width.
    depth:
        Number of downsampling steps.
    bilinear:
        Upsampling mode (see :class:`Up`).
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 4,
        base_channels: int = 16,
        depth: int = 4,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear

        # Encoder channel widths: base, 2*base, 4*base, ...
        widths = [base_channels * 2**i for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            [Down(widths[i], widths[i + 1]) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [
                Up(widths[i + 1], widths[i], widths[i], bilinear=bilinear)
                for i in reversed(range(depth))
            ]
        )
        # 1x1 conv to the class logits.
        self.head = nn.Conv2d(widths[0], n_classes, kernel_size=1)

    # -- introspection helpers -------------------------------------------

    @property
    def size_divisor(self) -> int:
        """Spatial sizes must be divisible by this for the skips to align."""
        return 2**self.depth

    def check_spatial_size(self, height: int, width: int) -> None:
        """Raise early with a clear message if the input size will not work."""
        divisor = self.size_divisor
        if height % divisor or width % divisor:
            raise ValueError(
                f"Input {height}x{width} is not divisible by {divisor} "
                f"(required for depth={self.depth}). Sprint 1 produces "
                f"352x256, which works for depth<=5."
            )

    def count_parameters(self) -> dict[str, int]:
        """Total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def describe(self) -> str:
        """One-line description used in logs and reports."""
        counts = self.count_parameters()
        return (
            f"UNet(in={self.in_channels}, classes={self.n_classes}, "
            f"base={self.base_channels}, depth={self.depth}, "
            f"bilinear={self.bilinear}) "
            f"{counts['total']:,} params"
        )

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape ``(N, n_classes, H, W)``."""
        if x.dim() != 4:
            raise ValueError(f"Expected a 4-D tensor (N,C,H,W), got shape {tuple(x.shape)}")
        self.check_spatial_size(x.size(-2), x.size(-1))

        skips: list[torch.Tensor] = []
        x = self.stem(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.head(x)


def build_unet(
    n_classes: int = 4,
    base_channels: int = 16,
    depth: int = 4,
    in_channels: int = 1,
    bilinear: bool = True,
) -> UNet:
    """Factory used by the training/evaluation scripts.

    Keeping construction in one function means the checkpoint and the
    evaluation script cannot disagree about the architecture.
    """
    return UNet(
        in_channels=in_channels,
        n_classes=n_classes,
        base_channels=base_channels,
        depth=depth,
        bilinear=bilinear,
    )
