"""Dataset and loaders for the preprocessed Sprint 1 slices.

Reads ``data/processed/slice_index.csv`` and the per-slice ``.npz`` files that
Sprint 1 wrote. **Sprint 1 output is treated as read-only** - nothing here
regenerates or modifies it.

Leakage safety
--------------
The split is taken from the ``split`` column of ``slice_index.csv``, which was
assigned per *patient* in Sprint 1. :func:`build_dataloaders` additionally
asserts that no patient id appears in two splits, so a corrupted index file
cannot silently reintroduce leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.labels import SEMANTIC_CLASSES
from src.utils.paths import PROJECT_ROOT, SLICE_INDEX_CSV

N_CLASSES = len(SEMANTIC_CLASSES)  # 4: background, vertebra, IVD, canal


class LumbarSliceDataset(Dataset):
    """Preprocessed sagittal slices with their 4-class semantic masks.

    Parameters
    ----------
    index:
        Rows of ``slice_index.csv`` for one split.
    augment:
        Apply light training-time augmentation (see :meth:`_augment`).
    downsample:
        Integer factor to shrink the slice by before returning it. ``1`` keeps
        the native Sprint 1 size of 352 x 256. This exists purely as a
        compute lever for CPU-only training and does **not** alter the stored
        data. Masks are downsampled by strided subsampling, never by
        interpolation, so label values are preserved exactly.
    return_instance:
        Also return the instance mask (individual vertebra/disc identities).
        Needed by the disc-extraction stage, not by segmentation training.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        *,
        augment: bool = False,
        downsample: int = 1,
        return_instance: bool = False,
        seed: int = 0,
    ) -> None:
        self.index = index.reset_index(drop=True)
        self.augment = augment
        self.downsample = int(downsample)
        self.return_instance = return_instance
        self._rng = np.random.default_rng(seed)

        if self.downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")
        if "npz_path" not in self.index.columns:
            raise KeyError("slice_index.csv is missing the 'npz_path' column")

    def __len__(self) -> int:
        return len(self.index)

    # -- augmentation RNG state (for exact resume) ------------------------
    # The augmentation stream is part of the training trajectory, so a resumed
    # run that rebuilds it from the seed would replay the same augmentations it
    # already used. Exposing the state lets the trainer checkpoint and restore
    # it. This does not change the augmentation policy, only its bookkeeping.

    def get_rng_state(self) -> dict:
        """Return the augmentation generator's state, for checkpointing."""
        return dict(self._rng.bit_generator.state)

    def set_rng_state(self, state: dict) -> None:
        """Restore a previously saved augmentation generator state."""
        self._rng.bit_generator.state = state

    def _load(self, row: pd.Series) -> dict[str, np.ndarray]:
        path = PROJECT_ROOT / row["npz_path"]
        with np.load(path) as bundle:
            out = {
                "image": bundle["image"].astype(np.float32),
                "mask": bundle["mask"],
            }
            if self.return_instance:
                out["mask_instance"] = bundle["mask_instance"]
        return out

    def _subsample(self, array: np.ndarray) -> np.ndarray:
        """Shrink by integer striding - exact for masks, adequate for images."""
        if self.downsample == 1:
            return array
        return array[:: self.downsample, :: self.downsample]

    def _augment_pair(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Light, anatomy-preserving augmentation.

        Deliberately conservative. Horizontal and vertical flips are **not**
        used: a sagittal lumbar slice has a fixed anatomical orientation
        (superior up, anterior left), so flipping would produce images that
        cannot occur and would teach the model that anterior/posterior does
        not matter - when in fact the spinal canal is defined by being
        posterior to the vertebral bodies.

        What is applied:

        * small translation (up to 5% of each axis) - plausible patient
          positioning variation
        * multiplicative intensity scaling and an additive offset - plausible
          residual contrast variation after normalisation

        Translation uses ``np.roll`` on both image and mask identically, so
        they stay aligned, and no interpolation touches the labels.
        """
        rows, cols = image.shape
        shift_y = int(self._rng.integers(-rows // 20, rows // 20 + 1))
        shift_x = int(self._rng.integers(-cols // 20, cols // 20 + 1))
        if shift_y or shift_x:
            image = np.roll(image, (shift_y, shift_x), axis=(0, 1))
            mask = np.roll(mask, (shift_y, shift_x), axis=(0, 1))

        scale = float(self._rng.uniform(0.9, 1.1))
        offset = float(self._rng.uniform(-0.05, 0.05))
        image = np.clip(image * scale + offset, 0.0, 1.0)
        return image, mask

    def __getitem__(self, position: int) -> dict:
        row = self.index.iloc[position]
        bundle = self._load(row)

        image = self._subsample(bundle["image"])
        mask = self._subsample(bundle["mask"])

        if self.augment:
            image, mask = self._augment_pair(image, mask)

        sample = {
            # (1, H, W) float32 - a single grayscale channel.
            "image": torch.from_numpy(np.ascontiguousarray(image[None])).float(),
            # (H, W) int64 - class indices, as CrossEntropyLoss expects.
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "slice_id": row["slice_id"],
            "image_id": row["image_id"],
            "patient_id": int(row["patient_id"]),
            "modality": row["modality"],
        }
        if self.return_instance:
            sample["mask_instance"] = torch.from_numpy(
                np.ascontiguousarray(self._subsample(bundle["mask_instance"]))
            ).long()
        return sample


def load_slice_index(path: Path | str = SLICE_INDEX_CSV) -> pd.DataFrame:
    """Load ``slice_index.csv``, failing loudly if Sprint 1 has not been run."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/03_preprocess.py and "
            f"scripts/04_split.py (Sprint 1) first."
        )
    index = pd.read_csv(path)
    if "split" not in index.columns:
        raise KeyError(
            f"{path} has no 'split' column. Run scripts/04_split.py to add it."
        )
    return index


def assert_no_patient_leakage(index: pd.DataFrame) -> dict[str, set[int]]:
    """Verify the splits are patient-disjoint. Raises if they are not.

    Re-checked here rather than assumed: this is the one invariant that, if
    broken, would silently invalidate every metric produced downstream.
    """
    patients = {
        split: set(group["patient_id"].unique())
        for split, group in index.groupby("split")
    }
    names = sorted(patients)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = patients[a] & patients[b]
            if shared:
                raise RuntimeError(
                    f"Patient leakage between '{a}' and '{b}': {sorted(shared)}"
                )
    return patients


def build_dataloaders(
    *,
    batch_size: int = 8,
    downsample: int = 1,
    augment_train: bool = True,
    num_workers: int = 0,
    max_train_slices: int | None = None,
    max_eval_slices: int | None = None,
    seed: int = 42,
    index: pd.DataFrame | None = None,
) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    """Build train/val/test loaders from the Sprint 1 slice index.

    Parameters
    ----------
    max_train_slices, max_eval_slices:
        Optional caps for smoke tests and for keeping a CPU-only run within a
        sane wall-clock budget. Sampling is **stratified by patient** and
        seeded, so a capped run still covers every training patient rather
        than over-representing whichever patients happen to sort first.
    num_workers:
        Left at 0 by default. This machine has 8 GB of RAM, and worker
        processes each hold their own copy of the index and decompression
        buffers.

    Returns
    -------
    (loaders, frames)
        ``loaders`` maps split name -> DataLoader. ``frames`` maps split name
        -> the index rows actually used, so downstream reporting can state
        exactly what was trained and evaluated on.
    """
    index = load_slice_index() if index is None else index
    assert_no_patient_leakage(index)

    rng = np.random.default_rng(seed)
    loaders: dict[str, DataLoader] = {}
    frames: dict[str, pd.DataFrame] = {}

    for split in ["train", "val", "test"]:
        subset = index[index["split"] == split]
        if subset.empty:
            continue

        cap = max_train_slices if split == "train" else max_eval_slices
        if cap is not None and len(subset) > cap:
            subset = _sample_per_patient(subset, cap, rng)

        is_train = split == "train"
        dataset = LumbarSliceDataset(
            subset,
            augment=augment_train and is_train,
            downsample=downsample,
            seed=seed,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            # drop_last only on train, so evaluation covers every slice.
            drop_last=is_train,
            pin_memory=False,  # no CUDA on this machine
        )
        frames[split] = subset.reset_index(drop=True)

    return loaders, frames


def _sample_per_patient(
    subset: pd.DataFrame, cap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample ``cap`` slices spread evenly across the patients present.

    A plain ``head(cap)`` would take all slices from the first few patients and
    none from the rest, which would make a capped run unrepresentative.
    """
    patients = subset["patient_id"].unique()
    per_patient = max(1, cap // len(patients))

    chosen: list[pd.DataFrame] = []
    for patient_id in patients:
        rows = subset[subset["patient_id"] == patient_id]
        take = min(per_patient, len(rows))
        positions = rng.choice(len(rows), size=take, replace=False)
        chosen.append(rows.iloc[sorted(positions)])

    sampled = pd.concat(chosen)
    # If rounding left room, top up randomly from what is left over.
    if len(sampled) < cap:
        remaining = subset.drop(index=sampled.index)
        extra = min(cap - len(sampled), len(remaining))
        if extra > 0:
            positions = rng.choice(len(remaining), size=extra, replace=False)
            sampled = pd.concat([sampled, remaining.iloc[sorted(positions)]])
    return sampled.sort_values("slice_id")


def describe_loaders(
    loaders: dict[str, DataLoader], frames: dict[str, pd.DataFrame]
) -> str:
    """Readable summary of what each loader will iterate over."""
    lines = []
    for split, loader in loaders.items():
        frame = frames[split]
        lines.append(
            f"  {split:5s}: {len(frame):6,d} slices  "
            f"{frame['patient_id'].nunique():3d} patients  "
            f"{len(loader):5,d} batches  "
            f"augment={loader.dataset.augment}"
        )
    return "\n".join(lines)
