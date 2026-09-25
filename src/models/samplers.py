"""Epoch-varying slice sampling (Sprint 3 audit, section C proposal).

The problem this solves
-----------------------
The current sampler (``src.models.data._sample_per_patient``) is called **once**,
before training, with a run-level seed. The resulting 2,560-slice subset is
therefore fixed for the whole run: every epoch iterates the same slices in a
different order. Both completed runs never saw the other 6,568 training slices -
72% of the available training data.

The approach
------------
Keep the per-epoch cost identical (so wall-clock per epoch is unchanged) but make
the *subset* a function of the epoch. Over a number of epochs the model then sees
all 9,128 training slices.

Two strategies are provided:

``rotating``
    Partition each patient's slices into deterministic shards and use shard
    ``epoch % n_shards`` in each epoch. Coverage is exact and predictable: every
    slice is seen once every ``n_shards`` epochs, and per-epoch composition
    stays balanced across patients.

``reshuffled``
    Draw a fresh per-patient random sample each epoch, seeded by
    ``(seed, epoch)``. Simpler, but coverage is probabilistic - some slices may
    be seen several times and others not at all within a given number of epochs.

``rotating`` is the recommended default because coverage is guaranteed rather
than expected.

Leakage safety
--------------
Both strategies only ever partition rows that are already ``split == 'train'``.
They never look at validation or test rows. :func:`verify_no_leakage_in_schedule`
checks that property against the full index for every epoch of a schedule.

Reproducibility
---------------
Both are pure functions of ``(index, seed, epoch)``. Re-running or resuming an
epoch reproduces that epoch's subset exactly.

Nothing here is wired into training yet - this module exists so the Sprint 3
proposal can be verified before any long run is launched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_SPLIT = "train"


def _patient_shards(
    slice_ids: np.ndarray, n_shards: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Split one patient's slice ids into ``n_shards`` near-equal shards.

    Slices are shuffled once with a patient-independent generator before
    splitting, so a shard is not a contiguous block of adjacent (highly
    correlated) sagittal slices.
    """
    shuffled = slice_ids.copy()
    rng.shuffle(shuffled)
    return [np.asarray(part) for part in np.array_split(shuffled, n_shards)]


def rotating_epoch_sample(
    index: pd.DataFrame,
    *,
    epoch: int,
    slices_per_epoch: int = 2560,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the training slices for ``epoch`` under the rotating strategy.

    ``n_shards`` is derived from the requested per-epoch size, so the per-epoch
    cost matches the current configuration while the content rotates.

    Parameters
    ----------
    index:
        The full slice index (all splits). Only ``split == 'train'`` rows are
        considered.
    epoch:
        1-based epoch number.
    slices_per_epoch:
        Target number of slices per epoch. Pass ``0`` or ``None`` to use every
        training slice in every epoch.
    """
    train = index[index["split"] == TRAIN_SPLIT]
    if not slices_per_epoch or slices_per_epoch >= len(train):
        return train.reset_index(drop=True)

    n_shards = max(1, int(round(len(train) / slices_per_epoch)))
    shard_position = (epoch - 1) % n_shards

    chosen: list[np.ndarray] = []
    for patient_id, group in train.groupby("patient_id", sort=True):
        # Seeded per patient so a patient's shard layout is stable across
        # epochs and independent of how other patients are processed.
        rng = np.random.default_rng([seed, int(patient_id)])
        shards = _patient_shards(group["slice_id"].to_numpy(), n_shards, rng)
        chosen.append(shards[shard_position])

    selected = np.concatenate(chosen) if chosen else np.array([])
    return (
        train[train["slice_id"].isin(selected)]
        .sort_values("slice_id")
        .reset_index(drop=True)
    )


def reshuffled_epoch_sample(
    index: pd.DataFrame,
    *,
    epoch: int,
    slices_per_epoch: int = 2560,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the training slices for ``epoch`` by fresh per-epoch sampling."""
    train = index[index["split"] == TRAIN_SPLIT]
    if not slices_per_epoch or slices_per_epoch >= len(train):
        return train.reset_index(drop=True)

    patients = train["patient_id"].unique()
    per_patient = max(1, slices_per_epoch // len(patients))

    chosen: list[pd.DataFrame] = []
    for patient_id in patients:
        group = train[train["patient_id"] == patient_id]
        rng = np.random.default_rng([seed, epoch, int(patient_id)])
        take = min(per_patient, len(group))
        positions = rng.choice(len(group), size=take, replace=False)
        chosen.append(group.iloc[sorted(positions)])

    sampled = pd.concat(chosen)
    if len(sampled) < slices_per_epoch:
        remaining = train.drop(index=sampled.index)
        extra = min(slices_per_epoch - len(sampled), len(remaining))
        if extra > 0:
            rng = np.random.default_rng([seed, epoch, 999])
            positions = rng.choice(len(remaining), size=extra, replace=False)
            sampled = pd.concat([sampled, remaining.iloc[sorted(positions)]])
    return sampled.sort_values("slice_id").reset_index(drop=True)


SAMPLERS = {
    "rotating": rotating_epoch_sample,
    "reshuffled": reshuffled_epoch_sample,
    "fixed": None,  # the current behaviour, for comparison
}


def measure_coverage(
    index: pd.DataFrame,
    *,
    strategy: str = "rotating",
    n_epochs: int = 30,
    slices_per_epoch: int = 2560,
    seed: int = 42,
) -> dict:
    """Measure how much of the training set a schedule actually exposes.

    Simulates the schedule without loading any image data, so it is cheap and
    can be run as a verification step before committing to a long training run.
    """
    train = index[index["split"] == TRAIN_SPLIT]
    all_ids = set(train["slice_id"])

    if strategy == "fixed":
        # Reproduce the current behaviour: sample once, reuse every epoch.
        from src.models.data import _sample_per_patient

        fixed = _sample_per_patient(
            train, slices_per_epoch, np.random.default_rng(seed)
        )
        per_epoch_ids = [set(fixed["slice_id"])] * n_epochs
    else:
        sampler = SAMPLERS[strategy]
        per_epoch_ids = [
            set(
                sampler(
                    index, epoch=epoch, slices_per_epoch=slices_per_epoch, seed=seed
                )["slice_id"]
            )
            for epoch in range(1, n_epochs + 1)
        ]

    seen_counts: dict[str, int] = {}
    cumulative_coverage: list[float] = []
    seen: set[str] = set()
    for ids in per_epoch_ids:
        for slice_id in ids:
            seen_counts[slice_id] = seen_counts.get(slice_id, 0) + 1
        seen |= ids
        cumulative_coverage.append(round(100 * len(seen) / len(all_ids), 2))

    counts = np.array([seen_counts.get(s, 0) for s in all_ids])
    epochs_to_full = next(
        (i + 1 for i, pct in enumerate(cumulative_coverage) if pct >= 99.99), None
    )

    return {
        "strategy": strategy,
        "n_epochs_simulated": n_epochs,
        "slices_per_epoch_requested": slices_per_epoch,
        "slices_per_epoch_actual": int(np.mean([len(ids) for ids in per_epoch_ids])),
        "train_slices_total": len(all_ids),
        "unique_slices_seen": int((counts > 0).sum()),
        "coverage_pct": round(100 * float((counts > 0).mean()), 2),
        "slices_never_seen": int((counts == 0).sum()),
        "epochs_to_full_coverage": epochs_to_full,
        "times_seen": {
            "min": int(counts.min()),
            "mean": round(float(counts.mean()), 2),
            "max": int(counts.max()),
            "std": round(float(counts.std()), 2),
        },
        "cumulative_coverage_pct_by_epoch": cumulative_coverage,
    }


def verify_no_leakage_in_schedule(
    index: pd.DataFrame,
    *,
    strategy: str = "rotating",
    n_epochs: int = 30,
    slices_per_epoch: int = 2560,
    seed: int = 42,
) -> dict:
    """Assert no validation/test slice or patient ever enters a training epoch.

    Checks both identifiers, because either alone would be insufficient: a slice
    id could in principle be reused, and a patient id could appear via a
    different series.
    """
    val_test = index[index["split"] != TRAIN_SPLIT]
    forbidden_slices = set(val_test["slice_id"])
    forbidden_patients = set(val_test["patient_id"])
    train_patients = set(index[index["split"] == TRAIN_SPLIT]["patient_id"])

    violations: list[dict] = []
    for epoch in range(1, n_epochs + 1):
        if strategy == "fixed":
            from src.models.data import _sample_per_patient

            sample = _sample_per_patient(
                index[index["split"] == TRAIN_SPLIT],
                slices_per_epoch,
                np.random.default_rng(seed),
            )
        else:
            sample = SAMPLERS[strategy](
                index, epoch=epoch, slices_per_epoch=slices_per_epoch, seed=seed
            )

        bad_slices = set(sample["slice_id"]) & forbidden_slices
        bad_patients = set(sample["patient_id"]) & forbidden_patients
        wrong_split = set(sample["split"].unique()) - {TRAIN_SPLIT}
        if bad_slices or bad_patients or wrong_split:
            violations.append(
                {
                    "epoch": epoch,
                    "leaked_slices": sorted(bad_slices)[:10],
                    "leaked_patients": sorted(bad_patients)[:10],
                    "unexpected_splits": sorted(wrong_split),
                }
            )

    return {
        "strategy": strategy,
        "n_epochs_checked": n_epochs,
        "train_patients": len(train_patients),
        "val_test_patients_excluded": len(forbidden_patients),
        "val_test_slices_excluded": len(forbidden_slices),
        "violations": violations,
        "leakage_free": not violations,
    }


def verify_determinism(
    index: pd.DataFrame,
    *,
    strategy: str = "rotating",
    epoch: int = 7,
    slices_per_epoch: int = 2560,
    seed: int = 42,
) -> dict:
    """Confirm an epoch's subset is reproducible, and differs between epochs."""
    sampler = SAMPLERS[strategy]
    first = set(sampler(index, epoch=epoch, slices_per_epoch=slices_per_epoch,
                        seed=seed)["slice_id"])
    again = set(sampler(index, epoch=epoch, slices_per_epoch=slices_per_epoch,
                        seed=seed)["slice_id"])
    other = set(sampler(index, epoch=epoch + 1, slices_per_epoch=slices_per_epoch,
                        seed=seed)["slice_id"])
    other_seed = set(sampler(index, epoch=epoch, slices_per_epoch=slices_per_epoch,
                             seed=seed + 1)["slice_id"])

    return {
        "strategy": strategy,
        "same_epoch_same_seed_identical": first == again,
        "different_epoch_differs": first != other,
        "overlap_with_next_epoch": len(first & other),
        "different_seed_differs": first != other_seed,
        "n_slices": len(first),
    }
