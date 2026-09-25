"""Patient-level train / validation / test splitting.

Why patient level and not slice level
-------------------------------------
Splitting individual slices at random would be badly wrong for this dataset:

* A single series contributes 8-154 adjacent sagittal slices. Neighbouring
  slices of the same patient are nearly identical, so random slice splitting
  puts near-copies of the same image in both train and test.
* A patient contributes 1-3 series (T1, T2, sometimes 3-D T2 SPACE) of the
  *same* anatomy. Inspection additionally found that the T1 and T2 masks of a
  patient are frequently byte-identical - the same annotation shared across
  co-registered series.

Either effect alone would inflate the eventual Dice/IoU scores. The split is
therefore made on ``patient_id``, and every series and slice of a patient
lands in exactly one split. :func:`verify_no_leakage` re-checks this
afterwards rather than trusting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.paths import RANDOM_SEED

DEFAULT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


def _stratification_key(patient_frame: pd.DataFrame) -> pd.Series:
    """Build a coarse key so the splits stay comparable in composition.

    Stratifying on the exact vertebra count would create tiny strata (some
    counts occur once or twice) that cannot be divided three ways. The key is
    therefore deliberately coarse: sex plus a bucketed annotated-vertebra
    count. Rare combinations are folded into an ``other`` bucket by the caller.
    """
    sex = patient_frame["sex"].fillna("unknown").astype(str)
    vertebrae = pd.cut(
        patient_frame["num_vertebrae"].fillna(-1),
        bins=[-2, 5, 6, 7, 8, 99],
        labels=["<=5", "6", "7", "8", ">=9"],
    ).astype(str)
    return sex + "|" + vertebrae


def create_dataset_split(
    pairs: pd.DataFrame,
    *,
    fractions: dict[str, float] | None = None,
    seed: int = RANDOM_SEED,
    stratify: bool = True,
    min_stratum_size: int = 6,
) -> pd.DataFrame:
    """Assign every series to ``train`` / ``val`` / ``test`` by patient.

    Parameters
    ----------
    pairs:
        Series-level table containing ``patient_id`` (and, for stratification,
        ``sex`` and ``num_vertebrae``).
    fractions:
        Target proportions **of patients**. Defaults to 70/15/15.
    seed:
        Random seed. Fixed by default so the split is reproducible.
    stratify:
        Balance the splits by sex and annotated-vertebra count. Strata smaller
        than ``min_stratum_size`` are pooled into one ``other`` stratum so
        every stratum can actually be divided three ways.

    Returns
    -------
    pandas.DataFrame
        ``pairs`` with an added ``split`` column.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    # Collapse to one row per patient: the split is decided at this level.
    patient_columns = [
        c for c in ["patient_id", "sex", "num_vertebrae", "num_discs"] if c in pairs.columns
    ]
    patients = (
        pairs[patient_columns]
        .groupby("patient_id", as_index=False)
        .agg("first")
        .sort_values("patient_id")  # deterministic starting order
        .reset_index(drop=True)
    )

    if stratify and {"sex", "num_vertebrae"} <= set(patients.columns):
        key = _stratification_key(patients)
        counts = key.value_counts()
        small = counts[counts < min_stratum_size].index
        patients["_stratum"] = key.where(~key.isin(small), "other")
    else:
        patients["_stratum"] = "all"

    rng = np.random.default_rng(seed)
    assignments: dict[int, str] = {}

    # Split each stratum independently, then concatenate.
    for stratum, group in patients.groupby("_stratum", sort=True):
        ids = group["patient_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)

        n_train = int(round(fractions["train"] * n))
        n_val = int(round(fractions["val"] * n))
        # Give the remainder to test so the three parts always sum to n.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        for patient_id in ids[:n_train]:
            assignments[int(patient_id)] = "train"
        for patient_id in ids[n_train : n_train + n_val]:
            assignments[int(patient_id)] = "val"
        for patient_id in ids[n_train + n_val :]:
            assignments[int(patient_id)] = "test"

    out = pairs.copy()
    out["split"] = out["patient_id"].map(assignments)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "patient_id"].unique()
        raise RuntimeError(f"Patients left unassigned by the split: {missing}")
    return out


def verify_no_leakage(split_frame: pd.DataFrame, id_column: str = "patient_id") -> dict:
    """Confirm no patient appears in more than one split.

    Returns a dict with ``ok`` plus the offending ids, so the check can be
    asserted in a script and printed in a report.
    """
    per_split = split_frame.groupby("split")[id_column].apply(set).to_dict()
    overlaps: dict[str, list] = {}
    names = sorted(per_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = per_split[a] & per_split[b]
            if shared:
                overlaps[f"{a}&{b}"] = sorted(shared)

    counted = sum(len(v) for v in per_split.values())
    return {
        "ok": not overlaps,
        "overlaps": overlaps,
        "ids_per_split": {k: len(v) for k, v in per_split.items()},
        "total_ids_counted": counted,
        "total_ids_unique": int(split_frame[id_column].nunique()),
    }


def summarise_split(
    split_frame: pd.DataFrame, slice_index: pd.DataFrame | None = None
) -> dict:
    """Count patients, series and slices per split.

    ``slice_index`` is the per-slice table produced by preprocessing; when it
    is supplied the slice counts are real rather than estimated.
    """
    summary: dict = {}
    for split in ["train", "val", "test"]:
        rows = split_frame[split_frame["split"] == split]
        entry = {
            "patients": int(rows["patient_id"].nunique()),
            "series": int(len(rows)),
            "slices": 0,
            "modalities": rows["modality"].value_counts().to_dict(),
        }
        if slice_index is not None and not slice_index.empty:
            entry["slices"] = int((slice_index["split"] == split).sum())
        summary[split] = entry
    return summary


def split_fraction_table(summary: dict) -> pd.DataFrame:
    """Render the split summary as a table with achieved percentages."""
    records = []
    totals = {
        key: sum(summary[s][key] for s in summary) for key in ["patients", "series", "slices"]
    }
    for split, counts in summary.items():
        record = {"split": split}
        for key in ["patients", "series", "slices"]:
            value = counts[key]
            share = 100 * value / totals[key] if totals[key] else 0.0
            record[key] = value
            record[f"{key}_pct"] = round(share, 1)
        records.append(record)
    return pd.DataFrame(records)
