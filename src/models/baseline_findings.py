"""Stage E - baseline disc-level radiological finding estimators.

Deliberately classical
----------------------
Sprint 2 asks for a baseline before any large architecture, and the data
supports that choice rather than merely permitting it: the Sprint 2 scoping
analysis measured **1,518 disc records from 218 patients**. That is far too
little to train a deep classifier from scratch, and it is exactly the regime
where regularised linear models and small tree ensembles are appropriate.

So this module uses:

* logistic regression and a small random forest for the binary findings,
* an **ordinal** decomposition for the Pfirrmann grade,

on two feature families that can be compared and combined:

``geometric``
    Measurements derived from the segmentation masks (disc height, area, AP
    extent, relative-height ratios, vertebral offset, canal width).
``intensity``
    Signal statistics inside the disc and its local references. Relevant
    because the Pfirrmann grade is read from nucleus T2 signal.

Leakage
-------
Every split is taken from the Sprint 1 **patient-level** assignment, and the
feature table is deduplicated to one series per patient before fitting, so a
patient's replicated gradings cannot appear on both sides of the split.
:func:`check_split_integrity` asserts both properties.

Ordinal handling of Pfirrmann
-----------------------------
The grade is ordered (1 < 2 < ... < 5), so it is modelled with the Frank & Hall
decomposition: K-1 binary classifiers estimate ``P(y > k)`` for each threshold,
and the class probabilities are recovered from consecutive differences. This
keeps the ordering information that plain multinomial classification discards,
and needs nothing beyond scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

#: Geometry from the segmentation masks, in millimetres.
GEOMETRIC_FEATURES = [
    "ivd_label",                       # anatomical level; prevalence varies by level
    "height_mm_central",
    "height_mm_anterior",
    "height_mm_posterior",
    "height_mm_min",
    "height_mm_max",
    "area_mm2",
    "ap_extent_mm",
    "aspect_ratio",
    "fill_ratio",
    "height_ratio_to_series_median",
    "height_ratio_to_neighbours",
    "disc_to_vertebra_height_ratio",
    "area_ratio_to_series_median",
    "height_ap_asymmetry",
    "vertebral_ap_offset_mm",
    "canal_width_at_disc_mm",
    "vertebra_above_height_mm",
    "vertebra_below_height_mm",
]

#: Signal statistics inside the disc and its local references.
INTENSITY_FEATURES = [
    "intensity_mean",
    "intensity_std",
    "intensity_p10",
    "intensity_median",
    "intensity_p90",
    "intensity_cv",
    "intensity_nucleus_mean",
    "intensity_annulus_mean",
    "intensity_nucleus_annulus_ratio",
    "intensity_disc_vertebra_ratio",
    "intensity_disc_canal_ratio",
]

FEATURE_SETS: dict[str, list[str]] = {
    "geometric": GEOMETRIC_FEATURES,
    "intensity": INTENSITY_FEATURES,
    "geometric+intensity": GEOMETRIC_FEATURES + INTENSITY_FEATURES,
}

#: Binary targets, using the column names produced by Stage D.
BINARY_TARGETS = [
    "bulging",
    "narrowing",
    "herniation",
    "spondylolisthesis",
    "any_modic",
    "upper_endplate",
    "lower_endplate",
]

ORDINAL_TARGET = "pfirrmann_grade"
PFIRRMANN_CLASSES = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_modelling_table(
    analysis: pd.DataFrame, *, primary_series_only: bool = True
) -> pd.DataFrame:
    """Select the rows used for modelling.

    ``primary_series_only`` keeps one series per patient (chosen in Stage D with
    a T2 preference). This is the correct default: gradings are a patient-level
    annotation, so keeping all 1-3 series would train on the same label two or
    three times and would make cross-validation over-optimistic.
    """
    table = analysis[analysis["has_grading"]].copy()
    if primary_series_only:
        table = table[table["is_primary_series"]]
    return table.reset_index(drop=True)


def check_split_integrity(table: pd.DataFrame) -> dict:
    """Assert patient-disjoint splits and one row per (patient, disc).

    Raises on violation. This is the invariant that makes the Stage E numbers
    meaningful, so it is verified rather than assumed.
    """
    patients = {
        split: set(group["patient_id"]) for split, group in table.groupby("split")
    }
    names = sorted(patients)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = patients[a] & patients[b]
            if shared:
                raise RuntimeError(f"Patient leakage between {a} and {b}: {sorted(shared)}")

    duplicated = table.duplicated(subset=["patient_id", "ivd_label"]).sum()
    if duplicated:
        raise RuntimeError(
            f"{duplicated} duplicated (patient_id, ivd_label) rows - gradings would "
            f"be counted more than once."
        )

    return {
        "patients_per_split": {k: len(v) for k, v in patients.items()},
        "rows_per_split": table["split"].value_counts().to_dict(),
        "duplicate_patient_disc_rows": int(duplicated),
        "leakage_free": True,
    }


def split_xy(
    table: pd.DataFrame, features: list[str], target: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build ``(X, y)`` arrays per split, dropping rows with a missing target."""
    available = [f for f in features if f in table.columns]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        subset = table[(table["split"] == split) & table[target].notna()]
        if subset.empty:
            continue
        out[split] = (
            subset[available].to_numpy(dtype=np.float64),
            subset[target].to_numpy(),
        )
    return out


def make_pipeline(kind: str, *, seed: int = 42) -> Pipeline:
    """Build an estimator pipeline.

    Median imputation is included because some measurements are legitimately
    undefined: the lowest disc has no vertebra below it (the sacrum is excluded
    from this dataset), and the spinal canal is not visible on every slice.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if kind == "logreg":
        steps += [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            )),
        ]
    elif kind == "forest":
        steps += [
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,          # shallow: ~1k training rows
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )),
        ]
    elif kind == "majority":
        steps += [("model", DummyClassifier(strategy="prior"))]
    else:
        raise ValueError(f"Unknown estimator kind: {kind!r}")
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Binary findings
# ---------------------------------------------------------------------------


def evaluate_binary(
    y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Metrics for one binary finding.

    PR-AUC (average precision) is the headline metric rather than ROC-AUC,
    because several of these findings are rare - spondylolisthesis is 2.8% of
    discs - and ROC-AUC is optimistic under that imbalance. The positive count
    is reported alongside every score so no number can be read without knowing
    how much support it has.
    """
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))

    result = {
        "n": int(len(y_true)),
        "n_positive": positives,
        "n_negative": negatives,
        "prevalence": round(positives / len(y_true), 4) if len(y_true) else float("nan"),
    }

    # With no positives (or no negatives) these metrics are undefined. Report
    # NaN rather than a misleading 0 or 1.
    if positives == 0 or negatives == 0:
        result.update({
            "pr_auc": float("nan"), "roc_auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "note": "undefined - test split contains only one class",
        })
        return result

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result.update({
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) else float("nan"), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else float("nan"), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        # A prevalence-rate baseline: what average precision a random ranker
        # would achieve. PR-AUC must beat this to mean anything.
        "pr_auc_baseline": round(positives / len(y_true), 4),
    })
    return result


def fit_binary_target(
    table: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    estimator: str = "logreg",
    seed: int = 42,
) -> dict:
    """Fit and evaluate one binary finding on the patient-level splits."""
    data = split_xy(table, features, target)
    if "train" not in data or "test" not in data:
        return {"target": target, "error": "missing train or test split"}

    x_train, y_train = data["train"]
    y_train = y_train.astype(int)
    if len(np.unique(y_train)) < 2:
        return {"target": target, "error": "training split has a single class"}

    pipeline = make_pipeline(estimator, seed=seed)
    pipeline.fit(x_train, y_train)

    out = {
        "target": target,
        "estimator": estimator,
        "n_features": x_train.shape[1],
        "splits": {},
    }
    for split, (x, y) in data.items():
        y = y.astype(int)
        scores = pipeline.predict_proba(x)[:, 1]
        predictions = pipeline.predict(x)
        out["splits"][split] = evaluate_binary(y, scores, predictions)

    if estimator == "forest":
        importances = pipeline.named_steps["model"].feature_importances_
        available = [f for f in features if f in table.columns]
        order = np.argsort(importances)[::-1][:8]
        out["top_features"] = [
            {"feature": available[i], "importance": round(float(importances[i]), 4)}
            for i in order
        ]
    return out


# ---------------------------------------------------------------------------
# Ordinal Pfirrmann grade
# ---------------------------------------------------------------------------


@dataclass
class OrdinalClassifier:
    """Frank & Hall ordinal classifier built from binary sub-problems.

    For an ordered target with classes ``c_1 < ... < c_K``, fit ``K-1`` binary
    models estimating ``P(y > c_k)``. Class probabilities follow from
    consecutive differences::

        P(y = c_1)     = 1 - P(y > c_1)
        P(y = c_k)     = P(y > c_{k-1}) - P(y > c_k)
        P(y = c_K)     = P(y > c_{K-1})

    Chosen over multinomial classification because it preserves the ordering,
    and over plain regression because it yields calibrated per-class
    probabilities that a report can present. Cumulative probabilities are
    enforced monotone before differencing, since independently fitted binary
    models can otherwise cross and produce small negative probabilities.
    """

    estimator: str = "logreg"
    seed: int = 42
    classes: tuple[int, ...] = PFIRRMANN_CLASSES

    def __post_init__(self) -> None:
        self.models_: list[Pipeline] = []
        self.thresholds_: list[int] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OrdinalClassifier":
        self.models_, self.thresholds_ = [], []
        for threshold in self.classes[:-1]:
            binary_y = (y > threshold).astype(int)
            if len(np.unique(binary_y)) < 2:
                # This threshold is not observed in training; skip it and
                # record nothing, rather than fitting a degenerate model.
                continue
            pipeline = make_pipeline(self.estimator, seed=self.seed)
            pipeline.fit(x, binary_y)
            self.models_.append(pipeline)
            self.thresholds_.append(threshold)
        if not self.models_:
            raise RuntimeError("No usable ordinal thresholds in the training data")
        return self

    def predict_cumulative(self, x: np.ndarray) -> np.ndarray:
        """``P(y > threshold)`` for each fitted threshold, made monotone."""
        cumulative = np.column_stack(
            [model.predict_proba(x)[:, 1] for model in self.models_]
        )
        # Enforce non-increasing cumulative probabilities across thresholds.
        return np.minimum.accumulate(cumulative, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Per-class probabilities over the fitted thresholds' class range."""
        cumulative = self.predict_cumulative(x)
        n_samples = cumulative.shape[0]
        classes = [self.classes[0]] + [t + 1 for t in self.thresholds_]

        probabilities = np.zeros((n_samples, len(classes)))
        probabilities[:, 0] = 1.0 - cumulative[:, 0]
        for i in range(1, len(classes) - 1):
            probabilities[:, i] = cumulative[:, i - 1] - cumulative[:, i]
        probabilities[:, -1] = cumulative[:, -1]

        probabilities = np.clip(probabilities, 0.0, None)
        totals = probabilities.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return probabilities / totals

    def predict(self, x: np.ndarray) -> np.ndarray:
        classes = np.array([self.classes[0]] + [t + 1 for t in self.thresholds_])
        return classes[self.predict_proba(x).argmax(axis=1)]

    def predict_expected(self, x: np.ndarray) -> np.ndarray:
        """Probability-weighted expected grade - a continuous severity estimate.

        Useful for ranking and for future change detection, where a continuous
        value is more sensitive than a rounded integer grade.
        """
        classes = np.array([self.classes[0]] + [t + 1 for t in self.thresholds_])
        return self.predict_proba(x) @ classes


def evaluate_ordinal(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics for the ordinal Pfirrmann target.

    Quadratic weighted kappa is the headline metric: it is the standard for
    ordinal radiological grading and penalises a 1-vs-5 confusion far more
    than a 4-vs-5 confusion, which plain accuracy does not.
    """
    exact = float(np.mean(y_true == y_pred))
    within_one = float(np.mean(np.abs(y_true - y_pred) <= 1))
    return {
        "n": int(len(y_true)),
        "quadratic_weighted_kappa": round(
            float(cohen_kappa_score(y_true, y_pred, weights="quadratic")), 4
        ),
        "linear_weighted_kappa": round(
            float(cohen_kappa_score(y_true, y_pred, weights="linear")), 4
        ),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "exact_agreement": round(exact, 4),
        "within_one_grade": round(within_one, 4),
        "class_support": {
            int(c): int(np.sum(y_true == c)) for c in sorted(np.unique(y_true))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(PFIRRMANN_CLASSES)
        ).tolist(),
        "confusion_matrix_labels": list(PFIRRMANN_CLASSES),
    }


def fit_pfirrmann(
    table: pd.DataFrame,
    features: list[str],
    *,
    estimator: str = "logreg",
    seed: int = 42,
    t2_only: bool = True,
) -> dict:
    """Fit and evaluate the ordinal Pfirrmann model.

    ``t2_only`` restricts to T2-weighted series. The Pfirrmann grade is defined
    on T2 signal, so predicting it from a T1 series is not a well-posed problem;
    the Sprint 2 scoping analysis established that 212 of 218 patients have a
    T2-type series, so the restriction costs little.
    """
    subset = table
    if t2_only and "modality" in table.columns:
        subset = table[table["modality"].isin(["t2", "t2_SPACE"])]

    data = split_xy(subset, features, ORDINAL_TARGET)
    if "train" not in data or "test" not in data:
        return {"target": ORDINAL_TARGET, "error": "missing train or test split"}

    x_train, y_train = data["train"]
    y_train = y_train.astype(int)

    model = OrdinalClassifier(estimator=estimator, seed=seed).fit(x_train, y_train)

    out = {
        "target": ORDINAL_TARGET,
        "estimator": f"ordinal({estimator}) Frank & Hall",
        "t2_only": t2_only,
        "n_features": x_train.shape[1],
        "thresholds_fitted": model.thresholds_,
        "splits": {},
    }
    for split, (x, y) in data.items():
        y = y.astype(int)
        predictions = model.predict(x)
        metrics = evaluate_ordinal(y, predictions)
        expected = model.predict_expected(x)
        metrics["mae_expected_grade"] = round(
            float(mean_absolute_error(y, expected)), 4
        )
        metrics["spearman_expected_vs_true"] = round(
            float(pd.Series(expected).corr(pd.Series(y), method="spearman")), 4
        )
        out["splits"][split] = metrics

    # Majority-class reference: quadratic kappa of a constant prediction is 0
    # by construction, so accuracy is the informative comparison.
    majority = int(pd.Series(y_train).mode().iloc[0])
    y_test = data["test"][1].astype(int)
    out["majority_baseline"] = {
        "predicted_class": majority,
        "exact_agreement": round(float(np.mean(y_test == majority)), 4),
        "mae": round(float(np.mean(np.abs(y_test - majority))), 4),
    }
    return out
