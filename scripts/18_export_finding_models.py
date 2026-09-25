"""Sprint 6 - persist the already-validated finding estimators for serving.

No segmentation model is retrained and no configuration is chosen here. Sprint 2
already selected and published the best estimator and feature set for every
finding target in ``outputs/reports/baseline_findings.md``. Those estimators were
refitted in-process every time they were needed, which is wrong for a served
application: the median-imputation statistics and the fitted coefficients would
be rebuilt per request.

This script fits each published configuration **once on the training split** and
writes it to disk with its provenance, so the backend loads instead of fitting.

Control
-------
Each fitted model is scored on the test split and compared against the value
published in Sprint 2. Agreement proves the configuration was reproduced rather
than re-selected. The test split is used only for that verification; nothing is
chosen from it.

Deliberately excluded
---------------------
``spondylolisthesis``
    5 positive discs in the test split. Sprint 2 concluded the point estimate is
    too uncertain to use, so no model is shipped and the API reports the field as
    unsupported.
``modic`` type (0 / I / II / III)
    Never modelled. Only the binarised ``any_modic`` was. The nominal type stays
    unsupported rather than being guessed from the binary model.

Outputs
-------
    outputs/models/findings/<target>.joblib
    outputs/models/findings/manifest.json

Usage
-----
    python scripts/18_export_finding_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline_findings import (  # noqa: E402
    FEATURE_SETS,
    ORDINAL_TARGET,
    PFIRRMANN_CLASSES,
    OrdinalClassifier,
    check_split_integrity,
    evaluate_binary,
    evaluate_ordinal,
    make_pipeline,
    prepare_modelling_table,
    split_xy,
)
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT  # noqa: E402
from src.utils.reporting import save_json  # noqa: E402

MODEL_DIR = OUTPUTS_DIR / "models" / "findings"
ANALYSIS_CSV = OUTPUTS_DIR / "reports" / "disc_analysis.csv"

#: Published Sprint 2 selections and their test scores, transcribed from
#: outputs/reports/baseline_findings.md. Used as the reproduction control.
BINARY_PLAN = {
    "narrowing": {
        "features": "geometric+intensity", "estimator": "logreg",
        "published_pr_auc": 0.8779, "prevalence": 0.3230, "n_positive": 73,
        "label": "Disc narrowing",
        "strength": "strong",
    },
    "lower_endplate": {
        "features": "geometric+intensity", "estimator": "logreg",
        "published_pr_auc": 0.7962, "prevalence": 0.3894, "n_positive": 88,
        "label": "Lower endplate change",
        "strength": "moderate",
    },
    "bulging": {
        "features": "geometric+intensity", "estimator": "forest",
        "published_pr_auc": 0.7610, "prevalence": 0.4469, "n_positive": 101,
        "label": "Disc bulging",
        "strength": "moderate",
    },
    "upper_endplate": {
        "features": "geometric+intensity", "estimator": "logreg",
        "published_pr_auc": 0.6688, "prevalence": 0.3363, "n_positive": 76,
        "label": "Upper endplate change",
        "strength": "modest",
    },
    "any_modic": {
        "features": "geometric+intensity", "estimator": "logreg",
        "published_pr_auc": 0.6505, "prevalence": 0.2876, "n_positive": 65,
        "label": "Modic-type change present",
        "strength": "modest",
    },
    "herniation": {
        "features": "geometric", "estimator": "logreg",
        "published_pr_auc": 0.4362, "prevalence": 0.0796, "n_positive": 18,
        "label": "Disc herniation",
        "strength": "weak",
    },
}

ORDINAL_PLAN = {
    "features": "geometric+intensity", "estimator": "forest",
    "published_qwk": 0.6718, "published_within_one": 0.8531,
    "published_mae": 0.7678, "n_test": 211,
    "label": "Pfirrmann grade",
    "strength": "moderate",
    "t2_only": True,
}

UNSUPPORTED = {
    "spondylolisthesis": (
        "Not served. Only 5 positive discs exist in the held-out test split, so "
        "Sprint 2 measured a PR-AUC between 0.19 and 0.37 and concluded the "
        "estimate is too uncertain to report."
    ),
    "modic_type": (
        "Not served. The nominal Modic type (0 / I / II / III) was never "
        "modelled; only the binary 'any Modic change' target was. Reporting a "
        "type would require guessing beyond what was validated."
    ),
}


def main() -> None:
    if not ANALYSIS_CSV.exists():
        raise SystemExit(f"Missing {ANALYSIS_CSV}. Run scripts/06_disc_analysis.py first.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    analysis = pd.read_csv(ANALYSIS_CSV)
    table = prepare_modelling_table(analysis, primary_series_only=True)
    integrity = check_split_integrity(table)
    print(f"Modelling table: {len(table):,} disc rows, "
          f"{table['patient_id'].nunique()} patients")
    print(f"  patient-disjoint splits verified: {integrity['leakage_free']}")
    print(f"  rows per split: {integrity['rows_per_split']}")

    manifest: dict = {
        "generated_for": "sprint6 serving",
        "source_table": str(ANALYSIS_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "fitted_on_split": "train",
        "test_used_for": "reproduction control only, no selection",
        "split_integrity": integrity,
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "supported": {},
        "unsupported": UNSUPPORTED,
    }
    controls: list[dict] = []

    # ---------------- binary findings ----------------
    print("\nBinary findings")
    for target, plan in BINARY_PLAN.items():
        features = FEATURE_SETS[plan["features"]]
        data = split_xy(table, features, target)
        if "train" not in data:
            print(f"  {target}: no training rows, skipped")
            continue
        x_train, y_train = data["train"]
        pipeline = make_pipeline(plan["estimator"], seed=42)
        pipeline.fit(x_train, y_train.astype(int))

        measured = None
        if "test" in data:
            x_test, y_test = data["test"]
            score = pipeline.predict_proba(x_test)[:, 1]
            prediction = pipeline.predict(x_test)
            measured = evaluate_binary(y_test.astype(int), score, prediction)

        used_features = [f for f in features if f in table.columns]
        payload = {
            "kind": "binary",
            "target": target,
            "label": plan["label"],
            "estimator": plan["estimator"],
            "feature_set": plan["features"],
            "features": used_features,
            "pipeline": pipeline,
            "validated": {
                "metric": "test_pr_auc",
                "published": plan["published_pr_auc"],
                "measured": round(float(measured["pr_auc"]), 4) if measured else None,
                "prevalence_baseline": plan["prevalence"],
                "n_test_positive": plan["n_positive"],
                "strength": plan["strength"],
            },
        }
        path = MODEL_DIR / f"{target}.joblib"
        joblib.dump(payload, path)

        delta = (payload["validated"]["measured"] - plan["published_pr_auc"]
                 if measured else None)
        agrees = delta is not None and abs(delta) < 0.0051
        controls.append({"target": target, "published": plan["published_pr_auc"],
                         "measured": payload["validated"]["measured"],
                         "agrees": bool(agrees)})
        manifest["supported"][target] = {
            k: v for k, v in payload.items() if k != "pipeline"
        }
        manifest["supported"][target]["artifact"] = path.name
        print(f"  {target:<18} PR-AUC published {plan['published_pr_auc']:.4f}  "
              f"measured {payload['validated']['measured']}  "
              f"{'OK' if agrees else 'MISMATCH'}  ({plan['strength']})")

    # ---------------- Pfirrmann ----------------
    print("\nPfirrmann grade (ordinal)")
    features = FEATURE_SETS[ORDINAL_PLAN["features"]]
    ordinal_table = table
    if ORDINAL_PLAN["t2_only"]:
        ordinal_table = table[table["modality"].isin(["t2", "t2_SPACE"])]
    data = split_xy(ordinal_table, features, ORDINAL_TARGET)
    x_train, y_train = data["train"]
    model = OrdinalClassifier(estimator=ORDINAL_PLAN["estimator"], seed=42)
    model.fit(x_train, y_train.astype(int))

    measured = None
    if "test" in data:
        x_test, y_test = data["test"]
        measured = evaluate_ordinal(y_test.astype(int), model.predict(x_test))

    used_features = [f for f in features if f in ordinal_table.columns]
    payload = {
        "kind": "ordinal",
        "target": ORDINAL_TARGET,
        "label": ORDINAL_PLAN["label"],
        "estimator": ORDINAL_PLAN["estimator"],
        "feature_set": ORDINAL_PLAN["features"],
        "features": used_features,
        "classes": list(PFIRRMANN_CLASSES),
        "modalities": ["t2", "t2_SPACE"],
        "model": model,
        "validated": {
            "metric": "test_quadratic_weighted_kappa",
            "published": ORDINAL_PLAN["published_qwk"],
            "measured": round(float(measured["quadratic_weighted_kappa"]), 4)
            if measured else None,
            "published_within_one_grade": ORDINAL_PLAN["published_within_one"],
            "measured_within_one_grade": round(float(measured["within_one_grade"]), 4)
            if measured else None,
            "published_mae": ORDINAL_PLAN["published_mae"],
            "n_test": ORDINAL_PLAN["n_test"],
            "strength": ORDINAL_PLAN["strength"],
            "end_to_end_qwk_sprint5": 0.6543,
            "note": (
                "The published value is measured from ground-truth masks. The "
                "end-to-end value through predicted masks and Sprint 5 indexing "
                "is 0.6543, which is the figure that applies to an uploaded study."
            ),
        },
    }
    path = MODEL_DIR / "pfirrmann_grade.joblib"
    joblib.dump(payload, path)
    delta = (payload["validated"]["measured"] - ORDINAL_PLAN["published_qwk"]
             if measured else None)
    agrees = delta is not None and abs(delta) < 0.0051
    controls.append({"target": ORDINAL_TARGET,
                     "published": ORDINAL_PLAN["published_qwk"],
                     "measured": payload["validated"]["measured"],
                     "agrees": bool(agrees)})
    manifest["supported"][ORDINAL_TARGET] = {
        k: v for k, v in payload.items() if k != "model"
    }
    manifest["supported"][ORDINAL_TARGET]["artifact"] = path.name
    print(f"  pfirrmann_grade    QWK published {ORDINAL_PLAN['published_qwk']:.4f}  "
          f"measured {payload['validated']['measured']}  "
          f"{'OK' if agrees else 'MISMATCH'}")

    manifest["reproduction_control"] = controls
    manifest["all_reproduced"] = all(c["agrees"] for c in controls)
    save_json(manifest, MODEL_DIR / "manifest.json")

    print("\n" + "=" * 70)
    n_ok = sum(1 for c in controls if c["agrees"])
    print(f"Persisted {len(controls)} estimators to "
          f"{MODEL_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Reproduction control: {n_ok}/{len(controls)} match the published "
          f"Sprint 2 values")
    print(f"Unsupported and reported as such: {', '.join(UNSUPPORTED)}")
    if not manifest["all_reproduced"]:
        print("WARNING: at least one estimator did not reproduce its published score.")
    print("=" * 70)
    if not manifest["all_reproduced"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
