# Stage E - Baseline Disc-Level Finding Assessment

*Classical baselines on segmentation geometry and ROI intensity features*

Generated: 2026-09-22 23:10:14

**This is a baseline, not a finished system, and not a clinical result.** Its purpose is to establish what the available disc-level data supports before any larger architecture is considered. Every score below is on **held-out test patients** from the Sprint 1 patient-level split.


## 1. Data and leakage control

| Field | Value |
| --- | --- |
| Disc records used | 1511 |
| Patients | 218 |
| One series per patient | True |
| Rows per split | `train`=1057, `val`=228, `test`=226 |
| Patients per split | `test`=33, `train`=152, `val`=33 |
| Duplicate (patient, disc) rows | 0 |
| Patient-disjoint splits verified | True |

Gradings are a **patient-level** annotation shared by a patient's 1-3 series. The table is therefore deduplicated to one series per patient (T2 preferred, because the Pfirrmann grade is defined on T2 signal) before fitting. Keeping all series would train on the same label two or three times and inflate every score. Both the deduplication and the patient-disjointness are asserted in code, not assumed.


## 2. Feature families

| family | n | source | examples |
| --- | --- | --- | --- |
| `geometric` | 19 | segmentation masks, measured in mm at Sprint 1's 1.0 mm/px | central/anterior/posterior disc height, area, AP extent, height relative to series median and to neighbours, disc-to-vertebra height ratio, vertebral AP offset, canal width |
| `intensity` | 11 | preprocessed image inside the disc and its references | disc mean/median/percentile signal, coefficient of variation, nucleus vs annulus, disc-to-vertebra and disc-to-canal signal ratios |
| `geometric+intensity` | 30 | both | union of the two |

The two families are reported separately on purpose. The Pfirrmann grade is defined by nucleus **signal** plus disc height, so intensity features should matter for it; disc narrowing and spondylolisthesis are **geometric** by definition. Splitting the comparison shows whether the models behave consistently with that expectation, which is a useful check that the features measure what they claim to.


## 3. Binary findings - test-set results

**PR-AUC is the headline metric**, not ROC-AUC: several findings are rare (spondylolisthesis is ~2.8% of discs) and ROC-AUC is optimistic under that imbalance. The `prevalence baseline` column is what a random ranker scores - a model must clearly beat it to carry information. `majority` is a predict-the-prior reference.

| target | features | estimator | test_n | test_n_positive | test_pr_auc | test_pr_auc_baseline | test_roc_auc | test_balanced_accuracy | test_sensitivity | test_specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulging | geometric | logreg | 226 | 101 | 0.685 | 0.4469 | 0.8088 | 0.7631 | 0.7822 | 0.744 |
| bulging | geometric | forest | 226 | 101 | 0.6921 | 0.4469 | 0.8156 | 0.7827 | 0.8614 | 0.704 |
| bulging | intensity | logreg | 226 | 101 | 0.6841 | 0.4469 | 0.7856 | 0.7608 | 0.8416 | 0.68 |
| bulging | intensity | forest | 226 | 101 | 0.681 | 0.4469 | 0.7791 | 0.72 | 0.7921 | 0.648 |
| bulging | geometric+intensity | logreg | 226 | 101 | 0.7545 | 0.4469 | 0.8535 | 0.8059 | 0.8119 | 0.8 |
| bulging | geometric+intensity | forest | 226 | 101 | 0.761 | 0.4469 | 0.859 | 0.7998 | 0.8317 | 0.768 |
| narrowing | geometric | logreg | 226 | 73 | 0.8354 | 0.323 | 0.9063 | 0.8254 | 0.7945 | 0.8562 |
| narrowing | geometric | forest | 226 | 73 | 0.8254 | 0.323 | 0.9128 | 0.845 | 0.7945 | 0.8954 |
| narrowing | intensity | logreg | 226 | 73 | 0.776 | 0.323 | 0.8715 | 0.76 | 0.7945 | 0.7255 |
| narrowing | intensity | forest | 226 | 73 | 0.7339 | 0.323 | 0.85 | 0.7391 | 0.7397 | 0.7386 |
| narrowing | geometric+intensity | logreg | 226 | 73 | 0.8779 | 0.323 | 0.9286 | 0.842 | 0.8082 | 0.8758 |
| narrowing | geometric+intensity | forest | 226 | 73 | 0.8373 | 0.323 | 0.9099 | 0.8388 | 0.8082 | 0.8693 |
| herniation | geometric | logreg | 226 | 18 | 0.4362 | 0.0796 | 0.8478 | 0.7057 | 0.5556 | 0.8558 |
| herniation | geometric | forest | 226 | 18 | 0.4215 | 0.0796 | 0.8365 | 0.68 | 0.3889 | 0.9712 |
| herniation | intensity | logreg | 226 | 18 | 0.2211 | 0.0796 | 0.805 | 0.7666 | 0.8889 | 0.6442 |
| herniation | intensity | forest | 226 | 18 | 0.1268 | 0.0796 | 0.7081 | 0.5572 | 0.2778 | 0.8365 |
| herniation | geometric+intensity | logreg | 226 | 18 | 0.4348 | 0.0796 | 0.855 | 0.7431 | 0.6111 | 0.875 |
| herniation | geometric+intensity | forest | 226 | 18 | 0.2963 | 0.0796 | 0.8632 | 0.5617 | 0.1667 | 0.9567 |
| spondylolisthesis | geometric | logreg | 226 | 5 | 0.1928 | 0.0221 | 0.905 | 0.9118 | 1 | 0.8235 |
| spondylolisthesis | geometric | forest | 226 | 5 | 0.3343 | 0.0221 | 0.9348 | 0.5932 | 0.2 | 0.9864 |
| spondylolisthesis | intensity | logreg | 226 | 5 | 0.2281 | 0.0221 | 0.8552 | 0.7575 | 0.8 | 0.7149 |
| spondylolisthesis | intensity | forest | 226 | 5 | 0.3666 | 0.0221 | 0.8127 | 0.6955 | 0.4 | 0.991 |
| spondylolisthesis | geometric+intensity | logreg | 226 | 5 | 0.2857 | 0.0221 | 0.8932 | 0.8253 | 0.8 | 0.8507 |
| spondylolisthesis | geometric+intensity | forest | 226 | 5 | 0.2473 | 0.0221 | 0.9403 | 0.5955 | 0.2 | 0.991 |
| any_modic | geometric | logreg | 226 | 65 | 0.6038 | 0.2876 | 0.7659 | 0.671 | 0.6154 | 0.7267 |
| any_modic | geometric | forest | 226 | 65 | 0.6049 | 0.2876 | 0.7619 | 0.7145 | 0.6154 | 0.8137 |
| any_modic | intensity | logreg | 226 | 65 | 0.6151 | 0.2876 | 0.8305 | 0.7526 | 0.7846 | 0.7205 |
| any_modic | intensity | forest | 226 | 65 | 0.5464 | 0.2876 | 0.7925 | 0.694 | 0.6923 | 0.6957 |
| any_modic | geometric+intensity | logreg | 226 | 65 | 0.6505 | 0.2876 | 0.8437 | 0.7573 | 0.7692 | 0.7453 |
| any_modic | geometric+intensity | forest | 226 | 65 | 0.6052 | 0.2876 | 0.8178 | 0.7018 | 0.6769 | 0.7267 |
| upper_endplate | geometric | logreg | 226 | 76 | 0.6557 | 0.3363 | 0.7927 | 0.7255 | 0.6711 | 0.78 |
| upper_endplate | geometric | forest | 226 | 76 | 0.605 | 0.3363 | 0.7796 | 0.7026 | 0.6053 | 0.8 |
| upper_endplate | intensity | logreg | 226 | 76 | 0.5055 | 0.3363 | 0.7217 | 0.6814 | 0.7895 | 0.5733 |
| upper_endplate | intensity | forest | 226 | 76 | 0.4836 | 0.3363 | 0.6943 | 0.6618 | 0.7237 | 0.6 |
| upper_endplate | geometric+intensity | logreg | 226 | 76 | 0.6688 | 0.3363 | 0.8107 | 0.7348 | 0.7763 | 0.6933 |
| upper_endplate | geometric+intensity | forest | 226 | 76 | 0.5717 | 0.3363 | 0.7542 | 0.6625 | 0.6184 | 0.7067 |
| lower_endplate | geometric | logreg | 226 | 88 | 0.77 | 0.3894 | 0.8158 | 0.7524 | 0.6932 | 0.8116 |
| lower_endplate | geometric | forest | 226 | 88 | 0.7209 | 0.3894 | 0.7938 | 0.7137 | 0.5795 | 0.8478 |
| lower_endplate | intensity | logreg | 226 | 88 | 0.6426 | 0.3894 | 0.7589 | 0.685 | 0.7614 | 0.6087 |
| lower_endplate | intensity | forest | 226 | 88 | 0.6336 | 0.3894 | 0.7343 | 0.667 | 0.6818 | 0.6522 |
| lower_endplate | geometric+intensity | logreg | 226 | 88 | 0.7962 | 0.3894 | 0.8353 | 0.7518 | 0.75 | 0.7536 |
| lower_endplate | geometric+intensity | forest | 226 | 88 | 0.7062 | 0.3894 | 0.7789 | 0.6857 | 0.625 | 0.7464 |


## 4. Pfirrmann grade - test-set results

Modelled as an **ordinal** target with the Frank & Hall decomposition: four binary models estimate `P(grade > k)` and the per-class probabilities come from consecutive differences. Quadratic weighted kappa (QWK) is the headline metric - it is the convention for ordinal radiological grading and penalises a 1-vs-5 confusion far more than a 4-vs-5 one. Restricted to T2 / T2 SPACE series.

| features | estimator | test_n | test_qwk | test_mae | test_exact | test_within_one | test_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| geometric | logreg | 211 | 0.4793 | 1.237 | 0.2417 | 0.6635 | 0.6561 |
| geometric | forest | 211 | 0.52 | 1.152 | 0.2607 | 0.7014 | 0.6985 |
| intensity | logreg | 211 | 0.6258 | 0.9763 | 0.3033 | 0.782 | 0.7201 |
| intensity | forest | 211 | 0.6376 | 0.8531 | 0.3697 | 0.8246 | 0.7171 |
| geometric+intensity | logreg | 211 | 0.6633 | 0.8341 | 0.3886 | 0.8199 | 0.7526 |
| geometric+intensity | forest | 211 | 0.6718 | 0.7678 | 0.4171 | 0.8531 | 0.7411 |

| Field | Value |
| --- | --- |
| Majority-class baseline predicted grade | 3 |
| Majority-class exact agreement | 0.2749 |
| Majority-class MAE | 0.9526 |

A constant prediction has a quadratic weighted kappa of 0 by construction, so accuracy is the meaningful comparison against it.


## 5. How to read these numbers

- **Test splits are small.** The test set holds 33 patients; for a finding with ~3% prevalence that is a single-digit number of positive discs. Point estimates for the rare findings are therefore very uncertain, and the positive count is printed next to every score for that reason.
- **These are the features a baseline can see.** Geometry and summary intensity describe a disc coarsely. A focal herniation is a local shape detail that a handful of summary statistics is not expected to capture well - a low score for herniation is an honest reflection of the feature set, not evidence the task is impossible.
- **No composite severity score is produced.** Each finding is reported on its own, as the dataset annotates it.
- **The segmentation used here is ground truth.** These numbers isolate the finding-assessment step from segmentation error. Running the same pipeline on predicted masks is what quantifies the combined error, and the disc-identification audit in `outputs/metrics/disc_identification_test.json` shows how much identity error a predicted mask introduces.

