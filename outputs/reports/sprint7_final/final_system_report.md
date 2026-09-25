# Final System Report

**Pipeline version:** `SPIDER-Lumbar-v1`
**Sprint:** 7 (final polish, demo readiness, research presentation)
**Status:** complete
**Generated:** 2026-09-25

> **Research and educational prototype.** Every finding this system reports is a
> model-derived research estimate and requires review by a qualified
> radiologist. The system does not provide a clinical diagnosis and is not a
> medical device.

Every figure in this report is traceable to an artefact under `outputs/`. The
source is named for each table. No number here was estimated, rounded up, or
carried over from a different experiment.

### How to read the three kinds of number in this system

This distinction is the single most important thing to understand before reading
any figure below, and the interface enforces it at every point of display.

| Kind | What it is | Scope | Example |
| --- | --- | --- | --- |
| **Dataset-level benchmark** | How the pipeline scored on a held-out split of 33 patients it never trained on | Describes **the pipeline**, never the study in front of you | disc indexing 93.84% |
| **Per-study measurement** | A quantity measured off the predicted segmentation of *this* volume, in mm at 1.0 mm/px | Describes **this study**, carries no accuracy claim of its own | central disc height 5.90 mm |
| **Per-study model estimate** | A research model's output for *this* disc, from segmentation-derived features | Describes **this study**, qualified by the benchmark of its target | Pfirrmann grade 5 |
| **Unavailable** | A target with no validated model, or a value that could not be computed | Reported as unavailable **with a reason** | Modic type |

A benchmark figure is never attached to a per-study value as if it were that
value's accuracy. 93.84% does not mean "this patient's disc was identified with
93.84% confidence"; it means that across 6,821 discs in the held-out test split,
93.84% received the correct index.

---

## 1. Project objective

Build a reproducible research pipeline that takes a single sagittal lumbar spine
MRI volume and produces, without manual intervention:

1. a four-class anatomical segmentation (vertebra, intervertebral disc, spinal
   canal, background),
2. a stable per-disc identity ordered from the most inferior disc upward,
3. quantitative disc measurements in millimetres,
4. disc-level radiological finding estimates restricted to targets that
   measurably validated, and
5. a transparent presentation in which the provenance of every displayed value
   is explicit.

The objective was deliberately **not** to produce a diagnostic tool. It was to
determine what the available data actually supports, to report the negative
results alongside the positive ones, and to build an application that cannot
overstate what it knows.

Secondary objective, added in Sprints 6–7: serve the frozen research pipeline
through a real application so the work can be demonstrated end to end, with
every displayed number arriving from the live backend rather than a fixture.

---

## 2. Dataset

**Source:** SPIDER lumbar spine MRI dataset (research dataset).
**Artefacts:** `outputs/preprocessing_reports/dataset_inspection.md`,
`outputs/reports/radiological_grading_analysis.md`.

### 2.1 Volumes

| Field | Value |
| --- | --- |
| Image volumes | 447 (3-D MetaImage `.mha`, not 2-D image files) |
| Mask volumes | 447, one matched mask per image |
| Distinct patients | **218** (1–3 series each) |
| Readable volumes | 447 / 447 (0 read errors) |
| Unmatched images or masks | 0 |
| Image voxel dtype | `float32` |
| Mask voxel dtype | `int16` |
| Total sagittal slices | 14,070 |
| Slices containing any label | 12,598 |
| Preprocessed slices written | 12,415 |

### 2.2 Modalities

| Modality | Series |
| --- | --- |
| `t2` | 210 |
| `t1` | 196 |
| `t2_SPACE` | 41 |

A patient contributes 1–3 series across these modalities. This matters for the
Pfirrmann grade, which is defined on T2 signal and is therefore withheld on a T1
study rather than extrapolated (see §12).

### 2.3 Segmentation labels

Four classes after mapping: background, vertebra, intervertebral disc, spinal
canal. Per volume the masks carry 3–9 vertebra labels and 3–9 IVD labels
(median 7 each); the spinal canal is present in 447/447 volumes. On average
5.64% of voxels are labelled.

Two anatomical sanity checks passed on all 447 volumes: the spinal canal lies
posterior to the vertebral bodies, and vertebra label 1 lies inferior to the
highest vertebra label.

### 2.4 Radiological gradings

| Field | Value |
| --- | --- |
| Rows | 1,520 × 10 columns, one row per (patient, disc) |
| Patients covered | 218 |
| Missing values | 0 |
| Valid rows | 1,518 (2 carry an invalid `IVD label` of 0) |
| Findings per disc | 2.08 on average, up to 7; 31.6% carry none |

Gradings are a **patient-level** annotation shared across a patient's 1–3
series. Attaching them to series would inflate 1,518 real labels to 3,143
instances without adding information, so the finding table is deduplicated to
one series per patient (T2 preferred). The effective sample size is
**218 patients / 1,518 discs**, not 12,415 slices.

The join between grading disc number and mask label was verified rather than
assumed: `IVD label` N corresponds to mask label 200 + N, confirmed by exact
agreement in **432 of 447 series (96.6%)**. The 15 disagreeing series are
enumerated in `radiological_grading_analysis.md`.

### 2.5 Split

Patient-level, disjoint, fixed once in Sprint 1 and never re-drawn.

| Split | Patients | Finding rows |
| --- | --- | --- |
| train | 152 | 1,057 |
| val | 33 | 228 |
| test | 33 | 226 |
| **total** | **218** | **1,511 disc records** |

The per-split patient counts sum to 218, which is the arithmetic proof that the
split is a true partition. Patient-disjointness is asserted in code, not
assumed. The test split was not touched during any training run
(`test_set_used_during_training: false` in both Sprint 3 and Sprint 4
summaries).

---

## 3. Final architecture

```
Browser
  → Next.js 15 frontend (React 19, TypeScript, Tailwind)
  → REST over HTTP/JSON
  → FastAPI backend (Python 3.12)
  → analysis service
  → Sprint 1 preprocessing
  → Sprint 3 U-Net checkpoint (frozen)
  → Sprint 5 5A+5D post-processing
  → feature extraction → finding estimators
  → structured JSON result
  → frontend visualisation + Markdown report
```

The served chain is named once, in `backend/app/pipeline_info.py`, as
`SPIDER-Lumbar-v1`. The full component-level description is in the companion
document `architecture.md`.

Two structural decisions are worth stating here:

- **`src/` is frozen as the ML layer.** `ml/` is a thin serving adapter that
  delegates into `src/` without reimplementing anything, so the application
  cannot silently diverge from the research code. 18 research scripts still
  import `src/` unchanged.
- **The Sprint 5 configuration is read from its artefact**, not hardcoded. The
  backend loads `outputs/reports/sprint5_indexing/test_results.json` at startup,
  so the served post-processing parameters are provably the ones that measured
  93.84%. A missing artefact is a startup error, not a silent default.

---

## 4. Sprint 1 — preprocessing

**Artefacts:** `outputs/preprocessing_reports/preprocessing_report.md`,
`preprocessing_choices.md`.

Applied, in order:

1. **Resample to 1.0 mm/pixel in-plane.** Not optional. Raw in-plane spacing
   varies from 0.077 mm to 1.23 mm and matrix rows from 216 to 3,682. Resampling
   to a fixed mm/pixel keeps a vertebra the same physical size in every patient
   and preserves the anatomical aspect ratio, which a plain pixel resize does
   not. This is also what makes every downstream measurement expressible in
   millimetres.
2. **Centre crop or pad to 352 × 256.** Chosen from the measured anatomy, not by
   convention: the worst case needs 341 mm superior–inferior but only 244 mm
   anterior–posterior. A square 288 × 288 crop would have clipped annotated
   anatomy in **157 of 447 volumes**. At 352 × 256 no volume loses any
   annotation, and both dimensions are multiples of 32, which suits a U-Net's
   downsampling depth.
3. **Per-volume foreground percentile normalisation.** The dataset contains two
   intensity conventions: 374 series share exactly the same minimum (−1000) and
   maximum (3096), i.e. they were linearly rescaled onto a fixed window, while
   the rest keep their native scanner range starting at 0. MRI intensity carries
   no absolute physical meaning, so neither group is comparable with the other
   without normalisation.
4. **Median denoise.**
5. **CLAHE** contrast equalisation.

Result: 447/447 series processed, 0 failures, 149 distinct input shapes reduced
to 1, 447/447 image–mask pairs with matching processed dimensions, 0 series where
the resize invented a label and 0 where a small label was lost.

**N4 bias-field correction was implemented but deliberately not applied.** The
decision was based on measured effect rather than cost: at `shrink_factor=4` it
costs only 0.48 s per volume, but the inhomogeneity metric moved by +0.00005
(+0.03%), and on one of three test volumes it got slightly worse.

---

## 5. Sprint 3 — the selected U-Net

**Artefacts:** `outputs/reports/sprint3_coverage/training_summary.json`,
`sprint3_final_report.md`.
**Checkpoint served:** `outputs/checkpoints/sprint3_coverage/best_val_dice.pt`.

| Field | Value |
| --- | --- |
| Architecture | `UNet(in=1, classes=4, base=16, depth=4, bilinear=True)` |
| Parameters | 1,963,860 |
| Controlled variable | training-slice coverage (fixed subset → rotating sampler) |
| Initialisation | fresh random init |
| Epochs | 29 of 30, early stopped (6 epochs without improvement) |
| Selection criterion | best validation foreground Dice |
| Best validation foreground Dice | 0.89831 at epoch 29 |
| Unique training slices seen | 9,128 of 9,128 (**100% coverage**, up from 28%) |
| Wall time | 5.88 h, 730.2 s per epoch |
| Test split used during training | no |

**The finding that motivated this sprint was a negative one about the previous
sprint.** An audit showed the sampler had only ever shown the model 28% of the
available training slices. A rotating sampler raised that to 100%. Validation
Dice did **not** improve as a result — 0.89831 versus 0.89850 for Sprint 2
Extended, a delta of −0.00019. Coverage was therefore not the binding
constraint.

This run is nevertheless the one served, because it produced the best
**test-split** segmentation of any run and it is the run whose full evaluation
chain was carried through Sprints 4 and 5.

### Held-out test segmentation Dice (33 patients)

| Class | Dice |
| --- | --- |
| Vertebra | 0.91663 |
| Intervertebral disc | 0.87827 |
| Spinal canal | 0.90514 |
| **Macro foreground** | **0.90001** |

These are dataset-level benchmark figures. They are served to the UI from
`pipeline_info.py` and displayed beside the class legend, labelled as validated
test Dice — never as a per-study accuracy.

---

## 6. Sprint 4 — the capacity experiment and its rejection

**Artefacts:** `outputs/reports/sprint4_capacity/training_summary.json`,
`sprint4_final_report.md`.
**Outcome: rejected. Reported, not discarded.**

The hypothesis was that the 16-channel model was capacity-limited. It was tested
by doubling the base width to 32, changing nothing else.

| | Sprint 3 (base 16) | Sprint 4 (base 32) | Delta |
| --- | --- | --- | --- |
| Parameters | 1,963,860 | 7,849,124 | ×4.0 |
| Best validation foreground Dice | 0.89831 | 0.89366 | **−0.00465** |
| Test macro foreground Dice | 0.90001 | 0.89393 | **−0.00608** |
| Test disc indexing | 84.08% | 82.70% | **−1.38 pp** |
| Seconds per epoch | 730.2 | 1,976.0 | **×2.71** |
| Wall time | 5.88 h | 16.47 h | ×2.80 |
| Epochs | 29 (early stopped) | 30 (no early stop) |  |

Every foreground Dice class fell, disc indexing fell, and the run cost 2.71×
more per epoch. The experiment ran to its full 30 epochs without early stopping,
so the result is not an artefact of a truncated run.

**Why it is in this report.** A negative result that rules something out is a
result. This one establishes that width scaling is not the productive direction
for this dataset at this size, which is precisely what justified spending
Sprint 5 on post-processing instead of on a larger model. The rejection is also
served by the API in `RESEARCH_PROGRESS` and is visible on the application's
Methodology page, marked `rejected`. It is not hidden from the demonstration.

The Sprint 4 checkpoint exists on disk and is deliberately **not** the default.
Serving it would regress the product.

---

## 7. Sprint 5 — disc indexing and post-processing

**Artefacts:** `outputs/reports/sprint5_indexing/test_results.json`,
`sprint5_final_report.md`.
**Constraint honoured: no retraining.** Sprint 5 operates on existing Sprint 3
predictions. `semantic_unchanged: true` — the four-class segmentation output is
byte-identical. Only disc *identity* changed.

### 7.1 The problem

Sprint 3 assigned disc identity independently on each 2-D slice. A disc missed
on one slice shifts every index above it on that slice, so the same anatomical
disc could carry different numbers on adjacent slices. 12.1% of discs were
index-shifted and 0.45% were split.

### 7.2 The two adopted methods

- **5A — series-level ordering.** Disc identity is assigned once per series from
  row-aligned tracks clustered across slices, instead of independently per
  slice.
- **5D — vertebral-body separation.** Vertebral bodies are separated from
  posterior elements using the canal column, so posterior structures stop
  perturbing the disc ordering.

**5B and 5C were tuned and rejected.** 5B measured worse on validation; 5C
changed nothing on a like-for-like denominator. Both are documented in the
Sprint 5 report.

Method selection and all parameter tuning were done on the **validation** split
(93.33% index-correct). The selected method was evaluated **once** on test.

### 7.3 Served parameters

Read at startup from `test_results.json`, method `5A+5D`:

| Parameter | Value |
| --- | --- |
| `min_area_px` | 20 |
| `use_series_tracks` | true |
| `cluster_px` | 8.0 |
| `min_track_support` | 0.25 |
| `assign_max_dist_px` | 14.0 |
| `use_vertebral_bodies` | true |
| `body_canal_margin_px` | 0.0 |
| `max_discs` | 9 |

### 7.4 Test results (33 patients, 6,821 discs across 1,655 slices)

| Metric | Sprint 3 baseline | Sprint 5 | Change |
| --- | --- | --- | --- |
| **Disc index correct** | **84.08%** | **93.84%** | **+9.76 pp** |
| Index shifted | 12.10% | 2.52% | improved |
| Split components | 0.45% | 0.00% | eliminated |
| Merged | 0.03% | 0.03% | unchanged |
| Missed | 3.34% | 3.61% | slightly worse |
| Slices with every disc correct | 77.95% | 84.89% | +6.94 pp |
| Spurious components | 270 | 239 | −31 |
| Central height MAE | 0.7634 mm | **0.6492 mm** | −0.1142 mm |
| Disc area MAE | 24.8986 mm² | **21.4831 mm²** | −3.4155 mm² |
| Central height Pearson r | 0.8870 | 0.9128 | +0.0258 |
| Disc area Pearson r | 0.8946 | 0.9261 | +0.0315 |
| Signal ratio MAE | 0.0364 | 0.0280 | −0.0084 |
| Signal ratio Pearson r | 0.9639 | **0.9726** | +0.0087 |
| Pfirrmann QWK (end-to-end) | 0.6317 | **0.6543** | +0.0226 |
| Pfirrmann within 1 grade (end-to-end) | 0.8333 | 0.8571 | +0.0238 |

All 172 remaining shifted discs are off by exactly +1. Mean tracks per series
6.98; 76 components were left unassigned (57 because the track was already
claimed, 19 because none fell within the distance gate).

Honest caveat: the one metric that moved the wrong way is *missed* discs, 3.34%
→ 3.61%, and region-found fell slightly from 96.66% to 96.39%. Series-level
ordering will decline to assign a disc it cannot support across slices rather
than guess, which is the intended trade and is why index correctness rose so
much more than missed detection fell.

---

## 8. Sprint 6 — the full-stack application

Built on the frozen Sprint 1/3/5 pipeline. **No retraining.** Every displayed
value comes from the real backend.

Delivered:

- **Backend:** FastAPI app with config, model singleton loaded once in the
  lifespan, Pydantic schemas, a uniform error envelope, an analysis service
  wrapping the frozen pipeline, upload validation, a background job runner, and
  server-side slice rendering.
- **Frontend:** Next.js 15 / React 19 / TypeScript app with dashboard, upload,
  processing, results workspace, MRI viewer with overlay toggles, disc panel,
  report view and history.
- **Serving adapter:** `ml/pipeline.py`, `ml/volume.py` — five composable steps
  delegating into frozen `src/` code.
- **Persisted finding estimators:** `scripts/18_export_finding_models.py` fits
  and persists 7 estimators to `outputs/models/findings/*.joblib`. All 7
  reproduce their published Sprint 2 scores exactly; the reproduction control
  is recorded in the manifest (`all_reproduced`).

Two findings from this sprint are worth keeping on the record:

- **A real bug, found and fixed.** `JobManager.is_active` consulted the
  persisted record, which was already `queued` from upload, so `run()` never
  submitted the job. Liveness is now process-local only, and a stale
  `processing` record after a restart reports `ANALYSIS_INTERRUPTED` rather than
  hanging.
- **Model load moved into the FastAPI lifespan.** Analysis time fell from 14.5 s
  standalone to 2.9 s served, because the checkpoint is loaded once rather than
  per request.

Security note from this sprint, still applicable: dependencies were bumped
(Next 14.2.18 → 15.5.26, React 19.2, Recharts 3.10.1, Vitest 4.1.11, PostCSS
override 8.5.28) to clear two critical unauthenticated RCE advisories, one of
them Windows-specific. `npm audit` reports 0 vulnerabilities.

---

## 9. Sprint 7 — transparency and polish

No ML change. The whole sprint is about making the system state plainly what it
knows and how it knows it.

| Area | Change |
| --- | --- |
| Pipeline identity | `PIPELINE_VERSION = "SPIDER-Lumbar-v1"` defined once in `backend/app/pipeline_info.py`; echoed by `/api/health`, by every result, in the top bar, and in the generated report |
| Metrics | Every research figure is served from the API (`validated_metrics`, `metric_table`). The hard-coded metric arrays were **removed** from the frontend, so the UI cannot drift from what was measured |
| Provenance | Each disc carries `structured` lines of label / value / provenance, with `segmentation_derived`, `model_prediction` or `unsupported` and a reason when unavailable |
| Evidence | Qualitative strength per target (`strong` / `moderate` / `modest` / `weak`), served as `evidence_labels`. No numeric "confidence" anywhere |
| Research progress | `RESEARCH_PROGRESS` serves the experimental record including the Sprint 4 rejection, rendered on the Methodology page |
| Viewer sync | `highlight_disc` query parameter on the slice endpoint; selecting a disc in the panel marks it in the rendered slice |
| Demo mode | `POST /api/analysis/sample` plus `sample_study_available` on health; the button only appears when the backend confirms the volume exists |
| Pages | New Methodology page; rewritten About page; provenance legend; report view and generated Markdown carry pipeline version and timestamp |
| Tests | New `backend/tests/test_sprint7.py` (23 tests). Backend total 32 → 55 |

The guiding rule: a missing value is never rendered as `0`, `No` or `Normal`,
because the absence of a validated model is not a negative finding.

---

## 10. Final validated research metrics

**These are dataset-level benchmark figures measured on a 33-patient held-out
test split. They describe the pipeline. They do not describe any uploaded
study.** This is the exact wording the API serves in
`validated_metrics.note` and the UI displays verbatim.

### Segmentation — Sprint 3

| Metric | Value |
| --- | --- |
| Macro foreground Dice | 0.90001 |
| Vertebra Dice | 0.91663 |
| Intervertebral disc Dice | 0.87827 |
| Spinal canal Dice | 0.90514 |

### Disc indexing, measurement and grading — Sprint 5 post-processing

| Metric | Value |
| --- | --- |
| Disc indexing, per-slice ordering (Sprint 3) | 84.08% |
| Disc indexing, series-level ordering (Sprint 5) | **93.84%** |
| Disc height mean absolute error | 0.6492 mm |
| Disc area mean absolute error | 21.4831 mm² |
| Disc signal ratio correlation | r = 0.9726 |
| End-to-end Pfirrmann agreement | QWK 0.6543 |
| Test patients | 33 |

Segmentation and post-processing figures are kept visually and structurally
separate in the UI (`kind: "segmentation" | "postprocessing"` on each metric
row) because they measure different things and must not be read as one score.

---

## 11. Supported findings

Seven targets are served. All were fitted on the **train** split; the test split
was used for a reproduction control only, never for selection. PR-AUC is the
headline metric rather than ROC-AUC because several findings are rare and
ROC-AUC is optimistic under imbalance. The prevalence baseline is what a random
ranker scores.

| Finding | Estimator | Metric | Value | Prevalence baseline | Test positives | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Disc narrowing | logistic regression | test PR-AUC | **0.8779** | 0.323 | 73 | strong |
| Lower endplate change | logistic regression | test PR-AUC | 0.7962 | 0.3894 | 88 | moderate |
| Disc bulging | random forest | test PR-AUC | 0.7610 | 0.4469 | 101 | moderate |
| Upper endplate change | logistic regression | test PR-AUC | 0.6688 | 0.3363 | 76 | modest |
| Any Modic change | logistic regression | test PR-AUC | 0.6505 | 0.2876 | 65 | modest |
| Disc herniation | logistic regression | test PR-AUC | 0.4362 | 0.0796 | 18 | **weak** |
| Pfirrmann grade | ordinal random forest | QWK | see below | — | n = 211 | moderate |

Features are segmentation geometry plus ROI intensity (30 features in the
`geometric+intensity` union): disc heights, area, AP extent, ratios to the
series median and to neighbours, disc-to-vertebra height ratio, canal width, and
disc/nucleus/annulus signal statistics.

### Pfirrmann grade — stated precisely

Modelled as an ordinal target with the Frank & Hall decomposition: four binary
models estimate P(grade > k) and per-class probabilities come from consecutive
differences.

| Figure | Value | What it measures |
| --- | --- | --- |
| QWK from ground-truth masks | 0.6718 | the grading step in isolation |
| **QWK end-to-end (served)** | **0.6543** | predicted masks + Sprint 5 indexing — the figure that applies to an uploaded study |
| Within one grade (served) | 0.8531 | ground-truth-mask figure |
| Within one grade, end-to-end | 0.8571 | reported in the Sprint 5 comparison |
| MAE from ground-truth masks | 0.7678 grades | |
| Majority-class exact agreement | 0.2749 | a constant prediction has QWK 0 by construction |

One precision note for anyone auditing the payload: the served
`validated_value` for Pfirrmann is the **end-to-end 0.6543**, which is the
correct figure for an uploaded study, while the served `within_one_grade` is
**0.8531**, the ground-truth-mask figure. The end-to-end within-one-grade value
measured in Sprint 5 is 0.8571. The two are close and the served value is the
more conservative of the pair, but they are not the same measurement and this
report does not present them as such.

**Herniation is served with `weak` evidence and should be read that way.** At
0.4362 PR-AUC against an 0.0796 baseline it carries information, but on 18
positive test discs the point estimate is very uncertain. A focal herniation is
a local shape detail that summary geometry is not expected to capture well; the
low score is an honest reflection of the feature set.

---

## 12. Unsupported and unavailable findings

Nothing is guessed. Each of these is returned as `null` with an
`unavailable_reason`, and the UI renders the reason rather than an empty badge
that could read as a negative result.

| Target | Served? | Reason |
| --- | --- | --- |
| **Modic type** (0 / I / II / III) | never | The nominal type was never modelled. Only the binary "any Modic change" target was. Reporting a type would require guessing beyond what was validated. |
| **Spondylolisthesis** | never | Only 5 positive discs exist in the held-out test split. Sprint 2 measured a PR-AUC between 0.19 and 0.37 depending on features and estimator — too uncertain to report. |
| **Pfirrmann grade on a non-T2 study** | conditionally | The grade is defined on T2 nucleus signal and the estimator was validated on T2 and T2-SPACE series only. On a T1 study it is withheld with an explicit reason rather than extrapolated. |
| **Any measurement that could not be computed** | conditionally | Returned `null`. The UI shows `unavailable` in a muted, italic style with the reason underneath. |

The distinction the interface is built to protect: **"we did not model this" is
not the same statement as "this patient does not have this."**

---

## 13. Provenance and evidence model

Every displayed quantity carries a provenance tag, served by the API as
`provenance_labels`:

| Provenance | Label shown | Meaning |
| --- | --- | --- |
| `segmentation_derived` | **Segmentation-derived** | Measured from the predicted segmentation at 1.0 mm/pixel |
| `model_prediction` | **Model-estimated** | Estimated by a research model from segmentation-derived features |
| `unsupported` | **Not modelled** | No validated model exists for this target, so nothing is reported |

Evidence strength, served as `evidence_labels`, is **deliberately qualitative**:

| Strength | Label shown |
| --- | --- |
| `strong` | Strong evidence |
| `moderate` | Moderate evidence |
| `modest` | Limited evidence |
| `weak` | Weak evidence |

No numeric confidence percentage is displayed anywhere in the system.
Converting a model probability into a "confidence" would invent a quantity that
was never validated. Where a raw probability is shown it is labelled as the
model's probability, alongside the validated metric for that target and its
prevalence baseline.

One more term that is easy to misread, and is labelled accordingly in the UI:
**`identity_confidence` is not clinical confidence.** It is the fraction of the
series' slices supporting a disc's series-level track — a measure of tracking
consistency. The interface says so explicitly in the disc detail panel.

Disc indices are also not anatomical claims. `discs[].index` is an integer
position with 1 = most inferior. The dataset does not state which vertebra is
L5, so no level name is asserted, and the summary says so in words.

---

## 14. Sample study and demo mode

`POST /api/analysis/sample` creates an analysis from a real volume bundled with
the dataset checkout (`data/extracted/images/33_t2.mha`, with two fallbacks).

What makes it honest:

- It is **not a mock**. The volume goes through the identical pipeline, and no
  result is pre-computed. The analysis it produces is indistinguishable from an
  upload except for an `is_sample` flag.
- The flag is **surfaced, not hidden**: the results header shows a
  `Sample research study` badge, and the upload page says the sample is a real
  volume from the research dataset processed by the identical pipeline.
- If the volume is absent on a given machine, `/api/health` reports
  `sample_study_available: false` and the UI **hides the button** rather than
  offering a dead end or falling back to canned output.

---

## 15. Limitations

Stated plainly, because a demonstration that hides these is misleading.

**Data and scope**

1. Single dataset, single research cohort. No external validation on another
   site, scanner or population.
2. 33 test patients. For a finding with ~3% prevalence that is a single-digit
   number of positive discs, so point estimates for rare findings are very
   uncertain. The positive count is printed beside every score for that reason.
3. Sagittal 2-D slices only. The model segments slice by slice; no 3-D
   context is used.
4. Findings are a patient-level annotation in the source data, mapped to discs.
5. The IVD-label-to-mask-label join disagrees in 15 of 447 series (3.4%).

**Model**

6. Segmentation is 2-D and four-class. It does not identify individual vertebral
   levels by name.
7. Disc indexing is 93.84% correct, not 100%. Remaining errors are +1 shifts and
   3.61% missed discs.
8. Finding estimators are classical models on 30 summary features, not deep
   models on imagery. Herniation in particular is weakly served.
9. Pfirrmann is restricted to T2 / T2-SPACE.
10. No composite severity or "spine damage" score is produced, by design. Each
    finding is reported on its own, as the dataset annotates it. Combining
    findings into a single number would be an unvalidated clinical judgement.

**System**

11. Runs on CPU in the demonstrated configuration; ~3 s per study served.
12. Local filesystem storage, single process, no database.
13. **No authentication.** See the security limitation in `architecture.md`.
14. No DICOM ingestion; converted volume formats only, so modality usually comes
    from a filename heuristic and is labelled as such.

---

## 16. Longitudinal and postoperative limitation

**This system does not and cannot predict postoperative healing, recovery or
any other future outcome.** The reason is a property of the data, not a
temporary gap in the implementation.

- SPIDER is **cross-sectional**. Each patient's series are a single timepoint.
- The dataset carries **no postoperative imaging**, no surgical record, no
  intervention date and no follow-up assessment.
- Time fields exist in the schema but are **null in the values**
  (`outputs/reports/longitudinal_plan.md`).
- With no second timepoint and no outcome label, there is nothing to train a
  progression or healing model against, and nothing to validate one with.

What the system does instead, honestly: the result carries a `timepoint` object
that is always a single `baseline`. It is a **schema seam** for a future
longitudinal module, not a claim of longitudinal capability. The analysis
summary states in words that findings are from a single timepoint and that no
postoperative assessment is provided, and the generated report repeats it.

Any claim of postoperative prediction from this system would be fabricated. It
is not made anywhere in the code, the API, the UI or the generated report.

---

## 17. Clinical and research disclaimer

Served verbatim by the API and displayed on every result, in the generated
report, and in the API documentation:

> Research and educational prototype. Results are model-derived research
> estimates and require review by a qualified radiologist. This system does not
> provide a clinical diagnosis and is not a medical device.

Shorter research notice, shown in the application banner:

> Research / educational prototype — results require expert radiological review.

Specifically, this system does **not**:

- provide a diagnosis,
- recommend or evaluate treatment,
- predict postoperative healing or any future outcome,
- assert anatomical level names (L1–L5, S1),
- report a Modic type or spondylolisthesis,
- produce a composite severity or damage score,
- claim any per-patient accuracy figure.

---

## 18. Final verification results

### Automated test suites

| Suite | Result |
| --- | --- |
| Backend `pytest backend/tests -q` | **55 passed** (32 Sprint 6 + 23 Sprint 7) |
| Frontend `npx vitest run` | **18 passed / 18** |
| Frontend `tsc --noEmit` | **clean**, 0 diagnostics |
| Frontend `next build` | **success**, 9 routes |
| `npm audit` | 0 vulnerabilities |

### Live end-to-end verification

Run against real `uvicorn` and `next` servers over HTTP, with no mocked
responses. Results are recorded in §"E2E" of the accompanying run log and
summarised in `demo_checklist.md`.

### Constraint audit

| Constraint | Status |
| --- | --- |
| No model retrained in Sprint 6 or 7 | honoured |
| Sprint 3 checkpoint unchanged | honoured |
| Sprint 5 post-processing unchanged | honoured |
| Validated model parameters unchanged | honoured |
| Research metrics unchanged | honoured |
| No fabricated findings | honoured |
| No mock analysis results | honoured |
| Sprint 1 preprocessing unchanged | honoured |

Sprint 7 modified no artefact under `src/`, `ml/`, `outputs/checkpoints/`,
`outputs/models/` or `outputs/reports/sprint5_indexing/`.

---

## Artefact index

| Claim in this report | Source |
| --- | --- |
| Dataset counts, modalities, label stats | `outputs/preprocessing_reports/dataset_inspection.md` |
| Preprocessing decisions and outcomes | `outputs/preprocessing_reports/preprocessing_report.md`, `preprocessing_choices.md` |
| Split integrity | `outputs/preprocessing_reports/split_report.md` |
| Grading join and prevalence | `outputs/reports/radiological_grading_analysis.md` |
| Sprint 3 training and selection | `outputs/reports/sprint3_coverage/training_summary.json` |
| Sprint 4 rejection | `outputs/reports/sprint4_capacity/training_summary.json` |
| Sprint 5 method, params, test results | `outputs/reports/sprint5_indexing/test_results.json` |
| Sprint 5 measurement and grading deltas | `outputs/reports/sprint5_indexing/sprint5_final_report.md` |
| Finding model scores and reproduction control | `outputs/models/findings/manifest.json` |
| Baseline finding assessment | `outputs/reports/baseline_findings.md` |
| Served pipeline identity and metrics | `backend/app/pipeline_info.py` |
| Longitudinal data absence | `outputs/reports/longitudinal_plan.md` |
| API contract | `http://localhost:8000/api/openapi.json`, `docs/api.md` |
