# Automated Analysis of Lumbar Spine MRI: Segmentation and Disc-Level Degeneration Assessment

## Problem

Manual segmentation and grading of lumbar spine MRI is time-consuming and can
vary between observers. That variability limits both clinical throughput and the
reproducibility of any measurement derived from the images.

## Objective

Develop an automated pipeline that:

1. analyses sagittal lumbar spine MRI,
2. identifies vertebrae and intervertebral disc regions,
3. estimates per-disc radiological degeneration findings,
4. produces a structured per-disc report of those findings,
5. is built so that longitudinal comparison of repeat scans becomes possible
   **once longitudinal data exists** (see the limitation below).

> **Verified dataset limitation.** The current dataset is **cross-sectional —
> exactly one study per patient.** It contains no acquisition-date or timepoint
> field, no patient has more than one series of the same sequence, and there is
> no surgery or outcome record. Preoperative vs 6-month vs 9-month comparison
> therefore **cannot be implemented or validated** with it. Evidence and the
> additional data that would be required: section 8 of
> [`outputs/reports/radiological_grading_analysis.md`](outputs/reports/radiological_grading_analysis.md).

---

# The Application

Sprint 6 turns the validated research pipeline into a local full-stack web
application. This section is self-contained: overview, architecture, setup,
API, workflow, limitations and disclaimer. The research documentation for
Sprints 1–5 continues below it.

> **Research / educational prototype — results require expert radiological
> review.** Every finding the application shows is a model-derived research
> estimate. It does not provide a clinical diagnosis, it does not replace a
> radiologist, and it is not a medical device.

## 1. Project overview

A researcher uploads a sagittal lumbar spine MRI volume and the application runs
the frozen, validated pipeline end to end: preprocessing, segmentation, disc
indexing, per-disc measurement and model-derived findings. It then presents a
slice viewer with a segmentation overlay, disc-level results, quantitative
measurements and a downloadable report.

Nothing in the interface is mocked. Every number displayed comes from the backend
running the real pipeline on the uploaded volume. Where a value cannot be
computed, or a finding was never validated, the field is `null` with a stated
reason instead of a guess.

## 2. Architecture

```
Browser
  │
  ▼
Next.js 15 (frontend/)          TypeScript · Tailwind · Recharts
  │  REST over HTTP
  ▼
FastAPI (backend/)              Pydantic schemas · background jobs
  │
  ▼
Analysis service                backend/app/services/mri_analysis_service.py
  │
  ▼
ML adapter (ml/)                stable serving seam over src/
  │
  ├─ Preprocessing              Sprint 1
  ├─ Segmentation               Sprint 3 U-Net
  ├─ Disc indexing              Sprint 5 post-processing
  ├─ Measurements               disc-level features
  └─ Findings                   persisted estimators
  │
  ▼
Result JSON  ──▶  Frontend visualisation
```

```
lumbar-spine-segmentation/
├── frontend/          Next.js app. Never imports Python.
├── backend/           FastAPI. The only place inference happens.
│   ├── app/
│   │   ├── main.py            app factory + lifespan (loads the model once)
│   │   ├── config.py          env-driven settings, all paths via pathlib
│   │   ├── schemas.py         explicit Pydantic request/response models
│   │   ├── errors.py          uniform error envelope
│   │   ├── routers/           health, analysis
│   │   └── services/          model_service, mri_analysis_service, jobs,
│   │                          storage, imaging, report_writer
│   └── tests/                 32 backend tests
├── ml/                Serving adapter over the research code
│   ├── pipeline.py            the five pipeline steps
│   └── volume.py              upload inspection and validation
├── src/               Frozen Sprint 1–5 research code (unchanged)
├── data/              Raw, extracted and preprocessed data
├── outputs/           Checkpoints, metrics, reports, and app_data/
└── docs/api.md        API reference with request/response examples
```

Two deliberate structural notes:

- **`src/` is the ML layer.** The brief sketched a top-level `ml/` directory.
  Renaming `src/` would break eighteen research scripts and every reproducibility
  claim in this README, so `src/` stays frozen and `ml/` was added as a genuine
  adapter: a small set of stable entry points the backend depends on, so the web
  layer never reaches into six research submodules. The separation the brief asked
  for is enforced; the history is intact.
- **The frontend never imports Python.** It talks only to the REST API, and the
  backend is the single source of truth for inference.

## 3. ML pipeline

| Stage | What happens | Source |
| --- | --- | --- |
| Preprocessing | Resample to 1.0 mm/px, centre crop or pad to 352×256, normalise on per-volume foreground percentiles, median denoise, CLAHE | Sprint 1 |
| Segmentation | 16-channel U-Net, depth 4, bilinear, 1,963,860 parameters, 4 classes | Sprint 3 |
| Disc indexing | Series-level track ordering (5A) plus vertebral-body separation (5D) | Sprint 5 |
| Measurements | Per-disc height, area, AP extent, signal ratios, canal width | Sprint 2/5 |
| Findings | Ordinal Pfirrmann model plus six binary finding estimators | Sprint 2, persisted in Sprint 6 |

Every stage parameter is either a code constant or computed from the uploaded
volume, so an unseen study is processed identically to a dataset study. The
Sprint 5 configuration is **read from the published artefact** at startup rather
than hardcoded, so the served pipeline cannot drift from the one that was
measured.

**Sprint 4 (width 32) is deliberately not served.** It measured worse than
Sprint 3 on every Dice class and on disc indexing, at 2.8× the training cost.

## 4. Sprint results

Measured on the 33-patient held-out test split. **Segmentation metrics and
post-processing metrics answer different questions and are reported separately.**

Segmentation quality — Sprint 3:

| metric | value |
| --- | --- |
| macro foreground Dice | **0.90001** |
| vertebra Dice | 0.91663 |
| intervertebral disc Dice | 0.87827 |
| spinal canal Dice | 0.90514 |

Disc identity and measurement — Sprint 5 post-processing on top of that
segmentation:

| metric | value |
| --- | --- |
| disc indexing accuracy | **93.84%** (from 84.08% with per-slice ordering) |
| disc height MAE | **0.6492 mm** |
| disc area MAE | **21.4831 mm²** |
| disc signal ratio correlation | **r = 0.9726** |
| end-to-end Pfirrmann agreement | **QWK 0.6543** |

Finding estimators — Sprint 2, test-split PR-AUC against prevalence baseline:

| target | PR-AUC | baseline | served |
| --- | --- | --- | --- |
| disc narrowing | 0.8779 | 0.323 | yes (strong) |
| lower endplate | 0.7962 | 0.389 | yes (moderate) |
| disc bulging | 0.7610 | 0.447 | yes (moderate) |
| upper endplate | 0.6688 | 0.336 | yes (modest) |
| any Modic change | 0.6505 | 0.288 | yes (modest) |
| disc herniation | 0.4362 | 0.080 | yes, flagged weak (18 positives) |
| spondylolisthesis | 0.19–0.37 | 0.022 | **no** — 5 positives, not validated |
| Modic *type* (0/I/II/III) | — | — | **no** — never modelled |

Sprint progression: 1 preprocessing → 2 baseline segmentation and findings →
2 Extended convergence → 3 data coverage (accepted) → 4 capacity (rejected,
worse) → 5 indexing post-processing (accepted, +9.76 pp) → 6 application.

## 5. Backend setup

```bash
# from the repository root
pip install -r requirements.txt
pip install -r backend/requirements.txt

# one-off: persist the finding estimators the API serves
python scripts/18_export_finding_models.py

cp backend/.env.example backend/.env    # optional; defaults work as-is

# Terminal 1
uvicorn backend.app.main:app --reload --port 8000
```

Confirm it came up correctly:

```bash
curl http://localhost:8000/api/health
```

`status` should be `ok` and `segmentation_model.loaded` should be `true`. If it
reports `degraded`, `MODEL_CHECKPOINT` is not pointing at a checkpoint.

The model is loaded **once** during application startup and reused for every
request. Inference runs on CPU by default and uses CUDA automatically when a
device is available.

## 6. Frontend setup

```bash
cd frontend
npm install
cp .env.example .env.local              # NEXT_PUBLIC_API_URL=http://localhost:8000

# Terminal 2
npm run dev
```

Open <http://localhost:3000>.

| command | purpose |
| --- | --- |
| `npm run dev` | development server on port 3000 |
| `npm run build` | production build |
| `npm run typecheck` | TypeScript, no emit |
| `npm test` | Vitest component and flow tests |

## 7. API documentation

Full reference with request and response examples, error codes and field
semantics: **[`docs/api.md`](docs/api.md)**.

FastAPI generates the authoritative schema:

- Swagger UI <http://localhost:8000/api/docs>
- ReDoc <http://localhost:8000/api/redoc>
- OpenAPI JSON <http://localhost:8000/api/openapi.json>

| method | path | purpose |
| --- | --- | --- |
| GET | `/api/health` | what is loaded, and which findings are served |
| POST | `/api/analysis/upload` | upload and validate a volume |
| POST | `/api/analysis/{id}/run` | start processing, returns immediately |
| GET | `/api/analysis/{id}/status` | poll real progress 0–100 |
| GET | `/api/analysis/{id}/result` | structured result |
| GET | `/api/analysis/{id}/slice/{n}` | one rendered slice as PNG |
| GET | `/api/analysis/{id}/report` | report payload |
| GET | `/api/analysis/{id}/download` | report as Markdown or JSON |
| GET | `/api/analysis` | analysis history |
| DELETE | `/api/analysis/{id}` | delete an analysis |

Every error, at every status code, returns the same envelope. A Python traceback
is never exposed:

```json
{ "error": { "code": "UNREADABLE_VOLUME", "message": "...", "details": {} } }
```

## 8. Model location

| artefact | default path | override |
| --- | --- | --- |
| Segmentation checkpoint | `outputs/checkpoints/sprint3_coverage/best_val_dice.pt` | `MODEL_CHECKPOINT` |
| Finding estimators | `outputs/models/findings/*.joblib` | `FINDING_MODEL_DIR` |
| Sprint 5 configuration | `outputs/reports/sprint5_indexing/test_results.json` | `SPRINT5_PARAMS` |
| Analysis storage | `outputs/app_data/` | `STORAGE_DIR` |

All paths are resolved with `pathlib` relative to the repository root. No
absolute machine-specific path is committed. Relative overrides are resolved
against the repository root; absolute overrides are used as given.

## 9. Example workflow

Through the interface: **New Analysis** → drop a `.mha` volume → the server
validates it and reports its geometry → **Start analysis** → watch the real stage
progression → the results workspace opens with the viewer, disc findings and
measurements → click a disc for its detail panel → **Download** for the report.

The same thing over HTTP:

```bash
BASE=http://localhost:8000

ID=$(curl -s -F "file=@data/extracted/images/33_t2.mha" \
  $BASE/api/analysis/upload | python -c "import json,sys;print(json.load(sys.stdin)['analysis_id'])")

curl -s -X POST $BASE/api/analysis/$ID/run
curl -s $BASE/api/analysis/$ID/status
curl -s $BASE/api/analysis/$ID/result | python -m json.tool
curl -s -o slice.png "$BASE/api/analysis/$ID/slice/12?mode=overlay"
curl -s -o report.md "$BASE/api/analysis/$ID/download?fmt=md"
```

### Tests and the end-to-end smoke test

```bash
python -m pytest backend/tests -q      # 32 backend tests
cd frontend && npm test                # 18 frontend tests

# with both servers running, prove the whole chain over real HTTP
python scripts/19_smoke_e2e.py
```

The smoke test walks frontend → API → ML pipeline → result → slice → frontend and
asserts, among other things, that progress is monotonic and never synthetic, that
unsupported findings are null with a reason, and that no traceback reaches the
client.

## 9a. 3D viewing and the longitudinal workflow demonstration

### Real 3D volume rendering

`GET /api/analysis/{id}/volume` serves a completed analysis's image volume and
both label maps in one cacheable response, and the results workspace renders it
with **VTK.js over WebGL** behind a `2D View | 3D View` switch. Rotate, zoom, pan,
reset camera, fullscreen, MRI visibility and opacity, segmentation and per-class
visibility, findings visibility and selected-disc highlighting are all supported.

This **intentionally reverses** the earlier property that the volume never crossed
the network — volumetric rendering needs the voxels. The 2D slice path is
unchanged and still filters classes server-side. The trade, and what it costs, is
recorded in `outputs/reports/sprint7_final/architecture.md` §11a. Measured on the
sample study: 6,337 KB of voxels leaves as **962 KB** gzipped, once per analysis.

Both viewers read one selected-disc state and one server-side finding decision
(carried in the volume header), so selecting a disc in either view updates the
other and they cannot disagree.

### Recovery Tracker — a simulated workflow, not follow-up data

**The SPIDER dataset provides cross-sectional studies rather than true
longitudinal postoperative follow-up. The Recovery Tracker therefore uses a
simulated demonstration workflow and does not claim postoperative recovery
outcomes from SPIDER.**

`/recovery` demonstrates how a longitudinal workflow would operate, using **four
different real SPIDER studies from four different patients**, all from the
held-out test split:

| Timeline position | Source study | Discs | Mean disc height | Finding-associated |
| --- | --- | --- | --- | --- |
| Diagnosis | `177_t2` | 9 | 4.64 mm | 8 |
| Surgical Evaluation | *(assessment of the baseline imaging)* | — | — | — |
| Post-Surgery | `106_t2` | 7 | 5.80 mm | 7 |
| 3-Month Recovery | `16_t2` | 7 | 7.79 mm | 2 |
| 6-Month Recovery | `6_t2` | 6 | 8.49 mm | 1 |

Every number above is real pipeline output. **The timeline is what is simulated.**
Any trend across the stages follows from which studies were selected; it is not an
observed recovery trajectory, and the comparison panel is captioned *"Cross-study
demonstration trend — not patient recovery."* No recovery percentage or composite
score is produced anywhere.

The interface states this permanently, not only here:

> These stages use different SPIDER studies to demonstrate the longitudinal
> workflow. They are not postoperative follow-up scans from the same patient.

Definitions live in `demo/longitudinal_cases/<case>/manifest.json` — text only, no
imaging. Each stage references a gitignored local dataset path; if it is absent the
API reports the stage unavailable and **never substitutes another scan**. The
loader refuses any manifest declaring itself real follow-up, so enabling genuine
longitudinal data is an explicit, reviewable change. See `demo/README.md`.

For an ordinary single study the application shows *Single-study analysis* and
*Longitudinal follow-up unavailable for this study*, and invents no stages.

## 10. Limitations

- **Validated performance is not per-study performance.** Every metric above was
  measured on a 33-patient held-out test split. Accuracy on any particular upload
  is unknown.
- **Disc identity is derived, not predicted.** The network outputs four semantic
  classes; identity comes from ordered geometry. `identity_confidence` reports how
  consistently a disc was tracked across slices — it is not a clinical confidence.
- **Integer disc indices only.** The dataset does not state which vertebra is L5,
  so no L1–L5 or S1 level name is produced anywhere.
- **Pfirrmann applies to T2 and T2-SPACE only**, because the grade is defined on
  T2 signal. On a T1 study the grade is withheld rather than extrapolated.
- **Modic type and spondylolisthesis are not reported.** The former was never
  modelled, the latter had five positive test cases.
- **`disc_to_vertebra_height_ratio` remains unreliable** even after Sprint 5
  improved it from 71.87% to 33.72% error.
- **Single timepoint.** No change over time is measured and no postoperative or
  longitudinal assessment is offered, because the data this system was validated
  on contains no longitudinal follow-up. The result schema carries a single
  `baseline` timepoint so a longitudinal module can be added later without
  altering any existing field.
- **No composite severity score.** There is deliberately no single "percentage
  damage" number, because no such quantity was defined or validated.
- **Modality is a filename heuristic.** A converted `.mha` carries no DICOM series
  description; the API always reports how the modality was determined.
- **Local prototype, not a deployment.** Storage is the local filesystem, there is
  no authentication, and the API is intended to be bound to localhost. Do not
  expose it to a network without adding access control.

## 11. Research disclaimer

This application performs **AI-assisted lumbar spine MRI research analysis**.

All outputs are **model-derived research estimates requiring review by a
qualified radiologist**. The system does not perform clinical diagnosis, does not
replace radiological assessment, and is not a medical device. It must not be used
to make care decisions. Segmentation and indexing metrics measure agreement with
one annotation protocol; they do not establish clinical effectiveness.

No postoperative healing, recovery or longitudinal-improvement capability is
implemented or claimed, because the dataset contains no longitudinal
postoperative follow-up data.

---

# Research Documentation (Sprints 1–5)

## Current sprint

**Sprint 1 — Dataset Inspection and Preprocessing: complete.**
**Sprint 2 — Segmentation and Disc-Level Analysis: complete (baseline).**
**Sprint 2 extended — controlled 30-epoch training experiment: complete.**
**Sprint 3 — training-data coverage experiment: complete (coverage was not the constraint).**
**Sprint 4 — model capacity experiment: complete (width 32 measurably worse; rejected).**
**Sprint 5 — disc indexing post-processing: complete (84.08% → 93.84%).**
**Sprint 6 — full-stack application: complete (see [The Application](#the-application)).**

> **The 16-channel U-Net has converged.** Extending training from 12 to 30
> epochs improved 43 of 53 tracked metrics, but the gain is small and the model
> plateaued: over the final six epochs validation Dice moved 0.00045 in total.
> Further epochs at this width and this training subset are not worthwhile —
> architecture and data-volume changes are now justified. Full comparison:
> [`outputs/reports/sprint2_extended/comparison.md`](outputs/reports/sprint2_extended/comparison.md)

### Sprint 2 extended — headline

Controlled experiment: **only the epoch budget changed** (12 → 30). Architecture,
patient-level split, seed, preprocessing, classes, loss, optimiser, augmentation
and slices-per-epoch were all held fixed. Resumed from the baseline's epoch-12
checkpoint; 18 further epochs in 231 min. The test set was untouched until the
single final evaluation of the selected checkpoint.

| test metric | baseline (12 ep) | extended (30 ep) | Δ |
| --- | --- | --- | --- |
| vertebra Dice | 0.9094 | **0.9133** | +0.0039 |
| IVD Dice | 0.8760 | **0.8779** | +0.0019 |
| spinal canal Dice | 0.9004 | **0.9026** | +0.0022 |
| macro foreground Dice | 0.8953 | **0.8979** | +0.0027 |
| macro foreground IoU | 0.8107 | **0.8151** | +0.0044 |
| macro foreground recall | 0.8934 | **0.9022** | +0.0087 |
| macro foreground precision | 0.8972 | 0.8938 | −0.0033 |
| **disc index accuracy** | 78.4% | **82.6%** | **+4.2 pp** |
| disc height MAE (predicted masks) | 0.848 mm | **0.807 mm** | −0.041 |
| disc area MAE (predicted masks) | 27.03 mm² | **25.34 mm²** | −1.69 |
| end-to-end Pfirrmann QWK (forest) | 0.6294 | **0.6375** | +0.0081 |

The biggest gain is **disc indexing (+4.2 pp)**, which is the metric that
actually gates the disc-level stages — a numbering shift attaches a grading to
the wrong disc. Recall improved on every class while precision fell slightly:
the extended model segments marginally more liberally.

> **No composite severity score is defined** anywhere in this project. The
> dataset grades eight findings independently; collapsing them into one number
> is a clinical judgement this data does not license. The reports are titled
> "Radiological Finding Assessment" and are explicitly not diagnoses.

### Sprint 2 headline results

Trained on CPU only (verified: no CUDA device on this machine), so these are
**baseline results under a compute budget, not final performance**. All figures
are on the 33 held-out **test patients** (1,655 slices) from the Sprint 1
patient-level split.

Segmentation, aggregate (dataset-level) Dice / IoU / precision / recall:

| class | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- |
| background | 0.9950 | 0.9900 | 0.9947 | 0.9953 |
| vertebra | 0.9094 | 0.8338 | 0.9172 | 0.9017 |
| intervertebral disc | 0.8760 | 0.7794 | 0.8782 | 0.8739 |
| spinal canal | 0.9004 | 0.8188 | 0.8960 | 0.9047 |
| **macro (foreground)** | **0.8953** | **0.8107** | 0.8972 | 0.8934 |

Per-patient Dice (mean ± sd over 33 patients): vertebra 0.910 ± 0.018,
disc 0.884 ± 0.029, canal 0.900 ± 0.024. Pixel accuracy 0.9904.

Baseline disc-level finding assessment (test set, best configuration per target):

| target | metric | score | baseline | test support |
| --- | --- | --- | --- | --- |
| Pfirrmann grade (ordinal) | quadratic weighted kappa | **0.672** | 0 (constant) | 211 discs |
| | MAE / within-1-grade | 0.768 / 85.3% | | |
| disc narrowing | PR-AUC | **0.878** | 0.323 | 73 pos / 226 |
| lower endplate defect | PR-AUC | 0.796 | 0.389 | 88 pos / 226 |
| disc bulging | PR-AUC | 0.761 | 0.447 | 101 pos / 226 |
| upper endplate defect | PR-AUC | 0.669 | 0.336 | 76 pos / 226 |
| any Modic change | PR-AUC | 0.651 | 0.288 | 65 pos / 226 |
| disc herniation | PR-AUC | 0.436 | 0.080 | 18 pos / 226 |
| spondylolisthesis | PR-AUC | 0.367 | 0.022 | **5 pos** / 226 |

`baseline` is what a random ranker scores (the prevalence). Every score is
printed next to its positive count because the rare findings rest on very few
test cases.

**The feature families behave as the anatomy predicts**, which is a useful
validity check: geometry beats intensity for herniation (PR-AUC 0.436 vs 0.221),
while intensity beats geometry for the Pfirrmann grade (kappa 0.638 vs 0.520) —
the grade is defined on T2 nucleus signal.

### Completed — Sprint 1

- Dataset organisation
- Dataset extraction
- Image–mask pairing
- Dataset validation
- Image preprocessing
- Mask preprocessing
- Visualisation
- Train/validation/test split (patient-level)

### Completed — Sprint 2 scoping

- Full analysis of `radiological_gradings.csv` (schema, value domains, class
  balance, quality problems)
- Verified linkage between `Patient`, `IVD label`, the MRI series and the
  segmentation mask label space
- Distribution figures for every radiological finding
- Longitudinal capability audit
- Proposed Sprint 2 architecture

### Completed — Sprint 2 implementation

- **Stage A** — baseline 2-D multi-class U-Net (PyTorch), with a 14-check smoke
  test run before training
- **Stage B** — Dice / IoU / precision / recall, overall and per class, in three
  aggregations (dataset-level, per-slice, per-patient); 12 prediction figures
- **Stage C** — deterministic disc-instance extraction with millimetre
  measurements; 3,147 disc records
- **Stage D** — gradings linked on `patient_id + ivd_label`
- **Stage E** — ordinal Pfirrmann model and 7 binary finding baselines, on
  patient-level splits
- **Stage F** — structured per-disc "Radiological Finding Assessment" reports
- **Stage G** — longitudinal-ready record schema and plan

### Completed — Sprint 2 extended

- Resumable training with per-epoch checkpoints, dual best-checkpoints
  (validation Dice and validation loss), early stopping, and RNG-state capture
- 30 epochs total, converged; test set evaluated once, on the selected checkpoint
- Disc-level analysis re-run and compared (justified by the +4.2 pp indexing gain)
- End-to-end Pfirrmann evaluation: models fitted on ground-truth-mask features,
  evaluated on **predicted**-mask features, which is the honest pipeline number

### Future — Sprint 3

- Retrain segmentation on all 9,128 training slices with a wider network. The
  extended run establishes that **width 16 on 2,560 slices/epoch has converged**,
  so the remaining levers are model capacity and training-data volume
- Post-processing of predicted masks (largest-component filtering, morphological
  cleanup, enforcing disc-count plausibility)
- Learned ROI-based finding classifier to replace the summary-statistic baseline,
  particularly for focal findings such as herniation
- Probability calibration before any probability is shown in a report
- Confidence intervals on every metric, by patient-level bootstrap

---

## Dataset

Sagittal lumbar spine MRI with voxel-level annotations of vertebrae,
intervertebral discs and the spinal canal. Supplied as four files, kept
untouched in `data/raw/`:

| file | size | content |
| --- | --- | --- |
| `images.zip` | 3.70 GB | 447 MRI volumes (`.mha`) |
| `masks.zip` | 58.2 MB | 447 matching mask volumes (`.mha`) |
| `overview.csv` | 120 KB | 447 rows × 39 columns of per-series metadata |
| `radiological_gradings.csv` | 34 KB | 1,520 rows of per-patient/per-disc gradings |

This is the publicly described SPIDER lumbar spine dataset
([Grand Challenge](https://spider.grand-challenge.org/data/);
[van der Graaf et al., *Scientific Data* 11, 264, 2024](https://www.nature.com/articles/s41597-024-03090-w)).
Every property listed below was nevertheless **measured from the files**, not
taken from the description.

### What the inspection found

| property | measured value |
| --- | --- |
| Format | MetaImage `.mha`, 3-D volumes (not 2-D image files) |
| Voxel dtype | `int16` |
| Series | 447 |
| **Patients** | **218** (1–3 series each: `t1`, `t2`, `t2_SPACE`) |
| Modalities | `t2` 210, `t1` 196, `t2_SPACE` 41 |
| Unreadable volumes | 0 |
| Unmatched image/mask files | 0 |
| In-plane matrix | 216–3682 × 264–1168 px, 123 distinct row sizes |
| In-plane pixel spacing | 0.077–1.233 mm |
| Slice spacing | 0.86–9.63 mm |
| Sagittal slices per volume | 8–154 (median 24) |
| Stored orientation | `LPS` for 374 series, `PIR` for 73 |
| Mask label values | `0`, `1–9`, `100`, `201–209` — 20 values, none undocumented |
| Intensity conventions | **two**: `[-1000, 3096]` (374 series) and `[0, …]` (73 series) |

### Mask labels

| raw value | meaning |
| --- | --- |
| `0` | background |
| `1…9` | vertebrae, numbered from the most inferior upward |
| `100` | spinal canal |
| `201…209` | intervertebral discs, numbered from the most inferior upward |

### Radiological gradings

`radiological_gradings.csv` holds **1,520 rows × 10 columns** — one row per
(patient, intervertebral disc) for all 218 patients. It has **zero missing
values**, no duplicate rows and no duplicate keys. 1,518 rows are valid; 2 carry
an invalid `IVD label` of 0.

| column | kind | domain | prevalence / distribution |
| --- | --- | --- | --- |
| `Patient` | identifier | 1…257 | matches the MRI filename prefix |
| `IVD label` | identifier | 1…9 (+2 invalid `0`) | maps to mask label `200 + N` |
| `Modic` | nominal | 0,1,2,3 | none 66.1%, I 0.3%, II 33.1%, III 0.5% |
| `UP endplate` | binary | 0,1 | 40.4% present |
| `LOW endplate` | binary | 0,1 | 41.0% present |
| `Spondylolisthesis` | binary | 0,1 | 2.8% present |
| `Disc herniation` | binary | 0,1 | 4.7% present |
| `Disc narrowing` | binary | 0,1 | 35.8% present |
| `Disc bulging` | binary | 0,1 | 49.1% present |
| `Pfirrman grade` | ordinal | 1…5 | 18.7 / 22.5 / 27.5 / 19.2 / 12.1% |

`UP endplate` / `LOW endplate` are the endplate-defect / Schmorl's-node labels
for the upper and lower endplate of each disc space.

Key structural facts, all verified:

- **Gradings are per patient, not per series.** One grading set is shared by a
  patient's 1–3 series, so attaching labels to series inflates 1,518 real labels
  to 3,143 instances (factor 2.07) without adding information. The effective
  sample size is **218 patients / 1,518 discs**.
- `IVD label` N ↔ mask label `200 + N`, confirmed by exact agreement in
  **432/447 series (96.6%)**. The 15 disagreeing series are listed in the report.
- The task is **multi-label**: a disc carries 2.08 findings on average (up to 7);
  31.6% carry none.
- Findings concentrate in the lower lumbar discs, the expected anatomical
  gradient. Record counts collapse at upper levels (218 at levels 1–6, then 144,
  55, 12).

Full analysis:
[`outputs/reports/radiological_grading_analysis.md`](outputs/reports/radiological_grading_analysis.md)

Preprocessing derives two label spaces (see `data/processed/label_mapping.json`):

- **semantic**, the project target — `0` background, `1` vertebra, `2` IVD, `3` spinal canal
- **instance**, loss-less — `0` background, `1–9` vertebrae, `10` canal, `11–19` IVDs

---

## Project structure

```
lumbar-spine-segmentation/
├── tutorials/                     EHR tutorial notebooks, unmodified reference only
│   ├── Tutorial_1_Health_care.ipynb
│   └── Tutorial_2_Health_care.ipynb
├── data/
│   ├── raw/                       the four original files, never modified
│   ├── extracted/{images,masks}/  447 + 447 .mha volumes
│   └── processed/
│       ├── slices/                preprocessed 2-D slices (.npz)
│       ├── slice_index.csv        one row per slice, incl. split assignment
│       ├── splits.csv             series → train/val/test
│       └── label_mapping.json     label space definitions
├── notebooks/
│   └── 01_preprocessing.ipynb     presentation-ready Sprint 1 walkthrough
├── src/
│   ├── preprocessing/
│   │   ├── extract.py             reproducible archive extraction
│   │   ├── volume_io.py           .mha reading + RAS reorientation
│   │   ├── labels.py              label semantics and remapping
│   │   ├── pairing.py             identifier-driven image↔mask pairing
│   │   ├── validate.py            dataset inspection and integrity checks
│   │   ├── transforms.py          image/mask preprocessing operations
│   │   ├── quality_metrics.py     objective metrics that justify each step
│   │   ├── dataset.py             builds the preprocessed slice dataset
│   │   ├── splits.py              patient-level splitting
│   │   └── visualize.py           figures
│   ├── analysis/
│   │   ├── gradings.py            grading schema, distributions, linkage audit
│   │   ├── grading_plots.py       grading distribution figures
│   │   ├── disc_features.py       Stage C disc instances + mm measurements
│   │   └── report_generator.py    Stage F/G structured finding reports
│   ├── models/
│   │   ├── unet.py                Stage A baseline 2-D multi-class U-Net
│   │   ├── data.py                Dataset/DataLoader over Sprint 1 slices
│   │   ├── losses.py              Dice + cross-entropy
│   │   ├── metrics.py             Dice / IoU / precision / recall
│   │   ├── predict_viz.py         Stage B prediction figures
│   │   └── baseline_findings.py   Stage E ordinal + binary estimators
│   └── utils/
│       ├── paths.py               single source of truth for all paths
│       └── reporting.py           Markdown/JSON report builders
├── scripts/
│   ├── 01_extract.py              Sprint 1
│   ├── 02_inspect.py
│   ├── 02b_justify_steps.py
│   ├── 03_preprocess.py
│   ├── 04_split.py
│   ├── 05_grading_analysis.py     Sprint 2 scoping
│   ├── train_unet.py              Stage A (has --smoke-test)
│   ├── evaluate_unet.py           Stage B
│   ├── 06_disc_analysis.py        Stages C + D
│   ├── 07_baseline_findings.py    Stage E
│   ├── 08_generate_reports.py     Stages F + G
│   ├── 09_compare_gt_vs_predicted.py  error propagation audit
│   └── build_notebook.py
├── outputs/
│   ├── models/                    U-Net checkpoints
│   ├── metrics/                   segmentation + baseline metrics
│   ├── visualizations/
│   │   └── predictions/           Stage B 6-panel prediction figures
│   ├── preprocessing_reports/     Sprint 1 preprocessing reports
│   └── reports/
│       ├── finding_assessments/   Stage F per-patient reports
│       └── longitudinal_plan.md   Stage G
├── requirements.txt
└── README.md
```

> `tutorials/` holds two notebooks that use a **different Electronic Health
> Record dataset**. They are preserved byte-for-byte as learning material and
> are not part of this project's pipeline.

---

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` lists only packages the code actually imports. `scikit-learn`,
`scikit-image` and `Pillow` are deliberately excluded — the stratified split and
the image-quality metrics are implemented with `numpy`/`scipy` directly.

## Running the pipeline

Each step is independent and re-runnable. `data/raw/` is only ever read.

```bash
python scripts/01_extract.py          # ZIPs -> data/extracted/   (resume-safe)
python scripts/02_inspect.py          # inspect all 447 pairs -> reports
python scripts/02b_justify_steps.py   # score preprocessing variants -> report
python scripts/03_preprocess.py       # preprocess all slices + figures
python scripts/04_split.py            # patient-level train/val/test split
python scripts/05_grading_analysis.py # grading analysis + Sprint 2 scoping
```

### Sprint 2

```bash
python scripts/train_unet.py --smoke-test   # 14 plumbing checks, ~2 min
python scripts/train_unet.py                # baseline training (~2.6 h on CPU)
python scripts/evaluate_unet.py --split test --save-predictions
python scripts/06_disc_analysis.py          # disc instances + grading linkage
python scripts/07_baseline_findings.py      # baseline finding estimators
python scripts/08_generate_reports.py       # per-disc reports + Stage G table
python scripts/06_disc_analysis.py --mask-source prediction
python scripts/09_compare_gt_vs_predicted.py  # error propagation
```

### Sprint 2 extended (controlled 30-epoch experiment)

Resumes from the baseline checkpoint and writes to separate directories, so the
baseline results and checkpoints are never overwritten. Safe to interrupt and
re-run — it resumes from `last.pt` with optimiser, scheduler and RNG state.

```bash
python scripts/train_unet_extended.py --max-epochs 30 --patience 6

python scripts/evaluate_unet.py --split test \
    --checkpoint outputs/checkpoints/sprint2_extended/best_val_dice.pt \
    --tag sprint2_extended --save-predictions \
    --predictions-dir data/processed/predictions_sprint2_extended

python scripts/06_disc_analysis.py --mask-source prediction \
    --predictions-dir data/processed/predictions_sprint2_extended \
    --suffix _sprint2_extended

python scripts/09_compare_gt_vs_predicted.py \
    --predicted outputs/reports/disc_analysis_sprint2_extended.csv \
    --out-dir outputs/reports/sprint2_extended/metrics \
    --viz-dir outputs/visualizations/sprint2_extended

python scripts/10_compare_experiments.py    # baseline vs extended verdict
```

**Always run the smoke test first.** It verifies tensor shapes, dtypes, class
IDs, forward pass, loss, backpropagation (including that gradients reach the
first layer and that weights change), and prediction generation, before
committing to a multi-hour run.

Then open the notebook:

```bash
jupyter lab notebooks/01_preprocessing.ipynb
```

Useful flags: `--limit N` on `02_inspect.py` / `03_preprocess.py` for a fast
smoke test, and `--seed` / `--train` / `--val` / `--test` on `04_split.py`.

---

## Preprocessing pipeline

Applied to every sagittal slice. Image and mask go through the **same**
geometric transform, differing only in interpolation, which is what keeps them
pixel-aligned.

| # | step | rationale |
| --- | --- | --- |
| 1 | Reorient volume to canonical `RAS` | stored orientation varies (`LPS`/`PIR`), so the sagittal axis is not the same array axis in every file. Pure axis permutation + flips, therefore loss-less and mask-safe |
| 2 | Extract sagittal slices | the data is 3-D; the project target is 2-D segmentation |
| 3 | Drop slices with < 10 mm² of annotation | lateral slices fall outside the spine. Threshold is a **physical area**, not a pixel count, because pixel spacing varies ~16× |
| 4 | Resample in-plane to 1.0 mm/px | pixel spacing varies 0.077–1.233 mm; a pixel-count resize would render the same vertebra at different physical sizes. Image: area/bilinear. **Mask: nearest neighbour** |
| 5 | Centre crop/pad to 352 × 256 px | measured, not guessed — see below |
| 6 | Clip to foreground percentiles (1, 99) and scale to [0, 1] | two incompatible intensity conventions; background padding is up to ~70% of a volume, so statistics are restricted to foreground voxels |
| 7 | Median 3 × 3 denoise | best class contrast of the variants tested while retaining ~80% of boundary sharpness |
| 8 | CLAHE (clip 2.0, 8 × 8 tiles) | highest vertebra-vs-disc contrast of all seven variants; applied *after* denoising because CLAHE amplifies noise |
| 9 | Remap labels to semantic + instance spaces | remap happens *after* the resize, so resizing never operates on a collapsed label space |
| 10 | Verify no label value was invented | automated hard check on every series |

### Why 352 × 256 and not a square

For every volume, the smallest **centred** crop that still contains the entire
annotation was computed from the inspection results. The lumbar spine is tall and
narrow: the worst case needs 341 mm superior–inferior but only 244 mm
anterior–posterior. A square 288 × 288 crop would have clipped annotated anatomy
in **157 of 447 volumes**. At 352 × 256 no volume loses any annotation, and both
dimensions are multiples of 32, which suits the downsampling depth of a U-Net in
a later sprint.

### Why masks use nearest neighbour only

Label values are categorical. Averaging vertebra `3` and vertebra `4` yields
`3.5`, which is not a structure; averaging a disc (`201`) with background (`0`)
yields ≈`100`, which is the spinal canal's label. Measured on one real slice:

| interpolation | label values invented |
| --- | --- |
| **nearest (used)** | **0** |
| bilinear | 159 |
| bicubic | 244 |

See `outputs/visualizations/mask_interpolation_comparison.png`.

### Steps considered and rejected

- **N4 bias field correction** — implemented in `transforms.py` and available via
  `PreprocessConfig.bias_field_correction`, but **off**. The decision is based on
  effect, not cost: at `shrink_factor=4` it takes ~0.5 s per volume, but the
  measured inhomogeneity changed by well under 1% and got slightly worse on one
  test volume. CLAHE already halves the same metric as a side effect of being a
  local equalisation.
- **Gaussian blur** — removes more noise than the median filter but loses more
  boundary sharpness and scores lower on vertebra-vs-disc contrast.
- **Bilateral filter** — preserves marginally more edge but is ~10× slower for no
  measurable gain here.

Full evidence: `outputs/preprocessing_reports/preprocessing_choices.md`.

---

## Train/validation/test split

Split on **`patient_id`**, never on individual slices, with a fixed seed (42) and
stratification by sex and annotated-vertebra count. Three independent reasons,
all established during inspection:

1. A series contributes 8–154 adjacent sagittal slices that are nearly identical
   to their neighbours.
2. A patient contributes 1–3 series of the *same* anatomy.
3. 107 groups of mask volumes are byte-identical, always within one patient — a
   single annotation shared across co-registered T1/T2 series.

Random slice splitting would place near-copies of the same image in both training
and test and would inflate the eventual Dice/IoU. `verify_no_leakage()` asserts
the partition afterwards rather than trusting it.

`overview.csv` ships its own `subset` column, but it only separates
training/validation (no test set) and is defined per series, so Sprint 1 derives
its own patient-level three-way split.

Exact counts: `outputs/preprocessing_reports/split_report.md`.

---

## Outputs

### Reports — `outputs/preprocessing_reports/`

| file | content |
| --- | --- |
| `dataset_inspection.md` / `.json` | formats, geometry, labels, intensity, duplicates, anatomical checks, problems found |
| `volume_inspection.csv` | one row per series, ~50 measured columns |
| `image_mask_pairs.csv` | the verified pairing table |
| `preprocessing_choices.md` / `.csv` / `.json` | measured comparison of preprocessing variants |
| `preprocessing_report.md` / `.json` | Sprint 1 data quality report |
| `series_preprocessing.csv` | per-series preprocessing metrics |
| `split_report.md` / `.json` | split strategy, counts, leakage verification |

### Reports — `outputs/reports/`

| file | content |
| --- | --- |
| `radiological_grading_analysis.md` / `.json` | full grading analysis, linkage audit, longitudinal capability audit, proposed Sprint 2 architecture |
| `grading_disc_level_prevalence.csv` | finding prevalence per disc level |
| `grading_series_linkage.csv` | grading↔mask agreement per series |
| `grading_patient_linkage.csv` | the same per patient |
| `disc_analysis.csv` | **Stage D analysis table** — 3,147 rows, one per (patient, series, disc), with gradings + measurements |
| `disc_analysis_predicted.csv` | the same measured from predicted masks (test series) |
| `disc_slice_measurements.csv` | Stage C per-slice measurements (51,558 records) |
| `disc_records_longitudinal.csv` | **Stage G** flat table with null time columns |
| `baseline_findings.md` | Stage E report |
| `longitudinal_plan.md` | **Stage G** plan |
| `finding_assessments/patient_XXX.{json,md,txt}` | **Stage F** per-disc reports |

### Metrics — `outputs/metrics/`

| file | content |
| --- | --- |
| `smoke_test.json` | the 14 Stage A verification checks and their results |
| `training_config.json` | exact training configuration |
| `training_history.csv` | per-epoch losses and validation metrics |
| `segmentation_metrics_test.json` | Dice/IoU/precision/recall, 3 aggregations |
| `segmentation_per_class_test.csv` | per-class table |
| `segmentation_per_patient_test.csv` | per-patient Dice |
| `segmentation_per_slice_test.csv` | per-slice Dice for all 1,655 test slices |
| `disc_identification_test.json` | do derived disc indices match ground truth? |
| `baseline_findings.json` / `_summary.csv` | Stage E results |
| `measurement_agreement.json` / `.csv` | predicted vs ground-truth measurement error |

### Figures — `outputs/visualizations/`

- `sample_*_A-E.png` — original MRI, original mask, preprocessed MRI,
  preprocessed mask, overlay
- `sample_*_before_after.png` — before/after with intensity histograms
- `stages_*.png` — stage-by-stage pipeline breakdown
- `mask_interpolation_comparison.png` — nearest vs bilinear vs bicubic label corruption
- `dataset_overview_grid.png` — consistency across samples
- `class_distribution.png` — semantic class pixel distribution
- `split_distribution.png` — patients/series/slices per split
- `grading_finding_distributions.png` — distribution of all eight findings
- `grading_prevalence_by_disc_level.png` — prevalence and record counts per level
- `grading_cooccurrence.png` — Spearman correlation between findings
- `grading_findings_per_disc.png` — how many findings co-occur per disc
- `grading_split_class_balance.png` — grading class counts per split

---

## Dataset problems found

| problem | handling |
| --- | --- |
| Matrix size, pixel spacing, slice thickness and stored orientation all vary | RAS reorientation + resampling to a common physical scale |
| Two incompatible intensity conventions (`[-1000, 3096]` vs `[0, …]`) | foreground-restricted percentile normalisation |
| Background padding up to ~70% of a volume; some series ~5% saturated at the ceiling | padding detected and excluded from statistics; percentile clipping |
| 107 groups of byte-identical mask volumes (shared T1/T2 annotation, same patient) | not corruption; makes patient-level splitting mandatory |
| `sex` has trailing-whitespace duplicates (`'F'` vs `'F '`) | all text columns stripped on load |
| A few series annotate an unusually short span of the spine | kept, flagged as outliers in the report |
| Severe class imbalance; intervertebral discs are the smallest target class | quantified in the report; will drive loss choice next sprint |

No cross-patient duplicate volumes, no unmatched image/mask files, no unreadable
volumes, and no undocumented label values were found.

---

## Sprint 2 limitations

Stated plainly, because several of these materially bound what the numbers mean.

### Compute

- **Trained on CPU only.** No CUDA device is present. Measured throughput at
  352×256 with `channels_last`: 3.99 img/s training, 12.93 img/s inference.
- Consequences: U-Net width **16** (1.96 M parameters) rather than the paper's 64
  (31.4 M); **2,560 slices per epoch** (sampled per patient, covering all 152
  training patients) rather than all 9,128; **12 epochs**, 158 min total.
- **The model had not converged.** Validation loss and Dice were still improving
  at the final epoch. These numbers are a floor, not a ceiling.
- `bfloat16` autocast was measured and rejected — **21× slower** on this CPU,
  which lacks native bf16 support.

### Segmentation → disc identity

- The U-Net predicts four **semantic** classes; it does not predict which disc is
  which. Disc identity is derived afterwards by ordered connected components.
- Measured propagation: **95.3%** of test discs have their region found, but only
  **78.4%** get the correct index **per slice** — the numbering shifts when a disc
  is merged or missed. Aggregating across slices per series recovers **99.3%**,
  so series-level aggregation is far more robust than per-slice identity.
- **The Stage D/E/F tables therefore use ground-truth instance masks**, which
  isolates finding assessment from segmentation error. The predicted-mask
  comparison is reported separately rather than mixed in.

### Measurement error from predicted masks

Compared on the 444 correctly-identified test discs:

| measurement | MAE | % of mean | r |
| --- | --- | --- | --- |
| `intensity_disc_vertebra_ratio` | 0.037 | 9.3% | **0.96** |
| `ap_extent_mm` | 2.13 mm | 6.3% | 0.77 |
| `height_mm_central` | 0.85 mm | 12.0% | 0.87 |
| `area_mm2` | 27.0 mm² | 12.7% | 0.89 |
| `disc_to_vertebra_height_ratio` | 0.134 | **70.3%** | 0.63 |

**`disc_to_vertebra_height_ratio` is not usable from predicted masks.** It
depends on separating individual vertebrae, which is harder than separating
discs, so its denominator is unreliable. Intensity ratios are the most robust.

### Finding assessment

- **Effective sample size is 218 patients / 1,518 discs**, not 12,415 slices.
  Gradings are a patient-level annotation, so the table is deduplicated to one
  series per patient before fitting.
- The test split holds 33 patients. For spondylolisthesis that is **5 positive
  discs** — its PR-AUC of 0.367 is a very uncertain point estimate and no
  confidence interval has been computed yet.
- Herniation scores lowest of the well-supported findings (PR-AUC 0.436). It is a
  *focal* shape abnormality, and summary statistics over a whole disc are not
  expected to capture it. This reflects the feature set, not an impossible task.
- **Modic type is not modelled.** Types I and III have 4 and 7 records in the
  whole dataset, with **zero type I in the test split**. Only binary "any Modic
  change" is used.
- **Pfirrmann is restricted to T2 / T2 SPACE series.** The grade is defined on T2
  signal; 6 patients are T1-only and are excluded from that target.
- **Probabilities are not calibrated.** They should not be read as
  well-calibrated risks until temperature scaling is applied on the validation
  split.

### Data

- **Level names are provisional.** The dataset numbers discs from the lowest
  upward and documents that the lowest vertebra is usually L5 but can be L4 or
  L6. Names such as `L5-S1` are flagged `level_name_is_provisional: true`; the
  integer disc index is authoritative.
- 12 of 3,147 disc rows have no matching grading, and 5 measured (patient, disc)
  pairs do not exist in the grading file.
- **No longitudinal capability.** See `outputs/reports/longitudinal_plan.md`.

### Sprint 2 extended — additional limitations

- **The resume was warm-restarted, not bit-exact.** The baseline checkpoint stored
  weights only, with no optimiser or scheduler state, so Adam's moment estimates
  restarted from zero. The baseline also used `CosineAnnealingLR(T_max=12)`, whose
  LR had annealed to ~0 by epoch 12; the extended run re-cast the schedule over
  30 epochs and fast-forwarded it to epoch 12 (LR 6.545e-4). **This run is
  therefore not equivalent to an uninterrupted 30-epoch run.** The warm restart
  is visible in the curves: validation Dice dropped to 0.851 at epoch 13 before
  recovering and exceeding the baseline at epoch 22. Checkpoints written from
  now on are fully resumable.
- **Early stopping never fired, and that was a configuration flaw.** `min_delta`
  was 0, so an improvement of +0.00004 reset the patience counter. With
  `min_delta ≈ 0.001` the run would have stopped around epoch 25–26 at
  effectively the same result. Fix this before the next long run.
- **The improvement is small and the test set is one sample of 33 patients.** A
  +0.0027 macro Dice change has no confidence interval attached. It should not be
  treated as a significant difference, only as a consistent direction across 43
  of 53 metrics.
- **`disc_to_vertebra_height_ratio` remains unusable from predicted masks**
  (MAE 71% of the mean, essentially unchanged). Vertebra instance separation did
  not improve.
- **Same 2,560-slice training subset.** The controlled design holds the data
  fixed, so this experiment says nothing about what more unique training slices
  would do.

### Not claimed

No composite severity score, no "damage percentage", no recovery percentage, no
true longitudinal or postoperative capability, no healing or recovery
metric, no 6-/9-month comparison, no longitudinal improvement, no diagnosis, and
no final clinical performance claim. The dataset contains a single timepoint per
patient and no postoperative follow-up imaging.

---

## Conventions

- Slices are stored and displayed in standard radiological sagittal orientation:
  **superior at the top, anterior on the left.**
- Preprocessed slices are one compressed `.npz` per slice containing
  `image` (`float16`, [0, 1]), `mask` (`uint8`, semantic) and `mask_instance`
  (`uint8`).
- Slices are stored flat and the split is recorded in `slice_index.csv`, so the
  split can be re-seeded without re-running preprocessing.
- Single random seed (42) in `src/utils/paths.py`.
- All paths derive from `src/utils/paths.py`, so the project can be moved without
  edits.
