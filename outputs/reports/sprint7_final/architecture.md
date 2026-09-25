# Architecture

**Pipeline version:** `SPIDER-Lumbar-v1`
**Scope:** the architecture as actually implemented and running. Nothing
aspirational is described here. Where a component was considered and rejected,
that is stated explicitly.

---

## 1. Request path, end to end

```
Browser (Chrome / Edge / Firefox)
   │  HTTP
   ▼
Next.js 15 frontend                     :3000
   │  fetch() → REST, JSON + PNG
   ▼
FastAPI backend                         :8000
   │
   ├─ routers/health.py      → pipeline identity, served metrics
   └─ routers/analysis.py    → upload, sample, run, status, result,
                                slice, report, download, history, delete
        │
        ▼
   services/jobs.py          JobManager, ThreadPoolExecutor(max_workers=1)
        │
        ▼
   services/mri_analysis_service.py      the analysis service
        │
        ▼
   ml/pipeline.py            serving adapter, five composable steps
        │
        ├─ 1. load volume          ml/volume.py → SimpleITK
        ├─ 2. preprocess           src/  Sprint 1: resample 1.0 mm/px,
        │                                crop/pad 352×256, normalise,
        │                                median denoise, CLAHE
        ├─ 3. segment              src/  Sprint 3 U-Net checkpoint (frozen)
        │                                UNet(in=1, classes=4, base=16,
        │                                depth=4, bilinear) 1,963,860 params
        ├─ 4. post-process         src/analysis/disc_postprocess.py
        │                                Sprint 5 5A (series-level ordering)
        │                              + 5D (vertebral-body separation)
        └─ 5. features + findings  src/  30 geometric+intensity features
                                         → 7 persisted estimators (.joblib)
        │
        ▼
   structured JSON result  +  arrays.npz (images, semantic, instance)
        │
        ├─ services/storage.py       → outputs/app_data/<analysis_id>/
        └─ services/report_writer.py → deterministic Markdown report
        │
        ▼
Frontend: results workspace, MRI viewer with server-rendered overlays,
          disc panel, report view, download
```

The arrow from step 2 to step 5 crosses into `src/`, which is the frozen
research code. `ml/` delegates; it does not reimplement. This is what keeps the
served pipeline and the published research identical.

---

## 2. Frontend technologies

| Concern | Choice | Notes |
| --- | --- | --- |
| Framework | Next.js 15.5.26, App Router | 9 routes, 7 static + 2 dynamic |
| UI library | React 19.2 | |
| Language | TypeScript, strict | `tsc --noEmit` is a gate |
| Styling | Tailwind CSS | dark workstation theme, tokens in `tailwind.config` |
| Charts | Recharts 3.10.1 | disc height context, Pfirrmann distribution |
| Icons | lucide-react | |
| Tests | Vitest 4.1.11 + Testing Library + jsdom | 18 behaviour tests |
| API client | `lib/api.ts`, `fetch` | single module, typed, throws `ApiError` |
| Types | `lib/types.ts` | hand-mirrored from the FastAPI OpenAPI schema |
| Display config | `lib/theme.ts` | segmentation colours, provenance and evidence styles |

Route inventory as built:

| Route | Rendering | Purpose |
| --- | --- | --- |
| `/` | static | dashboard: counts, recent analyses, active pipeline + served metrics |
| `/new` | static | upload, drag-and-drop, Load Sample Study |
| `/analysis/[id]` | dynamic | processing view, then results workspace |
| `/reports` | static | report index |
| `/reports/[id]` | dynamic | report view |
| `/history` | static | all analyses |
| `/methodology` | static | pipeline flow, served metric table, research progress |
| `/about` | static | scope, limitations, disclaimer |
| `/_not-found` | static | |

**No metric is hard-coded in the frontend.** Every research figure arrives from
`/api/health` (`validated_metrics`, `metric_table`, `research_progress`) or from
the analysis result. This was enforced in Sprint 7 by deleting the previously
hard-coded arrays.

Client-side state only. No Redux, no React Query, no server-side data fetching
for analysis data: pages fetch in `useEffect` and hold results in component
state, which is sufficient for a single-user local tool.

---

## 3. Backend technologies

| Concern | Choice | Notes |
| --- | --- | --- |
| Framework | FastAPI | OpenAPI at `/api/openapi.json`, Swagger at `/api/docs`, ReDoc at `/api/redoc` |
| Server | uvicorn | single process |
| Language | Python 3.12 | |
| Validation | Pydantic v2 | `backend/app/schemas.py` is the response contract |
| Imaging I/O | SimpleITK | `.mha`, `.mhd`, `.nii`, `.nii.gz` |
| Arrays | NumPy | |
| Inference | PyTorch, CPU | |
| Estimators | scikit-learn, persisted with joblib | |
| Rendering | Pillow | PNG encode of composited slices |
| Tests | pytest | 55 tests |

Configuration is a dataclass in `backend/app/config.py`, populated from the
environment with documented defaults:

| Setting | Default |
| --- | --- |
| `MODEL_CHECKPOINT` | `outputs/checkpoints/sprint3_coverage/best_val_dice.pt` |
| `FINDING_MODEL_DIR` | `outputs/models/findings` |
| `SPRINT5_PARAMS` | `outputs/reports/sprint5_indexing/test_results.json` |
| `STORAGE_DIR` | `outputs/app_data` |
| `JOB_TIMEOUT_SECONDS` | 1800 |
| `CORS_ORIGINS` | `http://localhost:3000, http://127.0.0.1:3000` |

The Sprint 4 checkpoint exists on disk but is deliberately not the default: it
measured worse on every Dice class and on disc indexing.

---

## 4. ML serving adapter

`ml/` is a thin adapter, added in Sprint 6, that exists for one reason: the
research layer in `src/` must stay frozen while still being callable from a web
request.

| Module | Responsibility |
| --- | --- |
| `ml/volume.py` | read a single uploaded volume, extract sagittal slices, report geometry |
| `ml/pipeline.py` | the five-step inference chain, each step delegating into `src/` |

Two design points that matter:

- **Single-volume preprocessing path.** The research preprocessing entry point
  (`preprocess_series`) requires a mask, which an upload does not have. The
  adapter instead composes the same primitives directly —
  `load_image` → `intensity_statistics` → `preprocess_image` per slice — which
  was verified to be self-contained with no dataset-level dependency. The
  arithmetic is identical to the research path.
- **Sprint 5 parameters are loaded from the published artefact**, not written
  into code. If `outputs/reports/sprint5_indexing/test_results.json` is missing,
  startup raises `FileNotFoundError` rather than falling back to dataclass
  defaults, because a silent default would change served behaviour.

A documented deviation: the brief asked for an `ml/` directory containing the
model code. `src/` was kept as the ML layer and `ml/` built as an adapter,
because renaming `src/` would have broken 18 research scripts and severed the
link between the application and the published results.

---

## 5. Model loading lifecycle

Models load **once per process**, in the FastAPI lifespan, not per request.

```
uvicorn start
  → lifespan enter
      → ModelService.load()
          ├─ torch.load(checkpoint)          Sprint 3 U-Net → eval mode, CPU
          ├─ joblib.load × 7                 finding estimators
          ├─ read manifest.json              supported/unsupported targets,
          │                                  per-target validated metrics
          └─ read sprint5 test_results.json  post-processing params
      → app.state.models = ModelService
      → app.state.settings, app.state.store, app.state.jobs
  → serving
  → lifespan exit → JobManager.shutdown(cancel_futures=True)
```

Measured effect of loading once: a served analysis takes **~2.9 s** versus
**14.5 s** when the checkpoint was loaded per invocation.

If the checkpoint fails to load, `/api/health` reports `status: "degraded"` and
analyses are refused with `MODEL_UNAVAILABLE` (503) rather than returning
partial output.

---

## 6. Storage

Local filesystem. One directory per analysis under `STORAGE_DIR`
(`outputs/app_data/` by default). No database.

```
outputs/app_data/<analysis_id>/
├── record.json      status, progress, stage, timestamps, study info, counts
├── result.json      the full structured result
├── arrays.npz       images (uint8), semantic (uint8), instance (uint8), slice_ids
└── <upload>.mha     the validated upload
```

Why this rather than a database: the data is already file-shaped, the tool is
single-user and local, and standing up Postgres would add a service to run for
no benefit.

Implementation details that are deliberate:

- **`analysis_id` is `secrets.token_hex(8)`** — a random 16-character hex token,
  never derived from the filename, so no identifying information appears in a
  URL.
- **Path traversal is refused at the boundary.** `AnalysisStore.directory()`
  rejects any id that is not pure hex before it is used to build a path.
- **Writes are atomic.** `record.json` and `result.json` are written to a `.tmp`
  sibling and then `replace()`d, so a crash mid-write cannot leave a truncated
  JSON file.
- **Display images are stored as `uint8`, not `float32`.** They exist only to be
  rendered as PNG, so a third of the size costs nothing visible.
- **Voxel data is never logged.** Filenames are sanitised to a basename and a
  conservative character set before touching the filesystem.

---

## 7. Asynchronous analysis lifecycle

`POST /run` returns immediately; the browser polls `GET /status`.

```
POST /upload  or  POST /sample
   → validate → write upload + record.json → 201 {analysis_id, status: "queued"}

POST /{id}/run
   → JobManager.submit() → ThreadPoolExecutor.submit() → 202 {status: "queued"}

worker thread
   → MriAnalysisService.run(progress=callback)
   → each real stage boundary calls progress(value, stage)
   → JobState updated in memory AND mirrored to record.json

GET /{id}/status   (polled)
   → live JobState if this process owns the job
   → otherwise rebuilt from record.json

on success  → result.json + arrays.npz written, status "completed", progress 100
on failure  → status "failed", error envelope with code + stage + exception type
on timeout  → ANALYSIS_TIMEOUT (limit 1800 s)
```

Why a single-process thread pool rather than Celery + Redis: inference is
CPU-bound and memory-constrained, the workload is one study at a time on a
laptop, and two extra services would buy nothing here.

Properties the implementation guarantees:

- **Progress comes from real stage boundaries, never a timer.**
  `Upload validated` 0 → `Loading MRI` 10 → `Preprocessing` 20 →
  `Segmentation` 40 → `Disc indexing` 60 → `Feature extraction` 75 →
  `Radiological analysis` 90 → `Complete` 100.
- **Progress is monotonic.** `max(state.progress, value)` — the bar cannot move
  backwards.
- **Liveness is process-local by definition.** `is_active()` consults only the
  in-memory table. A persisted `queued` record can exist simply because upload
  created it, so consulting the record would block resubmission — this was a real
  bug found in Sprint 6 and fixed.
- **A stale `processing` record is honest about it.** After a restart, `/status`
  reports `ANALYSIS_INTERRUPTED` instead of leaving the UI polling forever.
- **A job never crashes the app.** The worker catches broadly, logs the
  traceback server-side, and surfaces only the exception *type* plus the stage.

---

## 8. API endpoints

Base: `http://localhost:8000`. All endpoints under `/api`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | pipeline identity, served metrics, research progress, provenance/evidence labels, `sample_study_available`, model state |
| POST | `/api/analysis/upload` | `multipart/form-data`, one `file` field → 201 |
| POST | `/api/analysis/sample` | create an analysis from the bundled sample volume → 201 |
| POST | `/api/analysis/{id}/run` | queue the analysis → 202 |
| GET | `/api/analysis/{id}/status` | poll progress |
| GET | `/api/analysis/{id}/result` | full structured result; 409 until complete |
| GET | `/api/analysis/{id}/slice/{slice_id}` | one rendered slice as PNG |
| GET | `/api/analysis/{id}/volume` | the whole volume for browser-side 3D rendering (Sprint 8) |
| GET | `/api/longitudinal/demo-cases` | simulated longitudinal demonstration cases (Sprint 8) |
| GET | `/api/longitudinal/demo-cases/{case_id}` | one demonstration case |
| GET | `/api/longitudinal/demo-cases/{case_id}/stage/{stage_id}` | one timeline stage |
| GET | `/api/analysis/{id}/report` | report payload |
| GET | `/api/analysis/{id}/download` | `fmt=md` or `fmt=json`, as an attachment |
| GET | `/api/analysis` | history, `limit` 1–200 |
| DELETE | `/api/analysis/{id}` | remove upload, result and cached arrays |

Slice query parameters:

| Parameter | Default | Range |
| --- | --- | --- |
| `mode` | `overlay` | `original` \| `mask` \| `overlay` |
| `opacity` | `0.45` | 0–1 |
| `classes` | `1,2,3` | comma-separated class ids |
| `scale` | `1` | 1–3 integer upscale |
| `highlight_disc` | none | 1–9 |

**Uniform error envelope** at every status code, with no traceback ever returned:

```json
{ "error": { "code": "UNREADABLE_VOLUME", "message": "...", "details": { } } }
```

Upload validation runs in order — extension, size (2 KB–300 MB), readability,
dimensionality, slice count (3–512), in-plane spacing plausibility, finite
voxels — and a rejected upload is deleted with no analysis created. Codes
include `UNSUPPORTED_FORMAT`, `FILE_TOO_LARGE`, `UNREADABLE_VOLUME`,
`UNEXPECTED_DIMENSIONS`, `IMPLAUSIBLE_SPACING`, `RESULT_NOT_READY`,
`SLICE_OUT_OF_RANGE`, `MODEL_UNAVAILABLE`, `ANALYSIS_INTERRUPTED`.

> **Documentation note.** `docs/api.md` was written in Sprint 6 and does not yet
> describe the Sprint 7 additions (`/sample`, `highlight_disc`, the pipeline and
> metric fields on `/health`, `structured` lines on each disc). The FastAPI
> OpenAPI document is generated from the code and remains authoritative.

---

## 9. Result and provenance schema

Defined in `backend/app/schemas.py`, mirrored in `frontend/lib/types.ts`.

Top level of `AnalysisResult`:

```
analysis_id, status, created_at, completed_at, is_sample
pipeline           { version, preprocessing, segmentation, postprocessing,
                     *_detail, excluded, dataset }
study              { filename, format, modality, modality_source, slice_count,
                     dimensions, in_plane_spacing_mm, slice_spacing_mm,
                     native_orientation }
timepoint          { id: "baseline", label, acquired_at: null, is_baseline }
segmentation       { classes[] with validated_test_dice, slice_count,
                     informative_slice_count, macro_foreground_dice_validated }
processing          { model, source, postprocessing_method + params, device,
                     duration_seconds, stage_durations, disc_tracks_found,
                     components_rejected, components_unassigned }
discs[]            see below
summary            { disc_count, discs_with_findings_count, mean_disc_height_mm,
                     findings[], unsupported_targets }
validated_metrics  dataset-level benchmark figures + note
provenance_labels  { segmentation_derived, model_prediction, unsupported }
evidence_labels    { strong, moderate, modest, weak }
disclaimer
```

Each disc:

```
index                      1 = most inferior. NOT an anatomical level name.
structured[]               StructuredLine: label, value, provenance,
                           available, strength?, unavailable_reason?
measurements               DiscMeasurements, source "segmentation_derived_measurement"
findings[]                 FindingEstimate: value, probability, expected,
                           probabilities, strength, validated_metric,
                           validated_value, prevalence_baseline,
                           within_one_grade, source, caveat, unavailable_reason
identity_confidence        fraction of slices supporting this disc's track
representative_slice_*      the largest-area slice for this disc
pfirrmann_grade, modic_type, modic_any, bulging, narrowing,
herniation, spondylolisthesis, upper_endplate, lower_endplate
```

`StructuredLine` is the Sprint 7 addition that makes provenance a first-class
field rather than a UI convention. `provenance` is a closed set —
`segmentation_derived` | `model_prediction` | `unsupported` — and `available:
false` always comes with an `unavailable_reason`. The frontend renders the
label and reason verbatim; it never substitutes `0`, `No` or `Normal` for a
missing value.

Invariants enforced by the schema and covered by tests:

- `modic_type` and `spondylolisthesis` are always `null` with a reason.
- `pfirrmann_grade` is `null` with a reason on a non-T2 study.
- Any value that cannot be computed is `null` with a reason. Nothing is
  fabricated.
- `validated_metrics` is labelled as measured on a held-out split, not on the
  uploaded study.

---

## 10. Sample-study flow

```
frontend /new mounts
  → GET /api/health
  → health.sample_study_available === true  ? render the button : hide it

user clicks "Load Sample Study"
  → POST /api/analysis/sample
  → backend resolves sample_study_path(settings):
        data/extracted/images/33_t2.mha   (preferred)
        data/extracted/images/33_t1.mha
        data/extracted/images/1_t2.mha
     first candidate that exists and is > 2048 bytes
  → copied into the analysis directory and validated like any upload
  → 201 { analysis_id, study, is_sample: true }

user clicks "Start analysis"
  → identical POST /run path as an upload. No pre-computed result exists.

results header renders a "Sample research study" badge from is_sample
```

If no candidate is present the endpoint reports the volume is unavailable and
the button never appears. There is no canned-output fallback anywhere in this
path.

---

## 11. MRI slice and overlay flow

Rendering happens **server-side**. The volume never crosses the network.

```
completed analysis → arrays.npz { images, semantic, instance }

GET /slice/{i}?mode=overlay&opacity=0.45&classes=1,2,3&scale=1&highlight_disc=3
  → services/imaging.py
      render_slice(image[i], semantic[i], mode, opacity, classes, instance[i],
                   highlight_disc)
        ├─ mode original : grayscale only
        ├─ mode mask     : class colours only
        ├─ mode overlay  : alpha-composite colours over grayscale
        ├─ classes       : filtered BEFORE compositing
        └─ highlight_disc: draw_disc_highlight() marks that instance
      encode_png(rgb, scale)
  → image/png
     X-Slice-Count: <total>
     Cache-Control: public, max-age=86400, immutable
```

Class colours are defined once per side — `CLASS_COLOURS` in
`backend/app/services/imaging.py` and `SEG_CLASSES` in `frontend/lib/theme.ts` —
so the server-rendered overlay and the client-rendered legend provably agree.

Two consequences of rendering server-side:

- **A hidden class is never transmitted.** Toggling a class off changes the
  request, so the data does not reach the browser at all.
- **Slices of a completed analysis are immutable**, so the browser may cache
  them indefinitely and the viewer feels instant on re-navigation.

### 11a. Volume transfer for 3D — a deliberate reversal (Sprint 8)

> **This changes an architectural property this document previously stated
> without qualification.** Through Sprint 7, the volume never crossed the
> network. It does now, on one new path.

`GET /api/analysis/{id}/volume` sends the image volume and both label maps in
full. Browser-side volumetric rendering needs the voxels; there is no version of
a real 3D viewer that works on rendered PNGs.

What this costs: the data-minimisation guarantee described just above **does not
apply to this endpoint**. A client that calls it receives labels for classes the
user has hidden, because the whole volume goes at once. Class toggling in the 3D
view is a client-side transfer-function change, not a server-side filter.

What is retained:

| Property | Still true? |
| --- | --- |
| Only completed analyses are served | yes — `VOLUME_NOT_READY` (409) otherwise |
| Raw dataset files are never served | yes — only preprocessed `uint8` display arrays |
| One request per analysis, cached `immutable` | yes |
| Stored arrays are never mutated | yes |
| The 2D slice path is unchanged | yes — `imaging.py` untouched |

Wire format, defined once in `services/volume_export.py`:

```
[0:4]    uint32 little-endian  header_length
[4:4+H]  UTF-8 JSON header
[4+H:]   channel payloads, concatenated in header["channels"] order
```

Channels are `image`, `semantic`, `instance`, all `uint8`, all the same shape.
No base64, so nothing inflates by a third; gzip carries the compression. Measured
on the sample study: 6.2 MB of raw voxels leaves as roughly 1 MB, of which the
image volume is 943 KB and the two label maps are 13.7 KB and 10.7 KB.

The header carries `disc_instance_offset` and `finding_discs`, so the 3D view
highlights discs and marks findings from the **server's** decision rather than
re-deriving either in the browser. That is what keeps the 2D and 3D views from
disagreeing.

Why this is acceptable here: the deployment is localhost, single-user, already
unauthenticated (§14), and the voxels in question are preprocessed display
arrays from a licensed research dataset the operator already has on disk. It
would not be acceptable in a multi-tenant or networked deployment, and §14's
conclusion is unchanged — this must not be exposed without access control.

The viewer (`components/MriViewer.tsx`) supports next/previous slice, keyboard
navigation (arrows, Home/End), per-class visibility checkboxes, and mode
selection. Slice position is reported as `Slice n / total`.

---

## 12. Disc selection and highlight synchronisation

This is the Sprint 7 change that connects the two halves of the results
workspace.

```
ResultsView holds  selected: DiscResult | null

user clicks a row in DiscList
  → onSelect(disc)
  → selected = disc
      ├─ DiscDetail panel renders that disc
      │     (desktop: right-hand aside; mobile: bottom sheet)
      ├─ MriViewer focusSlice   = disc.representative_slice_index
      │     viewer jumps to the slice where this disc is largest
      └─ MriViewer highlightDisc = disc.index
            → slice request gains &highlight_disc=<index>
            → server re-renders that slice with the disc marked
```

The highlight is drawn from the `instance` array by the same code that produces
the overlay, so what is marked is exactly the component the pipeline assigned
that index. The panel and the image cannot disagree.

Closing the panel clears `selected`, which drops the `highlight_disc` parameter
and returns the viewer to a plain overlay.

---

## 13. Report generation

```
GET /api/analysis/{id}/report            → the structured payload, for the UI
GET /api/analysis/{id}/download?fmt=md   → Markdown, Content-Disposition: attachment
GET /api/analysis/{id}/download?fmt=json → the raw result
```

`services/report_writer.py` assembles Markdown from **deterministic templates
filled with measured values**. No language model is involved at any point, so the
same result always produces the same document.

Sections:

1. Study Information
2. Processing Information — including the pipeline version and generation
   timestamp added in Sprint 7
3. Segmentation Overview — per-class validated test Dice
4. Disc-Level Analysis
5. Quantitative Measurements
6. Model-Derived Findings — each statement generated by rule from the structured
   result
7. Limitations — including that validated performance is measured on a
   33-patient held-out test split, and that no postoperative assessment is
   provided
8. Research Disclaimer

---

## 14. Security limitation — read before exposing this

> **This deployment has no authentication, no authorisation and no transport
> security. It is built to run on `localhost` for a single researcher. It must
> not be exposed to a network or the public internet without an access-control
> layer in front of it.**

Concretely, as implemented:

| Property | State |
| --- | --- |
| Authentication | **none** — every endpoint is open |
| Authorisation | **none** — any caller can read or delete any analysis |
| Transport | plain HTTP, no TLS |
| Multi-tenancy | none — one shared analysis store |
| Rate limiting | none |
| Audit log | application log only, no per-user attribution |
| CORS | restricted to `localhost:3000` / `127.0.0.1:3000`, credentials disabled |

Anyone who can reach port 8000 can enumerate `GET /api/analysis`, read any
result, and `DELETE` any analysis. In a research setting with medical images
that is a meaningful exposure.

What exists already and helps, but is **not** a substitute for access control:
random non-identifying analysis ids, path-traversal rejection on the id,
filename sanitisation, no voxel data in logs, no traceback in responses, and a
CORS allowlist.

Minimum additions required before any shared or networked deployment:
authentication, per-user scoping of the analysis store, TLS, rate limiting, and
an audit trail. None of these are implemented.

---

## 15. What is deliberately not in this architecture

Listed so the diagram is not read as an abridgement of something larger.

| Not present | Why |
| --- | --- |
| Database | data is file-shaped; a local single-user tool does not need one |
| Celery / Redis / message broker | one CPU-bound study at a time; a thread pool suffices |
| Multi-GPU or batch serving | out of scope; CPU inference is ~3 s per study |
| DICOM ingestion | converted volume formats only; modality comes from a filename heuristic and is labelled as such |
| Model retraining or fine-tuning in the app | the pipeline is frozen by design |
| Real longitudinal / postoperative comparison | the dataset has a single timepoint and no outcome labels. Sprint 8 adds a **simulated** demonstration workflow (`demo/longitudinal_cases/`, `/api/longitudinal/*`) that stages four *different* real studies from four *different* patients to show how such a system would operate. It claims no recovery outcome, and the loader refuses any manifest declaring itself real follow-up. |
| Composite severity score | combining findings into one number would be an unvalidated clinical judgement |
| Server-side rendering of analysis data | client fetch is sufficient and keeps the API the single source of truth |
| Authentication | see §14 — this is a limitation, not a decision to be comfortable with |
