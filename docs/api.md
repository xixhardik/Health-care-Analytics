# API reference

Base URL `http://localhost:8000`. Everything lives under `/api`. The generated
OpenAPI document at `/api/openapi.json` is authoritative; this page is the
hand-written companion, and is current as of Sprint 8.

Interactive docs: `/api/docs` (Swagger) and `/api/redoc`.

> **Research prototype.** Every finding is a model-derived research estimate
> requiring review by a qualified radiologist. Not a clinical diagnosis, not a
> medical device.

> **Security.** No authentication, no authorisation, no TLS. Built for
> `localhost`, single user. Do not expose without an access-control layer in
> front of it. See `outputs/reports/sprint7_final/architecture.md` §14.

---

## Conventions

**Error envelope.** Every non-2xx response, at every status code:

```json
{ "error": { "code": "UNREADABLE_VOLUME", "message": "...", "details": { } } }
```

No traceback is ever returned. Field names may appear in `details`; raw input
values do not.

**Compression.** `GZipMiddleware` compresses responses over 1 KB. This matters
most for the volume endpoint, where it takes ~6.2 MB of voxels to ~1 MB.

**Analysis ids** are 16-character random hex (`secrets.token_hex(8)`), never
derived from a filename. Non-hex ids are rejected before touching the filesystem.

---

## Health

### `GET /api/health`

Service state plus everything the interface needs so it never hard-codes a
research figure.

Returns: `status` (`ok` | `degraded`), `api_version`, `pipeline`,
`segmentation_model`, `finding_models`, `postprocessing_method`,
`validated_metrics`, `metric_table`, `research_progress`, `provenance_labels`,
`evidence_labels`, `sample_study_available`, `disclaimer`, `research_notice`.

`status` is `degraded` when the checkpoint failed to load; analyses are then
refused with `MODEL_UNAVAILABLE` (503) rather than returning partial output.

---

## Analysis lifecycle

### `POST /api/analysis/upload`

`multipart/form-data`, one `file` field. → `201` `UploadResponse`.

Validation order: extension → size (2 KB–300 MB) → readability →
dimensionality → slice count (3–512) → in-plane spacing plausibility → finite
voxels. A rejected upload is deleted and no analysis is created.

Codes: `UNSUPPORTED_FORMAT`, `FILE_TOO_LARGE`, `FILE_TOO_SMALL`,
`UNREADABLE_VOLUME`, `UNEXPECTED_DIMENSIONS`, `IMPLAUSIBLE_SPACING`.

### `POST /api/analysis/sample`

Creates an analysis from the bundled sample study. → `201`, `is_sample: true`.
`404 SAMPLE_UNAVAILABLE` when the volume is absent locally — the UI hides the
button rather than offering a dead end. Nothing is pre-computed.

### `POST /api/analysis/demo-stage/{case_id}/{stage_id}`

Creates an analysis from one stage of a simulated longitudinal demonstration
(Sprint 8). The stage's source study is a real SPIDER volume processed by the
identical pipeline. → `201`.

The response record carries `demo_stage`, which every later read of the analysis
returns, so the disclaimer travels with the study. Codes:
`DEMO_CASE_NOT_FOUND`, `DEMO_CASE_INVALID`, `DEMO_STAGE_NOT_FOUND`,
`DEMO_STAGE_UNAVAILABLE`.

### `POST /api/analysis/{id}/run`

Queues the analysis. → `202`, returns immediately; inference happens on a
worker thread. `409 ANALYSIS_ALREADY_RUNNING` if already active.

### `GET /api/analysis/{id}/status`

Poll for progress. Stages, in order, from real pipeline boundaries:

`Upload validated` 0 → `Loading MRI` 10 → `Preprocessing` 20 →
`Segmentation` 40 → `Disc indexing` 60 → `Feature extraction` 75 →
`Radiological analysis` 90 → `Complete` 100.

Progress is monotonic by construction. A stale `processing` record after a
restart reports `ANALYSIS_INTERRUPTED` rather than polling forever.

### `GET /api/analysis/{id}/result`

Full `AnalysisResult`. `409 RESULT_NOT_READY` until complete, `409
ANALYSIS_FAILED` if it failed.

Two fields are derived at read time rather than stored, so analyses completed
before those features existed still carry them:

- `finding_overlay` — which disc regions the findings overlay may mark
- `demo_stage` — set only for demonstration-stage analyses, otherwise `null`

### `GET /api/analysis` · `DELETE /api/analysis/{id}`

History (`limit` 1–200, newest first) and deletion of upload, result and cached
arrays.

---

## Imaging

### `GET /api/analysis/{id}/slice/{slice_id}`

One rendered slice as PNG. Rendering is server-side, so a class hidden in the
interface is never transmitted.

| Parameter | Default | Range |
| --- | --- | --- |
| `mode` | `overlay` | `original` \| `mask` \| `overlay` |
| `opacity` | `0.45` | 0–1 |
| `classes` | `1,2,3` | comma-separated class ids |
| `scale` | `1` | 1–3 |
| `highlight_disc` | none | 1–9 |
| `highlight` | `disc` | `none` \| `disc` \| `findings` |
| `finding_opacity` | `0.5` | 0–1 |

`highlight=findings` tints disc regions the server decided are associated with a
supported positive model-estimated finding. The membership rule lives in
`services/findings_overlay.py`; a client cannot widen it. Red marks a
finding-associated *disc region*, drawn from the validated disc segmentation —
it is **not** a pixel-level pathology claim.

Headers: `X-Slice-Count`, `X-Finding-Discs`,
`Cache-Control: public, max-age=86400, immutable`.

Codes: `SLICE_OUT_OF_RANGE` (404), `SLICES_NOT_READY` (409), `INVALID_CLASSES`,
`INVALID_REQUEST` (422 for a bad `highlight` or out-of-range `finding_opacity`).

### `GET /api/analysis/{id}/volume`

**Sprint 8. This endpoint intentionally reverses the previous "the volume never
crosses the network" architecture**, because browser-side volumetric rendering
needs the voxels. The slice endpoint above is unchanged and remains the 2D
viewer's path. The trade is documented in `architecture.md` §11a.

Completed analyses only (`409 VOLUME_NOT_READY` otherwise). Serves preprocessed
`uint8` display arrays, never raw dataset files. One request per analysis,
cached `immutable`.

Wire format — one response, no base64:

```
[0:4]    uint32 little-endian  header_length
[4:4+H]  UTF-8 JSON header
[4+H:]   channel payloads, concatenated in header["channels"] order
```

Channels `image`, `semantic`, `instance`, all `uint8`, all the same shape. The
header declares `dimensions`, `spacing_mm`, `axis_order`, `semantic_labels`,
`disc_instance_offset` and `finding_discs` — the last two so the 3D view
highlights discs and marks findings from the **server's** decision rather than
re-deriving either. That is what keeps the 2D and 3D views consistent.

Headers: `X-Volume-Format-Version`, `X-Volume-Dimensions`, `X-Finding-Discs`.

Measured on the sample study: 6,337 KB of voxels → **962 KB** gzipped.

---

## Reports

### `GET /api/analysis/{id}/report`

The same validated payload as `/result`, so a second schema cannot drift from
the first.

### `GET /api/analysis/{id}/download?fmt=md|json`

Markdown or JSON, as an attachment. The Markdown is assembled from deterministic
templates filled with measured values — no language model — so the same result
always produces the same document. Sections: Study Information, Processing
Information (with pipeline version and timestamp), Segmentation Overview,
Disc-Level Analysis, Quantitative Measurements, Model-Derived Findings,
Limitations, Research Disclaimer.

---

## Longitudinal demonstration (Sprint 8)

These endpoints serve **simulated** demonstration cases from
`demo/longitudinal_cases/`. The SPIDER dataset is cross-sectional and contains no
postoperative follow-up, so a case arranges *different real studies from
different patients* into a timeline to demonstrate the workflow. Every response
carries the disclaimer and per-stage provenance.

The loader **refuses** any manifest whose `type` is not
`SIMULATED_LONGITUDINAL_DEMO`, or which sets `is_true_followup` or
`is_same_patient` true. Serving such a manifest would misrepresent the dataset.

### `GET /api/longitudinal/demo-cases`

All readable cases. An empty list is valid — it means none is installed — and is
not an error. Also returns `notice` and `research_statuses`.

### `GET /api/longitudinal/demo-cases/{case_id}`

One case: `disclaimer`, `ui_notice`, `interpretation`, `measurement_policy`,
`selection`, `stage_count`, `available_stage_count`, and ordered `stages`.

Stages are sorted by the manifest's own `order`, so timeline position never
depends on filesystem or JSON key ordering.

### `GET /api/longitudinal/demo-cases/{case_id}/stage/{stage_id}`

One stage. `404 DEMO_STAGE_NOT_FOUND` lists the valid ids in `details`.

Each stage carries `provenance`:

| Field | Meaning |
| --- | --- |
| `source_study_id` | the actual SPIDER study, e.g. `177_t2` |
| `source_patient_id` | integer research id |
| `source_dataset` | `SPIDER` |
| `source_split` | `test` for every current stage |
| `is_true_followup` | **always `false`** |
| `measurement_source` | `real_pipeline` or `simulated_demo_value` |

`display_reference` (e.g. "Demonstration study A") is what the interface shows,
so a patient identifier is not surfaced without reason.

A stage whose source volume is absent locally reports `available: false` with a
reason. **No substitute scan is ever used.**

`volume_path` is deliberately **not** in the response: a client has no use for a
server filesystem path. Paths in the manifest are validated against `data/`
before use.

---

## Error codes

| Code | Status | Meaning |
| --- | --- | --- |
| `UNSUPPORTED_FORMAT` | 400 | extension not accepted |
| `FILE_TOO_LARGE` / `FILE_TOO_SMALL` | 400 | outside 2 KB–300 MB |
| `UNREADABLE_VOLUME` | 400 | the file could not be parsed as a volume |
| `UNEXPECTED_DIMENSIONS` | 400 | not a usable 3-D volume |
| `IMPLAUSIBLE_SPACING` | 400 | in-plane spacing outside a plausible range |
| `INVALID_CLASSES` | 400 | malformed `classes` |
| `INVALID_REQUEST` | 422 | parameter validation failed |
| `ANALYSIS_NOT_FOUND` | 404 | unknown analysis id |
| `SLICE_OUT_OF_RANGE` | 404 | slice index outside the study |
| `SAMPLE_UNAVAILABLE` | 404 | bundled sample not on this machine |
| `DEMO_CASE_NOT_FOUND` | 404 | no such demonstration case |
| `DEMO_STAGE_NOT_FOUND` | 404 | no such stage in that case |
| `DEMO_STAGE_UNAVAILABLE` | 404 | the stage's source study is missing |
| `DEMO_CASE_INVALID` | 422 | manifest malformed, or claims to be real follow-up |
| `RESULT_NOT_READY` | 409 | analysis not complete |
| `SLICES_NOT_READY` | 409 | arrays not written yet |
| `VOLUME_NOT_READY` | 409 | volume unavailable until complete |
| `VOLUME_UNAVAILABLE` | 409 | stored volume could not be serialised |
| `ANALYSIS_FAILED` | 409 | the pipeline failed on this study |
| `ANALYSIS_ALREADY_RUNNING` | 409 | already active |
| `ANALYSIS_INTERRUPTED` | 409 | stale record after a restart |
| `ANALYSIS_TIMEOUT` | 500 | exceeded `JOB_TIMEOUT_SECONDS` (1800) |
| `MODEL_UNAVAILABLE` | 503 | checkpoint not loaded |
