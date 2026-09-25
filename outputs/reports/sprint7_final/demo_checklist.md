# Live Demo Checklist

**Pipeline version:** `SPIDER-Lumbar-v1`
**Duration:** ~12 minutes at a comfortable pace
**Prerequisites:** two terminals, both at the repository root; the dataset
extracted to `data/extracted/images/`; `frontend/node_modules` installed; a
browser at 1440 px or wider for the desktop layout.

Tick each box as you go. Steps 1–26 are the demo path in order. Everything after
step 26 is what to avoid saying and how to recover if something breaks.

---

## Before the room fills (5 minutes, not part of the demo)

- [ ] `python -m pytest backend/tests -q` → expect **55 passed**
- [ ] `cd frontend; npm run typecheck` → expect clean
- [ ] `cd frontend; npx vitest run` → expect **18 passed**
- [ ] Confirm the sample volume exists: `data/extracted/images/33_t2.mha`
- [ ] Delete stale demo analyses so the dashboard starts clean, or keep one
      completed analysis if you want a fast fallback (see Recovery)
- [ ] Close other heavy applications — inference is CPU-bound
- [ ] Have `outputs/reports/sprint7_final/final_system_report.md` open in a
      second tab for metric questions

---

## 1. Start the backend

- [ ] Terminal 1:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

- [ ] Wait for the startup log confirming the checkpoint loaded. The model loads
      **once** here, not per request.
- [ ] Sanity check in a browser or a third terminal:
      `http://localhost:8000/api/health` → `"status": "ok"`

**Say:** the model is loaded once at startup, which is why an analysis takes
about three seconds instead of fifteen.

---

## 2. Start the frontend

- [ ] Terminal 2:

```bash
cd frontend
npm run dev
```

- [ ] Wait for `Ready` and the port. Expect `http://localhost:3000`.

---

## 3. Open the dashboard

- [ ] Browse to `http://localhost:3000`
- [ ] Point out the persistent research banner at the top of the shell
- [ ] Point out the pipeline version badge in the top bar

---

## 4. Verify health and pipeline information

- [ ] Scroll to the **Active pipeline** panel
- [ ] Confirm the version badge reads `SPIDER-Lumbar-v1`
- [ ] Confirm `Segmentation Sprint 3` and `Post-processing Sprint 5`
- [ ] Confirm the metric table is populated

**Say:** every number in this panel is served by the API from a single source of
truth in `pipeline_info.py`. The frontend holds no hard-coded research figures —
that was removed deliberately so the interface cannot drift from what was
measured.

---

## 5. Open New Analysis

- [ ] Click **New Analysis**
- [ ] Point out the accepted formats and the 300 MB limit
- [ ] Point out the **What will run** panel listing the five real stages

---

## 6. Load Sample Study

- [ ] Click **Load Sample Study**

**Say:** this button only exists because the API reported
`sample_study_available: true`. If the volume were missing the button would not
render at all — there is no canned-result fallback.

---

## 7. Confirm the sample-study indication

- [ ] Confirm the **Validated** badge appears
- [ ] Confirm the note: *Sample research study — processed by the real pipeline*
- [ ] Confirm the **Volume information** panel filled in from the server:
      format, size, sagittal slices, in-plane size, spacing, orientation
- [ ] Point out the modality badge reading `filename heuristic`

**Say:** the modality is honest about its own provenance. A converted `.mha`
carries no DICOM series description, so this is a filename heuristic and the UI
says so rather than implying it read it from the header.

---

## 8. Start the analysis

- [ ] Click **Start analysis**
- [ ] Note that the call returns immediately and the browser navigates to the
      processing view

---

## 9. Show real progress

- [ ] Watch the stages advance: `Upload validated` → `Loading MRI` →
      `Preprocessing` → `Segmentation` → `Disc indexing` →
      `Feature extraction` → `Radiological analysis` → `Complete`
- [ ] Point out completed stages marked `done` and the active one `in progress`

**Say:** progress comes from real pipeline stage boundaries, not a timer. It is
monotonic by construction — the bar cannot move backwards, and it is not
animating toward a guess.

---

## 10. Open the completed result

- [ ] The view switches to the results workspace on completion
- [ ] Point out the header: filename, `N discs`, `N slices`, and the pipeline
      version
- [ ] Point out the **Sample research study** badge

**Say:** the pipeline version travels with the result, so any exported report is
traceable to the exact served configuration.

---

## 11. Show the research disclaimer

- [ ] Point out the banner: research / educational prototype, expert
      radiological review required
- [ ] Scroll to the full disclaimer in the findings area

**Say this early and plainly:** this is a research prototype. It does not
diagnose. Everything on screen needs a radiologist.

---

## 12. Show provenance: segmentation-derived, model-estimated, unavailable

- [ ] Point out the provenance legend
- [ ] Identify one **Segmentation-derived** value (a measurement)
- [ ] Identify one **Model-estimated** value (a finding)
- [ ] Identify one **Not modelled** value (Modic type or spondylolisthesis)

**Say:** three different kinds of statement, tagged differently on purpose. A
measured millimetre is not the same kind of claim as a model's guess, and
neither is the same as "we never modelled this."

---

## 13. Open the MRI viewer

- [ ] Point out the segmentation overlay with the class legend
- [ ] Point out the validated test Dice shown per class

**Say:** these Dice figures are dataset-level — how the segmentation scored on 33
held-out patients. They are not this study's accuracy; we have no ground truth
for this study.

---

## 14. Navigate slices

- [ ] Click next / previous; watch `Slice n / N` update
- [ ] Use arrow keys, then `Home` and `End`

**Say:** slices are rendered server-side, so the volume itself never crosses the
network.

---

## 15. Toggle segmentation classes

- [ ] Uncheck **Intervertebral discs** → the disc overlay disappears
- [ ] Uncheck **Vertebrae**, then **Spinal canal**
- [ ] Re-enable all three
- [ ] Switch modes: `original` → `mask` → `overlay`

**Say:** class filtering happens on the server, so a hidden class is not merely
invisible — it is never transmitted.

---

## 15b. Turn on the Findings Overlay

This is the most visually immediate moment of the demo. Take it slowly.

- [ ] Note the control reads `N discs marked` before you switch it on
- [ ] Switch **Findings Overlay** to On
- [ ] Red appears on the discs carrying a supported positive finding; the other
      discs stay amber
- [ ] Switch the mode to **MRI** so the red sits on plain greyscale
- [ ] Drag the **Finding** opacity slider up and down
- [ ] Read the legend: red = model-estimated finding, green = anatomical
      segmentation, white = MRI
- [ ] Switch the overlay **Off** and show the identical original image returning

**Say, before anyone asks:** the red is the disc segmentation for a disc that
carries a model-estimated finding. It marks *which disc*, not *which pixels*.
There is no pixel-level pathology model in this system, so nothing inside the red
region has been classified, and the shape of the red tells you nothing about
where within the disc anything is.

**Also worth saying:** the discs left unmarked are not certified healthy. They
have no *supported positive* finding, which is a different statement.

The decision about which discs turn red is made by the server from the served
findings, not by the browser. Only the six validated binary targets can trigger
it. The Pfirrmann grade cannot, because a grade is not a positive/negative
finding and thresholding it would invent a severity cut-off nobody validated.

---

## 16. Select a disc from the result panel

- [ ] Click a disc row in the list
- [ ] The detail panel opens with `Disc N` and its identity support

**Say:** the index is a position, 1 being the most inferior disc. It is not an
anatomical level name — the dataset does not tell us which vertebra is L5, so we
do not assert one.

---

## 17. Confirm the disc is highlighted in the viewer

- [ ] Confirm the viewer jumped to that disc's representative slice
- [ ] Confirm the disc is visibly marked in the rendered image
- [ ] Select a different disc and confirm the highlight follows
- [ ] With the Findings Overlay on, pick a disc carrying the red dot in the list
      and confirm its red region renders more strongly than the others

**Say:** the highlight is drawn from the same instance map that produced the
overlay, so the panel and the image cannot disagree about which disc is which.

---

## 18. Inspect measurements

- [ ] Point out central height in mm and area in mm²
- [ ] Point out the **Segmentation-derived** tag on each
- [ ] Point out the height-across-this-study chart and its caption

**Say:** measurements are in real millimetres because preprocessing resampled to
1.0 mm per pixel. The chart is context within this study only — no clinical
reference range is implied, because the dataset does not define one.

---

## 19. Inspect model-estimated findings

- [ ] Point out the Pfirrmann grade and its probability distribution
- [ ] Point out one binary finding with its validated metric and prevalence
      baseline, e.g. *Validated test pr auc 0.8779 against a prevalence baseline
      of 0.323*
- [ ] Point out the qualitative evidence labels: Strong / Moderate / Limited /
      Weak
- [ ] Point out a **Weak evidence** target if one is present (herniation)

**Say:** evidence strength is qualitative on purpose. We do not show a
confidence percentage, because turning a model probability into a medical
confidence would invent a number nobody validated. The prevalence baseline is
shown so you can see what a random ranker would have scored.

---

## 20. Show unavailable findings and their reasons

- [ ] **Modic type** → *Not served. The nominal Modic type was never modelled.*
- [ ] **Spondylolisthesis** → *Not served. Too few positive cases to validate.*
- [ ] Point out that these render as `unavailable`, never as `0`, `No` or
      `Normal`

**Say:** this is the distinction the interface is built to protect. "We did not
model this" is not the same statement as "this patient does not have this." Only
five positive spondylolisthesis discs exist in the test split, so reporting it
would be dressing up noise.

---

## 21. Open Methodology

- [ ] Navigate to **Methodology**
- [ ] Walk the pipeline flow: preprocessing → segmentation → post-processing
- [ ] Point out the **Research progress** section
- [ ] Point out **Sprint 4**, marked `rejected`

**Say:** the rejected experiment is on the page deliberately. Doubling model
width made every metric worse and cost 2.7× more per epoch. That negative result
is what justified spending the next sprint on post-processing instead of a bigger
model.

---

## 22. Show the research benchmark metrics

- [ ] Point out the segmentation block: macro foreground Dice 0.90001, vertebra
      0.91663, disc 0.87827, canal 0.90514
- [ ] Point out the post-processing block: disc indexing 84.08% → **93.84%**,
      height MAE 0.6492 mm, area MAE 21.4831 mm², signal ratio r = 0.9726,
      end-to-end Pfirrmann QWK 0.6543
- [ ] Point out that segmentation and post-processing are visually separate

**Say:** these are two different questions and two different measurements, which
is why they are not averaged into one score.

---

## 23. State explicitly that benchmarks are dataset-level

Do this as its own beat. Do not let it slide past.

- [ ] Read the served note aloud: these figures were measured on a 33-patient
      held-out test split; **they describe the pipeline, not this study**
- [ ] Give the concrete version:

> 93.84% means that across 6,821 discs in the held-out test split, 93.84%
> received the correct index. It does **not** mean this patient's disc was
> identified with 93.84% accuracy. For this study we have no ground truth, so we
> cannot state a per-patient accuracy at all — and we do not.

---

## 24. Open and download the report

- [ ] Click **Report** to open the report view
- [ ] Click **Download** for the Markdown file
- [ ] Open the downloaded file

---

## 25. Verify report provenance and version

In the downloaded Markdown, confirm:

- [ ] Pipeline version `SPIDER-Lumbar-v1` and a generation timestamp
- [ ] Section 3 — per-class validated test Dice
- [ ] Section 5 — quantitative measurements
- [ ] Section 6 — model-derived findings
- [ ] Section 7 — limitations, including the held-out-split statement
- [ ] Section 8 — the research disclaimer

**Say:** this document is assembled from deterministic templates filled with
measured values. No language model is involved, so the same result always
produces the same report.

---

## 26. Confirm no clinical diagnosis or postoperative prediction is claimed

Close on this.

- [ ] Point out that the report states findings are from a single timepoint and
      that no postoperative assessment is provided
- [ ] Point out the disclaimer: not a clinical diagnosis, not a medical device

**Say:** and to be explicit about a limitation people often ask about — this
system cannot predict postoperative healing. SPIDER is cross-sectional. There is
no follow-up imaging, no surgical record and no outcome label, so there is
nothing to train such a model on and nothing to validate it with. The result
carries a single `baseline` timepoint as a schema placeholder for future work,
not as a capability.

---

## Things NOT to say during the demo

Each of these is either false or unsupported by the data. If you are asked
directly, use the correction alongside it.

| Do not say | Say instead |
| --- | --- |
| "The system diagnoses the patient." | "The system produces research estimates that a radiologist must review. It does not diagnose." |
| "93.84% accuracy on this patient's disc." | "93.84% of 6,821 discs in the held-out test split got the correct index. It is a dataset-level benchmark. We have no ground truth for this study, so there is no per-patient accuracy." |
| "It predicts how well the patient will heal after surgery." | "It cannot. SPIDER has no postoperative imaging, no surgical record and no outcome labels, so no healing model could be trained or validated." |
| "It's validated longitudinally." | "There is a single timepoint per patient. The `timepoint` field is a schema seam for future work, not longitudinal validation." |
| "It reports the Modic type." | "Only binary 'any Modic change' was modelled. The nominal type 0/I/II/III was never modelled, so it is reported as unavailable with that reason." |
| "It detects spondylolisthesis." | "Not served. Five positive discs in the test split is too few to validate, so it is reported as unavailable with that reason." |
| "This disc is 87% damaged." / any composite spine-damage percentage | "No composite severity score is produced. Each finding is reported separately, as the dataset annotates it. Combining them would be an unvalidated clinical judgement." |
| "The model is 90% confident." | "No numeric confidence is displayed. Evidence strength is qualitative — strong, moderate, limited, weak — because a probability is not a validated medical confidence." |
| "Identity confidence shows how sure we are clinically." | "Identity support is the fraction of slices where this disc was tracked consistently. It describes tracking, not clinical certainty." |
| "This is disc L4–L5." | "Index 4 counting up from the most inferior disc. The dataset does not state which vertebra is L5, so no level name is asserted." |
| "The red shows the damaged tissue." / "the lesion" / "where the disease is" | "The red is the disc segmentation for a disc with a model-estimated finding. It marks which disc, not which pixels. There is no pixel-level pathology model here." |
| "The unmarked discs are healthy." | "They have no supported positive finding. That is not the same as certified normal." |
| "Red means high severity." | "Red is binary presence of a supported finding. It carries no severity or grade." |
| "It's production ready." | "It runs locally with no authentication. It must not be exposed on a network without an access-control layer." |

---

## Recovery, if something breaks mid-demo

| Symptom | Cause | Action |
| --- | --- | --- |
| Dashboard shows *Cannot reach the analysis API* | backend not running or wrong port | check Terminal 1; the error panel shows the code and a retry button |
| **Load Sample Study** button missing | `sample_study_available: false` — the volume is not on this machine | check `data/extracted/images/33_t2.mha`; upload a volume manually instead |
| Health badge reads *degraded* | checkpoint failed to load | check `MODEL_CHECKPOINT`; analyses are refused with `MODEL_UNAVAILABLE` rather than returning partial output |
| Processing view stuck | backend restarted mid-analysis | status reports `ANALYSIS_INTERRUPTED`; start the analysis again |
| Analysis takes much longer than ~3 s | CPU contention | close other applications; mention that this is CPU inference |
| Slices blank | analysis not complete | `SLICES_NOT_READY` (409) until the arrays are written |
| Frontend shows a stale page | dev-server hot reload glitch | hard refresh |

**Fallback:** keep one completed analysis from before the demo. If a live run
fails, open it from **History** and continue from step 10. Say that you are
opening an earlier real run — do not present it as a live one.
