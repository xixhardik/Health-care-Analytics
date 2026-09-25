# Longitudinal Readiness Plan

*What would be required to compare preoperative, 6-month and 9-month postoperative lumbar MRI, and what this project has already built towards it.*

---

## 1. Current status: verified, not assumed

**The dataset in use is cross-sectional — one study per patient. It cannot support any 6-month or 9-month comparison.**

This was established by direct checks against the files, not by assumption:

| Requirement for longitudinal analysis | Present? | Evidence |
| --- | --- | --- |
| Per-series acquisition date, visit number or timepoint field | **No** | All 39 columns of `overview.csv` were pattern-searched. The only match was `birth_date`, which holds values 14–84 (mean 59.6) and is an **age in years despite its name**; it is also missing for 73% of series. No `StudyDate`, `SeriesDate` or `AcquisitionDate` column exists. |
| More than one study per patient | **No** | Maximum series per (patient, modality) = 1, with 0 repeated patient+modality pairs. The 1–3 series a patient has are *different sequences from one sitting* (T1, T2, T2 SPACE), not repeat visits. |
| Treatment / surgery / outcome field | **No** | Pattern search for surgery, operation, treatment, intervention and outcome fields returned nothing. It is not recorded whether any patient was operated on. |

The published dataset description is consistent with all three: each patient contributed **one study** of up to three series, collected January 2019 – March 2022 across four hospitals. That date range is the span of the *collection campaign across different patients*, not a per-patient follow-up interval. Nothing in the description mentions follow-up, postoperative or repeat imaging ([SPIDER data page](https://spider.grand-challenge.org/data/); [van der Graaf et al., *Scientific Data* 11, 264, 2024](https://www.nature.com/articles/s41597-024-03090-w)). Content was rephrased for compliance with licensing restrictions.

> **Consequence.** Registration between timepoints, change metrics, progression or recovery modelling, and any pre-/post-operative statement are impossible to implement *or validate* here. Producing such a result from this data would require fabricating the second timepoint. No such claim appears anywhere in this project.

---

## 2. What has already been built towards it

Longitudinal readiness is a data-architecture property, and it costs nothing to establish now. Four things are already in place.

### 2.1 Stable disc identity

Every output record is keyed on `(patient_id, ivd_label)`, and `ivd_label` comes from the **mask label value**, never from position or filename order:

```
radiological_gradings.csv 'IVD label' N  <->  raw mask label 200 + N
                                         <->  Sprint 1 instance label 10 + N
```

This mapping was verified, not assumed: the graded and segmented disc sets agree exactly in **432 of 447 series (96.6%)**, and the 15 disagreeing series are enumerated in `radiological_grading_analysis.md`. A future comparison joins two timepoints on this key. Had disc identity been derived from ordering, a single missed disc at follow-up would shift every index above it and silently compare the wrong pair of discs.

### 2.2 Time fields present as schema, null as values

`outputs/reports/disc_records_longitudinal.csv` and every `finding_assessments/patient_XXX.json` carry:

| field | current value | purpose |
| --- | --- | --- |
| `record_schema_version` | `1.0` | lets a later sprint detect format change |
| `patient_id` | populated | join key |
| `series_id` | populated | join key |
| `ivd_label` | populated | join key |
| `study_id` | **null** | study-level identifier; absent in this dataset |
| `acquisition_timestamp` | **null** | absent in this dataset |
| `timepoint_label` | **null** | would hold `pre_op`, `6_month`, `9_month` |
| `timepoint_available` | `false` | explicit flag |

They are deliberately empty rather than populated with plausible-looking placeholders. A future sprint fills them in and the schema does not otherwise change, so a comparison becomes a join on `(patient_id, ivd_label)` across `timepoint_label`.

### 2.3 Measurements in physical units, not pixels

Sprint 1 resampled every slice to a fixed **1.0 mm/pixel**, and Stage C reports every measurement in millimetres and mm². This is what makes two scans comparable: the same disc imaged on a different scanner with different pixel spacing still yields a height in millimetres. Storing pixel counts instead would make any cross-timepoint difference a function of acquisition settings as much as of anatomy.

### 2.4 Deterministic, reproducible preprocessing

The whole pipeline is scripted with a fixed seed and reads the raw data as read-only. Two timepoints processed through it receive **identical** treatment — same reorientation to RAS, same resampling, same normalisation, same crop. Any difference between two processed scans is therefore attributable to the anatomy rather than to pipeline variation.

---

## 3. Additional data required

| # | Requirement | Detail | Why it is necessary |
| --- | --- | --- | --- |
| 1 | **Repeat MRI studies** | ≥2, ideally 3 per patient: preoperative, ~6 months, ~9 months. Same region, same sequences available at each timepoint. | Without a second observation of the same disc there is no change to measure. |
| 2 | **Timepoint identification** | Acquisition date per study **and** the surgery date, so the interval can be computed in days rather than assumed from a label. | "6-month" scans are rarely at exactly 6 months; using the nominal label as the interval introduces error into any rate-of-change estimate. |
| 3 | **Surgical record** | Procedure type, the vertebral/disc **levels operated on**, and the date. | Without the operated level, a per-disc change cannot be attributed to the intervention. An adjacent unoperated level is the natural internal control, and identifying it requires this field. |
| 4 | **Gradings at every timepoint** | The same eight findings re-graded at each follow-up under the same protocol. Ideally blinded to timepoint order, with inter-reader agreement reported. | Measuring *change* is far more sensitive to reader variability than measuring a single state: two readers who agree on grade 3 vs 4 at one visit can disagree on whether a disc changed. Without a reported agreement figure there is no way to tell a real change from reader noise. |
| 5 | **Clinical outcome measures** | Validated instruments at each timepoint — e.g. a disability index and a pain scale. | If "recovery" is to mean patient benefit rather than image appearance, it must be measured on the patient. Imaging change alone does not establish recovery. |
| 6 | **Protocol consistency** | Comparable acquisition across timepoints, or a documented harmonisation strategy. | The current dataset already spans four hospitals with in-plane pixel spacing varying 0.077–1.233 mm. Uncontrolled protocol drift *between* timepoints would confound measured change with scanner change. |
| 7 | **Governance** | Ethics approval and consent covering linkage of repeat studies and surgical records for the same individual. | Longitudinal linkage is a materially different privacy proposition from a single de-identified cross-sectional study. |

---

## 4. What the comparison would measure

Change should be quantified on **continuous measurements** first and categorical findings second. The reason is resolution, and it is worth stating plainly.

### 4.1 Continuous measurements — the sensitive signal

All of these are already computed per disc by Stage C and are directly differenceable:

| Measurement | Unit | Why it is informative |
| --- | --- | --- |
| `height_mm_central` | mm | Primary disc-height measure, taken across the middle 50% of the AP width so it excludes the tapering tips |
| `height_mm_anterior`, `height_mm_posterior` | mm | Separates uniform height loss from wedging |
| `height_ap_asymmetry` | ratio | Change in disc wedging |
| `area_mm2` | mm² | Cross-sectional disc area |
| `ap_extent_mm` | mm | Antero-posterior disc depth |
| `disc_to_vertebra_height_ratio` | ratio | Height normalised by the patient's own vertebra, removing patient-scale and scanner effects |
| `height_ratio_to_series_median` | ratio | Height relative to the patient's other discs — an internal control |
| `vertebral_ap_offset_mm` | mm | Vertebral slip; relevant to alignment change after instrumentation |
| `canal_width_at_disc_mm` | mm | Canal calibre at that level; relevant after decompression |
| `intensity_disc_vertebra_ratio` | ratio | Disc signal against a within-slice reference, so it survives scanner change better than absolute signal |

### 4.2 Categorical findings — coarse, and honest about it

The eight graded findings are **six binary flags, one nominal category and one 5-point ordinal grade**. There are no continuous measurements in the grading file: no disc height in mm, no slip distance, no signal-intensity ratio.

A 5-point scale has limited resolution over a 6–9 month interval: a real but modest change may not cross a grade boundary, so a categorical comparison will under-report change. This is a measurement-resolution point, not a clinical claim. It is the main reason §4.1 comes first.

### 4.3 Comparison design

```
for each (patient_id, ivd_label) present at two timepoints:
    delta_height_mm   = height_mm_central[t2]  - height_mm_central[t1]
    delta_area_mm2    = area_mm2[t2]           - area_mm2[t1]
    delta_pfirrmann   = pfirrmann_grade[t2]    - pfirrmann_grade[t1]
    finding_transition= (finding[t1], finding[t2])     # e.g. (1, 0) = resolved
    interval_days     = acquisition_date[t2] - acquisition_date[t1]
```

Three controls that the design needs:

1. **Unoperated adjacent levels as an internal control.** Change at an operated level is only interpretable against the change seen at levels that were not operated on.
2. **A measurement repeatability estimate.** Before any difference is called real, the pipeline's own test–retest variability must be known — ideally from same-session repeat scans, or at minimum from the cross-slice standard deviations Stage C already records (`*_across_slices_std`). A change smaller than measurement noise is not a change.
3. **Segmentation error propagation.** The measurements come from a segmentation. Two timepoints segmented independently will differ partly because of model variability. This must be quantified, and the Stage B disc-identification audit is the starting point for it.

---

## 5. What must not be claimed

- **No "healing percentage" or "recovery score"** unless a validated clinical definition is supplied, with the mapping from measurements to that definition specified and validated externally. Inventing a percentage from disc height change would produce a number with no defensible meaning.
- **No composite severity or damage score.** The source data grades eight findings independently; collapsing them into one number is a clinical judgement, not a data-processing step.
- **No causal attribution** of an observed change to the surgery without the operated-level record (§3 item 3) and an unoperated control level.
- **No claim that imaging change equals clinical improvement.** These are different endpoints and require the outcome measures in §3 item 5.
- **No diagnosis.** The reports are titled "Radiological Finding Assessment" / "Automated MRI Finding Report" and carry an explicit non-diagnostic disclaimer.

---

## 6. Summary

| Item | Status |
| --- | --- |
| Stable disc identity across scans | **Ready** — keyed on verified mask label values |
| Physical-unit measurements | **Ready** — millimetres at a fixed 1.0 mm/px |
| Deterministic reproducible preprocessing | **Ready** — scripted, seeded, raw data read-only |
| Schema fields for time and timepoint | **Ready** — present and explicitly null |
| Continuous change metrics defined | **Ready** — §4.1, computed per disc already |
| Second/third timepoint imaging | **Missing** — required, §3 item 1 |
| Acquisition and surgery dates | **Missing** — required, §3 items 2–3 |
| Follow-up gradings and outcome scores | **Missing** — required, §3 items 4–5 |
| Validated definition of "recovery" | **Missing** — required before any such metric |

The pipeline is structured so that acquiring the data in §3 is the only blocker. No code change to Sprint 1 or Sprint 2 is needed to ingest a second timepoint: the schema, the keys and the physical units are already in place.
