# Radiological Grading Analysis

*data/raw/radiological_gradings.csv - scoping the extended project goal (segmentation + disc-level degeneration assessment)*

Generated: 2026-09-22 22:06:57

**Scope of this document.** It inspects the grading file, establishes how it links to the MRI series and segmentation masks, and uses that to propose a Sprint 2 architecture. No model is trained here, and **no severity score is defined** - the dataset ships eight independently graded findings, and combining them into one number is a clinical decision rather than a data-processing one.


## 1. Headline figures

| Field | Value |
| --- | --- |
| File | `data/raw/radiological_gradings.csv` |
| Rows (raw) | 1520 |
| Columns | 10 |
| Patients | 218 |
| Valid IVD records (`IVD label` >= 1) | 1518 |
| Invalid IVD records (`IVD label` = 0) | 2 |
| Missing values (entire file) | 0 |
| Duplicate rows | 0 |
| Duplicate (Patient, IVD label) keys | 0 |
| Graded discs per patient | 5-9 (median 7) |

The file is **complete**: not a single missing value, no duplicate rows and no duplicate patient/disc keys. Its granularity is **one row per (patient, intervertebral disc)**.


## 2. All columns

| # | column | dtype | kind | values present | missing | unexpected |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `Patient` | int64 | identifier | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35... | 0 | none |
| 1 | `IVD label` | int64 | identifier | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | 0 | none |
| 2 | `Modic` | int64 | nominal | 0, 1, 2, 3 | 0 | none |
| 3 | `UP endplate` | int64 | binary | 0, 1 | 0 | none |
| 4 | `LOW endplate` | int64 | binary | 0, 1 | 0 | none |
| 5 | `Spondylolisthesis` | int64 | binary | 0, 1 | 0 | none |
| 6 | `Disc herniation` | int64 | binary | 0, 1 | 0 | none |
| 7 | `Disc narrowing` | int64 | binary | 0, 1 | 0 | none |
| 8 | `Disc bulging` | int64 | binary | 0, 1 | 0 | none |
| 9 | `Pfirrman grade` | int64 | ordinal | 1, 2, 3, 4, 5 | 0 | none |

All ten columns are integer-typed. Two are identifiers and eight are graded findings. No undocumented value appears in any finding column.


### Meaning of each finding

| column | kind | domain | meaning |
| --- | --- | --- | --- |
| `Modic` | nominal | 0, 1, 2, 3 | Modic change - vertebral endplate/bone-marrow signal change. 0 = none; 1, 2, 3 = Modic type I, II, III. The types describe different marrow states (oedema, fatty, sclerotic), so they are categories rather than an ordered severity scale. |
| `UP endplate` | binary | 0, 1 | Endplate defect / Schmorl's node at the UPPER endplate of the disc space. 0 = absent, 1 = present. |
| `LOW endplate` | binary | 0, 1 | Endplate defect / Schmorl's node at the LOWER endplate of the disc space. 0 = absent, 1 = present. |
| `Spondylolisthesis` | binary | 0, 1 | Forward slippage of one vertebra relative to the one below it. 0 = absent, 1 = present. |
| `Disc herniation` | binary | 0, 1 | Focal displacement of disc material beyond the disc space (protrusion/extrusion). 0 = absent, 1 = present. |
| `Disc narrowing` | binary | 0, 1 | Reduced disc height. 0 = absent, 1 = present. |
| `Disc bulging` | binary | 0, 1 | Generalised extension of disc material beyond the endplate margin, circumferential rather than focal. 0 = absent, 1 = present. |
| `Pfirrman grade` | ordinal | 1, 2, 3, 4, 5 | Pfirrmann grade of disc degeneration on T2, from 1 (normal, bright homogeneous nucleus, full height) to 5 (collapsed disc space, dark nucleus). An ordered severity scale for the disc itself. |

The distinction between *kinds* is not cosmetic and constrains the modelling:

- **`Pfirrman grade` is ordinal** (1 < 2 < ... < 5). It must be modelled in a way that respects the ordering; treating it as 5 unordered classes throws away the fact that confusing grade 4 with 5 is a smaller error than confusing 1 with 5.
- **`Modic` is nominal.** Types I, II and III describe different marrow states (oedema, fatty, sclerotic), not increasing severity, so it must **not** be treated as an ordered scale.
- **The remaining six are binary** presence/absence flags.
- `UP endplate` / `LOW endplate` are the endplate-defect / Schmorl's-node labels, recorded separately for the upper and lower endplate bounding each disc space.

The clinical naming follows the published dataset description ([SPIDER data page](https://spider.grand-challenge.org/data/); [van der Graaf et al., *Scientific Data* 11, 264, 2024](https://www.nature.com/articles/s41597-024-03090-w)); the value domains above were measured from the file. Content was rephrased for compliance with licensing restrictions.


## 3. Distribution of every finding

All percentages are over the **1,518 valid disc records**. See `outputs/visualizations/grading_finding_distributions.png`.


### 3.1 Pfirrmann grade (ordinal, 1-5)

| grade | label | records | share |
| --- | --- | --- | --- |
| 1 | grade 1 (normal) | 284 | 18.71% |
| 2 | grade 2 | 341 | 22.46% |
| 3 | grade 3 | 418 | 27.54% |
| 4 | grade 4 | 291 | 19.17% |
| 5 | grade 5 (most degenerate) | 184 | 12.12% |

| Field | Value |
| --- | --- |
| Grades present | 1, 2, 3, 4, 5 |
| Mean grade | 2.835 |
| Median grade | 3 |


### 3.2 Modic category (nominal, 0-3)

| value | category | records | share |
| --- | --- | --- | --- |
| 0 | none | 1,004 | 66.14% |
| 1 | type I | 4 | 0.26% |
| 2 | type II | 503 | 33.14% |
| 3 | type III | 7 | 0.46% |


### 3.3 Binary findings

| finding | absent (0) | present (1) | prevalence |
| --- | --- | --- | --- |
| `UP endplate` | 904 | 614 | 40.45% |
| `LOW endplate` | 895 | 623 | 41.04% |
| `Spondylolisthesis` | 1,476 | 42 | 2.77% |
| `Disc herniation` | 1,446 | 72 | 4.74% |
| `Disc narrowing` | 974 | 544 | 35.84% |
| `Disc bulging` | 772 | 746 | 49.14% |


### 3.4 How many findings occur on one disc

| findings on the disc | disc records | share |
| --- | --- | --- |
| 0 | 479 | 31.55% |
| 1 | 238 | 15.68% |
| 2 | 215 | 14.16% |
| 3 | 160 | 10.54% |
| 4 | 160 | 10.54% |
| 5 | 233 | 15.35% |
| 6 | 29 | 1.91% |
| 7 | 4 | 0.26% |

| Field | Value |
| --- | --- |
| Mean findings per disc | 2.078 |
| Discs with no finding at all | 31.55% |
| Counting rule | Counts the 6 binary findings plus 'any Modic change'. Pfirrmann is excluded because every disc always has a grade. |

**This makes the task multi-label, not multi-class.** A disc routinely carries several findings simultaneously, so Sprint 2 cannot treat the assessment as picking one class out of a set.


## 4. Findings across disc levels

`IVD label` is anatomically ordered: 1 is the most inferior disc, counting upward. Prevalence per level is therefore a sanity check on the labels as well as useful context.

| ivd_label | n_records | modic | upper_endplate_defect | lower_endplate_defect | spondylolisthesis | disc_herniation | disc_narrowing | disc_bulging | pfirrmann_grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 218 | 39.91 | 32.57 | 36.24 | 8.72 | 9.63 | 57.8 | 69.27 | 3.427 |
| 2 | 218 | 48.62 | 42.66 | 47.71 | 5.05 | 15.6 | 59.17 | 74.77 | 3.317 |
| 3 | 218 | 40.37 | 47.25 | 45.87 | 1.83 | 4.59 | 44.95 | 68.81 | 2.972 |
| 4 | 218 | 37.61 | 45.41 | 44.95 | 1.38 | 1.83 | 36.24 | 60.09 | 2.803 |
| 5 | 218 | 32.11 | 46.33 | 44.04 | 1.38 | 0.46 | 22.94 | 38.07 | 2.583 |
| 6 | 217 | 19.35 | 33.64 | 32.72 | 0.92 | 0.46 | 13.36 | 16.59 | 2.286 |
| 7 | 144 | 18.75 | 38.89 | 38.89 | 0 | 0.69 | 17.36 | 16.67 | 2.403 |
| 8 | 55 | 14.55 | 27.27 | 27.27 | 0 | 0 | 12.73 | 10.91 | 2.564 |
| 9 | 12 | 33.33 | 25 | 33.33 | 0 | 0 | 8.33 | 16.67 | 2.417 |

Two things to take from this table. First, the record count collapses at higher levels - disc 1 through 6 are present for essentially every patient, while the upper levels appear in only a fraction of them. Any per-level metric in Sprint 2 will therefore be far less reliable for the upper discs. Second, the degeneration measures concentrate in the lower lumbar discs, which is the expected anatomical pattern and indicates the labels behave sensibly. Figure: `outputs/visualizations/grading_prevalence_by_disc_level.png`.


### 4.1 Pfirrmann grade against Modic category

| grade | Modic none | Modic type I | Modic type II | Modic type III |
| --- | --- | --- | --- | --- |
| Pfirrmann 1 | 265 | 0 | 18 | 1 |
| Pfirrmann 2 | 301 | 0 | 40 | 0 |
| Pfirrmann 3 | 275 | 0 | 140 | 3 |
| Pfirrmann 4 | 140 | 2 | 148 | 1 |
| Pfirrmann 5 | 23 | 2 | 157 | 2 |

Read this as context, not as a severity mapping: Modic categories are nominal, so a higher Modic value in a row does not mean a worse disc.


### 4.2 Co-occurrence between findings

Spearman correlation (the findings are ordinal/binary, so Pearson would be inappropriate). Descriptive only - it informs whether Sprint 2 should predict the findings jointly. Figure: `outputs/visualizations/grading_cooccurrence.png`.

| finding | modic | upper_endplate_defect | lower_endplate_defect | spondylolisthesis | disc_herniation | disc_narrowing | disc_bulging | pfirrmann_grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modic | 1 | 0.475 | 0.496 | 0.136 | 0.152 | 0.443 | 0.387 | 0.515 |
| upper_endplate_defect | 0.475 | 1 | 0.854 | 0.09 | 0.081 | 0.369 | 0.342 | 0.422 |
| lower_endplate_defect | 0.496 | 0.854 | 1 | 0.096 | 0.104 | 0.415 | 0.359 | 0.451 |
| spondylolisthesis | 0.136 | 0.09 | 0.096 | 1 | 0.17 | 0.201 | 0.107 | 0.176 |
| disc_herniation | 0.152 | 0.081 | 0.104 | 0.17 | 1 | 0.234 | -0.021 | 0.203 |
| disc_narrowing | 0.443 | 0.369 | 0.415 | 0.201 | 0.234 | 1 | 0.513 | 0.619 |
| disc_bulging | 0.387 | 0.342 | 0.359 | 0.107 | -0.021 | 0.513 | 1 | 0.559 |
| pfirrmann_grade | 0.515 | 0.422 | 0.451 | 0.176 | 0.203 | 0.619 | 0.559 | 1 |


## 5. Relationship between Patient, IVD label and MRI series

This is the part that determines what is actually trainable.

- **`Patient` joins to the MRI filenames.** Filenames are `<patient_id>_<modality>.mha`, and `Patient` matches the numeric prefix. All 218 patients in the grading file have imaging, and every imaged patient is graded.
- **`IVD label` N corresponds to mask label 200 + N.** Masks number the lowest IVD 201 upward, and the grading file numbers the lowest disc 1 upward. Verified: the two label sets agree exactly in 432/447 series (96.64%).
- **Gradings are per patient, not per series.** There is exactly one grading row per (patient, disc) regardless of how many MRI series that patient has. A patient has 1-3 series (T1, T2, sometimes 3D T2 SPACE) of the same anatomy, and they all share one set of gradings.


### 5.1 Label replication across series

| Field | Value |
| --- | --- |
| Patients | 218 |
| MRI series | 447 |
| Unique grading records | 1518 |
| Disc-series instances if gradings are attached to every series | 3143 |
| Replication factor | 2.07 |
| Series per patient | `1`=28, `2`=151, `3`=39 |

Attaching gradings to series inflates 1,518 real labels into 3,143 training instances - a factor of 2.07. Those extra instances are **not** extra information. The effective sample size for the grading task is **218 patients / 1,518 discs**, and this is the reason the Sprint 1 patient-level split must be carried over unchanged.


### 5.2 Where gradings and segmentations disagree

| Field | Value |
| --- | --- |
| Series with exact agreement | 432 / 447 (96.64%) |
| Series with a mismatch | 15 |
| Patients affected | 8 |
| Segmented disc instances across all series | 3,147 |
| Segmented disc instances with a matching grading | 3,135 |
| Graded discs never segmented in any series of that patient | 0 |
| Segmented discs with no grading for that patient | 5 |

The mismatching series in full:

| series | patient | graded discs | segmented discs | segmented but not graded | graded but not segmented |
| --- | --- | --- | --- | --- | --- |
| `35_t2` | 35 | 9 | 3 | - | 4, 5, 6, 7, 8, 9 |
| `55_t1` | 55 | 7 | 8 | 8 | - |
| `55_t2_SPACE` | 55 | 7 | 8 | 8 | - |
| `57_t1` | 57 | 7 | 8 | 8 | - |
| `57_t2_SPACE` | 57 | 7 | 8 | 8 | - |
| `64_t2` | 64 | 7 | 6 | - | 7 |
| `98_t1` | 98 | 7 | 8 | 8 | - |
| `98_t2_SPACE` | 98 | 7 | 8 | 8 | - |
| `107_t1` | 107 | 6 | 7 | 7 | - |
| `107_t2` | 107 | 6 | 7 | 7 | - |
| `107_t2_SPACE` | 107 | 6 | 7 | 7 | - |
| `118_t2_SPACE` | 118 | 8 | 7 | - | 8 |
| `152_t1` | 152 | 8 | 9 | 9 | - |
| `152_t2` | 152 | 8 | 9 | 9 | - |
| `152_t2_SPACE` | 152 | 8 | 9 | 9 | - |

These are the discs that **cannot be used as disc-level training targets without a decision**: either there is a segmented region with no label, or a label with no region to attach it to. The practical handling is to train on the intersection and record the excluded discs, rather than silently mis-aligning labels by index. Per-series and per-patient detail: `outputs/reports/grading_series_linkage.csv` and `grading_patient_linkage.csv`.


## 6. Data quality problems found

- No missing values anywhere in the file; no duplicate rows; no duplicate (Patient, IVD label) keys.
- **2 rows carry `IVD label` = 0** (patients 107, 152). Zero is not a valid disc index - the mask label space reserves 0 for background and numbers IVDs from 201 upward - so these rows cannot be attached to any segmented structure. In both cases the patient also has a contiguous 1..N run, and counting the 0 row makes the patient's disc total match `num_discs` in `overview.csv`, which is consistent with a corrupted label value rather than an extra disc. They are excluded from every statistic in this report and should be excluded from training; recovering the intended index would require re-annotation.
- Disc indices form a contiguous 1..N run for every patient (after excluding the `IVD label` = 0 rows).
- 15 series (8 patients) disagree between the graded disc set and the segmented disc set - see section 5.2.
- Severe class imbalance in the rarest categories. Pfirrmann grades 1 and 5 and Modic type III are each a small fraction of records, and spondylolisthesis is the rarest binary finding. This caps what Sprint 2 can credibly report per class.
- The grading file carries **no laterality, no disc sub-region and no measurement values** (no disc height in mm, no slip distance, no signal intensity). Every finding is a per-disc categorical judgement, so the model output can only be as granular as that.


### 6.1 Class balance inside the Sprint 1 split

The existing patient-level split partitions the grading labels too, since gradings are keyed on patient. Counts per split:

| split | patients | disc_records | pfirrmann_1 | pfirrmann_2 | pfirrmann_3 | pfirrmann_4 | pfirrmann_5 | modic_0 | modic_1 | modic_2 | modic_3 | upper_endplate_defect_pos | lower_endplate_defect_pos | spondylolisthesis_pos | disc_herniation_pos | disc_narrowing_pos | disc_bulging_pos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 152 | 1063 | 222 | 228 | 287 | 192 | 134 | 696 | 3 | 359 | 5 | 433 | 428 | 30 | 43 | 373 | 515 |
| val | 33 | 228 | 21 | 51 | 73 | 56 | 27 | 146 | 1 | 80 | 1 | 105 | 107 | 7 | 11 | 98 | 130 |
| test | 33 | 227 | 41 | 62 | 58 | 43 | 23 | 162 | 0 | 64 | 1 | 76 | 88 | 5 | 18 | 73 | 101 |

Figure: `outputs/visualizations/grading_split_class_balance.png`. Note how thin the rare classes become: a class with only a handful of test records cannot support a trustworthy per-class score, so Sprint 2 should report those with explicit uncertainty or group them.


## 7. Sequence availability and what it constrains

| Field | Value |
| --- | --- |
| Series by modality | `t2`=210, `t1`=196, `t2_SPACE`=41 |
| Patients with at least one T2 / T2 SPACE series | 212 / 218 |
| Patients with only T1 | 6 ([72, 77, 126, 132, 180, 219]) |
| Disc records on patients that have T2 | 1,474 / 1,518 |
| Disc records lost if a target is restricted to T2 | 44 |

**This splits the eight findings into two groups, and the split is not arbitrary.** The Pfirrmann grade is defined on T2-weighted images: it is read from nucleus signal brightness and disc height on T2. Asking a model to predict it from a T1 series is not a well-posed problem, because the signal characteristic the grade is built on is not the one T1 shows. A Pfirrmann head should therefore be trained and evaluated on T2 / T2 SPACE series only, which still covers 1,474 of 1,518 disc records.

The other findings - endplate defects, spondylolisthesis, herniation, narrowing, bulging - are morphological rather than signal-based, so they are not restricted in the same way and can use all series. Modic changes sit in between: they are marrow *signal* changes whose type is conventionally distinguished using T1 together with T2, which is a further reason the Modic *type* is not a safe target here (see section 9).


## 8. Longitudinal / postoperative analysis: what is not possible

The revised goal includes comparing preoperative scans with 6-month and 9-month postoperative scans. **This dataset cannot support that.** That is a verified finding, not an assumption - three independent checks were run against the actual files, and the published dataset description was consulted.


### 8.1 Evidence

| requirement for longitudinal analysis | present? | evidence |
| --- | --- | --- |
| A per-series acquisition date, visit number or timepoint field | **No** | All 39 columns of `overview.csv` were pattern-searched for date/time/visit/follow-up fields. The only match is `birth_date`, which holds values 14-84 (mean 59.6) and is an age in years despite its name - not an acquisition date. It is also missing for 73% of series. There is no StudyDate, SeriesDate or AcquisitionDate column at all. |
| More than one study per patient | **No** | No patient has more than one series of the same modality (max = 1, 0 repeated patient+modality pairs). The 1-3 series a patient has are *different sequences of one sitting* (T1, T2, T2 SPACE), not repeat visits. |
| A treatment / surgery / outcome field | **No** | Pattern search for surgery, operation, treatment, intervention and outcome fields returned nothing. Neither CSV records whether a patient was operated on, at which level, when, or with what result. |

The published dataset description is consistent with all three findings: it states that studies were gathered from patients with a history of low back pain, that **each study consisted of up to three MRI series**, and that collection ran from January 2019 to March 2022 across four hospitals. That date range is the span of the *collection campaign* across different patients - it is not a per-patient follow-up interval. Nothing in the description mentions follow-up, postoperative or repeat imaging ([SPIDER data page](https://spider.grand-challenge.org/data/)). Content was rephrased for compliance with licensing restrictions.

**Conclusion: the dataset is cross-sectional - exactly one timepoint per patient.** Any 6-month or 9-month comparison, recovery trajectory, or pre-/post-operative change measurement is impossible to implement *or validate* with it. Reporting such a result from this data would require fabricating the second timepoint.


### 8.2 Specifically, what cannot be built

- **Inter-timepoint registration** - there is no second scan to register to.
- **Change / delta metrics** (disc height loss, grade progression, herniation resolution) - a delta needs two observations of the same disc at different times; the dataset has one.
- **Recovery or progression modelling** - no outcome variable and no time axis exist.
- **Postoperative assessment** - it is not even recorded which patients had surgery, so a pre-/post-operative label cannot be assigned.
- **Validation of any of the above** - even if such a module were written, there is no ground truth in this dataset to test it against.


### 8.3 Additional data that would be required

| requirement | detail |
| --- | --- |
| Repeat MRI studies | At least 2, ideally 3 studies per patient (preoperative, ~6 months, ~9 months) of the same region, with the same sequences available at each timepoint. |
| Timepoint identification | An acquisition date or visit label per study, plus the surgery date, so an interval in days can be computed rather than assumed. |
| Surgical record | Procedure type, the vertebral / disc levels operated on, and the date. Without the operated level, a per-disc change cannot be attributed to the intervention. |
| Gradings at every timepoint | The same eight findings re-graded at each follow-up under the same protocol. Ideally blinded and with inter-reader agreement reported, because measuring change is far more sensitive to reader variability than measuring a single state. |
| Clinical outcome measures | If 'recovery' is to mean patient benefit rather than image appearance, validated scores such as a disability index or pain scale are needed at each timepoint. Imaging change alone does not establish recovery. |
| Protocol consistency | Comparable acquisition across timepoints, or a documented harmonisation strategy. This dataset already spans four hospitals with pixel spacing varying by ~16x; adding uncontrolled protocol drift between timepoints would confound any measured change. |
| Governance | Ethics approval and consent covering linkage of repeat studies and surgical records for the same individual. |


### 8.4 A measurement-resolution caveat worth raising early

Even with ideal longitudinal data, the eight findings available here are **coarse categorical judgements** - six binary flags, one nominal category and one 5-point ordinal grade. There are no continuous measurements in the file: no disc height in millimetres, no slip distance, no signal-intensity ratio. A 5-point scale has limited resolution for detecting change over a 6-9 month interval, since a real but small change may not cross a grade boundary. If quantifying change is a project goal, the pipeline should also output **continuous geometric measurements in millimetres** (disc height, disc height relative to neighbours, anterior-posterior extent, inter-vertebral offset), which are derivable from the segmentation masks and are far more sensitive to change than a categorical grade. Those measurements are *computable now* from Sprint 1 output; only their validation against follow-up ground truth has to wait.


## 9. Proposed Sprint 2 architecture

The design below follows from the measurements in this report rather than from a generic template. **Sprint 1 preprocessing is reused unchanged**: the same 352 x 256 slices at 1.0 mm/px, the same label spaces, and the same patient-level split with the same seed.


### 9.1 The three constraints that drive the design

- **The grading task is small.** 1,518 disc records from 218 patients. Attaching them to series inflates this to 3,143 instances (factor 2.07), but that is replication, not information. This is orders of magnitude too little to train a large classifier from scratch, and it rules out an end-to-end jointly-trained segmentation + grading network.
- **The labels are per-disc, not per-slice.** A grading describes a disc as a 3-D structure. Assigning it to every 2-D slice of that disc would create label noise, because a focal finding such as a herniation is visible on only some slices. So the unit of prediction must be a **disc instance**, not a slice.
- **The task is multi-label with very unequal support.** A disc carries 2.08 findings on average and up to 7; meanwhile spondylolisthesis occurs in 2.8% of discs and herniation in 4.7%.


### 9.2 Staged architecture

```text

STAGE A - Segmentation                                    [Sprint 1 output, unchanged]
  2-D U-Net, input 352x256x1, output 4 semantic classes
  (background / vertebra / IVD / spinal canal)
  Also predicts the instance label space already saved in Sprint 1
  -> trained on 12,415 preprocessed slices, 152 training patients
  -> metrics: Dice, IoU per class

STAGE B - Disc instance extraction                        [geometric, no learning]
  From the instance mask, isolate each disc (index 1..9) and the two
  vertebrae bounding it
  For each disc instance:
    - disc-centred ROI, taken as a 2.5-D stack (mid-disc slice +/- k neighbours)
      so a focal finding is not missed, and including the adjacent vertebral
      bodies because endplate defects, Modic change and spondylolisthesis live
      in or between the vertebrae, not in the disc
    - geometric descriptors in millimetres, from the mask and the known
      1.0 mm/px scale:
          disc height (mean / min / anterior / posterior)
          disc height relative to the two neighbouring discs
          anterior-posterior extent, posterior disc margin position
          antero-posterior offset between the bounding vertebrae  (slip)
          canal cross-section at the disc level
  -> output: one feature record + one image ROI per disc instance

STAGE C - Per-disc multi-task prediction                  [the new model]
  Shared encoder over the ROI (small CNN or a pretrained backbone,
  fine-tuned; the Stage A encoder is a reasonable initialisation)
  Concatenate the Stage B geometric descriptors before the heads
  Heads, matched to each target's measurement kind:
    - Pfirrmann      ordinal (cumulative-link / CORAL), T2 series only
    - Modic          binary "any Modic change"        (see 9.4)
    - 6 x binary     endplate x2, spondylolisthesis, herniation,
                     narrowing, bulging
  Loss = weighted sum of per-head losses; class-weighted or focal loss for the
  rare binary findings

STAGE D - Per-disc report assembly                        [deterministic, no scoring]
  One row per disc: level index, each finding with its predicted class and
  calibrated probability, plus the Stage B measurements in millimetres
  Findings are reported SEPARATELY. No composite severity number is produced.
  Low-confidence and low-support cases (e.g. upper disc levels) are flagged.

STAGE E - Evaluation                                      [held-out test patients]
  Segmentation : Dice, IoU per class
  Binary       : PR-AUC (primary, because of imbalance), ROC-AUC,
                 sensitivity/specificity at a validation-chosen threshold
  Pfirrmann    : quadratic weighted kappa (primary), MAE,
                 exact and within-1-grade agreement
  Modic        : balanced accuracy, F1
  All aggregated at PATIENT level, plus per-disc-level breakdown where the
  record count supports it

```


### 9.3 Why two stages rather than end-to-end

With 1,518 disc labels, an end-to-end model would have to learn segmentation and grading simultaneously from the grading signal, which is the scarcer of the two by a wide margin (12,415 segmentation slices vs 1,518 grading records). Keeping the stages separate means the segmentation model uses all of its supervision, the grading model starts from an anatomically normalised ROI instead of a whole slice, and the two error sources stay separable at evaluation time. The last point is practical: Stage C should be evaluated **twice** - once with ROIs taken from ground-truth masks and once from predicted masks - so that a drop in grading accuracy can be attributed to either segmentation error or classification error rather than being confounded.


### 9.4 Targets that must be reduced or dropped

| target | problem | recommendation |
| --- | --- | --- |
| Modic **type** (I / II / III) | Type I has only 4 records and type III only 7, out of 1,518. Across the patient-level split that is type I: train 3 / val 1 / test 0; type III: train 5 / val 1 / test 1. **Type I has no test records at all**, so a per-type metric is not merely unreliable, it is undefined. | Collapse to **binary 'any Modic change'** (1,004 absent vs 514 present, a workable balance). Do not report per-type metrics. |
| Spondylolisthesis | 42 positives overall, split train 30 / val 7 / test 5. | Keep, because the Stage B inter-vertebral offset measurement is directly informative, but report PR-AUC with a confidence interval and state the positive count alongside every metric. |
| Disc herniation | 72 positives (4.7%), split train 43 / val 11 / test 18. | Keep with class weighting; expect wide confidence intervals and do not present a single point estimate as definitive. |
| Pfirrmann on T1 series | The grade is defined on T2 signal; T1 does not show it. | Restrict the Pfirrmann head to T2 / T2 SPACE (1,474 disc records; 6 T1-only patients excluded from this target only). |
| Upper disc levels (index 8, 9) | Very few records exist at these levels. | Train on them, but do not report per-level metrics where the count is negligible; fold them into an 'upper levels' group. |


### 9.5 Data-handling rules Sprint 2 must follow

- **Keep the Sprint 1 patient-level split, same seed.** Gradings are keyed on patient, so a series- or slice-level split would put the same label in train and test.
- **Exclude the 15 series where the graded and segmented disc sets disagree**, or restrict them to the intersecting discs. Never align by position/index order - that is exactly how labels get silently attached to the wrong disc.
- **Exclude the 2 rows with `IVD label` = 0.** They cannot be mapped to a structure.
- **Count each label once per patient in metrics.** Using a patient's 2-3 series as extra training samples is legitimate augmentation, but averaging metrics over series would weight multi-series patients 2-3x and overstate agreement.
- **Calibrate probabilities** on the validation split (e.g. temperature scaling) before they appear in a per-disc report, since an uncalibrated probability presented next to a clinical finding is misleading.
- **Persist the disc identity key `(patient_id, ivd_label)`** in every output record. This is what a future longitudinal module would join on, so building it in now costs nothing and makes the follow-up extension possible later.


### 9.6 What Sprint 2 will not claim

- No composite severity score. The dataset grades eight findings independently; collapsing them into one number would be a clinical judgement that this data does not license.
- No 6-month or 9-month comparison, and no recovery or progression statement - see section 8.
- No diagnostic claim. The output is an estimate of radiological findings for review, not a diagnosis.
- No metric without its supporting count. Every per-class figure is reported next to the number of test records behind it.


## 10. Figures produced

- `outputs/visualizations/grading_finding_distributions.png`
- `outputs/visualizations/grading_prevalence_by_disc_level.png`
- `outputs/visualizations/grading_cooccurrence.png`
- `outputs/visualizations/grading_findings_per_disc.png`
- `outputs/visualizations/grading_split_class_balance.png`


## 11. Companion data files

- `outputs/reports/radiological_grading_analysis.json` - every number above, machine-readable
- `outputs/reports/grading_disc_level_prevalence.csv` - prevalence per disc level
- `outputs/reports/grading_series_linkage.csv` - grading/mask agreement per series
- `outputs/reports/grading_patient_linkage.csv` - the same per patient

