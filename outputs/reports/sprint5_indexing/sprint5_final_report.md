# Sprint 5 Indexing - Final Report

*Disc identity from post-processing, with the segmentation model frozen*

Generated: 2026-09-24 20:05:09


## 1. Objective

Improve disc identification by post-processing the existing predictions, without retraining any segmentation model.

Sprints 3 and 4 both tested the network itself - more data coverage, then more capacity - and neither moved the plateau. What both sprints did establish is where the loss actually sits: the model finds the disc region on 96.66% of ground-truth discs but assigns the correct integer index on only 84.08%. That gap is an *identity* problem on regions that were already detected correctly, so it is addressable without touching the network.

The pipeline under test is: segmentation prediction -> disc extraction -> disc ordering/indexing -> disc-level measurements -> Pfirrmann/finding analysis. Only the ordering/indexing stage changes.

**Integer disc indices only.** The dataset does not state which vertebra is L5, so no anatomical level name (L1/L2/.../S1) is produced or asserted anywhere in this sprint.


## 2. Sprint 3 Baseline

Sprint 3 remains the strongest segmentation model and is the input to everything here. Its predictions are read frozen.

| Field | Value |
| --- | --- |
| Segmentation model | Sprint 3 Coverage, 16-channel U-Net, 1,963,860 parameters |
| Checkpoint | `outputs/checkpoints/sprint3_coverage/best_val_dice.pt` |
| Test macro foreground Dice | 0.90001 |
| Test vertebra Dice | 0.91663 |
| Test IVD Dice | 0.87827 |
| Test canal Dice | 0.90514 |
| Test corrected-taxonomy disc indexing | 84.08% |
| Sprint 4 (width 32) for reference | test macro Dice 0.89393, indexing 82.70% - worse, so architecture scaling was stopped |

**How indexing currently works.** Disc identity is not predicted by the network. It is derived afterwards by taking connected components of the IVD class *within a single slice*, dropping components under 20 px, and numbering them bottom-up. The index of a disc is therefore a function of how many components that one slice happens to contain, which is the structural weakness this sprint attacks.


### Estimator control

| Field | Value |
| --- | --- |
| Published Sprint 3 test index-correct | 84.08% |
| Re-measured here with the Sprint 5 scorer | 84.08% |
| Difference | +0.0000 pp |
| Estimator identical to Sprint 3's | True |

The Sprint 5 scorer takes a disc *index -> region* mapping rather than re-deriving components from the semantic map, which is what allows a post-processed identity to be scored at all. Because that is a change of plumbing, it was verified to reproduce the published Sprint 3 taxonomy exactly - every category count matches - so every comparison below is like-for-like against 84.08%.


## 3. Indexing Error Analysis

The corrected taxonomy from the Sprint 3 audit is preserved unchanged: every ground-truth disc on every slice is classified as exactly one of `correct`, `shifted`, `merged`, `split` or `missed`, with a 20 px minimum component area and best-overlap matching.

| category | validation n | validation % | test n | test % |
| --- | --- | --- | --- | --- |
| correct | 5454 | 80.82 | 5735 | 84.08 |
| shifted | 990 | 14.67 | 825 | 12.10 |
| merged | 0 | 0.00000 | 2 | 0.03000 |
| split | 25 | 0.37000 | 31 | 0.45000 |
| missed | 279 | 4.13 | 228 | 3.34 |

| Field | Value |
| --- | --- |
| Total discs evaluated (validation) | 6,748 over 1,632 slices |
| Total discs evaluated (test) | 6,821 over 1,655 slices |
| Region found (validation / test) | 95.87% / 96.66% |
| Shifted by exactly +/-1 (validation / test) | 9.13% / 9.79% of all discs |
| Spurious components (validation / test) | 237 / 270 |


### Shift offsets, test baseline

| offset | n discs |
| --- | --- |
| -5.0 | 3 |
| -4.0 | 10 |
| -3.0 | 24 |
| -2.0 | 108 |
| -1.0 | 321 |
| 1.0 | 347 |
| 2.0 | 12 |

The diagnosis is clear from these numbers. Detection is not the problem - only 3.34% of discs are missed outright, and merged and split together account for 0.48%. The dominant failure is `shifted` at 12.10%: the disc was found, and given the wrong number. Most shifts are by a single position, which is the signature of one extra or one absent component low in the stack renumbering everything above it.


## 4. Method 5A - Series-Level Ordering

Replace independent per-slice numbering with identity assigned once at series level, then propagated to every slice.

- **Extract** disc candidates from each sagittal slice as connected components of the IVD class, with their area, mean row and mean column.
- **Associate** candidates across the slices of a series by single-linkage clustering on mean row. Slices of one series share a voxel grid, so the same physical disc occupies nearly the same rows on every slice it appears in.
- **Build** a series-level representation: each cluster becomes a *track* with a median row and a support count, the fraction of slices in which it appears. Tracks below a support threshold are discarded as spurious.
- **Order** the surviving tracks once, most inferior first, matching the dataset's bottom-up convention. This ordering is the series' disc identity.
- **Propagate** by assigning each per-slice candidate the index of its nearest track, subject to a distance gate. Within a slice a track can be claimed only once; the larger candidate wins.

The reason this fixes shifts is that a disc's index no longer depends on the component count of the slice it appears in. If a disc is missed on one slice, the remaining discs keep their series-level identity instead of sliding by one.


### Geometry verified before use

Three conventions the method depends on were measured on ground-truth training data rather than assumed:

| property | measured | use |
| --- | --- | --- |
| vertebra N is superior to disc N | 97.7% of ground-truth discs | ties vertebra identity to disc identity in 5D |
| adjacent disc row gap | median 33.8 px, 5th percentile 25.7 px | clustering threshold must stay well under 25 px |
| vertebral body vs canal column | bodies median 23.5 px anterior, posterior elements 10.7 px behind | separates bodies from posterior elements in 5D |


## 5. Method 5B - Spurious Component Rejection

Spurious components can renumber every disc above them, so rejecting them before ordering is a plausible fix. Two conservative rules were tested: a raised minimum component area, and a column band rejecting candidates far from the series' median disc column.

| min_area_px | col_tolerance_px | validation % correct | % missed | % shifted |
| --- | --- | --- | --- | --- |
| 20 | - | 80.82 | 4.13 | 14.67 |
| 20 | 40.00 | 79.83 | 5.35 | 14.45 |
| 20 | 30.00 | 77.87 | 8.55 | 13.20 |
| 20 | 20.00 | 72.63 | 13.81 | 13.32 |
| 30 | - | 80.81 | 4.89 | 13.97 |
| 30 | 40.00 | 79.88 | 6.06 | 13.74 |
| 30 | 30.00 | 77.95 | 9.25 | 12.48 |
| 30 | 20.00 | 72.66 | 14.40 | 12.74 |
| 40 | - | 79.70 | 5.88 | 14.20 |
| 40 | 40.00 | 78.81 | 6.98 | 13.99 |
| 40 | 30.00 | 76.90 | 10.15 | 12.73 |
| 40 | 20.00 | 71.47 | 15.23 | 13.16 |
| 60 | - | 75.37 | 9.07 | 15.53 |
| 60 | 40.00 | 74.53 | 10.11 | 15.34 |
| 60 | 30.00 | 72.84 | 13.14 | 13.99 |
| 60 | 20.00 | 67.01 | 18.03 | 14.95 |
| 80 | - | 69.26 | 13.49 | 17.25 |
| 80 | 40.00 | 68.46 | 14.46 | 17.07 |
| 80 | 30.00 | 66.89 | 17.32 | 15.78 |
| 80 | 20.00 | 61.43 | 21.83 | 16.75 |

**5B is rejected on validation evidence.** Not one configuration beat the 80.82% baseline; the best was 80.82%, and the best of those is the configuration that rejects nothing. Both rules trade shifted discs for missed discs at a losing rate - tightening the column band to 20 px drops accuracy to 72.63% while raising missed discs from 4.13% to 13.81%. The reason is that a genuine disc on a lateral slice is small and can sit well off the series' median column, so these filters remove real discs faster than spurious ones. Rejection is not carried into the final method.


## 6. Method 5C - Lateral Slice Handling

Indexing accuracy falls on slices carrying little annotated disc area, because a lateral slice holds only fragments. The question is whether indexing should be restricted to sufficiently informative slices.

**The criterion is prediction-derived**, so it is available at test time without any annotation: a slice is informative when its predicted IVD area reaches a threshold. No ground-truth area is used.

| informative_min_ivd_px | all slices % correct | informative subset % correct | discs retained | % of slices retained |
| --- | --- | --- | --- | --- |
| 0 | 93.33 | 93.33 | 6,748 / 6,748 | 100.00 |
| 200 | 93.32 | 95.10 | 6,486 / 6,748 | 66.12 |
| 400 | 93.32 | 95.64 | 6,232 / 6,748 | 61.34 |
| 800 | 90.89 | 94.66 | 5,186 / 6,748 | 47.86 |
| 1200 | 91.21 | 92.15 | 4,702 / 6,748 | 53.43 |

Read the two accuracy columns together with the coverage column. Restricting the *reported* set to informative slices raises accuracy to 95.64% at a 400 px threshold, but it does so by dropping 516 of 6,748 discs and 38.7% of slices from the denominator. That is a coverage trade, not an improvement, and counting it as a gain would be exactly the artificial inflation this sprint was told to avoid.

On the like-for-like all-slices denominator the criterion changes nothing (93.32% against 93.33%). **5C is therefore not adopted as a filter.** Its measurement is still useful: it quantifies how much of the residual error is concentrated in genuinely uninformative slices, and it is reported here so that a future measurement stage can decide to *measure* only on informative slices while still *indexing* all of them.


## 7. Method 5D - Vertebral Body Separation

Vertebrae must not be numbered by counting connected components. The Sprint 3 audit established that one vertebra occupies roughly two components in a sagittal slice - the body, and the posterior elements - so ordering components interleaves fragments of different vertebrae regardless of segmentation quality. This is the root cause of the unusable disc-to-vertebra height ratio.

- Identify the **vertebral body** as the component anterior to the spinal canal centroid column, measured per slice with a series-level fallback when the canal is absent. Posterior elements keep their semantic class but receive no identity.
- Number each body by **the disc tracks below it**, so body N is the one directly above disc N. Tying vertebra identity to disc identity avoids reintroducing a second count-dependent ordering, and it matches the convention the measurement code already assumes.

| category | ordered components n | ordered components % | vertebral bodies n | vertebral bodies % |
| --- | --- | --- | --- | --- |
| correct | 951 | 11.63 | 5115 | 62.53 |
| shifted | 1679 | 20.53 | 727 | 8.89 |
| merged | 89 | 1.09 | 89 | 1.09 |
| split | 3427 | 41.89 | 50 | 0.61000 |
| missed | 2034 | 24.87 | 2199 | 26.88 |

| Field | Value |
| --- | --- |
| Ground-truth vertebrae scored (validation) | 8,180 |
| Ordered components, index correct | 11.63% |
| Vertebral bodies (5D), index correct | 62.53% |
| Absolute change | +50.90 pp |
| Adopted | True |

The ordered-component scheme scores 41.89% `split`, which is the audit's prediction showing up directly in the measurement: a single ground-truth vertebra is routinely covered by two differently-numbered predicted components. Body identification cuts that to 0.61% and raises correct identification from 11.63% to 62.53%. It is scored with the same taxonomy rules as the discs, so the two are directly readable against each other. **5D is adopted.** Note that at 62.53% it remains the weakest stage in the pipeline, with 26.88% of vertebrae still unmatched.


## 8. Validation Results

All parameters were selected on the validation split. 78 configurations were scored; the test split was not read during tuning.

| Field | Value |
| --- | --- |
| Split used for tuning | val |
| Test set touched during tuning | False |
| Configurations evaluated | 78 |
| Validation baseline | 80.82% |
| Selected method | 5A+5D |
| Selected validation accuracy | 93.33% |
| Absolute change on validation | +12.51 pp |


### Selected parameters

| parameter | value |
| --- | --- |
| min_area_px | 20 |
| col_tolerance_px | None |
| use_series_tracks | True |
| cluster_px | 8.0 |
| min_track_support | 0.25 |
| assign_max_dist_px | 14.0 |
| informative_min_ivd_px | 0 |
| informative_min_components | 0 |
| use_vertebral_bodies | True |
| body_canal_margin_px | 0.0 |
| max_discs | 9 |


### Best configuration per method family

| method | validation % correct | vs baseline (pp) | % shifted | % missed | discs scored | runtime s |
| --- | --- | --- | --- | --- | --- | --- |
| 5B | 80.82 | +0.00 | 14.67 | 4.13 | 6748 | 0.07000 |
| 5A | 93.33 | +12.51 | 2.52 | 4.15 | 6748 | 0.09000 |
| 5A+5B | 93.33 | +12.51 | 2.52 | 4.15 | 6748 | 0.10000 |
| 5C(all slices) | 93.33 | +12.51 | 2.52 | 4.15 | 6748 | 0.09000 |
| 5C(informative subset) | 95.64 | +14.82 | 2.36 | 2.01 | 6232 | 0.09000 |

The 5C rows are shown for completeness. Only the all-slices variants were eligible for selection, because the informative-subset variant scores a smaller denominator and is not comparable with the baseline.


### Sensitivity

The clustering threshold behaves exactly as the ground-truth geometry predicts. Thresholds of 8 px and 12 px are indistinguishable (93.33%), 16 px costs a little (92.81%), and 20 px collapses to 79.48% - because the 5th percentile gap between adjacent discs is 25.7 px, so a 20 px threshold starts merging neighbouring discs into one track. The method is not sensitive within the range the anatomy allows, and its failure mode outside that range is understood rather than mysterious.


## 9. Final Test Results

The single selected configuration was applied once to the held-out test set. No parameter was changed after seeing these numbers.

| Field | Value |
| --- | --- |
| Method applied | 5A+5D |
| Selected on | validation |
| Validation accuracy of this method | 93.33% |
| Test baseline | 84.08% |
| Test Sprint 5 | 93.84% |
| Absolute change | +9.76 pp |
| Slices written | 1,655 |
| Segmentation predictions unchanged | True |

| category | baseline n | baseline % | Sprint 5 n | Sprint 5 % | Δ pp |
| --- | --- | --- | --- | --- | --- |
| correct | 5735 | 84.08 | 6401 | 93.84 | 9.76 |
| shifted | 825 | 12.10 | 172 | 2.52 | -9.58 |
| merged | 2 | 0.03000 | 2 | 0.03000 | 0.00000 |
| split | 31 | 0.45000 | 0 | 0.00000 | -0.45000 |
| missed | 228 | 3.34 | 246 | 3.61 | 0.27000 |

| Field | Value |
| --- | --- |
| Region found | 96.66% -> 96.39% |
| Slices with correct disc count | 77.16% -> 78.55% |
| Slices with every disc correct | 77.95% -> 84.89% |
| Spurious components | 270 -> 239 |
| Mean tracks per series | 6.98 |
| Post-processing runtime | 0.09s for 1,655 slices (plus 26.6s to write the masks) |


## 10. Sprint 3 vs Sprint 5

| Metric | Sprint 3 | Sprint 5 | Absolute Δ | Verdict |
| --- | --- | --- | --- | --- |
| indexing correct % | 84.08 | 93.84 | +9.7600 | improved |
| indexing shifted % | 12.10 | 2.52 | -9.5800 | improved |
| indexing merged % | 0.03 | 0.03 | +0.0000 | unchanged |
| indexing split % | 0.45 | 0.00 | -0.4500 | improved |
| indexing missed % | 3.34 | 3.61 | +0.2700 | worsened |
| indexing region found % | 96.66 | 96.39 | -0.2700 | worsened |
| slices with every disc correct % | 77.95 | 84.89 | +6.9400 | improved |
| spurious components | 270 | 239 | -31.0000 | improved |
| height_mm_central MAE | 0.7634 | 0.6492 | -0.1142 | improved |
| area_mm2 MAE | 24.8986 | 21.4831 | -3.4155 | improved |
| height_mm_central Pearson r | 0.8870 | 0.9128 | +0.0258 | improved |
| area_mm2 Pearson r | 0.8946 | 0.9261 | +0.0315 | improved |
| intensity ratio MAE | 0.0364 | 0.0280 | -0.0084 | improved |
| intensity ratio Pearson r | 0.9639 | 0.9726 | +0.0087 | improved |
| disc_to_vertebra_height_ratio MAE % of mean | 71.87 | 33.72 | -38.1500 | improved |
| % of truth discs recovered (series level) | 99.33 | 99.11 | -0.2200 | worsened |
| predicted discs not in truth | 21 | 4 | -17.0000 | improved |
| Pfirrmann forest QWK | 0.6317 | 0.6543 | +0.0226 | improved |
| Pfirrmann logreg QWK | 0.6038 | 0.6574 | +0.0536 | improved |
| Pfirrmann forest within 1 grade | 0.8333 | 0.8571 | +0.0238 | improved |


### Are the segmentation predictions unchanged?

| Field | Value |
| --- | --- |
| `semantic` maps byte-identical to Sprint 3 | True |
| Segmentation model retrained | False |
| Segmentation metrics affected | No - macro foreground Dice, per-class Dice, IoU, precision and recall are all properties of the `semantic` map, which is copied through unchanged |
| What did change | Only the `instance` map, i.e. which integer identity is attached to each already-segmented disc and vertebral body |

This is the central property of the sprint. Sprint 5 adds no segmentation quality and claims none: pixel-level accuracy is exactly Sprint 3's, verified byte-for-byte on all 1,655 test slices. What improves is everything that depends on a disc being correctly *identified* - which is the whole downstream chain.


## 11. Failure Cases

420 of 6,821 test discs are still not correctly indexed (6.16%). The residual is dominated by a different category than the baseline's.

| category | n | % of all discs | % of remaining errors |
| --- | --- | --- | --- |
| shifted | 172 | 2.52 | 40.95 |
| merged | 2 | 0.03000 | 0.48000 |
| split | 0 | 0.00000 | 0.00000 |
| missed | 246 | 3.61 | 58.57 |


### Remaining shift offsets

| offset | n discs |
| --- | --- |
| 1 | 172 |


### Accuracy by disc index

| disc index | n | baseline % correct | Sprint 5 % correct | Δ pp |
| --- | --- | --- | --- | --- |
| 1 | 1028 | 90.08 | 92.22 | 2.14 |
| 2 | 1072 | 86.19 | 95.62 | 9.43 |
| 3 | 1048 | 82.82 | 94.47 | 11.65 |
| 4 | 994 | 81.49 | 94.16 | 12.67 |
| 5 | 925 | 82.70 | 94.38 | 11.68 |
| 6 | 854 | 84.66 | 95.43 | 10.77 |
| 7 | 583 | 82.16 | 94.34 | 12.18 |
| 8 | 269 | 74.35 | 80.30 | 5.95 |
| 9 | 48 | 83.33 | 100.00 | 16.67 |

The character of the residual error has changed. In the baseline the largest category was `shifted` at 12.10%; after series-level ordering `shifted` falls to 2.52% and `missed` is now the largest remaining category at 3.61%. The weakest disc index is 8 at 80.30% correct over 269 discs. High indices are the hardest because they sit at the superior end of the field of view where a disc may be only partly imaged, and because an error anywhere below them can still displace a track.

**The one cost of the method, stated plainly.** Missed discs rose by 18 (3.34% -> 3.61%), and region-found correspondingly fell from 96.66% to 96.39%. This is not a segmentation change - the pixels are identical - it is the track-assignment gate refusing to number a candidate that sits further than the allowed distance from any series track, plus the rule that a track may be claimed only once per slice. Those candidates were previously given a number, usually the wrong one. So the trade is 18 discs moved from 'numbered wrongly' to 'not numbered', against 653 discs moved from 'numbered wrongly' to 'numbered correctly'. That is a favourable exchange at roughly 36:1, and an unnumbered disc is more honest downstream than a confidently mis-numbered one, because it is excluded from the measurement join rather than attaching its measurements to the wrong grading. It is still a genuine regression on those two metrics and is counted as such in the comparison table.

Beyond that, the remaining `missed` population is a segmentation limitation rather than an indexing one: post-processing cannot number a disc the network never predicted. That puts a hard ceiling of 96.39% on what any indexing method can reach on these frozen predictions, and Sprint 5 now sits 2.55 pp below it - against 12.58 pp for the baseline. Most of the headroom that existed in the indexing stage has been taken.


## 12. Limitations

- **A whole-series shift is not detectable.** Series-level ordering fixes inconsistency *between* slices, but if the most inferior disc is absent from the prediction on every slice of a series, every track shifts by one and the method has no internal evidence of it. This is the residual `shifted` population, still 2.52% of discs.
- **No anatomical anchor.** Nothing here identifies the sacrum or any named level, so the numbering is relative to the most inferior detected disc, not to anatomy. Correcting a whole-series shift would need such an anchor, which the current annotation does not provide.
- **Vertebral body identification is the weak stage.** 5D raises vertebra identity to 62.53% on validation, a large gain, but 26.88% of vertebrae are still unmatched, and the canal-column split degrades on slices where the canal is not predicted.
- **disc_to_vertebra_height_ratio is improved but still not reliable.** Its error falls from 71.87% to 33.72% of the ground-truth mean. That is a large improvement and still too high to use as a derived feature.
- **Single split, single model.** All numbers come from one patient-level split and one frozen segmentation model. The test set is 33 patients.
- **Parameters were selected on 39 validation patients.** The validation baseline (80.82%) is lower than the test baseline (84.08%), so the two splits are not equally difficult; the gain transferred (+12.51 pp validation, +9.76 pp test) but was not identical.
- **Segmentation metrics are unchanged and are not clinical evidence.** A Dice score or an indexing accuracy measures agreement with one annotation protocol. Neither establishes clinical effectiveness or readiness for medical use, and none is claimed.
- **Pfirrmann grades are dataset annotations** used as labels for a measurement exercise. Nothing here grades a patient.
- **No longitudinal or postoperative scope.** The SPIDER data used here contains no longitudinal postoperative follow-up, so postoperative healing stays outside the validated scope of this work and no claim about it is made or supported. No change over time is measured, and no 'percentage spine damage' score exists in this pipeline.


## 13. Recommended Pipeline

The recommended configuration is the one evaluated above. It is reproducible from the frozen Sprint 3 predictions and adds well under a minute of CPU time for the whole test set.

| stage | component | note |
| --- | --- | --- |
| 1. segmentation | Sprint 3 Coverage 16-channel U-Net, frozen | no retraining; `semantic` output used unchanged |
| 2. disc extraction | connected components of the IVD class, 20 px minimum | unchanged from the baseline; no rejection filter (5B rejected) |
| 3. disc ordering | series-level tracks (5A) | cluster 8.0 px, support 0.25, gate 14.0 px |
| 4. vertebra identity | vertebral-body separation (5D) | canal-column split, numbered from the disc tracks below |
| 5. measurements | existing disc_features pipeline, unchanged | consumes the corrected `instance` map |
| 6. Pfirrmann / findings | existing end-to-end evaluation, unchanged | grade models refitted per run as before |

5B and 5C are explicitly **not** part of the recommendation. 5B was measured to hurt on validation, and 5C changes nothing on a like-for-like denominator.


## 14. Next Step

**Anchor the series-level numbering to an anatomical reference, then re-measure.** Series-level ordering has taken indexing from 84.08% to 93.84%, and the residual error is now dominated by two things it cannot fix: discs the network never predicted, and whole-series shifts where every track is displaced together. The first needs better segmentation recall at the ends of the stack; the second needs an anchor, because relative ordering alone cannot detect a uniform offset.

- The measurable ceiling is now explicit: region-found is 96.39% on these frozen predictions, so no indexing method can exceed that without improving detection.
- Whole-series shift is the largest remaining *indexing* error and is invisible to a relative method by construction, so it should be attacked with an anchor (for example the most inferior fully-imaged disc, or a sacrum cue) rather than with more ordering heuristics.
- Vertebral-body identity at 62.53% is the weakest stage and the one gating disc-to-vertebra measurements, which remain unreliable at 33.72% error.
- Any such work is again post-processing on frozen predictions, so it is cheap to try and cannot disturb the segmentation results.

**This has not been started.** It is a recommendation only, and no further experiment was launched after this report.


## Artefact Locations

| artefact | path |
| --- | --- |
| baseline audit (validation + test) | `outputs/reports/sprint5_indexing/baseline_audit.json` |
| validation sweep (all configurations) | `outputs/reports/sprint5_indexing/validation_tuning.json`, `validation_sweep.csv` |
| method 5D on validation | `outputs/reports/sprint5_indexing/validation_5d.json` |
| final test results | `outputs/reports/sprint5_indexing/test_results.json` |
| per-disc failure records (test) | `outputs/reports/sprint5_indexing/indexing_failures_sprint5_test.csv` |
| measurement agreement + Pfirrmann | `outputs/reports/sprint5_indexing/metrics/measurement_agreement.json` |
| disc measurements | `outputs/reports/disc_analysis_sprint5_indexing.csv` |
| corrected predictions (semantic unchanged) | `data/processed/predictions_sprint5_indexing/` |
| validation-split predictions (Sprint 3 model) | `data/processed/predictions_sprint5_val_raw/` |
| post-processing implementation | `src/analysis/disc_postprocess.py` |
| driver script | `scripts/16_sprint5_indexing.py` |
| figures | `outputs/visualizations/sprint5_indexing/` |
| this report | `outputs/reports/sprint5_indexing/sprint5_final_report.md` (+ .json) |


### Figures

- `outputs/visualizations/sprint5_indexing/indexing_before_after.png`
- `outputs/visualizations/sprint5_indexing/validation_sweep.png`
- `outputs/visualizations/sprint5_indexing/method_5d_vertebrae.png`
- `outputs/visualizations/sprint5_indexing/measurement_improvement.png`
- `outputs/visualizations/sprint5_indexing/measurement_agreement.png`

