# Sprint 3 Coverage - Final Report

*Controlled single-variable experiment: training-data coverage*

Generated: 2026-09-23 22:06:02


## 1. Objective

Answer one question: **was insufficient training-data coverage the main limitation of the 16-channel U-Net?**

Sprint 2 Extended established that the 16-channel model had converged, but the Sprint 3 audit found it had only ever seen **28.05%** of the available training slices - the sampler drew a 2,560-slice subset once, before training, and reused it every epoch. A plateau on 28% of the data cannot be attributed to insufficient capacity. This experiment removes that confound by rotating the subset so the model sees all 9,128 training slices, while changing nothing else.


## 2. Experimental Setup


### Changed (the single variable)

| Field | Value |
| --- | --- |
| Sampler | rotating shard sampler (was: fixed subset sampled once) |
| Slices per epoch | 2284 mean (2234-2327) |
| Intended coverage | 100% of 9,128 training slices by ~epoch 4 |


### Unchanged from Sprint 2 Extended

| Field | Value |
| --- | --- |
| architecture | UNet(in=1, classes=4, base=16, depth=4, bilinear=True) 1,963,860 params |
| base_channels | 16 |
| depth | 4 |
| batch_size | 8 |
| optimiser | Adam |
| lr | 0.00100 |
| scheduler | CosineAnnealingLR(T_max=30) |
| loss | CrossEntropy + soft Dice (equal weight) |
| augmentation | translate +/-5% + intensity jitter, train only |
| preprocessing | Sprint 1 slices, 352x256 at 1.0 mm/px |
| seed | 42 |
| split | Sprint 1 patient-level |
| validation_slices | 800 |
| n_classes | 4 |
| input_size | 352, 256 |
| memory_format | channels_last |
| bf16_autocast | False |


### Different by design

| Field | Value |
| --- | --- |
| initialisation | random (NOT resumed, NOT warm-started) |
| early_stopping_min_delta | 0.00100 |
| rationale | Warm-starting from the converged Sprint 2 weights would confound the coverage variable. min_delta was 0 in Sprint 2, which made early stopping ineffective. |

Random initialisation is required: warm-starting from the converged Sprint 2 weights would confound the coverage variable with the effect of already-trained weights. `min_delta` was raised from 0 to 0.001 because Sprint 2 showed that 0 makes early stopping ineffective - any improvement, however small, reset the patience counter.


### Verification that the run completed

| Field | Value |
| --- | --- |
| All expected artefacts present | True |
| Missing artefacts | none |
| Epochs in history.csv | 29 |
| Epochs reported in summary | 29 |
| history matches summary | True |
| Early stopping triggered | True |
| Stop reason | validation foreground Dice did not improve for 6 consecutive epochs (best 0.8981 at epoch 28) |


## 3. Training Results

| Field | Value |
| --- | --- |
| Total epochs completed | 29 / 30 |
| Early stopping triggered | True |
| Stop reason | validation foreground Dice did not improve for 6 consecutive epochs (best 0.8981 at epoch 28) |
| Epochs without improvement at end | 6 |
| Best validation foreground Dice | 0.89831 (epoch 29) |
| Best validation loss | 0.13369 (epoch 28) |
| Final validation Dice | 0.89831 |
| Final validation loss | 0.13371 |
| Train loss at best epoch | 0.11045 |
| Validation loss at best epoch | 0.13371 |
| Learning rate at best epoch | 1.093e-05 |
| Per-class Dice at best epoch | vertebra 0.9105, IVD 0.8766, canal 0.9078 |
| Total training time | 5.88 h (21175 s summed over epochs) |
| Mean epoch time | 730.2 s |
| Min / max epoch time | 675.3 s / 929.3 s |


### Per-epoch history

| epoch | learning_rate | n_train_slices_this_epoch | unique_train_slices_seen | train_loss | val_loss | val_fg_dice | val_dice_vertebra | val_dice_intervertebral_disc | val_dice_spinal_canal | epoch_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00100 | 2327 | 2327 | 0.88734 | 0.43070 | 0.75349 | 0.81688 | 0.80636 | 0.63724 | 792.60 |
| 2 | 0.00100 | 2314 | 4641 | 0.26104 | 0.22701 | 0.85717 | 0.86536 | 0.84628 | 0.85986 | 712.10 |
| 3 | 0.000990 | 2253 | 6894 | 0.19036 | 0.19637 | 0.87226 | 0.87912 | 0.85405 | 0.88362 | 706.50 |
| 4 | 0.000980 | 2234 | 9128 | 0.16019 | 0.18205 | 0.87782 | 0.88666 | 0.86544 | 0.88136 | 700.60 |
| 5 | 0.000960 | 2327 | 9128 | 0.15200 | 0.20569 | 0.86031 | 0.87950 | 0.82592 | 0.87552 | 701.30 |
| 6 | 0.000930 | 2314 | 9128 | 0.15920 | 0.18972 | 0.87101 | 0.88745 | 0.86148 | 0.86410 | 694.40 |
| 7 | 0.000900 | 2253 | 9128 | 0.15005 | 0.19027 | 0.86611 | 0.89118 | 0.83018 | 0.87698 | 678.90 |
| 8 | 0.000870 | 2234 | 9128 | 0.14376 | 0.16815 | 0.88493 | 0.89852 | 0.86818 | 0.88808 | 677.40 |
| 9 | 0.000830 | 2327 | 9128 | 0.13807 | 0.16868 | 0.88679 | 0.89957 | 0.86888 | 0.89190 | 704.90 |
| 10 | 0.000790 | 2314 | 9128 | 0.14005 | 0.17430 | 0.88075 | 0.89794 | 0.86244 | 0.88186 | 702.30 |
| 11 | 0.000750 | 2253 | 9128 | 0.13032 | 0.16098 | 0.89020 | 0.90277 | 0.86534 | 0.90250 | 682.50 |
| 12 | 0.000700 | 2234 | 9128 | 0.12925 | 0.15859 | 0.89134 | 0.90353 | 0.87014 | 0.90035 | 685.90 |
| 13 | 0.000650 | 2327 | 9128 | 0.12627 | 0.15961 | 0.88966 | 0.90006 | 0.87131 | 0.89762 | 706.70 |
| 14 | 0.000600 | 2314 | 9128 | 0.12764 | 0.15505 | 0.89086 | 0.90494 | 0.87121 | 0.89645 | 706.40 |
| 15 | 0.000550 | 2253 | 9128 | 0.13939 | 0.16942 | 0.87988 | 0.89232 | 0.86362 | 0.88369 | 687.30 |
| 16 | 0.000500 | 2234 | 9128 | 0.13370 | 0.15232 | 0.88985 | 0.90236 | 0.87207 | 0.89510 | 676.90 |
| 17 | 0.000450 | 2327 | 9128 | 0.12684 | 0.15279 | 0.89116 | 0.90413 | 0.87192 | 0.89742 | 701.30 |
| 18 | 0.000400 | 2314 | 9128 | 0.12681 | 0.14229 | 0.89575 | 0.90769 | 0.87570 | 0.90385 | 687.40 |
| 19 | 0.000350 | 2253 | 9128 | 0.12601 | 0.14529 | 0.89391 | 0.90509 | 0.87433 | 0.90230 | 747.80 |
| 20 | 0.000300 | 2234 | 9128 | 0.11743 | 0.14535 | 0.89332 | 0.90721 | 0.87382 | 0.89894 | 929.30 |
| 21 | 0.000250 | 2327 | 9128 | 0.11730 | 0.13580 | 0.89575 | 0.90811 | 0.87550 | 0.90365 | 848.70 |
| 22 | 0.000210 | 2314 | 9128 | 0.12123 | 0.13886 | 0.89584 | 0.90903 | 0.87401 | 0.90448 | 840.50 |
| 23 | 0.000170 | 2253 | 9128 | 0.11525 | 0.13778 | 0.89717 | 0.90946 | 0.87599 | 0.90605 | 822.50 |
| 24 | 0.000130 | 2234 | 9128 | 0.11783 | 0.13889 | 0.89727 | 0.90978 | 0.87655 | 0.90548 | 791.20 |
| 25 | 0.000100 | 2327 | 9128 | 0.11363 | 0.13665 | 0.89766 | 0.91015 | 0.87620 | 0.90661 | 779.30 |
| 26 | 0.000070 | 2314 | 9128 | 0.11416 | 0.13545 | 0.89761 | 0.91052 | 0.87537 | 0.90695 | 754.00 |
| 27 | 0.000040 | 2253 | 9128 | 0.11282 | 0.13706 | 0.89702 | 0.90960 | 0.87474 | 0.90671 | 679.40 |
| 28 | 0.000020 | 2234 | 9128 | 0.11006 | 0.13369 | 0.89811 | 0.91037 | 0.87637 | 0.90759 | 675.30 |
| 29 | 0.000010 | 2327 | 9128 | 0.11045 | 0.13371 | 0.89831 | 0.91052 | 0.87664 | 0.90777 | 701.70 |


## 4. Training Data Coverage

| Field | Value |
| --- | --- |
| Training slices available | 9,128 |
| Unique slices seen | 9,128 |
| Final coverage | 100.0% |
| Sprint 2 Extended coverage (for reference) | 28.05% |
| Epoch reaching 100% coverage | 4 |
| 100% achieved by epoch 4 as intended | True |
| Times each slice was seen | mean 7.25, range 7-8 (sd 0.436) |
| Slices never seen | 0 |


### Leakage verification

| Field | Value |
| --- | --- |
| Leakage-free across every epoch | True |
| Violations found | 0 |
| Validation/test patients excluded | 66 |
| Validation/test slices excluded | 3287 |

The sampler partitions only rows already marked `split == 'train'`, and the training script additionally asserts per epoch that the subset contains no other split. No validation or test slice **or patient** entered training.


## 5. Convergence Analysis

| Field | Value |
| --- | --- |
| Validation Dice, last 5 epochs | 0.89702 - 0.89831 (spread 0.00129) |
| Mean gain per epoch, last 5 | +0.000163 |
| Mean gain per epoch, epochs 5-10 | +0.004086 |
| Final learning rate | 1.093e-05 |
| Best epoch is the final epoch | True |
| Early stopping triggered | True |
| Assessed as converged | False |

Validation Dice was still moving +0.000163 per epoch over the final 5 epochs, against +0.004086 per epoch during epochs 5-10, so the run had not fully flattened when it ended.


## 6. Final Test Segmentation Results

Evaluated **once**, after training finished, on the untouched held-out test set: 1,655 slices from 33 patients. The checkpoint was selected by the predefined validation criterion, not by test performance.


### Aggregate (dataset-level) metrics

| class | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- |
| background | 0.99527 | 0.99058 | 0.99537 | 0.99516 |
| vertebra | 0.91663 | 0.84608 | 0.91950 | 0.91377 |
| intervertebral_disc | 0.87827 | 0.78297 | 0.86208 | 0.89509 |
| spinal_canal | 0.90514 | 0.82671 | 0.89521 | 0.91529 |
| **macro foreground** | 0.90001 | 0.81859 | 0.89226 | 0.90805 |

| Field | Value |
| --- | --- |
| Pixel accuracy | 0.99092 |


### Per-patient Dice (mean +/- sd over test patients)

| class | mean Dice | sd | patients |
| --- | --- | --- | --- |
| background | 0.99531 | 0.00102 | 33 |
| vertebra | 0.91637 | 0.01572 | 33 |
| intervertebral_disc | 0.88684 | 0.02938 | 33 |
| spinal_canal | 0.90418 | 0.02432 | 33 |


## 7. Sprint 2 Extended vs Sprint 3 Comparison

`Absolute Δ` is Sprint 3 minus Sprint 2 Extended. `Relative Δ` is that as a percentage of the Sprint 2 value. Verdicts account for metric direction (lower is better for MAE, loss and standard deviation).


### Primary criterion

| Metric | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ | Relative Δ | Verdict |
| --- | --- | --- | --- | --- | --- |
| validation macro foreground Dice (best) | 0.89850 | 0.89831 | -0.00019 | -0.022% | worsened |


### Segmentation metrics

| Metric | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ | Relative Δ | Verdict |
| --- | --- | --- | --- | --- | --- |
| test background/dice | 0.99514 | 0.99527 | +0.00013 | +0.013% | improved |
| test background/iou | 0.99032 | 0.99058 | +0.00026 | +0.026% | improved |
| test background/precision | 0.99509 | 0.99537 | +0.00028 | +0.028% | improved |
| test background/recall | 0.99518 | 0.99516 | -0.00002 | -0.002% | worsened |
| test vertebra/dice | 0.91332 | 0.91663 | +0.00331 | +0.362% | improved |
| test vertebra/iou | 0.84047 | 0.84608 | +0.00562 | +0.668% | improved |
| test vertebra/precision | 0.91777 | 0.91950 | +0.00172 | +0.188% | improved |
| test vertebra/recall | 0.90891 | 0.91377 | +0.00486 | +0.535% | improved |
| test intervertebral_disc/dice | 0.87790 | 0.87827 | +0.00037 | +0.042% | improved |
| test intervertebral_disc/iou | 0.78238 | 0.78297 | +0.00059 | +0.075% | improved |
| test intervertebral_disc/precision | 0.87109 | 0.86208 | -0.00901 | -1.034% | worsened |
| test intervertebral_disc/recall | 0.88483 | 0.89509 | +0.01026 | +1.160% | improved |
| test spinal_canal/dice | 0.90256 | 0.90514 | +0.00258 | +0.285% | improved |
| test spinal_canal/iou | 0.82242 | 0.82671 | +0.00429 | +0.521% | improved |
| test spinal_canal/precision | 0.89258 | 0.89521 | +0.00263 | +0.295% | improved |
| test spinal_canal/recall | 0.91277 | 0.91529 | +0.00252 | +0.276% | improved |
| test macro_foreground/dice | 0.89793 | 0.90001 | +0.00208 | +0.232% | improved |
| test macro_foreground/iou | 0.81509 | 0.81859 | +0.00350 | +0.429% | improved |
| test macro_foreground/precision | 0.89381 | 0.89226 | -0.00155 | -0.174% | worsened |
| test macro_foreground/recall | 0.90217 | 0.90805 | +0.00588 | +0.652% | improved |
| test pixel_accuracy | 0.99067 | 0.99092 | +0.00025 | +0.025% | improved |
| test per_patient_dice/background | 0.99522 | 0.99531 | +0.00009 | +0.009% | improved |
| test per_patient_dice_sd/background | 0.00103 | 0.00102 | -0.00001 | -1.249% | improved |
| test per_patient_dice/vertebra | 0.91405 | 0.91637 | +0.00232 | +0.254% | improved |
| test per_patient_dice_sd/vertebra | 0.01676 | 0.01572 | -0.00105 | -6.244% | improved |
| test per_patient_dice/intervertebral_disc | 0.88678 | 0.88684 | +0.00006 | +0.006% | improved |
| test per_patient_dice_sd/intervertebral_disc | 0.02991 | 0.02938 | -0.00053 | -1.759% | improved |
| test per_patient_dice/spinal_canal | 0.90253 | 0.90418 | +0.00165 | +0.183% | improved |
| test per_patient_dice_sd/spinal_canal | 0.02251 | 0.02432 | +0.00181 | +8.030% | worsened |


### Tally across all compared metrics

| Field | Value |
| --- | --- |
| worsened | 19 |
| improved | 48 |
| unavailable | 1 |


## 8. Disc Indexing Results

Corrected taxonomy: every ground-truth disc is classified as `correct`, `shifted`, `merged`, `split` or `missed`. Analysis uses the **integer disc index only** - the dataset does not state which vertebra is L5, so no anatomical level name is asserted. The invalid vertebra-component-vs-instance comparison identified in the Sprint 3 audit is not used.

| Field | Value |
| --- | --- |
| Discs analysed | 6,821 |
| Slices analysed | 1,655 |
| Region found | 96.66% |
| Index correct | 84.08% |
| Slices with correct disc count | 77.16% |
| Slices with every disc correct | 77.95% |
| Spurious components | 270 |
| Shift offsets | `-5.0`=3, `-4.0`=10, `-3.0`=24, `-2.0`=108, `-1.0`=321, `1.0`=347, `2.0`=12 |

| Metric | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ | Verdict |
| --- | --- | --- | --- | --- |
| indexing pct_index_correct | 82.25 | 84.08 | +1.830 | improved |
| indexing pct_region_found | 95.73 | 96.66 | +0.930 | improved |
| indexing pct_slices_count_correct | 75.53 | 77.16 | +1.630 | improved |
| indexing pct_slices_all_discs_correct | 75.11 | 77.95 | +2.840 | improved |
| indexing total_spurious_components | 249 | 270 | +21.000 | worsened |
| indexing pct_slices_with_spurious | 13.29 | 14.56 | +1.270 | worsened |
| indexing category %/correct | 82.25 | 84.08 | +1.830 | improved |
| indexing category %/shifted | 12.97 | 12.10 | -0.870 | improved |
| indexing category %/split | 0.51000 | 0.45000 | -0.060 | improved |
| indexing category %/missed | 4.27 | 3.34 | -0.930 | improved |


### Reconciliation: two estimators of indexing accuracy

The brief quotes the Sprint 2 Extended disc-indexing baseline as **82.6%**. That figure comes from the `evaluate_unet.py` disc-identification metric, which is a *different estimator* from the corrected-taxonomy `pct_index_correct` used above. The two differ because the taxonomy additionally resolves merged and split components before deciding whether an index is correct. Both are listed here against their own baseline so the improvement is never computed across estimators.

| Estimator | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ |
| --- | --- | --- | --- |
| `evaluate_unet` pct_discs_index_correct (the brief's 82.6% baseline) | 82.60% | 84.42% | +1.82 pp |
| corrected taxonomy `pct_index_correct` (headline above) | 82.25% | 84.08% | +1.83 pp |

The two estimators agree on the size and direction of the change (about +1.8 pp), which is why the indexing conclusion does not depend on which one is quoted.


### Accuracy by disc index

| truth_index | n | correct | shifted | merged | split | missed | pct_correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1028 | 926 | 24 | 0 | 29 | 49 | 90.08 |
| 2 | 1072 | 924 | 129 | 0 | 0 | 19 | 86.19 |
| 3 | 1048 | 868 | 149 | 0 | 1 | 30 | 82.82 |
| 4 | 994 | 810 | 151 | 0 | 0 | 33 | 81.49 |
| 5 | 925 | 765 | 130 | 0 | 0 | 30 | 82.70 |
| 6 | 854 | 723 | 113 | 1 | 1 | 16 | 84.66 |
| 7 | 583 | 479 | 87 | 1 | 0 | 16 | 82.16 |
| 8 | 269 | 200 | 34 | 0 | 0 | 35 | 74.35 |
| 9 | 48 | 40 | 8 | 0 | 0 | 0 | 83.33 |


### Accuracy by slice annotation area

| area_band | n | pct_correct | pct_shifted | pct_missed |
| --- | --- | --- | --- | --- |
| <1k | 135 | 28.15 | 23.70 | 48.15 |
| 1k-3k | 587 | 55.88 | 32.03 | 11.93 |
| 3k-6k | 1327 | 76.87 | 18.54 | 4.07 |
| 6k-10k | 3401 | 91.50 | 7.23 | 0.76000 |
| >10k | 1371 | 90.23 | 8.24 | 0.95000 |


## 9. Disc-Level Measurement Results

Measurements taken from predicted masks, compared against the same discs measured from ground-truth masks. Only discs whose identity the prediction recovered correctly are compared - detection failures are counted separately rather than averaged into the measurement error.

| Field | Value |
| --- | --- |
| Discs compared | 444 |
| Ground-truth discs on predicted series | 447 |
| % of truth discs recovered | 99.33% |

| Metric | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ | Verdict |
| --- | --- | --- | --- | --- |
| MAE ap_extent_mm | 1.99 | 2.00 | +0.00900 | worsened |
| r ap_extent_mm | 0.77660 | 0.77120 | -0.00540 | worsened |
| MAE area_mm2 | 25.34 | 24.90 | -0.44450 | improved |
| r area_mm2 | 0.90790 | 0.89460 | -0.01330 | worsened |
| MAE canal_width_at_disc_mm | 1.13 | 1.15 | +0.02150 | worsened |
| r canal_width_at_disc_mm | 0.81070 | 0.80650 | -0.00420 | worsened |
| MAE disc_to_vertebra_height_ratio | 0.13610 | 0.13720 | +0.00110 | worsened |
| r disc_to_vertebra_height_ratio | 0.64670 | 0.67500 | +0.02830 | improved |
| MAE height_mm_anterior | 0.87800 | 0.84250 | -0.03550 | improved |
| r height_mm_anterior | 0.85350 | 0.84370 | -0.00980 | worsened |
| MAE height_mm_central | 0.80650 | 0.76340 | -0.04310 | improved |
| r height_mm_central | 0.88750 | 0.88700 | -0.00050 | worsened |
| MAE height_mm_posterior | 0.75310 | 0.73990 | -0.01320 | improved |
| r height_mm_posterior | 0.74780 | 0.76140 | +0.01360 | improved |
| MAE height_ratio_to_series_median | 0.11860 | 0.11230 | -0.00630 | improved |
| r height_ratio_to_series_median | 0.83560 | 0.84810 | +0.01250 | improved |
| MAE intensity_disc_vertebra_ratio | 0.03600 | 0.03640 | +0.00040 | worsened |
| r intensity_disc_vertebra_ratio | 0.96380 | 0.96390 | +0.00010 | improved |
| disc detection %truth recovered | 99.11 | 99.33 | +0.22000 | improved |


## 10. End-to-End Pfirrmann Results

Models fitted on ground-truth-mask features from **training** patients, then applied to **predicted**-mask features from test patients. This is the only Pfirrmann figure that responds to the segmentation model, because the Stage E table itself uses ground-truth masks on both sides.

| Field | Value |
| --- | --- |
| Note | Fitted on ground-truth-mask features from training patients, evaluated on predicted-mask features from test patients. Includes segmentation error, unlike the Stage E report. |
| Training discs | 1036 |

| Estimator | Discs evaluated | QWK | MAE | Exact agreement | Within 1 grade |
| --- | --- | --- | --- | --- | --- |
| logreg | 210 | 0.60380 | 0.88100 | 0.37140 | 0.81430 |
| forest | 210 | 0.63170 | 0.77620 | 0.43810 | 0.83330 |

| Metric | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ | Verdict |
| --- | --- | --- | --- | --- |
| Pfirrmann logreg/quadratic_weighted_kappa | 0.61230 | 0.60380 | -0.00850 | worsened |
| Pfirrmann logreg/mae | 0.90000 | 0.88100 | -0.01900 | improved |
| Pfirrmann logreg/within_one_grade | 0.80000 | 0.81430 | +0.01430 | improved |
| Pfirrmann logreg/exact_agreement | 0.35710 | 0.37140 | +0.01430 | improved |
| Pfirrmann forest/quadratic_weighted_kappa | 0.63750 | 0.63170 | -0.00580 | worsened |
| Pfirrmann forest/mae | 0.80000 | 0.77620 | -0.02380 | improved |
| Pfirrmann forest/within_one_grade | 0.84290 | 0.83330 | -0.00960 | worsened |
| Pfirrmann forest/exact_agreement | 0.40480 | 0.43810 | +0.03330 | improved |


## 11. Visual Results

- `outputs/visualizations/sprint3_coverage/training_curves.png`
- `outputs/visualizations/sprint3_coverage/sprint2_vs_sprint3_curves.png`
- `outputs/visualizations/sprint3_coverage/test_metric_comparison.png`
- `outputs/visualizations/sprint3_coverage/indexing_comparison.png`
- `outputs/visualizations/sprint3_coverage/measurement_agreement.png`
- `outputs/visualizations/sprint3_coverage/segmentation_metrics.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_best_219_t1_s007.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_best_229_t1_s001.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_best_58_t1_s018.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_median_144_t2_s018.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_median_166_t1_s005.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_median_218_t2_s001.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_midsagittal_11_t2_s010.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_midsagittal_166_t2_SPACE_s060.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_midsagittal_171_t1_s008.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_worst_11_t1_s000.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_worst_11_t2_SPACE_s018.png`
- `outputs/visualizations/sprint3_coverage/predictions/pred_test_worst_144_t1_s023.png`


## 12. What Improved

| Metric | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ |
| --- | --- | --- | --- |
| test background/dice | 0.99514 | 0.99527 | +0.00013 |
| test background/iou | 0.99032 | 0.99058 | +0.00026 |
| test background/precision | 0.99509 | 0.99537 | +0.00028 |
| test vertebra/dice | 0.91332 | 0.91663 | +0.00331 |
| test vertebra/iou | 0.84047 | 0.84608 | +0.00562 |
| test vertebra/precision | 0.91777 | 0.91950 | +0.00172 |
| test vertebra/recall | 0.90891 | 0.91377 | +0.00486 |
| test intervertebral_disc/dice | 0.87790 | 0.87827 | +0.00037 |
| test intervertebral_disc/iou | 0.78238 | 0.78297 | +0.00059 |
| test intervertebral_disc/recall | 0.88483 | 0.89509 | +0.01026 |
| test spinal_canal/dice | 0.90256 | 0.90514 | +0.00258 |
| test spinal_canal/iou | 0.82242 | 0.82671 | +0.00429 |
| test spinal_canal/precision | 0.89258 | 0.89521 | +0.00263 |
| test spinal_canal/recall | 0.91277 | 0.91529 | +0.00252 |
| test macro_foreground/dice | 0.89793 | 0.90001 | +0.00208 |
| test macro_foreground/iou | 0.81509 | 0.81859 | +0.00350 |
| test macro_foreground/recall | 0.90217 | 0.90805 | +0.00588 |
| test pixel_accuracy | 0.99067 | 0.99092 | +0.00025 |
| test per_patient_dice/background | 0.99522 | 0.99531 | +0.00009 |
| test per_patient_dice_sd/background | 0.00103 | 0.00102 | -0.00001 |
| test per_patient_dice/vertebra | 0.91405 | 0.91637 | +0.00232 |
| test per_patient_dice_sd/vertebra | 0.01676 | 0.01572 | -0.00105 |
| test per_patient_dice/intervertebral_disc | 0.88678 | 0.88684 | +0.00006 |
| test per_patient_dice_sd/intervertebral_disc | 0.02991 | 0.02938 | -0.00053 |
| test per_patient_dice/spinal_canal | 0.90253 | 0.90418 | +0.00165 |
| indexing pct_index_correct | 82.25 | 84.08 | +1.83000 |
| indexing pct_region_found | 95.73 | 96.66 | +0.93000 |
| indexing pct_slices_count_correct | 75.53 | 77.16 | +1.63000 |
| indexing pct_slices_all_discs_correct | 75.11 | 77.95 | +2.84000 |
| indexing category %/correct | 82.25 | 84.08 | +1.83000 |
| indexing category %/shifted | 12.97 | 12.10 | -0.87000 |
| indexing category %/split | 0.51000 | 0.45000 | -0.06000 |
| indexing category %/missed | 4.27 | 3.34 | -0.93000 |
| MAE area_mm2 | 25.34 | 24.90 | -0.44450 |
| r disc_to_vertebra_height_ratio | 0.64670 | 0.67500 | +0.02830 |
| MAE height_mm_anterior | 0.87800 | 0.84250 | -0.03550 |
| MAE height_mm_central | 0.80650 | 0.76340 | -0.04310 |
| MAE height_mm_posterior | 0.75310 | 0.73990 | -0.01320 |
| r height_mm_posterior | 0.74780 | 0.76140 | +0.01360 |
| MAE height_ratio_to_series_median | 0.11860 | 0.11230 | -0.00630 |
| r height_ratio_to_series_median | 0.83560 | 0.84810 | +0.01250 |
| r intensity_disc_vertebra_ratio | 0.96380 | 0.96390 | +0.00010 |
| disc detection %truth recovered | 99.11 | 99.33 | +0.22000 |
| Pfirrmann logreg/mae | 0.90000 | 0.88100 | -0.01900 |
| Pfirrmann logreg/within_one_grade | 0.80000 | 0.81430 | +0.01430 |
| Pfirrmann logreg/exact_agreement | 0.35710 | 0.37140 | +0.01430 |
| Pfirrmann forest/mae | 0.80000 | 0.77620 | -0.02380 |
| Pfirrmann forest/exact_agreement | 0.40480 | 0.43810 | +0.03330 |


## 13. What Worsened

| Metric | Sprint 2 Extended | Sprint 3 Coverage | Absolute Δ |
| --- | --- | --- | --- |
| validation macro foreground Dice (best) | 0.89850 | 0.89831 | -0.00019 |
| test background/recall | 0.99518 | 0.99516 | -0.00002 |
| test intervertebral_disc/precision | 0.87109 | 0.86208 | -0.00901 |
| test macro_foreground/precision | 0.89381 | 0.89226 | -0.00155 |
| test per_patient_dice_sd/spinal_canal | 0.02251 | 0.02432 | +0.00181 |
| indexing total_spurious_components | 249 | 270 | +21.00000 |
| indexing pct_slices_with_spurious | 13.29 | 14.56 | +1.27000 |
| MAE ap_extent_mm | 1.99 | 2.00 | +0.00900 |
| r ap_extent_mm | 0.77660 | 0.77120 | -0.00540 |
| r area_mm2 | 0.90790 | 0.89460 | -0.01330 |
| MAE canal_width_at_disc_mm | 1.13 | 1.15 | +0.02150 |
| r canal_width_at_disc_mm | 0.81070 | 0.80650 | -0.00420 |
| MAE disc_to_vertebra_height_ratio | 0.13610 | 0.13720 | +0.00110 |
| r height_mm_anterior | 0.85350 | 0.84370 | -0.00980 |
| r height_mm_central | 0.88750 | 0.88700 | -0.00050 |
| MAE intensity_disc_vertebra_ratio | 0.03600 | 0.03640 | +0.00040 |
| Pfirrmann logreg/quadratic_weighted_kappa | 0.61230 | 0.60380 | -0.00850 |
| Pfirrmann forest/quadratic_weighted_kappa | 0.63750 | 0.63170 | -0.00580 |
| Pfirrmann forest/within_one_grade | 0.84290 | 0.83330 | -0.00960 |


## 14. Interpretation

**Question.** Was insufficient training-data coverage the main limitation of the previous 16-channel model?

**Verdict: not meaningfully supported (primary criterion tied).** Raising training-data coverage from 28.05% to 100.0% left the primary criterion effectively unchanged: validation macro foreground Dice moved -0.00019, which is smaller than the 0.001 min_delta the experiment was configured to treat as real. The secondary metrics do move consistently in the right direction - test macro Dice +0.00208, disc indexing +1.83 pp, and 45 of 63 compared metrics improved - but all of the segmentation gains are well inside the noise band of a single 33-patient test set. **The conclusion is that the 16-channel model was not meaningfully data-limited**: it reaches the same performance whether it sees 28% or 100% of the training slices. That makes model capacity, rather than data volume, the credible next variable.


### Primary criterion

| Field | Value |
| --- | --- |
| name | best validation macro foreground Dice |
| sprint2_extended | 0.89850 |
| sprint3_coverage | 0.89831 |
| absolute_delta | -0.000190 |
| exceeds_baseline | False |
| exceeds_by_meaningful_margin | False |
| meaningful_threshold | 0.00500 |
| statistically_tied | True |
| tie_threshold | 0.00100 |


### Supporting evidence

| Field | Value |
| --- | --- |
| test_macro_fg_dice_delta | 0.00208 |
| disc_indexing_delta_pp | 1.83 |
| pfirrmann_forest_qwk_delta | -0.00580 |
| metrics_improved | 45 |
| metrics_worsened | 18 |
| coverage_achieved_pct | 100.00 |
| coverage_vs_sprint2_pct | 28.05 |
| early_stopping_triggered | True |
| best_epoch | 29 |
| total_epochs | 29 |

The decision deliberately does not rest on a single metric. It weighs the primary validation criterion together with the test macro Dice, the per-class Dice values, disc indexing, the predicted-mask measurement errors, the end-to-end Pfirrmann agreement, and the convergence behaviour.

**One asymmetry worth stating plainly.** Sprint 3 trained from random initialisation, whereas Sprint 2 Extended continued from an already-converged 12-epoch model. Sprint 3 therefore had fewer effective optimisation steps on any given slice, and the comparison is not a perfectly matched pair. It is the correct comparison for the question - warm-starting would have confounded coverage with prior training - but the margin should be read with that in mind.


## 15. Limitations

- **CPU-only training.** No CUDA device is present. Width 16 and the per-epoch slice budget are compute-budget decisions, not modelling conclusions.
- **No confidence intervals.** Differences of a few thousandths of a Dice point on a single 33-patient test set should be read as a direction, not a significant difference. No bootstrap was run.
- **Random vs warm start.** See the asymmetry noted in section 14.
- **Disc identity is derived, not predicted.** The network outputs four semantic classes; disc index comes from ordering connected components per slice, which is why indexing accuracy trails detection accuracy.
- **Vertebra-normalised features remain unreliable from predicted masks.** The Sprint 3 audit established that a vertebra occupies ~1.73 connected components per instance in a sagittal plane (body plus posterior elements), so ordered components cannot identify individual vertebrae. This is a property of the anatomy, not of the model.
- **Level names are provisional.** The dataset documents that the lowest annotated vertebra is usually L5 but can be L4 or L6, so only integer disc indices are treated as authoritative.
- **Segmentation metrics are not clinical evidence.** A Dice score measures overlap with one annotation protocol. It does not establish clinical effectiveness, diagnostic accuracy or deployment readiness, and none is claimed.
- **No longitudinal scope.** The dataset contains exactly one study per patient, with no acquisition date, no repeat imaging and no surgical or outcome record. Postoperative healing and longitudinal change remain outside the validated scope of this project and are not claimed anywhere.


## 16. Recommended Next Experiment

**Option B: Controlled capacity experiment: 32-channel U-Net at batch size 2**

Full training-data coverage did not meaningfully improve the primary criterion - the 16-channel model reaches effectively the same validation Dice on 28% and on 100% of the training slices. It is therefore not data-limited at this configuration, and capacity becomes the credible next variable. The Sprint 3 audit measured width 32 at batch 2 as needing ~1.82 GB peak commit, which is *less* than the proven width-16/batch-8 configuration, so memory is not the obstacle. Time is: roughly 17 h for 30 rotating-coverage epochs.


### Proposed steps

- Width 32, batch size 2. Keep the rotating sampler (it costs nothing and removes coverage as a confound), plus the same loss, optimiser, augmentation, split and seed.
- 30 epochs, patience 6, min_delta 0.001, random initialisation.
- Budget ~17 h; new output directories; test set evaluated once at the end on the validation-selected checkpoint.
- Success criterion to fix in advance: validation macro foreground Dice above 0.8985 by a margin of at least 0.005.


### Orthogonal, and not part of this recommendation

- The disc-indexing post-processing work identified in the Sprint 3 audit remains the highest-value change that needs **no retraining**: series-level 3-D grouping was measured to recover 99.3% of discs against ~84% per slice. It is orthogonal to the capacity question and can proceed in parallel, but it is not the controlled experiment being recommended here.


### Explicitly not recommended

- Width 64 - measured at ~6.8 GB peak commit and 216-326 h for 30 epochs. No new measured reason to revisit it.
- Changing width together with batch size, loss, augmentation or any other variable in the same run.

**This experiment has not been started.** It is a recommendation only.


## 17. Artifact Locations

| artifact | path |
| --- | --- |
| Training checkpoints | `outputs/checkpoints/sprint3_coverage/` |
| Selected checkpoint | `outputs/checkpoints/sprint3_coverage/best_val_dice.pt` |
| Per-epoch history | `outputs/checkpoints/sprint3_coverage/history.csv` / `.json` |
| Training config / summary | `outputs/reports/sprint3_coverage/training_config.json`, `training_summary.json` |
| Test segmentation metrics | `outputs/reports/sprint3_coverage/metrics/` |
| Disc analysis (predicted masks) | `outputs/reports/disc_analysis_sprint3_coverage.csv` |
| Measurement agreement + Pfirrmann | `outputs/reports/sprint3_coverage/metrics/measurement_agreement.json` |
| Indexing failure detail | `outputs/reports/sprint3_coverage/indexing_failures_sprint3_coverage.csv` |
| Comparison table | `outputs/reports/sprint3_coverage/comparison_table.csv` |
| Figures | `outputs/visualizations/sprint3_coverage/` |
| This report | `outputs/reports/sprint3_coverage/sprint3_final_report.md` |
| Sprint 2 Extended baseline (unmodified) | `outputs/reports/sprint2_extended/`, `outputs/checkpoints/sprint2_extended/` |

