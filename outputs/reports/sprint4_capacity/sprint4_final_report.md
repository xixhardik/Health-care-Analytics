# Sprint 4 Capacity - Final Report

*Controlled experiment: U-Net width 16 -> 32 after coverage was fixed*

Generated: 2026-09-24 19:22:55


## 1. Objective

Answer one question: **does increasing U-Net capacity from width 16 to width 32 provide a meaningful improvement after training-data coverage has already been fixed?**

Sprint 3 removed the data-coverage confound: the width-16 model was trained on 100% of the 9,128 training slices instead of 28.05%, and its validation Dice did not move. That left model capacity as the next credible variable, and the Sprint 3 audit had already measured width 32 at batch 2 as the only affordable way to test it on this hardware. This sprint runs that test.


## 2. Experimental Setup


### Changed

| Field | Value |
| --- | --- |
| Base channels (width) | 16 -> 32 |
| Parameters | 1,963,860 -> 7,849,124 (3.997x) |
| Batch size | 8 -> 2 (co-varying dependency, not an independent choice) |

The batch size is **not** an independent choice. Width 32 at batch 8 was measured in the Sprint 3 audit at ~3,834 MB peak commit (1.5x the memory-proven level) and 80 h for 30 full-coverage epochs. At batch 2 it needs ~1,823 MB, below the proven width-16/batch-8 level of ~2,534 MB, and ~16.9 h for 30 rotating epochs. Batch 2 is what makes width 32 runnable on this hardware.


### Unchanged from Sprint 3 Coverage

- `topology`: plain U-Net, BatchNorm, bilinear upsampling
- `depth`: 4
- `bilinear`: True
- `sampler`: rotating shard sampler
- `slices_per_epoch`: 2284
- `coverage_pct_over_run`: 100.0
- `epochs_to_full_coverage`: 4
- `optimiser`: Adam
- `lr`: 0.001
- `scheduler`: CosineAnnealingLR(T_max=30)
- `loss`: CrossEntropy + soft Dice (equal weight)
- `augmentation`: translate +/-5% + intensity jitter, train only
- `preprocessing`: Sprint 1 slices, 352x256 at 1.0 mm/px
- `seed`: 42
- `split`: Sprint 1 patient-level
- `validation_slices`: 800
- `n_classes`: 4
- `input_size`: [352, 256]
- `memory_format`: channels_last
- `bf16_autocast`: False

Early stopping: monitor `val_fg_dice`, patience 6, min_delta 0.001 - identical to Sprint 3.


### Different by design

- **initialisation**: random. NOT resumed from Sprint 3, NOT warm-started from Sprint 2 - the layer shapes differ, and warm-starting would confound the capacity variable.

- **validation_batch_size**: 2 rather than Sprint 3's 8. Verified in the smoke test to leave the validation Dice bit-identical: BatchNorm runs in eval mode and the confusion matrix accumulates over the full 800 slices, so the metric stays directly comparable to Sprint 3's 0.89831.


### Feasibility check before the long run

A short smoke test was run first and had to pass before training started. Verdict **PASS** (25 checks passed, 0 failed).

| Field | Value |
| --- | --- |
| Parameter count verified | 7,849,124 (3.997x Sprint 3) |
| Loss finite on first step | 2.19 |
| Global gradient L2 norm | 3.45 |
| Measured train throughput | 1.273 img/s (1.5716 s/step at batch 2) |
| Projected 30-epoch time | 16.4 h (audit estimate 16.9 h) |
| Peak commit charge | 1931.4 MB = 0.757x the memory-proven level |
| Validation Dice invariant to batch size | batch 2 = batch 8 to 0.0e+00 |
| CUDA assumed | False |
| Test set touched | False |


### Verification that the run completed

| Field | Value |
| --- | --- |
| All training artefacts present | True |
| Epochs in history.csv | 30 |
| Epochs reported by the summary | 30 |
| History matches summary | True |
| Early stopping triggered | False |
| Stop reason | reached the maximum of 30 epochs without triggering early stopping |
| Test set used during training | False |


## 3. Architecture Comparison

Same topology in both runs - plain U-Net, depth 4, BatchNorm, bilinear upsampling, 4 output classes. Only the channel widths differ, and each encoder stage doubles from the base.

| Property | Sprint 3 (width 16) | Sprint 4 (width 32) |
| --- | --- | --- |
| base channels | 16 | 32 |
| encoder stage widths | 16, 32, 64, 128, 256 | 32, 64, 128, 256, 512 |
| depth | 4 | 4 |
| parameters | 1,963,860 | 7,849,124 |
| parameter ratio | 1.00x | 4.00x |
| batch size | 8 | 2 |
| checkpoint size | 7.9 MB (weights-only) | 31.5 MB (weights-only) |

Parameters scale about 4x per width doubling, as expected for a convolutional encoder-decoder: each layer's weight tensor grows with the product of input and output channels.


## 4. Training Results

| Field | Value |
| --- | --- |
| Total epochs completed | 30 / 30 |
| Early stopping triggered | False |
| Stop reason | reached the maximum of 30 epochs without triggering early stopping |
| Epochs without improvement at end | 4 |
| Best validation foreground Dice | 0.89366 (epoch 26) |
| Best validation loss | 0.15270 (epoch 29) |
| Final validation Dice | 0.89178 |
| Final validation loss | 0.15384 |
| Learning rate at best epoch | 6.699e-05 |
| Train loss at best epoch | 0.13612 |
| Validation loss at best epoch | 0.15591 |
| Per-class Dice at best epoch | vertebra 0.90640, IVD 0.87340, canal 0.90119 |

The run used its full 30-epoch budget: early stopping did **not** fire, and the counter stood at 4 of the patience-6 limit when the budget ran out. Sprint 3 by contrast stopped early at epoch 29.


### Checkpoint selection

| Field | Value |
| --- | --- |
| Checkpoint | `outputs/checkpoints/sprint4_capacity/best_val_dice.pt` |
| Selection criterion | best validation foreground Dice (predefined) |
| Epoch | 26 |
| Validation foreground Dice | 0.89366 |
| Validation loss | 0.15591 |
| Matches the training summary | True |
| Test performance used for selection | False |


### Per-epoch history

| epoch | learning_rate | slices | seen | cov% | train_loss | val_loss | val_fg_dice | vert | ivd | canal | sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00100 | 2327 | 2327 | 25.49 | 0.50056 | 0.59977 | 0.64999 | 0.63750 | 0.73107 | 0.58141 | 2092.00 |
| 2 | 0.000997 | 2314 | 4641 | 50.84 | 0.32598 | 0.37058 | 0.73760 | 0.75781 | 0.75005 | 0.70495 | 1971.50 |
| 3 | 0.000989 | 2253 | 6894 | 75.53 | 0.25236 | 0.31184 | 0.76411 | 0.82772 | 0.80830 | 0.65630 | 1918.60 |
| 4 | 0.000976 | 2234 | 9128 | 100.00 | 0.23702 | 0.21442 | 0.84798 | 0.86975 | 0.84897 | 0.82522 | 1946.80 |
| 5 | 0.000957 | 2327 | 9128 | 100.00 | 0.19800 | 0.20594 | 0.86419 | 0.88071 | 0.85584 | 0.85602 | 1977.90 |
| 6 | 0.000933 | 2314 | 9128 | 100.00 | 0.20535 | 0.20770 | 0.85214 | 0.87354 | 0.85020 | 0.83269 | 1942.70 |
| 7 | 0.000905 | 2253 | 9128 | 100.00 | 0.19882 | 0.20369 | 0.85828 | 0.87639 | 0.85313 | 0.84533 | 1956.70 |
| 8 | 0.000872 | 2234 | 9128 | 100.00 | 0.17830 | 0.18901 | 0.87135 | 0.88237 | 0.86500 | 0.86667 | 1901.00 |
| 9 | 0.000835 | 2327 | 9128 | 100.00 | 0.17477 | 0.18686 | 0.87465 | 0.89101 | 0.86358 | 0.86938 | 1972.30 |
| 10 | 0.000794 | 2314 | 9128 | 100.00 | 0.18934 | 0.19268 | 0.87172 | 0.88769 | 0.85893 | 0.86853 | 1953.00 |
| 11 | 0.000750 | 2253 | 9128 | 100.00 | 0.16922 | 0.19711 | 0.86983 | 0.88121 | 0.86010 | 0.86817 | 1893.00 |
| 12 | 0.000703 | 2234 | 9128 | 100.00 | 0.17047 | 0.17780 | 0.87933 | 0.89311 | 0.86638 | 0.87851 | 1872.10 |
| 13 | 0.000655 | 2327 | 9128 | 100.00 | 0.15577 | 0.17886 | 0.88181 | 0.89681 | 0.86856 | 0.88006 | 1950.10 |
| 14 | 0.000604 | 2314 | 9128 | 100.00 | 0.15803 | 0.17094 | 0.88359 | 0.89938 | 0.86121 | 0.89019 | 1955.80 |
| 15 | 0.000552 | 2253 | 9128 | 100.00 | 0.15729 | 0.17296 | 0.87893 | 0.89610 | 0.86466 | 0.87604 | 1948.50 |
| 16 | 0.000500 | 2234 | 9128 | 100.00 | 0.15953 | 0.17698 | 0.88105 | 0.89768 | 0.86277 | 0.88269 | 1895.40 |
| 17 | 0.000448 | 2327 | 9128 | 100.00 | 0.15195 | 0.17405 | 0.88512 | 0.90263 | 0.87098 | 0.88176 | 2027.30 |
| 18 | 0.000396 | 2314 | 9128 | 100.00 | 0.15278 | 0.16975 | 0.88108 | 0.89847 | 0.86336 | 0.88141 | 1983.60 |
| 19 | 0.000345 | 2253 | 9128 | 100.00 | 0.14728 | 0.15996 | 0.88900 | 0.90237 | 0.87073 | 0.89389 | 1964.40 |
| 20 | 0.000297 | 2234 | 9128 | 100.00 | 0.14462 | 0.16453 | 0.88787 | 0.90233 | 0.87199 | 0.88929 | 1964.60 |
| 21 | 0.000250 | 2327 | 9128 | 100.00 | 0.13885 | 0.16103 | 0.88851 | 0.90326 | 0.87294 | 0.88933 | 2035.90 |
| 22 | 0.000206 | 2314 | 9128 | 100.00 | 0.14180 | 0.16189 | 0.88920 | 0.90396 | 0.87092 | 0.89271 | 1988.40 |
| 23 | 0.000165 | 2253 | 9128 | 100.00 | 0.13758 | 0.16240 | 0.88539 | 0.90361 | 0.86786 | 0.88469 | 1937.90 |
| 24 | 0.000128 | 2234 | 9128 | 100.00 | 0.13578 | 0.16019 | 0.89009 | 0.90513 | 0.87271 | 0.89242 | 1902.80 |
| 25 | 0.000095 | 2327 | 9128 | 100.00 | 0.13923 | 0.15743 | 0.89163 | 0.90545 | 0.87287 | 0.89657 | 1951.10 |
| 26 | 0.000067 | 2314 | 9128 | 100.00 | 0.13612 | 0.15591 | 0.89366 | 0.90640 | 0.87340 | 0.90119 | 2006.20 |
| 27 | 0.000043 | 2253 | 9128 | 100.00 | 0.13157 | 0.15287 | 0.89287 | 0.90581 | 0.87310 | 0.89969 | 1931.30 |
| 28 | 0.000024 | 2234 | 9128 | 100.00 | 0.13390 | 0.15297 | 0.89226 | 0.90554 | 0.87240 | 0.89883 | 2202.40 |
| 29 | 0.000011 | 2327 | 9128 | 100.00 | 0.13145 | 0.15270 | 0.89209 | 0.90624 | 0.87266 | 0.89737 | 2286.20 |
| 30 | 0.000003 | 2314 | 9128 | 100.00 | 0.13380 | 0.15384 | 0.89178 | 0.90536 | 0.87249 | 0.89749 | 1950.80 |


### Training-data coverage

| Field | Value |
| --- | --- |
| Training slices available | 9,128 |
| Unique slices seen | 9,128 |
| Final coverage | 100.0% |
| Epoch reaching 100% coverage | 4 |
| Mean slices per epoch | 2,285 |
| Sampler leakage-free | True |
| Sampler violations | 0 |
| Validation/test patients excluded from training | 66 |

Coverage is identical to Sprint 3 by construction - the same rotating shard sampler with the same seed and the same per-epoch budget - so data exposure is held constant and cannot explain the difference between the two runs.


## 5. Convergence

| Field | Value |
| --- | --- |
| Best epoch | 26 |
| Epochs after the best | 4 |
| Validation Dice over the last 5 epochs | 0.89178 - 0.89366 (spread 0.00188) |
| Train loss over the last 5 epochs | 0.13145 - 0.13612 |
| Final train loss | 0.13380 |
| Final validation loss | 0.15384 |
| Validation - train loss gap | 0.02004 |
| Minimum validation loss | 0.15270 (epoch 29) |

The curve is flat at the end: validation Dice moves only 0.00188 across the last five epochs, and the best epoch is 4 epochs before the end. The larger model has converged - it has not been cut short. The 0.02004 gap between validation and training loss is the margin to watch: a 4x larger model on the same 9,128 slices has more room to fit the training split, and the validation metric does not follow the training loss down.


## 6. Test Segmentation Results

Evaluated **once**, after training finished, on the untouched held-out test set: 1,655 slices from 33 patients. The checkpoint was selected on validation Dice, not on test performance.


### Aggregate (dataset-level) metrics

| class | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- |
| background | 0.99503 | 0.99011 | 0.99452 | 0.99554 |
| vertebra | 0.91113 | 0.83676 | 0.92498 | 0.89768 |
| intervertebral_disc | 0.87313 | 0.77482 | 0.86835 | 0.87796 |
| spinal_canal | 0.89755 | 0.81414 | 0.89589 | 0.89922 |
| **macro foreground** | 0.89393 | 0.80858 | 0.89640 | 0.89162 |

| Field | Value |
| --- | --- |
| Pixel accuracy | 0.99046 |


### Per-patient Dice (mean +/- sd over test patients)

| class | mean Dice | sd | patients |
| --- | --- | --- | --- |
| background | 0.99507 | 0.00113 | 33 |
| vertebra | 0.91106 | 0.01963 | 33 |
| intervertebral_disc | 0.88156 | 0.03106 | 33 |
| spinal_canal | 0.89608 | 0.02277 | 33 |

Note on aggregation: these are dataset-level metrics computed from a pooled confusion matrix, which is the same aggregation Sprint 2 and Sprint 3 reported. The evaluation script also prints a per-slice mean, which is systematically lower for every run because lateral slices holding only small structure fragments get equal weight. The two are not interchangeable and only the aggregate values are compared here.


## 7. Sprint 2 vs Sprint 3 vs Sprint 4

`Δ S4 vs S3` is Sprint 4 minus Sprint 3 Coverage, the immediate predecessor and the only run that differs from Sprint 4 by capacity alone. Verdicts account for metric direction - lower is better for MAE, loss, standard deviation and spurious components.


### Primary criterion

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| validation macro foreground Dice (best) | 0.89850 | 0.89831 | 0.89366 | -0.00465 | worsened |


### Headline metrics

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| test vertebra/dice | 0.91332 | 0.91663 | 0.91113 | -0.00550 | worsened |
| test intervertebral_disc/dice | 0.87790 | 0.87827 | 0.87313 | -0.00515 | worsened |
| test spinal_canal/dice | 0.90256 | 0.90514 | 0.89755 | -0.00759 | worsened |
| test macro_foreground/dice | 0.89793 | 0.90001 | 0.89393 | -0.00608 | worsened |
| indexing pct_index_correct | 82.25 | 84.08 | 82.70 | -1.38000 | worsened |
| disc_identification pct_discs_index_correct | 82.60 | 84.42 | 83.01 | -1.41000 | worsened |
| height_mm_central/mae | 0.80650 | 0.76340 | 0.86580 | +0.10240 | worsened |
| intensity_disc_vertebra_ratio/pearson_r | 0.96380 | 0.96390 | 0.95630 | -0.00760 | worsened |
| Pfirrmann forest/quadratic_weighted_kappa | 0.63750 | 0.63170 | 0.64310 | +0.01140 | improved |


### All segmentation metrics

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| test background/dice | 0.99514 | 0.99527 | 0.99503 | -0.00024 | worsened |
| test background/iou | 0.99032 | 0.99058 | 0.99011 | -0.00047 | worsened |
| test background/precision | 0.99509 | 0.99537 | 0.99452 | -0.00085 | worsened |
| test background/recall | 0.99518 | 0.99516 | 0.99554 | +0.00038 | improved |
| test vertebra/dice | 0.91332 | 0.91663 | 0.91113 | -0.00550 | worsened |
| test vertebra/iou | 0.84047 | 0.84608 | 0.83676 | -0.00932 | worsened |
| test vertebra/precision | 0.91777 | 0.91950 | 0.92498 | +0.00548 | improved |
| test vertebra/recall | 0.90891 | 0.91377 | 0.89768 | -0.01609 | worsened |
| test intervertebral_disc/dice | 0.87790 | 0.87827 | 0.87313 | -0.00515 | worsened |
| test intervertebral_disc/iou | 0.78238 | 0.78297 | 0.77482 | -0.00814 | worsened |
| test intervertebral_disc/precision | 0.87109 | 0.86208 | 0.86835 | +0.00626 | improved |
| test intervertebral_disc/recall | 0.88483 | 0.89509 | 0.87796 | -0.01713 | worsened |
| test spinal_canal/dice | 0.90256 | 0.90514 | 0.89755 | -0.00759 | worsened |
| test spinal_canal/iou | 0.82242 | 0.82671 | 0.81414 | -0.01257 | worsened |
| test spinal_canal/precision | 0.89258 | 0.89521 | 0.89589 | +0.00068 | improved |
| test spinal_canal/recall | 0.91277 | 0.91529 | 0.89922 | -0.01607 | worsened |
| test macro_foreground/dice | 0.89793 | 0.90001 | 0.89393 | -0.00608 | worsened |
| test macro_foreground/iou | 0.81509 | 0.81859 | 0.80858 | -0.01001 | worsened |
| test macro_foreground/precision | 0.89381 | 0.89226 | 0.89640 | +0.00414 | improved |
| test macro_foreground/recall | 0.90217 | 0.90805 | 0.89162 | -0.01643 | worsened |
| test pixel_accuracy | 0.99067 | 0.99092 | 0.99046 | -0.00046 | worsened |
| test per_patient_dice/background | 0.99522 | 0.99531 | 0.99507 | -0.00024 | worsened |
| test per_patient_dice_sd/background | 0.00103 | 0.00102 | 0.00113 | +0.00011 | worsened |
| test per_patient_dice/vertebra | 0.91405 | 0.91637 | 0.91106 | -0.00531 | worsened |
| test per_patient_dice_sd/vertebra | 0.01676 | 0.01572 | 0.01963 | +0.00391 | worsened |
| test per_patient_dice/intervertebral_disc | 0.88678 | 0.88684 | 0.88156 | -0.00528 | worsened |
| test per_patient_dice_sd/intervertebral_disc | 0.02991 | 0.02938 | 0.03106 | +0.00168 | worsened |
| test per_patient_dice/spinal_canal | 0.90253 | 0.90418 | 0.89608 | -0.00809 | worsened |
| test per_patient_dice_sd/spinal_canal | 0.02251 | 0.02432 | 0.02277 | -0.00155 | improved |


### Tally of Sprint 4 against Sprint 3

| Field | Value |
| --- | --- |
| worsened | 61 |
| improved | 18 |
| unchanged | 1 |


## 8. Disc Indexing

Corrected taxonomy: every ground-truth disc is classified as `correct`, `shifted`, `merged`, `split` or `missed`. Analysis uses the **integer disc index only** - the dataset does not state which vertebra is L5, so no anatomical level name is asserted. The invalid vertebra-component-vs-instance comparison identified in the Sprint 3 audit is not used.

| Field | Value |
| --- | --- |
| Discs analysed | 6,821 |
| Slices analysed | 1,655 |
| Region found | 95.4% |
| Index correct | 82.7% |
| Slices with correct disc count | 75.77% |
| Slices with every disc correct | 75.11% |
| Spurious components | 233 |
| Shift offsets | `-5.0`=4, `-4.0`=22, `-3.0`=22, `-2.0`=106, `-1.0`=243, `1.0`=412, `2.0`=22 |

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| indexing pct_index_correct | 82.25 | 84.08 | 82.70 | -1.38000 | worsened |
| indexing pct_region_found | 95.73 | 96.66 | 95.40 | -1.26000 | worsened |
| indexing pct_slices_count_correct | 75.53 | 77.16 | 75.77 | -1.39000 | worsened |
| indexing pct_slices_all_discs_correct | 75.11 | 77.95 | 75.11 | -2.84000 | worsened |
| indexing total_spurious_components | 249 | 270 | 233 | -37 | improved |
| indexing pct_slices_with_spurious | 13.29 | 14.56 | 12.69 | -1.87000 | improved |
| indexing category %/correct | 82.25 | 84.08 | 82.70 | -1.38000 | worsened |
| indexing category %/shifted | 12.97 | 12.10 | 12.18 | +0.08000 | worsened |
| indexing category %/merged | 0.00000 | 0.03000 | 0.00000 | -0.03000 | improved |
| indexing category %/split | 0.51000 | 0.45000 | 0.51000 | +0.06000 | worsened |
| indexing category %/missed | 4.27 | 3.34 | 4.60 | +1.26000 | worsened |


### Second estimator (evaluate_unet)

The evaluation script computes indexing accuracy with a different estimator from the corrected taxonomy. Both are listed so the comparison is never made across estimators.

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| disc_identification pct_discs_index_correct | 82.60 | 84.42 | 83.01 | -1.41000 | worsened |
| disc_identification pct_discs_region_found | 95.68 | 96.66 | 95.40 | -1.26000 | worsened |
| disc_identification pct_slices_with_matching_disc_count | 75.71 | 77.16 | 75.77 | -1.39000 | worsened |

Both estimators agree in direction: width 32 indexes discs less accurately than width 16.


### Accuracy by disc index

| truth_index | n | correct | shifted | merged | split | missed | pct_correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1028 | 923 | 31 | 0 | 32 | 42 | 89.79 |
| 2 | 1072 | 930 | 108 | 0 | 3 | 31 | 86.75 |
| 3 | 1048 | 866 | 143 | 0 | 0 | 39 | 82.63 |
| 4 | 994 | 790 | 153 | 0 | 0 | 51 | 79.48 |
| 5 | 925 | 742 | 138 | 0 | 0 | 45 | 80.22 |
| 6 | 854 | 708 | 123 | 0 | 0 | 23 | 82.90 |
| 7 | 583 | 466 | 92 | 0 | 0 | 25 | 79.93 |
| 8 | 269 | 179 | 36 | 0 | 0 | 54 | 66.54 |
| 9 | 48 | 37 | 7 | 0 | 0 | 4 | 77.08 |


### Accuracy by slice annotation area

| area_band | n | pct_correct | pct_shifted | pct_missed |
| --- | --- | --- | --- | --- |
| <1k | 135 | 31.85 | 17.04 | 51.11 |
| 1k-3k | 587 | 56.56 | 25.89 | 17.38 |
| 3k-6k | 1327 | 74.98 | 18.61 | 5.95 |
| 6k-10k | 3401 | 89.41 | 8.53 | 1.53 |
| >10k | 1371 | 89.72 | 8.68 | 0.88000 |

The structural gap remains the story: the model finds the disc region for 95.4% of ground-truth discs but assigns the right integer index for only 82.7%. The difference is ordering and fragmentation of regions that were already detected, which is a post-processing problem rather than a segmentation-quality one.


## 9. Disc-Level Measurements

Measurements taken from predicted masks, compared against the same discs measured from ground-truth masks. Only discs whose identity the prediction recovered correctly are compared - detection failures are counted separately rather than averaged into the measurement error.

| Field | Value |
| --- | --- |
| Discs compared | 445 |
| Ground-truth discs on predicted series | 447 |
| % of truth discs recovered | 99.55% |


### Sprint 4 agreement detail

| measurement | unit | n | gt_mean | bias | mae | mae_pct_of_gt_mean | pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| height_mm_central | mm | 445 | 7.05 | 0.51840 | 0.86580 | 12.29 | 0.88510 |
| height_mm_anterior | mm | 445 | 6.52 | 0.31210 | 0.97590 | 14.97 | 0.83300 |
| height_mm_posterior | mm | 445 | 4.85 | 0.34180 | 0.73990 | 15.25 | 0.79130 |
| area_mm2 | mm2 | 445 | 212.87 | 17.78 | 28.46 | 13.37 | 0.90780 |
| ap_extent_mm | mm | 445 | 33.68 | 0.72130 | 2.17 | 6.44 | 0.75760 |
| height_ratio_to_series_median | ratio | 445 | 1.04 | 0.00320 | 0.11920 | 11.50 | 0.83730 |
| disc_to_vertebra_height_ratio | ratio | 445 | 0.19080 | 0.13650 | 0.13840 | 72.51 | 0.63330 |
| canal_width_at_disc_mm | mm | 443 | 11.15 | 0.04340 | 1.17 | 10.46 | 0.78270 |
| intensity_disc_vertebra_ratio | ratio | 445 | 0.39580 | -0.00780 | 0.03840 | 9.70 | 0.95630 |


### Three-way comparison

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| height_mm_central/mae | 0.80650 | 0.76340 | 0.86580 | +0.10240 | worsened |
| height_mm_central/pearson_r | 0.88750 | 0.88700 | 0.88510 | -0.00190 | worsened |
| height_mm_central/mae_pct_of_gt_mean | 11.42 | 10.82 | 12.29 | +1.47000 | worsened |
| height_mm_anterior/mae | 0.87800 | 0.84250 | 0.97590 | +0.13340 | worsened |
| height_mm_anterior/pearson_r | 0.85350 | 0.84370 | 0.83300 | -0.01070 | worsened |
| height_mm_anterior/mae_pct_of_gt_mean | 13.44 | 12.91 | 14.97 | +2.06000 | worsened |
| height_mm_posterior/mae | 0.75310 | 0.73990 | 0.73990 | +0.00000 | unchanged |
| height_mm_posterior/pearson_r | 0.74780 | 0.76140 | 0.79130 | +0.02990 | improved |
| height_mm_posterior/mae_pct_of_gt_mean | 15.50 | 15.23 | 15.25 | +0.02000 | worsened |
| area_mm2/mae | 25.34 | 24.90 | 28.46 | +3.56430 | worsened |
| area_mm2/pearson_r | 0.90790 | 0.89460 | 0.90780 | +0.01320 | improved |
| area_mm2/mae_pct_of_gt_mean | 11.87 | 11.68 | 13.37 | +1.69000 | worsened |
| ap_extent_mm/mae | 1.99 | 2.00 | 2.17 | +0.17300 | worsened |
| ap_extent_mm/pearson_r | 0.77660 | 0.77120 | 0.75760 | -0.01360 | worsened |
| ap_extent_mm/mae_pct_of_gt_mean | 5.89 | 5.92 | 6.44 | +0.52000 | worsened |
| height_ratio_to_series_median/mae | 0.11860 | 0.11230 | 0.11920 | +0.00690 | worsened |
| height_ratio_to_series_median/pearson_r | 0.83560 | 0.84810 | 0.83730 | -0.01080 | worsened |
| height_ratio_to_series_median/mae_pct_of_gt_mean | 11.42 | 10.82 | 11.50 | +0.68000 | worsened |
| disc_to_vertebra_height_ratio/mae | 0.13610 | 0.13720 | 0.13840 | +0.00120 | worsened |
| disc_to_vertebra_height_ratio/pearson_r | 0.64670 | 0.67500 | 0.63330 | -0.04170 | worsened |
| disc_to_vertebra_height_ratio/mae_pct_of_gt_mean | 71.29 | 71.87 | 72.51 | +0.64000 | worsened |
| canal_width_at_disc_mm/mae | 1.13 | 1.15 | 1.17 | +0.01520 | worsened |
| canal_width_at_disc_mm/pearson_r | 0.81070 | 0.80650 | 0.78270 | -0.02380 | worsened |
| canal_width_at_disc_mm/mae_pct_of_gt_mean | 10.13 | 10.33 | 10.46 | +0.13000 | worsened |
| intensity_disc_vertebra_ratio/mae | 0.03600 | 0.03640 | 0.03840 | +0.00200 | worsened |
| intensity_disc_vertebra_ratio/pearson_r | 0.96380 | 0.96390 | 0.95630 | -0.00760 | worsened |
| intensity_disc_vertebra_ratio/mae_pct_of_gt_mean | 9.09 | 9.20 | 9.70 | +0.50000 | worsened |
| detection % of truth discs recovered | 99.11 | 99.33 | 99.55 | +0.22000 | improved |

`disc_to_vertebra_height_ratio` remains unusable in all three runs - its mean absolute error is over 70% of the ground-truth mean - so it is reported but must not be used as a derived feature.


## 10. Pfirrmann Results

End-to-end evaluation: the grade models are fitted on ground-truth-mask features from **training** patients, then applied to **predicted**-mask features from test patients, so segmentation error is included. Pfirrmann grades are dataset annotations used as labels for this measurement exercise. Nothing here is a clinical grading of a patient.

| Field | Value |
| --- | --- |
| Features | geometric+intensity |
| Training discs (ground-truth masks) | 1036 |
| Test discs evaluated (predicted masks) | 212 |

| estimator | QWK | MAE | exact agreement | within 1 grade |
| --- | --- | --- | --- | --- |
| logreg | 0.62460 | 0.87740 | 0.35850 | 0.82080 |
| forest | 0.64310 | 0.76420 | 0.43400 | 0.85380 |


### Three-way comparison

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| Pfirrmann logreg/quadratic_weighted_kappa | 0.61230 | 0.60380 | 0.62460 | +0.02080 | improved |
| Pfirrmann logreg/mae | 0.90000 | 0.88100 | 0.87740 | -0.00360 | improved |
| Pfirrmann logreg/within_one_grade | 0.80000 | 0.81430 | 0.82080 | +0.00650 | improved |
| Pfirrmann logreg/exact_agreement | 0.35710 | 0.37140 | 0.35850 | -0.01290 | worsened |
| Pfirrmann forest/quadratic_weighted_kappa | 0.63750 | 0.63170 | 0.64310 | +0.01140 | improved |
| Pfirrmann forest/mae | 0.80000 | 0.77620 | 0.76420 | -0.01200 | improved |
| Pfirrmann forest/within_one_grade | 0.84290 | 0.83330 | 0.85380 | +0.02050 | improved |
| Pfirrmann forest/exact_agreement | 0.40480 | 0.43810 | 0.43400 | -0.00410 | worsened |

This is the one family where Sprint 4 comes out ahead. It is reported as measured, and it is also the weakest evidence in the report: it rests on roughly 212 test discs, quadratic weighted kappa on that sample size moves easily, and the downstream grade model is refitted for each run. It is not enough to offset a consistent segmentation and indexing regression.


## 11. Computational Cost

| Metric | Sprint 2 Extended | Sprint 3 | Sprint 4 | Δ S4 vs S3 | Verdict |
| --- | --- | --- | --- | --- | --- |
| parameters | 1963860 | 1963860 | 7849124 | +5,885,264 | worsened |
| mean epoch seconds | - | 730.20 | 1976.00 | +1245.80000 | worsened |
| total wall hours | - | 5.88 | 16.47 | +10.59000 | worsened |

| Field | Value |
| --- | --- |
| Sprint 3 mean epoch | 730.2 s |
| Sprint 4 mean epoch | 1976.0 s (2.71x) |
| Sprint 4 epoch range | 1872.1 - 2286.2 s |
| Sprint 3 total wall time | 5.88 h (29 epochs) |
| Sprint 4 total wall time | 16.47 h (30 epochs) |
| Measured train throughput (smoke test) | 1.273 img/s at batch 2 |
| Peak commit charge | 1931.4 MB (0.757x the proven level) |
| Checkpoint size | 31.5 MB per weights-only file |

The capacity increase cost 2.71x the time per epoch and 2.80x the total wall time, and it bought a negative change in the primary metric. Memory was never the limit: peak commit charge stayed at 0.757x the proven level because the batch size dropped to 2. Time was the cost, exactly as the Sprint 3 audit predicted.


## 12. Limitations

- **Batch size is a confound.** Width 32 could not be run at batch 8 on this hardware within a sane time budget, so batch size dropped from 8 to 2 alongside the width change. Batch size affects BatchNorm statistics and gradient noise at a fixed learning rate. This experiment therefore measures 'width 32 as it can actually be trained here', not width 32 in isolation. A cleaner separation would need either batch 8 at width 32 (measured at ~80 h) or batch 2 at width 16 as a control (not run).
- **Learning rate was not re-tuned.** The learning rate was held at 1e-3 to keep the comparison controlled, but the optimal learning rate generally shifts with batch size. A tuned width-32 run could do better than this one; that possibility is not excluded by this result.
- **One seed, one split.** Every number here is a single run at seed 42 on one patient-level split. Differences of a few thousandths of a Dice point are within the range that seed choice alone can produce, which is why the decision rule was set at a margin rather than at zero.
- **Test set is 33 patients.** 1,655 slices from 33 patients is a small held-out set. It was evaluated once, after training, but it still gives wide uncertainty on any single metric.
- **Pfirrmann evaluation rests on ~212 discs**, and the grade model is refitted per run, so its quadratic weighted kappa is the least stable number in the report.
- **Segmentation metrics are not clinical evidence.** A Dice score measures overlap with one annotation protocol. It does not establish clinical effectiveness or readiness for medical use, and none is claimed.
- **Disc identity is derived, not predicted.** The network outputs four semantic classes; the disc index comes from ordering connected components, which is why indexing accuracy trails detection accuracy.
- **No longitudinal or postoperative scope.** The SPIDER dataset used here contains no longitudinal postoperative follow-up, so postoperative healing is outside the validated scope of this work and no claim about it is made or supported. No change over time is measured, and no 'percentage spine damage' score exists in this pipeline.


## 13. Interpretation

**Verdict: no improvement - width 32 is measurably worse.** Quadrupling the parameter count from 1,963,860 to 7,849,124 moved the primary criterion the wrong way: best validation foreground Dice went 0.89831 -> 0.89366 (-0.00465), a drop 4.6x larger than the 0.001 min_delta the experiment was configured to treat as real. The held-out test set agrees rather than contradicting: macro foreground Dice -0.00608, and every foreground class is down (vertebra -0.00550, IVD -0.00515, canal -0.00759). Disc indexing falls too (-1.38 pp on the corrected taxonomy, -1.41 pp on the evaluate_unet estimator). Across all compared quality metrics 18 improved and 61 worsened. The exception worth naming is the end-to-end Pfirrmann agreement, which rose +0.0114 QWK - but that is measured on ~212 discs and is not enough to offset a consistent segmentation regression. All of this cost 2.71x the time per epoch (16.47 h against 5.88 h). **Increasing U-Net width is therefore not the lever that improves this pipeline.**


### Against the predefined decision rule

| Field | Value |
| --- | --- |
| Question | Does increasing U-Net capacity from width 16 to width 32 provide a meaningful improvement after training-data coverage has already been fixed? |
| Answer | No. |
| Primary criterion | best validation foreground Dice |
| min_delta on validation Dice | 0.00100 |
| Material margin on test macro Dice | 0.00500 |
| Verdict | no improvement - width 32 is measurably worse |
| Hypothesis supported | False |


### Evidence weighed

| Field | Value |
| --- | --- |
| val_fg_dice_delta | -0.00465 |
| val_delta_in_min_deltas | 4.65 |
| test_macro_fg_dice_delta | -0.00608 |
| per_class_dice_delta | `vertebra`=-0.0055, `intervertebral_disc`=-0.00515, `spinal_canal`=-0.00759 |
| indexing_taxonomy_delta_pp | -1.38 |
| indexing_identification_delta_pp | -1.41 |
| pfirrmann_forest_qwk_delta | 0.01140 |
| height_mm_central_mae_delta | 0.10240 |
| n_quality_metrics_improved | 18 |
| n_quality_metrics_worsened | 61 |
| epoch_slowdown_vs_sprint3 | 2.71 |
| test_patients | 33 |
| test_slices | 1655 |

The decision deliberately does not rest on a single metric. It weighs the primary validation criterion together with the test macro Dice, the per-class Dice values, per-patient variability, disc indexing under two estimators, the predicted-mask measurement errors, the end-to-end Pfirrmann agreement, convergence behaviour and training cost. The validation and test evidence point the same way, which is what makes the conclusion safe to draw despite the Pfirrmann exception.

Read together with Sprint 3, the picture is consistent: neither more data exposure nor more capacity moves this plateau. Both were reasonable hypotheses and both have now been tested and rejected on measured evidence, which is a useful result even though neither produced a better model. The remaining measurable loss sits in the disc-indexing stage, not in the segmentation network.


## 14. Recommended Next Step

**Option B - Disc-indexing and post-processing pipeline (no retraining).**

Width 32 did not improve segmentation, and Sprint 3 already showed that data coverage was not the binding constraint either. Two independent capacity/data experiments have now failed to move the segmentation plateau, which is evidence that the remaining error is not where the last two sprints looked. The disc-indexing stage is where the measurable loss actually is: the network recovers the disc region on about 95-97% of discs but assigns the correct integer index on only about 83-84%, so roughly 12-13 pp of usable accuracy is lost to ordering and fragmentation of already-detected regions, not to segmentation quality. That gap is addressable with post-processing on the existing predictions and needs no new training run.


### Segmentation candidate to keep

Sprint 3 Coverage, the width-16 model: it holds the best validation Dice (0.89831), the best test macro foreground Dice (0.90001) and the best disc indexing, at a quarter of the parameters and a third of the training time.


### Proposed steps

- Work from the existing saved predictions - no retraining, no new architecture.
- Enforce geometric ordering across a whole series rather than per-slice, using the fact that series-level disc ordering is far more reliable than per-slice ordering.
- Merge fragmented components before indexing, and reject components below a size threshold calibrated on the training split only.
- Re-score the corrected taxonomy on the test set once, after the post-processing rule is fixed on train/validation data.

| Field | Value |
| --- | --- |
| Success criterion | Raise the corrected-taxonomy index-correct rate from its current level (Sprint 3 remains the best at 84.08%) towards the region-found ceiling, without retraining. |


### Explicitly not recommended

Width 32 cost 2.71x the time per epoch for a -0.00465 validation change. Width 64 was measured in the Sprint 3 audit at ~6.8 GB peak commit and 216-326 h for 30 epochs, so it remains infeasible on this hardware, and there is now direct evidence that more width does not help at this data scale.

**This next experiment has not been started.** It is a recommendation only.


## Artefact Locations

| artefact | path |
| --- | --- |
| training history | `outputs/checkpoints/sprint4_capacity/history.csv` (+ .json) |
| selected checkpoint | `outputs/checkpoints/sprint4_capacity/best_val_dice.pt` |
| per-epoch checkpoints | `outputs/checkpoints/sprint4_capacity/epochs/` |
| training config + summary | `outputs/reports/sprint4_capacity/training_config.json`, `training_summary.json` |
| smoke test | `outputs/reports/sprint4_capacity/smoke_test.json` |
| test segmentation metrics | `outputs/reports/sprint4_capacity/metrics/` |
| disc measurements | `outputs/reports/disc_analysis_sprint4_capacity.csv` |
| indexing failures | `outputs/reports/sprint4_capacity/indexing_failures_sprint4_capacity.csv` |
| comparison table | `outputs/reports/sprint4_capacity/comparison_table.csv` |
| figures | `outputs/visualizations/sprint4_capacity/` |
| test predictions | `data/processed/predictions_sprint4_capacity/` |
| this report | `outputs/reports/sprint4_capacity/sprint4_final_report.md` (+ .json) |


### Figures

- `outputs/visualizations/sprint4_capacity/training_curves.png`
- `outputs/visualizations/sprint4_capacity/sprint3_vs_sprint4_curves.png`
- `outputs/visualizations/sprint4_capacity/three_way_test_comparison.png`
- `outputs/visualizations/sprint4_capacity/computational_cost.png`
- `outputs/visualizations/sprint4_capacity/indexing_comparison.png`
- `outputs/visualizations/sprint4_capacity/measurement_agreement.png`
- `outputs/visualizations/sprint4_capacity/segmentation_metrics.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_best_12_t1_s006.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_best_171_t2_s008.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_best_219_t1_s007.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_median_166_t2_SPACE_s022.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_median_80_t1_s007.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_median_98_t2_SPACE_s049.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_midsagittal_11_t2_s010.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_midsagittal_166_t2_SPACE_s060.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_midsagittal_171_t1_s008.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_worst_11_t2_s018.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_worst_11_t2_SPACE_s018.png`
- `outputs/visualizations/sprint4_capacity/predictions/pred_test_worst_53_t1_s015.png`

