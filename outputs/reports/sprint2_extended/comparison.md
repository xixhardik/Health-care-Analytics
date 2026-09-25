# Sprint 2 Extended Training - Controlled Experiment Report

*Same 16-channel U-Net, same split, same seed; only the epoch budget changed*

Generated: 2026-09-23 05:53:59

**Controlled variable: number of epochs (12 -> up to 30).** Architecture, patient-level split, seed, preprocessing, class definitions, loss, optimiser, augmentation policy and slices-per-epoch were all held fixed. The test set was not touched during training or model selection.


## 1. Training outcome

| Field | Value |
| --- | --- |
| Resume mode | weights_only_from_baseline |
| Resumed at epoch | 12 |
| Optimiser state restored | False |
| Epochs completed (total) | 30 |
| Epochs run this session | 18 |
| Max epochs | 30 |
| Stopped early | False |
| Stop reason | reached the maximum of 30 epochs without triggering early stopping |
| Best validation foreground Dice | 0.898513214251773 (epoch 30) |
| Best validation loss | 0.13474805168807508 (epoch 29) |
| Wall time this session (min) | 231.1 |
| Mean epoch time (s) | 770.2 |
| Checkpoint evaluated | val_dice |

**Resume caveat.** The baseline checkpoint stored weights only - no optimiser or scheduler state - so Adam's moment estimates restart from zero and the LR schedule was re-derived. This run is therefore a warm-restarted continuation, NOT bit-identical to an uninterrupted 30-epoch run. The baseline also used CosineAnnealingLR(T_max=12), so its LR had annealed to ~0 by epoch 12; here the schedule is re-cast over 30 epochs.


## 1b. Did it converge?

| Field | Value |
| --- | --- |
| Validation Dice, last 6 epochs | 0.89806 - 0.89851 (spread 0.00045) |
| Mean gain per epoch, last 6 | +0.000085 |
| Mean gain per epoch, epochs 14-20 | +0.001020 |
| Ratio (mid / late) | 12 |
| Final learning rate | 2.739e-06 |
| Early stopping fired | False |

**The model has converged for this configuration.** Over the final six epochs validation foreground Dice moved a total of 0.00045, i.e. +0.000085 per epoch, against +0.001020 per epoch during epochs 14-20 - roughly 12.0x slower. The cosine schedule had annealed the learning rate to ~0 by epoch 30.

**Early stopping did not fire, and that is a configuration flaw rather than evidence of continued learning.** `min_delta` was 0, so an improvement of +0.00004 still reset the patience counter. With a meaningful `min_delta` (~0.001) the run would have stopped around epoch 25-26 at effectively the same result. This should be corrected in the next run.


## 2. Test-set segmentation: baseline vs extended

Aggregate (dataset-level) metrics on the 33 held-out test patients. `delta` is extended minus baseline; higher is better for all of these.

| metric | baseline | extended | delta | verdict |
| --- | --- | --- | --- | --- |
| background/dice | 0.9950 | 0.9951 | +0.0001 | improved |
| background/iou | 0.9900 | 0.9903 | +0.0003 | improved |
| background/precision | 0.9947 | 0.9951 | +0.0004 | improved |
| background/recall | 0.9953 | 0.9952 | -0.0001 | worsened |
| vertebra/dice | 0.9094 | 0.9133 | +0.0039 | improved |
| vertebra/iou | 0.8338 | 0.8405 | +0.0067 | improved |
| vertebra/precision | 0.9172 | 0.9178 | +0.0006 | improved |
| vertebra/recall | 0.9017 | 0.9089 | +0.0072 | improved |
| intervertebral_disc/dice | 0.8760 | 0.8779 | +0.0019 | improved |
| intervertebral_disc/iou | 0.7794 | 0.7824 | +0.0029 | improved |
| intervertebral_disc/precision | 0.8782 | 0.8711 | -0.0071 | worsened |
| intervertebral_disc/recall | 0.8739 | 0.8848 | +0.0109 | improved |
| spinal_canal/dice | 0.9004 | 0.9026 | +0.0022 | improved |
| spinal_canal/iou | 0.8188 | 0.8224 | +0.0037 | improved |
| spinal_canal/precision | 0.8960 | 0.8926 | -0.0035 | worsened |
| spinal_canal/recall | 0.9047 | 0.9128 | +0.0081 | improved |
| macro_foreground/dice | 0.8953 | 0.8979 | +0.0027 | improved |
| macro_foreground/iou | 0.8107 | 0.8151 | +0.0044 | improved |
| macro_foreground/precision | 0.8972 | 0.8938 | -0.0033 | worsened |
| macro_foreground/recall | 0.8934 | 0.9022 | +0.0087 | improved |
| pixel_accuracy | 0.9904 | 0.9907 | +0.0003 | improved |
| per_patient_dice/background | 0.9951 | 0.9952 | +0.0002 | improved |
| per_patient_dice/vertebra | 0.9101 | 0.9141 | +0.0040 | improved |
| per_patient_dice/intervertebral_disc | 0.8842 | 0.8868 | +0.0026 | improved |
| per_patient_dice/spinal_canal | 0.9004 | 0.9025 | +0.0022 | improved |
| disc_identification/pct_slices_with_matching_disc_count | 74.2600 | 75.7100 | +1.4500 | improved |
| disc_identification/pct_discs_region_found | 95.3400 | 95.6800 | +0.3400 | improved |
| disc_identification/pct_discs_index_correct | 78.4000 | 82.6000 | +4.2000 | improved |
| disc_detection/pct_truth_discs_recovered | 99.3300 | 99.1100 | -0.2200 | worsened |


## 3. Disc measurements from predicted masks

Agreement between measurements taken from predicted masks and from ground-truth masks. For `measurement_mae/*` **lower is better**; for `measurement_r/*` higher is better.

| metric | baseline | extended | delta | verdict |
| --- | --- | --- | --- | --- |
| measurement_mae/ap_extent_mm | 2.1284 | 1.9865 | -0.1419 | improved |
| measurement_r/ap_extent_mm | 0.7651 | 0.7766 | +0.0115 | improved |
| measurement_mae/area_mm2 | 27.0315 | 25.3431 | -1.6884 | improved |
| measurement_r/area_mm2 | 0.8895 | 0.9079 | +0.0184 | improved |
| measurement_mae/canal_width_at_disc_mm | 1.1614 | 1.1298 | -0.0316 | improved |
| measurement_r/canal_width_at_disc_mm | 0.7936 | 0.8107 | +0.0171 | improved |
| measurement_mae/disc_to_vertebra_height_ratio | 0.1341 | 0.1361 | +0.0020 | worsened |
| measurement_r/disc_to_vertebra_height_ratio | 0.6265 | 0.6467 | +0.0202 | improved |
| measurement_mae/height_mm_anterior | 0.9559 | 0.8780 | -0.0779 | improved |
| measurement_r/height_mm_anterior | 0.8306 | 0.8535 | +0.0229 | improved |
| measurement_mae/height_mm_central | 0.8476 | 0.8065 | -0.0411 | improved |
| measurement_r/height_mm_central | 0.8715 | 0.8875 | +0.0160 | improved |
| measurement_mae/height_mm_posterior | 0.7493 | 0.7531 | +0.0038 | worsened |
| measurement_r/height_mm_posterior | 0.7531 | 0.7478 | -0.0053 | worsened |
| measurement_mae/height_ratio_to_series_median | 0.1247 | 0.1186 | -0.0061 | improved |
| measurement_r/height_ratio_to_series_median | 0.8124 | 0.8356 | +0.0232 | improved |
| measurement_mae/intensity_disc_vertebra_ratio | 0.0369 | 0.0360 | -0.0009 | improved |
| measurement_r/intensity_disc_vertebra_ratio | 0.9609 | 0.9638 | +0.0029 | improved |
| end_to_end_pfirrmann/logreg/quadratic_weighted_kappa | 0.6195 | 0.6123 | -0.0072 | worsened |
| end_to_end_pfirrmann/logreg/mae | 0.9147 | 0.9000 | -0.0147 | improved |
| end_to_end_pfirrmann/logreg/within_one_grade | 0.8104 | 0.8000 | -0.0104 | worsened |
| end_to_end_pfirrmann/forest/quadratic_weighted_kappa | 0.6294 | 0.6375 | +0.0081 | improved |
| end_to_end_pfirrmann/forest/mae | 0.8057 | 0.8000 | -0.0057 | improved |
| end_to_end_pfirrmann/forest/within_one_grade | 0.8389 | 0.8429 | +0.0040 | improved |


## 5. Decision on the disc-level re-run

| Field | Value |
| --- | --- |
| Macro foreground Dice (baseline) | 0.8953 |
| Macro foreground Dice (extended) | 0.8979 |
| Delta | 0.00267 |
| Meaningful-change threshold | 0.005 |
| Verdict | improved |
| Disc-level re-run justified | True |
| Reason | |delta macro foreground Dice| = 0.0027 < threshold 0.005; |delta disc-index accuracy| = 4.20 pp >= threshold 1.0 pp |


## 6. Figures

- `outputs/visualizations/sprint2_extended/training_curves_combined.png`
- `outputs/visualizations/sprint2_extended/metric_comparison.png`
- `outputs/visualizations/sprint2_extended/predictions/` - prediction figures

No clinical diagnosis, no postoperative healing prediction and no longitudinal improvement is claimed. The dataset contains a single timepoint per patient and no follow-up imaging.

