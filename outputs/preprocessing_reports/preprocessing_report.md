# Preprocessing and Data Quality Report

*Automated Segmentation of Vertebrae and Intervertebral Discs in Lumbar Spine MRI Images - Sprint 1*

Generated: 2026-09-22 00:32:01


## 1. Dataset size

| Field | Value |
| --- | --- |
| Series processed | 447 / 447 |
| Series failed | 0 |
| Distinct patients | 218 |
| Sagittal slices available | 14070 |
| Slices written (annotated) | 12415 |
| Slices dropped (no/low annotation) | 1655 |
| Slices per series | 8 - 120 (median 22) |

Slices carrying less than 10.0 mm2 of annotation were dropped. Those are the lateral sagittal slices that fall outside the spine; keeping them would have added mostly-empty targets and worsened an already severe class imbalance. The threshold is a physical area rather than a pixel count because in-plane pixel spacing varies by a factor of ~16 across this dataset.


## 2. Image dimensions before preprocessing

| Field | Value |
| --- | --- |
| Rows (superior-inferior) | 216 - 3682 px |
| Columns (anterior-posterior) | 264 - 1168 px |
| Distinct in-plane shapes | 149 |
| Row spacing | 0.077 - 1.233 mm |
| Column spacing | 0.248 - 1.062 mm |


## 3. Image dimensions after preprocessing

| Field | Value |
| --- | --- |
| Rows x Columns | 352 x 256 px |
| Pixel spacing | 1.0 mm (isotropic) |
| Field of view | 352 mm superior-inferior x 256 mm anterior-posterior |
| Distinct shapes after | 1 |

The target is deliberately **not square**. For every volume the smallest centred crop that still contains the entire annotation was measured: the worst case needs 341 mm superior-inferior but only 244 mm anterior-posterior, because the lumbar spine is tall and narrow. A square 288 x 288 crop would have clipped annotated anatomy in 157 of 447 volumes. Both dimensions are multiples of 32, which suits the downsampling depth of a U-Net in a later sprint.

Every slice now has identical dimensions and a consistent physical scale, reducing 149 distinct input shapes to 1. Image and mask are guaranteed to have the same shape because both go through the same geometric transform, differing only in interpolation.


## 4. Intensity statistics


### Before preprocessing

| Field | Value |
| --- | --- |
| Distinct raw minimum values | `-1000.0`=374, `0.0`=73 |
| Distinct raw maximum values | 70 |
| Most common raw maximum | 3096 in 374 series |
| Range of raw maxima | 349 - 3096 |
| Series with a constant padding floor | 447 |

This is the clearest evidence of the **two intensity conventions**: 374 series share exactly the same minimum (-1000) and maximum (3096), i.e. they were linearly rescaled onto a fixed window with clipping at both ends, while the remaining series keep their native scanner range starting at 0 with a per-series maximum. Because MRI intensity carries no absolute physical meaning, neither group can be compared with the other without normalisation.


### After preprocessing

| Field | Value |
| --- | --- |
| Global min / max | 0.0009 / 1.0000 |
| Mean slice intensity | 0.3183 |
| Mean slice std | 0.2721 |
| Per-slice mean range | 0.0810 - 0.4732 |

All slices now occupy the same bounded `[0, 1]` range regardless of which acquisition convention they came from.


## 5. Missing values and files

| Field | Value |
| --- | --- |
| Series in the pairing table but not processed | 0 |
| Series that failed to read | none |
| Slices with an unwritable output | 0 |


## 6. Image-mask matching status

| Field | Value |
| --- | --- |
| Pairs with matching processed dimensions | 447 / 447 |
| Series where the resize invented a label | 0 |
| Series where a small label was lost | 0 |
| Annotated area retained (mean) | 100.06% |
| Annotated area retained (worst series) | 99.33% |
| Series retaining < 95% of annotated area | 0 |

Annotated area is compared in **mm^2**, not pixels, because resampling deliberately changes the pixel count. A retention close to 100% is evidence that the centre crop did not cut off annotated anatomy. A label *disappearing* is acceptable when a structure is only a few pixels wide; a label *appearing* would mean interpolation corrupted the mask and is treated as a failure.


## 7. Mask class distribution

| class id | class | pixels | % of all pixels | % of labelled pixels |
| --- | --- | --- | --- | --- |
| 0 | background | 1,060,235,393 | 94.7705% | - |
| 1 | vertebra | 42,031,210 | 3.7570% | 71.842% |
| 2 | intervertebral_disc | 8,839,638 | 0.7901% | 15.109% |
| 3 | spinal_canal | 7,634,239 | 0.6824% | 13.049% |

**The class imbalance is severe and must shape the later modelling sprint.** Background dominates by orders of magnitude, and the intervertebral discs - one of the two structures the project targets - are the smallest foreground class. A plain pixel-wise cross-entropy loss would be minimised by predicting background almost everywhere, so a Dice-based or class-weighted loss will be needed.


## 8. Valid and rejected samples

| Field | Value |
| --- | --- |
| Valid image-mask series | 447 |
| Valid preprocessed slices | 12415 |
| Rejected series (read failure) | 0 |
| Rejected slices (insufficient annotation) | 1655 |
| Series failing mask-integrity checks | 0 |

Note the distinction: the dropped slices are *deliberately excluded* by the slice-selection rule, not corrupted data. No series was rejected for a quality failure.


## 9. Preprocessing operations performed

- `reorient every volume to a canonical RAS frame (loss-less axis permutation)`
- `extract sagittal slices along the through-plane axis`
- `drop slices with less than 10.0 mm2 of annotation`
- `resample in-plane to 1.0 mm/px (image: area/bilinear, mask: nearest neighbour)`
- `centre crop or zero-pad to 352x256 px`
- `clip to foreground percentiles (1.0, 99.0) and scale to [0, 1]`
- `median denoise, kernel 3`
- `CLAHE clip=2.0 tiles=(8, 8)`
- `collapse raw labels to 4 semantic classes and to a contiguous instance space`
- `verify no label value was invented by the resize`
- `save image (float16) + semantic mask (uint8) + instance mask (uint8) per slice`


### Configuration used

| Field | Value |
| --- | --- |
| target_spacing_mm | 1 |
| target_size | 352, 256 |
| clip_percentiles | 1.0, 99.0 |
| normalize | True |
| denoise | True |
| denoise_method | median |
| denoise_kernel | 3 |
| enhance_contrast | True |
| clahe_clip_limit | 2 |
| clahe_tile_grid | 8, 8 |
| bias_field_correction | False |
| keep_only_annotated_slices | True |
| min_labelled_area_mm2 | 10 |


## 10. Per-series metrics (first 20)

| image_id | modality | n_slices_total | n_slices_kept | rows_before | cols_before | row_spacing_before_mm | col_spacing_before_mm | rows_after | cols_after | annotated_area_retained | mask_labels_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1_t1 | t1 | 50 | 30 | 578 | 448 | 0.5006 | 0.625 | 352 | 256 | 0.9989 | True |
| 1_t2 | t2 | 50 | 30 | 578 | 448 | 0.5006 | 0.625 | 352 | 256 | 0.9989 | True |
| 2_t1 | t1 | 17 | 16 | 294 | 320 | 0.8893 | 0.8125 | 352 | 256 | 1.001 | True |
| 2_t2 | t2 | 17 | 16 | 346 | 384 | 0.7539 | 0.6771 | 352 | 256 | 1.001 | True |
| 3_t1 | t1 | 18 | 18 | 389 | 320 | 0.6742 | 0.8125 | 352 | 256 | 0.9983 | True |
| 3_t2 | t2 | 17 | 17 | 486 | 384 | 0.5388 | 0.6771 | 352 | 256 | 1.002 | True |
| 4_t1 | t1 | 27 | 25 | 553 | 448 | 0.5106 | 0.625 | 352 | 256 | 0.9968 | True |
| 4_t2 | t2 | 27 | 25 | 553 | 448 | 0.5106 | 0.625 | 352 | 256 | 0.9968 | True |
| 5_t1 | t1 | 17 | 17 | 512 | 512 | 0.5859 | 0.5859 | 352 | 256 | 1.002 | True |
| 5_t2 | t2 | 17 | 17 | 512 | 512 | 0.5859 | 0.5859 | 352 | 256 | 1.002 | True |
| 5_t2_SPACE | t2_SPACE | 120 | 109 | 640 | 512 | 0.4688 | 0.4688 | 352 | 256 | 1 | True |
| 6_t2 | t2 | 16 | 14 | 329 | 384 | 0.7962 | 0.6771 | 352 | 256 | 1.002 | True |
| 7_t1 | t1 | 17 | 17 | 512 | 512 | 0.5859 | 0.5859 | 352 | 256 | 1.002 | True |
| 7_t2 | t2 | 17 | 17 | 512 | 512 | 0.5859 | 0.5859 | 352 | 256 | 1.003 | True |
| 7_t2_SPACE | t2_SPACE | 120 | 111 | 640 | 512 | 0.4688 | 0.4688 | 352 | 256 | 1.004 | True |
| 8_t1 | t1 | 15 | 15 | 384 | 384 | 0.7292 | 0.7292 | 352 | 256 | 1 | True |
| 8_t2 | t2 | 15 | 15 | 384 | 384 | 0.7292 | 0.7292 | 352 | 256 | 1 | True |
| 9_t1 | t1 | 19 | 16 | 464 | 320 | 0.5667 | 0.8125 | 352 | 256 | 0.9996 | True |
| 9_t2 | t2 | 19 | 16 | 610 | 384 | 0.4315 | 0.6771 | 352 | 256 | 0.9974 | True |
| 10_t1 | t1 | 15 | 15 | 896 | 896 | 0.3125 | 0.3125 | 352 | 256 | 1 | True |

_...427 more rows_

Full table: `outputs/preprocessing_reports/series_preprocessing.csv`


## 11. Figures produced

- `outputs/visualizations/sample_22_t1_s007_A-E.png`
- `outputs/visualizations/sample_22_t1_s007_before_after.png`
- `outputs/visualizations/sample_202_t1_s011_A-E.png`
- `outputs/visualizations/sample_202_t1_s011_before_after.png`
- `outputs/visualizations/sample_109_t2_s015_A-E.png`
- `outputs/visualizations/sample_109_t2_s015_before_after.png`
- `outputs/visualizations/sample_108_t2_s008_A-E.png`
- `outputs/visualizations/sample_108_t2_s008_before_after.png`
- `outputs/visualizations/sample_152_t2_SPACE_s073_A-E.png`
- `outputs/visualizations/sample_152_t2_SPACE_s073_before_after.png`
- `outputs/visualizations/sample_18_t2_SPACE_s061_A-E.png`
- `outputs/visualizations/sample_18_t2_SPACE_s061_before_after.png`
- `outputs/visualizations/stages_22_t1_s007.png`
- `outputs/visualizations/mask_interpolation_comparison.png`
- `outputs/visualizations/dataset_overview_grid.png`
- `outputs/visualizations/class_distribution.png`


## 12. Outputs

- `data/processed/slices/` - 12415 `.npz` files (image float16, semantic mask uint8, instance mask uint8)
- `data/processed/slice_index.csv` - one row per slice with metadata
- `data/processed/label_mapping.json` - label space definitions
- `outputs/preprocessing_reports/series_preprocessing.csv` - per-series metrics

**No segmentation model has been trained and no Dice or IoU score has been computed.** Those belong to the next sprint; this report covers preprocessing only.

