# Dataset Inspection Report

*Automated Segmentation of Vertebrae and Intervertebral Discs in Lumbar Spine MRI Images - Sprint 1*

Generated: 2026-09-22 00:01:27


## 1. Headline figures

| Field | Value |
| --- | --- |
| Total image volumes | 447 |
| Total mask volumes | 447 |
| Readable volumes | 447 |
| Read errors | 0 |
| Distinct patients | 218 |
| Series per modality | `t2`=210, `t1`=196, `t2_SPACE`=41 |
| Image format | `mha`=447 |
| Mask format | `mha`=447 |
| Image voxel dtype | `float32`=447 |
| Mask voxel dtype | `int16`=447 |
| Total sagittal slices | 14070 |
| Slices containing any label | 12598 |
| Slices with >=50 labelled px | 12337 |


## 2. Format, dimensions and geometry

Volumes are stored as **MetaImage** (`.mha`) 3-D sagittal series, not 2-D image files. Every series has one mask volume of the same name.

| Field | Value |
| --- | --- |
| In-plane rows (sup-inf) | 216 - 3682 px (123 distinct values) |
| In-plane cols (ant-post) | 264 - 1168 px (45 distinct values) |
| Slices per volume | 8 - 154 (median 24) |
| Row spacing | 0.077 - 1.233 mm |
| Col spacing | 0.248 - 1.062 mm |
| Slice spacing | 0.859 - 9.627 mm |
| Native orientations | `LPS`=374, `PIR`=73 |

**Finding - image dimensions are not constant.** Both the matrix size and the physical voxel size vary between series, so a fixed resize in pixels would place the same vertebra at different physical scales. Preprocessing therefore resamples to a common mm/pixel scale before cropping to a fixed matrix.

**Finding - storage orientation is not constant.** The 2-D TSE series are stored `LPS` while the 3-D `t2_SPACE` series are stored `PIR`, meaning the array axis that steps through sagittal slices differs between files. Every volume is reoriented to a canonical `RAS` frame on load; this is a pure axis permutation plus flips, so it is loss-less and safe for label masks.


### Per-series geometry (first 15 series)

| image_id | modality | img_native_orientation | img_rows | img_cols | img_n_slices | img_row_spacing_mm | img_col_spacing_mm | img_slice_spacing_mm | img_fov_rows_mm | img_fov_cols_mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1_t1 | t1 | LPS | 578 | 448 | 50 | 0.5006 | 0.625 | 3.321 | 289.4 | 280 |
| 1_t2 | t2 | LPS | 578 | 448 | 50 | 0.5006 | 0.625 | 3.321 | 289.4 | 280 |
| 2_t1 | t1 | LPS | 294 | 320 | 17 | 0.8893 | 0.8125 | 4.786 | 261.5 | 260 |
| 2_t2 | t2 | LPS | 346 | 384 | 17 | 0.7539 | 0.6771 | 4.788 | 260.9 | 260 |
| 3_t1 | t1 | LPS | 389 | 320 | 18 | 0.6742 | 0.8125 | 4.821 | 262.3 | 260 |
| 3_t2 | t2 | LPS | 486 | 384 | 17 | 0.5388 | 0.6771 | 4.817 | 261.9 | 260 |
| 4_t1 | t1 | LPS | 553 | 448 | 27 | 0.5106 | 0.625 | 3.32 | 282.4 | 280 |
| 4_t2 | t2 | LPS | 553 | 448 | 27 | 0.5106 | 0.625 | 3.32 | 282.4 | 280 |
| 5_t1 | t1 | LPS | 512 | 512 | 17 | 0.5859 | 0.5859 | 3.3 | 300 | 300 |
| 5_t2 | t2 | LPS | 512 | 512 | 17 | 0.5859 | 0.5859 | 3.3 | 300 | 300 |
| 5_t2_SPACE | t2_SPACE | PIR | 640 | 512 | 120 | 0.4688 | 0.4688 | 0.9 | 300 | 240 |
| 6_t2 | t2 | LPS | 329 | 384 | 16 | 0.7962 | 0.6771 | 4.782 | 262 | 260 |
| 7_t1 | t1 | LPS | 512 | 512 | 17 | 0.5859 | 0.5859 | 3.3 | 300 | 300 |
| 7_t2 | t2 | PIR | 512 | 512 | 17 | 0.5859 | 0.5859 | 3.3 | 300 | 300 |
| 7_t2_SPACE | t2_SPACE | PIR | 640 | 512 | 120 | 0.4688 | 0.4688 | 0.9 | 300 | 240 |

_...432 more rows_


## 3. Mask labels and classes

Masks are integer label volumes (not binary, not one-hot). The complete label vocabulary observed across the dataset is:

| raw value | meaning |
| --- | --- |
| 0 | background |
| 1 | vertebra #1 (counted from the most inferior) |
| 2 | vertebra #2 (counted from the most inferior) |
| 3 | vertebra #3 (counted from the most inferior) |
| 4 | vertebra #4 (counted from the most inferior) |
| 5 | vertebra #5 (counted from the most inferior) |
| 6 | vertebra #6 (counted from the most inferior) |
| 7 | vertebra #7 (counted from the most inferior) |
| 8 | vertebra #8 (counted from the most inferior) |
| 9 | vertebra #9 (counted from the most inferior) |
| 100 | spinal canal |
| 201 | IVD #1 (counted from the most inferior) |
| 202 | IVD #2 (counted from the most inferior) |
| 203 | IVD #3 (counted from the most inferior) |
| 204 | IVD #4 (counted from the most inferior) |
| 205 | IVD #5 (counted from the most inferior) |
| 206 | IVD #6 (counted from the most inferior) |
| 207 | IVD #7 (counted from the most inferior) |
| 208 | IVD #8 (counted from the most inferior) |
| 209 | IVD #9 (counted from the most inferior) |

| Field | Value |
| --- | --- |
| Distinct raw label values | 20 |
| Undocumented label values found | none |
| Vertebra labels per volume | 3 - 9 (median 7) |
| IVD labels per volume | 3 - 9 (median 7) |
| Volumes containing spinal canal | 447 / 447 |
| Labelled voxel fraction | 5.64% of voxels on average |

**Class imbalance is severe**: on average only a few percent of voxels carry any label at all, and the discs are far smaller than the vertebrae. This is recorded now because it will drive the loss function choice in a later sprint.


### Semantic class volume distribution (all volumes pooled)

| class | voxels | share of labelled |
| --- | --- | --- |
| vertebra | 148114894 | 71.7% |
| intervertebral_disc | 31440866 | 15.2% |
| spinal_canal | 27022473 | 13.1% |


## 4. Intensity characteristics

MRI intensities have no absolute physical meaning, and this dataset makes that concrete: **two incompatible intensity conventions coexist.**

| Field | Value |
| --- | --- |
| Series with a constant padding floor | 447 |
| Series without a padding floor | 0 |
| Distinct raw minimum values | `-1000.0`=374, `0.0`=73 |
| Distinct raw maximum values | `349.0`=1, `379.0`=1, `398.0`=1, `408.0`=1, `433.0`=1, `444.0`=1, `451.0`=1, `465.0`=1, `470.0`=1, `472.0`=1, `475.0`=2, `480.0`=1, `484.0`=1, `487.0`=1, `490.0`=1, `501.0`=1, `512.0`=1, `514.0`=1, `516.0`=1, `519.0`=1, `520.0`=1, `521.0`=1, `530.0`=1, `539.0`=1, `542.0`=1, `543.0`=1, `548.0`=1, `557.0`=1, `560.0`=1, `567.0`=1, `574.0`=1, `575.0`=1, `582.0`=1, `590.0`=1, `600.0`=1, `610.0`=1, `618.0`=1, `621.0`=1, `629.0`=1, `656.0`=1, `675.0`=1, `680.0`=2, `688.0`=1, `692.0`=1, `698.0`=1, `700.0`=1, `702.0`=1, `718.0`=1, `720.0`=1, `729.0`=2, `730.0`=1, `733.0`=1, `736.0`=1, `749.0`=1, `750.0`=1, `762.0`=1, `772.0`=1, `785.0`=2, `796.0`=1, `798.0`=1, `840.0`=1, `850.0`=1, `878.0`=1, `920.0`=1, `922.0`=1, `962.0`=1, `1345.0`=1, `1367.0`=1, `1501.0`=1, `3096.0`=374 |

In the padded group a single extreme value (typically -1000) fills the empty region of the resampled grid and can account for a large share of all voxels, while a further share sits pinned at the maximum. Any mean/std or min-max normalisation would be dominated by those two spikes rather than by tissue. Preprocessing therefore computes statistics over **foreground voxels only** and clips at robust percentiles.


### Per-series intensity (first 15 series)

| image_id | raw_min | raw_max | padding_value | foreground_fraction | saturated_fraction | fg_p1 | fg_p50 | fg_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1_t1 | -1,000 | 3,096 | -1,000 | 0.2914 | 0.0501 | -986 | 1,767 | 3,096 |
| 1_t2 | -1,000 | 3,096 | -1,000 | 0.291 | 0.05015 | -982 | 1,467 | 3,096 |
| 2_t1 | -1,000 | 3,096 | -1,000 | 0.8973 | 0.05068 | -986 | 279 | 3,096 |
| 2_t2 | -1,000 | 3,096 | -1,000 | 0.9188 | 0.05087 | -980 | -272 | 3,096 |
| 3_t1 | -1,000 | 3,096 | -1,000 | 0.8881 | 0.05027 | -989 | 146 | 3,096 |
| 3_t2 | -1,000 | 3,096 | -1,000 | 0.9072 | 0.0501 | -984 | -390 | 3,096 |
| 4_t1 | -1,000 | 3,096 | -1,000 | 0.5506 | 0.05004 | -983 | 558 | 3,096 |
| 4_t2 | -1,000 | 3,096 | -1,000 | 0.5493 | 0.05006 | -983 | 796 | 3,096 |
| 5_t1 | -1,000 | 3,096 | -1,000 | 0.9293 | 0.05005 | -978 | -215 | 3,096 |
| 5_t2 | -1,000 | 3,096 | -1,000 | 0.9419 | 0.05001 | -977 | -645 | 3,096 |
| 5_t2_SPACE | 0 | 398 | 0 | 0.9588 | 2.543e-08 | 1 | 15 | 197 |
| 6_t2 | -1,000 | 3,096 | -1,000 | 0.8357 | 0.05037 | -971 | -541 | 3,096 |
| 7_t1 | -1,000 | 3,096 | -1,000 | 0.9065 | 0.05007 | -982 | -352 | 3,096 |
| 7_t2 | 0 | 656 | 0 | 0.9774 | 4.488e-07 | 2 | 17 | 397 |
| 7_t2_SPACE | 0 | 530 | 0 | 0.9737 | 2.543e-08 | 1 | 19 | 273 |

_...432 more rows_

| Field | Value |
| --- | --- |
| Mean foreground fraction | 74.4% |
| Mean saturated fraction | 4.20% |
| Max saturated fraction | 5.12% |


## 5. Image-mask correspondence

Filenames follow `<patient_id>_<modality>.mha`, and `images/X.mha` pairs with `masks/X.mha`. Pairing uses that parsed identifier, never directory order, so a missing file surfaces as an explicit mismatch instead of silently shifting every subsequent pair.

| Field | Value |
| --- | --- |
| Matched pairs | 447 |
| Images without a mask | none |
| Masks without an image | none |
| Filenames not matching the convention | none |
| Series missing an overview.csv row | none |
| Image/mask shape disagreements | 0 |
| Image/mask spacing disagreements | 0 |
| Image/mask orientation disagreements | 0 |


### Example verified pairs

```
Matched image/mask pairs : 447
Distinct patients        : 218
Modalities               : {'t2': 210, 't1': 196, 't2_SPACE': 41}
Images without a mask    : 0
Masks without an image   : 0
Unparsable filenames     : 0
Missing overview.csv row : 0

Example pairs (first 5):
  [1_t1] patient=1 modality=t1
      image: data/extracted/images/1_t1.mha
      mask : data/extracted/masks/1_t1.mha
  [1_t2] patient=1 modality=t2
      image: data/extracted/images/1_t2.mha
      mask : data/extracted/masks/1_t2.mha
  [2_t1] patient=2 modality=t1
      image: data/extracted/images/2_t1.mha
      mask : data/extracted/masks/2_t1.mha
  [2_t2] patient=2 modality=t2
      image: data/extracted/images/2_t2.mha
      mask : data/extracted/masks/2_t2.mha
  [3_t1] patient=3 modality=t1
      image: data/extracted/images/3_t1.mha
      mask : data/extracted/masks/3_t1.mha
```


## 6. Anatomical sanity checks

These check that the reorientation actually placed the axes where the code assumes they are. They are computed from the mask geometry alone.

| check | passed | failed |
| --- | --- | --- |
| spinal canal lies posterior to the vertebral bodies | 447 | 0 |
| vertebra label 1 lies inferior to the highest vertebra label | 447 | 0 |


## 7. Duplicate detection

Grouping by SHA-256 of the file contents.


### images

| Field | Value |
| --- | --- |
| Duplicate groups | 0 |
| Groups within a single patient | 0 |
| Groups spanning different patients | 0 |
| Examples (same patient) | none |
| Cross-patient groups (would be a problem) | none |


### masks

| Field | Value |
| --- | --- |
| Duplicate groups | 107 |
| Groups within a single patient | 107 |
| Groups spanning different patients | 0 |
| Examples (same patient) | ['100_t1.mha', '100_t2.mha'], ['105_t1.mha', '105_t2.mha'], ['107_t1.mha', '107_t2.mha'], ['109_t1.mha', '109_t2.mha'... |
| Cross-patient groups (would be a problem) | none |

**Interpretation.** No image volume is duplicated. A large number of *mask* volumes are byte-identical, but always within one patient: the T1 and T2 series of a patient were acquired on the same grid and share a single annotation. This is a property of the annotation process, not corruption. It does, however, mean the T1/T2 series of a patient are highly correlated, which is exactly why the train/val/test split must be made **per patient**.


## 8. Metadata files

| Field | Value |
| --- | --- |
| overview.csv | 447 rows x 41 columns (one row per series) |
| radiological_gradings.csv | 1520 rows, 218 patients (one row per patient x disc) |
| Identifier in overview.csv | `new_file_name` matches the volume filename stem exactly |
| Identifier in gradings | `Patient` matches the numeric prefix of the filename |
| Official subset column | `training`=360, `validation`=87 |


### Relevant metadata columns

- `new_file_name` - series identifier, joins to the volume files
- `num_vertebrae`, `num_discs` - annotated structure counts per series
- `sex`, `subset` - patient/series level attributes
- `Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength` - scanner
- `MRAcquisitionType`, `ScanningSequence`, `SeriesDescription` - sequence
- `PixelSpacing`, `SliceThickness`, `SpacingBetweenSlices` - geometry
- `EchoTime`, `RepetitionTime` - contrast weighting
- gradings: `IVD label`, `Pfirrman grade`, `Modic`, `Disc herniation`, `Disc narrowing`, `Disc bulging`, `Spondylolisthesis`, `UP/LOW endplate` - per-disc radiological gradings (classification targets for a later sprint, not segmentation targets)


### Missing values in overview.csv

| Field | Value |
| --- | --- |
| birth_date | 328 |
| AngioFlag | 211 |
| BodyPartExamined | 119 |
| DeviceSerialNumber | 299 |
| SequenceName | 211 |
| SeriesDescription | 180 |
| SoftwareVersions | 119 |
| TransmitCoilName | 211 |

**Finding - dirty categorical values.** `sex` is stored with inconsistent trailing whitespace ({'F': 276, 'M': 171}), i.e. `'F'` and `'F '` appear as two distinct values. The loader strips whitespace from all text columns. The heavily-missing columns are optional DICOM acquisition fields and are not used by preprocessing.


## 9. Problems found

- No unreadable volumes: all series loaded successfully.
- No missing or unmatched files: every image has exactly one mask.
- No undocumented label values in any mask.
- Image and mask geometry agree for every pair.
- Heterogeneous acquisition: matrix size, pixel spacing, slice thickness and storage orientation all vary between series - handled by resampling to a common physical scale and reorienting to RAS.
- Two incompatible intensity conventions coexist (padded -1000..3096 vs plain 0..~700) - handled by foreground-restricted percentile normalisation.
- 107 groups of byte-identical mask volumes, all within a single patient (shared T1/T2 annotation). Not corruption, but it makes patient-level splitting mandatory.
- `sex` contains trailing-whitespace duplicates in the raw CSV; stripped on load.
- 2 series annotate fewer than 6 vertebrae/discs (['246_t2', '35_t2']), i.e. a much smaller field of view than the rest. Kept, but flagged as outliers.
- Every series contains at least one usably-annotated slice.
- The official `subset` column only splits training/validation (no test set) and is series-level, so Sprint 1 derives its own patient-level 3-way split.

