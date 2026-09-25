# Preprocessing Step Justification

*Which optional preprocessing steps are actually worth applying, measured on real slices*

Generated: 2026-09-22 00:10:42


## Method

Each variant was applied to the mid-sagittal slice of **40 randomly sampled series** (seed fixed). Every variant shares identical geometry handling and intensity normalisation, so score differences are attributable only to the step under test. Metrics are averaged over the sample.

- **noise_sigma** - estimated noise level (Immerkaer). *Lower is better.*
- **boundary_sharpness** - mean image gradient on the true vertebra/disc boundary taken from the ground-truth mask. *Higher is better* - it means the anatomical edge the model must find is still crisp.
- **class_contrast_cnr** - contrast-to-noise ratio between the vertebra and disc classes. *Higher is better* - this is the separability the project goal depends on.
- **dynamic_range_std** - spread of intensities actually used within the tissue region. Higher means less of [0, 1] is wasted.
- **bias_inhomogeneity** - low-frequency brightness drift across vertebral bone. *Lower is better.*


## Measured results

| variant | noise_sigma | boundary_sharpness | class_contrast_cnr | dynamic_range_std | bias_inhomogeneity |
| --- | --- | --- | --- | --- | --- |
| normalised only | 0.01048 | 0.1174 | 2.131 | 0.2692 | 0.19 |
| median 3x3 | 0.00676 | 0.0943 | 2.296 | 0.2664 | 0.191 |
| gaussian 3x3 | 0.00181 | 0.08696 | 2.305 | 0.2618 | 0.1899 |
| bilateral | 0.00626 | 0.1042 | 2.266 | 0.2671 | 0.1901 |
| CLAHE only | 0.03147 | 0.1584 | 2.255 | 0.2694 | 0.08855 |
| median + CLAHE | 0.01232 | 0.1174 | 2.556 | 0.2576 | 0.1096 |
| gaussian + CLAHE | 0.00613 | 0.1227 | 2.479 | 0.2599 | 0.08445 |


## Decisions


### 1. Geometry: resample to 1.0 mm/px, then centre crop/pad to 352x256

**Applied.** Not optional - inspection showed in-plane pixel spacing varies from 0.077 mm to 1.23 mm and matrix sizes range from 216 to 3682 rows. Resampling to a fixed mm/pixel keeps a vertebra the same physical size in every patient and preserves the anatomical aspect ratio, which a plain pixel resize does not.

The crop size was **derived from the data, not chosen by convention.** For every volume the smallest centred crop that still contains the entire annotation was computed: the worst case needs 341 mm superior-inferior but only 244 mm anterior-posterior, because the lumbar spine is tall and narrow. A square 288x288 crop would have clipped annotated anatomy in **157 of 447 volumes**. At 352x256 no volume loses any annotation, and both dimensions are multiples of 32, which suits the downsampling depth of a U-Net in a later sprint.


### 2. Intensity normalisation (foreground percentile clip)

**Applied.** Not optional either: two incompatible intensity conventions coexist in the raw data, and MRI intensity has no absolute meaning. Statistics are restricted to foreground voxels because the padded background can be up to ~70% of a volume and would otherwise dominate them.


### 3. Noise reduction

Median 3x3 reduces the noise estimate from `0.01048` to `0.00676` (35.5% lower) while retaining 80.3% of the boundary sharpness.

Gaussian 3x3 removes more noise (`0.00181`, 82.7% lower) but keeps only 74.1% of the boundary sharpness - it blurs exactly the vertebra/disc edge the segmentation has to find. **Removing the most noise is not the goal**; separating the two target classes is, and on that metric median 3x3 combined with CLAHE scores highest of all seven variants (`2.5562` vs `2.4793` for gaussian + CLAHE).

**Applied - median 3x3.** A median filter suppresses the speckle in MRI magnitude images while keeping edges comparatively intact. Bilateral filtering preserves slightly more edge (88.8%) but is roughly an order of magnitude slower and scored lower on class contrast, so it was rejected.


### 4. Contrast enhancement (CLAHE)

CLAHE alone moves vertebra-vs-disc CNR from `2.1310` to `2.2546`, and median + CLAHE reaches `2.5562` (+20.0% over normalisation alone) - the best score in the table. Usable dynamic range within the tissue region changes by -4.3%, i.e. essentially unchanged, so the gain is genuine local contrast rather than simple histogram stretching.

**Applied** (clip limit 2.0, 8x8 tiles). It increases the separability of exactly the two structures the project has to distinguish. It is applied *after* denoising, because CLAHE amplifies whatever noise is present - visible in the table as the raised noise figure for `CLAHE only` (`0.03147`) compared with `median + CLAHE` (`0.01232`).

The best-scoring variant overall on class contrast is **median + CLAHE**, which is the combination the pipeline uses.


### 5. MRI intensity inhomogeneity

Worth noting from the table: CLAHE reduces the measured bias-field inhomogeneity from `0.18995` to `0.08855` on its own, and median + CLAHE to `0.10964`. That is not a coincidence - CLAHE equalises each tile independently, so a smooth brightness drift across the image is partly cancelled. Intensity inhomogeneity is therefore already being addressed by a step that is applied for a different reason.


#### N4 bias field correction

| Field | Value |
| --- | --- |
| Volumes tested | 3 |
| Mean time per volume | 0.48 s |
| Projected time for all 447 series | 0.06 h |
| Mean inhomogeneity before N4 | 0.1566 |
| Mean inhomogeneity after N4 | 0.1566 |

**Not applied in Sprint 1 - implemented and available.** The decision is based on effect, not cost: at `shrink_factor=4` N4 takes only 0.48 s per volume (0.06 h projected for all 447 series), which would be affordable. But the measured effect is negligible - the inhomogeneity metric moved by +0.00005 (+0.03%), and on one of the three test volumes it got slightly *worse*. Compared with the ~50% reduction that CLAHE already delivers, N4 adds nothing measurable here. `correct_bias_field()` is implemented in `src/preprocessing/transforms.py` and can be enabled via `PreprocessConfig.bias_field_correction` if a later sprint finds evidence that it helps segmentation accuracy.


### 6. Mask handling

**Nearest-neighbour interpolation only.** Labels are categorical, so any averaging interpolation would fabricate label values that do not exist (e.g. blending vertebra 3 and vertebra 4 into 3.5). The geometric transform is applied to the raw label values first and the semantic/instance remap happens afterwards, so resizing never sees a collapsed label space. `build_processed_dataset()` records the label set before and after for every series and flags any invented label as a hard failure.


## Per-series spread (first 20 rows)

Individual series vary; the decisions above are based on the sample mean, and the full per-series scores are in `preprocessing_choices.csv`.

| image_id | variant | noise_sigma | boundary_sharpness | class_contrast_cnr | dynamic_range_std | bias_inhomogeneity |
| --- | --- | --- | --- | --- | --- | --- |
| 251_t2 | normalised only | 0.01093 | 0.08913 | 1.537 | 0.2471 | 0.1304 |
| 251_t2 | median 3x3 | 0.00761 | 0.0684 | 1.731 | 0.2448 | 0.1313 |
| 251_t2 | gaussian 3x3 | 0.00213 | 0.06434 | 1.725 | 0.2394 | 0.1304 |
| 251_t2 | bilateral | 0.0067 | 0.07338 | 1.677 | 0.2443 | 0.1306 |
| 251_t2 | CLAHE only | 0.01846 | 0.141 | 1.526 | 0.2747 | 0.05654 |
| 251_t2 | median + CLAHE | 0.01095 | 0.09798 | 1.806 | 0.2599 | 0.07722 |
| 251_t2 | gaussian + CLAHE | 0.00507 | 0.1095 | 1.703 | 0.2659 | 0.05554 |
| 20_t2 | normalised only | 0.00951 | 0.1114 | 2.11 | 0.2813 | 0.1373 |
| 20_t2 | median 3x3 | 0.00814 | 0.09096 | 2.196 | 0.2783 | 0.1369 |
| 20_t2 | gaussian 3x3 | 0.00191 | 0.08625 | 2.197 | 0.2742 | 0.1373 |
| 20_t2 | bilateral | 0.00587 | 0.09812 | 2.205 | 0.2784 | 0.1375 |
| 20_t2 | CLAHE only | 0.01395 | 0.131 | 2.509 | 0.298 | 0.1009 |
| 20_t2 | median + CLAHE | 0.01043 | 0.1038 | 2.581 | 0.2885 | 0.1108 |
| 20_t2 | gaussian + CLAHE | 0.00394 | 0.1051 | 2.592 | 0.2864 | 0.08725 |
| 225_t2 | normalised only | 0.00964 | 0.1245 | 2.038 | 0.2568 | 0.2021 |
| 225_t2 | median 3x3 | 0.00636 | 0.1027 | 2.22 | 0.255 | 0.2037 |
| 225_t2 | gaussian 3x3 | 0.00175 | 0.09416 | 2.225 | 0.2502 | 0.2021 |
| 225_t2 | bilateral | 0.00563 | 0.1118 | 2.166 | 0.2552 | 0.2024 |
| 225_t2 | CLAHE only | 0.02617 | 0.1637 | 2.05 | 0.2561 | 0.07056 |
| 225_t2 | median + CLAHE | 0.01127 | 0.1248 | 2.403 | 0.2372 | 0.1015 |

_...260 more rows_

