# Train / Validation / Test Split Report

*Patient-level split - Sprint 1*

Generated: 2026-09-22 00:37:32


## 1. Splitting strategy

- **Split unit: `patient_id`.** Every series and every slice of a patient lands in exactly one split.
- **Seed: 42** - fixed, so the split is reproducible.
- **Requested proportions of patients:** {'train': 0.7, 'val': 0.15, 'test': 0.15}.
- **Stratified:** True (by sex and annotated-vertebra count, with rare combinations pooled so every stratum can be divided three ways).

**Why not split slices at random.** A single series contributes 8-154 adjacent sagittal slices that are nearly identical to their neighbours, and a patient contributes 1-3 series of the *same* anatomy. Dataset inspection also found that the T1 and T2 masks of a patient are frequently byte-identical - one annotation shared across co-registered series. Random slice splitting would therefore place near-copies of the same image in both training and test sets and inflate the eventual Dice/IoU scores. Splitting on the patient removes that path entirely.


## 2. Resulting counts

| split | patients | patients_pct | series | series_pct | slices | slices_pct |
| --- | --- | --- | --- | --- | --- | --- |
| train | 152 | 69.7 | 319 | 71.4 | 9128 | 73.5 |
| val | 33 | 15.1 | 64 | 14.3 | 1632 | 13.1 |
| test | 33 | 15.1 | 64 | 14.3 | 1655 | 13.3 |


## 3. Leakage verification

| Field | Value |
| --- | --- |
| No patient in more than one split | True |
| Overlapping patients | none |
| Patients per split | `test`=33, `train`=152, `val`=33 |
| Sum of per-split patient counts | 218 |
| Distinct patients in the dataset | 218 |

The last two rows matching is the arithmetic proof that the split is a true partition: no patient was counted twice and none was dropped.


## 4. Composition per split


### train

| Field | Value |
| --- | --- |
| Patients | 152 |
| Series | 319 |
| Modalities | `t2`=147, `t1`=140, `t2_SPACE`=32 |
| Sex | `F`=195, `M`=124 |
| Mean annotated vertebrae | 7.01 |
| Mean annotated discs | 7.07 |


### val

| Field | Value |
| --- | --- |
| Patients | 33 |
| Series | 64 |
| Modalities | `t2`=32, `t1`=28, `t2_SPACE`=4 |
| Sex | `F`=40, `M`=24 |
| Mean annotated vertebrae | 6.91 |
| Mean annotated discs | 6.95 |


### test

| Field | Value |
| --- | --- |
| Patients | 33 |
| Series | 64 |
| Modalities | `t2`=31, `t1`=28, `t2_SPACE`=5 |
| Sex | `F`=41, `M`=23 |
| Mean annotated vertebrae | 6.97 |
| Mean annotated discs | 6.98 |


## 5. Relationship to the dataset's own subset column

`overview.csv` ships a `subset` column, but it only distinguishes training from validation (no test set) and is defined per *series*. Sprint 1 therefore derives its own patient-level three-way split. The overlap between the two is shown below for reference.

| subset | test | train | val |
| --- | --- | --- | --- |
| training | 47 | 260 | 53 |
| validation | 17 | 59 | 11 |


## 6. Files

- `data/processed/splits.csv` - series-level split assignment
- `data/processed/slice_index.csv` - per-slice index including `split`
- `outputs/visualizations/split_distribution.png` - split chart

**No model has been trained.** The test split has not been looked at beyond counting it, and no Dice or IoU value exists yet.

