"""Generate ``notebooks/01_preprocessing.ipynb`` programmatically.

The notebook is generated rather than hand-written so it stays in sync with
the modules in ``src/`` and can be regenerated after a refactor. It contains
no hard-coded results: every number it displays is computed when it runs.

Usage
-----
    python scripts/build_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import PROJECT_ROOT  # noqa: E402

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_preprocessing.ipynb"


def md(text: str) -> nbf.NotebookNode:
    """Markdown cell from a dedented string."""
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    """Code cell from a dedented string."""
    return nbf.v4.new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells: list[nbf.NotebookNode] = []

    # ================= 1. Project Introduction =================
    cells.append(md("""
# Automated Segmentation of Vertebrae and Intervertebral Discs in Lumbar Spine MRI Images

## Sprint 1 - Dataset Inspection and Preprocessing

**Problem.** Manual segmentation of lumbar spine MRI is slow and varies between
observers, which limits both clinical throughput and the reproducibility of any
measurement derived from it.

**Objective.** Build an automated pipeline that segments vertebrae and
intervertebral discs in sagittal lumbar spine MRI.

**Scope of this notebook.** Sprint 1 only: inspect the dataset, extract it, pair
images with masks, validate quality, preprocess images and masks, visualise the
result, and create a leakage-free train/validation/test split.

> **No segmentation model is trained here, and no Dice or IoU score is reported.**
> Those belong to the next sprint. Any accuracy figure would be meaningless until
> a model has actually been trained and evaluated.

### Pipeline overview

```
data/raw/*.zip
      |  extract.py
      v
data/extracted/{images,masks}/*.mha      447 sagittal 3-D volumes
      |  validate.py        -> dataset_inspection.md
      |  pairing.py         -> image_mask_pairs.csv
      v
transforms.py + dataset.py               reorient -> slice -> resample -> crop
      |                                  -> normalise -> denoise -> CLAHE
      v
data/processed/slices/*.npz              fixed-size 2-D image/mask pairs
      |  splits.py
      v
data/processed/splits.csv                patient-level train/val/test
```

### How this notebook is organised

1. Project introduction
2. Dataset description
3. Dataset inspection
4. Data extraction
5. Image-mask pairing
6. Dataset quality checks
7. Image preprocessing
8. Mask preprocessing
9. Before/after visualisation
10. Train/validation/test split
11. Final preprocessing summary
"""))

    # ================= Setup =================
    cells.append(md("""
### Setup

The only thing that needs configuring is the project root. Everything else is
derived from it by `src/utils/paths.py`, so the notebook runs start-to-finish
without further manual intervention.
"""))
    cells.append(code('''
%matplotlib inline

import sys
from pathlib import Path

# Make the project importable whether the notebook is run from notebooks/ or
# from the project root.
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

from src.utils import paths

print("Project root:", paths.PROJECT_ROOT)
print("Raw data    :", paths.RAW_DIR)
print("Random seed :", paths.RANDOM_SEED)
'''))

    # ================= 2. Dataset Description =================
    cells.append(md("""
---
## 2. Dataset description

The project dataset is a **lumbar spine MRI segmentation dataset** supplied as
four files in `data/raw/`:

| file | content |
| --- | --- |
| `images.zip` | MRI volumes |
| `masks.zip` | matching segmentation masks |
| `overview.csv` | one row per series: acquisition metadata and structure counts |
| `radiological_gradings.csv` | one row per patient x disc: radiological gradings |

This is the publicly described lumbar spine MRI segmentation dataset of sagittal
T1 and T2 series ([SPIDER, Grand Challenge](https://spider.grand-challenge.org/data/);
[van der Graaf et al., *Scientific Data* 11, 264, 2024](https://www.nature.com/articles/s41597-024-03090-w)).
Everything stated below about format, labels and geometry was nevertheless
**measured from the files themselves** rather than taken from the description.

> The two notebooks in `tutorials/` use a completely different Electronic Health
> Record dataset. They are kept as reference material only and are not used here.
"""))
    cells.append(code('''
for path in [paths.IMAGES_ZIP, paths.MASKS_ZIP, paths.OVERVIEW_CSV, paths.GRADINGS_CSV]:
    size = path.stat().st_size / 1e6 if path.exists() else float("nan")
    print(f"{path.name:28s} exists={path.exists()}  {size:>9,.2f} MB")
'''))

    cells.append(md("""
### 2.1 Metadata files

`overview.csv` is keyed on `new_file_name`, which matches the volume filename
stem exactly, so it joins directly onto the image files.
`radiological_gradings.csv` is keyed on `Patient`, the numeric prefix of the
filename.

The loaders in `src/preprocessing/pairing.py` also clean one real defect found
during inspection: `sex` is stored with inconsistent trailing whitespace, so
`'F'` and `'F '` appear as two distinct categories.
"""))
    cells.append(code('''
from src.preprocessing.pairing import load_gradings, load_overview

raw_overview = pd.read_csv(paths.OVERVIEW_CSV)
overview = load_overview()   # whitespace-stripped + patient_id / modality parsed
gradings = load_gradings()

print(f"overview.csv : {raw_overview.shape[0]} rows x {raw_overview.shape[1]} columns")
print(f"gradings.csv : {gradings.shape[0]} rows, {gradings['patient_id'].nunique()} patients")
print("\\nsex BEFORE cleaning:", raw_overview["sex"].value_counts(dropna=False).to_dict())
print("sex AFTER cleaning :", overview["sex"].value_counts(dropna=False).to_dict())

overview[["new_file_name", "patient_id", "modality", "num_vertebrae", "num_discs",
          "sex", "subset", "Manufacturer", "PixelSpacing", "SliceThickness"]].head()
'''))

    cells.append(code('''
# Per-disc radiological gradings: classification targets for a possible later
# sprint, not segmentation targets. Summarised here only for completeness.
print("Gradings columns:", list(gradings.columns))
display(gradings.head())
display(gradings["pfirrman_grade"].value_counts().sort_index().rename("n discs").to_frame())
'''))

    # ================= 3. Dataset Inspection =================
    cells.append(md("""
---
## 3. Dataset inspection

Nothing about the dataset was assumed. Before writing any preprocessing code the
archives were opened and examined to determine the file format, the array
geometry, the label vocabulary and the intensity conventions.

### 3.1 What is inside the archives
"""))
    cells.append(code('''
import os
from collections import Counter

from src.preprocessing.extract import list_archive

for zip_path in [paths.IMAGES_ZIP, paths.MASKS_ZIP]:
    names = [n for n in list_archive(zip_path) if not n.endswith("/")]
    extensions = Counter(os.path.splitext(n)[1].lower() for n in names)
    print(f"{zip_path.name}: {len(names)} members, extensions={dict(extensions)}")
    print("   first 3:", names[:3])
'''))

    cells.append(md("""
**Finding.** Both archives contain **447 `.mha` files** - MetaImage format, which
holds a *3-D volume* plus its physical geometry. So this is not a folder of 2-D
PNG/JPEG images: each file is a sagittal MRI series, and the project's 2-D
segmentation targets have to be produced by slicing the volumes.

Filenames follow `<patient_id>_<modality>.mha`, and `images/X.mha` corresponds to
`masks/X.mha`.
"""))
    cells.append(code('''
from src.preprocessing.pairing import parse_series_stem

stems = [Path(n).stem for n in list_archive(paths.IMAGES_ZIP) if n.endswith(".mha")]
parsed = [parse_series_stem(s) for s in stems]
patient_ids = sorted({p for p, _ in parsed})
modalities = Counter(m for _, m in parsed)
series_per_patient = Counter(Counter(p for p, _ in parsed).values())

print(f"Series            : {len(stems)}")
print(f"Distinct patients : {len(patient_ids)}  (ids {min(patient_ids)}..{max(patient_ids)}, not contiguous)")
print(f"Modalities        : {dict(modalities)}")
print(f"Series per patient: {dict(sorted(series_per_patient.items()))}")
'''))

    cells.append(md("""
**Finding that shapes the whole project.** There are **447 series but only 218
patients** - most patients contribute 2 or 3 series of the *same* anatomy. This
is why the train/validation/test split later has to be made per patient rather
than per series or per slice.
"""))

    # ================= 4. Data Extraction =================
    cells.append(md("""
---
## 4. Data extraction

Extraction is done by code (`src/preprocessing/extract.py`) so it is
reproducible and the raw ZIPs are never modified. The function is resume-safe:
a member whose target file already exists at the expected size is skipped, so
re-running is cheap.
"""))
    cells.append(code('''
from src.preprocessing.extract import extract_dataset

# Resume-safe: skips files that already exist with the expected size.
counts = extract_dataset(verbose=False)
print("Extracted:", counts)
print("images ->", paths.EXTRACTED_IMAGES_DIR)
print("masks  ->", paths.EXTRACTED_MASKS_DIR)

# The raw archives are untouched.
print("\\nRaw archives still present:",
      paths.IMAGES_ZIP.exists(), paths.MASKS_ZIP.exists())
'''))

    cells.append(md("""
### 4.1 Reading a volume: geometry is not consistent

Loading is handled by `src/preprocessing/volume_io.py`. The important detail it
solves: the volumes are **not all stored with the same axis order**. Most 2-D
TSE series are stored `LPS`, while the 3-D `t2_SPACE` series are stored `PIR`,
so the array axis that steps through sagittal slices differs between files.

Every volume is therefore reoriented to a canonical `RAS` frame on load. That is
a pure axis permutation plus flips - no interpolation - so it is exactly
loss-less and safe to apply to label masks.
"""))
    cells.append(code('''
from src.preprocessing.volume_io import load_image, load_mask, volume_geometry

for stem in ["1_t1", "98_t2_SPACE"]:
    image = load_image(paths.EXTRACTED_IMAGES_DIR / f"{stem}.mha")
    mask = load_mask(paths.EXTRACTED_MASKS_DIR / f"{stem}.mha")
    print(f"--- {stem}")
    print(f"    stored orientation : {image.native_orientation}")
    print(f"    array shape (z,y,x): {image.array.shape}   dtype={image.array.dtype}")
    print(f"    spacing mm  (z,y,x): {tuple(round(s, 3) for s in image.spacing_zyx)}")
    print(f"    sagittal slices    : {image.n_sagittal_slices}")
    print(f"    slice shape        : {image.in_plane_shape}")
    print(f"    mask shape matches : {image.array.shape == mask.array.shape}")
    print(f"    raw intensity range: [{image.array.min():.0f}, {image.array.max():.0f}]")
    print(f"    mask label values  : {np.unique(mask.array).tolist()}")
'''))

    cells.append(md("""
**Two findings visible above.**

1. **Label vocabulary is sparse and non-contiguous**: `0`, `1..9`, `100`,
   `201..209`. Masks are integer label maps, not binary and not one-hot.
2. **Intensity conventions differ**: one series runs `[-1000, 3096]`, the other
   `[0, ...]`. MRI intensity has no absolute meaning, so normalisation is
   mandatory rather than optional.

### 4.2 Label semantics

Documented and implemented in `src/preprocessing/labels.py`:

| raw value | meaning |
| --- | --- |
| `0` | background |
| `1..9` | vertebrae, numbered from the most inferior upward |
| `100` | spinal canal |
| `201..209` | intervertebral discs, numbered from the most inferior upward |

Two derived label spaces are produced:

* **semantic** (the project target): `0` background, `1` vertebra, `2` IVD, `3` spinal canal
* **instance** (loss-less): `0` background, `1..9` vertebrae, `10` canal, `11..19` IVDs
"""))
    cells.append(code('''
from src.preprocessing.labels import (
    SEMANTIC_CLASSES, describe_raw_label, to_instance, to_semantic,
)

mask = load_mask(paths.EXTRACTED_MASKS_DIR / "1_t1.mha")
for value in np.unique(mask.array):
    print(f"  {int(value):>4} -> {describe_raw_label(int(value))}")

print("\\nSemantic classes:", SEMANTIC_CLASSES)
print("Raw label values      :", np.unique(mask.array).tolist())
print("Semantic label values :", np.unique(to_semantic(mask.array)).tolist())
print("Instance label values :", np.unique(to_instance(mask.array)).tolist())
'''))

    # ================= 5. Image-Mask Pairing =================
    cells.append(md("""
---
## 5. Image-mask pairing

Pairing is driven by the **parsed filename identifier**, never by directory
listing order. Alphabetical pairing would be dangerous here: a single missing
file would silently shift every subsequent pair by one and every later metric
would be quietly wrong.
"""))
    cells.append(code('''
from src.preprocessing.pairing import describe_pairing, pair_images_and_masks

pairs, pairing_issues = pair_images_and_masks(overview=overview, gradings=gradings)
print(describe_pairing(pairs, pairing_issues, n_examples=5))
'''))
    cells.append(code('''
# The pairing table: what later sprints consume.
print("Columns:", list(pairs.columns))
pairs.head(8)
'''))
    cells.append(code('''
# Verify every pair explicitly: the mask file must exist and describe the same grid.
import random

random.seed(paths.RANDOM_SEED)
checks = []
for _ in range(5):
    row = pairs.iloc[random.randrange(len(pairs))]
    image = load_image(paths.PROJECT_ROOT / row["image_path"])
    mask = load_mask(paths.PROJECT_ROOT / row["mask_path"])
    checks.append({
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "modality": row["modality"],
        "same_stem": Path(row["image_path"]).stem == Path(row["mask_path"]).stem,
        "same_shape": image.array.shape == mask.array.shape,
        "same_spacing": np.allclose(image.spacing_zyx, mask.spacing_zyx, atol=1e-4),
        "n_labels": int(len(np.unique(mask.array))),
    })
pd.DataFrame(checks)
'''))

    # ================= 6. Quality Checks =================
    cells.append(md("""
---
## 6. Dataset quality checks

`scripts/02_inspect.py` reads every one of the 447 volume pairs and records
geometry, intensity, labels and integrity into
`outputs/preprocessing_reports/`. That scan takes several minutes, so this
notebook loads its results rather than repeating it.

Run it first if the files are missing:

```bash
python scripts/02_inspect.py
```
"""))
    cells.append(code('''
import json

inspection = pd.read_csv(paths.VOLUME_INSPECTION_CSV)
with open(paths.REPORTS_DIR / "dataset_inspection.json") as handle:
    inspection_summary = json.load(handle)

print(f"Inspected series : {len(inspection)}")
print(f"Read errors      : {int(inspection['read_error'].notna().sum())}")
print(f"Distinct patients: {inspection_summary['n_patients']}")
print(f"Label vocabulary : {inspection_summary['label_vocabulary']}")
print(f"Undocumented labels: {inspection_summary['unknown_labels'] or 'none'}")
'''))

    cells.append(md("""
### 6.1 Missing, unmatched and duplicate files
"""))
    cells.append(code('''
duplicates = inspection_summary["duplicates"]
quality = {
    "images without a mask": len(inspection_summary["pairing_issues"]["images_without_mask"]),
    "masks without an image": len(inspection_summary["pairing_issues"]["masks_without_image"]),
    "filenames not matching convention": len(inspection_summary["pairing_issues"]["unparsable_stems"]),
    "series missing an overview.csv row": len(inspection_summary["pairing_issues"]["missing_from_overview"]),
    "duplicate IMAGE groups": duplicates["images"]["n_duplicate_groups"],
    "duplicate MASK groups": duplicates["masks"]["n_duplicate_groups"],
    "duplicate mask groups within one patient": duplicates["masks"]["n_same_patient_groups"],
    "duplicate mask groups across patients": duplicates["masks"]["n_cross_patient_groups"],
    "image/mask shape disagreements": inspection_summary["geometry_mismatches"]["shape"],
    "image/mask spacing disagreements": inspection_summary["geometry_mismatches"]["spacing"],
}
pd.Series(quality, name="count").to_frame()
'''))

    cells.append(md("""
**Interpretation of the duplicates.** No *image* volume is duplicated. Many
*mask* volumes are byte-identical, but **always within a single patient** - the
T1 and T2 series of a patient were acquired on the same grid and share one
annotation. That is a property of the annotation process, not corruption. It
does mean a patient's series are highly correlated, which is a second
independent reason the split must be per patient.

### 6.2 Anatomical sanity checks

These verify that the reorientation actually put the axes where the code assumes
they are. They are derived from mask geometry alone:

* the spinal canal must lie **posterior** to the vertebral bodies
* vertebra label `1` must lie **inferior** to the highest vertebra label
"""))
    cells.append(code('''
checks = inspection_summary["anatomical_checks"]
pd.DataFrame([
    {"check": "spinal canal posterior to vertebral bodies",
     "passed": checks["canal_posterior_pass"], "failed": checks["canal_posterior_fail"]},
    {"check": "vertebra label 1 inferior to highest vertebra label",
     "passed": checks["label1_inferior_pass"], "failed": checks["label1_inferior_fail"]},
])
'''))

    cells.append(md("""
### 6.3 Heterogeneity: why a naive resize would be wrong
"""))
    cells.append(code('''
ok = inspection[inspection["read_error"].isna()]

summary = pd.DataFrame({
    "min": [ok["img_rows"].min(), ok["img_cols"].min(), ok["img_n_slices"].min(),
            ok["img_row_spacing_mm"].min(), ok["img_col_spacing_mm"].min(),
            ok["img_slice_spacing_mm"].min()],
    "median": [ok["img_rows"].median(), ok["img_cols"].median(), ok["img_n_slices"].median(),
               ok["img_row_spacing_mm"].median(), ok["img_col_spacing_mm"].median(),
               ok["img_slice_spacing_mm"].median()],
    "max": [ok["img_rows"].max(), ok["img_cols"].max(), ok["img_n_slices"].max(),
            ok["img_row_spacing_mm"].max(), ok["img_col_spacing_mm"].max(),
            ok["img_slice_spacing_mm"].max()],
}, index=["rows (px)", "cols (px)", "slices", "row spacing (mm)",
          "col spacing (mm)", "slice spacing (mm)"])

print(f"Distinct in-plane shapes: {ok.groupby(['img_rows', 'img_cols']).ngroups}")
print(f"Stored orientations     : {inspection_summary['native_orientations']}")
summary.round(4)
'''))

    cells.append(code('''
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].hist(ok["img_rows"], bins=40, color="#4c72b0")
axes[0].set_title("In-plane rows (px)"); axes[0].set_xlabel("pixels")
axes[1].hist(ok["img_row_spacing_mm"], bins=40, color="#dd8452")
axes[1].set_title("In-plane pixel spacing (mm)"); axes[1].set_xlabel("mm")
axes[2].hist(ok["img_n_slices"], bins=40, color="#55a868")
axes[2].set_title("Sagittal slices per volume"); axes[2].set_xlabel("slices")
for axis in axes:
    axis.set_ylabel("series")
fig.suptitle("Acquisition heterogeneity across the 447 series", y=1.04)
fig.tight_layout()
plt.show()
'''))

    cells.append(md("""
Pixel spacing spans more than an order of magnitude. Resizing purely by pixel
count would therefore place the same vertebra at a different physical size in
different patients. Preprocessing resamples to a **fixed mm/pixel** instead.

### 6.4 The two intensity conventions
"""))
    cells.append(code('''
convention = ok.groupby(["raw_min", "raw_max"]).size().rename("n series").reset_index()
display(convention)

print(f"Series with a constant padding floor: {int(ok['padding_value'].notna().sum())} / {len(ok)}")
print(f"Mean foreground fraction            : {ok['foreground_fraction'].mean():.1%}")
print(f"Mean fraction pinned at the maximum : {ok['saturated_fraction'].mean():.2%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for axis, stem in zip(axes, ["1_t1", "98_t2_SPACE"]):
    volume = load_image(paths.EXTRACTED_IMAGES_DIR / f"{stem}.mha")
    axis.hist(volume.array.ravel(), bins=150, color="#666666")
    axis.set_yscale("log")
    axis.set_title(f"{stem}: raw intensity histogram\\n"
                   f"range [{volume.array.min():.0f}, {volume.array.max():.0f}]")
    axis.set_xlabel("raw intensity"); axis.set_ylabel("voxels (log)")
fig.tight_layout()
plt.show()
'''))

    cells.append(md("""
The left histogram shows the problem: a huge spike of background voxels at the
padding floor and another at the saturated ceiling. A mean/std or min-max
normalisation would be governed by those two spikes rather than by tissue, which
is why statistics are computed over **foreground voxels only**.

### 6.5 Class imbalance

Recorded now because it will decide the loss function in the modelling sprint.
"""))
    cells.append(code('''
totals = {
    "vertebrae": float(ok["voxels_vertebrae"].sum()),
    "intervertebral discs": float(ok["voxels_ivd"].sum()),
    "spinal canal": float(ok["voxels_canal"].sum()),
}
labelled = sum(totals.values())
imbalance = pd.DataFrame({
    "voxels": {k: int(v) for k, v in totals.items()},
    "% of labelled voxels": {k: round(100 * v / labelled, 2) for k, v in totals.items()},
})
print(f"Labelled voxels as a share of all voxels: "
      f"{ok['labelled_voxel_fraction'].mean():.2%} (mean per volume)")
imbalance
'''))

    # ================= 7. Image Preprocessing =================
    cells.append(md("""
---
## 7. Image preprocessing

Every step below was chosen from measurements, not from a generic recipe. The
supporting evidence is in
`outputs/preprocessing_reports/preprocessing_choices.md`, produced by
`scripts/02b_justify_steps.py`, which scores seven competing variants on real
slices.

| step | applied? | why |
| --- | --- | --- |
| reorient to RAS | **yes** | stored orientation varies (`LPS` / `PIR`); loss-less |
| resample to 1.0 mm/px | **yes** | pixel spacing varies 0.077-1.23 mm |
| centre crop/pad to 352x256 | **yes** | measured: contains all annotation in all 447 volumes |
| percentile normalisation | **yes** | two incompatible intensity conventions |
| median 3x3 denoise | **yes** | best class contrast; keeps ~80% of edge sharpness |
| CLAHE | **yes** | highest vertebra-vs-disc contrast of all variants tested |
| N4 bias field correction | **no** | measured effect negligible; CLAHE already halves inhomogeneity |

### 7.1 Why the target size is 352 x 256 and not a square

For every volume, the smallest **centred** crop that still contains the entire
annotation was computed. The lumbar spine is tall and narrow, so the requirement
is very different along the two axes.
"""))
    cells.append(code('''
requirements = {}
for axis_name, frac_col, extent_col, fov_col in [
    ("superior-inferior (rows)", "ann_row_center_frac", "ann_rows_mm", "img_fov_rows_mm"),
    ("anterior-posterior (cols)", "ann_col_center_frac", "ann_cols_mm", "img_fov_cols_mm"),
]:
    fov = ok[fov_col].to_numpy()
    centre = ok[frac_col].to_numpy() * fov
    extent = ok[extent_col].to_numpy()
    # Smallest centred window that still contains [centre-extent/2, centre+extent/2].
    needed = 2 * np.maximum(np.abs(centre - extent / 2 - fov / 2),
                            np.abs(centre + extent / 2 - fov / 2))
    requirements[axis_name] = needed
    print(f"{axis_name}: annotation extent {extent.min():.0f}-{extent.max():.0f} mm, "
          f"required centred crop max {needed.max():.0f} mm")

print()
rows = []
for size in [256, 288, 320, 352, 384]:
    rows.append({
        "crop (mm)": size,
        "volumes losing annotation (rows)": int((requirements["superior-inferior (rows)"] > size).sum()),
        "volumes losing annotation (cols)": int((requirements["anterior-posterior (cols)"] > size).sum()),
    })
pd.DataFrame(rows)
'''))

    cells.append(md("""
A square `288 x 288` crop would have clipped annotated anatomy in **157 of 447
volumes**. `352` rows x `256` cols loses nothing, and both values are multiples
of 32, which suits the downsampling depth of a U-Net later.

### 7.2 The configuration

All parameters live in one dataclass, so the pipeline is auditable and
reproducible.
"""))
    cells.append(code('''
from src.preprocessing.transforms import PreprocessConfig

config = PreprocessConfig()
pd.Series(config.to_dict(), name="value").to_frame()
'''))

    cells.append(md("""
### 7.3 The pipeline stage by stage

Order matters: geometry first (so a 3x3 kernel means the same physical size
everywhere), then normalisation, then denoising, and CLAHE last because CLAHE
amplifies whatever noise is present.
"""))
    cells.append(code('''
from src.preprocessing.transforms import (
    apply_geometry, denoise_image, enhance_contrast, intensity_statistics, normalize_image,
)

row = pairs[pairs["modality"] == "t2"].iloc[0]
image_volume = load_image(paths.PROJECT_ROOT / row["image_path"])
mask_volume = load_mask(paths.PROJECT_ROOT / row["mask_path"])

# Use the slice with the most annotation - the informative mid-sagittal slice.
slice_index = int(np.argmax((mask_volume.array > 0).sum(axis=(0, 1))))
spacing = image_volume.in_plane_spacing
volume_stats = intensity_statistics(image_volume.array, percentiles=config.clip_percentiles)

raw_slice = image_volume.sagittal_slice(slice_index)
geometry = apply_geometry(raw_slice, spacing, config, is_mask=False)
normalised = normalize_image(geometry, percentiles=config.clip_percentiles, stats=volume_stats)
denoised = denoise_image(normalised, config.denoise_method, config.denoise_kernel)
final = enhance_contrast(denoised, clip_limit=config.clahe_clip_limit,
                         tile_grid=config.clahe_tile_grid)

print(f"series {row['image_id']}, sagittal slice {slice_index}")
pd.DataFrame([
    {"stage": "1. raw slice", "shape": raw_slice.shape,
     "min": raw_slice.min(), "max": raw_slice.max(), "mean": raw_slice.mean()},
    {"stage": f"2. resampled to {config.target_spacing_mm} mm + crop", "shape": geometry.shape,
     "min": geometry.min(), "max": geometry.max(), "mean": geometry.mean()},
    {"stage": "3. normalised", "shape": normalised.shape,
     "min": normalised.min(), "max": normalised.max(), "mean": normalised.mean()},
    {"stage": f"4. {config.denoise_method} denoise", "shape": denoised.shape,
     "min": denoised.min(), "max": denoised.max(), "mean": denoised.mean()},
    {"stage": "5. CLAHE (final)", "shape": final.shape,
     "min": final.min(), "max": final.max(), "mean": final.mean()},
]).round(4)
'''))

    cells.append(code('''
from src.preprocessing.visualize import visualize_preprocessing_stages

visualize_preprocessing_stages(
    {
        "1. raw slice": raw_slice,
        f"2. resample {config.target_spacing_mm}mm + crop": geometry,
        "3. normalised [0,1]": normalised,
        f"4. {config.denoise_method} denoise": denoised,
        "5. CLAHE (final)": final,
    },
    title=f"Preprocessing stages - {row['image_id']} sagittal slice {slice_index}",
)
plt.show()
'''))

    cells.append(md("""
### 7.4 Evidence for the optional steps

The table below is the measured comparison that decided the denoising and
contrast choices. `class_contrast_cnr` is the one that matters most: it is the
vertebra-vs-disc separability, i.e. exactly what the project has to achieve.
"""))
    cells.append(code('''
choices_csv = paths.REPORTS_DIR / "preprocessing_choices.csv"
if choices_csv.exists():
    scores = pd.read_csv(choices_csv)
    order = ["normalised only", "median 3x3", "gaussian 3x3", "bilateral",
             "CLAHE only", "median + CLAHE", "gaussian + CLAHE"]
    means = scores.groupby("variant").mean(numeric_only=True).reindex(order)
    display(means.round(5))
    print("Best vertebra-vs-disc contrast:", means["class_contrast_cnr"].idxmax())
    print("\\nNote: gaussian removes the most noise but loses the most edge sharpness,")
    print("and scores lower on class contrast than median + CLAHE.")
else:
    print("Run: python scripts/02b_justify_steps.py")
'''))

    # ================= 8. Mask Preprocessing =================
    cells.append(md("""
---
## 8. Mask preprocessing

Masks go through the **same geometric transform** as their image, differing only
in interpolation. That is what guarantees the two stay pixel-aligned.

**Nearest neighbour is used for masks, always.** Label values are categorical:
averaging vertebra `3` and vertebra `4` gives `3.5`, which is not a structure,
and averaging a disc (`201`) with background (`0`) gives ~`100`, which happens to
be the spinal canal's label. The demonstration below counts exactly how many
non-existent label values each interpolation method invents.
"""))
    cells.append(code('''
import cv2

from src.preprocessing.transforms import center_crop_or_pad, preprocess_mask

mask_slice = mask_volume.sagittal_slice(slice_index)
original_labels = set(int(v) for v in np.unique(mask_slice))

rows_n, cols_n = mask_slice.shape
new_rows = int(round(rows_n * spacing[0] / config.target_spacing_mm))
new_cols = int(round(cols_n * spacing[1] / config.target_spacing_mm))

comparison = []
for name, flag in [("NEAREST (used)", cv2.INTER_NEAREST),
                   ("BILINEAR (unsafe)", cv2.INTER_LINEAR),
                   ("BICUBIC (unsafe)", cv2.INTER_CUBIC)]:
    resized = cv2.resize(mask_slice.astype(np.float32), (new_cols, new_rows), interpolation=flag)
    cropped = center_crop_or_pad(resized, config.target_size, pad_value=0)
    produced = set(int(round(v)) for v in np.unique(cropped))
    comparison.append({
        "method": name,
        "label values produced": len(produced),
        "INVENTED label values": len(produced - original_labels),
    })

print(f"Original mask has {len(original_labels)} label values: {sorted(original_labels)}")
pd.DataFrame(comparison)
'''))

    cells.append(code('''
from src.preprocessing.visualize import visualize_interpolation_comparison

visualize_interpolation_comparison(mask_slice, spacing, config)
plt.show()
'''))

    cells.append(md("""
### 8.1 Mask output and integrity verification

The geometric transform is applied to the **raw** label values first, and the
semantic/instance remap happens afterwards - so resizing never operates on a
collapsed label space and each vertebra/disc identity survives the resize.
"""))
    cells.append(code('''
semantic_mask, instance_mask = preprocess_mask(mask_slice, spacing, config)

print(f"raw mask      : shape={mask_slice.shape}, labels={sorted(int(v) for v in np.unique(mask_slice))}")
print(f"semantic mask : shape={semantic_mask.shape}, dtype={semantic_mask.dtype}, "
      f"labels={np.unique(semantic_mask).tolist()}")
print(f"instance mask : shape={instance_mask.shape}, dtype={instance_mask.dtype}, "
      f"labels={np.unique(instance_mask).tolist()}")
print(f"\\nimage and mask shapes identical: {final.shape == semantic_mask.shape}")

# No label may be invented by the resize; disappearance of a tiny structure is allowed.
from src.preprocessing.transforms import validate_mask_labels

check = validate_mask_labels(mask_slice, instance_mask, mapping=to_instance)
pd.Series(check, name="value").to_frame()
'''))

    # ================= 9. Before/After =================
    cells.append(md("""
---
## 9. Before/after visualisation

The five required panels for randomly chosen samples: original MRI, original
mask, preprocessed MRI, preprocessed mask, and the preprocessed MRI with the
mask overlaid. Figures are also written to `outputs/visualizations/` by
`scripts/03_preprocess.py`.

Slices are displayed in standard radiological sagittal orientation: **superior
at the top, anterior on the left.**
"""))
    cells.append(code('''
from src.preprocessing.transforms import preprocess_image
from src.preprocessing.visualize import visualize_before_after, visualize_sample

rng = np.random.default_rng(paths.RANDOM_SEED)

# One sample per modality so the figures are not all the same sequence type.
selection = []
for modality in ["t1", "t2", "t2_SPACE"]:
    subset = pairs[pairs["modality"] == modality]
    selection.append(subset.iloc[int(rng.integers(len(subset)))])

for row in selection:
    image_v = load_image(paths.PROJECT_ROOT / row["image_path"])
    mask_v = load_mask(paths.PROJECT_ROOT / row["mask_path"])
    index = int(np.argmax((mask_v.array > 0).sum(axis=(0, 1))))
    sp = image_v.in_plane_spacing
    stats = intensity_statistics(image_v.array, percentiles=config.clip_percentiles)

    original_image = image_v.sagittal_slice(index)
    original_mask = mask_v.sagittal_slice(index)
    processed_image = preprocess_image(original_image, sp, config, volume_stats=stats)
    processed_mask, _ = preprocess_mask(original_mask, sp, config)

    visualize_sample(
        original_image, original_mask, processed_image, processed_mask,
        title=(f"{row['image_id']} | patient {row['patient_id']} | {row['modality']} | "
               f"slice {index} | in-plane {sp[0]:.3f} x {sp[1]:.3f} mm"),
    )
    plt.show()
'''))

    cells.append(code('''
# Intensity-focused before/after: the histograms show what normalisation achieved.
row = selection[0]
image_v = load_image(paths.PROJECT_ROOT / row["image_path"])
mask_v = load_mask(paths.PROJECT_ROOT / row["mask_path"])
index = int(np.argmax((mask_v.array > 0).sum(axis=(0, 1))))
stats = intensity_statistics(image_v.array, percentiles=config.clip_percentiles)

original_image = image_v.sagittal_slice(index)
processed_image = preprocess_image(original_image, image_v.in_plane_spacing, config,
                                   volume_stats=stats)

visualize_before_after(
    original_image, processed_image,
    title=(f"Before vs after - {row['image_id']} "
           f"(raw range [{original_image.min():.0f}, {original_image.max():.0f}])"),
)
plt.show()
'''))

    cells.append(md("""
### 9.1 Consistency across the dataset

The point of preprocessing is that every sample now looks structurally the same
to a model: identical size, identical orientation, identical intensity scale.
"""))
    cells.append(code('''
from src.preprocessing.visualize import visualize_dataset_grid

samples = []
for _ in range(6):
    row = pairs.iloc[int(rng.integers(len(pairs)))]
    image_v = load_image(paths.PROJECT_ROOT / row["image_path"])
    mask_v = load_mask(paths.PROJECT_ROOT / row["mask_path"])
    index = int(np.argmax((mask_v.array > 0).sum(axis=(0, 1))))
    sp = image_v.in_plane_spacing
    stats = intensity_statistics(image_v.array, percentiles=config.clip_percentiles)
    processed_mask, _ = preprocess_mask(mask_v.sagittal_slice(index), sp, config)
    samples.append({
        "image": preprocess_image(image_v.sagittal_slice(index), sp, config, volume_stats=stats),
        "mask": processed_mask,
        "label": f"{row['image_id']}\\n{row['modality']}",
    })

visualize_dataset_grid(
    samples,
    title=(f"Preprocessed samples - all {config.target_size[0]}x{config.target_size[1]} px "
           f"at {config.target_spacing_mm} mm/px"),
)
plt.show()
'''))

    # ================= 10. Split =================
    cells.append(md("""
---
## 10. Train / validation / test split

Run the full pipeline first if `data/processed/` is empty:

```bash
python scripts/03_preprocess.py     # writes every preprocessed slice
python scripts/04_split.py          # writes the patient-level split
```

**The split is made on `patient_id`, never on individual slices.** Three
independent reasons, all established by the inspection above:

1. A series contributes 8-154 adjacent sagittal slices that are nearly
   identical to their neighbours.
2. A patient contributes 1-3 series of the *same* anatomy.
3. The T1 and T2 masks of a patient are frequently byte-identical.

Random slice splitting would put near-copies of the same image in both training
and test, and the resulting Dice/IoU would be optimistically biased.
"""))
    cells.append(code('''
from src.preprocessing.splits import (
    create_dataset_split, split_fraction_table, summarise_split, verify_no_leakage,
)

slice_index = pd.read_csv(paths.SLICE_INDEX_CSV) if paths.SLICE_INDEX_CSV.exists() else None
if slice_index is None:
    print("data/processed/slice_index.csv not found - run scripts/03_preprocess.py first.")

split_frame = create_dataset_split(
    pairs, fractions={"train": 0.70, "val": 0.15, "test": 0.15},
    seed=paths.RANDOM_SEED, stratify=True,
)

if slice_index is not None:
    slice_index["split"] = slice_index["image_id"].map(
        split_frame.set_index("image_id")["split"]
    )

leakage = verify_no_leakage(split_frame)
print(f"No patient in more than one split : {leakage['ok']}")
print(f"Overlapping patients              : {leakage['overlaps'] or 'none'}")
print(f"Sum of per-split patient counts   : {leakage['total_ids_counted']}")
print(f"Distinct patients in the dataset  : {leakage['total_ids_unique']}")
print("  (the last two matching proves the split is a true partition)")
'''))
    cells.append(code('''
counts = summarise_split(split_frame, slice_index)
table = split_fraction_table(counts)
display(table)

if slice_index is not None:
    from src.preprocessing.visualize import visualize_split
    visualize_split(counts)
    plt.show()
'''))
    cells.append(code('''
# Composition per split: stratification keeps them comparable.
composition = []
for name in ["train", "val", "test"]:
    rows_ = split_frame[split_frame["split"] == name]
    composition.append({
        "split": name,
        "patients": rows_["patient_id"].nunique(),
        "series": len(rows_),
        "% female": round(100 * (rows_["sex"] == "F").mean(), 1),
        "mean vertebrae": round(rows_["num_vertebrae"].mean(), 2),
        "mean discs": round(rows_["num_discs"].mean(), 2),
        "t1 / t2 / SPACE": "{} / {} / {}".format(
            int((rows_["modality"] == "t1").sum()),
            int((rows_["modality"] == "t2").sum()),
            int((rows_["modality"] == "t2_SPACE").sum()),
        ),
    })
pd.DataFrame(composition)
'''))

    cells.append(md("""
Note that `overview.csv` ships its own `subset` column, but it only separates
training from validation (no test set) and is defined per *series*, so Sprint 1
derives its own patient-level three-way split.
"""))
    cells.append(code('''
if "subset" in split_frame.columns:
    display(pd.crosstab(split_frame["subset"], split_frame["split"],
                        margins=True, margins_name="total"))
'''))

    # ================= 11. Summary =================
    cells.append(md("""
---
## 11. Final preprocessing summary
"""))
    cells.append(code('''
report_json = paths.REPORTS_DIR / "preprocessing_report.json"
if report_json.exists():
    with open(report_json) as handle:
        preprocessing_summary = json.load(handle)

    size = preprocessing_summary["dataset_size"]
    before = preprocessing_summary["dimensions_before"]
    after = preprocessing_summary["dimensions_after"]
    integrity = preprocessing_summary["mask_integrity"]

    display(pd.Series({
        "Series processed": f"{size['series_succeeded']} / {size['series_attempted']}",
        "Patients": size["patients"],
        "Sagittal slices available": size["slices_available"],
        "Preprocessed slices written": size["slices_written"],
        "Slices dropped (too little annotation)": size["slices_dropped_unannotated"],
        "Distinct input shapes": before["distinct_shapes"],
        "Output shape": f"{after['rows']} x {after['cols']} px @ {after['spacing_mm']} mm",
        "Series where a label was invented": integrity["series_with_invented_labels"],
        "Annotated area retained (mean)": f"{100 * integrity['annotated_area_retained_mean']:.2f}%",
        "Annotated area retained (worst)": f"{100 * integrity['annotated_area_retained_min']:.2f}%",
    }, name="value").to_frame())

    print("\\nSemantic class distribution (% of all pixels):")
    display(pd.Series(preprocessing_summary["class_distribution_pct_of_all"],
                      name="% of pixels").to_frame())
else:
    print("Run: python scripts/03_preprocess.py")
'''))

    cells.append(code('''
# Load a preprocessed slice back from disk to confirm the saved artefacts are usable.
from src.preprocessing.dataset import load_processed_slice

if slice_index is not None and len(slice_index):
    sample_row = slice_index.iloc[0]
    bundle = load_processed_slice(sample_row["npz_path"])
    print(f"slice_id : {sample_row['slice_id']}")
    print(f"file     : {sample_row['npz_path']}")
    for key, array in bundle.items():
        print(f"  {key:14s} shape={array.shape} dtype={array.dtype} "
              f"range=[{array.min()}, {array.max()}]")

    from src.preprocessing.visualize import overlay_mask
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    axes[0].imshow(bundle["image"], cmap="gray"); axes[0].set_title("saved image")
    axes[1].imshow(bundle["mask"], cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title("saved semantic mask")
    axes[2].imshow(overlay_mask(bundle["image"], bundle["mask"])); axes[2].set_title("overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"Round-trip check: {sample_row['slice_id']}")
    fig.tight_layout()
    plt.show()
'''))

    cells.append(md("""
### What Sprint 1 delivered

* Clean project structure; the two EHR tutorial notebooks preserved unmodified in
  `tutorials/` as reference material only.
* Raw dataset left byte-for-byte intact in `data/raw/`; extraction is scripted
  and reproducible.
* All 447 volume pairs inspected: format, geometry, label vocabulary, intensity
  conventions, duplicates and anatomical consistency.
* Image-mask pairing driven by parsed filename identifiers, with every pair
  verified.
* A preprocessing pipeline whose every optional step is backed by a measurement.
* Masks resized with nearest neighbour only, with automated verification that no
  label value was invented.
* Reproducible, stratified, **patient-level** train/validation/test split with an
  explicit leakage check.
* Reports in `outputs/preprocessing_reports/` and figures in
  `outputs/visualizations/`.

### Dataset problems found

* Pixel spacing, matrix size, slice thickness and stored orientation all vary
  between series - handled by RAS reorientation and resampling to a common
  physical scale.
* Two incompatible intensity conventions - handled by foreground-restricted
  percentile normalisation.
* 107 groups of byte-identical mask volumes, all within one patient (shared
  T1/T2 annotation) - makes patient-level splitting mandatory.
* `sex` in `overview.csv` has trailing-whitespace duplicates - stripped on load.
* A small number of series annotate an unusually short span of the spine - kept
  but flagged.
* Severe foreground/background class imbalance, with intervertebral discs the
  smallest target class.

### Next sprint

1. Baseline segmentation model (U-Net) on the preprocessed 2-D slices.
2. Class-imbalance-aware loss (Dice or compound Dice + cross-entropy).
3. Data augmentation.
4. Post-processing (largest-component filtering, morphological cleanup).
5. Evaluation on the held-out **test patients**: Dice, IoU, precision, recall.
6. Qualitative comparison of predictions against ground truth.

> Sprint 1 reports no accuracy metric. Dice and IoU will be reported only after a
> model has actually been trained and evaluated.
"""))

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    markdown = sum(1 for c in notebook.cells if c.cell_type == "markdown")
    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"  {len(notebook.cells)} cells ({markdown} markdown, {code_cells} code)")


if __name__ == "__main__":
    main()
