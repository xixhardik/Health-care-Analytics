# Demonstration assets

This directory holds **demonstration workflow definitions only**. It contains no
medical imaging, no patient data and no dataset files, and nothing here is
research output.

It is deliberately separate from:

| Path | Contents |
| --- | --- |
| `data/` | the SPIDER dataset and everything derived from it (never committed) |
| `outputs/` | research results, metrics and reports |
| `demo/` | **this directory** - workflow definitions for demonstrations |

## Why this exists

The SPIDER dataset is **cross-sectional**: each patient is imaged at a single
timepoint. It contains no postoperative imaging, no surgical record and no
follow-up assessment. A longitudinal recovery tracker therefore cannot be
validated on it, and this project does not claim otherwise.

To demonstrate *how* such a workflow would operate, `longitudinal_cases/`
defines simulated cases. A case stages **four different real SPIDER studies from
four different patients** as four timeline positions. The imaging is real; the
timeline is not.

> Four real SPIDER studies arranged into a simulated UI workflow.
> **Not** one patient followed through surgery and recovery.

## What is and is not stored here

- **Stored:** `manifest.json` per case - stage order, labels, the source study
  each stage references, and explicit provenance flags. Small, text, committed.
- **Not stored:** MRI volumes. Each stage references an existing local SPIDER
  study by project-relative path. Those files live under `data/extracted/` and
  are gitignored. If they are absent on a machine, the API reports the stage
  unavailable rather than substituting anything.

## Reading a manifest

Every stage carries provenance that cannot be dropped:

| Field | Meaning |
| --- | --- |
| `source_study_id` | the actual SPIDER study backing this stage, e.g. `177_t2` |
| `source_dataset` | always `SPIDER` |
| `source_split` | which split the study belongs to (`test` for all current stages) |
| `display_reference` | the non-identifying label the interface shows |
| `is_true_followup` | always `false` |
| `measurement_source` | `real_pipeline` or `simulated_demo_value` |

`measurement_source` is `real_pipeline` for every stage in `LS-DEMO-001`: the
numbers shown come from running the frozen Sprint 1/3/5 pipeline on the real
study behind that stage. Nothing is invented. What is simulated is only the
*arrangement* of four unrelated studies into a timeline.

The four studies were chosen from the **held-out test split**, so the
segmentation on display was never trained on, and each was verified to pass
preprocessing, segmentation, Sprint 5 indexing and rendering before selection.
