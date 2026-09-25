# Sprint 3 Audit - Architecture and Data Utilisation Investigation

*Feasibility investigation before any new model is trained. No long training run was launched.*

Generated: 2026-09-23 12:57:35

Sprint 2 Extended established that the 16-channel U-Net has converged at its current configuration. This audit determines what to change next, and answers it with measurements taken on this machine rather than estimates.


## 1. Current architecture summary

| Field | Value |
| --- | --- |
| Model | UNet |
| Input / output | 1 channel in, 4 classes out |
| Base channels (width) | 16 |
| Depth | 4 |
| Encoder widths | 16, 32, 64, 128, 256 |
| Spatial sizes per level | 352x256, 176x128, 88x64, 44x32, 22x16 |
| Bottleneck resolution | 22x16 |
| Upsampling | bilinear + 1x1 conv |
| Normalisation / activation | BatchNorm2d / ReLU |
| Parameters | 1,963,860 (1.964M), all trainable |


### Where the capacity sits

| module | parameters | share |
| --- | --- | --- |
| downs | 1,176,960 | 59.9% |
| ups | 784,320 | 39.9% |
| stem | 2,512 | 0.1% |
| head | 68 | 0.0% |

- Plain U-Net (Ronneberger topology) with BatchNorm added.
- No attention, no residual blocks, no deep supervision, no 3-D convs.
- padding=1 on every 3x3 conv, so skip connections align without cropping.


### Loss, optimiser, schedule, augmentation

| Field | Value |
| --- | --- |
| Loss | ce_weight * CrossEntropy + dice_weight * (1 - mean soft Dice) |
| CE weight / Dice weight | 1.0 / 1.0 |
| CE class weights used | False |
| Dice includes background | True |
| Dice smoothing | 1 |
| Optimiser | Adam, lr=0.001, weight_decay=0.0 |
| Schedule | CosineAnnealingLR (T_max 12 baseline, 30 extended) |
| Gradient clipping | none |
| Batch size | 8 |
| Augmentation | integer translation, intensity scale and offset |


### Measured runtime resources

| Field | Value |
| --- | --- |
| Device | CPU only (no CUDA device present) |
| Logical cores / torch threads | 16 / 12 |
| CPU utilisation during a training step | 1138.0% of one core-equivalent scale, i.e. ~71.1% of all 16 cores |
| RAM installed / available at audit | 8.19 GB / 1.74 GB |
| Process RSS during training | 799.3 MB |
| Data-loader throughput | 231.0 img/s (0.0346s per batch of 8) |
| Training throughput | 3.62 img/s (2.2091s per batch of 8) |
| Loader share of step time | 1.57% |
| Bottleneck | compute |

**The pipeline is compute-bound, not I/O-bound.** Data loading is 1.57% of step time (231.0 img/s available against 3.62 img/s consumed), so worker processes cannot help - Sprint 2 measured them making training slightly *slower* by competing for the same cores.

- Data loading is a negligible fraction of step time, so worker processes do not help; they were measured in Sprint 2 to make training slightly slower by competing for the same cores.
- bf16 autocast was measured at ~21x slower on this CPU (no native bf16 support) and is not used.


## 2. Current data sampling - why only 2,560 of 9,128 slices

| Field | Value |
| --- | --- |
| Training slices available | 9,128 |
| Training patients / series | 152 / 319 |
| Configured cap (`max_train_slices`) | 2560 |
| Slices used per epoch | 2,560 |
| Fraction of training data per epoch | 28.1% |
| Slices never seen | 6,568 |
| Slices per patient available | 14-158 (median 48) |
| Slices per patient sampled | 14-22 |
| Sampling function | `src.models.data._sample_per_patient` |
| Strategy | cap // n_patients slices are drawn without replacement from each training patient (2560 // 152 = 16), then the remainder is topped up at random from whatever is left |


### Why the cap exists

- The cap is a CPU wall-clock decision, not a data or method decision. Measured training throughput is ~3.9 img/s at 352x256 width 16, so a full 9,128-slice epoch costs ~39 min of training plus validation.
- At 2,560 slices an epoch costs ~11-13 min, which made a 12-epoch run fit in ~2.6 h and a 30-epoch run in ~3.9 h.
- The cap is passed as max_train_slices; nothing in the architecture, split or preprocessing requires it.


### The actual defect

**The sampling RNG is seeded once per run, not per epoch, and the DataLoader is constructed once before training. The SAME 2,560 slices are therefore reused for every epoch - shuffling only reorders them. The remaining 6,568 training slices (72% of the available training data) were never seen by either completed run.**

Confirmed by simulation: the subset is a deterministic function of the seed alone (identical across calls: True), and simulating 30 epochs of the current `fixed` strategy reaches only **28.05% coverage** with 6,568 slices never seen. This is a *sampling* limitation, not a capacity limitation - and it was in force for both completed runs.

**Leakage status.** No leakage. Sampling draws exclusively from rows where split == 'train', and build_dataloaders asserts the splits are patient-disjoint on every construction.


## 3. Width 16 / 32 / 64 feasibility

Measured on this machine: each configuration ran in a fresh subprocess doing real forward+backward steps at 352x256. **Peak commit charge** is reported rather than resident-set size, because under memory pressure Windows pages memory out and RSS understates allocation - it can even *fall* as the model grows. Feasibility is judged against ~2500 MB of realistically free RAM (the machine has 8.19 GB installed but runs a browser and editor) and a 10-hour overnight budget.

| width | batch | params | vs w16 | peak commit | memory | img/s | full epoch | 30 ep (full) | 30 ep (rotating) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 8 | 1.964M | 1.0x | 2534 MB (0.99x proven) | proven | 3.061 | 52 min | 26 h | 7 h | memory-proven but too slow |
| 32 | 8 | 7.849M | 4.0x | 3834 MB (1.5x proven) | at risk | 0.988 | 160 min | 80 h | 22 h | not feasible |
| 32 | 4 | 7.849M | 4.0x | 2593 MB (1.02x proven) | at risk | 1.149 | 138 min | 69 h | 19 h | not feasible |
| 32 | 2 | 7.849M | 4.0x | 1823 MB (0.71x proven) | proven | 1.35 | 118 min | 59 h | 17 h | memory-proven but too slow |
| 64 | 8 | 31.384M | 15.98x | 6772 MB (2.66x proven) | exceeds practical memory | 0.24 | 652 min | 326 h | 88 h | not feasible |
| 64 | 2 | 31.384M | 15.98x | 3078 MB (1.21x proven) | at risk | 0.331 | 478 min | 239 h | 67 h | not feasible |

- **Width 16 / batch 8 is memory-proven**: ~2.5 GB peak commit, and this exact configuration completed two multi-hour runs on this machine. It is the reference point for the memory column.
- **Memory is not what rules out width 32 - time is.** At batch 2, width 32 needs ~1.8 GB, which is *less* than the proven width-16 configuration. Reducing the batch also makes it slightly faster per image (1.29 -> 1.41 img/s) because it pages less. But even at its best it is ~2.6x slower per image than width 16, so 30 full-coverage epochs cost ~57 h against ~21 h.
- **Width 64 is not feasible on this hardware.** ~6.8 GB peak commit at batch 8 (2.7x the proven level) and 216-288 h for 30 full epochs. Reducing to batch 2 brings memory to ~3.1 GB but still needs ~216 h. **Not launched, as instructed.**
- Parameters grow ~4x per width doubling (1.96M -> 7.85M -> 31.4M), but measured time per image grows faster still (1x -> ~2.7x -> ~13.6x), because the extra memory traffic pushes the process into paging on an 8 GB machine.


## 4. Recommended next architecture

**Keep width 16 for the next experiment. Do not increase capacity yet.**

The reasoning is that capacity has not been shown to be the binding constraint. The 16-channel model converged while seeing only 28% of the available training data. Until it has been trained on all of it, a plateau cannot be attributed to insufficient capacity - it may simply be a plateau on 2,560 slices. Adding width now would change two variables at once and cost 3-6x the wall-clock time for an unmeasurable reason.

The ordering is therefore: **fix data utilisation first (it is free), then reassess capacity.** If full-coverage training also plateaus at a similar score, that is the evidence that justifies width 32 - and at that point the batch-size finding above makes it affordable.


## 5. Recommended data sampling strategy

**Adopt the `rotating` shard sampler.** It exposes the model to every training slice while keeping the per-epoch cost unchanged, so it is a strict improvement over the current fixed subset.

| strategy | slices/epoch | coverage after 30 ep | never seen | epochs to 100% | times each slice seen |
| --- | --- | --- | --- | --- | --- |
| fixed | 2560 | 28.05% | 6568 | never | 0-30 (sd 13.48) |
| rotating | 2284 | 100.0% | 0 | 4 | 7-8 (sd 0.5) |
| reshuffled | 2560 | 99.49% | 47 | never | 0-30 (sd 5.78) |

- `rotating` partitions each patient's slices into deterministic shards and uses shard `epoch % n_shards` per epoch. Coverage is **exact**: 100% after 4 epochs, every slice seen 7-8 times over 30 epochs.
- `reshuffled` re-samples per epoch. Simpler, but coverage is only 99.49% and exposure is very uneven (some slices 30 times, 47 never) - so it is not recommended.
- Shards are shuffled before splitting, so a shard is not a block of adjacent, near-identical sagittal slices.


### Verification (already run, not assumed)

| strategy | leakage-free over 30 epochs | violations | val/test patients excluded | same seed reproducible | epochs differ | overlap with next epoch |
| --- | --- | --- | --- | --- | --- | --- |
| rotating | True | 0 | 66 | True | True | 0 |
| reshuffled | True | 0 | 66 | True | True | 996 |

The patient-level split is untouched: the sampler partitions only rows already marked `split == 'train'`, and the check confirms no validation or test slice **or patient** enters any of the 30 epochs. `rotating` also has zero overlap between consecutive epochs, confirming the shards are disjoint.


## 6. Recommended loss-function experiment


### Measured class balance on the training split

| class | % of pixels | % of slices containing it | inverse-frequency weight |
| --- | --- | --- | --- |
| background | 94.65 | 100 | 1.1 |
| vertebra | 3.849 | 100 | 26 |
| intervertebral_disc | 0.827 | 74 | 120.9 |
| spinal_canal | 0.6759 | 40 | 148 |

| Field | Value |
| --- | --- |
| Slices measured | 400 |
| Foreground share | 5.352% |
| background : disc pixel ratio | 114.5 : 1 |


### What the current loss actually does

| model state | CE term | Dice term | total | Dice share of total |
| --- | --- | --- | --- | --- |
| untrained | 1.444 | 0.8851 | 2.329 | 38% |
| trained_extended | 0.02588 | 0.08528 | 0.1112 | 77% |

At convergence the cross-entropy term shrinks far faster than the Dice term, because CE is dominated by the 94.8% background pixels that become easy. The Dice term therefore supplies most of the remaining gradient - which is the intended behaviour, but it also means the equal 1:1 weighting is effectively Dice-dominated late in training.

- Cross-entropy is unweighted: every pixel contributes equally, so the 94.8% background majority dominates that term.
- The Dice term is what counteracts the imbalance, because it is normalised per class.
- Background is INCLUDED in the Dice average by default, so one of the four averaged terms is a class that is trivially easy.


### Assessment and recommendation

**The current loss is appropriate and is not the main problem.** Cross-entropy alone would be minimised by predicting background almost everywhere (94.6483% of pixels), and the soft Dice term is what prevents that, because it is normalised per class and so weights the 0.8% disc class equally with background. The achieved per-class Dice (vertebra 0.913, disc 0.878, canal 0.903 on test) is evidence it is working - a broken loss would show the small classes collapsing, which is not what happens.

Two specific, low-risk refinements are worth testing - **after** the data-coverage experiment, and one at a time:

- **Exclude background from the Dice average.** Background is currently one of four equally-weighted Dice terms and sits at ~0.995, so it contributes almost no gradient while diluting the three terms that matter by 25%. `PreprocessConfig`-equivalent support already exists as `ignore_background_in_dice`, so this is a one-flag experiment.
- **Do NOT add inverse-frequency CE weights on top.** The measured weights would be ~121x for disc against ~1x for background. Stacking that on a loss that already contains a class-normalised Dice term risks over-weighting the small classes and destabilising training at batch size 8. The existing `inverse_frequency_class_weights` helper should stay unused by default.
- A compound Dice + focal loss is a reasonable third option but changes two things at once (class weighting and hard-example weighting), so it is not the next experiment.


## 7. Disc-indexing failure analysis

Every ground-truth disc on every test slice was classified into one failure category. Analysis uses the **integer disc index only** - the dataset does not state which vertebra is L5, so no anatomical level name is asserted.

| run | discs analysed | correct | shifted | merged | split | missed |
| --- | --- | --- | --- | --- | --- | --- |
| sprint2_baseline | 6,821 | 77.97% | 16.76% | 0% | 0.85% | 4.43% |
| sprint2_extended | 6,821 | 82.25% | 12.97% | 0% | 0.51% | 4.27% |


### Root causes, in order of contribution

| Field | Value |
| --- | --- |
| Discs whose region was found | 95.73% |
| Discs correctly indexed | 82.25% |
| Slices where the disc COUNT is right | 75.53% |
| Slices where EVERY disc is right | 75.11% |
| Shifted by exactly +/-1 | 10.01% |
| Spurious components (total) | 249 |
| Slices with a spurious component | 13.29% |
| Predicted-vs-truth disc count delta | `-4`=1, `-3`=7, `-2`=29, `-1`=183, `0`=1250, `1`=161, `2`=22, `3`=2 |
| Shift offset histogram | `-5.0`=1, `-4.0`=21, `-3.0`=32, `-2.0`=124, `-1.0`=325, `1.0`=358, `2.0`=24 |

**The gap between 95.73% detection and 82.25% correct indexing is the bottleneck, and it is a counting problem, not a segmentation problem.** Indexing is derived by ordering connected components from the most inferior upward, so the index of every disc depends on how many components were found below it. One extra, missing or merged component shifts every index above it.

- **Component-count errors are the dominant cause.** The disc count is correct on only 75.53% of slices, and the index is derived from that count.
- **10.01% of all discs are shifted by exactly +/-1**, which is the signature of a single spurious or missing component below them rather than of poor localisation.
- **Lateral slices are the hard case.** Indexing accuracy rises sharply with the amount of annotation on the slice: 27% correct on slices with under 1,000 annotated pixels, against ~90% on slices with 6,000-10,000. On a lateral slice a disc appears as a small fragment that is easily missed or broken up.
- **Small discs fail disproportionately.** Discs under 50 px are 52% correct with 30% missed; discs over 400 px are 93% correct with 0% missed.
- **Merged discs are not a problem at all** (0.0% of cases), which rules out one plausible hypothesis: adjacent discs are not being fused together.


### A structural finding: why the vertebra ratio fails

Sprint 2 found that `disc_to_vertebra_height_ratio` had ~70% error when computed from predicted masks. The cause is now identified, and it is **not** segmentation quality - it is measured on the ground truth, so it is a property of the anatomy in a sagittal plane:

| structure | instances | 2-D components | components per instance | % instances split | ordered components valid? |
| --- | --- | --- | --- | --- | --- |
| intervertebral_disc | 955 | 972 | 1.018 | 1.88 | True |
| vertebra | 1275 | 2182 | 1.711 | 60.31 | False |

Discs occupy 1.018 components per instance (1.88% are split), so numbering discs by ordered connected components is structurally sound. Vertebrae occupy 1.711 components per instance, because the vertebral body and the posterior elements are separate regions in a sagittal plane. Numbering vertebrae by ordered connected components is therefore structurally invalid regardless of segmentation quality - and that is the root cause of the unreliable disc-to-vertebra height ratio.

Verified directly on well-annotated test slices: a slice with 8 annotated vertebrae contains 15-18 connected components in the **ground truth**, and the prediction finds 16-18 - i.e. the model reproduces the component structure correctly. The predicted vertebra component count matches the ground-truth component count on 36.44% of slices (mean absolute difference 1.1 components), whereas comparing components against *instances* would have appeared to be wrong on 84% of slices. **The metric, not the model, was at fault.**

**Consequence for the design.** Numbering discs by ordered connected components is sound (1.018 components per disc). Numbering *vertebrae* the same way is not (1.711 components each), so `instances_from_semantic` cannot reliably identify which vertebra is which, and any feature with a vertebra in its denominator inherits that. Either the vertebral bodies must be separated from the posterior elements before numbering, or vertebra-normalised features must be dropped when working from predicted masks.


### Accuracy by disc index

| truth_index | n | correct | shifted | merged | split | missed | pct_correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1028 | 910 | 27 | 0 | 23 | 68 | 88.52 |
| 2 | 1072 | 900 | 144 | 0 | 10 | 18 | 83.96 |
| 3 | 1048 | 848 | 165 | 0 | 0 | 35 | 80.92 |
| 4 | 994 | 789 | 165 | 0 | 0 | 40 | 79.38 |
| 5 | 925 | 749 | 136 | 0 | 2 | 38 | 80.97 |
| 6 | 854 | 715 | 122 | 0 | 0 | 17 | 83.72 |
| 7 | 583 | 479 | 83 | 0 | 0 | 21 | 82.16 |
| 8 | 269 | 182 | 34 | 0 | 0 | 53 | 67.66 |
| 9 | 48 | 38 | 9 | 0 | 0 | 1 | 79.17 |


### Accuracy by slice annotation area

| area_band | n | pct_correct | pct_shifted | pct_missed |
| --- | --- | --- | --- | --- |
| <1k | 135 | 27.41 | 23.7 | 48.89 |
| 1k-3k | 587 | 50.6 | 34.41 | 14.65 |
| 3k-6k | 1327 | 73.02 | 20.5 | 6.1 |
| 6k-10k | 3401 | 90.5 | 7.67 | 1.35 |
| >10k | 1371 | 89.64 | 8.61 | 0.88 |


### Accuracy by disc size

| size_band | n | pct_correct | pct_missed |
| --- | --- | --- | --- |
| <50 | 473 | 52.01 | 29.6 |
| 50-100 | 1357 | 72.73 | 9.58 |
| 100-200 | 2964 | 85.7 | 0.57 |
| 200-400 | 1968 | 90.55 | 0.2 |
| >400 | 59 | 93.22 | 0 |


### Baseline vs extended

| metric | baseline | extended | delta | verdict |
| --- | --- | --- | --- | --- |
| pct_index_correct | 77.97 | 82.25 | +4.280 | improved |
| pct_region_found | 95.57 | 95.73 | +0.160 | improved |
| pct_slices_count_correct | 73.35 | 75.53 | +2.180 | improved |
| pct_slices_all_discs_correct | 72.33 | 75.11 | +2.780 | improved |
| pct_shifted_by_one | 13.39 | 10.01 | -3.380 | improved |
| total_spurious_components | 288 | 249 | -39.000 | improved |
| pct_slices_with_spurious | 15.65 | 13.29 | -2.360 | improved |
| pct_slices_vertebra_components_correct | 34.68 | 36.44 | +1.760 | improved |
| mean_abs_vertebra_component_delta | 1.177 | 1.1 | -0.077 | improved |
| category_pct/correct | 77.97 | 82.25 | +4.280 | improved |
| category_pct/shifted | 16.76 | 12.97 | -3.790 | improved |
| category_pct/split | 0.85 | 0.51 | -0.340 | improved |
| category_pct/missed | 4.43 | 4.27 | -0.160 | improved |


### Recommended fixes, cheapest first

**None of these require retraining the segmentation model.** The indexing bottleneck is a post-processing problem, and that is the most useful conclusion of this section: the cheapest available improvement to the disc pipeline does not involve a new model at all.

- **1. Aggregate identity across slices instead of deciding per slice.** Sprint 2 measured that per-series aggregation already recovers 99.3% of discs against 82% per slice. Deciding disc identity at the *series* level - grouping components in 3-D across adjacent slices and numbering once - directly addresses the dominant failure mode. Highest value, no retraining.
- **2. Suppress spurious components before numbering.** 249 spurious components appear on 13.29% of slices, and each one shifts every index above it. Filter on physical area and on plausible disc aspect ratio.
- **3. Restrict per-disc measurement to slices carrying enough annotation.** Accuracy is ~90% above 6,000 annotated pixels and 27% below 1,000. Those near-empty lateral slices contribute little anatomical information anyway, so excluding them costs almost nothing and removes the worst-behaved cases.
- **4. Separate vertebral bodies from posterior elements before numbering vertebrae**, or drop vertebra-normalised features when working from predicted masks. See the structural finding above.
- **5. Only then** consider predicting instance labels directly. It is a much larger change and should not be attempted before the cheap geometric fixes are exhausted.


## 8. Estimated training time

| configuration | epoch | 30 epochs | data coverage |
| --- | --- | --- | --- |
| width 16, batch 8, fixed 2,560 slices/epoch (what ran in Sprint 2) | ~12.8 min (measured) | ~6.4 h | 28% |
| **width 16, batch 8, rotating ~2,282 slices/epoch (recommended)** | ~15 min | ~7.4 h | **100% (by epoch 4)** |
| width 16, batch 8, all 9,128 slices every epoch | ~52 min | ~26.0 h | 100% every epoch |
| width 32, batch 4, rotating ~2,282 slices/epoch | ~39 min | ~19.4 h | 100% (by epoch 4) |

The recommended configuration costs **the same wall-clock time as Sprint 2** while raising data coverage from 28% to 100%. That is why it is the next experiment rather than a wider model: it is the only available change that improves the setup at no cost.


## 9. Exact proposed Sprint 3 experiment

**One controlled variable: the sampling strategy.** Everything else is held at the Sprint 2 Extended values so the comparison is clean.

| setting | value |
| --- | --- |
| architecture | 16-channel U-Net, depth 4, bilinear (1,963,860 params) - UNCHANGED |
| sampling | **rotating shard sampler, ~2,282 slices/epoch - CHANGED** |
| epochs | 30, early stopping patience 6, **min_delta 0.001** (fixing the Sprint 2 flaw where min_delta=0 made early stopping ineffective) |
| split | Sprint 1 patient-level split - UNCHANGED |
| seed | 42 - UNCHANGED |
| loss | CE + soft Dice, equal weight - UNCHANGED |
| optimiser | Adam lr 1e-3, CosineAnnealingLR T_max=30 - UNCHANGED |
| augmentation | translate + intensity jitter - UNCHANGED |
| batch size | 8 - UNCHANGED |
| initialisation | from random, NOT resumed - a warm restart from the converged 16-channel weights would confound the data-coverage variable |
| output directory | `outputs/checkpoints/sprint3_coverage/` and `outputs/reports/sprint3_coverage/` - new, nothing overwritten |
| estimated wall time | ~6.5 h (overnight) |
| test set | untouched until one final evaluation of the validation-selected checkpoint |


### Success criteria, stated in advance

- **Primary:** validation macro foreground Dice beats 0.8985 (Sprint 2 Extended). If it does, data utilisation was the binding constraint.
- **Secondary:** test disc-indexing accuracy beats 82.6%.
- **Decision rule:** if full coverage does *not* improve on 0.8985, then capacity becomes the credible next variable and width 32 at batch 4 is justified - with the measured ~19 h cost accepted deliberately.
- Either outcome is informative, which is what makes this the right next experiment.


### Explicitly out of scope

- Width 64 - measured as infeasible on this hardware (~6.8 GB commit, hundreds of hours).
- Loss changes - deferred to a later single-variable experiment.
- 3-D U-Net, attention, transformers, ensembles.
- Any clinical diagnosis, postoperative healing prediction, longitudinal claim, or composite 'percentage spine damage' score. The pipeline reports measurable segmentation, disc-level radiological findings from dataset annotations, and quantitative imaging features in millimetres.


## 10. Figures

- `outputs/visualizations/sprint3_audit/sampling_coverage.png`
- `outputs/visualizations/sprint3_audit/width_feasibility.png`
- `outputs/visualizations/sprint3_audit/indexing_failures.png`

