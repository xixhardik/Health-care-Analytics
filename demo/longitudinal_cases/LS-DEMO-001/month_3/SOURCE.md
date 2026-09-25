# Stage 3 — 3-Month Recovery (simulated position)

**No imaging is stored in this directory, by design.**

| | |
| --- | --- |
| Source study | `16_t2` (SPIDER, held-out test split) |
| Referenced path | `data/extracted/images/16_t2.mha` (gitignored, local only) |
| Display reference | Demonstration study C |
| True follow-up? | **No** |

A different patient again, and deliberately a different scanner — Philips
Healthcare 3.0T against SIEMENS 1.5T for the other stages — so the demonstration
also shows the pipeline handling acquisition variation.

No three-month interval was observed. The label is a position in a demonstration
timeline.

If the referenced volume is absent locally, the API reports this stage
unavailable. It never substitutes another scan.
