# Stage 1 — Pre-Surgery (simulated position)

**No imaging is stored in this directory, by design.**

| | |
| --- | --- |
| Source study | `177_t2` (SPIDER, held-out test split) |
| Referenced path | `data/extracted/images/177_t2.mha` (gitignored, local only) |
| Display reference | Demonstration study A |
| True follow-up? | **No** |

This stage is a real SPIDER study from one patient, placed at the "pre-surgery"
position of a simulated timeline. It is not a preoperative scan of a patient who
later appears in the other stages — the four stages are four different patients.

If the referenced volume is absent locally, the API reports this stage
unavailable. It never substitutes another scan.
