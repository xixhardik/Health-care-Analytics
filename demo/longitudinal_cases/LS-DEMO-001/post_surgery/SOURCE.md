# Stage 2 — Post-Surgery (simulated position)

**No imaging is stored in this directory, by design.**

| | |
| --- | --- |
| Source study | `106_t2` (SPIDER, held-out test split) |
| Referenced path | `data/extracted/images/106_t2.mha` (gitignored, local only) |
| Display reference | Demonstration study B |
| True follow-up? | **No** |

This is a different patient from stage 1. **No surgery is depicted anywhere in
this demonstration**, because the SPIDER dataset contains no postoperative
imaging. The stage name describes a position in a demonstration timeline, not an
observed postoperative state.

If the referenced volume is absent locally, the API reports this stage
unavailable. It never substitutes another scan.
