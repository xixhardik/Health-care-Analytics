# Stage 4 — 6-Month Recovery (simulated position)

**No imaging is stored in this directory, by design.**

| | |
| --- | --- |
| Source study | `6_t2` (SPIDER, held-out test split) |
| Referenced path | `data/extracted/images/6_t2.mha` (gitignored, local only) |
| Display reference | Demonstration study D |
| True follow-up? | **No** |

The least degenerate of the four studies, and a fourth distinct patient. It
occupies the final position in the demonstration timeline.

The apparent improvement from stage 1 to stage 4 is a consequence of **which
studies were selected**. It is not a recovery trajectory, and no recovery outcome
is claimed.

If the referenced volume is absent locally, the API reports this stage
unavailable. It never substitutes another scan.
