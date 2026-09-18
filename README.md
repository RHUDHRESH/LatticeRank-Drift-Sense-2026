# LatticeRank Drift-Sense

CPU-only registration for the SEMICON India Drift-Sense hackathon. The two
submissions are independent and are judged separately; each folder is
self-contained and carries its own README, requirements and tests.

| Submission | Entry point (run from inside the folder) |
|---|---|
| **[Phase 2 — SEM to SEM](phase_2/README.md)** | `python register.py --input pairs.csv --output predictions.csv` |
| **[Phase 3 — CAD to SEM](phase_3/README.md)** | `python phase3.py --input pairs.csv --output predictions.csv` |

Both write `pair_id,x,y,theta,scale,found,score`, one row per `pair_id`.

`tools/validate_submission.py` runs either phase the way the judges run it —
blind split, no network, contract checked — and `tools/make_tiles.py` renders
the per-case evidence tiles. No dataset is stored in this repository.
