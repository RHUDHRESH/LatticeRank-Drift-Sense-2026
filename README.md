# LatticeRank Drift-Sense

CPU-only edge registration for the SEMICON India hackathon.

## Project view

| Phase 2: SEM → SEM | Phase 3: CAD → SEM |
|---|---|
| ![Phase 2 edge registration](assets/phase2-overview.svg) | ![Phase 3 CAD registration](assets/phase3-overview.svg) |
| **Measured results** | **Work status** |
| ![Measured results](assets/measured-results.svg) | ![Completed work and next fixes](assets/work-status.svg) |

- **Done:** edge detection, scale/rotation proposal, multi-peak translation,
  continuous subpixel refinement, confidence, no-match output, Phase 3
  layer-aware CAD matching, and exact judge CSV handling.
- **Working well:** zero execution failures on both 25-pair runs; Phase 2
  measured subtotal `72.9838/85`; Phase 3 placed all 24 generated positives
  within two pixels.
- **Weak now:** Phase 2 harsh degradation can lose or reject correct
  candidates; Phase 3 accepts difficult no-match pairs and its bounded vector
  voting is biased by polygon traversal order.
- **Next:** remove Phase 3 vote-order bias, add independent negative evidence,
  fuse vector and image candidates, then improve Phase 2 candidate retention
  and recalibrate confidence.

Detailed measurements and algorithms are in
[FINDINGS_AND_PROCESS.md](FINDINGS_AND_PROCESS.md).

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.11 or newer is required. Inference is offline and defaults to one CPU
thread.

## Phase 2: SEM to SEM

```bash
python register.py --input pairs.csv --output predictions.csv
```

Required input fields: `pair_id`, `reference_path`, `search_path`.
Rotation is reported in degrees and scale is reported in the range 8–12.

## Phase 3: CAD to SEM

```bash
python phase3.py --input pairs.csv --output predictions.csv
```

Required input fields: `pair_id`, `search_path`, `reference_gds_path`.
`search_gds_path` is used when supplied. `reference_sem_path` and
`params_json_path` are training metadata and are never read during inference.
Phase 3 reports fixed `theta=0` and `scale=10`, while fitting visible GDS layers
against the SEM image; the first or last layer may be faint or absent.

## Output

Both entry points write exactly:

```csv
pair_id,x,y,theta,scale,found,score
```

Each unique input ID receives one finite output row. `found` is 0 or 1;
rejected rows have `x=y=theta=scale=0`; `score` is in `[0,1]`.

Run the focused checks with `python -m pytest -q`.
