# LatticeRank Drift-Sense

CPU-only edge registration for the SEMICON India hackathon.

## Project view

| GDS layer tiles | Merged CAD → yield → SEM |
|---|---|
| ![Cumulative GDS layer tiles and full merged view](assets/gds-layer-stack.png) | ![Merged CAD, inferred yield raster, and SEM search](assets/cad-yield-sem.png) |
| **Phase 2: SEM → SEM** | **Measured results** |
| ![Phase 2 edge registration](assets/phase2-overview.svg) | ![Measured results](assets/measured-results.svg) |
| **Completed and next** | |
| ![Completed work and next fixes](assets/work-status.svg) | |

The organizer's **tiles** are cumulative GDS layer views: layers 0–1,
layers 0–3, layers 0–5, and finally all eight layers. The **merged view** is
the complete stack of colored polygons. It has exact geometry and no image
brightness or SEM noise. The solver then infers a grayscale layer appearance
and registers that geometry against the supplied noisy SEM search. The two
Phase 3 images above were generated from the supplied organizer generator,
not copied from the photographed slides.

- **Done:** edge detection, scale/rotation proposal, multi-peak translation,
  continuous subpixel refinement, confidence, no-match output, Phase 3
  layer-aware CAD matching, and exact judge CSV handling.
- **Judge-style validation:** Phase 2 returned 20 TP, 0 FP, and 0 FN on the
  supplied 25-pair cut, with all 20 positives within 2 px. Phase 3 returned
  24 TP, 0 FP, and 0 FN on the frozen 32-pair CAD/SEM suite, all at the exact
  labelled centre.
- **Confidence:** Phase 2 standardizes each correlation peak against its own
  sidelobes (presence AUC `0.940` on the 25-pair cut). Phase 3 additionally
  measures the CAD-layer/SEM correlation ratio and held-out yield fit.
- **Known limit:** repeated Phase 2 lattice cells can remain visually
  indistinguishable; on the separate 40-pair periodic stress set, 13 of 29
  positives were within 5 px. Fisher/CRB diagnostics describe local subpixel
  precision but deliberately do not certify that the correct lattice lobe was
  selected.

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
