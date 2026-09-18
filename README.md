# LatticeRank Drift-Sense

CPU-only registration for the SEMICON India Phase 2 and Phase 3 tasks. Each
judge submission is self-contained in its own folder.

| Task | Judge entry point | Input |
|---|---|---|
| [Phase 2: SEM → SEM](phase_2/README.md) | `python phase_2/register.py --input pairs.csv --output predictions.csv` | SEM reference and SEM search |
| [Phase 3: CAD → SEM](phase_3/README.md) | `python phase_3/phase3.py --input pairs.csv --output predictions.csv` | reference GDS, search SEM, optional search GDS |

## Install

Python 3.11 or newer is required.

```bash
python -m pip install -r phase_2/requirements.txt
python -m pip install -r phase_3/requirements.txt
```

Inference is offline. Native numerical libraries default to one CPU thread for
predictable execution on shared and low-specification machines.

## Common output contract

Both phases write exactly:

```csv
pair_id,x,y,theta,scale,found,score
```

Each unique input ID produces one finite row. `found` is `0` or `1`, `score`
is clipped to `[0,1]`, and rejected rows use `x=y=theta=scale=0`. Output is
written atomically, and a malformed pair does not terminate the batch.

## Technical distinction

Phase 2 combines local SIFT/RANSAC evidence, global directional-spectrum
proposals, explicit scale-boundary hypotheses and spatial edge correlation.
It retains remote lattice aliases, refines candidates continuously in
`(x,y,theta,scale)`, and constructs confidence from independent evidence rather
than copying the winning correlation value.

Phase 3 keeps GDS layers separate and estimates how each visible layer appears
in the SEM at every candidate. A layer may fit to zero, so a faint first or
last layer does not invalidate the template. Search-side CAD can add exact
geometry proposals, but SEM evidence still decides presence.

These choices address the two central ambiguities in the supplied data:
repeated semiconductor lattices create convincing remote aliases, and CAD
layer identity does not determine SEM brightness.

## Visual evidence

### Cumulative GDS layer tiles

![Cumulative GDS tiles](phase_3/assets/gds-layer-stack.png)

### CAD → inferred layer appearance → SEM

![CAD and SEM merged view](phase_3/assets/cad-yield-sem.png)

### Phase 2 organizer coverage

![Phase 2 parameter grid](phase_2/assets/phase2-parameter-grid.png)

Green cards are present references; red cards are true no-match cases. The
bold band exposes scale, rotation and severity before the detailed acquisition
and uniqueness measurements.

### Phase 3 distortion coverage

![Phase 3 parameter grid](phase_3/assets/phase3-parameter-grid.png)

The bold band exposes the principal geometric and dose settings; the rows
below it retain the remaining acquisition, noise, fabrication and layout
parameters.

Every grid tile is generated from the supplied organizer data or generator and
prints the parameters needed to inspect that case. Only the rendered figures
are included; organizer datasets, ground truth and generators are not shipped.

## Measured validation

- Phase 2 organizer 25-pair cut: `20 TP / 0 FP / 0 FN`; measured score
  `81.3/85` before efficiency and generator-analysis points.
- Phase 2 independent certified stress sets: `30/0/0` and `30/2/0`
  `(TP/FP/FN)`.
- Phase 3 organizer-compatible image-only set: `24 TP / 0 FP / 0 FN / 1 TN`,
  all accepted positives within 2 px.
- Phase 3 frozen 32-pair CAD/SEM set: `24 TP / 0 FP / 0 FN / 8 TN`, all
  positives at the labelled centre.
- Submitted-head regression suite: `59 passed`, including both judge-facing
  phase-folder entry points.

These are corpus measurements, not claims about an unseen jury set. A repeated
layout can be genuinely indistinguishable when the available crop has no unique
structure; the solvers expose that ambiguity through `score` and may abstain.

## Repository map

```text
phase_2/          self-contained SEM-to-SEM submission
phase_3/          self-contained CAD-to-SEM submission
```

## Verify

```bash
python phase_2/register.py --help
python phase_3/phase3.py --help
```
