#!/usr/bin/env python3
"""Phase 3 CLI: register a layered GDS reference against a SEM search image.

The current Phase 3 statement fixes rotation at zero and the CAD-to-search
scale at 10. Translation is estimated from layer boundaries, while per-layer
SEM contrast is learned independently at each candidate. Training-only image,
parameter, and ground-truth columns are never inspected.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Bound native numeric libraries before importing NumPy/scipy through the
# solver. Jury processes may run several pairs/submissions concurrently.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("DRIFTFORGE_NUM_THREADS", "1")

import numpy as np

PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import register
from driftforge.cad_edges import NOMINAL_SCALE, solve_cad_edges
from driftforge.cadref import rasterize_gds

COLUMNS = register.COLUMNS
BUDGET_S = register.BUDGET_S                 # compatibility for external runners
FAILURE_SCORE = register.FAILURE_SCORE
PIXEL_SIZE_REF_NM = 1.0
PIXEL_SIZE_SEARCH_NM = 10.0
PHASE3_SCALES = (NOMINAL_SCALE,)             # fixed by the Phase 3 statement

GDS_KEYS = (
    "reference_gds_path", "reference_gds", "gds_path", "reference_path",
    "reference", "ref_gds_path", "ref_path",
)

# These fields are labels, development aids, or alternate/search-side assets.
# Extension fallback must not accidentally select any of them as the reference.
FORBIDDEN_REFERENCE_KEYS = frozenset({
    "gt_x", "gt_y", "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
    "match_found", "found", "theta", "scale", "seed",
    "reference_sem_path", "params_json_path", "reference_preview_path",
    "search_gds_path", "search_gds", "search_cad_path",
})
TRAINING_ONLY_KEYS = frozenset({
    "gt_x", "gt_y", "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
    "match_found", "found", "theta", "scale", "seed",
    "reference_sem_path", "params_json_path", "reference_preview_path",
})

SEARCH_GDS_KEYS = ("search_gds_path", "search_gds", "search_cad_path")


def _pick_gds(row: dict) -> str | None:
    names = {str(name).strip().lower(): name for name in row if name}
    for key in GDS_KEYS:
        source_name = names.get(key)
        if source_name is None:
            continue
        value = row[source_name]
        if value not in (None, ""):
            return str(value)
    for name in row:
        if not name:
            continue
        lowered_name = str(name).strip().lower()
        if lowered_name in FORBIDDEN_REFERENCE_KEYS or "search" in lowered_name:
            continue
        value = row[name]
        if value in (None, ""):
            continue
        if str(value).strip().lower().endswith((".gds", ".gds2", ".gdsii")):
            return str(value)
    return None


def _pick_search(row: dict) -> str | None:
    names = {str(name).strip().lower(): name for name in row if name}
    for key in register.SEARCH_KEYS:
        source_name = names.get(key)
        if source_name is not None:
            value = row[source_name]
            if value not in (None, ""):
                return str(value)
    for lowered_name, source_name in names.items():
        if lowered_name in FORBIDDEN_REFERENCE_KEYS:
            continue
        if any(hint in lowered_name for hint in register.REF_HINTS):
            continue
        if any(hint in lowered_name for hint in register.SEARCH_HINTS):
            value = row[source_name]
            if value not in (None, "") and str(value).strip().lower().endswith(register.IMAGE_SUFFIXES):
                return str(value)
    return None


def _pick_search_gds(row: dict) -> str | None:
    """Return the optional judge-supplied full-canvas CAD path.

    The search-side GDS is a legal inference input, but it must never be
    confused with the cropped reference GDS.
    """
    names = {str(name).strip().lower(): name for name in row if name}
    for key in SEARCH_GDS_KEYS:
        source_name = names.get(key)
        if source_name is not None and row[source_name] not in (None, ""):
            return str(row[source_name])
    return None


def _pick_id(row: dict, fallback: str) -> str:
    names = {str(name).strip().lower(): name for name in row if name}
    for key in register.ID_KEYS:
        source_name = names.get(key)
        if source_name is not None:
            value = row[source_name]
            if value not in (None, ""):
                return str(value)
    return fallback


def load_reference(gds_path: Path) -> np.ndarray:
    """Compatibility preview helper; inference uses geometry directly."""
    if gds_path.suffix.lower() in (".gds", ".gds2", ".gdsii"):
        return rasterize_gds(gds_path)
    return register.load_image(gds_path)


def _failure(pair_id: str) -> dict:
    return {"pair_id": pair_id, "x": 0.0, "y": 0.0, "theta": 0.0,
            "scale": 0.0, "found": 0, "score": float(FAILURE_SCORE)}


def _finite_result(pair_id: str, result: dict) -> dict:
    output = {"pair_id": pair_id}
    output.update(result)
    numeric = ("x", "y", "theta", "scale", "score")
    if any(not math.isfinite(float(output[key])) for key in numeric):
        raise ValueError("solver returned a non-finite value")
    output["found"] = int(bool(output["found"]))
    if output["found"]:
        # The fixed-pose contract is explicit even if a future internal helper
        # grows extra pose diagnostics.
        output["theta"] = 0.0
        output["scale"] = float(NOMINAL_SCALE)
    else:
        output.update(x=0.0, y=0.0, theta=0.0, scale=0.0)
    output["score"] = float(np.clip(float(output["score"]), 0.0, 1.0))
    return output


def process(pair_id: str, gds_path: str | None, search_path: str | None,
            search_gds_path: str | None = None,
            *_unused_models) -> tuple[dict, str | None]:
    """Return one contract row and an optional failure reason."""
    failed = _failure(pair_id)
    try:
        if not gds_path or not search_path:
            return failed, "missing reference GDS or search path in input row"
        resolved_gds = register.resolve_path(gds_path)
        search = register.load_image(register.resolve_path(search_path))
        resolved_search_gds = (register.resolve_path(search_gds_path)
                               if search_gds_path else None)
        result = solve_cad_edges(resolved_gds, search, scale=NOMINAL_SCALE,
                                 search_gds_path=resolved_search_gds)
        return _finite_result(pair_id, result), None
    except (FileNotFoundError, ValueError, OSError) as exc:
        return failed, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - every input row still emits
        traceback.print_exc(file=sys.stderr)
        return failed, f"unexpected {type(exc).__name__}: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 CAD-edge to SEM registration.")
    parser.add_argument("--input", required=True, type=Path, help="path to pairs.csv")
    parser.add_argument("--output", required=True, type=Path, help="path to predictions.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    register.SEARCH_ROOTS.clear()
    for root in (args.input.resolve().parent, PROJECT):
        if root not in register.SEARCH_ROOTS:
            register.SEARCH_ROOTS.append(root)

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"error: input has no rows: {args.input}", file=sys.stderr)
        return 2

    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    timings: list[float] = []
    seen: set[str] = set()
    started = time.perf_counter()
    for index, row in enumerate(rows):
        pair_id = _pick_id(row, str(index))
        if pair_id in seen:
            print(f"warning: duplicate pair_id {pair_id}; keeping first", file=sys.stderr)
            continue
        seen.add(pair_id)
        began = time.perf_counter()
        output, reason = process(
            pair_id,
            _pick_gds(row),
            _pick_search(row),
            _pick_search_gds(row),
        )
        timings.append(time.perf_counter() - began)
        results.append(output)
        if reason:
            failures.append((pair_id, reason))
            print(f"pair {pair_id}: FAILED -- {reason}", file=sys.stderr)

    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for output in results:
            writer.writerow({key: output[key] for key in COLUMNS})
    temporary.replace(destination)

    elapsed = time.perf_counter() - started
    found = sum(int(output["found"]) for output in results)
    median = float(np.median(timings)) if timings else 0.0
    worst = float(np.max(timings)) if timings else 0.0
    print(f"processed {len(results)} pairs in {elapsed:.1f}s "
          f"(median {median:.2f}s/pair, max {worst:.2f}s) | found={found} "
          f"({found/max(len(results), 1):.1%}) | failures={len(failures)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
