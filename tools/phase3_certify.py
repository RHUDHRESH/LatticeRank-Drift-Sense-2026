#!/usr/bin/env python3
"""Certify Phase 3 stress pairs after SEM degradation.

The organizer generator can label a crop absent even when a repeated layout
produces an equally good observable match.  This utility removes those
contradictory cases.  It never changes ground truth: ambiguous samples are
discarded and reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import register
from driftforge import cad_edges as ce


def _truth(row: dict[str, str]) -> tuple[bool, float | None, float | None]:
    raw = str(row.get("match_found", row.get("present", "0"))).lower()
    present = raw in {"1", "true", "yes"}
    return (present,
            float(row["gt_x"]) if present else None,
            float(row["gt_y"]) if present else None)


def _analyse(pair: dict[str, str], truth: dict[str, str], root: Path) -> dict:
    register.SEARCH_ROOTS[:] = [root]
    search = ce._normalise_image(register.load_image(
        register.resolve_path(pair["search_path"])))
    template = ce.load_layered_template(register.resolve_path(
        pair["reference_gds_path"]))
    _gx, _gy, edges = ce._gradient(search)
    raw = ce._proposal_candidates(edges, template, search)
    fits = [ce._fit_candidate(search, template, float(col), float(row), score)
            for row, col, score in raw]
    rank = lambda item: (0.58 * item.fit + 0.27 * item.appearance_corr
                         + 0.05 * item.proposal)
    fits.sort(key=rank, reverse=True)
    pool = [ce._refine_candidate(search, template, item) for item in fits[:3]]
    pool.extend(fits[3:])
    pool.sort(key=rank, reverse=True)
    best = pool[0]
    vector: list[tuple[int, int, float]] = []
    search_gds = pair.get("search_gds_path", "").strip()
    if search_gds:
        vector = ce._vector_cad_proposal_candidates(
            register.resolve_path(pair["reference_gds_path"]),
            register.resolve_path(search_gds), search.shape, ce.NOMINAL_SCALE)
    present, gt_x, gt_y = _truth(truth)
    result = {
        "pair_id": pair["pair_id"], "present": present,
        "best_fit": best.fit, "best_edge": best.edge_corr,
        "best_appearance": best.appearance_corr, "best_eta": best.eta_squared,
        "best_x": best.left + 50.0, "best_y": best.top + 50.0,
    }
    # A negative is certifiable only when no searched CAD placement has the
    # joint photometric, directional-edge, and layer-membership evidence of a
    # real positive.  One weak channel is sufficient to disprove the match.
    positive_like = (best.fit >= 0.68 and best.edge_corr >= 0.42 and
                     (best.appearance_corr >= 0.67 or best.eta_squared >= 0.50))
    if not present:
        if vector:
            vector_fits = [ce._fit_candidate(search, template, float(col), float(row), score,
                                             score)
                           for row, col, score in vector if score >= ce.MIN_VECTOR_COVERAGE]
            # Exact CAD alone never proves SEM presence.  A negative becomes
            # contradictory only when both the design and observed image
            # support the same placement.
            positive_like = any(
                item.fit >= 0.06 and
                (item.edge_corr >= 0.020 or
                 (item.eta_squared >= 0.035 and item.yield_fit >= 0.10))
                for item in vector_fits)
        result.update(valid=not positive_like,
                      reason="accidental_observable_match" if positive_like else "valid_negative")
        return result

    assert gt_x is not None and gt_y is not None
    side = template.regions.shape[-1]
    left = min(max(gt_x - side / 2.0, 0.0), search.shape[1] - side)
    top = min(max(gt_y - side / 2.0, 0.0), search.shape[0] - side)
    gt = ce._fit_candidate(search, template, left, top, 1.0)
    error = math.hypot(result["best_x"] - gt_x, result["best_y"] - gt_y)
    remote = max((rank(item) for item in pool
                  if math.hypot(item.left + side / 2.0 - gt_x,
                                item.top + side / 2.0 - gt_y) > 5.0),
                 default=-1.0)
    gt_rank = rank(gt)
    if vector:
        vector_near = [item for item in vector
                       if item[2] >= ce.MIN_VECTOR_COVERAGE and
                       math.hypot(item[1] + side / 2.0 - gt_x,
                                  item[0] + side / 2.0 - gt_y) <= 2.0]
        vector_fits = [ce._fit_candidate(search, template, float(col), float(row), score,
                                         score)
                       for row, col, score in vector if score >= ce.MIN_VECTOR_COVERAGE]
        gt_vector_rank = max((rank(item) for item in vector_fits
                              if math.hypot(item.left + side / 2.0 - gt_x,
                                            item.top + side / 2.0 - gt_y) <= 2.0),
                             default=-1.0)
        remote_vector_rank = max((rank(item) for item in vector_fits
                                  if math.hypot(item.left + side / 2.0 - gt_x,
                                                item.top + side / 2.0 - gt_y) > 5.0),
                                 default=-1.0)
        observable = (gt.fit >= 0.06 and
                      (gt.edge_corr >= 0.020 or
                       (gt.eta_squared >= 0.035 and gt.yield_fit >= 0.10)))
        unique = bool(vector_near) and gt_vector_rank >= remote_vector_rank + 0.01
        result.update(vector_candidates=len(vector),
                      gt_vector_rank=gt_vector_rank,
                      remote_vector_rank=remote_vector_rank)
    else:
        observable = (gt.fit >= 0.68 and gt.edge_corr >= 0.42 and
                      (gt.appearance_corr >= 0.67 or gt.eta_squared >= 0.50))
        unique = error <= 5.0 and gt_rank >= remote + 0.015
    valid = observable and unique
    reason = "valid_positive" if valid else (
        "unobservable_after_degradation" if not observable else "periodic_alias")
    result.update(valid=valid, reason=reason, gt_fit=gt.fit,
                  gt_edge=gt.edge_corr, gt_appearance=gt.appearance_corr,
                  gt_eta=gt.eta_squared, gt_rank=gt_rank,
                  remote_rank=remote, localization_error=error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.pairs.resolve().parent
    pairs = list(csv.DictReader(args.pairs.open(newline="", encoding="utf-8-sig")))
    truths = list(csv.DictReader(args.truth.open(newline="", encoding="utf-8-sig")))
    truth_by_id = {str(row.get("pair_id", row.get("id"))): row for row in truths}
    results = [_analyse(row, truth_by_id[row["pair_id"]], root) for row in pairs]
    summary = {
        "generated": len(results),
        "generated_positive": sum(item["present"] for item in results),
        "generated_negative": sum(not item["present"] for item in results),
        "certified": sum(item["valid"] for item in results),
        "certified_positive": sum(item["valid"] and item["present"] for item in results),
        "certified_negative": sum(item["valid"] and not item["present"] for item in results),
        "discarded_unobservable": sum(item["reason"] == "unobservable_after_degradation" for item in results),
        "discarded_alias": sum(item["reason"] == "periodic_alias" for item in results),
        "discarded_accidental_match": sum(item["reason"] == "accidental_observable_match" for item in results),
        "cases": results,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
