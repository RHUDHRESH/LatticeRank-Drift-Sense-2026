#!/usr/bin/env python3
"""Phase 2 edge registration.

Usage: python register.py --input pairs.csv --output predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "LOKY_MAX_CPU_COUNT", "DRIFTFORGE_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

COLUMNS = ("pair_id", "x", "y", "theta", "scale", "found", "score")
ID_KEYS = ("pair_id", "id", "pairid", "pair")
REF_KEYS = ("reference_path", "reference", "reference_image", "ref_path", "ref", "ref_image")
SEARCH_KEYS = ("search_path", "search", "search_image", "src_path", "search_img")
REF_HINTS = ("ref", "template", "patch", "query")
SEARCH_HINTS = ("search", "wide", "scene", "haystack")
IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp")
COLOR_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}
NUMERIC_MODES = {"I", "F", "I;16", "I;16L", "I;16B", "I;16N"}
FAILURE_SCORE = 1e-6
BUDGET_S = 12.0  # compatibility with external runners
SEARCH_ROOTS: list[Path] = []


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items() if key}
    return next((str(lowered[key]) for key in keys
                 if lowered.get(key) not in (None, "")), None)


def _pick_image(row: dict, keys: tuple[str, ...], hints: tuple[str, ...],
                anti: tuple[str, ...]) -> str | None:
    exact = _pick(row, keys)
    if exact:
        return exact
    for name, value in row.items():
        if not name or value in (None, ""):
            continue
        lowered = str(name).strip().lower()
        text = str(value).strip()
        if (not any(token in lowered for token in anti)
                and any(token in lowered for token in hints)
                and text.lower().endswith(IMAGE_SUFFIXES)):
            return text
    return None


def resolve_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for root in SEARCH_ROOTS:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"image must contain one frame: {path}")
        image.load()
        if image.mode in COLOR_MODES:
            return np.asarray(image.convert("L"), dtype=np.uint8)
        if image.mode in NUMERIC_MODES:
            values = np.asarray(image, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"image must contain one plane: {path}")
            low, high = float(values.min()), float(values.max())
            if high <= low:
                return np.zeros(values.shape, dtype=np.uint8)
            return np.clip(np.rint((values - low) * 255.0 / (high - low)), 0, 255).astype(np.uint8)
        raise ValueError(f"unsupported image mode {image.mode!r}: {path}")


def solve_edge(reference: np.ndarray, search: np.ndarray) -> dict:
    from driftforge.edge_registration import solve_edges

    match = solve_edges(reference, search)
    values = np.asarray((match.x, match.y, match.theta, match.scale, match.score))
    if not np.isfinite(values).all():
        raise ValueError("solver returned non-finite output")
    score = float(np.clip(match.score, 0.0, 1.0))
    if not match.found:
        return {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0,
                "found": 0, "score": score}
    height, width = search.shape[:2]
    return {
        "x": float(np.clip(match.x, 0.0, width - 1)),
        "y": float(np.clip(match.y, 0.0, height - 1)),
        "theta": float(np.clip(match.theta, -10.0, 10.0)),
        "scale": float(np.clip(match.scale, 8.0, 12.0)),
        "found": 1,
        "score": score,
    }


def process(pair_id: str, reference_path: str | None,
            search_path: str | None, *_unused, **_unused_kw) -> tuple[dict, str | None]:
    row = {"pair_id": pair_id, "x": 0.0, "y": 0.0, "theta": 0.0,
           "scale": 0.0, "found": 0, "score": FAILURE_SCORE}
    try:
        if not reference_path or not search_path:
            return row, "missing reference or search path"
        result = solve_edge(load_image(resolve_path(reference_path)),
                            load_image(resolve_path(search_path)))
        row.update(result)
        return row, None
    except (FileNotFoundError, UnidentifiedImageError, ValueError, OSError) as exc:
        return row, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # every input still receives a valid row
        traceback.print_exc(file=sys.stderr)
        return row, f"unexpected {type(exc).__name__}: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 edge registration")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("edge",), default="edge", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    SEARCH_ROOTS.clear()
    SEARCH_ROOTS.extend((args.input.resolve().parent, Path(__file__).resolve().parent))
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("error: input CSV has no rows", file=sys.stderr)
        return 2

    results: list[dict] = []
    seen: set[str] = set()
    failures = 0
    started = time.perf_counter()
    for index, source in enumerate(rows):
        pair_id = _pick(source, ID_KEYS) or str(index)
        if pair_id in seen:
            continue
        seen.add(pair_id)
        result, reason = process(
            pair_id,
            _pick_image(source, REF_KEYS, REF_HINTS, SEARCH_HINTS),
            _pick_image(source, SEARCH_KEYS, SEARCH_HINTS, REF_HINTS),
        )
        results.append(result)
        if reason:
            failures += 1
            print(f"pair {pair_id}: {reason}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in COLUMNS} for row in results)
    temporary.replace(args.output)
    print(f"processed {len(results)} pairs in {time.perf_counter() - started:.2f}s; "
          f"failures={failures}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
