"""Small, inference-only CSV and image I/O helpers for Phase 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


COLUMNS = ("pair_id", "x", "y", "theta", "scale", "found", "score")
ID_KEYS = ("pair_id", "id", "pairid", "pair")
SEARCH_KEYS = ("search_path", "search", "search_image", "src_path", "search_img")
REF_HINTS = ("ref", "template", "patch", "query")
SEARCH_HINTS = ("search", "wide", "scene", "haystack")
IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp")
COLOR_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}
NUMERIC_MODES = {"I", "F", "I;16", "I;16L", "I;16B", "I;16N"}
FAILURE_SCORE = 1e-6
BUDGET_S = 12.0
SEARCH_ROOTS: list[Path] = []


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
            scaled = (values - low) * 255.0 / (high - low)
            return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
        raise ValueError(f"unsupported image mode {image.mode!r}: {path}")
