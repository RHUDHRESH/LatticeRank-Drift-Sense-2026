"""GDS geometry loading and rasterization for Phase 3.

The challenge generator writes coordinates numerically in nanometres but uses
``gdstk.Library()``'s default micrometre unit. Production GDS files normally
mean what their unit record says. :func:`read_gds_geometry` supports both:
``unit_mode="gds"`` obeys the file, ``unit_mode="numeric-nm"`` treats numeric
coordinates as nanometres, and the default ``"auto"`` selects the convention
whose layout extent is plausible for the documented 1000 nm reference window.

The old :func:`rasterize_gds` API remains available. The Phase 3 matcher uses
:func:`rasterize_layer_masks`, keeping layers separate so their SEM contrast
can be estimated from the observed image and an invisible layer can get zero
weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REFERENCE_SIZE_PX = 1000
REFERENCE_SIZE_NM = 1000.0
BASE_YIELD = 0.20
TOP_YIELD = 0.85
BACKGROUND_YIELD = 0.12


@dataclass(frozen=True)
class GDSGeometry:
    """Flattened top-level GDS polygons expressed in nanometres."""

    polygons: dict[tuple[int, int], tuple[np.ndarray, ...]]
    origin_nm: tuple[float, float]
    bounds_nm: tuple[float, float, float, float]
    nm_per_user_unit: float
    unit_mode: str


def layer_intensity(layer: int, num_layers: int) -> int:
    """Legacy preview intensity derived from a layer's numeric ID."""
    frac = 1.0 if num_layers <= 1 else layer / (num_layers - 1)
    value = BASE_YIELD + (TOP_YIELD - BASE_YIELD) * frac
    return int(round(np.clip(value, 0.0, 1.0) * 255))


def background_intensity() -> int:
    return int(round(BACKGROUND_YIELD * 255))


def _read_library(gds_path: str | Path):
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError("reading a GDS reference requires the 'gdstk' package") from exc
    library = gdstk.read_gds(str(gds_path))
    cells = library.top_level()
    if not cells:
        raise ValueError(f"GDS has no top-level cell: {gds_path}")
    return library, cells


def read_top_cell(gds_path: str | Path):
    """Return the first top-level cell (compatibility with the original API)."""
    return _read_library(gds_path)[1][0]


def cell_layer_polygons(cell) -> dict[int, list[np.ndarray]]:
    """Group flattened polygons by layer ID in the cell's numeric units."""
    grouped: dict[int, list[np.ndarray]] = {}
    for polygon in cell.get_polygons():
        grouped.setdefault(int(polygon.layer), []).append(
            np.asarray(polygon.points, dtype=np.float64)
        )
    return grouped


def _plausibility(extent_nm: float, window_nm: float) -> float:
    if not np.isfinite(extent_nm) or extent_nm <= 0:
        return float("inf")
    ratio = extent_nm / window_nm
    penalty = abs(float(np.log10(ratio)))
    if ratio < 0.02 or ratio > 20.0:
        penalty += 4.0
    return penalty


def read_gds_geometry(
    gds_path: str | Path,
    *,
    unit_mode: str = "auto",
    nm_per_user_unit: float | None = None,
    window_nm: float = REFERENCE_SIZE_NM,
    origin_nm: tuple[float, float] | None = None,
) -> GDSGeometry:
    """Load flattened top-level polygons and convert coordinates to nm.

    ``unit_mode`` is ``"auto"``, ``"gds"``, or ``"numeric-nm"``. An
    explicit ``nm_per_user_unit`` overrides it. The documented reference
    frame always has origin ``(0, 0)`` unless ``origin_nm`` is explicitly
    supplied. In particular, this function never tight-crops a sparse layout:
    empty margins are part of the 1000 nm reference footprint.
    """
    library, cells = _read_library(gds_path)
    raw: dict[tuple[int, int], list[np.ndarray]] = {}
    for cell in cells:
        for polygon in cell.get_polygons():
            key = (int(polygon.layer), int(polygon.datatype))
            points = np.asarray(polygon.points, dtype=np.float64)
            if len(points) >= 3 and np.isfinite(points).all():
                raw.setdefault(key, []).append(points)
    if not raw:
        raise ValueError(f"GDS has no polygons: {gds_path}")

    all_points = np.concatenate([p for values in raw.values() for p in values], axis=0)
    lo_raw = all_points.min(axis=0)
    hi_raw = all_points.max(axis=0)
    raw_extent = float(np.max(hi_raw - lo_raw))
    file_nm = float(library.unit) * 1e9

    if nm_per_user_unit is not None:
        factor, selected = float(nm_per_user_unit), "explicit"
    elif unit_mode == "gds":
        factor, selected = file_nm, "gds"
    elif unit_mode in {"numeric-nm", "nm"}:
        factor, selected = 1.0, "numeric-nm"
    elif unit_mode == "auto":
        raw_score = _plausibility(raw_extent, window_nm)
        gds_score = _plausibility(raw_extent * file_nm, window_nm)
        factor, selected = ((1.0, "numeric-nm") if raw_score <= gds_score
                            else (file_nm, "gds"))
    else:
        raise ValueError(f"unknown GDS unit mode: {unit_mode}")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("nm_per_user_unit must be a positive finite number")

    converted = {
        key: tuple(np.asarray(points * factor, dtype=np.float64) for points in values)
        for key, values in raw.items()
    }
    lo = lo_raw * factor
    hi = hi_raw * factor
    origin = ((0.0, 0.0) if origin_nm is None
              else (float(origin_nm[0]), float(origin_nm[1])))
    return GDSGeometry(
        polygons=converted,
        origin_nm=origin,
        bounds_nm=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
        nm_per_user_unit=factor,
        unit_mode=selected,
    )


def _draw_mask(points_groups: list[np.ndarray], *, size_px: int,
               window_nm: float, origin_nm: tuple[float, float]) -> np.ndarray:
    canvas = Image.new("L", (size_px, size_px), 0)
    painter = ImageDraw.Draw(canvas)
    scale = size_px / float(window_nm)
    origin = np.asarray(origin_nm, dtype=np.float64)
    for points in points_groups:
        pixels = (points - origin) * scale
        flat = [tuple(map(float, point)) for point in pixels]
        if len(flat) >= 3:
            painter.polygon(flat, fill=255)
    return np.asarray(canvas, dtype=np.uint8)


def rasterize_layer_masks(
    gds_path: str | Path,
    *,
    size_px: int = REFERENCE_SIZE_PX,
    window_nm: float = REFERENCE_SIZE_NM,
    supersample: int = 1,
    unit_mode: str = "auto",
    nm_per_user_unit: float | None = None,
    origin_nm: tuple[float, float] | None = None,
    max_layers: int | None = None,
) -> tuple[dict[int, np.ndarray], GDSGeometry]:
    """Return per-layer fractional-coverage masks and geometry metadata."""
    if size_px <= 0 or supersample <= 0 or window_nm <= 0:
        raise ValueError("raster dimensions and window_nm must be positive")
    geometry = read_gds_geometry(
        gds_path, unit_mode=unit_mode, nm_per_user_unit=nm_per_user_unit,
        window_nm=window_nm, origin_nm=origin_nm,
    )
    grouped: dict[int, list[np.ndarray]] = {}
    for (layer, _datatype), polygons in geometry.polygons.items():
        grouped.setdefault(layer, []).extend(polygons)
    if max_layers is not None and len(grouped) > max_layers:
        raise ValueError(
            f"GDS has {len(grouped)} layers; bounded matcher limit is {max_layers}"
        )

    render_size = int(size_px * supersample)
    masks: dict[int, np.ndarray] = {}
    for layer in sorted(grouped):
        mask = _draw_mask(grouped[layer], size_px=render_size,
                          window_nm=window_nm, origin_nm=geometry.origin_nm)
        values = mask.astype(np.float32) / 255.0
        if supersample > 1:
            values = values.reshape(size_px, supersample, size_px, supersample).mean(axis=(1, 3))
        masks[layer] = values.astype(np.float32, copy=False)
    return masks, geometry


def rasterize_gds(gds_path: str | Path, size_px: int = REFERENCE_SIZE_PX,
                  nm_per_px: float = 1.0, return_layers: bool = False,
                  *, unit_mode: str = "auto",
                  nm_per_user_unit: float | None = None):
    """Rasterize a GDS preview while preserving the original public API."""
    masks, _geometry = rasterize_layer_masks(
        gds_path, size_px=size_px, window_nm=float(size_px) * float(nm_per_px),
        unit_mode=unit_mode, nm_per_user_unit=nm_per_user_unit,
    )
    raster = np.full((size_px, size_px), background_intensity(), dtype=np.uint8)
    num_layers = max(masks) + 1
    boolean_masks: dict[int, np.ndarray] = {}
    for layer, coverage in masks.items():
        hit = coverage > 0.0
        raster[hit] = layer_intensity(layer, num_layers)
        boolean_masks[layer] = hit
    if return_layers:
        return raster, boolean_masks
    return raster
