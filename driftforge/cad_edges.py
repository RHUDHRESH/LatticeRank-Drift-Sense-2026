"""Fixed-pose, layered CAD-edge registration for Phase 3.

This submission's current Phase 3 scope is axis-aligned at the documented
10 nm/search-pixel scale. The unknowns are translation and how each CAD layer appears in SEM.
This module searches translation only, then estimates layer contrasts at each
candidate with a tiny ridge solve. A faint layer is allowed to fit to zero.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

from .cadref import REFERENCE_SIZE_NM, rasterize_layer_masks, read_gds_geometry

NOMINAL_SCALE = 10.0
MAX_CANDIDATES = 18
MAX_LAYERS = 32


@dataclass(frozen=True)
class LayeredTemplate:
    layer_ids: tuple[int, ...]
    regions: np.ndarray
    proposal_edges: np.ndarray
    appearance: np.ndarray
    size_nm: float
    unit_mode: str


@dataclass(frozen=True)
class CandidateFit:
    left: float
    top: float
    fit: float
    intensity_corr: float
    edge_corr: float
    appearance_corr: float
    proposal: float
    geometry_support: float
    visible_layers: int


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ValueError("Phase 3 edge registration requires OpenCV (cv2)") from exc
    return cv2


@contextmanager
def _opencv_thread_scope():
    """Temporarily honor the repository's native thread budget."""
    cv2 = _cv2()
    previous = int(cv2.getNumThreads())
    raw = os.environ.get("DRIFTFORGE_NUM_THREADS", "1")
    try:
        requested = int(raw)
    except ValueError:
        requested = 1
    requested = max(1, min(requested, 32))
    cv2.setNumThreads(requested)
    try:
        yield cv2
    finally:
        cv2.setNumThreads(previous)


def _normalise_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values.mean(axis=-1)
    if values.ndim != 2 or min(values.shape) < 8:
        raise ValueError("search image must be a two-dimensional raster")
    finite = np.isfinite(values)
    if not finite.all():
        replacement = float(np.median(values[finite])) if finite.any() else 0.0
        values = np.where(finite, values, replacement)
    lo, hi = np.percentile(values, (1.0, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), 0.65, mode="reflect")
    gy, gx = np.gradient(smooth)
    magnitude = np.hypot(gx, gy).astype(np.float32)
    cap = float(np.percentile(magnitude, 99.0))
    if cap > 1e-8:
        magnitude = np.clip(magnitude / cap, 0.0, 1.0)
    return gx.astype(np.float32), gy.astype(np.float32), magnitude


def load_layered_template(gds_path: str | Path, *, scale: float = NOMINAL_SCALE,
                          unit_mode: str = "auto") -> LayeredTemplate:
    """Rasterize CAD layers at the search sampling pitch."""
    side = int(round(REFERENCE_SIZE_NM / float(scale)))
    if side < 8:
        raise ValueError("nominal scale makes the CAD footprint too small")
    masks, geometry = rasterize_layer_masks(
        gds_path, size_px=side, window_nm=REFERENCE_SIZE_NM,
        supersample=2, unit_mode=unit_mode, max_layers=MAX_LAYERS,
    )
    layer_ids = tuple(sorted(masks))
    if not layer_ids:
        raise ValueError(f"GDS has no rasterizable layers: {gds_path}")

    # GDS layer numbers identify masks; they are not a universal z-order.
    # Keep overlapping masks independent and let ridge fitting suppress an
    # invisible layer. This also permits a dropped top-layer hypothesis to
    # recover geometry underneath instead of permanently occluding it here.
    regions = np.stack([
        np.clip(masks[layer], 0.0, 1.0) for layer in layer_ids
    ]).astype(np.float32)

    edge_channels = []
    for mask in regions:
        _gx, _gy, edge = _gradient(mask)
        edge[:2, :] = edge[-2:, :] = 0.0
        edge[:, :2] = edge[:, -2:] = 0.0
        if float(edge.max()) > 0:
            edge_channels.append(edge)
    if not edge_channels:
        raise ValueError(f"GDS polygons have no usable boundaries: {gds_path}")
    stack = np.stack(edge_channels).astype(np.float32)
    pooled = 0.55 * np.max(stack, axis=0) + 0.45 * np.mean(stack, axis=0)
    pooled /= max(float(pooled.max()), 1e-6)
    proposals = np.concatenate((pooled[None, ...], stack), axis=0)
    appearance = np.full(regions.shape[1:], 0.12, dtype=np.float32)
    num_layers = max(layer_ids) + 1
    for layer, coverage in zip(layer_ids, regions):
        intensity = 0.85 if num_layers <= 1 else 0.20 + 0.65 * layer / (num_layers - 1)
        appearance = appearance * (1.0 - coverage) + float(intensity) * coverage
    return LayeredTemplate(layer_ids, regions, proposals.astype(np.float32), appearance,
                           REFERENCE_SIZE_NM, geometry.unit_mode)


def _proposal_candidates(search_edges: np.ndarray, template: LayeredTemplate,
                         search: np.ndarray | None = None
                         ) -> list[tuple[int, int, float]]:
    """Union pooled and per-layer peaks, then deduplicate nearby sites.

    The pooled channel is strongest when most layers are visible. Per-layer
    channels keep the true site in the candidate set when a geometrically
    dominant top layer is faint or entirely absent from the SEM image.
    """
    cv2 = _cv2()
    proposed: list[tuple[int, int, float]] = []
    if search is not None:
        response = cv2.matchTemplate(
            np.asarray(search, dtype=np.float32), template.appearance,
            cv2.TM_CCOEFF_NORMED,
        )
        proposed.extend(_distinct_candidates(
            response, 4, radius=max(8, template.appearance.shape[0] // 3)
        ))
    for index, edges in enumerate(template.proposal_edges):
        response = cv2.matchTemplate(search_edges, edges, cv2.TM_CCORR_NORMED)
        count = 8 if index == 0 else 3
        proposed.extend(_distinct_candidates(response, count, radius=max(8, edges.shape[0] // 3)))
    proposed.sort(key=lambda item: item[2], reverse=True)
    accepted: list[tuple[int, int, float]] = []
    # Only merge essentially identical peaks. Distinct periodic sites must
    # survive so alias ambiguity can lower confidence honestly.
    merge_radius_sq = 5 * 5
    for row, col, value in proposed:
        if any((row - ar) ** 2 + (col - ac) ** 2 <= merge_radius_sq
               for ar, ac, _av in accepted):
            continue
        accepted.append((row, col, value))
        if len(accepted) >= MAX_CANDIDATES:
            break
    return accepted


def _polygon_signature(points: np.ndarray, quantum_nm: float = 0.01) -> tuple:
    """Translation-invariant polygon signature, insensitive to vertex start."""
    values = np.rint(np.asarray(points, dtype=np.float64) / quantum_nm).astype(np.int64)
    centre = np.rint(values.mean(axis=0)).astype(np.int64)
    relative = [tuple(map(int, point - centre)) for point in values]
    variants = []
    for sequence in (relative, list(reversed(relative))):
        variants.extend(tuple(sequence[index:] + sequence[:index])
                        for index in range(len(sequence)))
    return min(variants)


def _vector_cad_proposal_candidates(reference_gds_path: str | Path,
                                    search_gds_path: str | Path,
                                    search_shape: tuple[int, int],
                                    scale: float
                                    ) -> list[tuple[int, int, float]]:
    """Vote for translation using exact ideal-design polygon geometry."""
    height, width = search_shape
    reference = read_gds_geometry(reference_gds_path, window_nm=REFERENCE_SIZE_NM)
    search = read_gds_geometry(search_gds_path, window_nm=float(width) * scale)
    search_index: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for layer_key, polygons in search.polygons.items():
        for points in polygons:
            search_index[(layer_key, _polygon_signature(points))].append(points.mean(axis=0))

    votes: dict[tuple[int, int], dict[str, object]] = {}
    matchable_reference_count = 0
    matchable_layers: set[tuple[int, int]] = set()
    max_pairs = 500_000
    used_pairs = 0
    for layer_key, polygons in reference.polygons.items():
        for points in polygons:
            matches = search_index.get((layer_key, _polygon_signature(points)), ())
            if not matches:
                continue
            matchable_reference_count += 1
            matchable_layers.add(layer_key)
            anchor = points.mean(axis=0)
            for search_anchor in matches:
                if used_pairs >= max_pairs:
                    break
                delta = np.asarray(search_anchor) - anchor
                key = (int(round(float(delta[0]) * 10.0)),
                       int(round(float(delta[1]) * 10.0)))  # 0.1 nm bins
                record = votes.setdefault(key, {"count": 0, "layers": set()})
                record["count"] = int(record["count"]) + 1
                record["layers"].add(layer_key)
                used_pairs += 1
            if used_pairs >= max_pairs:
                break
        if used_pairs >= max_pairs:
            break
    if not votes or not matchable_reference_count:
        return []

    proposals = []
    layer_denominator = max(len(matchable_layers), 1)
    max_left = width - int(round(REFERENCE_SIZE_NM / scale))
    max_top = height - int(round(REFERENCE_SIZE_NM / scale))
    for (dx_bin, dy_bin), record in votes.items():
        left = dx_bin / (10.0 * scale)
        top = dy_bin / (10.0 * scale)
        if not (0.0 <= left <= max_left and 0.0 <= top <= max_top):
            continue
        support = min(int(record["count"]) / matchable_reference_count, 1.0)
        layer_support = len(record["layers"]) / layer_denominator
        score = float(np.clip(0.75 * support + 0.25 * layer_support, 0.0, 1.0))
        proposals.append((int(round(top)), int(round(left)), score))
    proposals.sort(key=lambda item: item[2], reverse=True)
    return proposals[:MAX_CANDIDATES // 2]


def _raster_cad_proposal_candidates(search_gds_path: str | Path,
                                    template: LayeredTemplate,
                                    search_shape: tuple[int, int]
                                    ) -> list[tuple[int, int, float]]:
    """Propose translations by matching the cropped CAD in full-canvas CAD.

    The organizer exports both files from the same ideal design database.
    This path uses geometry only for proposals; the observed SEM still ranks
    candidates and determines presence, so an unrelated reference is not
    accepted merely because periodic CAD produces a lookalike.
    """
    height, width = search_shape
    if height != width:
        return []
    masks, _geometry = rasterize_layer_masks(
        search_gds_path, size_px=width,
        window_nm=float(width) * NOMINAL_SCALE,
        supersample=1, max_layers=MAX_LAYERS,
    )
    cv2 = _cv2()
    votes: list[tuple[int, int, int, float]] = []
    eligible_layers = 0
    for layer, reference in zip(template.layer_ids, template.regions):
        search_layer = masks.get(layer)
        if search_layer is None or float(np.std(reference)) <= 2e-3:
            continue
        eligible_layers += 1
        response = cv2.matchTemplate(
            np.asarray(search_layer, dtype=np.float32),
            np.asarray(reference, dtype=np.float32),
            cv2.TM_CCOEFF_NORMED,
        )
        votes.extend((layer, row, col, value) for row, col, value in
                     _distinct_candidates(
                         response, 3,
                         radius=max(8, reference.shape[0] // 3)
                     ))

    # A position supported independently by several layers is much stronger
    # design evidence than a high peak from one periodic layer. Cluster first,
    # retaining one vote per layer, before applying the bounded CAD quota.
    clusters: list[dict] = []
    for layer, row, col, value in sorted(votes, key=lambda item: item[3], reverse=True):
        cluster = next((item for item in clusters
                        if (row - item["row"]) ** 2 +
                           (col - item["col"]) ** 2 <= 25), None)
        if cluster is None:
            cluster = {"row": float(row), "col": float(col),
                       "weight": 0.0, "layers": {}}
            clusters.append(cluster)
        previous = cluster["layers"].get(layer)
        if previous is None or value > previous:
            cluster["layers"][layer] = value
            weight = max(value, 1e-6)
            total = cluster["weight"] + weight
            cluster["row"] = (cluster["row"] * cluster["weight"] + row * weight) / total
            cluster["col"] = (cluster["col"] * cluster["weight"] + col * weight) / total
            cluster["weight"] = total

    distinct: list[tuple[int, int, float]] = []
    denominator = max(eligible_layers, 1)
    for cluster in clusters:
        layer_scores = tuple(cluster["layers"].values())
        support = len(layer_scores) / denominator
        mean_corr = float(np.mean(layer_scores)) if layer_scores else 0.0
        geometry_score = float(np.clip(0.65 * support + 0.35 * mean_corr,
                                       0.0, 1.0))
        distinct.append((int(round(cluster["row"])),
                         int(round(cluster["col"])), geometry_score))
    distinct.sort(key=lambda item: item[2], reverse=True)
    # Preserve half of the bounded pool for image-only proposals. This keeps
    # the optional CAD channel helpful without making it a hard dependency.
    return distinct[:MAX_CANDIDATES // 2]


def _cad_proposal_candidates(search_gds_path: str | Path,
                             template: LayeredTemplate,
                             search_shape: tuple[int, int],
                             *, reference_gds_path: str | Path | None = None,
                             scale: float = NOMINAL_SCALE
                             ) -> list[tuple[int, int, float]]:
    """Prefer fast exact vector votes, falling back to layered raster CAD."""
    if reference_gds_path is not None:
        vector = _vector_cad_proposal_candidates(
            reference_gds_path, search_gds_path, search_shape, scale
        )
        if vector:
            return vector
    return _raster_cad_proposal_candidates(search_gds_path, template, search_shape)


def _distinct_candidates(response: np.ndarray, count: int,
                         radius: int) -> list[tuple[int, int, float]]:
    work = np.asarray(response, dtype=np.float32).copy()
    candidates: list[tuple[int, int, float]] = []
    for _ in range(count):
        index = int(np.argmax(work))
        row, col = np.unravel_index(index, work.shape)
        value = float(work[row, col])
        if not np.isfinite(value):
            break
        candidates.append((int(row), int(col), value))
        y0, y1 = max(0, row - radius), min(work.shape[0], row + radius + 1)
        x0, x1 = max(0, col - radius), min(work.shape[1], col + radius + 1)
        work[y0:y1, x0:x1] = -np.inf
    return candidates


def _extract_patch(image: np.ndarray, left: float, top: float,
                   side: int) -> np.ndarray:
    cv2 = _cv2()
    centre = (float(left + (side - 1) / 2.0), float(top + (side - 1) / 2.0))
    return cv2.getRectSubPix(np.asarray(image, dtype=np.float32), (side, side), centre)


def _detrend_plane(patch: np.ndarray) -> np.ndarray:
    height, width = patch.shape
    yy, xx = np.mgrid[-1.0:1.0:height * 1j, -1.0:1.0:width * 1j]
    nuisance = np.column_stack((np.ones(height * width), xx.ravel(), yy.ravel()))
    target = patch.ravel().astype(np.float64)
    beta, *_ = np.linalg.lstsq(nuisance, target, rcond=None)
    return (target - nuisance @ beta).reshape(height, width).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denominator) if denominator > 1e-12 else 0.0


def _fit_candidate(search: np.ndarray, template: LayeredTemplate,
                   left: float, top: float, proposal: float,
                   geometry_support: float = 0.0) -> CandidateFit:
    side = template.regions.shape[-1]
    patch = _detrend_plane(_extract_patch(search, left, top, side))
    features = ndimage.gaussian_filter(template.regions, sigma=(0.0, 0.55, 0.55), mode="nearest")
    matrix = features.reshape(features.shape[0], -1).T.astype(np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    useful = np.std(matrix, axis=0) > 2e-3
    matrix = matrix[:, useful]
    if matrix.shape[1] == 0:
        return CandidateFit(left, top, 0.0, 0.0, 0.0, 0.0, proposal,
                            geometry_support, 0)

    target = patch.ravel().astype(np.float64)
    target -= target.mean()
    yy, xx = np.indices((side, side))
    train = ((xx + yy) & 1).ravel() == 0
    train_matrix = matrix[train]
    normal = train_matrix.T @ train_matrix
    ridge = max(float(np.trace(normal)) / max(matrix.shape[1], 1) * 1e-3, 1e-8)
    beta = np.linalg.solve(normal + ridge * np.eye(matrix.shape[1]),
                           train_matrix.T @ target[train])
    predicted = (matrix @ beta).reshape(side, side).astype(np.float32)
    observed = target.reshape(side, side).astype(np.float32)

    intensity_corr = max(0.0, _cosine(predicted, observed))
    appearance = _detrend_plane(template.appearance)
    appearance_corr = max(0.0, _cosine(appearance, observed))
    pgx, pgy, _pm = _gradient(predicted)
    ogx, ogy, _om = _gradient(observed)
    edge_corr = max(0.0, _cosine(np.stack((pgx, pgy)), np.stack((ogx, ogy))))
    fit = float(np.clip(0.62 * intensity_corr + 0.38 * edge_corr, 0.0, 1.0))

    feature_std = np.std(matrix, axis=0)
    effects = np.abs(beta) * feature_std
    threshold = max(float(np.std(target)) * 0.025, 1e-4)
    visible_layers = int(np.count_nonzero(effects > threshold))
    return CandidateFit(float(left), float(top), fit, intensity_corr,
                        edge_corr, appearance_corr, float(proposal),
                        float(geometry_support),
                        visible_layers)


def _parabolic_offset(low: float, centre: float, high: float) -> float:
    denominator = low - 2.0 * centre + high
    if not np.isfinite(denominator) or denominator >= -1e-8:
        return 0.0
    return float(np.clip(0.5 * (low - high) / denominator, -0.75, 0.75))


def _refine_candidate(search: np.ndarray, template: LayeredTemplate,
                      candidate: CandidateFit) -> CandidateFit:
    x0, y0 = candidate.left, candidate.top
    side = template.regions.shape[-1]
    max_x = float(search.shape[1] - side)
    max_y = float(search.shape[0] - side)
    xm = _fit_candidate(search, template, max(0.0, x0 - 1.0), y0,
                        candidate.proposal, candidate.geometry_support)
    xp = _fit_candidate(search, template, min(max_x, x0 + 1.0), y0,
                        candidate.proposal, candidate.geometry_support)
    dx = _parabolic_offset(xm.fit, candidate.fit, xp.fit)
    refined_x = float(np.clip(x0 + dx, 0.0, max_x))
    shifted = _fit_candidate(search, template, refined_x, y0,
                             candidate.proposal, candidate.geometry_support)
    ym = _fit_candidate(search, template, refined_x, max(0.0, y0 - 1.0),
                        candidate.proposal, candidate.geometry_support)
    yp = _fit_candidate(search, template, refined_x, min(max_y, y0 + 1.0),
                        candidate.proposal, candidate.geometry_support)
    dy = _parabolic_offset(ym.fit, shifted.fit, yp.fit)
    refined_y = float(np.clip(y0 + dy, 0.0, max_y))
    refined = _fit_candidate(search, template, refined_x, refined_y,
                             candidate.proposal, candidate.geometry_support)
    return refined if refined.fit >= candidate.fit else candidate


def _solve_cad_edges(gds_path: str | Path, search_image: np.ndarray, *,
                     scale: float, unit_mode: str,
                     search_gds_path: str | Path | None = None
                     ) -> dict[str, float | int]:
    search = _normalise_image(search_image)
    template = load_layered_template(gds_path, scale=scale, unit_mode=unit_mode)
    side = int(template.regions.shape[-1])
    if search.shape[0] < side or search.shape[1] < side:
        raise ValueError("search image is smaller than the CAD footprint")
    _sx, _sy, search_edges = _gradient(search)
    image_candidates = _proposal_candidates(search_edges, template, search)
    raw_candidates = [(row, col, proposal, 0.0)
                      for row, col, proposal in image_candidates]
    if search_gds_path is not None:
        cad_candidates = _cad_proposal_candidates(
            search_gds_path, template, search.shape,
            reference_gds_path=gds_path, scale=scale,
        )
        combined = ([(row, col, score, score)
                     for row, col, score in cad_candidates] + raw_candidates)
        deduplicated: list[tuple[int, int, float, float]] = []
        for row, col, proposal, geometry_support in combined:
            if any((row - old_row) ** 2 + (col - old_col) ** 2 <= 25
                   for old_row, old_col, _old_proposal, _old_geometry in deduplicated):
                continue
            deduplicated.append((row, col, proposal, geometry_support))
            if len(deduplicated) >= MAX_CANDIDATES:
                break
        raw_candidates = deduplicated
    fitted = [
        _fit_candidate(search, template, float(col), float(row), proposal,
                       geometry_support)
        for row, col, proposal, geometry_support in raw_candidates
    ]
    if not fitted:
        return {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0,
                "found": 0, "score": 0.0}

    order_value = lambda item: (0.58 * item.fit + 0.27 * item.appearance_corr +
                                0.05 * item.proposal +
                                0.10 * item.geometry_support)
    fitted.sort(key=order_value, reverse=True)
    refined = [_refine_candidate(search, template, item) for item in fitted[:3]]
    pool = refined + fitted[3:]
    pool.sort(key=order_value, reverse=True)
    best = pool[0]
    second_fit = pool[1].fit if len(pool) > 1 else 0.0
    ambiguity_margin = max(0.0, best.fit - second_fit)

    evidence = float(np.clip((best.fit - 0.10) / 0.70, 0.0, 1.0))
    ambiguity = 0.55 + 0.45 * float(np.clip(ambiguity_margin / 0.10, 0.0, 1.0))
    layer_support = 0.75 + 0.25 * min(best.visible_layers, 3) / 3.0
    confidence = float(np.clip(evidence * ambiguity * layer_support, 0.0, 1.0))
    found = int(best.fit >= 0.20 and best.edge_corr >= 0.06 and best.proposal >= 0.04)
    if not found:
        return {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0,
                "found": 0, "score": confidence}
    return {
        "x": float(best.left + side / 2.0),
        "y": float(best.top + side / 2.0),
        "theta": 0.0,
        "scale": float(scale),
        "found": 1,
        "score": confidence,
    }


def solve_cad_edges(gds_path: str | Path, search_image: np.ndarray, *,
                    scale: float = NOMINAL_SCALE,
                    unit_mode: str = "auto",
                    search_gds_path: str | Path | None = None
                    ) -> dict[str, float | int]:
    """Locate a 1000 nm CAD footprint in a SEM search image."""
    with _opencv_thread_scope():
        return _solve_cad_edges(gds_path, search_image, scale=scale,
                                unit_mode=unit_mode,
                                search_gds_path=search_gds_path)
