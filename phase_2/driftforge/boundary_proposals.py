"""Bounded photometric pose proposals with explicit boundary hypotheses.

This module only proposes poses and locations.  Callers must verify every
proposal with their independent edge objective before declaring a match.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundaryProposal:
    x: float
    y: float
    scale: float
    theta: float
    score: float
    margin: float
    boundary: str


def _render(reference: np.ndarray, scale: float, theta: float,
            border_mode: int) -> np.ndarray:
    height = max(8, int(round(reference.shape[0] / scale)))
    width = max(8, int(round(reference.shape[1] / scale)))
    source_center = ((reference.shape[1] - 1.0) / 2.0,
                     (reference.shape[0] - 1.0) / 2.0)
    target_center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    matrix = cv2.getRotationMatrix2D(source_center, theta, 1.0 / scale)
    matrix[:, 2] += np.subtract(target_center, source_center)
    # Integrate roughly one output footprint before shrinking.  This avoids
    # changing the proposal ranking with high-frequency raster phase.
    source = cv2.GaussianBlur(
        reference.astype(np.float32, copy=False), (0, 0),
        max(1.0, scale / 3.2), borderType=cv2.BORDER_REFLECT_101)
    return cv2.warpAffine(
        source, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=border_mode, borderValue=0.0)


def _surface_peaks(surface: np.ndarray, count: int,
                   radius: int) -> list[tuple[float, int, int]]:
    work = np.nan_to_num(surface, nan=-1.0).copy()
    peaks: list[tuple[float, int, int]] = []
    for _ in range(max(1, count)):
        _low, high, _low_at, high_at = cv2.minMaxLoc(work)
        col, row = high_at
        peaks.append((float(high), int(col), int(row)))
        work[max(0, row-radius):row+radius+1,
             max(0, col-radius):col+radius+1] = -1.0
    return peaks


def boundary_pose_proposals(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    scales: tuple[float, ...] = (8.0, 8.35, 9.0, 10.0, 11.0, 11.65, 12.0),
    angles: tuple[float, ...] = (-10.0, -5.0, -2.5, 0.0, 2.5, 5.0, 10.0),
    peaks_per_surface: int = 1,
    shortlist: int = 8,
) -> list[BoundaryProposal]:
    """Return a small pose/location shortlist from two FOV boundary models.

    Absolute ZNCC handles SEM contrast reversal.  Constant and reflected
    borders cover the two common ways a finite Reference crop is resampled.
    The fixed pose bank is intentionally small and CPU bounded.
    """
    ref = np.asarray(reference, dtype=np.float32)
    scene = np.asarray(search, dtype=np.float32)
    if (ref.ndim != 2 or scene.ndim != 2 or min(ref.shape) < 16
            or min(scene.shape) < 32 or float(ref.std()) < 1e-5):
        return []
    ranked: list[BoundaryProposal] = []
    modes = (("constant", cv2.BORDER_CONSTANT),
             ("reflect", cv2.BORDER_REFLECT_101))
    for boundary, border_mode in modes:
        for scale in scales:
            for theta in angles:
                template = _render(ref, float(scale), float(theta), border_mode)
                if (template.shape[0] >= scene.shape[0]
                        or template.shape[1] >= scene.shape[1]
                        or float(template.std()) < 1e-5):
                    continue
                surface = np.abs(cv2.matchTemplate(
                    scene, template, cv2.TM_CCOEFF_NORMED))
                radius = max(3, int(round(min(template.shape) * 0.3)))
                half_x = (template.shape[1] - 1.0) / 2.0
                half_y = (template.shape[0] - 1.0) / 2.0
                for score, col, row in _surface_peaks(
                        surface, peaks_per_surface, radius):
                    remote = surface.copy()
                    remote[max(0, row-radius):row+radius+1,
                           max(0, col-radius):col+radius+1] = -1.0
                    margin = float(score - remote.max())
                    ranked.append(BoundaryProposal(
                        x=col + half_x, y=row + half_y,
                        scale=float(scale), theta=float(theta), score=score,
                        margin=margin,
                        boundary=boundary))
    ranked.sort(key=lambda item: item.score, reverse=True)
    kept: list[BoundaryProposal] = []
    for item in ranked:
        if all(np.hypot(item.x-old.x, item.y-old.y) >= 4.0
               or abs(item.scale-old.scale) >= 0.3
               or abs(item.theta-old.theta) >= 1.0
               for old in kept):
            kept.append(item)
        if len(kept) >= max(1, shortlist):
            break
    return kept


__all__ = ["BoundaryProposal", "boundary_pose_proposals"]
