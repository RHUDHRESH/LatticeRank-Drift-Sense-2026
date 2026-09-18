"""Cheap local Cramer--Rao diagnostics for an aligned image pair.

The bound describes local pose jitter around an already selected match.  It
does not account for choosing the wrong correlation lobe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FisherConfidence:
    """Finite uncertainty diagnostics for ``(tx, ty, rotation, log_scale)``."""

    covariance: np.ndarray
    information: np.ndarray
    std_x: float
    std_y: float
    std_rotation_rad: float
    std_rotation_deg: float
    std_log_scale: float
    noise_sigma: float
    effective_samples: float
    correlation_inflation: float
    condition_number: float
    confidence: float
    identifiable: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _lag_correlation(values: np.ndarray, axis: int) -> float:
    """Robust-ish lag-one correlation, clipped to a useful finite range."""
    a = np.take(values, range(values.shape[axis] - 1), axis=axis).ravel()
    b = np.take(values, range(1, values.shape[axis]), axis=axis).ravel()
    if a.size == 0:
        return 0.0
    a = a - np.median(a)
    b = b - np.median(b)
    # Winsorisation prevents a handful of failed-fit pixels owning the ESS.
    limit = 6.0 * max(1.4826 * np.median(np.abs(values - np.median(values))),
                      np.finfo(np.float64).eps)
    a = np.clip(a, -limit, limit)
    b = np.clip(b, -limit, limit)
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    rho = 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)
    return float(np.clip(rho, 0.0, 0.95))


def fisher_confidence(
    template: np.ndarray,
    observed: np.ndarray,
    *,
    blur_sigma: float = 0.8,
    max_correlation_inflation: float = 100.0,
) -> FisherConfidence:
    """Estimate a local pose CRB from two already aligned grayscale patches.

    Rotation is expressed in radians and scale as log-scale, so the latter's
    standard deviation is approximately fractional scale uncertainty for small
    errors.  Residual MAD estimates noise while adjacent residual correlation
    reduces the nominal independent sample count.
    """
    reference = np.asarray(template, dtype=np.float64)
    data = np.asarray(observed, dtype=np.float64)
    if reference.ndim != 2 or data.ndim != 2 or reference.shape != data.shape:
        raise ValueError("template and observed must be same-shaped 2-D arrays")
    if reference.size == 0:
        raise ValueError("patches must not be empty")
    reference = np.nan_to_num(reference, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    residual = data - reference
    residual -= np.median(residual)
    mad_sigma = 1.4826 * float(np.median(np.abs(residual)))
    signal_range = float(np.percentile(reference, 99) - np.percentile(reference, 1))
    sigma_floor = max(signal_range * 1e-6, np.finfo(np.float32).eps)
    noise_sigma = max(mad_sigma, sigma_floor)

    rho_x = _lag_correlation(residual, 1)
    rho_y = _lag_correlation(residual, 0)
    inflation = ((1.0 + rho_x) / (1.0 - rho_x)) * ((1.0 + rho_y) / (1.0 - rho_y))
    inflation = float(np.clip(inflation, 1.0, max_correlation_inflation))
    effective_samples = float(reference.size / inflation)

    source = reference.astype(np.float32)
    if blur_sigma > 0:
        source = cv2.GaussianBlur(source, (0, 0), blur_sigma,
                                  borderType=cv2.BORDER_REFLECT101)
    # OpenCV's Scharr response is 32 times the derivative in pixel units.
    ix = cv2.Scharr(source, cv2.CV_64F, 1, 0,
                    borderType=cv2.BORDER_REFLECT101) / 32.0
    iy = cv2.Scharr(source, cv2.CV_64F, 0, 1,
                    borderType=cv2.BORDER_REFLECT101) / 32.0
    height, width = reference.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    xx -= (width - 1.0) / 2.0
    yy -= (height - 1.0) / 2.0
    jacobian = np.column_stack((
        ix.ravel(),
        iy.ravel(),
        (-yy * ix + xx * iy).ravel(),
        (xx * ix + yy * iy).ravel(),
    ))
    information = (jacobian.T @ jacobian) / (noise_sigma * noise_sigma * inflation)

    eigvals, eigenvectors = np.linalg.eigh(information)
    largest = max(float(eigvals[-1]), np.finfo(np.float64).tiny)
    tolerance = largest * 1e-10
    identifiable = bool(eigvals[0] > tolerance)
    condition = float(largest / max(float(eigvals[0]), np.finfo(np.float64).tiny))
    condition = min(condition, np.finfo(np.float64).max)

    # A bounded eigenvalue floor keeps downstream scoring JSON-safe on flat or
    # one-dimensional patches while identifiable explicitly communicates rank.
    regularization = max(tolerance, noise_sigma * noise_sigma * 1e-12,
                         np.finfo(np.float64).tiny)
    safe_eigenvalues = np.maximum(eigvals, regularization)
    covariance = (eigenvectors * (1.0 / safe_eigenvalues)) @ eigenvectors.T
    covariance = np.nan_to_num(covariance, nan=np.finfo(np.float64).max,
                               posinf=np.finfo(np.float64).max,
                               neginf=-np.finfo(np.float64).max)
    standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    # Pixel uncertainty is the most portable scalar for ranking local matches.
    translation_rms = float(np.hypot(standard[0], standard[1]))
    confidence = float(np.clip(1.0 / (1.0 + translation_rms), 0.0, 1.0))
    if not identifiable:
        confidence = 0.0

    return FisherConfidence(
        covariance=covariance,
        information=information,
        std_x=float(standard[0]),
        std_y=float(standard[1]),
        std_rotation_rad=float(standard[2]),
        std_rotation_deg=float(np.degrees(standard[2])),
        std_log_scale=float(standard[3]),
        noise_sigma=float(noise_sigma),
        effective_samples=effective_samples,
        correlation_inflation=inflation,
        condition_number=condition,
        confidence=confidence,
        identifiable=identifiable,
    )


# A concise alternate name reads naturally at call sites.
estimate_fisher_confidence = fisher_confidence
