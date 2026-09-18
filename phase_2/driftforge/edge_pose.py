"""Bounded scale/rotation proposals from directional edge spectra.

The Reference and Search have different fields of view, so their Fourier
spectra cannot establish translation and must never be treated as a complete
registration.  This helper keeps the direction of a small set of prominent
2-D frequencies, votes only inside the disclosed pose bounds, and reports a
few deliberately capped-confidence seeds for spatial verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SpectralPose:
    """One bounded pose seed and the spread of the supporting frequency votes.

    ``scale_uncertainty`` and ``theta_uncertainty_deg`` are robust one-sigma
    spreads with a floor from the finite FFT-bin resolution.  ``confidence``
    is evidence for proposing this pose, not a probability that the target is
    present; it is capped at 0.45 because the Search has a larger field of view.
    """

    scale: float
    theta: float
    confidence: float
    scale_uncertainty: float
    theta_uncertainty_deg: float
    support: int
    directional_diversity_deg: float


@dataclass(frozen=True)
class _Peak:
    fx: float
    fy: float
    radius: float
    angle_deg: float
    weight: float
    relative_radial_resolution: float
    angular_resolution_deg: float


@dataclass(frozen=True)
class _Vote:
    scale: float
    theta: float
    weight: float
    ref_index: int
    search_index: int
    scale_floor: float
    theta_floor_deg: float


def _wrap_axis_angle(angle_deg: float) -> float:
    """Wrap an unoriented spectral-axis angle to ``[-90, 90)`` degrees."""
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _directional_diversity(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 0.0
    differences = np.abs(angles[:, None] - angles[None, :])
    differences = np.minimum(differences, 180.0 - differences)
    return float(np.max(differences))


def _spectral_peaks(edge: np.ndarray, max_peaks: int) -> list[_Peak]:
    values = np.asarray(edge, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 16 or max_peaks <= 0:
        return []
    finite = np.isfinite(values)
    if not finite.all():
        replacement = float(np.median(values[finite])) if finite.any() else 0.0
        values = np.where(finite, values, replacement)
    values = values - float(values.mean())
    if float(np.sqrt(np.mean(values * values))) <= 1e-7:
        return []

    height, width = values.shape
    taper = (np.hanning(height).astype(np.float32)[:, None]
             * np.hanning(width).astype(np.float32)[None, :])
    spectrum = np.fft.rfft2(values * taper)
    power = np.log1p(np.abs(spectrum) ** 2)
    fy = np.broadcast_to(np.fft.fftfreq(height)[:, None], power.shape)
    fx = np.broadcast_to(np.fft.rfftfreq(width)[None, :], power.shape)
    radius = np.hypot(fx, fy)
    valid = (radius >= 0.018) & (radius <= 0.42)
    # The rFFT half-plane already removes antipodal duplicates except on fx=0.
    valid &= (fx > 0.0) | (fy >= 0.0)
    if not np.any(valid):
        return []

    # Subtract the radial noise floor so broadband edges do not displace narrow
    # lattice peaks merely because high-frequency rings contain more samples.
    radial_bins = 64
    slots = np.minimum((radius * radial_bins / 0.42).astype(np.int32),
                       radial_bins - 1)
    background = np.zeros(radial_bins, dtype=np.float64)
    for slot in range(radial_bins):
        members = valid & (slots == slot)
        if np.any(members):
            background[slot] = float(np.median(power[members]))
    prominence = power - background[slots]
    candidate_mask = valid & (prominence > 0.0)
    candidate_ids = np.flatnonzero(candidate_mask)
    if not len(candidate_ids):
        return []
    keep = min(len(candidate_ids), max(128, max_peaks * 40))
    if keep < len(candidate_ids):
        local = np.argpartition(prominence.ravel()[candidate_ids], -keep)[-keep:]
        candidate_ids = candidate_ids[local]
    candidate_ids = candidate_ids[
        np.argsort(prominence.ravel()[candidate_ids], kind="stable")[::-1]
    ]

    maximum_prominence = max(float(np.max(prominence[candidate_mask])), 1e-9)
    peaks: list[_Peak] = []
    for flat_index in candidate_ids:
        row, col = np.unravel_index(int(flat_index), power.shape)
        x_frequency, y_frequency = float(fx[row, col]), float(fy[row, col])
        # Greedy NMS in native FFT-bin coordinates. The vertical axis wraps.
        duplicate = False
        for old in peaks:
            dx_bins = (x_frequency - old.fx) * width
            dy_bins = abs(y_frequency - old.fy) * height
            dy_bins = min(dy_bins, abs(dy_bins - height))
            # A Hann-windowed sinusoid occupies several adjacent FFT cells.
            # Suppress that main lobe so it cannot masquerade as several
            # independent lattice directions or harmonics.
            if dx_bins * dx_bins + dy_bins * dy_bins < 4.5 * 4.5:
                duplicate = True
                break
        if duplicate:
            continue
        radial = float(radius[row, col])
        angle = math.degrees(math.atan2(y_frequency, x_frequency))
        dfx, dfy = 1.0 / width, 1.0 / height
        # Treat a peak's location inside one FFT cell as a uniform rounding
        # uncertainty, then project that cell onto radial/angular directions.
        radial_sigma = math.hypot(
            (x_frequency / radial) * dfx,
            (y_frequency / radial) * dfy,
        ) / math.sqrt(12.0)
        angular_sigma = math.hypot(
            (y_frequency / (radial * radial)) * dfx,
            (x_frequency / (radial * radial)) * dfy,
        ) / math.sqrt(12.0)
        peaks.append(_Peak(
            fx=x_frequency,
            fy=y_frequency,
            radius=radial,
            angle_deg=_wrap_axis_angle(angle),
            weight=float(np.clip(
                prominence[row, col] / maximum_prominence, 0.01, 1.0)),
            relative_radial_resolution=float(radial_sigma / radial),
            angular_resolution_deg=float(math.degrees(angular_sigma)),
        ))
        if len(peaks) >= max_peaks:
            break
    return peaks


def _pose_votes(
    reference: list[_Peak],
    search: list[_Peak],
    nominal_scale: float,
    scale_bounds: tuple[float, float],
    theta_bounds: tuple[float, float],
) -> list[_Vote]:
    votes: list[_Vote] = []
    for ref_index, ref in enumerate(reference):
        for search_index, scene in enumerate(search):
            scale = nominal_scale * scene.radius / ref.radius
            if not (scale_bounds[0] <= scale <= scale_bounds[1]):
                continue
            # FFT y coordinates increase down the image. A positive cv2 image
            # rotation therefore shifts a frequency's y-down polar angle by
            # the negative of the reported theta.
            theta = -_wrap_axis_angle(scene.angle_deg - ref.angle_deg)
            if not (theta_bounds[0] <= theta <= theta_bounds[1]):
                continue
            relative_floor = math.hypot(
                ref.relative_radial_resolution,
                scene.relative_radial_resolution,
            )
            votes.append(_Vote(
                scale=float(scale),
                theta=float(theta),
                weight=float(math.sqrt(ref.weight * scene.weight)),
                ref_index=ref_index,
                search_index=search_index,
                scale_floor=float(scale * relative_floor),
                theta_floor_deg=float(math.hypot(
                    ref.angular_resolution_deg,
                    scene.angular_resolution_deg,
                )),
            ))
    return votes


def _one_to_one_members(votes: list[_Vote], indices: np.ndarray) -> list[_Vote]:
    """Keep the strongest correspondence for each Reference and Search peak."""
    ordered = sorted((votes[int(index)] for index in indices),
                     key=lambda vote: vote.weight, reverse=True)
    accepted: list[_Vote] = []
    used_reference: set[int] = set()
    used_search: set[int] = set()
    for vote in ordered:
        if vote.ref_index in used_reference or vote.search_index in used_search:
            continue
        accepted.append(vote)
        used_reference.add(vote.ref_index)
        used_search.add(vote.search_index)
    return accepted


def _cluster_votes(
    votes: list[_Vote],
    reference_peaks: list[_Peak],
    scale_bounds: tuple[float, float],
    theta_bounds: tuple[float, float],
    max_proposals: int,
) -> list[SpectralPose]:
    if len(votes) < 2:
        return []
    scales = np.asarray([vote.scale for vote in votes], dtype=np.float64)
    thetas = np.asarray([vote.theta for vote in votes], dtype=np.float64)
    weights = np.asarray([vote.weight for vote in votes], dtype=np.float64)
    active = np.ones(len(votes), dtype=bool)
    proposals: list[SpectralPose] = []
    # A 100-pixel nominal Reference contains only a few Fourier cells across
    # many useful lattice periods. Let several directions resolve one another's
    # coarse bin error; the reported robust spread still tells the caller how
    # widely it must verify the resulting seed.
    scale_band, theta_band = 0.65, 2.0

    for _ in range(max_proposals):
        active_ids = np.flatnonzero(active)
        if len(active_ids) < 2:
            break
        seed_count = min(128, len(active_ids))
        if seed_count < len(active_ids):
            chosen = np.argpartition(weights[active_ids], -seed_count)[-seed_count:]
            seed_ids = active_ids[chosen]
        else:
            seed_ids = active_ids
        densities = []
        for seed in seed_ids:
            ds = (scales[active_ids] - scales[seed]) / scale_band
            dt = (thetas[active_ids] - thetas[seed]) / theta_band
            kernel = np.exp(-0.5 * (ds * ds + dt * dt))
            densities.append(float(np.dot(weights[active_ids], kernel)))
        density_order = np.argsort(densities)[::-1]
        seed = int(seed_ids[int(density_order[0])])
        best_density = float(densities[int(density_order[0])])
        runner_density = (float(densities[int(density_order[1])])
                          if len(density_order) > 1 else 0.0)

        distance = ((scales - scales[seed]) / scale_band) ** 2 + (
            (thetas - thetas[seed]) / theta_band) ** 2
        member_ids = np.flatnonzero(active & (distance <= 1.0))
        members = _one_to_one_members(votes, member_ids)
        if len(members) < 2:
            active[seed] = False
            continue
        member_scales = np.asarray([vote.scale for vote in members])
        member_thetas = np.asarray([vote.theta for vote in members])
        member_weights = np.asarray([vote.weight for vote in members])
        scale = _weighted_median(member_scales, member_weights)
        theta = _weighted_median(member_thetas, member_weights)
        scale_mad = 1.4826 * _weighted_median(
            np.abs(member_scales - scale), member_weights)
        theta_mad = 1.4826 * _weighted_median(
            np.abs(member_thetas - theta), member_weights)
        independent_support = math.sqrt(len(members))
        scale_floor = _weighted_median(
            np.asarray([vote.scale_floor for vote in members]), member_weights
        ) / independent_support
        theta_floor = _weighted_median(
            np.asarray([vote.theta_floor_deg for vote in members]), member_weights
        ) / independent_support
        scale_uncertainty = float(max(scale_mad, scale_floor, 0.025))
        theta_uncertainty = float(max(theta_mad, theta_floor, 0.08))
        directions = np.asarray([
            reference_peaks[vote.ref_index].angle_deg for vote in members
        ])
        diversity = _directional_diversity(directions)

        support_quality = 1.0 - math.exp(-(len(members) - 1.0) / 2.5)
        coherence = math.exp(-0.5 * (
            (scale_uncertainty / 0.55) ** 2
            + (theta_uncertainty / 1.80) ** 2
        ))
        diversity_quality = 0.35 + 0.65 * min(diversity / 35.0, 1.0)
        density_margin = max(0.0, best_density - runner_density)
        margin_quality = 0.50 + 0.50 * min(
            density_margin / max(0.35 * best_density, 1e-9), 1.0)
        confidence = 0.06 + 0.39 * (
            support_quality * coherence * diversity_quality * margin_quality)
        if len(members) < 3:
            confidence = min(confidence, 0.14)

        proposals.append(SpectralPose(
            scale=float(np.clip(scale, *scale_bounds)),
            theta=float(np.clip(theta, *theta_bounds)),
            confidence=float(np.clip(confidence, 0.0, 0.45)),
            scale_uncertainty=scale_uncertainty,
            theta_uncertainty_deg=theta_uncertainty,
            support=len(members),
            directional_diversity_deg=diversity,
        ))
        # Remove this mode but retain distant harmonic hypotheses.
        active &= (((scales - scale) / (1.5 * scale_band)) ** 2
                   + ((thetas - theta) / (1.5 * theta_band)) ** 2 > 1.0)
    return proposals


def spectral_pose_proposals(
    reference_edge: np.ndarray,
    search_edge: np.ndarray,
    *,
    nominal_scale: float = 10.0,
    scale_bounds: tuple[float, float] = (8.0, 12.0),
    theta_bounds: tuple[float, float] = (-5.0, 5.0),
    max_proposals: int = 3,
    max_peaks: int = 32,
) -> list[SpectralPose]:
    """Return bounded scale/rotation seeds from matched 2-D spectral peaks.

    No translation is returned: differing fields of view make Fourier phase and
    global spectral agreement insufficient for localization. Callers must run a
    spatial verifier for every result. Empty/flat/unrelated inputs return an
    empty list rather than a nominal guess.
    """
    if not math.isfinite(nominal_scale) or nominal_scale <= 0:
        raise ValueError("nominal_scale must be positive and finite")
    if (len(scale_bounds) != 2 or not scale_bounds[0] < scale_bounds[1]
            or len(theta_bounds) != 2 or not theta_bounds[0] < theta_bounds[1]):
        raise ValueError("pose bounds must be increasing pairs")
    if max_proposals <= 0 or max_peaks < 2:
        return []
    reference = _spectral_peaks(reference_edge, max_peaks)
    search = _spectral_peaks(search_edge, max_peaks)
    votes = _pose_votes(reference, search, nominal_scale, scale_bounds, theta_bounds)
    return _cluster_votes(
        votes, reference, scale_bounds, theta_bounds, max_proposals,
    )


__all__ = ["SpectralPose", "spectral_pose_proposals"]
