"""Independent contracts for bounded directional-spectrum pose proposals."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from driftforge.edge_pose import spectral_pose_proposals


def _lattice(shape: tuple[int, int], frequencies: list[tuple[float, float]]) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    signal = np.ones(shape, dtype=np.float64)
    for index, (fx, fy) in enumerate(frequencies):
        signal += (0.32 / (index + 1)) * np.cos(2.0 * np.pi * (fx * xx + fy * yy))
    return signal.astype(np.float32)


def _transform_frequencies(
    frequencies: list[tuple[float, float]], scale: float, theta: float,
    nominal: float = 10.0,
) -> list[tuple[float, float]]:
    # A positive cv2 image rotation is counter-clockwise on screen. Frequency
    # vectors expressed in array coordinates (y down) therefore rotate by -theta.
    angle = np.deg2rad(-theta)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    ratio = scale / nominal
    return [
        (ratio * (cosine * fx - sine * fy),
         ratio * (sine * fx + cosine * fy))
        for fx, fy in frequencies
    ]


@pytest.mark.parametrize("scale,theta", [(8.7, -3.25), (11.2, 2.5)])
def test_directional_peaks_recover_bounded_pose(scale: float, theta: float) -> None:
    frequencies = [(0.052, 0.011), (0.014, 0.087), (0.096, -0.026)]
    reference = _lattice((192, 176), frequencies)
    search = _lattice((384, 352), _transform_frequencies(frequencies, scale, theta))

    proposals = spectral_pose_proposals(reference, search)

    assert proposals
    best = min(proposals, key=lambda item: abs(item.scale - scale)
               + abs(item.theta - theta))
    assert abs(best.scale - scale) <= max(0.20, 2.0 * best.scale_uncertainty)
    assert abs(best.theta - theta) <= max(0.55, 2.0 * best.theta_uncertainty_deg)
    assert best.support >= 2
    assert 0.0 <= best.confidence <= 0.45
    assert best.directional_diversity_deg > 15.0


def test_flat_inputs_abstain() -> None:
    flat = np.ones((96, 96), dtype=np.float32)
    assert spectral_pose_proposals(flat, flat) == []


def test_global_spectrum_is_only_a_capped_nonspatial_proposal() -> None:
    frequencies = [(0.043, 0.008), (0.012, 0.078), (0.091, -0.021)]
    reference = _lattice((160, 144), frequencies)
    # This field-wide lattice has no planted Reference location. It may support
    # a pose, but the helper must not manufacture translation or target presence.
    search = _lattice((400, 360), _transform_frequencies(frequencies, 9.4, 4.0))

    proposals = spectral_pose_proposals(reference, search)

    assert proposals
    assert all(item.confidence <= 0.45 for item in proposals)
    assert all(set(field.name for field in dataclasses.fields(item)).isdisjoint({"x", "y", "found"})
               for item in proposals)


def test_invalid_bounds_fail_before_spectral_work() -> None:
    image = np.ones((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="pose bounds"):
        spectral_pose_proposals(image, image, scale_bounds=(12.0, 8.0))
