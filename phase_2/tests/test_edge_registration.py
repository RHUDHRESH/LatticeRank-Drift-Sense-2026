"""Focused tests for the edge-first Phase 2 registration path."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from PIL import Image

from driftforge.edge_registration import (
    _Candidate,
    _PoseProposal,
    EdgeConfig,
    EdgeResult,
    _add_scale_boundary_proposals,
    _global_edge_pose_proposals,
    _make_template,
    _peak_to_sidelobe,
    _polarity_insensitive_orientation_agreement,
    _select_refinement_candidates,
    _spatial_candidates,
    _source_verified,
    _surface_peaks,
    solve_edges,
)


def _reference(seed: int = 7, side: int = 720) -> np.ndarray:
    """A deterministic line-art field with both local and spectral evidence."""
    rng = np.random.default_rng(seed)
    image = np.full((side, side), 24, dtype=np.uint8)
    # Repeated device rows exercise alias handling; sparse asymmetric marks
    # make the intended instance identifiable.
    for row in range(55, side - 30, 72):
        for col in range(45, side - 25, 64):
            value = int(rng.integers(125, 220))
            cv2.rectangle(image, (col, row), (col + 27, row + 13), value, -1)
            cv2.circle(image, (col + 14, row + 35), 9, min(250, value + 25), 3)
    for _ in range(18):
        p0 = tuple(int(v) for v in rng.integers(20, side - 20, size=2))
        p1 = tuple(int(v) for v in rng.integers(20, side - 20, size=2))
        cv2.line(image, p0, p1, int(rng.integers(85, 245)), int(rng.integers(2, 7)))
    cv2.putText(image, "DF", (side // 3, side // 2), cv2.FONT_HERSHEY_SIMPLEX,
                3.0, 250, 9, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (0, 0), 0.7)


def _place(
    reference: np.ndarray,
    *,
    x: float,
    y: float,
    scale: float,
    theta: float,
    side: int = 480,
    noise_seed: int = 91,
) -> np.ndarray:
    centre = ((reference.shape[1] - 1.0) / 2.0,
              (reference.shape[0] - 1.0) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, theta, 1.0 / scale)
    matrix[0, 2] += x - centre[0]
    matrix[1, 2] += y - centre[1]
    search = cv2.warpAffine(
        reference, matrix, (side, side), flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT, borderValue=24,
    ).astype(np.float32)
    noise = np.random.default_rng(noise_seed).normal(0.0, 0.8, search.shape)
    return np.clip(search + noise, 0, 255).astype(np.uint8)


def _assert_contract(result: EdgeResult, shape: tuple[int, int]) -> None:
    assert all(math.isfinite(value) for value in (
        result.x, result.y, result.theta, result.scale, result.score,
        result.edge_correlation, result.runner_up_correlation,
        result.ambiguity, result.pose_confidence,
    ))
    assert 0.0 <= result.x <= shape[1] - 1
    assert 0.0 <= result.y <= shape[0] - 1
    assert -10.0 <= result.theta <= 10.0
    assert 8.0 <= result.scale <= 12.0
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.ambiguity <= 1.0


@pytest.mark.parametrize(
    ("x", "y", "scale", "theta"),
    [
        (173.0, 296.0, 9.35, 2.4),
        (286.5, 181.5, 10.65, -3.1),
        # Exact disclosed pose boundaries catch optimizers that silently
        # exclude a closed interval endpoint or report the pre-warp centre.
        (107.0, 110.0, 8.0, -10.0),
        (371.0, 108.0, 12.0, 10.0),
    ],
)
def test_known_similarity_pose_is_localized_subpixel(
    x: float, y: float, scale: float, theta: float,
) -> None:
    reference = _reference()
    search = _place(reference, x=x, y=y, scale=scale, theta=theta)

    result = solve_edges(reference, search)

    _assert_contract(result, search.shape)
    assert result.found, result.diagnostics
    assert math.hypot(result.x - x, result.y - y) < 1.0
    assert abs(result.scale - scale) / scale < 0.05
    assert abs(result.theta - theta) < 0.8
    assert result.edge_correlation > result.runner_up_correlation
    assert result.diagnostics["dense_pose_fallback"] is False
    assert result.diagnostics["correlation_surfaces"] <= EdgeConfig().max_pose_proposals
    assert result.diagnostics["fisher"]["identifiable"]
    assert np.isfinite(result.diagnostics["fisher"]["std_x_px"])
    assert np.isfinite(result.diagnostics["fisher"]["std_y_px"])


def test_flat_and_unrelated_pairs_abstain_with_finite_values() -> None:
    flat = np.full((480, 480), 113, dtype=np.uint8)
    flat_result = solve_edges(np.full((720, 720), 113, dtype=np.uint8), flat)
    _assert_contract(flat_result, flat.shape)
    assert not flat_result.found
    assert flat_result.diagnostics["rejection_reason"] == "insufficient_edge_energy"

    reference = _reference(seed=11)
    unrelated = _reference(seed=411, side=480)
    # Make the second layout geometrically different, not merely a photometric
    # variant of the same repeated pattern.
    unrelated = cv2.warpPolar(
        unrelated, unrelated.shape[::-1], (240.0, 240.0), 310.0,
        cv2.WARP_POLAR_LINEAR,
    )
    unrelated_result = solve_edges(reference, unrelated)
    _assert_contract(unrelated_result, unrelated.shape)
    assert not unrelated_result.found, unrelated_result.diagnostics


def test_manual_paste_oracle_has_half_pixel_centre() -> None:
    """Check correlation decode without using production's affine warp."""
    reference = _reference(seed=31)
    cv2.rectangle(reference, (35, 60), (505, 235), 205, 18)
    cv2.line(reference, (110, 650), (675, 315), 245, 21)
    cv2.circle(reference, (590, 115), 63, 105, -1)
    scale = 9.0
    # PIL Lanczos plus direct integer slicing is independent of OpenCV's
    # getRotationMatrix2D/warpAffine centre and sign conventions.
    side = int(round(reference.shape[0] / scale))
    template = np.asarray(
        Image.fromarray(reference).resize((side, side), Image.Resampling.LANCZOS)
    )
    left, top = 207, 119
    template_edge = cv2.Canny(template, 30, 90).astype(np.float32) / 255.0
    search_edge = np.zeros((480, 480), dtype=np.float32)
    search_edge[top:top + side, left:left + side] = template_edge
    expected_x = left + (side - 1.0) / 2.0
    expected_y = top + (side - 1.0) / 2.0

    surface = cv2.matchTemplate(search_edge, template_edge, cv2.TM_CCOEFF_NORMED)
    value, col, row = _surface_peaks(surface, template_edge.shape, 1, 0.3)[0]
    actual_x = col + (template_edge.shape[1] - 1.0) / 2.0
    actual_y = row + (template_edge.shape[0] - 1.0) / 2.0

    assert value == pytest.approx(1.0, abs=1e-6)
    assert actual_x == expected_x
    assert actual_y == expected_y


def test_peak_to_sidelobe_standardizes_against_off_peak_surface() -> None:
    rng = np.random.default_rng(123)
    surface = rng.normal(0.05, 0.02, (81, 93)).astype(np.float32)
    surface[40, 46] = 0.45

    psr = _peak_to_sidelobe(surface, 40, 46, guard_radius=4)

    assert psr > 15.0
    assert _peak_to_sidelobe(surface, 12, 17, guard_radius=4) < 3.0


def test_peak_to_sidelobe_excludes_main_lobe_from_null_variance() -> None:
    surface = np.zeros((61, 61), dtype=np.float32)
    surface[27:34, 27:34] = 0.8
    surface[30, 30] = 1.0

    assert _peak_to_sidelobe(surface, 30, 30, guard_radius=4) == 20.0


def test_coarse_surface_recovers_broad_peak_omitted_by_fine_top_k() -> None:
    """Low-frequency proposals survive dense high-frequency distractors."""
    rng = np.random.default_rng(814)
    template = np.zeros((48, 48), dtype=np.float32)
    cv2.rectangle(template, (6, 8), (40, 37), 0.35, -1)
    template += rng.normal(0.0, 0.16, template.shape).astype(np.float32)
    search = rng.normal(0.0, 0.18, (180, 210)).astype(np.float32)
    row, col = 93, 121
    # Preserve the broad boundary but replace its fine texture, reproducing a
    # target whose exact Scharr response ranks below narrow clutter aliases.
    fine_template = _make_template(template, 1.0, 0.0, EdgeConfig())
    broad = cv2.GaussianBlur(fine_template, (0, 0), 2.0)
    search[row:row + 48, col:col + 48] += broad
    # One sharp alias exhausts the deliberately one-peak fine budget.
    search[18:66, 22:70] += fine_template
    proposal = _PoseProposal(1.0, 0.0, 0.3, "test")
    config = EdgeConfig(peaks_per_pose=1, coarse_peaks_per_pose=4)

    fine_only, _ = _spatial_candidates(
        template, search, [proposal],
        EdgeConfig(peaks_per_pose=1, coarse_peaks_per_pose=0),
    )
    candidates, surfaces = _spatial_candidates(template, search, [proposal], config)

    assert surfaces == 1
    assert all(math.hypot(item.x - (col + 23.5), item.y - (row + 23.5)) >= 5.0
               for item in fine_only)
    assert any(math.hypot(item.x - (col + 23.5), item.y - (row + 23.5)) < 5.0
               for item in candidates)


def test_refinement_admission_reserves_each_pose_before_more_aliases() -> None:
    candidates = [
        _Candidate(
            x=float(20 + 12 * index), y=20.0, scale=9.0, theta=-2.0,
            correlation=0.95 - 0.01 * index, pose_confidence=0.8,
            source="descriptor_ransac",
        )
        for index in range(8)
    ]
    candidates.extend([
        _Candidate(
            x=150.0, y=90.0, scale=10.2, theta=1.5,
            correlation=0.70, pose_confidence=0.3,
            source="directional_spectrum",
        ),
        _Candidate(
            x=260.0, y=180.0, scale=10.0, theta=0.0,
            correlation=0.65, pose_confidence=0.2,
            source="orientation_nominal",
        ),
    ])
    config = EdgeConfig(max_refine_candidates=4)

    selected = _select_refinement_candidates(candidates, config)

    assert len(selected) == 4
    assert {item.source for item in selected[:3]} == {
        "descriptor_ransac", "directional_spectrum", "orientation_nominal",
    }
    assert selected[3].source == "descriptor_ransac"


def test_near_endpoint_pose_gets_fresh_boundary_template_proposal() -> None:
    proposal = _PoseProposal(
        scale=8.31, theta=-1.1, confidence=0.20,
        source="directional_spectrum",
    )

    expanded = _add_scale_boundary_proposals([proposal], EdgeConfig())

    assert len(expanded) == 2
    assert expanded[1].scale == 8.0
    assert expanded[1].theta == proposal.theta
    assert expanded[1].source == "directional_spectrum_scale_boundary"
    assert expanded[1].confidence == proposal.confidence


def test_template_pyramid_area_integrates_subpixel_checkerboard() -> None:
    """Independent anti-alias oracle; warpAffine AREA does not integrate."""
    yy, xx = np.indices((800, 800))
    reference = (((xx + yy) % 2) * 255).astype(np.uint8)
    config = EdgeConfig()
    base = cv2.resize(reference.astype(np.float32) / 255.0, (100, 100),
                      interpolation=cv2.INTER_AREA)
    template = _make_template(
        base, scale=10.0, theta=0.0, config=config,
        source_scale=config.scale_min,
    )
    oracle_pixels = cv2.resize(
        reference.astype(np.float32) / 255.0, (80, 80),
        interpolation=cv2.INTER_AREA,
    )

    assert float(base.std()) < 1e-6
    assert float(oracle_pixels.std()) < 1e-6
    assert float(template.max()) < 1e-4


def test_result_mapping_contains_public_contract_fields() -> None:
    result = solve_edges(np.zeros((128, 128), dtype=np.uint8),
                         np.zeros((128, 128), dtype=np.uint8))
    assert set(result.as_dict()) == {
        "x", "y", "theta", "scale", "found", "score",
        "edge_correlation", "runner_up_correlation", "ambiguity",
        "pose_confidence", "diagnostics",
    }


@pytest.mark.parametrize(
    ("correlation", "gap", "expected"),
    [
        (0.74, 0.01, False),  # strong but repeated-layout alias
        (0.74, 0.06, True),   # strong agreement with modest isolation
        (0.52, 0.12, True),   # moderate agreement needs spatial isolation
        (0.65, 0.04, False),  # moderate repeated-layout alias
        (0.45, 0.20, False),  # isolated but weak edge coincidence
    ],
)
def test_spectral_presence_requires_strong_or_isolated_edge_evidence(
    correlation: float, gap: float, expected: bool,
) -> None:
    candidate = _Candidate(
        x=10.0, y=10.0, scale=10.0, theta=0.0,
        correlation=correlation, pose_confidence=0.16,
        source="directional_spectrum",
    )
    assert _source_verified(candidate, correlation, gap, EdgeConfig()) is expected


def test_degraded_spectral_match_uses_independent_orientation_evidence() -> None:
    candidate = _Candidate(
        x=10.0, y=10.0, scale=10.0, theta=0.0,
        correlation=0.38, pose_confidence=0.16,
        source="directional_spectrum", orientation_agreement=0.20,
    )
    assert _source_verified(candidate, 0.38, 0.06, EdgeConfig())


def test_synthetic_scale_boundary_requires_extra_isolation() -> None:
    candidate = _Candidate(
        x=10.0, y=10.0, scale=8.0, theta=0.0,
        correlation=0.68, pose_confidence=0.16,
        source="directional_spectrum_scale_boundary",
        orientation_agreement=0.50,
    )
    assert not _source_verified(candidate, 0.68, 0.09, EdgeConfig())
    assert _source_verified(candidate, 0.68, 0.13, EdgeConfig())


def test_orientation_agreement_is_polarity_insensitive_and_discriminative() -> None:
    image = np.zeros((96, 96), dtype=np.float32)
    cv2.rectangle(image, (18, 25), (74, 68), 1.0, -1)
    cv2.line(image, (20, 80), (78, 12), 0.6, 4)
    inverted = 1.0 - image
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    same = _polarity_insensitive_orientation_agreement(image, image)
    opposite_polarity = _polarity_insensitive_orientation_agreement(image, inverted)
    different_orientation = _polarity_insensitive_orientation_agreement(image, rotated)

    assert same > 0.95
    assert opposite_polarity > 0.95
    assert different_orientation < 0.35


def test_global_pose_rescue_requires_independent_fit_and_isolation() -> None:
    candidate = _Candidate(
        x=10.0, y=10.0, scale=10.0, theta=0.0,
        correlation=0.62, pose_confidence=0.12,
        source="coarse_global_edge", intensity_correlation=-0.70,
    )
    assert _source_verified(candidate, 0.62, 0.14, EdgeConfig())
    assert not _source_verified(candidate, 0.62, 0.08, EdgeConfig())
    candidate.intensity_correlation = 0.40
    assert not _source_verified(candidate, 0.62, 0.20, EdgeConfig())


def test_global_pose_bank_can_be_disabled_for_calibrated_first_pass() -> None:
    proposals, diagnostics = _global_edge_pose_proposals(
        np.zeros((128, 128), np.float32),
        np.zeros((128, 128), np.float32),
        EdgeConfig(global_pose_shortlist=0),
    )
    assert proposals == []
    assert diagnostics["coarse_global_pose_surfaces"] == 0
