"""Fast edge-first similarity registration for Phase 2.

The public :func:`solve_edges` path has three deliberately separate stages:

1. Gaussian smoothing followed by Scharr derivatives produces edge features.
2. Scale and rotation are proposed from edge evidence.  Local SIFT matches are
   fit with a similarity RANSAC, while orientation histograms and radial power
   spectra provide a bounded fallback for low-feature periodic layouts.
3. Only the proposed poses are spatially correlated.  Distinct aliases are
   retained, then the strongest few are refined continuously in
   ``(x, y, theta, scale)`` using sub-pixel patch sampling.

The calibrated local/spectral path runs first.  Honest abstentions may invoke
a bounded quarter-resolution pose bank; its shortlist must still pass the same
full-resolution edge refinement and independent verification.
All scales use the project convention: ``scale`` is the 8--12 down-scaling
factor from Reference pixels to Search pixels, and ``theta`` is the angle
accepted by :func:`driftforge.pose.build_template`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import os
from typing import Any

import numpy as np

try:  # Kept lazy-friendly for source inspection and helpful wheel failures.
    import cv2
except ImportError:  # pragma: no cover - the production dependency supplies it
    cv2 = None


@dataclass(frozen=True)
class EdgeConfig:
    """Bounded controls for the edge registration path."""

    scale_min: float = 8.0
    scale_max: float = 12.0
    # Phase 2 discloses rotations through ten degrees. Pose proposal remains
    # bounded to that interval; this is not a correlation sweep.
    rotation_limit_deg: float = 10.0
    nominal_scale: float = 10.0
    gaussian_sigma: float = 1.0
    # Search may contain a small, polarity-inverted target whose pixels occupy
    # barely one percentile; gradient energy remains the stronger flatness gate.
    min_contrast: float = 0.008
    min_gradient_rms: float = 0.0025
    max_keypoints: int = 1400
    # Periodic edge descriptors have close second neighbours; retain recall
    # here and let bounded-pose RANSAC plus edge correlation reject them.
    descriptor_ratio: float = 0.96
    # A similarity has four parameters; three non-collinear correspondences
    # overdetermine it while still covering official low-feature crops where
    # the nominal 100 px Reference supplies only three stable SIFT matches.
    min_descriptor_matches: int = 3
    min_ransac_inliers: int = 3
    ransac_threshold_px: float = 2.5
    max_local_proposals: int = 1
    descriptor_pose_confidence: float = 0.12
    spectral_pose_confidence: float = 0.10
    descriptor_scale_slack: float = 0.35
    descriptor_angle_slack_deg: float = 8.0
    spectral_scale_samples: int = 33
    spectral_angle_samples: int = 41
    spectral_proposals_per_axis: int = 2
    max_pose_proposals: int = 5
    # Keep enough spatially separated aliases for periodic layouts.  The
    # downstream refinement budget remains fixed, so this increases proposal
    # recall without increasing the number of continuous optimizations.
    peaks_per_pose: int = 8
    # A second, low-frequency surface proposes broad spatial modes that can be
    # weak on the fine Scharr map under blur, charging and line dropout.  Its
    # locations are always rescored on the original full-resolution surface.
    coarse_peaks_per_pose: int = 4
    coarse_downsample: int = 2
    coarse_blur_fraction: float = 0.035
    # A tiny global bank is evaluated only at quarter resolution.  It supplies
    # starting poses when periodic spectra select the wrong harmonic; every
    # shortlisted pose is still verified on the full-resolution edge map.
    global_pose_downsample: int = 4
    global_pose_shortlist: int = 8
    boundary_rescue_shortlist: int = 12
    intensity_rank_weight: float = 0.0
    boundary_rescue_only: bool = False
    # At least one site from every retained pose can reach continuous
    # refinement.  Otherwise several high harmonic aliases can fill a global
    # top-3 and prevent the nominal seed from ever correcting its scale.
    max_refine_candidates: int = 10
    spatial_nms_fraction: float = 0.32
    refine_xy_radius_px: float = 4.0
    # The spectral estimate is intentionally only a proposal: a 100 px
    # Reference crop may contain too few periods to identify the right harmonic.
    # Continuous verification is therefore allowed to traverse the full
    # disclosed band from the nominal 10x proposal.
    refine_scale_radius: float = 2.0
    # Global orientation evidence is only a proposal because the target covers
    # a small part of Search. Local patch evidence owns the final angle across
    # the disclosed interval.
    refine_theta_radius_deg: float = 10.0
    optimizer_maxiter: int = 20
    optimizer_maxfev: int = 100
    min_edge_correlation: float = 0.18
    min_pose_confidence: float = 0.08
    min_found_score: float = 0.30
    # ``None`` respects OpenCV's current process setting.  A caller can set an
    # explicit local budget here, or use DRIFTFORGE_NUM_THREADS without code
    # changes on a constrained evaluation host.
    opencv_threads: int | None = None


@dataclass
class EdgeResult:
    """Registration answer plus edge-specific confidence evidence."""

    x: float
    y: float
    theta: float
    scale: float
    found: bool
    score: float
    edge_correlation: float
    runner_up_correlation: float
    ambiguity: float
    pose_confidence: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PoseProposal:
    scale: float
    theta: float
    confidence: float
    source: str
    x: float | None = None
    y: float | None = None
    matches: int = 0
    inliers: int = 0
    scale_uncertainty: float = 0.0
    theta_uncertainty_deg: float = 0.0
    support: int = 0
    directional_diversity_deg: float = 0.0
    photometric_margin: float = 0.0


@dataclass
class _Candidate:
    x: float
    y: float
    scale: float
    theta: float
    correlation: float
    pose_confidence: float
    source: str
    matches: int = 0
    inliers: int = 0
    intensity_correlation: float = -1.0
    orientation_agreement: float = 0.0
    peak_to_sidelobe: float = 0.0
    photometric_margin: float = 0.0


def _as_gray(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image)
    if data.ndim == 3:
        if data.shape[2] >= 3 and cv2 is not None:
            data = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            data = data.mean(axis=-1)
    if data.ndim != 2:
        raise ValueError("reference and search must be 2-D grayscale or RGB images")
    out = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        out /= float(np.iinfo(data.dtype).max)
    elif out.size and float(np.nanmax(out)) > 1.5:
        out /= 255.0
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def _contrast_and_gradient(gray: np.ndarray, config: EdgeConfig) -> tuple[float, float]:
    sample = gray[::4, ::4] if gray.size >= 4096 else gray
    lo, hi = np.percentile(sample, (1.0, 99.0))
    contrast = float(hi - lo)
    smooth = cv2.GaussianBlur(gray, (0, 0), config.gaussian_sigma,
                              borderType=cv2.BORDER_REFLECT)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    gradient_rms = float(np.sqrt(np.mean(gx * gx + gy * gy)) / 16.0)
    return contrast, gradient_rms


def _edge_features(gray: np.ndarray, config: EdgeConfig) -> np.ndarray:
    """Return normalized Gaussian/Scharr edge magnitude as float32."""
    smooth = cv2.GaussianBlur(gray, (0, 0), config.gaussian_sigma,
                              borderType=cv2.BORDER_REFLECT)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy) / 16.0
    sample = magnitude[::3, ::3] if magnitude.size >= 4096 else magnitude
    floor, ceiling = np.percentile(sample, (35.0, 99.5))
    edge = np.maximum(magnitude - float(floor), 0.0)
    edge /= max(float(ceiling - floor), 1e-6)
    return np.clip(edge, 0.0, 1.0).astype(np.float32)


def _nominal_reference(gray: np.ndarray, config: EdgeConfig) -> np.ndarray:
    """Bring Reference near Search pixel scale before descriptor matching."""
    height = max(16, int(round(gray.shape[0] / config.nominal_scale)))
    width = max(16, int(round(gray.shape[1] / config.nominal_scale)))
    # INTER_AREA integrates the roughly 10x10 source footprint and is much less
    # sensitive to high-resolution SEM noise than asking a SIFT octave pyramid
    # to bridge the native 10:1 acquisition gap.
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def _wrap_angle(degrees: float) -> float:
    return float((degrees + 180.0) % 360.0 - 180.0)


def _pose_from_affine(matrix: np.ndarray, config: EdgeConfig) -> tuple[float, float] | None:
    a, c = float(matrix[0, 0]), float(matrix[0, 1])
    shrink_from_nominal = float(np.hypot(a, c))
    if not np.isfinite(shrink_from_nominal) or shrink_from_nominal <= 1e-6:
        return None
    scale = config.nominal_scale / shrink_from_nominal
    # cv2's image-plane positive angle has matrix row [cos, sin].
    theta = _wrap_angle(np.degrees(np.arctan2(c, a)))
    if not (config.scale_min <= scale <= config.scale_max):
        return None
    if abs(theta) > config.rotation_limit_deg:
        return None
    return float(scale), float(theta)


def _descriptor_proposals(
    nominal_gray: np.ndarray,
    search_gray: np.ndarray,
    config: EdgeConfig,
) -> tuple[list[_PoseProposal], dict[str, Any]]:
    """Fit a few similarity transforms to local edge-descriptor matches."""
    nominal_edge = _edge_features(nominal_gray, config)
    search_edge = _edge_features(search_gray, config)
    sift = cv2.SIFT_create(
        nfeatures=config.max_keypoints,
        nOctaveLayers=3,
        contrastThreshold=0.008,
        edgeThreshold=12,
        sigma=1.2,
    )
    kp_ref, desc_ref = sift.detectAndCompute(
        np.asarray(np.rint(nominal_edge * 255.0), dtype=np.uint8), None)
    kp_search, desc_search = sift.detectAndCompute(
        np.asarray(np.rint(search_edge * 255.0), dtype=np.uint8), None)
    diag = {
        "reference_keypoints": len(kp_ref),
        "search_keypoints": len(kp_search),
        "descriptor_matches": 0,
    }
    if desc_ref is None or desc_search is None or len(desc_ref) < 2 or len(desc_search) < 2:
        return [], diag

    raw = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(desc_ref, desc_search, k=2)
    accepted = []
    relative_lo = config.nominal_scale / config.scale_max
    relative_hi = config.nominal_scale / config.scale_min
    for pair in raw:
        if len(pair) != 2 or pair[0].distance >= config.descriptor_ratio * pair[1].distance:
            continue
        match = pair[0]
        left, right = kp_ref[match.queryIdx], kp_search[match.trainIdx]
        size_ratio = right.size / max(left.size, 1e-6)
        if not (relative_lo * (1.0 - config.descriptor_scale_slack)
                <= size_ratio <= relative_hi * (1.0 + config.descriptor_scale_slack)):
            continue
        if abs(_wrap_angle(right.angle - left.angle)) > (
            config.rotation_limit_deg + config.descriptor_angle_slack_deg
        ):
            continue
        accepted.append(match)
    # One search keypoint cannot support several unrelated transformations.
    best_by_train: dict[int, Any] = {}
    for match in accepted:
        old = best_by_train.get(match.trainIdx)
        if old is None or match.distance < old.distance:
            best_by_train[match.trainIdx] = match
    accepted = sorted(best_by_train.values(), key=lambda item: item.distance)
    diag["descriptor_matches"] = len(accepted)
    if len(accepted) < config.min_descriptor_matches:
        return [], diag

    src_all = np.float32([kp_ref[m.queryIdx].pt for m in accepted])
    dst_all = np.float32([kp_search[m.trainIdx].pt for m in accepted])
    remaining = np.arange(len(accepted), dtype=np.int32)
    proposals: list[_PoseProposal] = []
    nominal_center = np.array(
        [(nominal_gray.shape[1] - 1.0) / 2.0,
         (nominal_gray.shape[0] - 1.0) / 2.0, 1.0], dtype=np.float64)

    while (len(remaining) >= config.min_descriptor_matches
           and len(proposals) < config.max_local_proposals):
        matrix, mask = cv2.estimateAffinePartial2D(
            src_all[remaining], dst_all[remaining], method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_threshold_px,
            maxIters=1200, confidence=0.995, refineIters=10,
        )
        if matrix is None or mask is None:
            break
        local_inliers = np.flatnonzero(mask.ravel().astype(bool))
        if len(local_inliers) < config.min_ransac_inliers:
            break
        pose = _pose_from_affine(matrix, config)
        if pose is not None:
            mapped = matrix @ nominal_center
            inlier_ratio = len(local_inliers) / max(len(remaining), 1)
            support = 1.0 - np.exp(-len(local_inliers) / 7.0)
            confidence = float(np.clip(inlier_ratio * support, 0.0, 1.0))
            proposals.append(_PoseProposal(
                scale=pose[0], theta=pose[1], confidence=confidence,
                source="descriptor_ransac", x=float(mapped[0]), y=float(mapped[1]),
                matches=len(accepted), inliers=len(local_inliers),
            ))
        # Peel off this consensus so a repeated-layout alias can form another.
        keep = np.ones(len(remaining), dtype=bool)
        keep[local_inliers] = False
        remaining = remaining[keep]
    diag["ransac_proposals"] = len(proposals)
    return proposals, diag


def _orientation_histogram(gray: np.ndarray, bins: int = 180) -> np.ndarray:
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    slots = np.minimum((angle * bins / 180.0).astype(np.int32), bins - 1)
    hist = np.bincount(slots.ravel(), weights=magnitude.ravel(), minlength=bins).astype(np.float64)
    # Circular smoothing suppresses one-bin quantisation without erasing a
    # five-degree shift.
    hist = (np.roll(hist, -2) + 4 * np.roll(hist, -1) + 6 * hist
            + 4 * np.roll(hist, 1) + np.roll(hist, 2)) / 16.0
    hist -= hist.mean()
    hist /= np.linalg.norm(hist) + 1e-12
    return hist


def _radial_spectrum(edge: np.ndarray, bins: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Compact rotation-invariant spectrum in cycles per pixel."""
    data = edge.astype(np.float32) - float(edge.mean())
    # Hann taper keeps the unequal field boundaries out of the descriptor.
    data *= np.hanning(data.shape[0]).astype(np.float32)[:, None]
    data *= np.hanning(data.shape[1]).astype(np.float32)[None, :]
    power = np.log1p(np.abs(np.fft.rfft2(data)) ** 2)
    fy = np.fft.fftfreq(data.shape[0])[:, None]
    fx = np.fft.rfftfreq(data.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    lo, hi = 0.018, 0.42
    slots = np.floor((radius - lo) * bins / (hi - lo)).astype(np.int32)
    valid = (slots >= 0) & (slots < bins)
    sums = np.bincount(slots[valid], weights=power[valid], minlength=bins)
    counts = np.bincount(slots[valid], minlength=bins)
    profile = sums / np.maximum(counts, 1)
    profile = np.convolve(profile, np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
    profile -= np.mean(profile)
    profile /= np.linalg.norm(profile) + 1e-12
    frequencies = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins
    return frequencies, profile


def _top_separated(grid: np.ndarray, scores: np.ndarray, count: int,
                   separation: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []
    for index in order:
        if all(abs(float(grid[index] - grid[old])) >= separation for old in chosen):
            chosen.append(int(index))
            if len(chosen) >= count:
                break
    return chosen


def _spectral_proposals(
    nominal_gray: np.ndarray,
    search_gray: np.ndarray,
    config: EdgeConfig,
) -> tuple[list[_PoseProposal], dict[str, Any]]:
    """Propose pose from bounded directional edge-spectrum evidence.

    This is intentionally proposal evidence rather than a whole-image Fourier
    registration claim: Reference contains only a few periods and Search has a
    different field of view.  Its confidence is therefore capped, and spatial
    edge correlation must verify every proposal.
    """
    href = _orientation_histogram(nominal_gray)
    hsearch = _orientation_histogram(search_gray)
    angles = np.linspace(-config.rotation_limit_deg, config.rotation_limit_deg,
                         config.spectral_angle_samples)
    orientation_scores = np.array([
        float(np.dot(href, np.roll(hsearch, int(round(theta)))))
        for theta in angles
    ])
    angle_ids = _top_separated(angles, orientation_scores,
                               config.spectral_proposals_per_axis, 1.0)

    ref_edge = _edge_features(nominal_gray, config)
    search_edge = _edge_features(search_gray, config)
    from .edge_pose import spectral_pose_proposals

    directional = spectral_pose_proposals(
        ref_edge,
        search_edge,
        nominal_scale=config.nominal_scale,
        scale_bounds=(config.scale_min, config.scale_max),
        theta_bounds=(-config.rotation_limit_deg, config.rotation_limit_deg),
        # Reserve one of the configured pose slots for the nominal-scale
        # orientation seed. The remaining slots retain distinct directional
        # modes, including a correct low-confidence scale beside a harmonic.
        max_proposals=max(1, config.max_pose_proposals - 1),
        max_peaks=32,
    )

    # Orientation contrast remains useful for the single nominal-scale safety
    # seed, but no radial collapse is used for scale: collapsing direction was
    # what made a harmonic look like a well-supported pose on periodic fields.
    orientation_strength = float(np.clip(
        (orientation_scores.max() - np.median(orientation_scores)) / 0.20, 0.0, 1.0))
    proposals = [
        _PoseProposal(
            scale=float(item.scale), theta=float(item.theta),
            confidence=float(item.confidence), source="directional_spectrum",
            scale_uncertainty=float(item.scale_uncertainty),
            theta_uncertainty_deg=float(item.theta_uncertainty_deg),
            support=int(item.support),
            directional_diversity_deg=float(item.directional_diversity_deg),
        )
        for item in directional
    ]
    # Keep one nominal-scale hypothesis whenever orientation has a measurable
    # preference.  This is a single continuous-refinement seed, not a scale
    # sweep.  It matters on short periodic crops whose radial spectrum locks to
    # a harmonic: spatial correlation at 10x still finds the correct site, and
    # the bounded four-parameter optimizer then recovers the continuous scale.
    if angle_ids:
        nominal_confidence = float(np.clip(
            0.14 + 0.20 * orientation_strength, 0.16, 0.34))
        proposals.append(_PoseProposal(
            scale=float(config.nominal_scale), theta=float(angles[angle_ids[0]]),
            confidence=nominal_confidence, source="orientation_nominal",
        ))
    diag = {
        "spectral_scale": (
            float(directional[0].scale) if directional else config.nominal_scale
        ),
        "spectral_theta": float(angles[int(np.argmax(orientation_scores))]),
        "orientation_strength": orientation_strength,
        "spectral_confidence": (
            float(directional[0].confidence) if directional else 0.0
        ),
        "directional_spectral_proposals": [asdict(item) for item in directional],
    }
    return proposals, diag


def _deduplicate_proposals(proposals: list[_PoseProposal], config: EdgeConfig) -> list[_PoseProposal]:
    # A geometrically consistent local transform is more specific evidence
    # than a global periodic spectrum.  Reserve room for it; correlation still
    # has to verify the proposal before ``found`` can become true.
    ordered = sorted(
        proposals,
        key=lambda item: (item.source == "descriptor_ransac", item.confidence),
        reverse=True,
    )
    kept: list[_PoseProposal] = []
    for proposal in ordered:
        duplicate = any(
            abs(proposal.scale - old.scale) < 0.20
            and abs(proposal.theta - old.theta) < 0.45
            and (proposal.x is None or old.x is None
                 or np.hypot(proposal.x - old.x, proposal.y - old.y) < 5.0)
            for old in kept
        )
        if not duplicate:
            kept.append(proposal)
        if len(kept) >= config.max_pose_proposals:
            break
    return kept


def _add_scale_boundary_proposals(
    proposals: list[_PoseProposal], config: EdgeConfig,
) -> list[_PoseProposal]:
    """Add exact disclosed scale endpoints near an uncertain proposal.

    Template support changes with scale.  A continuous optimizer whose patch
    shape was created from an approximate seed cannot reliably move onto an
    8x or 12x endpoint.  Rendering a fresh endpoint template is cheap and
    gives spatial correlation the correct support before local refinement.
    """
    expanded = list(proposals)
    for proposal in proposals:
        for boundary in (config.scale_min, config.scale_max):
            if not (0.05 < abs(proposal.scale - boundary) <= 0.50):
                continue
            candidate = replace(
                proposal,
                scale=float(boundary),
                confidence=float(proposal.confidence),
                source=f"{proposal.source}_scale_boundary",
            )
            if any(abs(candidate.scale - old.scale) < 0.05
                   and abs(candidate.theta - old.theta) < 0.25
                   for old in expanded):
                continue
            expanded.append(candidate)
    return expanded


def _global_edge_pose_proposals(
    reference_gray: np.ndarray,
    search_gray: np.ndarray,
    config: EdgeConfig,
) -> tuple[list[_PoseProposal], dict[str, Any]]:
    """Shortlist global similarity modes on a quarter-resolution edge map."""
    if config.global_pose_shortlist <= 0:
        return [], {"coarse_global_pose_peaks": [],
                    "coarse_global_pose_surfaces": 0}
    factor = max(2, int(config.global_pose_downsample))
    search_edge = _edge_features(search_gray, config)
    coarse_search = cv2.resize(
        search_edge, None, fx=1.0/factor, fy=1.0/factor,
        interpolation=cv2.INTER_AREA)
    # Anti-alias once before rendering the small pose bank.  Five scales and
    # five angles cover the disclosed industrial extension with only 25 tiny
    # surfaces (roughly 128x128 for a 512px Search).
    source = cv2.GaussianBlur(
        reference_gray, (0, 0), 0.5*config.scale_min,
        borderType=cv2.BORDER_REFLECT)
    scales = np.linspace(config.scale_min, config.scale_max, 5)
    angles = np.linspace(-config.rotation_limit_deg,
                         config.rotation_limit_deg, 5)
    ranked: list[tuple[float, _PoseProposal]] = []
    for scale in scales:
        for theta in angles:
            full_template = _make_template(
                source, float(scale), float(theta), config)
            template = cv2.resize(
                full_template, None, fx=1.0/factor, fy=1.0/factor,
                interpolation=cv2.INTER_AREA)
            if (min(template.shape) < 6
                    or template.shape[0] >= coarse_search.shape[0]
                    or template.shape[1] >= coarse_search.shape[1]
                    or float(template.std()) < 1e-5):
                continue
            surface = cv2.matchTemplate(
                coarse_search, template, cv2.TM_CCOEFF_NORMED)
            _lo, peak, _lloc, location = cv2.minMaxLoc(surface)
            x = location[0]*factor + (full_template.shape[1]-1.0)/2.0
            y = location[1]*factor + (full_template.shape[0]-1.0)/2.0
            ranked.append((float(peak), _PoseProposal(
                scale=float(scale), theta=float(theta), confidence=0.12,
                source="coarse_global_edge", x=float(x), y=float(y),
                scale_uncertainty=0.5,
                theta_uncertainty_deg=config.rotation_limit_deg/4.0,
                support=int(template.size),
            )))
    ranked.sort(key=lambda item: item[0], reverse=True)
    kept: list[_PoseProposal] = []
    for _score, proposal in ranked:
        if all(abs(proposal.scale-old.scale) >= 0.70
               or abs(proposal.theta-old.theta) >= 2.0
               or np.hypot(proposal.x-old.x, proposal.y-old.y) >= 8.0
               for old in kept):
            kept.append(proposal)
        if len(kept) >= config.global_pose_shortlist:
            break
    return kept, {
        "coarse_global_pose_peaks": [
            {"scale": item.scale, "theta": item.theta,
             "x": item.x, "y": item.y}
            for item in kept
        ],
        "coarse_global_pose_surfaces": len(ranked),
    }


def _boundary_rescue_proposals(
    reference_gray: np.ndarray,
    search_gray: np.ndarray,
    config: EdgeConfig,
) -> list[_PoseProposal]:
    if config.boundary_rescue_shortlist <= 0:
        return []
    from .boundary_proposals import boundary_pose_proposals

    raw = boundary_pose_proposals(
        reference_gray, search_gray,
        shortlist=config.boundary_rescue_shortlist)
    # A reflected finite-FOV hypothesis is deliberately only an alternate
    # boundary model.  Require the physically usual constant-padding model
    # to corroborate its location before it can trigger a rescue.  This
    # prevents a reflected border from manufacturing a strong periodic SEM
    # alias while retaining cases where both boundary models identify the
    # same site (the pose can still come from the better reflected model).
    constant_sites = [
        item for item in raw
        if (item.boundary == "constant" and item.score >= 0.22
            and item.margin >= 0.09)
    ]

    def boundary_supported(item: Any) -> bool:
        return (item.boundary == "constant"
                or any(np.hypot(item.x-other.x, item.y-other.y) <= 4.0
                       for other in constant_sites))

    return [
        _PoseProposal(
            scale=item.scale, theta=item.theta, confidence=item.score,
            source="boundary_intensity", x=item.x, y=item.y,
            scale_uncertainty=0.7,
            theta_uncertainty_deg=2.5,
            support=1, photometric_margin=item.margin,
        )
        for item in raw
        if (item.score >= 0.22 and item.margin >= 0.09
            and boundary_supported(item))
    ]


def _warp_template_pixels(
    reference_gray: np.ndarray,
    scale: float,
    theta: float,
    output_shape: tuple[int, int] | None = None,
    source_scale: float = 1.0,
) -> np.ndarray:
    """Warp a pre-filtered Reference into Search sampling."""
    if output_shape is None:
        output_shape = (
            max(8, int(round(reference_gray.shape[0] * source_scale / scale))),
            max(8, int(round(reference_gray.shape[1] * source_scale / scale))),
        )
    height, width = output_shape
    # warpAffine does not provide true area integration while shrinking, even
    # when INTER_AREA is requested. Decimate first with resize so periodic
    # high-resolution features cannot alias, then rotate at Search resolution.
    source_center = ((reference_gray.shape[1] - 1.0) / 2.0,
                     (reference_gray.shape[0] - 1.0) / 2.0)
    target_center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    relative_scale = source_scale / scale
    matrix = cv2.getRotationMatrix2D(source_center, theta, relative_scale)
    matrix[0, 2] += target_center[0] - source_center[0]
    matrix[1, 2] += target_center[1] - source_center[1]
    warped = cv2.warpAffine(
        reference_gray, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def _make_template(
    reference_gray: np.ndarray,
    scale: float,
    theta: float,
    config: EdgeConfig,
    output_shape: tuple[int, int] | None = None,
    source_scale: float = 1.0,
) -> np.ndarray:
    """Warp Reference into a Search-pixel edge template at continuous pose."""
    warped = _warp_template_pixels(
        reference_gray, scale, theta, output_shape, source_scale)
    return _edge_features(warped, config)


def _surface_peaks(surface: np.ndarray, template_shape: tuple[int, int],
                   count: int, nms_fraction: float) -> list[tuple[float, int, int]]:
    work = np.nan_to_num(surface, nan=-1.0, posinf=-1.0, neginf=-1.0).copy()
    radius = max(3, int(round(min(template_shape) * nms_fraction)))
    peaks: list[tuple[float, int, int]] = []
    for _ in range(count):
        _minimum, maximum, _minloc, location = cv2.minMaxLoc(work)
        if not np.isfinite(maximum) or maximum <= -1.0:
            break
        col, row = location
        peaks.append((float(maximum), int(col), int(row)))
        y0, y1 = max(0, row - radius), min(work.shape[0], row + radius + 1)
        x0, x1 = max(0, col - radius), min(work.shape[1], col + radius + 1)
        work[y0:y1, x0:x1] = -1.0
    return peaks


def _peak_to_sidelobe(
    surface: np.ndarray,
    row: int,
    col: int,
    guard_radius: int,
    moments: tuple[float, float] | None = None,
) -> float:
    """Standardize a correlation peak against its off-peak response map.

    The response surface is already available, so this adds only two reductions
    and a small guard-band sum.  Excluding the peak neighbourhood prevents the
    target's main lobe from inflating its own null variance.
    """
    values = (np.nan_to_num(surface, nan=0.0, posinf=0.0, neginf=0.0)
              if moments is None else surface)
    count = int(values.size)
    if moments is None:
        total = float(values.sum())
        total_sq = float(np.square(values).sum())
    else:
        total, total_sq = moments
    y0, y1 = max(0, row - guard_radius), min(values.shape[0], row + guard_radius + 1)
    x0, x1 = max(0, col - guard_radius), min(values.shape[1], col + guard_radius + 1)
    guard = values[y0:y1, x0:x1]
    count -= int(guard.size)
    if count < 2:
        return 0.0
    total -= float(guard.sum())
    total_sq -= float(np.square(guard).sum())
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    if variance <= 1e-12:
        return 0.0
    return float(np.clip((float(surface[row, col]) - mean) / np.sqrt(variance), 0.0, 20.0))


def _spatial_candidates(
    template_source: np.ndarray,
    search_edge: np.ndarray,
    proposals: list[_PoseProposal],
    config: EdgeConfig,
) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    surfaces = 0
    for proposal in proposals:
        template = _make_template(
            template_source, proposal.scale, proposal.theta, config,
            source_scale=1.0,
        )
        if (template.shape[0] >= search_edge.shape[0]
                or template.shape[1] >= search_edge.shape[1]
                or float(template.std()) < 1e-5):
            continue
        surface = cv2.matchTemplate(search_edge, template, cv2.TM_CCOEFF_NORMED)
        surfaces += 1
        half_x = (template.shape[1] - 1.0) / 2.0
        half_y = (template.shape[0] - 1.0) / 2.0
        peaks = ([] if proposal.source == "boundary_intensity" else
                 _surface_peaks(surface, template.shape, config.peaks_per_pose,
                                config.spatial_nms_fraction))
        # Build an independent low-frequency proposal surface.  Smoothing at
        # a scale tied to template support preserves device boundaries while
        # suppressing fine periodic lines that otherwise occupy every retained
        # peak.  Downsampling makes this stage cheap; proposed locations are
        # converted back to the fine grid and scored by ``surface`` so coarse
        # evidence can improve recall but cannot manufacture match quality.
        coarse_factor = max(1, int(config.coarse_downsample))
        coarse_sigma = max(0.8, min(template.shape) * config.coarse_blur_fraction)
        coarse_template = cv2.GaussianBlur(
            template, (0, 0), coarse_sigma, borderType=cv2.BORDER_REFLECT)
        coarse_search = cv2.GaussianBlur(
            search_edge, (0, 0), coarse_sigma, borderType=cv2.BORDER_REFLECT)
        if coarse_factor > 1:
            coarse_template = cv2.resize(
                coarse_template, None, fx=1.0 / coarse_factor,
                fy=1.0 / coarse_factor, interpolation=cv2.INTER_AREA)
            coarse_search = cv2.resize(
                coarse_search, None, fx=1.0 / coarse_factor,
                fy=1.0 / coarse_factor, interpolation=cv2.INTER_AREA)
        if (proposal.source != "boundary_intensity"
                and coarse_template.shape[0] < coarse_search.shape[0]
                and coarse_template.shape[1] < coarse_search.shape[1]
                and float(coarse_template.std()) >= 1e-5):
            coarse_surface = cv2.matchTemplate(
                coarse_search, coarse_template, cv2.TM_CCOEFF_NORMED)
            coarse_peaks = _surface_peaks(
                coarse_surface, coarse_template.shape,
                config.coarse_peaks_per_pose, config.spatial_nms_fraction)
            for _coarse_value, coarse_col, coarse_row in coarse_peaks:
                col = int(np.clip(
                    round(coarse_col * coarse_factor), 0, surface.shape[1] - 1))
                row = int(np.clip(
                    round(coarse_row * coarse_factor), 0, surface.shape[0] - 1))
                # Snap within one coarse pixel to the best fine-grid response;
                # this removes phase error introduced by area decimation.
                radius = coarse_factor
                y0, y1 = max(0, row - radius), min(surface.shape[0], row + radius + 1)
                x0, x1 = max(0, col - radius), min(surface.shape[1], col + radius + 1)
                _lo, value, _lloc, offset = cv2.minMaxLoc(surface[y0:y1, x0:x1])
                fine_col, fine_row = x0 + offset[0], y0 + offset[1]
                if all(np.hypot(fine_col - old_col, fine_row - old_row)
                       >= max(2.0, min(template.shape) * 0.08)
                    for _old_value, old_col, old_row in peaks):
                    peaks.append((float(value), fine_col, fine_row))
        guard_radius = max(2, int(round(min(template.shape) * 0.16)))
        surface_mean, surface_std = cv2.meanStdDev(surface)
        mean_value = float(surface_mean[0, 0])
        std_value = float(surface_std[0, 0])
        surface_moments = (
            mean_value * surface.size,
            (std_value * std_value + mean_value * mean_value) * surface.size,
        )
        # A RANSAC centre is valuable when repetition makes several correlation
        # peaks equivalent.  Add it explicitly, with its correlation sampled
        # from the same edge surface.
        if proposal.x is not None and proposal.y is not None:
            col = int(round(proposal.x - half_x))
            row = int(round(proposal.y - half_y))
            if 0 <= row < surface.shape[0] and 0 <= col < surface.shape[1]:
                peaks.append((float(surface[row, col]), col, row))
        for value, col, row in peaks:
            candidates.append(_Candidate(
                x=float(col + half_x), y=float(row + half_y),
                scale=proposal.scale, theta=proposal.theta,
                correlation=float(value), pose_confidence=proposal.confidence,
                source=proposal.source, matches=proposal.matches,
                inliers=proposal.inliers,
                peak_to_sidelobe=_peak_to_sidelobe(
                    surface, row, col, guard_radius, surface_moments),
                photometric_margin=proposal.photometric_margin,
            ))
    candidates.sort(
        key=lambda item: item.correlation + 0.16 * item.pose_confidence,
        reverse=True,
    )
    return candidates, surfaces


def _select_refinement_candidates(
    candidates: list[_Candidate],
    config: EdgeConfig,
) -> list[_Candidate]:
    """Round-robin distinct pose modes into the bounded refinement budget.

    Correlation surfaces already apply spatial NMS within each pose.  Taking
    candidates in global source order can nevertheless spend the entire
    budget on aliases from one pose before a weaker, correct pose is seen.
    Grouping by source and pose, then taking one candidate from every group
    per pass preserves those alternatives while retaining correlation order
    within each surface.
    """
    groups: list[list[_Candidate]] = []
    for candidate in candidates:
        for group in groups:
            head = group[0]
            if (candidate.source == head.source
                    and abs(candidate.scale - head.scale) < 0.15
                    and abs(candidate.theta - head.theta) < 0.35):
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    selected: list[_Candidate] = []
    depth = 0
    while len(selected) < config.max_refine_candidates:
        added = False
        for group in groups:
            if depth >= len(group):
                continue
            candidate = group[depth]
            if all(np.hypot(candidate.x - old.x, candidate.y - old.y) >= 2.0
                   or abs(candidate.scale - old.scale) >= 0.15
                   or abs(candidate.theta - old.theta) >= 0.35
                   or candidate.source != old.source
                   for old in selected):
                selected.append(candidate)
                added = True
                if len(selected) >= config.max_refine_candidates:
                    break
        if not added and all(depth >= len(group) - 1 for group in groups):
            break
        depth += 1
    return selected


def _zncc(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float32).ravel()
    b = right.astype(np.float32).ravel()
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator <= 1e-9:
        return -1.0
    return float(np.dot(a, b) / denominator)


def _polarity_insensitive_orientation_agreement(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """Return edge-weighted gradient agreement, ignoring contrast polarity.

    Goshtasby's gradient-direction matching treats orientation as independent
    verification of an intensity or edge match.  Taking the absolute dot
    product makes bright-on-dark and dark-on-bright boundaries equivalent.
    The returned value is normalized so unrelated orientations (whose expected
    absolute cosine is ``2 / pi``) score near zero.
    """
    left32 = left.astype(np.float32, copy=False)
    right32 = right.astype(np.float32, copy=False)
    lx = cv2.Scharr(left32, cv2.CV_32F, 1, 0)
    ly = cv2.Scharr(left32, cv2.CV_32F, 0, 1)
    rx = cv2.Scharr(right32, cv2.CV_32F, 1, 0)
    ry = cv2.Scharr(right32, cv2.CV_32F, 0, 1)
    lm = cv2.magnitude(lx, ly)
    rm = cv2.magnitude(rx, ry)
    weights = np.minimum(lm, rm)
    valid = weights > max(float(np.percentile(weights, 55.0)), 1e-6)
    if int(np.count_nonzero(valid)) < 16:
        return 0.0
    cosine = np.abs((lx * rx + ly * ry) / np.maximum(lm * rm, 1e-12))
    raw = float(np.average(cosine[valid], weights=weights[valid]))
    random_baseline = 2.0 / np.pi
    return float(np.clip((raw - random_baseline) / (1.0 - random_baseline), 0.0, 1.0))


def _refine_candidate(
    reference_gray: np.ndarray,
    template_source: np.ndarray,
    search_gray: np.ndarray,
    search_edge: np.ndarray,
    candidate: _Candidate,
    config: EdgeConfig,
) -> _Candidate:
    """Bounded continuous four-parameter similarity refinement."""
    from scipy.optimize import minimize

    shape = (
        max(8, int(round(reference_gray.shape[0] / candidate.scale))),
        max(8, int(round(reference_gray.shape[1] / candidate.scale))),
    )
    half_x, half_y = (shape[1] - 1.0) / 2.0, (shape[0] - 1.0) / 2.0
    x_lo = max(half_x, candidate.x - config.refine_xy_radius_px)
    x_hi = min(search_edge.shape[1] - 1.0 - half_x,
               candidate.x + config.refine_xy_radius_px)
    y_lo = max(half_y, candidate.y - config.refine_xy_radius_px)
    y_hi = min(search_edge.shape[0] - 1.0 - half_y,
               candidate.y + config.refine_xy_radius_px)
    s_lo = max(config.scale_min, candidate.scale - config.refine_scale_radius)
    s_hi = min(config.scale_max, candidate.scale + config.refine_scale_radius)
    t_lo = max(-config.rotation_limit_deg,
               candidate.theta - config.refine_theta_radius_deg)
    t_hi = min(config.rotation_limit_deg,
               candidate.theta + config.refine_theta_radius_deg)
    if x_hi <= x_lo or y_hi <= y_lo or s_hi <= s_lo or t_hi <= t_lo:
        return candidate

    cache: dict[tuple[float, float], np.ndarray] = {}

    def objective(parameters: np.ndarray) -> float:
        x, y, theta, scale = (float(value) for value in parameters)
        key = (round(theta, 5), round(scale, 5))
        template = cache.get(key)
        if template is None:
            template = _make_template(
                template_source, scale, theta, config, output_shape=shape,
                source_scale=1.0,
            )
            cache[key] = template
        patch = cv2.getRectSubPix(search_edge, (shape[1], shape[0]), (x, y))
        return -_zncc(patch, template)

    initial = np.array([candidate.x, candidate.y, candidate.theta, candidate.scale],
                       dtype=np.float64)
    result = minimize(
        objective, initial, method="Powell",
        bounds=((x_lo, x_hi), (y_lo, y_hi), (t_lo, t_hi), (s_lo, s_hi)),
        options={
            "maxiter": config.optimizer_maxiter,
            "maxfev": config.optimizer_maxfev,
            "xtol": 0.025,
            "ftol": 2e-4,
        },
    )
    values = result.x if np.isfinite(result.fun) else initial
    correlation = float(-objective(values))
    intensity_template = _warp_template_pixels(
        template_source, float(values[3]), float(values[2]), shape, 1.0)
    intensity_patch = cv2.getRectSubPix(
        search_gray, (shape[1], shape[0]),
        (float(values[0]), float(values[1])))
    intensity_correlation = _zncc(intensity_patch, intensity_template)
    orientation_agreement = _polarity_insensitive_orientation_agreement(
        intensity_patch, intensity_template)
    return _Candidate(
        x=float(values[0]), y=float(values[1]), theta=float(values[2]),
        scale=float(values[3]), correlation=correlation,
        pose_confidence=candidate.pose_confidence, source=candidate.source,
        matches=candidate.matches, inliers=candidate.inliers,
        intensity_correlation=intensity_correlation,
        orientation_agreement=orientation_agreement,
        peak_to_sidelobe=candidate.peak_to_sidelobe,
        photometric_margin=candidate.photometric_margin,
    )


def _empty_result(search_shape: tuple[int, int], reason: str,
                  diagnostics: dict[str, Any] | None = None) -> EdgeResult:
    height, width = search_shape
    data = {} if diagnostics is None else dict(diagnostics)
    data["rejection_reason"] = reason
    return EdgeResult(
        x=float((width - 1.0) / 2.0), y=float((height - 1.0) / 2.0),
        theta=0.0, scale=10.0, found=False, score=0.0,
        edge_correlation=0.0, runner_up_correlation=0.0,
        ambiguity=1.0, pose_confidence=0.0, diagnostics=data,
    )


def _source_verified(candidate: _Candidate, correlation: float, gap: float,
                     config: EdgeConfig) -> bool:
    """Require independent geometric or spatial evidence for presence.

    A repeated layout can give high absolute correlation at an unrelated
    periodic site.  Spectral proposals therefore always need at least modest
    spatial isolation; moderate correlations need a larger margin.
    """
    strong_edge_agreement = correlation >= 0.72 and gap >= 0.05
    isolated_edge_agreement = correlation >= 0.47 and gap >= 0.10
    orientation_verified = (
        correlation >= 0.47 and gap >= 0.05
        and candidate.orientation_agreement >= 0.18
    )
    degraded_edge_agreement = (
        correlation >= 0.35 and gap >= 0.05
        and candidate.orientation_agreement >= 0.18
    )
    if candidate.source == "descriptor_ransac":
        return bool(
            candidate.inliers >= config.min_ransac_inliers
            and correlation >= 0.30
        )
    if candidate.source.endswith("_scale_boundary"):
        boundary_verified = (
            (correlation >= 0.80 and gap >= 0.05)
            or (correlation >= 0.35 and gap >= 0.12)
        )
        return bool(
            candidate.pose_confidence >= config.min_pose_confidence
            and boundary_verified
        )
    if candidate.source.startswith("directional_spectrum"):
        return bool(
            candidate.pose_confidence >= config.min_pose_confidence
            and (strong_edge_agreement or isolated_edge_agreement
                 or orientation_verified or degraded_edge_agreement)
        )
    if candidate.source.startswith("orientation_nominal"):
        return bool(
            candidate.pose_confidence >= 0.15
            and (strong_edge_agreement or isolated_edge_agreement
                 or orientation_verified or degraded_edge_agreement)
        )
    if candidate.source == "coarse_global_edge":
        photometric = abs(candidate.intensity_correlation)
        return bool(
            candidate.pose_confidence >= config.min_pose_confidence
            and ((photometric >= 0.45 and gap >= 0.13)
                 or (photometric >= 0.55
                     and candidate.orientation_agreement >= 0.55
                     and correlation >= 0.35)
                 or (photometric >= 0.45
                     and 0.35 <= correlation <= 0.50
                     and gap >= 0.07
                     and candidate.orientation_agreement >= 0.25))
        )
    if candidate.source == "boundary_intensity":
        return bool(
            candidate.pose_confidence >= 0.22
            and candidate.photometric_margin >= 0.09
            and correlation >= 0.15
        )
    return False


@contextmanager
def _opencv_budget(config: EdgeConfig):
    requested = config.opencv_threads
    if requested is None:
        value = os.environ.get("DRIFTFORGE_NUM_THREADS", "").strip()
        if value:
            try:
                requested = int(value)
            except ValueError:
                requested = None
    if requested is None or requested <= 0 or cv2 is None:
        yield
        return
    previous = int(cv2.getNumThreads())
    cv2.setNumThreads(int(requested))
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def _solve_edges_impl(
    reference: np.ndarray,
    search: np.ndarray,
    config: EdgeConfig,
) -> EdgeResult:
    """Register a high-magnification Reference in Search using edge evidence.

    Returned coordinates are ``x=column, y=row`` in Search pixels.  Every
    scalar is finite, including honest abstentions.  The solver never consults
    filenames, metadata, seeds, or ground truth.
    """
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("edge registration requires opencv-python-headless")
    reference_gray = _as_gray(reference)
    search_gray = _as_gray(search)
    if min(reference_gray.shape) < 64 or min(search_gray.shape) < 64:
        return _empty_result(search_gray.shape, "image_too_small")

    ref_contrast, ref_gradient = _contrast_and_gradient(reference_gray, config)
    search_contrast, search_gradient = _contrast_and_gradient(search_gray, config)
    base_diag: dict[str, Any] = {
        "method": "gaussian_scharr_similarity",
        "reference_contrast": ref_contrast,
        "search_contrast": search_contrast,
        "reference_gradient_rms": ref_gradient,
        "search_gradient_rms": search_gradient,
        "dense_pose_fallback": False,
    }
    if (ref_contrast < config.min_contrast or search_contrast < config.min_contrast
            or ref_gradient < config.min_gradient_rms
            or search_gradient < config.min_gradient_rms):
        return _empty_result(search_gray.shape, "insufficient_edge_energy", base_diag)

    nominal = _nominal_reference(reference_gray, config)
    if config.boundary_rescue_only:
        local, spectral = [], []
        local_diag = {"reference_keypoints": 0, "search_keypoints": 0,
                      "descriptor_matches": 0, "ransac_proposals": 0}
        spectral_diag = {"spectral_scale": config.nominal_scale,
                         "spectral_theta": 0.0,
                         "orientation_strength": 0.0,
                         "spectral_confidence": 0.0,
                         "directional_spectral_proposals": []}
    else:
        local, local_diag = _descriptor_proposals(nominal, search_gray, config)
        spectral, spectral_diag = _spectral_proposals(nominal, search_gray, config)
    global_edge, global_edge_diag = _global_edge_pose_proposals(
        reference_gray, search_gray, config)
    boundary = _boundary_rescue_proposals(reference_gray, search_gray, config)
    proposals = _add_scale_boundary_proposals(
        _deduplicate_proposals(local + spectral, config), config)
    proposals += global_edge + boundary
    base_diag.update(local_diag)
    base_diag.update(spectral_diag)
    base_diag.update(global_edge_diag)
    base_diag["pose_proposals"] = [
        {
            "scale": float(item.scale), "theta": float(item.theta),
            "confidence": float(item.confidence), "source": item.source,
            "inliers": int(item.inliers),
            "scale_uncertainty": float(item.scale_uncertainty),
            "theta_uncertainty_deg": float(item.theta_uncertainty_deg),
            "support": int(item.support),
            "directional_diversity_deg": float(item.directional_diversity_deg),
        }
        for item in proposals
    ]
    if not proposals:
        return _empty_result(search_gray.shape, "no_pose_proposal", base_diag)

    search_edge = _edge_features(search_gray, config)
    # Pre-filter once at the finest allowed output sampling rate. warpAffine
    # then only performs the geometric resampling; it no longer aliases native
    # high-frequency Reference pixels into every optimizer evaluation.
    template_source = cv2.GaussianBlur(
        reference_gray, (0, 0), 0.5 * config.scale_min,
        borderType=cv2.BORDER_REFLECT,
    )
    candidates, surface_count = _spatial_candidates(
        template_source, search_edge, proposals, config)
    base_diag["correlation_surfaces"] = int(surface_count)
    base_diag["spatial_candidates"] = int(len(candidates))
    if not candidates:
        return _empty_result(search_gray.shape, "no_spatial_candidate", base_diag)

    # Preserve at least one spatially NMS-separated candidate from every pose
    # before admitting second aliases.  This prevents a strong periodic mode
    # from exhausting the fixed continuous-refinement budget.
    selected = _select_refinement_candidates(candidates, config)
    refined = [_refine_candidate(
        reference_gray, template_source, search_gray, search_edge, item, config)
               for item in selected]
    refined.sort(
        key=lambda item: (item.correlation + 0.16 * item.pose_confidence
                          + 0.08 * item.orientation_agreement
                          + config.intensity_rank_weight
                          * abs(item.intensity_correlation)),
        reverse=True,
    )
    best = refined[0]

    template_side = min(reference_gray.shape) / best.scale
    remote = [
        item for item in refined[1:] + candidates
        if np.hypot(item.x - best.x, item.y - best.y)
        >= config.spatial_nms_fraction * template_side
    ]
    runner_up = max((item.correlation for item in remote), default=-1.0)
    runner_up = float(np.clip(runner_up, -1.0, 1.0))
    correlation = float(np.clip(best.correlation, -1.0, 1.0))
    gap = max(0.0, correlation - runner_up)
    ambiguity = float(np.clip(1.0 - gap / 0.12, 0.0, 1.0))
    corr_quality = float(np.clip((correlation - 0.10) / 0.55, 0.0, 1.0))
    gap_quality = float(np.clip(gap / 0.08, 0.0, 1.0))
    pose_confidence = float(np.clip(best.pose_confidence, 0.0, 1.0))
    sidelobe_quality = float(np.clip(
        (best.peak_to_sidelobe - 4.0) / 8.0, 0.0, 1.0))
    # Peak height measures local fit, while the standardized peak measures how
    # surprising that fit is for this image pair.  Keep margin, pose, and
    # orientation evidence in the score so periodic aliases do not win merely
    # by having a narrow response distribution.
    score = float(np.clip(
        0.30 * corr_quality + 0.20 * pose_confidence
        + 0.15 * gap_quality + 0.20 * sidelobe_quality
        + 0.15 * best.orientation_agreement,
        0.0, 1.0,
    ))
    strong_isolated_edge = bool(correlation >= 0.58 and gap >= 0.10)
    source_verified = _source_verified(best, correlation, gap, config)
    required_score = (0.10 if best.source == "boundary_intensity"
                      else config.min_found_score)
    found = bool(
        correlation >= config.min_edge_correlation
        and pose_confidence >= config.min_pose_confidence
        and score >= required_score
        and source_verified
    )
    # The Cramer--Rao diagnostic describes local jitter around the selected
    # correlation lobe.  It is deliberately reported rather than blended into
    # ``score``: a sharp but incorrect periodic alias can have excellent local
    # Fisher information even though the global location is wrong.
    from .fisher_confidence import fisher_confidence

    fisher_shape = (
        max(8, int(round(reference_gray.shape[0] / best.scale))),
        max(8, int(round(reference_gray.shape[1] / best.scale))),
    )
    fisher_template = _warp_template_pixels(
        template_source, best.scale, best.theta, fisher_shape, 1.0)
    fisher_patch = cv2.getRectSubPix(
        search_gray, (fisher_shape[1], fisher_shape[0]), (best.x, best.y))
    fisher = fisher_confidence(fisher_template, fisher_patch)
    base_diag.update({
        "proposal_source": best.source,
        "descriptor_inliers": int(best.inliers),
        "candidate_count_refined": len(refined),
        "candidate_aliases": [
            {
                "x": float(item.x), "y": float(item.y),
                "scale": float(item.scale), "theta": float(item.theta),
                "edge_correlation": float(item.correlation),
                "source": item.source,
            }
            for item in refined[:8]
        ],
        "correlation_gap": float(gap),
        "peak_to_sidelobe": float(best.peak_to_sidelobe),
        "orientation_agreement": float(best.orientation_agreement),
        "intensity_correlation": float(best.intensity_correlation),
        "fisher": {
            "std_x_px": float(fisher.std_x),
            "std_y_px": float(fisher.std_y),
            "std_rotation_deg": float(fisher.std_rotation_deg),
            "std_log_scale": float(fisher.std_log_scale),
            "noise_sigma": float(fisher.noise_sigma),
            "effective_samples": float(fisher.effective_samples),
            "confidence": float(fisher.confidence),
            "identifiable": bool(fisher.identifiable),
        },
        "strong_isolated_edge": strong_isolated_edge,
        "rejection_reason": None if found else "weak_edge_evidence",
    })
    return EdgeResult(
        x=float(np.clip(best.x, 0.0, search_gray.shape[1] - 1.0)),
        y=float(np.clip(best.y, 0.0, search_gray.shape[0] - 1.0)),
        theta=float(np.clip(best.theta, -config.rotation_limit_deg,
                            config.rotation_limit_deg)),
        scale=float(np.clip(best.scale, config.scale_min, config.scale_max)),
        found=found,
        score=score,
        edge_correlation=correlation,
        runner_up_correlation=runner_up,
        ambiguity=ambiguity,
        pose_confidence=pose_confidence,
        diagnostics=base_diag,
    )


def solve_edges(
    reference: np.ndarray,
    search: np.ndarray,
    config: EdgeConfig = EdgeConfig(),
) -> EdgeResult:
    """Register Reference in Search under an optional OpenCV thread budget.

    ``DRIFTFORGE_NUM_THREADS`` is an opt-in process setting for constrained
    hosts.  ``EdgeConfig.opencv_threads`` takes precedence when provided, and
    the previous OpenCV setting is restored before returning.
    """
    with _opencv_budget(config):
        # Preserve the fast, calibrated local/spectral path exactly.  The
        # global bank is a rescue for honest abstentions, never a competitor
        # that may displace an already verified answer.
        baseline_config = replace(
            config, global_pose_shortlist=0, boundary_rescue_shortlist=0)
        baseline = _solve_edges_impl(reference, search, baseline_config)
        if (baseline.found or (config.global_pose_shortlist <= 0
                              and config.boundary_rescue_shortlist <= 0)):
            return baseline
        edge_rescue_config = replace(
            config, max_refine_candidates=max(config.max_refine_candidates, 16),
            intensity_rank_weight=max(config.intensity_rank_weight, 0.12),
            boundary_rescue_shortlist=0)
        edge_rescue = _solve_edges_impl(reference, search, edge_rescue_config)
        if edge_rescue.found:
            edge_rescue.diagnostics["global_pose_rescue"] = True
            return edge_rescue
        if config.boundary_rescue_shortlist > 0:
            boundary_config = replace(
                config, global_pose_shortlist=0, boundary_rescue_only=True,
                max_refine_candidates=max(config.max_refine_candidates,
                                          config.boundary_rescue_shortlist))
            boundary_rescue = _solve_edges_impl(
                reference, search, boundary_config)
            if boundary_rescue.found:
                boundary_rescue.diagnostics["boundary_pose_rescue"] = True
                return boundary_rescue
        return baseline


__all__ = ["EdgeConfig", "EdgeResult", "solve_edges"]
