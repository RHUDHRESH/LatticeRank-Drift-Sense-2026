import cv2
import numpy as np
import pytest

from driftforge.fisher_confidence import fisher_confidence


def _textured_patch(size: int = 72) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    return (np.sin(xx / 3.7) + 0.7 * np.cos(yy / 5.1) +
            0.4 * np.sin((xx + yy) / 7.3))


def test_returns_finite_full_rank_diagnostics():
    template = _textured_patch()
    rng = np.random.default_rng(4)
    observed = template + rng.normal(0.0, 0.08, template.shape)
    result = fisher_confidence(template, observed)

    assert result.identifiable
    assert result.information.shape == (4, 4)
    assert result.covariance.shape == (4, 4)
    assert np.all(np.isfinite(result.information))
    assert np.all(np.isfinite(result.covariance))
    assert 0.0 < result.confidence <= 1.0
    assert result.noise_sigma == pytest.approx(0.08, rel=0.12)


def test_more_noise_increases_translation_uncertainty():
    template = _textured_patch()
    rng = np.random.default_rng(8)
    noise = rng.normal(size=template.shape)
    low = fisher_confidence(template, template + 0.02 * noise)
    high = fisher_confidence(template, template + 0.20 * noise)

    assert high.std_x > low.std_x * 8
    assert high.std_y > low.std_y * 8
    assert high.confidence < low.confidence


def test_correlated_residual_reduces_effective_sample_count():
    template = _textured_patch()
    rng = np.random.default_rng(12)
    white = rng.normal(0, 0.05, template.shape)
    correlated = cv2.GaussianBlur(white, (0, 0), 1.5)
    correlated *= white.std() / correlated.std()
    independent = fisher_confidence(template, template + white)
    dependent = fisher_confidence(template, template + correlated)

    assert dependent.correlation_inflation > independent.correlation_inflation
    assert dependent.effective_samples < independent.effective_samples
    assert dependent.std_x > independent.std_x


def test_flat_patch_degrades_to_finite_zero_confidence():
    template = np.ones((32, 32), dtype=np.float64)
    result = fisher_confidence(template, template.copy())

    assert not result.identifiable
    assert result.confidence == 0.0
    assert np.all(np.isfinite(result.covariance))
    assert all(np.isfinite(value) for value in (
        result.std_x, result.std_y, result.std_rotation_rad,
        result.std_log_scale, result.condition_number,
    ))


def test_rejects_incompatible_inputs():
    with pytest.raises(ValueError, match="same-shaped 2-D"):
        fisher_confidence(np.zeros((8, 8)), np.zeros((7, 8)))

