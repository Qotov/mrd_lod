"""Tests for the error models and the sample()/mean_rate() contract.

The headline guard here is test 9 (BUILD_SPEC 10): for the log-normal model the
sample mean must converge to ``mean_rate()`` -- so a median/mean mix-up fails
loudly rather than silently biasing the analytic background at ppm scale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrd_lod_sim.errors import (
    REGIME_RATES,
    ConstantError,
    ContextualError,
    EmpiricalError,
    LogNormalError,
    error_preset,
)


def rng() -> np.random.Generator:
    return np.random.default_rng(20260718)


def test_constant_sample_and_mean() -> None:
    m = ConstantError(1e-5)
    s = m.sample((3, 4), rng())
    assert s.shape == (3, 4)
    assert np.all(s == 1e-5)
    assert m.mean_rate() == 1e-5


def test_lognormal_mean_is_not_median() -> None:
    m = LogNormalError(median=1e-5, sigma=0.8)
    # E[eps] = median * exp(sigma^2/2), strictly greater than the median.
    assert m.mean_rate() == pytest.approx(1e-5 * math.exp(0.8**2 / 2))
    assert m.mean_rate() > m.median


def test_lognormal_sample_mean_converges_to_mean_rate() -> None:
    # BUILD_SPEC 10, test 9: the guard against the median/mean trap.
    m = LogNormalError(median=1e-5, sigma=0.9)
    draws = m.sample(4_000_000, rng())
    assert draws.mean() == pytest.approx(m.mean_rate(), rel=0.01)


def test_constant_preset_mean_matches_regime() -> None:
    for regime, rate in REGIME_RATES.items():
        assert error_preset(regime).mean_rate() == pytest.approx(rate)


def test_lognormal_preset_mean_still_matches_regime() -> None:
    # The whole point of the median-reparameterisation: mean == quoted rate.
    for regime, rate in REGIME_RATES.items():
        m = error_preset(regime, sigma=1.1)
        assert isinstance(m, LogNormalError)
        assert m.mean_rate() == pytest.approx(rate, rel=1e-12)


def test_regime_ordering() -> None:
    assert (
        error_preset("RAW").mean_rate()
        > error_preset("SSCS").mean_rate()
        > error_preset("DUPLEX").mean_rate()
    )


def test_unknown_regime_rejected() -> None:
    with pytest.raises(ValueError):
        error_preset("nonsense")


def test_contextual_mean_uniform_default() -> None:
    m = ContextualError({"CpG": 3e-5, "other": 1e-5})
    # Uniform default weights -> simple average.
    assert m.mean_rate() == pytest.approx(2e-5)


def test_contextual_mean_respects_panel_weights() -> None:
    m = ContextualError({"CpG": 3e-5, "other": 1e-5})

    class Panel:
        context_weights = {"CpG": 0.25, "other": 0.75}

    assert m.mean_rate(Panel()) == pytest.approx(0.25 * 3e-5 + 0.75 * 1e-5)


def test_contextual_sample_marginal_matches_mean() -> None:
    m = ContextualError(
        {"CpG": 3e-5, "other": 1e-5}, context_weights={"CpG": 0.2, "other": 0.8}
    )
    draws = m.sample(2_000_000, rng())
    assert draws.mean() == pytest.approx(m.mean_rate(), rel=0.02)


def test_empirical_mean_is_pooled_rate() -> None:
    mut = np.array([1.0, 0.0, 2.0, 1.0])
    tot = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    m = EmpiricalError(mut, tot)
    assert m.mean_rate() == pytest.approx(4.0 / 4000.0)


def test_empirical_sample_bootstrap_mean() -> None:
    mut = np.array([1.0, 3.0, 0.0, 2.0, 4.0])
    tot = np.full(5, 1000.0)
    m = EmpiricalError(mut, tot)
    draws = m.sample(1_000_000, rng())
    # Bootstrap of per-site rates -> mean of per-site rates (not pooled, but
    # equal here since depths are uniform).
    assert draws.mean() == pytest.approx((mut / tot).mean(), rel=0.02)


def test_empirical_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        EmpiricalError(np.array([1.0]), np.array([0.0]))  # zero depth
    with pytest.raises(ValueError):
        EmpiricalError(np.array([1.0, 2.0]), np.array([10.0]))  # shape mismatch
