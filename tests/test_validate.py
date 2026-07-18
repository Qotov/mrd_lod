"""Tests for the data-source-agnostic validation layer (BUILD_SPEC 5, 10)."""

from __future__ import annotations

import numpy as np
import pytest

from mrd_lod_sim.validate import (
    DetectionRecord,
    calibrate_threshold,
    fit_detection_curve,
    limit_of_blank,
    limit_of_quantification,
    lod_from_curve,
    power_analysis,
)


def make_records(true_lod: float, slope: float = 3.0, n_rep: int = 400, rng=None):
    """Synthetic detection data from a known probit LoD (BUILD_SPEC 10, test 4)."""
    from scipy.stats import norm

    rng = rng or np.random.default_rng(0)
    # Choose intercept so that p(true_lod) = 0.95.
    intercept = norm.ppf(0.95) - slope * np.log10(true_lod)
    levels = np.array([true_lod / 8, true_lod / 4, true_lod / 2, true_lod, true_lod * 2])
    recs = []
    for lvl in levels:
        p = norm.cdf(intercept + slope * np.log10(lvl))
        k = rng.binomial(n_rep, p)
        recs.append(DetectionRecord(float(lvl), n_rep, int(k)))
    return recs


def test_validate_does_not_import_simulate() -> None:
    import mrd_lod_sim.validate as v
    import inspect

    src = inspect.getsource(v)
    assert "import simulate" not in src and "from mrd_lod_sim.simulate" not in src


def test_probit_recovers_known_lod() -> None:
    # BUILD_SPEC 10, test 4.
    true_lod = 5e-6
    recs = make_records(true_lod, rng=np.random.default_rng(1))
    res = lod_from_curve(recs, hit_rate=0.95, n_bootstrap=500, rng=np.random.default_rng(2))
    assert res.ci_low <= true_lod <= res.ci_high
    assert res.lod == pytest.approx(true_lod, rel=0.4)


def test_lod_ci_ordering() -> None:
    recs = make_records(5e-6, rng=np.random.default_rng(3))
    res = lod_from_curve(recs, n_bootstrap=300, rng=np.random.default_rng(4))
    assert res.ci_low <= res.lod <= res.ci_high


def test_curve_is_monotone_increasing() -> None:
    recs = make_records(5e-6, rng=np.random.default_rng(5))
    curve = fit_detection_curve(recs)
    ps = curve.predict(np.array([1e-7, 1e-6, 1e-5, 1e-4]))
    assert np.all(np.diff(ps) >= 0)


def test_fit_needs_two_levels() -> None:
    with pytest.raises(ValueError):
        fit_detection_curve([DetectionRecord(1e-5, 10, 5)])


def test_limit_of_blank_normal_agrees() -> None:
    rng = np.random.default_rng(6)
    blanks = rng.normal(10.0, 2.0, size=20_000)
    lob = limit_of_blank(blanks, alpha=0.05)
    # For clean normal data the two methods should agree.
    assert not lob.diverges
    assert lob.nonparametric == pytest.approx(lob.clsi_parametric, rel=0.05)


def test_limit_of_blank_flags_divergence() -> None:
    rng = np.random.default_rng(7)
    # Sparse skewed counts at ppm scale -> percentile and mean+z*SD diverge.
    blanks = rng.poisson(0.1, size=20_000).astype(float)
    lob = limit_of_blank(blanks, alpha=0.05)
    assert lob.diverges


def test_calibrate_threshold_achieves_specificity() -> None:
    rng = np.random.default_rng(8)
    blanks = rng.poisson(2.0, size=50_000).astype(float)
    thr = calibrate_threshold(blanks, target_specificity=0.99)
    achieved = np.mean(blanks < thr)
    assert achieved >= 0.99


def test_calibrate_threshold_held_out() -> None:
    # BUILD_SPEC 10, test 5: threshold from one blank set holds on another.
    rng = np.random.default_rng(9)
    train = rng.poisson(3.0, size=40_000).astype(float)
    test = rng.poisson(3.0, size=40_000).astype(float)
    thr = calibrate_threshold(train, target_specificity=0.99)
    assert np.mean(test < thr) >= 0.985  # within sampling error of 0.99


def test_limit_of_quantification() -> None:
    # Precise at high level, imprecise at low level.
    ests = {
        1e-4: np.full(50, 1e-4) + np.random.default_rng(10).normal(0, 1e-6, 50),
        1e-6: np.random.default_rng(11).normal(1e-6, 1e-6, 50),
    }
    loq = limit_of_quantification(ests, max_cv=0.20)
    assert loq == 1e-4


def test_power_analysis_more_replicates_narrows_ci() -> None:
    recs = make_records(5e-6, rng=np.random.default_rng(12))
    curve = fit_detection_curve(recs)
    levels = [5e-6 / 4, 5e-6 / 2, 5e-6, 5e-6 * 2]
    res = power_analysis(
        curve, levels, target_ci_width=1e-6, replicate_grid=[50, 500],
        n_bootstrap=150, rng=np.random.default_rng(13),
    )
    assert res.achieved_ci_width[500] < res.achieved_ci_width[50]
