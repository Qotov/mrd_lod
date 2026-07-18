"""Tests for the detection rules -- the production caller (BUILD_SPEC 3).

These exercise the interface contract (rules consume only per-site counts + an
error model), the estimated_vaf optionality (KofN returns None), and basic
statistical behaviour (blanks are quiet, strong signal is detected, VAF
estimates are sane).
"""

from __future__ import annotations

import numpy as np
import pytest

from mrd_lod_sim.detect import (
    AggregatePoissonRule,
    KofNRule,
    LikelihoodRatioRule,
    MRDCall,
    SiteObservations,
)
from mrd_lod_sim.errors import ConstantError

RULES = [KofNRule(k=2, per_site_alpha=0.01), AggregatePoissonRule(alpha=0.05), LikelihoodRatioRule(alpha=0.05)]


def blank_obs(n: int = 500, depth: float = 3000.0) -> SiteObservations:
    return SiteObservations(
        mutant_counts=np.zeros(n), total_counts=np.full(n, depth)
    )


def signal_obs(
    n: int = 500, depth: float = 3000.0, mutant_per_site: float = 5.0, n_signal: int = 20
) -> SiteObservations:
    mut = np.zeros(n)
    mut[:n_signal] = mutant_per_site
    return SiteObservations(mutant_counts=mut, total_counts=np.full(n, depth))


@pytest.mark.parametrize("rule", RULES)
def test_blank_not_detected(rule) -> None:
    call = rule.call(blank_obs(), ConstantError(1e-5))
    assert isinstance(call, MRDCall)
    assert call.detected is False


@pytest.mark.parametrize("rule", RULES)
def test_strong_signal_detected(rule) -> None:
    call = rule.call(signal_obs(), ConstantError(1e-5))
    assert call.detected is True


def test_kofn_does_not_estimate_vaf() -> None:
    call = KofNRule().call(signal_obs(), ConstantError(1e-5))
    assert call.estimated_vaf is None
    assert call.estimated_ci is None
    assert call.n_positive_sites is not None


def test_aggregate_and_lr_estimate_vaf() -> None:
    for rule in (AggregatePoissonRule(), LikelihoodRatioRule()):
        call = rule.call(signal_obs(), ConstantError(1e-5))
        assert call.estimated_vaf is not None
        assert call.estimated_vaf > 0


def test_aggregate_vaf_estimate_is_reasonable() -> None:
    # 20 sites * 5 mutant = 100 mutant molecules; background lam_bg =
    # 500*3000*1e-5 = 15 molecules; total depth = 1.5e6. The estimate is
    # background-subtracted: (100 - 15) / 1.5e6.
    call = AggregatePoissonRule().call(signal_obs(), ConstantError(1e-5))
    assert call.estimated_vaf == pytest.approx((100 - 15) / 1.5e6, rel=0.02)
    lo, hi = call.estimated_ci
    assert lo <= call.estimated_vaf <= hi


def test_lr_vaf_estimate_is_reasonable() -> None:
    call = LikelihoodRatioRule().call(signal_obs(), ConstantError(1e-5))
    # LR estimates per-site VAF, likewise net of background.
    assert call.estimated_vaf == pytest.approx((100 - 15) / 1.5e6, rel=0.10)


def test_uses_per_site_error_rates_when_present() -> None:
    # One noisy site should not, by itself, trip the k-of-N rule when its own
    # high error rate is supplied.
    n = 100
    mut = np.zeros(n)
    mut[0] = 8.0  # elevated count at a site we will declare noisy
    eps = np.full(n, 1e-5)
    eps[0] = 3e-3  # this site is known-noisy -> 8 counts unremarkable
    obs = SiteObservations(
        mutant_counts=mut, total_counts=np.full(n, 3000.0), error_rates=eps
    )
    call = KofNRule(k=1, per_site_alpha=0.001).call(obs, ConstantError(1e-9))
    # With the site's true (high) rate, count of 8 vs lam_bg=9 is not significant.
    assert call.per_site["positive"][0] == False  # noqa: E712


def test_calibrated_threshold_overrides_alpha() -> None:
    rule = AggregatePoissonRule(decision_threshold=1e9)  # impossibly high
    call = rule.call(signal_obs(), ConstantError(1e-5))
    assert call.detected is False  # even strong signal can't clear the threshold


def test_statistic_monotone_in_signal() -> None:
    weak = AggregatePoissonRule().call(
        signal_obs(mutant_per_site=1.0), ConstantError(1e-5)
    )
    strong = AggregatePoissonRule().call(
        signal_obs(mutant_per_site=10.0), ConstantError(1e-5)
    )
    assert strong.statistic > weak.statistic


def test_observation_validation() -> None:
    with pytest.raises(ValueError):
        SiteObservations(np.array([5.0]), np.array([2.0]))  # mutant > total
    with pytest.raises(ValueError):
        SiteObservations(np.array([[1.0]]), np.array([[10.0]]))  # 2-D
