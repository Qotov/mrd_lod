"""Tests for the analytic fast path (BUILD_SPEC 4.2).

The MC-vs-analytic agreement (the headline test) lives in test_agreement.py;
here we cover the closed forms in isolation, the regime guard, monotonicity, and
the LR aggregate-proxy dispatch.
"""

from __future__ import annotations

import pytest

from mrd_lod_sim.analytic import (
    AnalyticRegimeError,
    count_threshold,
    detection_probability,
    uses_aggregate_proxy,
)
from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import (
    AggregatePoissonRule,
    KofNRule,
    LikelihoodRatioRule,
)
from mrd_lod_sim.errors import ConstantError
from mrd_lod_sim.panel import PanelModel
from mrd_lod_sim.params import MoleculeParams
from scipy.stats import poisson


def cfg(**overrides) -> AssayConfig:
    base = dict(
        molecules=MoleculeParams(input_ng=30.0, conversion_efficiency=0.4),
        panel=PanelModel.clonal(500),
        error_model=ConstantError(1e-5),
    )
    base.update(overrides)
    return AssayConfig(**base)


def test_count_threshold_satisfies_alpha() -> None:
    lam = 15.0
    t = count_threshold(lam, 0.05)
    assert poisson.sf(t - 1, lam) <= 0.05
    assert poisson.sf(t - 2, lam) > 0.05  # minimal


def test_aggregate_probability_bounds() -> None:
    rule = AggregatePoissonRule(alpha=0.05)
    p_zero = detection_probability(cfg(), rule, vaf=0.0)
    p_high = detection_probability(cfg(), rule, vaf=1e-3)
    assert 0.0 <= p_zero <= 0.10  # near the false-positive floor
    assert p_high == pytest.approx(1.0, abs=1e-6)


def test_detection_probability_monotone_in_vaf() -> None:
    rule = AggregatePoissonRule()
    ps = [detection_probability(cfg(), rule, v) for v in (1e-6, 1e-5, 1e-4, 1e-3)]
    assert all(a <= b + 1e-12 for a, b in zip(ps, ps[1:]))


def test_monotone_in_panel_size_and_input() -> None:
    rule = AggregatePoissonRule()
    vaf = 5e-6
    small = detection_probability(cfg(panel=PanelModel.clonal(100)), rule, vaf)
    big = detection_probability(cfg(panel=PanelModel.clonal(2000)), rule, vaf)
    assert big >= small
    lo_in = detection_probability(
        cfg(molecules=MoleculeParams(10.0, 0.4)), rule, vaf
    )
    hi_in = detection_probability(
        cfg(molecules=MoleculeParams(80.0, 0.4)), rule, vaf
    )
    assert hi_in >= lo_in


def test_lower_error_regime_improves_detection() -> None:
    rule = AggregatePoissonRule()
    vaf = 3e-6
    raw = detection_probability(cfg(error_model=ConstantError(1e-3)), rule, vaf)
    duplex = detection_probability(cfg(error_model=ConstantError(1e-7)), rule, vaf)
    assert duplex >= raw


def test_kofn_probability_monotone() -> None:
    rule = KofNRule(k=2, per_site_alpha=0.01)
    ps = [detection_probability(cfg(), rule, v) for v in (1e-6, 1e-5, 1e-4)]
    assert ps[0] <= ps[1] <= ps[2]


def test_lr_uses_aggregate_proxy() -> None:
    lr = LikelihoodRatioRule(alpha=0.05)
    agg = AggregatePoissonRule(alpha=0.05)
    assert uses_aggregate_proxy(lr)
    assert not uses_aggregate_proxy(agg)
    # Proxy => identical curve to the aggregate rule at matched alpha.
    for vaf in (1e-6, 1e-5, 1e-4):
        assert detection_probability(cfg(), lr, vaf) == pytest.approx(
            detection_probability(cfg(), agg, vaf)
        )


def test_regime_guard_raises_on_dropout() -> None:
    with pytest.raises(AnalyticRegimeError):
        detection_probability(
            cfg(panel=PanelModel(500, dropout_prob=0.1)),
            AggregatePoissonRule(),
            1e-5,
        )


def test_regime_guard_raises_on_dispersion() -> None:
    with pytest.raises(AnalyticRegimeError):
        detection_probability(
            cfg(panel=PanelModel(500, depth_dispersion=2.0)),
            AggregatePoissonRule(),
            1e-5,
        )
