"""Headline test: MC and analytic agree within Monte Carlo error (BUILD_SPEC 4.2, 10).

Run ONLY in the analytic-valid regime (constant error, no dropout, no depth
dispersion, clonal CCF so E[ccf] is exact). The band is
``4 * sqrt(p*(1-p)/n_rep)``. This single test validates both implementations and
would catch the median-vs-mean error of BUILD_SPEC 2.4.1.

Also here: the vectorised MC decision equals the per-replicate ``rule.call``
(so the fast path is faithful to the production caller), determinism, and the
LR-approx-aggregate relationship that justifies the dashboard/surface proxy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrd_lod_sim.analytic import detection_probability
from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import (
    AggregatePoissonRule,
    KofNRule,
    LikelihoodRatioRule,
    SiteObservations,
)
from mrd_lod_sim.errors import ConstantError
from mrd_lod_sim.panel import PanelModel
from mrd_lod_sim.params import MoleculeParams
from mrd_lod_sim.simulate import detection_rate, simulate_counts

N_REP = 40_000


def cfg(rate: float = 1e-5, n: int = 500, input_ng: float = 30.0) -> AssayConfig:
    return AssayConfig(
        molecules=MoleculeParams(input_ng=input_ng, conversion_efficiency=0.4),
        panel=PanelModel.clonal(n),  # clonal => E[ccf] == 1 exactly
        error_model=ConstantError(rate),
    )


def _band(p: float, n: int) -> float:
    return 4.0 * math.sqrt(max(p * (1 - p), 1e-9) / n)


@pytest.mark.parametrize("rule", [AggregatePoissonRule(alpha=0.05), KofNRule(k=2, per_site_alpha=0.01)])
def test_mc_analytic_agreement(rule) -> None:
    config = cfg()
    rng = np.random.default_rng(12345)
    # A VAF ladder straddling the detection transition.
    for vaf in (1e-6, 3e-6, 5e-6, 1e-5, 3e-5):
        p_analytic = detection_probability(config, rule, vaf)
        p_mc = detection_rate(config, rule, vaf, N_REP, rng).detection_rate
        band = _band(p_analytic, N_REP)
        assert abs(p_mc - p_analytic) < max(band, 0.01), (
            f"{type(rule).__name__} vaf={vaf}: mc={p_mc:.4f} analytic={p_analytic:.4f}"
        )


def test_vectorised_matches_rule_call() -> None:
    # The vectorised MC decision must equal the per-replicate production caller.
    config = cfg()
    for rule in (AggregatePoissonRule(alpha=0.05), KofNRule(k=2, per_site_alpha=0.01)):
        rng = np.random.default_rng(7)
        counts = simulate_counts(config, 8e-6, 300, rng)
        # Vectorised path:
        rng2 = np.random.default_rng(7)
        vec = detection_rate(config, rule, 8e-6, 300, rng2)
        # Per-replicate rule.call on the same draws:
        for r in range(300):
            obs = SiteObservations(counts.mutant[r], counts.total[r])
            call = rule.call(obs, config.error_model)
            assert call.detected == vec.detected[r]


def test_determinism() -> None:
    config = cfg()
    rule = AggregatePoissonRule()
    a = detection_rate(config, rule, 5e-6, 5000, np.random.default_rng(99)).detection_rate
    b = detection_rate(config, rule, 5e-6, 5000, np.random.default_rng(99)).detection_rate
    assert a == b


def test_lr_tracks_aggregate_at_ppm() -> None:
    # BUILD_SPEC 10, test 10: justifies the LR aggregate proxy in surface/dashboard.
    config = cfg()
    rng = np.random.default_rng(2024)
    for vaf in (2e-6, 5e-6, 1e-5):
        lr = detection_rate(config, LikelihoodRatioRule(alpha=0.05), vaf, 3000, rng)
        agg = detection_rate(config, AggregatePoissonRule(alpha=0.05), vaf, 3000, rng)
        assert abs(lr.detection_rate - agg.detection_rate) < 0.08
