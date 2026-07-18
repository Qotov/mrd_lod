"""Tests for the calibration hooks (BUILD_SPEC 6, 10).

The headline here is the round-trip (test 6): simulate with known parameters,
recover them via calibrate.py within tolerance -- this proves the "useful
later" claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from mrd_lod_sim.calibrate import (
    compare_predicted_observed,
    estimate_contextual_error,
    estimate_error_model,
    fit_conversion_efficiency,
)
from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import AggregatePoissonRule
from mrd_lod_sim.errors import ConstantError
from mrd_lod_sim.panel import PanelModel
from mrd_lod_sim.params import PG_PER_GENOME_EQUIVALENT, MoleculeParams
from mrd_lod_sim.simulate import detection_rate
from mrd_lod_sim.surface import achievable_lod
from mrd_lod_sim.validate import DetectionRecord


def test_error_model_round_trip() -> None:
    # Simulate healthy-donor counts with a known rate, recover it.
    true_rate = 3e-5
    rng = np.random.default_rng(1)
    n_sites = 5000
    depth = 3000.0
    mutant = rng.poisson(depth * true_rate, size=n_sites).astype(float)
    total = np.full(n_sites, depth)
    model = estimate_error_model(mutant, total)
    assert model.mean_rate() == pytest.approx(true_rate, rel=0.05)


def test_conversion_efficiency_round_trip() -> None:
    # BUILD_SPEC 10, test 6 (conversion arm).
    true_conv = 0.42
    rng = np.random.default_rng(2)
    masses = np.array([5.0, 10.0, 20.0, 30.0, 50.0, 100.0])
    ge_total = masses / PG_PER_GENOME_EQUIVALENT
    observed = rng.poisson(ge_total * true_conv).astype(float)  # count noise
    est = fit_conversion_efficiency(masses, observed)
    assert est == pytest.approx(true_conv, rel=0.05)


def test_conversion_efficiency_with_strand_recovery() -> None:
    true_conv, strand = 0.5, 0.5
    masses = np.array([10.0, 30.0, 60.0, 100.0])
    ge_total = masses / PG_PER_GENOME_EQUIVALENT
    observed = ge_total * true_conv * strand
    est = fit_conversion_efficiency(masses, observed, strand_recovery=strand)
    assert est == pytest.approx(true_conv, rel=1e-6)


def test_contextual_error_recovers_per_context_rates() -> None:
    rng = np.random.default_rng(3)
    depth = 5000.0
    rates = {"CpG": 5e-5, "other": 1e-5}
    mut, tot, ctx = [], [], []
    for c, rate in rates.items():
        n = 3000
        mut.extend(rng.poisson(depth * rate, size=n))
        tot.extend([depth] * n)
        ctx.extend([c] * n)
    model = estimate_contextual_error(np.array(mut, float), np.array(tot, float), ctx)
    assert model.context_rates["CpG"] == pytest.approx(5e-5, rel=0.1)
    assert model.context_rates["other"] == pytest.approx(1e-5, rel=0.15)


def cfg(rate: float = 1e-5, conv: float = 0.4) -> AssayConfig:
    return AssayConfig(
        molecules=MoleculeParams(30.0, conv),
        panel=PanelModel.clonal(500),
        error_model=ConstantError(rate),
    )


def test_compare_detects_sensitivity_loss() -> None:
    # Model assumes good conversion; "observed" comes from a much worse assay.
    assumed = cfg(conv=0.5)
    truth = cfg(conv=0.1)
    rule = AggregatePoissonRule(alpha=0.05)
    rng = np.random.default_rng(4)
    levels = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    records = []
    for lvl in levels:
        rate = detection_rate(truth, rule, lvl, 2000, rng).detection_rate
        records.append(DetectionRecord(lvl, 2000, int(round(rate * 2000))))
    diag = compare_predicted_observed(assumed, rule, records)
    assert diag.mean_signed_discrepancy < 0  # observed below predicted
    assert any("conversion" in c or "dropout" in c for c in diag.likely_causes)


def test_compare_reports_agreement_when_matched() -> None:
    config = cfg()
    rule = AggregatePoissonRule(alpha=0.05)
    rng = np.random.default_rng(5)
    levels = [1e-6, 5e-6, 1e-5, 5e-5]
    records = []
    for lvl in levels:
        rate = detection_rate(config, rule, lvl, 4000, rng).detection_rate
        records.append(DetectionRecord(lvl, 4000, int(round(rate * 4000))))
    diag = compare_predicted_observed(config, rule, records)
    assert abs(diag.mean_signed_discrepancy) < 0.1
    assert any("no material discrepancy" in c for c in diag.likely_causes)
