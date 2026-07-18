"""Tests for the LoD surface and lever analysis (BUILD_SPEC 7)."""

from __future__ import annotations

import numpy as np
import pytest

from mrd_lod_sim.analytic import detection_probability
from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import AggregatePoissonRule
from mrd_lod_sim.errors import ConstantError
from mrd_lod_sim.panel import PanelModel
from mrd_lod_sim.params import MoleculeParams
from mrd_lod_sim.surface import achievable_lod, lod_surface, what_would_it_take


def cfg(**over) -> AssayConfig:
    base = dict(
        molecules=MoleculeParams(30.0, 0.4),
        panel=PanelModel.clonal(500),
        error_model=ConstantError(1e-5),
    )
    base.update(over)
    return AssayConfig(**base)


RULE = AggregatePoissonRule(alpha=0.05)


def test_duplex_strand_penalty_is_not_free() -> None:
    """The duplex trade-off (BUILD_SPEC 2.1, test #7): duplex lowers epsilon AND
    reduces usable molecules via strand_recovery, so the net benefit depends on
    the regime. Guards the dashboard bug where duplex was modelled as a pure,
    cost-free error reduction."""
    panel = PanelModel(500, ccf_alpha=9.0, ccf_beta=1.0)
    sscs = AssayConfig(MoleculeParams(30.0, 0.4, 1.0), panel, ConstantError(1e-5))
    duplex_free = AssayConfig(MoleculeParams(30.0, 0.4, 1.0), panel, ConstantError(1e-7))
    duplex = AssayConfig(MoleculeParams(30.0, 0.4, 0.5), panel, ConstantError(1e-7))

    # The molecule cost is real: strand recovery 0.5 halves effective GE.
    assert duplex.ge_eff() == pytest.approx(0.5 * sscs.ge_eff())

    lod_free = achievable_lod(duplex_free, RULE, 0.95)
    lod_duplex = achievable_lod(duplex, RULE, 0.95)
    lod_sscs = achievable_lod(sscs, RULE, 0.95)

    # Duplex is NOT free: the strand penalty worsens (raises) the LoD vs the
    # physically wrong cost-free duplex -- here it roughly doubles.
    assert lod_duplex > lod_free * 1.5
    # But error suppression still beats SSCS at this background-limited operating
    # point -- so the net effect is a genuine trade-off, not a strict loss.
    assert lod_duplex < lod_sscs
    # At a fixed VAF the strand penalty lowers detection probability.
    vaf = 3e-6
    assert detection_probability(duplex, RULE, vaf) < detection_probability(duplex_free, RULE, vaf)


def test_achievable_lod_is_at_hit_rate() -> None:
    from mrd_lod_sim.analytic import detection_probability

    lod = achievable_lod(cfg(), RULE, 0.95)
    assert np.isfinite(lod)
    assert detection_probability(cfg(), RULE, lod) == pytest.approx(0.95, abs=1e-3)


def test_achievable_lod_improves_with_more_input() -> None:
    lo = achievable_lod(cfg(molecules=MoleculeParams(10.0, 0.4)), RULE)
    hi = achievable_lod(cfg(molecules=MoleculeParams(80.0, 0.4)), RULE)
    assert hi < lo  # more input -> lower (better) LoD


def test_surface_monotone_across_axes() -> None:
    surf = lod_surface(
        cfg(), RULE, panel_sizes=[50, 200, 1000], input_masses=[10.0, 30.0, 90.0]
    )
    assert surf.lod.shape == (3, 3)
    # LoD improves (decreases) with larger panel (across columns)...
    for row in surf.lod:
        assert np.all(np.diff(row) <= 1e-15)
    # ...and with larger input mass (down rows).
    for col in surf.lod.T:
        assert np.all(np.diff(col) <= 1e-15)


def test_surface_as_dict_roundtrips_shape() -> None:
    surf = lod_surface(cfg(), RULE, [100, 500], [20.0, 60.0])
    d = surf.as_dict()
    assert len(d["lod"]) == 2 and len(d["lod"][0]) == 2


def test_what_would_it_take_on_missed_target() -> None:
    # A demanding target the base config misses.
    weak = cfg(molecules=MoleculeParams(10.0, 0.2), error_model=ConstantError(1e-4))
    target = 2e-6
    assert achievable_lod(weak, RULE) > target  # confirm it misses
    changes = what_would_it_take(weak, RULE, target)
    levers = {c.lever for c in changes}
    assert levers == {"panel_size", "input_mass", "conversion_efficiency", "error_regime"}
    # Each reported requirement, when applied, should actually meet the target.
    from dataclasses import replace
    from mrd_lod_sim.analytic import detection_probability

    for c in changes:
        if c.required is None:
            continue
        if c.lever == "panel_size":
            cfg2 = replace(weak, panel=replace(weak.panel, n_variants=int(c.required)))
        elif c.lever == "input_mass":
            cfg2 = replace(weak, molecules=replace(weak.molecules, input_ng=c.required))
        elif c.lever == "conversion_efficiency":
            cfg2 = replace(weak, molecules=replace(weak.molecules, conversion_efficiency=c.required))
        else:
            cfg2 = replace(weak, error_model=ConstantError(c.required))
        assert detection_probability(cfg2, RULE, target) >= 0.95 - 1e-3


def test_what_would_it_take_flags_plausibility() -> None:
    weak = cfg(molecules=MoleculeParams(10.0, 0.2), error_model=ConstantError(1e-4))
    changes = {c.lever: c for c in what_would_it_take(weak, RULE, 2e-6)}
    # Conversion efficiency cannot exceed the physical ceiling.
    conv = changes["conversion_efficiency"]
    if conv.required is None:
        assert conv.plausible is False
