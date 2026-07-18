"""Tests for the molecule budget and VAF/TF converters (BUILD_SPEC 2.1, 2.2)."""

from __future__ import annotations

import math

import pytest

from mrd_lod_sim.molecules import molecule_budget
from mrd_lod_sim.params import (
    PG_PER_GENOME_EQUIVALENT,
    MoleculeParams,
    tf_from_vaf,
    vaf_from_tf,
)


def test_ge_total_from_mass() -> None:
    # 33 ng / 3.3 pg per GE = 10,000 genome equivalents.
    p = MoleculeParams(input_ng=33.0, conversion_efficiency=1.0)
    b = molecule_budget(p)
    assert b.ge_total == pytest.approx(10_000.0)


def test_ge_eff_applies_conversion_and_strand_recovery() -> None:
    p = MoleculeParams(
        input_ng=33.0, conversion_efficiency=0.5, strand_recovery=0.5
    )
    b = molecule_budget(p)
    assert b.ge_total == pytest.approx(10_000.0)
    # 10,000 * 0.5 * 0.5
    assert b.ge_eff == pytest.approx(2_500.0)


def test_duplex_strand_recovery_reduces_usable_molecules() -> None:
    single = molecule_budget(
        MoleculeParams(input_ng=30.0, conversion_efficiency=0.4, strand_recovery=1.0)
    )
    duplex = molecule_budget(
        MoleculeParams(input_ng=30.0, conversion_efficiency=0.4, strand_recovery=0.5)
    )
    assert duplex.ge_eff == pytest.approx(single.ge_eff * 0.5)


def test_default_pg_per_ge() -> None:
    assert PG_PER_GENOME_EQUIVALENT == pytest.approx(0.0033)


@pytest.mark.parametrize(
    "bad",
    [
        dict(input_ng=0.0, conversion_efficiency=0.5),
        dict(input_ng=-1.0, conversion_efficiency=0.5),
        dict(input_ng=30.0, conversion_efficiency=0.0),
        dict(input_ng=30.0, conversion_efficiency=1.5),
        dict(input_ng=30.0, conversion_efficiency=0.5, strand_recovery=0.0),
    ],
)
def test_invalid_params_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        MoleculeParams(**bad)


# --- VAF / TF converters ---------------------------------------------------


def test_clonal_het_is_factor_of_two() -> None:
    assert vaf_from_tf(0.01) == pytest.approx(0.005)
    assert tf_from_vaf(0.005) == pytest.approx(0.01)


def test_converters_round_trip() -> None:
    for tf in (1e-6, 1e-4, 1e-2, 0.5):
        assert tf_from_vaf(vaf_from_tf(tf)) == pytest.approx(tf)


def test_ploidy_and_copies() -> None:
    # Two altered copies out of four -> VAF = TF/2 still.
    assert vaf_from_tf(0.02, copies_altered=2, ploidy=4) == pytest.approx(0.01)
    # One copy out of three.
    assert vaf_from_tf(0.03, copies_altered=1, ploidy=3) == pytest.approx(0.01)


def test_ppm_scale_no_precision_loss() -> None:
    # 10 ppm TF -> 5 ppm VAF, exactly.
    assert vaf_from_tf(10e-6) == pytest.approx(5e-6, rel=1e-12)
    assert not math.isnan(vaf_from_tf(1e-9))
