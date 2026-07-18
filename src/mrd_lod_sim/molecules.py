"""The molecule budget (BUILD_SPEC 2.1).

Converts an input cfDNA mass into the number of *effective* genome equivalents
actually interrogated by the assay -- the quantity that ultimately sets the
Poisson sampling floor on detection.

    GE_total = input_ng / pg_per_genome_equivalent
    GE_eff   = GE_total * conversion_efficiency * strand_recovery
"""

from __future__ import annotations

from dataclasses import dataclass

from mrd_lod_sim.params import MoleculeParams


@dataclass(frozen=True, slots=True)
class MoleculeBudget:
    """Result of :func:`molecule_budget`.

    Attributes:
        ge_total: genome equivalents present in the input mass.
        ge_eff: effective genome equivalents recovered as usable molecules
            after conversion efficiency and strand recovery. This is the
            per-site sampling depth in the Poisson model (BUILD_SPEC 2.3).
    """

    ge_total: float
    ge_eff: float


def molecule_budget(params: MoleculeParams) -> MoleculeBudget:
    """Compute the molecule budget from assay input parameters."""
    ge_total = params.input_ng / params.pg_per_genome_equivalent
    ge_eff = ge_total * params.conversion_efficiency * params.strand_recovery
    return MoleculeBudget(ge_total=ge_total, ge_eff=ge_eff)
