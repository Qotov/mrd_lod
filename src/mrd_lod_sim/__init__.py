"""mrd-lod-sim: a limit-of-detection simulator and design tool for
tumour-informed ctDNA MRD assays.

See ``docs/mrd-lod-sim_BUILD_SPEC.md`` for the scientific model and the
architectural contracts (the detection rule is the production caller; the
validation layer is data-source agnostic; every assay assumption is a
parameter, never a constant).
"""

from mrd_lod_sim.molecules import MoleculeBudget, molecule_budget
from mrd_lod_sim.params import (
    PG_PER_GENOME_EQUIVALENT,
    MoleculeParams,
    tf_from_vaf,
    vaf_from_tf,
)

__all__ = [
    "PG_PER_GENOME_EQUIVALENT",
    "MoleculeParams",
    "MoleculeBudget",
    "molecule_budget",
    "vaf_from_tf",
    "tf_from_vaf",
]
