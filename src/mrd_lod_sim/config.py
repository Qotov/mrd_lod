"""The assay configuration -- one object bundling every assay assumption.

Every scientific assumption is a parameter, never a constant (BUILD_SPEC 0), so
both computation paths (analytic and Monte Carlo) and the surface/dashboard read
from a single :class:`AssayConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mrd_lod_sim.errors import ErrorModel
from mrd_lod_sim.molecules import molecule_budget
from mrd_lod_sim.panel import PanelModel
from mrd_lod_sim.params import MoleculeParams


@dataclass(frozen=True, slots=True)
class AssayConfig:
    """A complete assay configuration.

    Attributes:
        molecules: molecule-budget inputs (input mass, conversion, strand).
        panel: panel size and heterogeneity model.
        error_model: per-site background error distribution.
    """

    molecules: MoleculeParams
    panel: PanelModel
    error_model: ErrorModel

    def ge_eff(self) -> float:
        """Effective genome equivalents interrogated (per site depth)."""
        return molecule_budget(self.molecules).ge_eff

    def mean_error_rate(self) -> float:
        """Analytic background expectation E[eps], panel-context aware."""
        return self.error_model.mean_rate(self.panel)
