"""Panel heterogeneity model (BUILD_SPEC 2.5).

Not every tracked variant behaves identically. A ``PanelModel`` carries:

- a **CCF distribution** across tracked variants (default: mostly clonal with a
  subclonal tail, a Beta skewed toward 1.0). This links the simulator to panel
  *selection*: a panel of truncal variants beats one with subclonal variants at
  the same N.
- an optional **per-site dropout probability** (probe failure / poor capture).
- an optional **depth dispersion** across sites (negative-binomial rather than a
  fixed effective depth).

The two optional features put the config *outside the analytic-valid regime*
(BUILD_SPEC 4.2); ``analytic.py`` raises rather than silently approximating.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PanelModel:
    """A panel of ``n_variants`` tracked sites.

    Attributes:
        n_variants: number of tracked variants (N).
        ccf_alpha: Beta ``a`` for the CCF distribution, or ``None`` for a fully
            clonal panel (CCF == 1 at every site).
        ccf_beta: Beta ``b`` for the CCF distribution.
        dropout_prob: per-site probability of dropout (probe failure). 0 keeps
            the config in the analytic-valid regime.
        depth_dispersion: negative-binomial dispersion of per-site depth, or
            ``None`` for fixed depth. ``None`` keeps the analytic-valid regime.
        context_weights: optional trinucleotide composition, consumed by
            :class:`~mrd_lod_sim.errors.ContextualError` when computing E[eps].
    """

    n_variants: int
    ccf_alpha: float | None = 9.0
    ccf_beta: float = 1.0
    dropout_prob: float = 0.0
    depth_dispersion: float | None = None
    context_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.n_variants < 1:
            raise ValueError("n_variants must be >= 1")
        if self.ccf_alpha is not None and (self.ccf_alpha <= 0 or self.ccf_beta <= 0):
            raise ValueError("ccf Beta parameters must be positive")
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError("dropout_prob must be in [0, 1)")
        if self.depth_dispersion is not None and self.depth_dispersion <= 0:
            raise ValueError("depth_dispersion must be positive when set")

    @classmethod
    def clonal(cls, n_variants: int, **kwargs) -> "PanelModel":
        """A panel where every tracked variant is fully clonal (CCF == 1)."""
        return cls(n_variants=n_variants, ccf_alpha=None, **kwargs)

    @property
    def is_analytic_valid(self) -> bool:
        """Whether this panel stays inside the analytic-valid regime (4.2)."""
        return self.dropout_prob == 0.0 and self.depth_dispersion is None

    def mean_ccf(self) -> float:
        """E[CCF] across tracked variants."""
        if self.ccf_alpha is None:
            return 1.0
        return self.ccf_alpha / (self.ccf_alpha + self.ccf_beta)

    def sample_ccf(
        self, shape: int | tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """Draw per-site CCF values."""
        if self.ccf_alpha is None:
            return np.ones(shape, dtype=float)
        return rng.beta(self.ccf_alpha, self.ccf_beta, size=shape)
