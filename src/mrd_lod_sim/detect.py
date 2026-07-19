"""Detection rules -- the production caller (BUILD_SPEC 3).

This module is the seam between the simulator and a real pipeline. A
``DetectionRule`` accepts only real per-site observations
``(mutant_count, total_count, error_rate)`` via :class:`SiteObservations` --
never simulation-internal state. Simulation is one producer of that input; a
wet-lab pipeline is another, with no code change here.

Per-site background rates are resolved as follows (reconciling BUILD_SPEC 0 and
3): if ``SiteObservations.error_rates`` is supplied (the real-pipeline path,
where each site has a measured background estimate) the rule uses it directly;
otherwise it falls back to ``error_model.mean_rate()`` broadcast across sites
(the prior path). The ``error_model`` argument is thus both the prior and the
provider of the analytic expectation.

Two rules are implemented and comparing them is a first-class feature:

- :class:`KofNRule`          -- Signatera-style; intuitive, statistically
  inefficient; does **not** estimate a VAF (``estimated_vaf`` stays ``None``,
  per BUILD_SPEC 3).
- :class:`AggregatePoissonRule` -- efficient, closed form (BUILD_SPEC 4.2).

Every rule can operate at a **calibrated threshold** on its ``statistic``
(higher = more evidence for MRD) derived from blanks (BUILD_SPEC 5), rather than
a nominal alpha -- because the empirical false-positive rate is what matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, chi2, poisson

from mrd_lod_sim.errors import ErrorModel

__all__ = [
    "SiteObservations",
    "MRDCall",
    "DetectionRule",
    "KofNRule",
    "AggregatePoissonRule",
]


@dataclass(frozen=True, slots=True)
class SiteObservations:
    """Per-site observations -- exactly what a real pipeline emits (BUILD_SPEC 3).

    Attributes:
        mutant_counts: mutant molecule count per tracked site.
        total_counts: total molecule count (depth) per tracked site.
        site_ids: optional per-site identifiers.
        error_rates: optional per-site background rate estimates. When present,
            rules use these directly (real-pipeline path); when ``None`` they
            fall back to the passed ``error_model``'s ``mean_rate()``.
    """

    mutant_counts: np.ndarray
    total_counts: np.ndarray
    site_ids: np.ndarray | None = None
    error_rates: np.ndarray | None = None

    def __post_init__(self) -> None:
        mut = np.asarray(self.mutant_counts, dtype=float)
        tot = np.asarray(self.total_counts, dtype=float)
        if mut.shape != tot.shape:
            raise ValueError("mutant_counts and total_counts must have equal shape")
        if mut.ndim != 1:
            raise ValueError("observations must be 1-D (one entry per site)")
        if np.any(tot < 0) or np.any(mut < 0):
            raise ValueError("counts must be non-negative")
        if np.any(mut > tot):
            raise ValueError("mutant_counts cannot exceed total_counts")
        object.__setattr__(self, "mutant_counts", mut)
        object.__setattr__(self, "total_counts", tot)
        if self.error_rates is not None:
            er = np.asarray(self.error_rates, dtype=float)
            if er.shape != mut.shape:
                raise ValueError("error_rates must align with observations")
            if np.any(er < 0):
                raise ValueError("error_rates must be non-negative")
            object.__setattr__(self, "error_rates", er)

    @property
    def n_sites(self) -> int:
        return int(self.mutant_counts.size)


@dataclass(frozen=True, slots=True)
class MRDCall:
    """Result of a detection rule (BUILD_SPEC 3).

    ``estimated_vaf`` / ``estimated_ci`` are optional (``None``) because not
    every rule estimates a level -- ``KofNRule`` only counts significant sites.
    Any reported VAF is **per-site VAF** (BUILD_SPEC 2.2); convert to tumour
    fraction only through ``params.tf_from_vaf``.
    """

    detected: bool
    statistic: float
    p_value: float
    estimated_vaf: float | None = None
    estimated_ci: tuple[float, float] | None = None
    n_positive_sites: int | None = None
    per_site: dict[str, np.ndarray] | None = None


class DetectionRule(ABC):
    """Abstract base for detection rules."""

    #: Optional calibrated threshold on ``statistic``. When set, the sample is
    #: called positive iff ``statistic >= decision_threshold``, overriding the
    #: rule's nominal-alpha decision (BUILD_SPEC 3, 5).
    decision_threshold: float | None = None

    @abstractmethod
    def call(
        self, observations: SiteObservations, error_model: ErrorModel
    ) -> MRDCall:
        """Score observations and return an :class:`MRDCall`."""

    def _site_rates(
        self, observations: SiteObservations, error_model: ErrorModel
    ) -> np.ndarray:
        if observations.error_rates is not None:
            return observations.error_rates
        return np.full(observations.n_sites, error_model.mean_rate())


def _poisson_count_ci(
    k: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Exact (Garwood) Poisson confidence interval for a count ``k``."""
    lo = 0.0 if k <= 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k)
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1))
    return float(lo), float(hi)


@dataclass(frozen=True, slots=True)
class KofNRule(DetectionRule):
    """Signatera-style k-of-N rule (BUILD_SPEC 3).

    A site is positive if its mutant count is significant against its own
    background Poisson; the sample is positive if >= ``k`` sites are positive.
    Intuitive and robust, statistically inefficient. Does not estimate a VAF.

    Attributes:
        k: minimum number of positive sites for a positive sample.
        per_site_alpha: per-site significance level for calling a site positive.
        decision_threshold: optional calibrated threshold on the number of
            positive sites (overrides ``k`` when set).
    """

    k: int = 2
    per_site_alpha: float = 0.01
    decision_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 < self.per_site_alpha < 1.0:
            raise ValueError("per_site_alpha must be in (0, 1)")

    def call(
        self, observations: SiteObservations, error_model: ErrorModel
    ) -> MRDCall:
        eps = self._site_rates(observations, error_model)
        lam_bg = observations.total_counts * eps
        # P(X >= mutant_i | background only).
        p_site = poisson.sf(observations.mutant_counts - 1, lam_bg)
        positive = p_site <= self.per_site_alpha
        n_pos = int(np.count_nonzero(positive))
        statistic = float(n_pos)
        threshold = self.decision_threshold if self.decision_threshold is not None else self.k
        detected = statistic >= threshold
        # Sample-level p-value: P(>= n_pos positive sites | all background).
        # Number of false-positive sites ~ Binomial(N, per_site_alpha) under H0.
        n = observations.n_sites
        sample_p = float(binom.sf(n_pos - 1, n, self.per_site_alpha)) if n_pos > 0 else 1.0
        return MRDCall(
            detected=bool(detected),
            statistic=statistic,
            p_value=sample_p,
            estimated_vaf=None,
            estimated_ci=None,
            n_positive_sites=n_pos,
            per_site={"p_value": p_site, "positive": positive},
        )


@dataclass(frozen=True, slots=True)
class AggregatePoissonRule(DetectionRule):
    """Aggregate Poisson rule (BUILD_SPEC 3, 4.2).

    Total mutant molecules across the panel tested against total expected
    background. Efficient, closed form, and the rule the analytic fast path
    represents exactly.

    Attributes:
        alpha: nominal false-positive rate for the count threshold.
        decision_threshold: optional calibrated count threshold on total mutant
            molecules (overrides the alpha-derived threshold when set).
    """

    alpha: float = 0.05
    decision_threshold: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    def call(
        self, observations: SiteObservations, error_model: ErrorModel
    ) -> MRDCall:
        eps = self._site_rates(observations, error_model)
        lam_bg = float(np.sum(observations.total_counts * eps))
        total_mutant = float(np.sum(observations.mutant_counts))
        total_depth = float(np.sum(observations.total_counts))
        # P(X >= total_mutant | Poisson(lam_bg)).
        p_value = float(poisson.sf(total_mutant - 1, lam_bg))
        if self.decision_threshold is not None:
            detected = total_mutant >= self.decision_threshold
        else:
            detected = p_value <= self.alpha
        # Background-subtracted VAF estimate with an exact Poisson CI.
        vaf_hat = max(0.0, (total_mutant - lam_bg) / total_depth) if total_depth else 0.0
        lo, hi = _poisson_count_ci(total_mutant, alpha=0.05)
        ci = (
            max(0.0, (lo - lam_bg) / total_depth) if total_depth else 0.0,
            max(0.0, (hi - lam_bg) / total_depth) if total_depth else 0.0,
        )
        return MRDCall(
            detected=bool(detected),
            statistic=total_mutant,
            p_value=p_value,
            estimated_vaf=vaf_hat,
            estimated_ci=ci,
        )
