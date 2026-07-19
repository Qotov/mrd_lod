"""Analytic fast path (BUILD_SPEC 4.2).

Detection probability in microseconds instead of seconds, making the LoD surface
(BUILD_SPEC 7) and live dashboard (BUILD_SPEC 8) tractable.

One entry point, :func:`detection_probability`, dispatches by rule:

- ``AggregatePoissonRule`` -- exact: the sum of independent Poissons is Poisson.
- ``KofNRule`` -- exact under panel-expectation inputs: per-site hit probability
  is a Poisson tail; the sample is positive with probability ``P(K >= k)`` for
  ``K ~ Binomial(N, p)`` (the homogeneous reduction of the exact Poisson
  binomial).

All inputs come from ``mean_rate()`` and panel expectations -- never a median or
a per-replicate draw. The closed forms represent only a restricted model, so the
**analytic-valid regime is enforced**: :func:`detection_probability` raises on
any config it cannot faithfully represent (depth dispersion or dropout).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom, poisson

from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import (
    AggregatePoissonRule,
    DetectionRule,
    KofNRule,
)

__all__ = [
    "detection_probability",
    "AnalyticRegimeError",
    "count_threshold",
]


class AnalyticRegimeError(ValueError):
    """Raised when a config lies outside the analytic-valid regime (4.2)."""


def _require_analytic_regime(config: AssayConfig) -> None:
    if config.panel.depth_dispersion is not None:
        raise AnalyticRegimeError(
            "depth dispersion is outside the analytic-valid regime (BUILD_SPEC 4.2); "
            "use the Monte Carlo path"
        )
    if config.panel.dropout_prob != 0.0:
        raise AnalyticRegimeError(
            "per-site dropout is outside the analytic-valid regime (BUILD_SPEC 4.2); "
            "use the Monte Carlo path"
        )


def count_threshold(lam_bg: float, alpha: float) -> int:
    """Smallest integer count ``t`` with ``P(X >= t | Poisson(lam_bg)) <= alpha``.

    This is the calibrated positivity threshold for a Poisson background at a
    nominal false-positive rate ``alpha``.
    """
    # isf gives an integer near the target; nudge to satisfy the inequality
    # exactly (guards against float/rounding at the boundary).
    t = int(max(1, np.ceil(poisson.isf(alpha, lam_bg))))
    while t > 1 and poisson.sf(t - 2, lam_bg) <= alpha:
        t -= 1
    while poisson.sf(t - 1, lam_bg) > alpha:
        t += 1
    return t


def _aggregate_detection_probability(
    ge_eff: float, n: int, e_ccf: float, mean_eps: float, vaf: float,
    alpha: float, decision_threshold: float | None,
) -> float:
    lam_bg = ge_eff * n * mean_eps
    lam_sig = ge_eff * n * vaf * e_ccf
    if decision_threshold is not None:
        t = decision_threshold
    else:
        t = count_threshold(lam_bg, alpha)
    return float(poisson.sf(t - 1, lam_bg + lam_sig))


def _kofn_detection_probability(
    ge_eff: float, n: int, e_ccf: float, mean_eps: float, vaf: float,
    rule: KofNRule,
) -> float:
    lam_site_bg = ge_eff * mean_eps
    lam_site_sig = ge_eff * vaf * e_ccf
    # Per-site positivity threshold at per_site_alpha against its own background.
    t_site = count_threshold(lam_site_bg, rule.per_site_alpha)
    p_site = float(poisson.sf(t_site - 1, lam_site_bg + lam_site_sig))
    k = rule.decision_threshold if rule.decision_threshold is not None else rule.k
    # P(K >= k) for K ~ Binomial(N, p_site).
    return float(binom.sf(int(np.ceil(k)) - 1, n, p_site))


def detection_probability(
    config: AssayConfig, rule: DetectionRule, vaf: float
) -> float:
    """Probability of detection at per-site ``vaf`` for ``rule`` under ``config``.

    Raises:
        AnalyticRegimeError: if the config uses depth dispersion or dropout.
    """
    _require_analytic_regime(config)
    ge_eff = config.ge_eff()
    n = config.panel.n_variants
    e_ccf = config.panel.mean_ccf()
    mean_eps = config.mean_error_rate()

    if isinstance(rule, AggregatePoissonRule):
        return _aggregate_detection_probability(
            ge_eff, n, e_ccf, mean_eps, vaf, rule.alpha, rule.decision_threshold
        )
    if isinstance(rule, KofNRule):
        return _kofn_detection_probability(ge_eff, n, e_ccf, mean_eps, vaf, rule)
    raise TypeError(f"no analytic path for rule type {type(rule).__name__}")
