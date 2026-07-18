"""Monte Carlo generative core (BUILD_SPEC 4.1).

Fully vectorised with numpy -- draws arrays of shape ``(n_replicates,
n_variants)``, never Python loops over replicates for the *drawing*. A seeded
``numpy.random.Generator`` gives reproducibility (BUILD_SPEC 10, determinism).

Per-site sampling model (BUILD_SPEC 2.3), for tracked site ``i`` at per-site
``vaf``::

    vaf_i        = vaf * ccf_i                       # ccf_i from the panel
    depth_i      = GE_eff  (optionally dispersed)
    mutant_i     ~ Poisson(depth_i * vaf_i)          # true signal
    background_i ~ Poisson(depth_i * eps_i)          # eps_i from the error model
    observed_i   = mutant_i + background_i

Poisson (not Binomial) because ``vaf << 1`` -- valid below ~1% VAF.

This module is one *producer* of :class:`~mrd_lod_sim.detect.SiteObservations`;
the detection rules (the production caller) are applied to that output exactly as
they would be to a real pipeline's output. :func:`iter_observations` exposes the
per-replicate objects; :func:`detection_rate` applies a rule across replicates,
vectorising the two closed-form rules and looping the rule call for the rest.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import (
    AggregatePoissonRule,
    DetectionRule,
    KofNRule,
    SiteObservations,
)

__all__ = ["SimulatedCounts", "simulate_counts", "iter_observations", "detection_rate"]


@dataclass(frozen=True, slots=True)
class SimulatedCounts:
    """Raw draws from :func:`simulate_counts`.

    Shapes are ``(n_replicates, n_variants)``.
    """

    mutant: np.ndarray  # observed mutant molecule counts (signal + background)
    total: np.ndarray  # per-site depth (effective molecules)


def simulate_counts(
    config: AssayConfig, vaf: float, n_replicates: int, rng: np.random.Generator
) -> SimulatedCounts:
    """Draw ``(n_replicates, n_variants)`` per-site observations at per-site ``vaf``."""
    ge_eff = config.ge_eff()
    n = config.panel.n_variants
    shape = (n_replicates, n)

    ccf = config.panel.sample_ccf(shape, rng)
    eps = config.error_model.sample(shape, rng)

    if config.panel.depth_dispersion is None:
        total = np.full(shape, ge_eff, dtype=float)
    else:
        # Gamma-mixed depth with mean GE_eff (negative-binomial-like dispersion).
        k = 1.0 / config.panel.depth_dispersion
        total = rng.gamma(shape=k, scale=ge_eff / k, size=shape)

    if config.panel.dropout_prob > 0.0:
        dropped = rng.random(shape) < config.panel.dropout_prob
        total = np.where(dropped, 0.0, total)

    signal = rng.poisson(total * vaf * ccf)
    background = rng.poisson(total * eps)
    mutant = np.minimum(signal + background, np.floor(total)).astype(float)
    return SimulatedCounts(mutant=mutant, total=total)


def iter_observations(
    config: AssayConfig, vaf: float, n_replicates: int, rng: np.random.Generator
) -> Iterator[SiteObservations]:
    """Yield one :class:`SiteObservations` per replicate (the production input)."""
    counts = simulate_counts(config, vaf, n_replicates, rng)
    for r in range(n_replicates):
        yield SiteObservations(
            mutant_counts=counts.mutant[r], total_counts=counts.total[r]
        )


@dataclass(frozen=True, slots=True)
class DetectionOutcome:
    """Result of :func:`detection_rate`."""

    detection_rate: float
    detected: np.ndarray  # per-replicate bool
    statistic: np.ndarray  # per-replicate statistic (higher = more evidence)


def detection_rate(
    config: AssayConfig,
    rule: DetectionRule,
    vaf: float,
    n_replicates: int,
    rng: np.random.Generator,
) -> DetectionOutcome:
    """Estimate P(detect) at per-site ``vaf`` by Monte Carlo.

    The two closed-form rules are evaluated vectorised (identical decision to
    :meth:`DetectionRule.call`, which the equivalence test in ``tests`` pins);
    other rules fall back to a per-replicate ``rule.call``.
    """
    counts = simulate_counts(config, vaf, n_replicates, rng)
    mutant, total = counts.mutant, counts.total
    eps_est = config.mean_error_rate()  # what the caller knows (BUILD_SPEC 3)

    if isinstance(rule, AggregatePoissonRule):
        total_mutant = mutant.sum(axis=1)
        lam_bg = eps_est * total.sum(axis=1)
        if rule.decision_threshold is not None:
            detected = total_mutant >= rule.decision_threshold
        else:
            detected = poisson.sf(total_mutant - 1, lam_bg) <= rule.alpha
        statistic = total_mutant
    elif isinstance(rule, KofNRule):
        lam_bg_site = total * eps_est
        p_site = poisson.sf(mutant - 1, lam_bg_site)
        n_pos = (p_site <= rule.per_site_alpha).sum(axis=1)
        threshold = (
            rule.decision_threshold if rule.decision_threshold is not None else rule.k
        )
        detected = n_pos >= threshold
        statistic = n_pos.astype(float)
    else:
        detected = np.empty(n_replicates, dtype=bool)
        statistic = np.empty(n_replicates, dtype=float)
        for r in range(n_replicates):
            obs = SiteObservations(
                mutant_counts=mutant[r], total_counts=total[r]
            )
            call = rule.call(obs, config.error_model)
            detected[r] = call.detected
            statistic[r] = call.statistic

    detected = np.asarray(detected, dtype=bool)
    return DetectionOutcome(
        detection_rate=float(detected.mean()),
        detected=detected,
        statistic=statistic,
    )
