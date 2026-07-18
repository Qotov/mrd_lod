"""Calibration hooks -- the "later" path (BUILD_SPEC 6).

These make the tool durable: the same code that runs on literature priors today
consumes real data tomorrow. Interfaces are defined now with synthetic
round-trip tests, even though real data does not yet exist.

- :func:`estimate_error_model` -- healthy-donor counts -> empirical error model,
  replacing the literature prior (BUILD_SPEC 6).
- :func:`fit_conversion_efficiency` -- recover the real conversion efficiency
  from QC data.
- :func:`compare_predicted_observed` -- the diagnostic: overlay the model's
  predicted detection curve on observed results and hint at the likely cause of
  any discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrd_lod_sim.analytic import detection_probability
from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import DetectionRule
from mrd_lod_sim.errors import ContextualError, EmpiricalError
from mrd_lod_sim.params import PG_PER_GENOME_EQUIVALENT
from mrd_lod_sim.validate import DetectionRecord

__all__ = [
    "estimate_error_model",
    "estimate_contextual_error",
    "fit_conversion_efficiency",
    "Diagnostic",
    "compare_predicted_observed",
]


def estimate_error_model(
    mutant_counts: np.ndarray,
    total_counts: np.ndarray,
    contexts: list[str] | None = None,
) -> EmpiricalError:
    """Fit an :class:`EmpiricalError` from healthy-donor counts (BUILD_SPEC 6).

    Replaces the literature prior with an empirical per-site rate distribution.
    ``mean_rate()`` on the result is the pooled background rate.
    """
    return EmpiricalError(
        mutant_counts=np.asarray(mutant_counts, dtype=float),
        total_counts=np.asarray(total_counts, dtype=float),
        contexts=contexts,
    )


def estimate_contextual_error(
    mutant_counts: np.ndarray,
    total_counts: np.ndarray,
    contexts: list[str],
) -> ContextualError:
    """Per-trinucleotide-context error rates from healthy-donor data (BUILD_SPEC 6)."""
    mut = np.asarray(mutant_counts, dtype=float)
    tot = np.asarray(total_counts, dtype=float)
    ctx = np.asarray(contexts)
    if not (mut.shape == tot.shape == ctx.shape):
        raise ValueError("counts and contexts must align")
    rates: dict[str, float] = {}
    weights: dict[str, float] = {}
    total_depth = float(tot.sum())
    for c in np.unique(ctx):
        m = ctx == c
        rates[str(c)] = float(mut[m].sum() / tot[m].sum())
        weights[str(c)] = float(tot[m].sum() / total_depth)
    return ContextualError(context_rates=rates, context_weights=weights)


def fit_conversion_efficiency(
    input_masses: np.ndarray,
    observed_unique_molecules: np.ndarray,
    pg_per_genome_equivalent: float = PG_PER_GENOME_EQUIVALENT,
    strand_recovery: float = 1.0,
) -> float:
    """Recover conversion efficiency from QC data (BUILD_SPEC 6).

    ``observed_unique = GE_total * conversion_efficiency * strand_recovery`` with
    ``GE_total = input_ng / pg_per_genome_equivalent``. Fits the through-origin
    slope by least squares and divides out the (known) strand recovery.
    """
    masses = np.asarray(input_masses, dtype=float)
    obs = np.asarray(observed_unique_molecules, dtype=float)
    if masses.shape != obs.shape:
        raise ValueError("input_masses and observed_unique_molecules must align")
    ge_total = masses / pg_per_genome_equivalent
    # Through-origin OLS slope = sum(x*y) / sum(x^2).
    slope = float(np.sum(ge_total * obs) / np.sum(ge_total**2))
    return slope / strand_recovery


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Predicted-vs-observed comparison with a ranked cause hint (BUILD_SPEC 6)."""

    levels: np.ndarray
    predicted: np.ndarray  # model P(detect) at each level
    observed: np.ndarray  # empirical detection fraction at each level
    blank_false_positive: float | None  # observed detection at level ~ 0, if any
    mean_signed_discrepancy: float  # mean(observed - predicted)
    likely_causes: list[str]  # ranked, most likely first


def compare_predicted_observed(
    config: AssayConfig,
    rule: DetectionRule,
    observed_records: list[DetectionRecord],
) -> Diagnostic:
    """Overlay predicted vs observed detection and hint at the cause (BUILD_SPEC 6).

    The hint is heuristic and ranked: sensitivity loss across levels points to
    conversion efficiency below assumption or correlated dropout; a rightward
    shift with intact high-level detection points to a higher error floor;
    elevated detection on blanks points to contamination.
    """
    levels = np.array([r.level for r in observed_records], dtype=float)
    observed = np.array(
        [r.n_detections / r.n_replicates for r in observed_records], dtype=float
    )
    predicted = np.array(
        [detection_probability(config, rule, lvl) for lvl in levels]
    )
    disc = observed - predicted
    mean_disc = float(np.mean(disc))

    # Blank behaviour: any near-zero level acts as a contamination probe.
    blank_fp: float | None = None
    blank_mask = levels <= levels.min() * 1.0 if levels.size else np.array([])
    # Treat the smallest level as the blank proxy only if it is very small.
    if levels.size and levels.min() <= 1e-7:
        blank_fp = float(observed[np.argmin(levels)])

    high_mask = predicted >= 0.9
    high_gap = float(np.mean((predicted - observed)[high_mask])) if high_mask.any() else 0.0

    causes: list[str] = []
    if blank_fp is not None and blank_fp > 0.10:
        causes.append(
            "contamination or a mis-calibrated threshold: blanks are detected far "
            "above the assumed specificity"
        )
    if high_gap > 0.10:
        causes.append(
            "conversion efficiency below assumption or correlated site dropout: "
            "sensitivity is lost even where the model predicts near-certain detection"
        )
    if mean_disc < -0.10 and high_gap <= 0.10:
        causes.append(
            "error floor higher than the consensus regime implies: the detection "
            "curve is shifted toward higher VAF without losing high-level detection"
        )
    if not causes:
        causes.append("no material discrepancy: observed tracks predicted")

    return Diagnostic(
        levels=levels,
        predicted=predicted,
        observed=observed,
        blank_false_positive=blank_fp,
        mean_signed_discrepancy=mean_disc,
        likely_causes=causes,
    )
