"""Validation layer -- data-source agnostic (BUILD_SPEC 5).

Every function here consumes summary records, never simulation-internal state,
so the same code fits LoB/LoD from *simulated* replicates today and from
*wet-lab* replicates later with no change. **This module must not import
``simulate.py``** (enforced by a test).

Two input shapes appear, both rule-agnostic:

- detection records ``(level, n_replicates, n_detections)`` -- for the detection
  curve, LoD, and power analysis;
- raw ``blank_statistics`` arrays (the detection statistic on blanks) -- for LoB
  and threshold calibration. These are produced by whichever rule you ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

__all__ = [
    "DetectionRecord",
    "DetectionCurve",
    "fit_detection_curve",
    "lod_from_curve",
    "LimitOfBlank",
    "limit_of_blank",
    "limit_of_quantification",
    "calibrate_threshold",
    "power_analysis",
]


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """One dilution level's outcome (BUILD_SPEC 5).

    Attributes:
        level: the concentration (per-site VAF, unless the caller documents
            otherwise -- the fitter works in ``log10(level)``).
        n_replicates: replicates tested at this level.
        n_detections: replicates called positive.
    """

    level: float
    n_replicates: int
    n_detections: int

    def __post_init__(self) -> None:
        if self.level <= 0:
            raise ValueError("level must be positive (log10 is taken)")
        if self.n_replicates <= 0:
            raise ValueError("n_replicates must be positive")
        if not 0 <= self.n_detections <= self.n_replicates:
            raise ValueError("n_detections must be in [0, n_replicates]")


@dataclass(frozen=True, slots=True)
class DetectionCurve:
    """A fitted probit/logistic detection curve in ``log10(level)``.

    ``p(level) = link(intercept + slope * log10(level))``.
    """

    intercept: float
    slope: float
    link: str  # "probit" or "logit"

    def predict(self, level: float | np.ndarray) -> np.ndarray:
        x = np.log10(np.asarray(level, dtype=float))
        z = self.intercept + self.slope * x
        if self.link == "probit":
            return norm.cdf(z)
        return 1.0 / (1.0 + np.exp(-z))

    def level_at(self, hit_rate: float) -> float:
        """Level achieving ``hit_rate`` detection (inverse of the curve)."""
        if not 0.0 < hit_rate < 1.0:
            raise ValueError("hit_rate must be in (0, 1)")
        if self.slope <= 0:
            return float("nan")
        z = norm.ppf(hit_rate) if self.link == "probit" else np.log(hit_rate / (1 - hit_rate))
        return float(10.0 ** ((z - self.intercept) / self.slope))


def _neg_loglik(
    params: np.ndarray, x: np.ndarray, k: np.ndarray, n: np.ndarray, link: str
) -> float:
    z = params[0] + params[1] * x
    if link == "probit":
        p = norm.cdf(z)
    else:
        p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.sum(k * np.log(p) + (n - k) * np.log1p(-p)))


def fit_detection_curve(
    records: list[DetectionRecord], link: str = "probit"
) -> DetectionCurve:
    """Fit detection probability vs ``log10(level)`` (BUILD_SPEC 5.1).

    Args:
        records: at least two distinct levels.
        link: "probit" (default) or "logit".
    """
    if link not in ("probit", "logit"):
        raise ValueError("link must be 'probit' or 'logit'")
    levels = np.array([r.level for r in records], dtype=float)
    if np.unique(levels).size < 2:
        raise ValueError("need at least two distinct levels to fit a curve")
    x = np.log10(levels)
    k = np.array([r.n_detections for r in records], dtype=float)
    n = np.array([r.n_replicates for r in records], dtype=float)

    # Initial guess: slope positive, centre near the median level.
    x0 = np.array([-np.median(x), 1.0])
    res = minimize(
        _neg_loglik, x0, args=(x, k, n, link), method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 10_000},
    )
    return DetectionCurve(intercept=float(res.x[0]), slope=float(res.x[1]), link=link)


@dataclass(frozen=True, slots=True)
class LoDResult:
    """LoD point estimate with a bootstrap confidence interval (BUILD_SPEC 5.2)."""

    lod: float
    ci_low: float
    ci_high: float
    hit_rate: float


def lod_from_curve(
    records: list[DetectionRecord],
    hit_rate: float = 0.95,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    link: str = "probit",
    rng: np.random.Generator | None = None,
) -> LoDResult:
    """Level at ``hit_rate`` detection, with a bootstrap CI (BUILD_SPEC 5.2).

    Never returns a bare point estimate: the CI is resampled by redrawing each
    level's detections from ``Binomial(n_replicates, p_hat)`` and refitting.
    """
    rng = rng or np.random.default_rng()
    point = fit_detection_curve(records, link=link).level_at(hit_rate)

    boot = np.empty(n_bootstrap)
    n_arr = np.array([r.n_replicates for r in records])
    p_hat = np.array([r.n_detections / r.n_replicates for r in records])
    for b in range(n_bootstrap):
        k_star = rng.binomial(n_arr, p_hat)
        recs = [
            DetectionRecord(r.level, r.n_replicates, int(kk))
            for r, kk in zip(records, k_star)
        ]
        try:
            boot[b] = fit_detection_curve(recs, link=link).level_at(hit_rate)
        except ValueError:
            boot[b] = np.nan
    boot = boot[np.isfinite(boot)]
    tail = (1 - ci) / 2
    lo = float(np.quantile(boot, tail)) if boot.size else float("nan")
    hi = float(np.quantile(boot, 1 - tail)) if boot.size else float("nan")
    return LoDResult(lod=float(point), ci_low=lo, ci_high=hi, hit_rate=hit_rate)


@dataclass(frozen=True, slots=True)
class LimitOfBlank:
    """LoB by two methods, with a divergence warning (BUILD_SPEC 5.3)."""

    nonparametric: float  # (1-alpha) percentile of the blanks
    clsi_parametric: float  # mean + z * SD
    alpha: float
    diverges: bool  # methods disagree -> normality likely violated
    message: str


def limit_of_blank(
    blank_statistics: np.ndarray, alpha: float = 0.05, rel_tol: float = 0.25
) -> LimitOfBlank:
    """LoB from a blank statistic distribution (BUILD_SPEC 5.3).

    Returns both the non-parametric ``(1-alpha)`` percentile and the CLSI
    parametric ``mean + z_(1-alpha) * SD``, and warns when they diverge -- a
    normality-violation signal, common for count statistics at ppm scale.
    """
    x = np.asarray(blank_statistics, dtype=float)
    if x.size == 0:
        raise ValueError("blank_statistics must be non-empty")
    z = norm.ppf(1 - alpha)
    nonparam = float(np.percentile(x, 100 * (1 - alpha)))
    clsi = float(x.mean() + z * x.std(ddof=1)) if x.size > 1 else float(x.mean())
    denom = max(abs(nonparam), 1e-9)
    diverges = abs(nonparam - clsi) / denom > rel_tol
    msg = (
        "non-parametric and CLSI parametric LoB diverge; the blank statistic is "
        "likely non-normal (common for count statistics at ppm scale) -- prefer "
        "the non-parametric percentile"
        if diverges
        else "methods agree"
    )
    return LimitOfBlank(nonparam, clsi, alpha, diverges, msg)


def limit_of_quantification(
    level_estimates: dict[float, np.ndarray], max_cv: float = 0.20
) -> float | None:
    """Lowest level meeting a precision criterion (BUILD_SPEC 5.4).

    Args:
        level_estimates: level -> array of quantitative estimates (e.g. estimated
            VAF) across replicates at that level.
        max_cv: maximum acceptable coefficient of variation (SD / mean).

    Returns:
        The lowest level with ``CV <= max_cv``, or ``None`` if none qualifies.
    """
    qualifying = []
    for level, ests in level_estimates.items():
        arr = np.asarray(ests, dtype=float)
        if arr.size < 2 or arr.mean() <= 0:
            continue
        cv = arr.std(ddof=1) / arr.mean()
        if cv <= max_cv:
            qualifying.append(level)
    return min(qualifying) if qualifying else None


def calibrate_threshold(
    blank_statistics: np.ndarray, target_specificity: float
) -> float:
    """Statistic threshold achieving ``target_specificity`` on blanks (BUILD_SPEC 5.5).

    A sample is called positive iff ``statistic >= threshold``, so specificity is
    ``P(statistic < threshold | blank)``. **Run this before LoD determination**;
    report the LoD alongside the specificity it assumes.
    """
    if not 0.0 < target_specificity < 1.0:
        raise ValueError("target_specificity must be in (0, 1)")
    x = np.asarray(blank_statistics, dtype=float)
    if x.size == 0:
        raise ValueError("blank_statistics must be non-empty")
    # A sample is positive iff statistic >= threshold, so specificity is the
    # fraction of blanks STRICTLY below the threshold. Pick the smallest
    # candidate value (observed value, or a sentinel just above the max) whose
    # strict-below fraction meets the target -- correct for discrete/tied counts
    # where a plain quantile can undershoot.
    uniq = np.unique(x)
    sentinel = np.nextafter(uniq[-1], np.inf)
    for c in np.append(uniq, sentinel):
        if np.mean(x < c) >= target_specificity:
            return float(c)
    return float(sentinel)


@dataclass(frozen=True, slots=True)
class PowerResult:
    """Result of :func:`power_analysis`."""

    required_replicates: int | None  # smallest n meeting the target, or None
    achieved_ci_width: dict[int, float]  # n -> LoD CI width


def power_analysis(
    curve: DetectionCurve,
    levels: list[float],
    target_ci_width: float,
    replicate_grid: list[int],
    hit_rate: float = 0.95,
    n_bootstrap: int = 400,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> PowerResult:
    """Replicates per level needed for a target CI width on the LoD (BUILD_SPEC 5.6).

    Treats ``curve`` as ground truth, generates detection outcomes at ``levels``
    with ``n`` replicates each, refits, and measures the resulting LoD CI width.
    The experimental-design tool for the real study. (Uses numpy directly -- no
    dependency on ``simulate.py``.)
    """
    rng = rng or np.random.default_rng()
    p_true = curve.predict(np.array(levels))
    achieved: dict[int, float] = {}
    required: int | None = None
    for n in sorted(replicate_grid):
        widths = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            k = rng.binomial(n, p_true)
            recs = [
                DetectionRecord(lvl, n, int(kk)) for lvl, kk in zip(levels, k)
            ]
            try:
                lod = fit_detection_curve(recs, link=curve.link).level_at(hit_rate)
                widths[b] = lod
            except ValueError:
                widths[b] = np.nan
        finite = widths[np.isfinite(widths)]
        tail = (1 - ci) / 2
        width = (
            float(np.quantile(finite, 1 - tail) - np.quantile(finite, tail))
            if finite.size
            else float("inf")
        )
        achieved[n] = width
        if required is None and width <= target_ci_width:
            required = n
    return PowerResult(required_replicates=required, achieved_ci_width=achieved)
