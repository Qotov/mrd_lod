"""Error models -- a distribution over per-site background rates (BUILD_SPEC 2.4).

The background error rate ``eps`` is a *distribution*, not a scalar: some sites
are far noisier than others (deamination at CpG, oxidation, etc.).

The load-bearing contract (BUILD_SPEC 2.4.1) is that every model exposes **two**
methods, and the two computation paths use different ones:

- ``sample(shape, rng)`` -- per-site draws for the Monte Carlo path.
- ``mean_rate(panel)`` -- the analytic expectation ``E[eps]`` used by
  ``analytic.py``. **The analytic path must call this and never a median or a
  nominal value.** Conflating median and mean is a silent factor error at ppm
  scale (the classic lognormal trap), which is exactly why the split is
  mandatory and why a convergence test guards it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ErrorModel",
    "ConstantError",
    "LogNormalError",
    "ContextualError",
    "EmpiricalError",
    "REGIME_RATES",
    "error_preset",
]


class ErrorModel(ABC):
    """Abstract base for per-site background error models."""

    @abstractmethod
    def sample(
        self, shape: int | tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """Draw an array of per-site error rates for the Monte Carlo path."""

    @abstractmethod
    def mean_rate(self, panel: Any | None = None) -> float:
        """Return ``E[eps]`` for the analytic path.

        ``panel`` -- optional object exposing a ``context_weights`` mapping, used
        only by context-dependent models to weight the expectation by a panel's
        trinucleotide composition. Context-free models ignore it.
        """


@dataclass(frozen=True, slots=True)
class ConstantError(ErrorModel):
    """A single fixed per-base error rate. The simplest baseline (BUILD_SPEC 2.4)."""

    rate: float

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError("rate must be non-negative")

    def sample(
        self, shape: int | tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        return np.full(shape, self.rate, dtype=float)

    def mean_rate(self, panel: Any | None = None) -> float:
        return self.rate


@dataclass(frozen=True, slots=True)
class LogNormalError(ErrorModel):
    """Log-normal per-site error (BUILD_SPEC 2.4), the default model.

    Parameterised by the **median** and the log-scale ``sigma``. Note the
    analytic mean is *not* the median::

        E[eps] = median * exp(sigma**2 / 2)

    Attributes:
        median: median per-base error rate (the log-normal's geometric mean).
        sigma: standard deviation of the underlying normal (log-space spread).
    """

    median: float
    sigma: float

    def __post_init__(self) -> None:
        if self.median <= 0:
            raise ValueError("median must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def sample(
        self, shape: int | tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        return rng.lognormal(mean=math.log(self.median), sigma=self.sigma, size=shape)

    def mean_rate(self, panel: Any | None = None) -> float:
        return self.median * math.exp(self.sigma**2 / 2.0)


@dataclass(frozen=True, slots=True)
class ContextualError(ErrorModel):
    """Per-trinucleotide-context error rates (BUILD_SPEC 2.4).

    Captures that e.g. C>T at CpG (deamination) and G>T (oxidation) carry
    elevated rates. The expectation is defined *relative to a panel's context
    composition*, hence the ``panel`` argument to :meth:`mean_rate`.

    Attributes:
        context_rates: mapping of context label -> per-base error rate.
        context_weights: assumed composition (context label -> weight). Defaults
            to uniform over ``context_rates``. Weights are normalised to sum 1.
    """

    context_rates: Mapping[str, float]
    context_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.context_rates:
            raise ValueError("context_rates must be non-empty")
        if any(r < 0 for r in self.context_rates.values()):
            raise ValueError("context rates must be non-negative")
        if self.context_weights is not None:
            missing = set(self.context_weights) - set(self.context_rates)
            if missing:
                raise ValueError(f"weights reference unknown contexts: {missing}")

    def _weights(self, panel: Any | None) -> dict[str, float]:
        raw: Mapping[str, float] | None = None
        if panel is not None and getattr(panel, "context_weights", None) is not None:
            raw = panel.context_weights
        elif self.context_weights is not None:
            raw = self.context_weights
        if raw is None:
            # Uniform over known contexts.
            n = len(self.context_rates)
            return {c: 1.0 / n for c in self.context_rates}
        total = float(sum(raw.values()))
        if total <= 0:
            raise ValueError("context weights must sum to a positive value")
        return {c: w / total for c, w in raw.items()}

    def sample(
        self, shape: int | tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        weights = self._weights(None)
        contexts = list(weights)
        rates = np.array([self.context_rates[c] for c in contexts], dtype=float)
        probs = np.array([weights[c] for c in contexts], dtype=float)
        idx = rng.choice(len(contexts), size=shape, p=probs)
        return rates[idx]

    def mean_rate(self, panel: Any | None = None) -> float:
        weights = self._weights(panel)
        return float(sum(w * self.context_rates[c] for c, w in weights.items()))


@dataclass(frozen=True, slots=True)
class EmpiricalError(ErrorModel):
    """Error model estimated from real healthy-donor counts (BUILD_SPEC 2.4, 6).

    Produced by ``calibrate.estimate_error_model``; replaces the literature
    prior with an empirical distribution of per-site rates. ``mean_rate`` is the
    pooled rate (total mutant / total depth), which is what governs the analytic
    background expectation.

    Attributes:
        mutant_counts: per-site background mutant molecule counts.
        total_counts: per-site total molecule counts (depth).
        contexts: optional per-site trinucleotide context labels.
    """

    mutant_counts: np.ndarray
    total_counts: np.ndarray
    contexts: Sequence[str] | None = None
    _per_site_rate: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mut = np.asarray(self.mutant_counts, dtype=float)
        tot = np.asarray(self.total_counts, dtype=float)
        if mut.shape != tot.shape:
            raise ValueError("mutant_counts and total_counts must have equal shape")
        if mut.size == 0:
            raise ValueError("empirical counts must be non-empty")
        if np.any(tot <= 0):
            raise ValueError("total_counts must be positive")
        if np.any(mut < 0):
            raise ValueError("mutant_counts must be non-negative")
        object.__setattr__(self, "mutant_counts", mut)
        object.__setattr__(self, "total_counts", tot)
        object.__setattr__(self, "_per_site_rate", mut / tot)

    def sample(
        self, shape: int | tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        # Bootstrap resample of observed per-site rates.
        idx = rng.integers(0, self._per_site_rate.size, size=shape)
        return self._per_site_rate[idx]

    def mean_rate(self, panel: Any | None = None) -> float:
        return float(self.mutant_counts.sum() / self.total_counts.sum())


#: Literature-derived order-of-magnitude priors for the per-base error MEAN of
#: each consensus regime (BUILD_SPEC 2.4). These are priors, NOT measurements --
#: replace with an :class:`EmpiricalError` fitted from healthy-donor data as soon
#: as it exists. Rough citations: RAW ~ raw Illumina substitution error (~1e-3,
#: e.g. Schirmer et al. 2016); SSCS ~ single-strand UMI consensus (~1e-5, Kennedy
#: et al. 2014); DUPLEX ~ duplex consensus (~1e-7, Schmitt et al. 2012).
REGIME_RATES: dict[str, float] = {
    "RAW": 1e-3,
    "SSCS": 1e-5,
    "DUPLEX": 1e-7,
}


def error_preset(regime: str, sigma: float = 0.0) -> ErrorModel:
    """Construct a named regime preset whose ``mean_rate()`` equals the prior.

    Args:
        regime: one of ``REGIME_RATES`` (case-insensitive).
        sigma: log-space spread. ``0.0`` returns a :class:`ConstantError`;
            ``> 0`` returns a :class:`LogNormalError` whose median is chosen so
            the analytic mean still equals the regime's quoted rate
            (``median = rate * exp(-sigma**2 / 2)``). This keeps the quoted
            order-of-magnitude prior interpretable as the mean regardless of
            spread.
    """
    key = regime.upper()
    if key not in REGIME_RATES:
        raise ValueError(f"unknown regime {regime!r}; choose from {list(REGIME_RATES)}")
    rate = REGIME_RATES[key]
    if sigma == 0.0:
        return ConstantError(rate)
    median = rate * math.exp(-(sigma**2) / 2.0)
    return LogNormalError(median=median, sigma=sigma)
