"""LoD surface and the lever analysis (BUILD_SPEC 7).

Runs entirely on the analytic path (BUILD_SPEC 4.2), so depth dispersion and
dropout are ignored here by construction -- state that on any figure. Uses
root-finding on VAF to locate the 95%-detection point per cell, never a
simulated VAF ladder.

``what_would_it_take`` is the most decision-relevant output in the tool: given a
config that misses the target, the minimum single-lever change that would reach
it. It is a one-lever-at-a-time analysis by design -- the genuinely cheapest
route may combine levers, so each result is annotated as a single-lever bound,
not a global optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from mrd_lod_sim.analytic import detection_probability
from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import DetectionRule
from mrd_lod_sim.errors import REGIME_RATES, ConstantError

__all__ = [
    "achievable_lod",
    "SurfaceResult",
    "lod_surface",
    "LeverChange",
    "what_would_it_take",
]

_UNREACHABLE = float("inf")


def achievable_lod(
    config: AssayConfig,
    rule: DetectionRule,
    hit_rate: float = 0.95,
    vaf_bounds: tuple[float, float] = (1e-9, 0.5),
) -> float:
    """Per-site VAF at ``hit_rate`` detection, or ``inf`` if unreachable.

    Detection probability is monotone increasing in VAF, so a single ``brentq``
    on ``vaf -> P(detect) - hit_rate`` locates the operating point.
    """
    lo, hi = vaf_bounds

    def f(v: float) -> float:
        return detection_probability(config, rule, v) - hit_rate

    if f(hi) < 0:
        return _UNREACHABLE  # target not reachable even at high VAF
    if f(lo) >= 0:
        return lo
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-10))


@dataclass(frozen=True, slots=True)
class SurfaceResult:
    """A 2-D LoD grid (BUILD_SPEC 7, 8.3).

    ``lod[i, j]`` is the achievable per-site VAF LoD at
    ``input_masses[i]`` x ``panel_sizes[j]`` (``inf`` where unreachable).
    """

    panel_sizes: np.ndarray
    input_masses: np.ndarray
    lod: np.ndarray
    hit_rate: float

    def as_dict(self) -> dict:
        return {
            "panel_sizes": self.panel_sizes.tolist(),
            "input_masses": self.input_masses.tolist(),
            "lod": np.where(np.isfinite(self.lod), self.lod, None).tolist(),
            "hit_rate": self.hit_rate,
        }


def lod_surface(
    config: AssayConfig,
    rule: DetectionRule,
    panel_sizes: list[int],
    input_masses: list[float],
    hit_rate: float = 0.95,
) -> SurfaceResult:
    """Compute achievable LoD over a panel-size x input-mass grid (BUILD_SPEC 7)."""
    ns = np.array(panel_sizes)
    masses = np.array(input_masses, dtype=float)
    grid = np.empty((masses.size, ns.size), dtype=float)
    for i, mass in enumerate(masses):
        for j, n in enumerate(ns):
            cell = replace(
                config,
                molecules=replace(config.molecules, input_ng=float(mass)),
                panel=replace(config.panel, n_variants=int(n)),
            )
            grid[i, j] = achievable_lod(cell, rule, hit_rate)
    return SurfaceResult(ns, masses, grid, hit_rate)


@dataclass(frozen=True, slots=True)
class LeverChange:
    """A single-lever requirement to reach a target (BUILD_SPEC 7)."""

    lever: str
    current: float
    required: float | None  # None => not reachable by this lever alone
    unit: str
    plausible: bool
    note: str


def _meets(config: AssayConfig, rule: DetectionRule, target_vaf: float, hit_rate: float) -> bool:
    return detection_probability(config, rule, target_vaf) >= hit_rate


def _bisect_increasing(
    eval_meets, lo: float, hi: float, tol_rel: float = 1e-4
) -> float | None:
    """Smallest value in [lo, hi] for which ``eval_meets`` is True (monotone)."""
    if not eval_meets(hi):
        return None
    if eval_meets(lo):
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eval_meets(mid):
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol_rel * max(hi, 1e-30):
            break
    return hi


def what_would_it_take(
    config: AssayConfig,
    rule: DetectionRule,
    target_vaf: float,
    hit_rate: float = 0.95,
    *,
    max_panel: int = 100_000,
    plausible_panel: int = 5_000,
    max_input_ng: float = 1_000.0,
    plausible_input_ng: float = 100.0,
    max_conversion: float = 0.9,
) -> list[LeverChange]:
    """Minimum single-lever change to reach ``target_vaf`` (BUILD_SPEC 7).

    Returns one :class:`LeverChange` per lever (panel size, input mass,
    conversion efficiency, error regime). Levers are moved one at a time; each
    result is a single-lever bound, not a joint optimum.
    """
    changes: list[LeverChange] = []

    # --- Panel size N ---
    n0 = config.panel.n_variants

    def n_meets(n: float) -> bool:
        cfg = replace(config, panel=replace(config.panel, n_variants=max(1, int(np.ceil(n)))))
        return _meets(cfg, rule, target_vaf, hit_rate)

    n_req = _bisect_increasing(n_meets, float(n0), float(max_panel))
    n_req_val = int(np.ceil(n_req)) if n_req is not None else None
    changes.append(
        LeverChange(
            lever="panel_size",
            current=float(n0),
            required=float(n_req_val) if n_req_val is not None else None,
            unit="variants",
            plausible=n_req_val is not None and n_req_val <= plausible_panel,
            note=(
                f"grow panel to ~{n_req_val} tracked variants"
                if n_req_val is not None
                else f"unreachable below {max_panel} variants by panel size alone"
            ),
        )
    )

    # --- Input mass ---
    m0 = config.molecules.input_ng

    def m_meets(m: float) -> bool:
        cfg = replace(config, molecules=replace(config.molecules, input_ng=m))
        return _meets(cfg, rule, target_vaf, hit_rate)

    m_req = _bisect_increasing(m_meets, m0, max_input_ng)
    changes.append(
        LeverChange(
            lever="input_mass",
            current=m0,
            required=m_req,
            unit="ng",
            plausible=m_req is not None and m_req <= plausible_input_ng,
            note=(
                f"increase cfDNA input to ~{m_req:.0f} ng"
                if m_req is not None
                else f"unreachable below {max_input_ng:.0f} ng by input mass alone"
            ),
        )
    )

    # --- Conversion efficiency ---
    c0 = config.molecules.conversion_efficiency

    def c_meets(c: float) -> bool:
        cfg = replace(config, molecules=replace(config.molecules, conversion_efficiency=c))
        return _meets(cfg, rule, target_vaf, hit_rate)

    c_req = _bisect_increasing(c_meets, c0, max_conversion)
    changes.append(
        LeverChange(
            lever="conversion_efficiency",
            current=c0,
            required=c_req,
            unit="fraction",
            plausible=c_req is not None and c_req <= max_conversion,
            note=(
                f"improve conversion efficiency to ~{c_req * 100:.0f}%"
                if c_req is not None
                else f"unreachable below {max_conversion * 100:.0f}% conversion alone"
            ),
        )
    )

    # --- Error regime (lower epsilon) ---
    eps0 = config.mean_error_rate()

    def eps_meets(eps: float) -> bool:
        cfg = replace(config, error_model=ConstantError(eps))
        return _meets(cfg, rule, target_vaf, hit_rate)

    # Largest epsilon (least demanding) that still meets the target: bisect the
    # *decreasing* direction by negating.
    eps_req = _bisect_increasing(lambda neg: eps_meets(-neg), -eps0, -1e-12)
    eps_needed = -eps_req if eps_req is not None else None
    named = None
    if eps_needed is not None:
        for regime, rate in sorted(REGIME_RATES.items(), key=lambda kv: -kv[1]):
            if rate <= eps_needed:
                named = regime
                break
    changes.append(
        LeverChange(
            lever="error_regime",
            current=eps0,
            required=eps_needed,
            unit="per-base error",
            plausible=named is not None,
            note=(
                f"reduce error to <= {eps_needed:.1e}"
                + (f" (reachable with the {named} regime)" if named else " (below any named regime)")
                if eps_needed is not None
                else "target met regardless of error regime"
            ),
        )
    )

    return changes
