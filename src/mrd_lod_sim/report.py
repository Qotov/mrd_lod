"""Dashboard generation (BUILD_SPEC 8).

Assembles a data payload from a :class:`~mrd_lod_sim.scenario.Scenario` and
renders a single self-contained HTML file. The page reimplements the analytic
model (BUILD_SPEC 4.2) in JS for instant, lattice-free response; this module
supplies:

- the **golden vectors** ``(inputs -> P(detect))`` from ``analytic.py`` that the
  page re-derives on load, so Python/JS can never silently drift;
- a precomputed **Monte-Carlo-validated curve** for the optional overlay,
  demonstrating the JS analytic model agrees with the Python MC.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from mrd_lod_sim.analytic import _aggregate_detection_probability, detection_probability
from mrd_lod_sim.errors import REGIME_RATES
from mrd_lod_sim.molecules import molecule_budget
from mrd_lod_sim.params import tf_from_vaf
from mrd_lod_sim.scenario import Scenario
from mrd_lod_sim.simulate import detection_rate
from mrd_lod_sim.surface import achievable_lod
from mrd_lod_sim.validate import (
    DetectionRecord,
    fit_detection_curve,
    power_analysis,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Map detection-rule class name -> the dashboard's rule key.
_RULE_KEY = {"KofNRule": "kofn", "LikelihoodRatioRule": "lr"}


def _rule_key(rule) -> str:
    return _RULE_KEY.get(type(rule).__name__, "aggregate")


def _golden_vectors(alpha: float) -> list[dict]:
    """Reference (inputs -> P(detect)) rows for the JS cross-check.

    Kept in the moderate-lambda regime so the JS log-space summation matches
    scipy to tight tolerance.
    """
    rows: list[dict] = []
    pg = 0.0033
    for input_ng, conv, strand in ((30.0, 0.4, 1.0), (60.0, 0.5, 1.0)):
        ge_eff = input_ng / pg * conv * strand
        for n in (100, 500, 1000):
            for eps in (1e-5, 1e-4):
                for vaf in (0.0, 3e-6, 1e-5, 3e-5):
                    p = _aggregate_detection_probability(
                        ge_eff, n, 1.0, eps, vaf, alpha, None
                    )
                    rows.append(
                        {
                            "input_ng": input_ng,
                            "conv": conv,
                            "strand": strand,
                            "pg": pg,
                            "n": n,
                            "e_ccf": 1.0,
                            "eps": eps,
                            "alpha": alpha,
                            "vaf": vaf,
                            "p": p,
                        }
                    )
    return rows


def _mc_curve(scenario: Scenario, n_points: int = 12, n_replicates: int = 4000) -> list[dict]:
    """A Monte-Carlo detection curve at the scenario's current settings."""
    rng = np.random.default_rng(0)
    lod_guess = max(scenario.target_vaf, 1e-7)
    vafs = np.logspace(np.log10(lod_guess / 30), np.log10(lod_guess * 30), n_points)
    out = []
    for v in vafs:
        rate = detection_rate(scenario.config, scenario.rule, float(v), n_replicates, rng)
        out.append({"vaf": float(v), "p": rate.detection_rate})
    return out


def _mc_settings(scenario: Scenario, alpha: float) -> dict:
    """Every parameter the MC curve was generated under (BUILD_SPEC review P0.2).

    The dashboard hides the overlay whenever the live UI settings differ, so the
    reader never compares the analytic curve against an MC curve computed at a
    different threshold.
    """
    cfg = scenario.config
    return {
        "input_ng": cfg.molecules.input_ng,
        "conversion_efficiency": cfg.molecules.conversion_efficiency,
        "strand_recovery": cfg.molecules.strand_recovery,
        "pg_per_ge": cfg.molecules.pg_per_genome_equivalent,
        "n_variants": cfg.panel.n_variants,
        "e_ccf": cfg.panel.mean_ccf(),
        "eps": cfg.mean_error_rate(),
        "alpha": alpha,
        "rule": _rule_key(scenario.rule),
        "k": getattr(scenario.rule, "k", 2),
        "per_site_alpha": getattr(scenario.rule, "per_site_alpha", 0.01),
        "regime_label": _regime_label(cfg.mean_error_rate()),
    }


def _regime_label(eps: float) -> str:
    for name, rate in REGIME_RATES.items():
        if abs(eps - rate) / rate < 0.01:
            return name
    return "custom ε"


def _preset_states(presets: list[Scenario]) -> list[dict]:
    """Slider states for each example scenario (BUILD_SPEC review P3.2)."""
    out = []
    for sc in presets:
        cfg = sc.config
        out.append(
            {
                "name": sc.name,
                "input_ng": cfg.molecules.input_ng,
                "conv": cfg.molecules.conversion_efficiency,
                "strand": cfg.molecules.strand_recovery,
                "n": cfg.panel.n_variants,
                "eps": cfg.mean_error_rate(),
                "spec": sc.target_specificity,
                "rule": _rule_key(sc.rule),
                "targetTF": tf_from_vaf(sc.target_vaf),
            }
        )
    return out


def _power_estimate(scenario: Scenario) -> dict | None:
    """Replicates per level needed for a target CI width on the LoD (P3.1).

    Precomputed at generation settings (the bootstrap is not feasible in vanilla
    JS). Fits a probit curve to analytic points around the achievable LoD, then
    runs ``validate.power_analysis``.
    """
    cfg, rule, hit = scenario.config, scenario.rule, scenario.hit_rate
    lod = achievable_lod(cfg, rule, hit)
    if not math.isfinite(lod):
        return None
    n_fit = 100_000
    levels = [lod * f for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
    recs = [
        DetectionRecord(
            lvl, n_fit, int(round(detection_probability(cfg, rule, lvl) * n_fit))
        )
        for lvl in levels
    ]
    try:
        curve = fit_detection_curve(recs)
    except ValueError:
        return None
    target_ci = lod * 0.5  # +/- 25% of the LoD estimate
    grid = [50, 100, 200, 500, 1000, 2000]
    pr = power_analysis(
        curve, [lod * 0.5, lod, lod * 2], target_ci, grid,
        hit_rate=hit, n_bootstrap=200, rng=np.random.default_rng(0),
    )
    return {
        "lod_ppm": tf_from_vaf(lod) * 1e6,
        "target_ci_ppm": tf_from_vaf(target_ci) * 1e6,
        "required_replicates": pr.required_replicates,
        "grid": [
            {
                "n": n,
                "ci_ppm": tf_from_vaf(w) * 1e6 if math.isfinite(w) else None,
            }
            for n, w in sorted(pr.achieved_ci_width.items())
        ],
    }


def build_payload(scenario: Scenario, presets: list[Scenario] | None = None) -> dict:
    """Assemble the JSON payload embedded in the dashboard."""
    cfg = scenario.config
    budget = molecule_budget(cfg.molecules)
    alpha = getattr(scenario.rule, "alpha", 0.05)
    rule_name = type(scenario.rule).__name__

    return {
        "meta": {
            "name": scenario.name,
            "rule": rule_name,
            "uses_aggregate_proxy": rule_name == "LikelihoodRatioRule",
        },
        "current": {
            "input_ng": cfg.molecules.input_ng,
            "conversion_efficiency": cfg.molecules.conversion_efficiency,
            "strand_recovery": cfg.molecules.strand_recovery,
            "pg_per_ge": cfg.molecules.pg_per_genome_equivalent,
            "n_variants": cfg.panel.n_variants,
            "e_ccf": cfg.panel.mean_ccf(),
            "eps": cfg.mean_error_rate(),
            "alpha": alpha,
            "k": getattr(scenario.rule, "k", 2),
            "per_site_alpha": getattr(scenario.rule, "per_site_alpha", 0.01),
            "ge_eff": budget.ge_eff,
        },
        "target": {
            "vaf": scenario.target_vaf,
            "tf": tf_from_vaf(scenario.target_vaf),
            "specificity": scenario.target_specificity,
            "hit_rate": scenario.hit_rate,
        },
        "regimes": REGIME_RATES,
        "golden": _golden_vectors(alpha),
        "mc_curve": _mc_curve(scenario),
        "mc_settings": _mc_settings(scenario, alpha),
        "presets": _preset_states(presets) if presets else [],
        "power": _power_estimate(scenario),
    }


def render_dashboard(
    scenario: Scenario,
    out_path: str | Path,
    presets: list[Scenario] | None = None,
) -> Path:
    """Render the dashboard HTML for ``scenario`` to ``out_path``."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("dashboard.html.j2")
    payload = build_payload(scenario, presets=presets)
    html = template.render(payload_json=json.dumps(payload), name=scenario.name)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
