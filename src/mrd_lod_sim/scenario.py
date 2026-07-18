"""Scenario loading from TOML (BUILD_SPEC 9).

A scenario bundles an :class:`AssayConfig`, the detection rule, and the target,
so a non-programmer can edit whole design scenarios in a TOML file. TOML schema
(all sections optional except where noted)::

    [molecules]
    input_ng = 30.0
    conversion_efficiency = 0.4
    strand_recovery = 1.0

    [panel]
    n_variants = 500
    clonal = false        # true => CCF == 1 at every site
    ccf_alpha = 9.0
    ccf_beta = 1.0
    dropout_prob = 0.0
    # depth_dispersion = 0.5   # optional; leaves the analytic-valid regime

    [error]
    model = "preset"      # "preset" | "constant" | "lognormal"
    regime = "SSCS"       # for model = "preset"
    sigma = 0.0
    # rate = 1e-5         # for model = "constant"
    # median = 1e-5       # for model = "lognormal"

    [detection]
    rule = "aggregate"    # "aggregate" | "kofn" | "lr"
    alpha = 0.05
    k = 2
    per_site_alpha = 0.01

    [target]
    tf = 1e-5             # target tumour fraction (or give vaf directly)
    specificity = 0.99
    hit_rate = 0.95
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from mrd_lod_sim.config import AssayConfig
from mrd_lod_sim.detect import (
    AggregatePoissonRule,
    DetectionRule,
    KofNRule,
    LikelihoodRatioRule,
)
from mrd_lod_sim.errors import (
    ConstantError,
    ErrorModel,
    LogNormalError,
    error_preset,
)
from mrd_lod_sim.panel import PanelModel
from mrd_lod_sim.params import MoleculeParams, vaf_from_tf

__all__ = ["Scenario", "load_scenario", "build_error_model", "build_rule"]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete design scenario (config + rule + target)."""

    config: AssayConfig
    rule: DetectionRule
    target_vaf: float
    target_specificity: float
    hit_rate: float
    name: str


def build_error_model(section: dict) -> ErrorModel:
    model = section.get("model", "preset").lower()
    if model == "preset":
        return error_preset(section.get("regime", "SSCS"), float(section.get("sigma", 0.0)))
    if model == "constant":
        return ConstantError(float(section["rate"]))
    if model == "lognormal":
        return LogNormalError(float(section["median"]), float(section["sigma"]))
    raise ValueError(f"unknown error model {model!r}")


def build_rule(section: dict) -> DetectionRule:
    rule = section.get("rule", "aggregate").lower()
    if rule == "aggregate":
        return AggregatePoissonRule(alpha=float(section.get("alpha", 0.05)))
    if rule == "kofn":
        return KofNRule(
            k=int(section.get("k", 2)),
            per_site_alpha=float(section.get("per_site_alpha", 0.01)),
        )
    if rule == "lr":
        return LikelihoodRatioRule(alpha=float(section.get("alpha", 0.05)))
    raise ValueError(f"unknown detection rule {rule!r}")


def _build_panel(section: dict) -> PanelModel:
    n = int(section.get("n_variants", 500))
    kwargs = dict(
        dropout_prob=float(section.get("dropout_prob", 0.0)),
        depth_dispersion=section.get("depth_dispersion"),
    )
    if section.get("clonal", False):
        return PanelModel.clonal(n, **kwargs)
    return PanelModel(
        n_variants=n,
        ccf_alpha=float(section.get("ccf_alpha", 9.0)),
        ccf_beta=float(section.get("ccf_beta", 1.0)),
        **kwargs,
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load a :class:`Scenario` from a TOML file."""
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    mol = data.get("molecules", {})
    molecules = MoleculeParams(
        input_ng=float(mol.get("input_ng", 30.0)),
        conversion_efficiency=float(mol.get("conversion_efficiency", 0.4)),
        strand_recovery=float(mol.get("strand_recovery", 1.0)),
    )
    config = AssayConfig(
        molecules=molecules,
        panel=_build_panel(data.get("panel", {})),
        error_model=build_error_model(data.get("error", {})),
    )
    rule = build_rule(data.get("detection", {}))

    target = data.get("target", {})
    if "vaf" in target:
        target_vaf = float(target["vaf"])
    elif "tf" in target:
        target_vaf = vaf_from_tf(float(target["tf"]))
    else:
        target_vaf = vaf_from_tf(1e-5)  # 10 ppm TF default (BUILD_SPEC 8.2)

    return Scenario(
        config=config,
        rule=rule,
        target_vaf=target_vaf,
        target_specificity=float(target.get("specificity", 0.99)),
        hit_rate=float(target.get("hit_rate", 0.95)),
        name=path.stem,
    )
