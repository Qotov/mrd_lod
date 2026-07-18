"""Command-line interface (BUILD_SPEC 9).

    uv run mrd-lod simulate   --config configs/default.toml --out results/
    uv run mrd-lod surface    --config configs/default.toml --out results/
    uv run mrd-lod dashboard  --config configs/default.toml --out outputs/dashboard.html
    uv run mrd-lod calibrate  --background data/healthy.csv --out configs/calibrated.toml
    uv run mrd-lod compare    --config configs/calibrated.toml --observed data/dilution.csv

Scenarios are TOML so a non-programmer can edit them (see ``configs/``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import typer

from mrd_lod_sim.calibrate import (
    compare_predicted_observed,
    estimate_error_model,
)
from mrd_lod_sim.report import render_dashboard
from mrd_lod_sim.scenario import load_scenario
from mrd_lod_sim.simulate import detection_rate
from mrd_lod_sim.surface import (
    achievable_lod,
    lod_surface,
    what_would_it_take,
)
from mrd_lod_sim.validate import DetectionRecord

app = typer.Typer(
    add_completion=False,
    help="Limit-of-detection simulator and design tool for ctDNA MRD assays.",
)


def _echo_json(obj: dict, out: Path | None) -> None:
    text = json.dumps(obj, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command()
def simulate(
    config: Path = typer.Option(..., "--config", help="Scenario TOML."),
    out: Path | None = typer.Option(None, "--out", help="Output dir (writes simulate.json)."),
    replicates: int = typer.Option(20_000, help="Monte Carlo replicates per VAF."),
    points: int = typer.Option(9, help="VAF ladder points."),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """Monte Carlo detection rate across a VAF ladder + achievable LoD."""
    sc = load_scenario(config)
    rng = np.random.default_rng(seed)
    centre = max(sc.target_vaf, 1e-7)
    vafs = np.logspace(np.log10(centre / 30), np.log10(centre * 30), points)
    curve = [
        {"vaf": float(v), "p_detect": detection_rate(sc.config, sc.rule, float(v), replicates, rng).detection_rate}
        for v in vafs
    ]
    result = {
        "scenario": sc.name,
        "rule": type(sc.rule).__name__,
        "target_vaf": sc.target_vaf,
        "achievable_lod_vaf": achievable_lod(sc.config, sc.rule, sc.hit_rate),
        "curve": curve,
    }
    _echo_json(result, out / "simulate.json" if out else None)


@app.command()
def surface(
    config: Path = typer.Option(..., "--config"),
    out: Path | None = typer.Option(None, "--out", help="Output dir (writes surface.json)."),
) -> None:
    """Compute the achievable-LoD surface over panel size x input mass."""
    sc = load_scenario(config)
    panel_sizes = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    input_masses = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
    surf = lod_surface(sc.config, sc.rule, panel_sizes, input_masses, sc.hit_rate)
    levers = what_would_it_take(sc.config, sc.rule, sc.target_vaf, sc.hit_rate)
    result = {
        "scenario": sc.name,
        "surface": surf.as_dict(),
        "target_vaf": sc.target_vaf,
        "what_would_it_take": [
            {
                "lever": c.lever, "current": c.current, "required": c.required,
                "unit": c.unit, "plausible": c.plausible, "note": c.note,
            }
            for c in levers
        ],
    }
    _echo_json(result, out / "surface.json" if out else None)


@app.command()
def dashboard(
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(Path("outputs/dashboard.html"), "--out"),
) -> None:
    """Render the interactive HTML dashboard."""
    sc = load_scenario(config)
    written = render_dashboard(sc, out)
    typer.echo(f"wrote {written}")


@app.command()
def calibrate(
    background: Path = typer.Option(..., "--background", help="Healthy-donor CSV: mutant,total[,context]."),
    out: Path | None = typer.Option(None, "--out", help="Write a calibrated TOML fragment."),
) -> None:
    """Estimate an empirical error model from healthy-donor counts."""
    mutant, total, contexts = [], [], []
    with background.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mutant.append(float(row["mutant"]))
            total.append(float(row["total"]))
            if "context" in row and row["context"]:
                contexts.append(row["context"])
    model = estimate_error_model(
        np.array(mutant), np.array(total), contexts or None
    )
    rate = model.mean_rate()
    typer.echo(f"empirical pooled error rate: {rate:.3e}  (from {len(mutant)} sites)")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"[error]\nmodel = \"constant\"\nrate = {rate:.6e}\n", encoding="utf-8"
        )
        typer.echo(f"wrote {out}")


@app.command()
def compare(
    config: Path = typer.Option(..., "--config"),
    observed: Path = typer.Option(..., "--observed", help="Dilution CSV: level,n_replicates,n_detections."),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Compare model-predicted vs observed detection and diagnose discrepancy."""
    sc = load_scenario(config)
    records = []
    with observed.open() as fh:
        for row in csv.DictReader(fh):
            records.append(
                DetectionRecord(
                    float(row["level"]), int(row["n_replicates"]), int(row["n_detections"])
                )
            )
    diag = compare_predicted_observed(sc.config, sc.rule, records)
    result = {
        "scenario": sc.name,
        "mean_signed_discrepancy": diag.mean_signed_discrepancy,
        "blank_false_positive": diag.blank_false_positive,
        "levels": diag.levels.tolist(),
        "predicted": diag.predicted.tolist(),
        "observed": diag.observed.tolist(),
        "likely_causes": diag.likely_causes,
    }
    _echo_json(result, out)


if __name__ == "__main__":
    app()
