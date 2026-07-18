"""Tests for scenario loading, dashboard rendering, and the CLI (BUILD_SPEC 8, 9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mrd_lod_sim.analytic import _aggregate_detection_probability
from mrd_lod_sim.cli import app
from mrd_lod_sim.report import build_payload, render_dashboard
from mrd_lod_sim.scenario import load_scenario

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


def test_load_all_example_scenarios() -> None:
    for name in ("default", "conservative", "optimistic"):
        sc = load_scenario(CONFIGS / f"{name}.toml")
        assert sc.config.panel.n_variants >= 1
        assert 0 < sc.target_vaf < 1


def test_target_tf_to_vaf_factor_of_two() -> None:
    sc = load_scenario(CONFIGS / "default.toml")
    # default.toml sets tf = 1e-5 -> vaf = 5e-6.
    assert sc.target_vaf == pytest.approx(5e-6)


def test_golden_vectors_are_internally_consistent() -> None:
    # Every golden row must equal the analytic aggregate for its inputs, so the
    # JS-side assertion is checking against a correct reference.
    payload = build_payload(load_scenario(CONFIGS / "default.toml"))
    assert len(payload["golden"]) > 0
    for r in payload["golden"]:
        ge = r["input_ng"] / r["pg"] * r["conv"] * r["strand"]
        p = _aggregate_detection_probability(
            ge, r["n"], r["e_ccf"], r["eps"], r["vaf"], r["alpha"], None
        )
        assert p == pytest.approx(r["p"], abs=1e-12)
        assert 0.0 <= r["p"] <= 1.0


@pytest.mark.parametrize("name", ["default", "conservative", "optimistic"])
def test_dashboard_renders_self_contained(name, tmp_path) -> None:
    sc = load_scenario(CONFIGS / f"{name}.toml")
    out = render_dashboard(sc, tmp_path / "d.html")
    html = out.read_text()
    assert "<!doctype html>" in html.lower()
    assert "PAYLOAD" in html and "golden" in html
    # No local asset references beyond the documented Plotly CDN.
    assert "src=\"http" not in html or "cdn.plot.ly" in html


def test_rendered_units_and_labels(tmp_path) -> None:
    # BUILD_SPEC review P0.1 / P1.1-P1.3: molecules not moles, ppm everywhere.
    sc = load_scenario(CONFIGS / "default.toml")
    html = render_dashboard(sc, tmp_path / "d.html").read_text()
    # No standalone "mol" unit suffix (only the word "molecules").
    assert " mol'" not in html and "} mol)" not in html and " mol<" not in html
    assert "molecules" in html
    # ppm-labelled axes / colorbar, not log10 or per-site VAF titles.
    assert "tumour fraction (ppm)" in html
    assert "LoD (ppm TF)" in html
    assert "log₁₀ TF LoD" not in html
    assert "'per-site VAF'" not in html
    # Strand recovery is now an exposed control coupled to the regime.
    assert 'id="strand"' in html
    assert "regime_strand" in html
    # Hover glossary: every key term carries a plain-language explanation.
    assert "initTooltips" in html
    for term in (
        "cfDNA input", "Conversion efficiency", "Panel size (N)",
        "SSCS — single-strand consensus", "Duplex consensus", "Strand recovery",
        "Target specificity", "Detection rule", "Tumour fraction (TF)",
        "Effective genome equivalents",
    ):
        assert f'data-tip-title="{term}"' in html, term


def test_payload_mc_settings_and_extras(tmp_path) -> None:
    # BUILD_SPEC review P0.2 / P3.1 / P3.2.
    sc = load_scenario(CONFIGS / "default.toml")
    presets = [load_scenario(CONFIGS / f"{n}.toml") for n in ("conservative", "default", "optimistic")]
    payload = build_payload(sc, presets=presets)
    # mc_settings records the operating alpha (= 1 - target specificity) so the
    # overlay lines up with the analytic curve on load and hides on mismatch.
    mcs = payload["mc_settings"]
    assert mcs["alpha"] == pytest.approx(1.0 - sc.target_specificity)
    assert mcs["n_variants"] == sc.config.panel.n_variants
    assert mcs["rule"] == "aggregate"
    # presets carry every slider field.
    assert [p["name"] for p in payload["presets"]] == ["conservative", "default", "optimistic"]
    assert all({"input_ng", "conv", "n", "eps", "spec", "rule", "targetTF"} <= set(p) for p in payload["presets"])
    # power estimate is present as RELATIVE half-widths (so the panel can scale
    # live with the current LoD) plus a target fraction.
    pw = payload["power"]
    assert pw is not None
    assert len(pw["grid"]) >= 3
    assert "target_rel" in pw and 0 < pw["target_rel"] < 1
    assert all(("rel" in g) for g in pw["grid"])
    # regime -> strand coupling is exposed to the page.
    assert payload["regime_strand"]["DUPLEX"] < payload["regime_strand"]["SSCS"]


def test_cli_simulate(tmp_path) -> None:
    res = runner.invoke(
        app,
        ["simulate", "--config", str(CONFIGS / "default.toml"),
         "--replicates", "2000", "--points", "5", "--out", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    data = json.loads((tmp_path / "simulate.json").read_text())
    assert len(data["curve"]) == 5
    assert "achievable_lod_vaf" in data


def test_cli_surface(tmp_path) -> None:
    res = runner.invoke(
        app, ["surface", "--config", str(CONFIGS / "default.toml"), "--out", str(tmp_path)]
    )
    assert res.exit_code == 0, res.output
    data = json.loads((tmp_path / "surface.json").read_text())
    assert len(data["what_would_it_take"]) == 4


def test_cli_dashboard(tmp_path) -> None:
    out = tmp_path / "dash.html"
    res = runner.invoke(
        app, ["dashboard", "--config", str(CONFIGS / "default.toml"), "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert out.exists()


def test_cli_calibrate_and_compare(tmp_path) -> None:
    # Healthy-donor CSV -> calibrate.
    bg = tmp_path / "healthy.csv"
    rng = np.random.default_rng(0)
    lines = ["mutant,total"] + [f"{int(x)},3000" for x in rng.poisson(3000 * 2e-5, 500)]
    bg.write_text("\n".join(lines))
    res = runner.invoke(app, ["calibrate", "--background", str(bg), "--out", str(tmp_path / "cal.toml")])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "cal.toml").exists()

    # Dilution CSV -> compare.
    obs = tmp_path / "dilution.csv"
    obs.write_text(
        "level,n_replicates,n_detections\n"
        "1e-6,200,10\n5e-6,200,120\n1e-5,200,190\n5e-5,200,200\n"
    )
    res2 = runner.invoke(
        app, ["compare", "--config", str(CONFIGS / "default.toml"), "--observed", str(obs)]
    )
    assert res2.exit_code == 0, res2.output
    assert "likely_causes" in res2.output
