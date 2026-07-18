"""Generate the README hero contour figure (BUILD_SPEC 11.3).

Dev-only: uses matplotlib (not a runtime dependency). Run with::

    uv run python scripts/make_hero_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mrd_lod_sim.params import tf_from_vaf
from mrd_lod_sim.scenario import load_scenario
from mrd_lod_sim.surface import lod_surface

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sc = load_scenario(ROOT / "configs" / "default.toml")
    panel_sizes = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    input_masses = [5, 10, 20, 30, 50, 75, 100]
    surf = lod_surface(sc.config, sc.rule, panel_sizes, input_masses, sc.hit_rate)

    # Convert per-site VAF LoD to tumour fraction (ppm), log10 for contouring.
    tf_ppm = tf_from_vaf(surf.lod) * 1e6
    z = np.log10(tf_ppm)

    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(surf.panel_sizes, surf.input_masses, z, levels=14, cmap="viridis")
    target_ppm = tf_from_vaf(sc.target_vaf) * 1e6
    tc = ax.contour(
        surf.panel_sizes, surf.input_masses, z,
        levels=[np.log10(target_ppm)], colors="#c93838", linewidths=2.4,
    )
    ax.clabel(tc, fmt=lambda v: f"target {target_ppm:.0f} ppm TF", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Panel size (tracked variants, N)")
    ax.set_ylabel("cfDNA input (ng)")
    ax.set_title("Achievable LoD — aggregate-Poisson rule, SSCS regime, 40% conversion")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("log₁₀ ( LoD in ppm tumour fraction )")
    fig.tight_layout()

    out = ROOT / "docs" / "img" / "hero_surface.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
