# Build Specification — `mrd-lod-sim`
### A limit-of-detection simulator and design tool for tumour-informed ctDNA MRD assays

> **For Claude Code.** Build this as a complete, tested, documented Python package with an interactive HTML dashboard. Read this spec fully before writing code. Prefer correctness and clean interfaces over feature count — every module below has a defined afterlife in a production assay, so interfaces matter more than cleverness.

---

## 0. Project goals (read this first — it constrains every design choice)

This tool answers one question: **given an assay configuration, what limit of detection can it achieve, and what would it take to reach a target?**

It must be useful in three successive phases, using the *same* code:

1. **Now (no data):** run on literature-derived priors to map the feasibility envelope and size the design space.
2. **Soon (background data):** consume real healthy-donor sequencing to replace the assumed error model with an empirical one.
3. **Later (dilution data):** consume real dilution-series results to fit LoD, and compare predicted vs observed to diagnose assay underperformance.

Three architectural consequences that are **non-negotiable**:

- **The detection rule is the production caller.** It must accept real per-site observations `(mutant_count, total_count, error_rate)` — never simulation-internal state. Simulation is one producer of that input; a real pipeline is another.
- **The validation layer is data-source agnostic.** LoB/LoD fitting consumes `(level, n_replicates, n_detections)` tuples. Those come from simulation today and from wet-lab experiments later, with no code change.
- **Every assay assumption is a parameter, never a constant.** No hard-coded error rates, panel sizes, or efficiencies in logic.

---

## 1. Environment and tooling

- **Python 3.11+**, managed with **uv**.
- Initialise with `uv init`, dependencies via `uv add`, everything runs through `uv run`.
- Dependencies: `numpy`, `scipy`, `jinja2`, `typer` (CLI), `pytest` (dev), `pytest-cov` (dev).
- **Do not add plotly/matplotlib as a runtime dependency for the dashboard** — the HTML pulls Plotly from CDN with a bundled-fallback note in the README. Static plots for the README may use matplotlib as a dev dependency only.
- Type hints throughout; `dataclasses` for all parameter objects. Run `ruff` if available.

---

## 2. Scientific model — get these conventions exactly right

### 2.1 Molecule budget

```
GE_total  = input_ng / PG_PER_GENOME_EQUIVALENT      # default 0.0033 ng (3.3 pg), configurable
GE_eff    = GE_total × conversion_efficiency × strand_recovery
```

- `conversion_efficiency` — fraction of input molecules recovered as usable consensus families (typical 0.2–0.6). **This is the most commonly underestimated parameter and a frequent cause of disappointing real-world LoD; expose it prominently.**
- `strand_recovery` — additional penalty when duplex consensus is required (both strands must be recovered). Default 1.0 for single-strand consensus, ~0.5 for duplex. Keep separate from `conversion_efficiency` so the duplex trade-off (lower error, fewer usable molecules) is explicit and visible.

### 2.2 VAF vs tumour fraction — be explicit everywhere

Internally the model works in **per-site VAF**. For a clonal heterozygous SNV, `VAF ≈ TF / 2`. Provide explicit converters and label every output with which quantity it is. Ambiguity here is a factor-of-two error at ppm scale — the API must make it impossible to confuse them.

### 2.3 Per-site sampling model

For each tracked site `i`, given panel size `N`:

```
vaf_i        = vaf_clonal × ccf_i                       # ccf_i sampled from panel CCF distribution
mutant_i     ~ Poisson(GE_eff × vaf_i)                  # true signal
background_i ~ Poisson(GE_eff × eps_i)                  # eps_i sampled from error model
observed_i   = mutant_i + background_i
depth_i      = GE_eff                                   # optionally with dispersion, see 2.5
```

Poisson is used rather than Binomial because `vaf ≪ 1`; document this approximation and its validity range in the code.

### 2.4 Error model — a distribution, not a scalar

`eps_i` must be drawn from a distribution, not fixed. Support:

- **`ConstantError(rate)`** — simplest baseline.
- **`LogNormalError(median, sigma)`** — default; captures that some sites are far noisier than others.
- **`ContextualError(context_rates)`** — per-trinucleotide-context rates, so C>T at CpG (deamination) and G>T (oxidation) can carry elevated rates.
- **`EmpiricalError(counts_table)`** — estimated from real healthy-donor data (see §6).

Ship **literature-derived presets** as named regimes, clearly cited in docstrings as order-of-magnitude priors, not measurements:

| Regime | Approx. per-base error |
|---|---|
| `RAW` | ~1e-3 |
| `SSCS` (single-strand UMI consensus) | ~1e-5 |
| `DUPLEX` | ~1e-7 |

### 2.4.1 ErrorModel interface — separate MC sampling from the analytic expectation

Every `ErrorModel` exposes **two** methods, and the two computation paths use different ones. Conflating them is a silent factor error at ppm scale, so the split is mandatory:

- **`sample(shape, rng) -> np.ndarray`** — draw per-site rates for the Monte Carlo path (§4.1).
- **`mean_rate(panel=None) -> float`** — the analytic expectation `E[eps]` used by `analytic.py` (§4.2). **The analytic path must call this and never a median or nominal value.**

Concrete `mean_rate` definitions:

| Model | `mean_rate()` |
|---|---|
| `ConstantError(rate)` | `rate` |
| `LogNormalError(median, sigma)` | `median × exp(sigma² / 2)` — **the mean, not the median** |
| `ContextualError(context_rates)` | `Σ_c w_c · rate_c`, weighted by the panel's trinucleotide composition `w_c` (hence the `panel` argument) |
| `EmpiricalError(counts_table)` | total mutant / total depth across the table, per-context when context is supplied |

The `panel` argument makes the error-model ↔ panel-model coupling explicit: a context-dependent error expectation is only defined relative to a panel's context mix. A test asserts `sample(...).mean()` converges to `mean_rate()` for `LogNormalError`, so a median/mean mix-up fails loudly (§10).

### 2.5 Panel heterogeneity

Not every tracked variant behaves identically. Support a `PanelModel` with:
- A **CCF distribution** across tracked variants (default: mostly clonal with a subclonal tail — e.g. a Beta distribution skewed toward 1.0). This links the simulator to panel-*selection* strategy: a panel of truncal variants outperforms a panel with subclonal ones at identical N.
- Optional **per-site dropout probability** (probe failure / poor capture).
- Optional **depth dispersion** across sites (negative binomial rather than fixed `GE_eff`).

---

## 3. Detection rules — `detect.py` (the production caller)

Define an ABC `DetectionRule` with:

```python
def call(self, observations: SiteObservations, error_model: ErrorModel) -> MRDCall: ...
```

Where `SiteObservations` holds arrays `mutant_counts`, `total_counts`, optional `site_ids` — i.e. **exactly what a real pipeline emits.** `MRDCall` returns `detected: bool`, `statistic: float`, `p_value: float`, plus per-site detail.

**VAF estimation is an optional capability, not a mandatory field.** Only rules that naturally estimate a level should report one, so `estimated_vaf` / `estimated_ci` on `MRDCall` are `float | None` (default `None`). `AggregatePoissonRule` and `LikelihoodRatioRule` populate them (the LR rule's MLE VAF is its natural estimate); `KofNRule` — which only counts significant sites — leaves them `None` rather than fabricating a value. Every VAF a rule does report is per-site VAF and must be labelled as such (§2.2); downstream consumers convert to TF only through the §2.2 converter.

Implement **three** rules and make comparing them a first-class feature:

1. **`KofNRule(k, per_site_alpha)`** — Signatera-style. A site is positive if its mutant count is significant against its own background; the sample is positive if ≥`k` sites are positive. Intuitive and robust; statistically inefficient.
2. **`AggregatePoissonRule(alpha)`** — total mutant molecules across the panel tested against total expected background, `P(X ≥ observed | Poisson(λ_bg))`. Efficient; has a closed form (see §4).
3. **`LikelihoodRatioRule(alpha)`** — compares H0 (background only) against H1 (background + tumour at MLE VAF) using per-site error rates. Most statistically efficient; the reference implementation.

Threshold behaviour: each rule must support **operating at a calibrated threshold** derived from blanks (§5, item 5) rather than a nominal alpha, since the empirical false-positive rate is what actually matters.

**Analytic availability (resolves the surface/dashboard rule question).** Not every rule has a fast closed form, and §7–§8 must be honest about which is running:

| Rule | Analytic path | Basis |
|---|---|---|
| `AggregatePoissonRule` | **exact** | sum of independent Poissons is Poisson (§4.2) |
| `KofNRule` | **exact (homogeneous)** | per-site hit probability `p` is a Poisson tail; sample-positive probability is a Poisson-binomial tail `P(K ≥ k)`, which reduces to `Binomial(N, p)` under panel-expectation inputs |
| `LikelihoodRatioRule` | **none** | no tractable closed form for the LR statistic's null/alternative tails |

Consequence, made explicit rather than left implicit: `surface.py` and the dashboard compute `AggregatePoissonRule` and `KofNRule` analytically. `LikelihoodRatioRule` is served in the MC path and as the production caller, but wherever a fast path is required it falls back to the **aggregate-Poisson proxy, labelled as such in the UI** — justified because the LR rule is the *most* efficient and the aggregate sits just below it at ppm scale, a relationship a test pins down (§10). Never present the proxy as the LR rule silently.

---

## 4. Two computation paths — `simulate.py` and `analytic.py`

### 4.1 Monte Carlo (`simulate.py`)
General-purpose generative core. Fully vectorised with numpy — draw arrays of shape `(n_replicates, n_variants)`, never Python loops over replicates. Seeded RNG via `numpy.random.Generator` for reproducibility. Returns detection outcomes plus the raw statistic distribution.

### 4.2 Analytic fast path (`analytic.py`)

Expose one entry point, `detection_probability(config, rule, vaf) -> float`, that dispatches by rule (§3, analytic-availability table). All inputs come from `mean_rate()` (§2.4.1) and panel expectations — never a median or a per-replicate draw.

**Aggregate rule.** The sum of independent Poissons is Poisson:

```
λ_signal = GE_eff × N × vaf × E[ccf]
λ_bg     = GE_eff × N × mean_rate(panel)
P(detect) = 1 − PoissonCDF(threshold − 1, λ_signal + λ_bg)
```

**KofN rule.** Per-site rate `λ_site = GE_eff × (vaf × E[ccf] + mean_rate)`, with per-site threshold `t` calibrated to `per_site_alpha` against `Poisson(GE_eff × mean_rate)`:

```
p   = 1 − PoissonCDF(t − 1, λ_site)          # per-site hit probability
P(detect) = 1 − BinomialCDF(k − 1, N, p)     # ≥ k of N sites positive
```

(The homogeneous `Binomial(N, p)` is the panel-expectation reduction of the exact Poisson-binomial; use the full Poisson-binomial only if per-site heterogeneity is later required in the fast path.)

This gives detection probability in **microseconds instead of seconds**, making the LoD surface (§7) and live dashboard (§8) tractable. Implement with `scipy.stats.poisson` / `scipy.stats.binom`.

**Analytic-valid regime (a hard precondition, enforced in code).** The closed forms represent only a restricted model. `analytic.py` must **raise** if handed a config it cannot faithfully represent, so the fast path can never silently diverge from the MC truth:

- uniform depth `depth_i = GE_eff` — **depth dispersion (§2.5) must be off**;
- **no per-site dropout** (`dropout_prob = 0`);
- error entering only through `mean_rate()` — per-site `eps` heterogeneity is collapsed to its expectation;
- CCF entering only through `E[ccf]`;
- Poisson signal, valid for `vaf ≪ 1` (below ~1%).

**Required consistency test (headline).** In the analytic-valid regime *only*, MC and analytic detection probabilities must agree within Monte Carlo error. Concretely: assert `|p_MC − p_analytic| < 4 × sqrt(p_analytic·(1 − p_analytic) / n_replicates)` across a VAF ladder, for **both** the aggregate and KofN rules, with `n_replicates` large enough (e.g. ≥ 20,000) to make the band tight. This single test validates both implementations and would catch the median-vs-mean error of §2.4.1. Outside the valid regime the analytic path is not expected to agree and must not be called.

---

## 5. Validation layer — `validate.py` (data-source agnostic)

All functions here consume `(level, n_replicates, n_detections)` records. **They must not import from `simulate.py`.**

1. **`fit_detection_curve(records)`** — probit (or logistic) regression of detection probability on `log10(vaf)`; returns fitted curve object.
2. **`lod_from_curve(curve, hit_rate=0.95)`** — the level at 95% detection, **with a bootstrap 95% confidence interval.** Never return a bare point estimate.
3. **`limit_of_blank(blank_statistics, alpha=0.05)`** — 95th percentile of the blank statistic distribution; provide both the non-parametric percentile and the CLSI parametric `mean + 1.645×SD`, and warn when they diverge (a normality-violation signal, common at ppm scale).
4. **`limit_of_quantification(records, max_cv=0.20)`** — lowest level meeting a precision criterion.
5. **`calibrate_threshold(blank_statistics, target_specificity)`** — returns the statistic threshold achieving the target specificity on blanks. **This must be run before LoD determination**, and the resulting LoD must always be reported alongside the specificity it assumes.
6. **`power_analysis(...)`** — replicates needed per level to achieve a target CI width on the LoD estimate. This is the experimental-design tool for the real study.

---

## 6. Calibration hooks — `calibrate.py` (the "later" path)

These are what make the tool durable. Implement now with clear interfaces even though real data doesn't exist yet; include synthetic round-trip tests.

- **`estimate_error_model(healthy_donor_counts) -> EmpiricalError`** — takes per-site mutant/total counts from a healthy-donor cohort and returns a fitted error distribution (with per-context breakdown when context is supplied). Replaces the literature prior.
- **`fit_conversion_efficiency(input_masses, observed_unique_molecules)`** — recovers the real conversion efficiency from QC data.
- **`compare_predicted_observed(config, observed_records)`** — the diagnostic. Overlays the model's predicted detection curve on observed experimental results and reports the discrepancy, with a structured hint at the likely cause: conversion efficiency below assumption, error floor higher than the consensus regime implies, correlated site dropout, or contamination.

**Required test:** simulate with known parameters → run calibration → recover those parameters within tolerance. This proves the "useful later" claim.

---

## 7. LoD surface — `surface.py`

Grid computation over two axes (default: panel size `N` × cfDNA input mass), returning achievable LoD per cell, faceted by error regime. For each cell, root-find with `scipy.optimize.brentq` on `vaf ↦ detection_probability(config, rule, vaf) − 0.95` to locate the 95%-detection point — do not simulate a VAF ladder per cell. The rule comes from §3's dispatch: aggregate and KofN compute exactly, LR uses the labelled aggregate proxy. Because the surface runs entirely in the analytic-valid regime (§4.2), depth dispersion and dropout are ignored here by construction — state this on the figure.

Also expose **`what_would_it_take(config, target_vaf)`** — given a configuration that misses the target, return the minimum change in each single lever (panel size, input mass, conversion efficiency, error regime) that would reach it, flagging which are physically plausible. This is the most decision-relevant output in the whole tool.

---

## 8. Interactive HTML dashboard — `report.py` + `templates/dashboard.html.j2`

**Audience: a wet-lab founder, not a programmer.** The dashboard must be legible in five seconds and answer "what configuration do we need?"

### 8.1 Technical constraints
- **A single self-contained `.html` file** that works offline from an email attachment (Plotly via CDN with a documented offline-bundling option).
- **Live computation in JavaScript.** Reimplement the analytic model (§4.2) in JS — molecule-budget arithmetic, a stable log-space Poisson CDF (guard overflow for large λ), and **both** the aggregate and KofN closed forms; the LR option reuses the aggregate curve behind a visible "aggregate-Poisson approximation" label (§3). Controls must respond instantly to *any* combination, not snap to a precomputed lattice. **Guard against Python/JS drift with a golden-vector test:** emit a small JSON of `(inputs → P(detect))` from `analytic.py`, embed it, and have the page assert its JS reproduces every row on load (and a Python test assert the same vectors). A physics change then cannot desync the dashboard from the package.
- Embed precomputed Monte-Carlo-validated curves as JSON for an optional overlay, demonstrating the JS analytic model agrees with the Python MC.
- No build step, no server, no framework. Vanilla JS.

### 8.2 Controls
| Control | Type | Range / options |
|---|---|---|
| cfDNA input | slider | 5–100 ng |
| Conversion efficiency | slider | 10–90 % |
| Panel size (N) | slider, **log scale** | 10–5,000 variants |
| Error regime | segmented control | Raw / SSCS / Duplex (+ free log-slider for custom ε) |
| Target specificity | dropdown | 95 / 99 / 99.5 / 99.9 % |
| Detection rule | dropdown | the three rules from §3 (LR carries an "approximate" tag — see §3 analytic-availability) |
| Target LoD | numeric input | default 0.001 % **TF** (see units rule below) |

**Units rule (resolves the factor-of-two trap of §2.2 at the UI boundary).** The dashboard is the one surface where a non-computational reader will confuse TF and VAF, so it is settled once, globally:

- The UI operates in **tumour fraction (TF)** — that is what a founder reasons about — and **every** TF figure carries an explicit "TF" suffix. The default target `0.001 %` is `0.001 % TF` = 10 ppm TF.
- The internal JS model works in **per-site VAF** (mirroring §2.2). Conversion happens at exactly one place, `VAF = TF / 2` for clonal heterozygous SNVs, and that relationship is printed inline near the target input (e.g. "10 ppm TF ≈ 5 ppm VAF").
- Any axis, readout, or prose that shows a VAF-native quantity (e.g. the detection-probability curve's x-axis) is labelled "VAF"; everything user-facing that isn't is labelled "TF". No unlabelled ppm/percent appears anywhere.

### 8.3 Outputs
1. **Hero readout** — achievable LoD in large type, with the assumed hit-rate and specificity stated beneath it, and a clear pass/miss indicator against the target.
2. **Molecule budget panel** — expected mutant molecules vs expected background molecules, and their ratio, as a small horizontal bar pair. *This is the single most intuitive output for a non-computational reader — it makes the physics obvious.*
3. **LoD surface** — contour plot (panel size × input mass), current settings marked with a crosshair, target LoD drawn as a highlighted contour.
4. **Detection probability curve** — P(detect) vs VAF at current settings, with the 95% line and the LoD point marked.
5. **Plain-language interpretation box** — auto-generated prose, e.g. *"At 30 ng input with 40% conversion, you interrogate ~3,600 effective genome equivalents. Across 500 tracked variants at 10 ppm you expect ~18 mutant molecules against ~0.2 background — comfortably detectable. Dropping to single-strand consensus raises background to ~18 molecules, and the target becomes unreachable."*
6. **"What would it take?" panel** — when the target is missed, the §7 lever analysis rendered as plain sentences.
7. **Poisson-floor warning** — a prominent, unmissable banner when expected mutant molecules < 3: *"Below the sampling floor — the molecules are not present in the sample. No additional sequencing depth can recover this."*

### 8.4 Design direction
Clean and clinical, not a data-science notebook. Restrained palette (one accent colour; use red/amber only for the floor warning and target-miss states), generous whitespace, a real type scale, responsive down to laptop width, all numbers with sensible significant figures and explicit units. Every plot needs an axis label a non-specialist can read. Assume it will be viewed on a projector in a meeting.

---

## 9. CLI — `cli.py` (typer)

```
uv run mrd-lod simulate   --config configs/default.toml --out results/
uv run mrd-lod surface    --config configs/default.toml --out results/
uv run mrd-lod dashboard  --config configs/default.toml --out outputs/dashboard.html
uv run mrd-lod calibrate  --background data/healthy.csv --out configs/calibrated.toml
uv run mrd-lod compare    --config configs/calibrated.toml --observed data/dilution.csv
```

Configuration via TOML so a non-programmer can edit scenarios. Ship 2–3 example configs (`conservative`, `target`, `optimistic`).

---

## 10. Tests — `tests/`

Prioritise scientific correctness over coverage percentage:

1. **MC vs analytic agreement** — in the analytic-valid regime (§4.2), `|p_MC − p_analytic| < 4·SE` across a VAF ladder for **both** the aggregate and KofN rules (headline test). A companion test asserts `analytic.py` **raises** when handed depth dispersion or dropout.
2. **Poisson floor** — detection probability collapses toward the false-positive rate when expected mutants < 1.
3. **Monotonicity** — LoD improves with larger N, higher input, higher conversion, lower ε. Property-based if convenient.
4. **Probit recovery** — synthetic detection data generated from a known LoD is recovered by the fitter within CI.
5. **Threshold calibration** — the calibrated threshold achieves the requested specificity on held-out blanks.
6. **Calibration round-trip** — simulate with known parameters, recover them via `calibrate.py`.
7. **Duplex trade-off** — verify the model correctly represents that duplex lowers ε *and* reduces usable molecules via `strand_recovery`, so the net benefit depends on regime.
8. **Determinism** — identical seeds produce identical results.
9. **Error-model expectation** — `LogNormalError.sample(...).mean()` converges to `mean_rate()`; guards the §2.4.1 median-vs-mean split.
10. **LR ≈ aggregate at ppm** — the LR rule's MC detection probability tracks the aggregate proxy within MC error in the ppm regime, justifying the dashboard/surface fallback (§3).

---

## 11. README requirements

Must include, in this order:

1. One-paragraph statement of what problem the tool solves and why simulation is *permanently* necessary — real samples cannot provide ground truth at ppm, so a specified-truth model is the only way to unit-test a detection rule, regression-test after a chemistry change, or re-tune a threshold. **This is the core argument; state it clearly.**
2. Quickstart with `uv`.
3. The hero contour figure.
4. The scientific model, with the equations from §2 written out.
5. **An explicit assumptions and limitations section.** Name what the model does *not* capture: independence across sites (real dropout and error are correlated), no clonal-haematopoiesis modelling, no fragment-size or biological cfDNA variation, idealised probe capture, tumour shedding assumed uniform. *Being precise about limitations is a feature, not a caveat.*
6. The three-phase roadmap (priors → empirical background → fitted from dilution data) showing which module serves each phase.
7. Citations for the literature-derived priors.

---

## 12. Constraints and non-goals

- **No proprietary or company-specific content.** Generic tool, generic naming, literature parameters only.
- **Do not implement** read alignment, UMI collapsing, variant calling, or BAM/FASTQ handling. The tool starts from per-site counts. State this scope boundary explicitly in the README.
- Keep the package small and readable. A reviewer should understand the whole thing in twenty minutes.
- Every scientific constant needs a docstring citation or an explicit "order-of-magnitude prior" label. **Never present an assumed value as a measurement.**

---

## 13. Build order

1. `params.py` + `molecules.py` + tests
2. `detect.py` (production interface first — it constrains everything downstream)
3. `analytic.py`, then `simulate.py`, then the agreement test
4. `validate.py` + tests
5. `surface.py` + `what_would_it_take`
6. `calibrate.py` + round-trip test
7. `report.py` + dashboard template
8. CLI, README, example configs

Commit at each step with clear messages.
