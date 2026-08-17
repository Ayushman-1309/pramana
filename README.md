# PRAMANA v2.0.0 — Unified Cosmological Inference Suite

> Sanskrit *pramāṇa* (प्रमाण): a means of valid knowledge — the epistemological question of how you actually justify that something is true.

**Developed by Ayushman** · [MIT License](LICENSE)

A comprehensive cosmological inference toolkit spanning:
- **SN Ia** (Pantheon+SH0ES)
- **BAO** (DESI DR2)
- **CMB** (ACT DR6 lensing + primary)
- **JWST-era probes** (high-z SNe, H₀ tension, S₈ tension)

With multiple inference engines:
- **MCMC** (emcee)
- **Nested Sampling** (dynesty) — Bayesian evidence
- **HMC/NUTS** (numpyro/JAX) — gradient-based
- **Profile Likelihood** — frequentist cross-check
- **Simulation-Based Inference** (sbi) — likelihood-free
- **Fisher Forecasting** — fast Gaussian approximations
- **GP Emulation** — fast surrogates for expensive theory
- **MOPED Compression** — optimal data compression
- **Importance Resampling** — fast posterior reweighting

---

## Installation

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e ".[all]"
```

### GPU Support (Optional)

```bash
# NVIDIA CUDA 12
uv pip install "jaxlib==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Apple Metal
uv pip install jax-metal
```

### Optional Dependencies

- **CMB (ACT)**: `uv add camb cobaya` + download `act_dr6_lenslike`/`act_dr6_cmbonly` from [NASA LAMBDA](https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/)
- **Development**: `uv sync --group dev`

---

## Quick Start

### CLI

```bash
# Single-probe MCMC fit
pramana fit --method mcmc --model lcdm \
    --sn-data data/pantheon/Pantheon+SH0ES.dat \
    --sn-cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov

# Nested sampling (evidence + posteriors)
pramana fit --method nested --model cpl \
    --sn-data data/pantheon/... --sn-cov data/pantheon/...

# NUTS/HMC (requires JAX)
pramana fit --method nuts --model wcdm \
    --sn-data data/pantheon/... --sn-cov data/pantheon/...

# Joint SN+BAO fit
pramana joint --model cpl \
    --sn-data data/pantheon/... --sn-cov data/pantheon/... --bao

# Fisher forecast
pramana forecast --model cpl \
    --sn-data data/pantheon/... --sn-cov data/pantheon/...

# GP emulator
pramana emulate --model cpl --n-train 200

# MOPED compression
pramana compress --model cpl

# H₀/S₈ tension
pramana tension --h0 --plot
pramana tension --s8 --plot

# Append JWST high-z SNe
pramana tension append-sn --base-data data.npz \
    --z-new "1.5,1.8" --mb-new "26.5,27.2" --mb-err-new "0.15,0.18"
```

### Web UI

```bash
streamlit run -m pramana.web.app
```

Then navigate to `http://localhost:8501` for interactive analysis.

---

## Data Setup

Place your data files in the `data/` directory:

```
data/
├── pantheon/
│   ├── Pantheon+SH0ES.dat
│   └── Pantheon+SH0ES_STAT+SYS.cov
├── desi/          # Optional - built-in table used by default
└── act/           # Optional - for ACT DR6 likelihoods
    ├── act_dr6_lenslike/
    └── act_dr6_cmbonly/
```

Download from:
- **Pantheon+**: https://github.com/PantheonPlusSH0ES/DataRelease
- **ACT DR6**: https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/

---

## Project Structure

```
pramana/
├── src/pramana/
│   ├── core/           # 20 core modules (exact logic from PRAMANA skill)
│   │   ├── models.py              # LCDM, wCDM, CPL + registry
│   │   ├── data_io.py             # Pantheon+ loaders, synthetic data
│   │   ├── likelihood.py          # SN marginalized Gaussian likelihood
│   │   ├── mcmc.py                # emcee runner
│   │   ├── diagnostics.py         # Convergence, gelman-rubin
│   │   ├── plotting.py            # Corner, getdist, Hubble diagram
│   │   ├── bao_desi.py            # DESI DR2 BAO likelihood
│   │   ├── camb_theory.py         # CAMB wrapper (exact theory)
│   │   ├── cmb_act.py             # ACT DR6 lensing + primary
│   │   ├── jwst_probes.py         # High-z SNe, H₀/S₈ tension
│   │   ├── nested_sampling.py     # dynesty evidence/model comparison
│   │   ├── fisher_forecast.py     # Fisher matrix + validation
│   │   ├── profile_likelihood.py  # Frequentist profiling
│   │   ├── differentiable_models.py # JAX models for gradients
│   │   ├── hmc_numpyro.py         # NUTS/HMC sampler
│   │   ├── gp_emulator.py         # GP emulation (sklearn)
│   │   ├── sbi_inference.py       # Simulation-based inference (sbi)
│   │   ├── importance_resampling.py # Chain reweighting
│   │   ├── data_compression.py    # MOPED compression
│   │   └── joint_likelihood.py    # Multi-probe combiner
│   ├── cli/            # Typer-based CLI commands
│   ├── web/            # Streamlit web UI
│   └── utils/          # JAX config, optional imports, validators
├── data/               # User data files (gitignored)
├── tests/              # Validation tests
├── examples/           # Example scripts/notebooks
└── docs/               # Documentation
```

---

## Validation

The suite includes validated cross-checks from the PRAMANA skill:

```bash
# Run validation tests
pytest tests/test_validation.py -v
```

Key validated results:
- MCMC vs NUTS: consistent posteriors (Ωₘ=0.30-0.32, w₀≈-1)
- Nested sampling Occam penalty: ln K = 2.00 ± 0.28 (ΛCDM vs CPL)
- Fisher vs MCMC: flags non-Gaussianity (wₐ ratio ~3.7×)
- GP emulator calibration: predicted σ/residual ~2.5
- MOPED accuracy: max diff < 0.001 vs full likelihood
- SBI coverage: true θ in 68% CI
- Joint SN+BAO: degeneracy breaking (σ(Ωₘ): 0.100→0.013)

---

*PRAMANA v2.0.0 · Developed by Ayushman*
