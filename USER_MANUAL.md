# PRAMANA — Unified Cosmological Inference Suite
## Complete User Manual

> **Sanskrit *pramāṇa* (प्रमाण)**: a means of valid knowledge, the epistemological question of how you actually justify that something is true.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Theoretical Framework](#3-theoretical-framework)
4. [Data Preparation](#4-data-preparation)
5. [Command-Line Interface](#5-command-line-interface)
6. [Web Interface](#6-web-interface)
7. [Inference Methods](#7-inference-methods)
8. [Models & Priors](#8-models--priors)
9. [Advanced Workflows](#9-advanced-workflows)
10. [Troubleshooting](#10-troubleshooting)
11. [API Reference](#11-api-reference)

---

## 1. Installation

### 1.1 Requirements

- **Python**: 3.12+ (tested on 3.14)
- **Package Manager**: `uv` (recommended) or `pip`
- **OS**: Linux, macOS, Windows

### 1.2 Standard Installation

```bash
# Clone or navigate to project
cd pramana

# Install with uv (fast, modern)
uv sync

# Or with pip
pip install -e ".[all]"
```

This installs all core dependencies:
- `numpy`, `scipy`, `pandas` — numerical computing
- `emcee`, `dynesty` — MCMC & nested sampling
- `jax`, `jaxlib`, `numpyro` — differentiable models & HMC
- `scikit-learn` — GP emulation
- `torch`, `sbi` — simulation-based inference
- `camb`, `cobaya` — CMB theory (optional)
- `corner`, `getdist` — plotting
- `streamlit`, `plotly` — web UI
- `typer`, `rich` — CLI

### 1.3 GPU Acceleration (Optional)

For faster HMC/NUTS sampling with JAX:

```bash
# NVIDIA CUDA 12
uv pip install "jaxlib==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Apple Silicon (Metal)
uv pip install jax-metal
```

The CLI auto-detects GPU: `auto` → CUDA → Metal → CPU.

### 1.4 CMB Data (Optional)

For ACT DR6 likelihoods, download from [NASA LAMBDA](https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/):

```bash
# Lensing likelihood
pip install act_dr6_lenslike  # from local wheel/tarball

# Primary CMB (ACT-lite)
pip install "act_dr6_cmbonly @ git+https://github.com/ACTCollaboration/DR6-ACT-lite.git"
uv add cobaya
```

Place data in `data/act/`:
```
data/act/
├── act_dr6_lenslike/
└── act_dr6_cmbonly/
```

### 1.5 Development Installation

```bash
uv sync --group dev
# or
pip install -e ".[dev]"
```

Includes: `pytest`, `ruff`, `mypy`, `pre-commit`

---

## 2. Quick Start

### 2.1 Synthetic Data Test (No Downloads Needed)

```bash
# MCMC fit with synthetic data
pramana fit mcmc --model lcdm --synthetic --nwalkers 16 --nsteps 2000

# Nested sampling (evidence)
pramana fit nested --model lcdm --synthetic --nlive 200

# Joint SN+BAO fit
pramana joint fit --model lcdm --synthetic --bao

# Fisher forecast
pramana forecast run --model lcdm --synthetic

# H0 tension
pramana tension h0 --list
pramana tension h0 --tension "Planck 2018" "SH0ES"
```

### 2.2 With Real Data

```bash
# 1. Download Pantheon+SH0ES data
#    https://github.com/PantheonPlusSH0ES/DataRelease
#    Place in data/pantheon/
#    Pantheon+SH0ES.dat
#    Pantheon+SH0ES_STAT+SYS.cov

# 2. Run MCMC
pramana fit mcmc --model lcdm \
    --sn-data data/pantheon/Pantheon+SH0ES.dat \
    --sn-cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov \
    --nwalkers 32 --nsteps 8000

# 3. Joint SN+BAO
pramana joint fit --model cpl \
    --sn-data data/pantheon/Pantheon+SH0ES.dat \
    --sn-cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov \
    --bao
```

### 2.3 Web Interface

```bash
streamlit run -m pramana.web.app
# Opens at http://localhost:8501
```

---

## 3. Theoretical Framework

### 3.1 Cosmological Models

All models assume **flat FRW geometry** (standard Pantheon+ assumption). The distance modulus is:

$$\mu(z) = 25 + 5 \log_{10}\left(\frac{d_L(z)}{\text{Mpc}}\right)$$

where luminosity distance:

$$d_L(z) = (1+z) \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}$$

with $E(z) = H(z)/H_0$.

#### 3.1.1 ΛCDM (lcdm)
$$E(z)^2 = \Omega_m (1+z)^3 + (1-\Omega_m)$$
- **Parameters**: $\Omega_m \in [0.05, 0.6]$
- **Degeneracy**: $H_0$ and $M_B$ perfectly degenerate → fixed at $H_0=70$, analytically marginalized

#### 3.1.2 wCDM (wcdm) — Constant $w$
$$E(z)^2 = \Omega_m (1+z)^3 + (1-\Omega_m)(1+z)^{3(1+w)}$$
- **Parameters**: $\Omega_m \in [0.05, 0.6]$, $w \in [-3.0, 0.0]$

#### 3.1.3 CPL / w₀wₐCDM (cpl) — Time-varying $w$
$$w(a) = w_0 + w_a(1-a), \quad a = \frac{1}{1+z}$$

Dark energy density evolution:
$$\frac{\rho_{DE}(z)}{\rho_{DE}(0)} = (1+z)^{3(1+w_0+w_a)} \exp\left(-\frac{3w_a z}{1+z}\right)$$

$$E(z)^2 = \Omega_m (1+z)^3 + (1-\Omega_m) \times [\text{above}]$$
- **Parameters**: $\Omega_m \in [0.05, 0.6]$, $w_0 \in [-3.0, 1.0]$, $w_a \in [-3.0, 2.0]$

### 3.2 Likelihood Functions

#### 3.2.1 SN Ia Marginalized Gaussian Likelihood

Standard SN cosmology trick (Conley+2011, Betoule+2014, Pantheon+):

Observed magnitude: $m_{B,\text{obs}}(z) = \mu_{\text{model}}(z; \theta) + M_B + \text{noise}$

The offset $A = M_B$ (absorbs $M_B$ and $H_0$ mismatch) is analytically marginalized:

$$\chi^2_{\text{marg}} = \delta^T C^{-1} \delta - \frac{(\mathbf{1}^T C^{-1} \delta)^2}{\mathbf{1}^T C^{-1} \mathbf{1}}$$

where $\delta = m_{B,\text{obs}} - \mu_{\text{model}}(\theta)$, $C$ = covariance matrix.

**Log-likelihood**: $\ln\mathcal{L} = -\frac{1}{2}\chi^2_{\text{marg}}$

#### 3.2.2 BAO Likelihood (DESI DR2)

DESI DR2 provides 13 compressed measurements across 7 tracers ($0.295 \leq z \leq 2.33$):

Observables: $D_M/r_d$, $D_H/r_d$, $D_V/r_d$

- $D_M$ = comoving angular diameter distance
- $D_H = c/H(z)$
- $D_V = [z D_M^2 D_H]^{1/3}$
- $r_d$ = sound horizon at baryon drag epoch

Covariance: block-diagonal (intra-bin $D_M$-$D_H$ correlation $\rho_{MH}$, zero inter-bin)

**Log-likelihood**: $\ln\mathcal{L}_{\text{BAO}} = -\frac{1}{2} (\mathbf{d} - \mathbf{\mu})^T C^{-1} (\mathbf{d} - \mathbf{\mu})$

#### 3.2.3 Sound Horizon

Two options:

1. **Eisenstein & Hu (1998) fitting formula** (fast, ~1-3% accuracy):
   $$r_d = \frac{2}{3k_{\text{eq}}} \sqrt{\frac{6}{R_{\text{eq}}}} \ln\left(\frac{\sqrt{1+R_d} + \sqrt{R_d+R_{\text{eq}}}}{1+\sqrt{R_{\text{eq}}}}\right)$$
   where $R = 31.5 \Omega_b h^2 \Theta_{2.7}^{-4} (z/10^3)^{-1}$

2. **CAMB exact** (slow, precise): Boltzmann solver output

### 3.3 Joint Likelihood

Independence assumption (standard practice):
$$\ln\mathcal{L}_{\text{joint}} = \ln\mathcal{L}_{\text{SN}} + \ln\mathcal{L}_{\text{BAO}} + \ln\mathcal{L}_{\text{CMB}}$$

**H₀ Caveat**: SN is $H_0$-marginalized (fiducial $H_0=70$), BAO genuinely measures absolute distances via $r_d$ and IS sensitive to $H_0$. For real joint fits, either:
- Add $H_0$ as shared free parameter (recommended), or
- Fix $H_0$ to external CMB prior

---

## 4. Data Preparation

### 4.1 Pantheon+SH0ES

Download from: https://github.com/PantheonPlusSH0ES/DataRelease

Place in `data/pantheon/`:
```
data/pantheon/
├── Pantheon+SH0ES.dat              # Whitespace-delimited ASCII
└── Pantheon+SH0ES_STAT+SYS.cov     # Covariance: first line N, then N×N values
```

**Key columns used**:
- `zHD` — Hubble-diagram redshift (CMB frame)
- `m_b_corr` — Bias-corrected apparent B-band magnitude
- `IS_CALIBRATOR` — 1 if Cepheid host (excluded for shape-only fit)

### 4.2 DESI DR2 BAO

Built-in table (`bao_desi.DESI_DR2_BAO_TABLE`) — no download needed.

| Tracer | $z_{\text{eff}}$ | Observables |
|--------|------------------|-------------|
| BGS | 0.295 | $D_V/r_d$ |
| LRG1 | 0.510 | $D_M/r_d$, $D_H/r_d$ |
| LRG2 | 0.706 | $D_M/r_d$, $D_H/r_d$ |
| LRG3+ELG1 | 0.934 | $D_M/r_d$, $D_H/r_d$ |
| ELG2 | 1.321 | $D_M/r_d$, $D_H/r_d$ |
| QSO | 1.484 | $D_M/r_d$, $D_H/r_d$ |
| Lya | 2.330 | $D_M/r_d$, $D_H/r_d$ |

### 4.3 ACT DR6 CMB

Requires download from LAMBDA. Place in `data/act/`:
```
data/act/
├── act_dr6_lenslike/     # Lensing likelihood data
└── act_dr6_cmbonly/      # ACT-lite primary CMB data
```

### 4.4 JWST High-z SNe

Append new discoveries to Pantheon+:
```bash
pramana tension append-sn \
    --base-data data/pantheon.npz \
    --z-new "1.5,1.8,2.1" \
    --mb-new "26.5,27.2,27.8" \
    --mb-err-new "0.15,0.18,0.20"
```

---

## 5. Command-Line Interface

### 5.1 Global Options

```bash
pramana [OPTIONS] COMMAND [ARGS]...

Options:
  --jax-backend TEXT    JAX backend: auto, cpu, cuda, metal (default: auto)
  --help                Show help
```

Environment variable: `PRAMANA_JAX_BACKEND`

### 5.2 Commands Overview

| Command | Purpose |
|---------|---------|
| `fit` | Single-probe parameter inference |
| `joint` | Multi-probe joint fits |
| `forecast` | Fisher matrix forecasting |
| `emulate` | GP emulator training/validation |
| `compress` | MOPED optimal compression |
| `reweight` | Importance resampling |
| `tension` | H₀/S₈ tension analysis |
| `diagnose` | Convergence diagnostics |
| `data` | Data loading/validation |

### 5.3 Fit Commands

#### 5.3.1 MCMC (emcee)
```bash
pramana fit mcmc [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --nwalkers INT      Number of walkers (default: 32)
  --nsteps INT        MCMC steps (default: 4000)
  --out PATH          Output .npz file (default: chain.npz)
  --seed INT          Random seed (default: 42)
  --synthetic         Use synthetic data
  --plot              Generate corner plot
```

#### 5.3.2 Nested Sampling (dynesty)
```bash
pramana fit nested [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --nlive INT         Live points (default: 500)
  --out PATH          Output .npz file
  --seed INT          Random seed (default: 42)
  --synthetic         Use synthetic data
```

#### 5.3.3 NUTS/HMC (numpyro)
```bash
pramana fit nuts [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --warmup INT        Warmup steps (default: 1000)
  --samples INT       Samples per chain (default: 2000)
  --chains INT        Number of chains (default: 2)
  --out PATH          Output .npz file
  --seed INT          Random seed (default: 0)
  --synthetic         Use synthetic data
  --plot              Generate corner plot
```

#### 5.3.4 Profile Likelihood
```bash
pramana fit profile [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --param TEXT        Parameter to profile (Om, w, w0, wa)
  --points INT        Scan points (default: 30)
  --out PATH          Output .npz file
  --synthetic         Use synthetic data
```

#### 5.3.5 SBI (Neural Posterior Estimation)
```bash
pramana fit sbi [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --sims INT          Training simulations (default: 2000)
  --samples INT       Posterior samples (default: 5000)
  --out PATH          Output .npz file
  --seed INT          Random seed (default: 0)
  --synthetic         Use synthetic data
```

### 5.4 Joint Fit

```bash
pramana joint fit [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --synthetic         Use synthetic data
  --bao / --no-bao    Include DESI DR2 BAO (default: yes)
  --bao-H0 FLOAT      H0 for BAO (default: 70.0)
  --rd-mode TEXT      rd mode: eh98, free, planck_prior (default: eh98)
  --nwalkers INT      Walkers (default: 32)
  --nsteps INT        Steps (default: 8000)
  --out PATH          Output .npz file
  --seed INT          Random seed (default: 42)
  --plot              Generate corner plot
  --per-probe / --no-per-probe  Print per-probe χ² (default: yes)
```

### 5.5 Model Comparison

```bash
pramana joint compare [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --out PATH          Output .npz file
```
Computes Bayes factors between models using nested sampling.

### 5.6 Fisher Forecast

```bash
pramana forecast run [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --fiducial JSON     Fiducial params: '{"Om": 0.3, "w0": -1, "wa": 0}'
  --out PATH          Output .npz file
  --synthetic         Use synthetic data
  --compare-mcmc PATH MCMC chain for Fisher vs MCMC comparison
```

```bash
pramana forecast ellipse [OPTIONS]
  --fisher PATH       Fisher .npz file
  --param-i TEXT      Parameter X
  --param-j TEXT      Parameter Y
  --out PATH          Output plot
```

### 5.7 GP Emulation

```bash
pramana emulate train [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --n-train INT       Training points (default: 200)
  --n-test INT        Test points (default: 50)
  --out PATH          Output .pkl file
  --synthetic         Use synthetic data
```

```bash
pramana emulate predict [OPTIONS]
  --emulator PATH     Trained emulator .pkl file
  --theta JSON        Parameters: '{"Om": 0.3, "w0": -1, "wa": 0}'
```

### 5.8 MOPED Compression

```bash
pramana compress run [OPTIONS]
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      Pantheon+ .dat file
  --sn-cov PATH       Covariance .cov file
  --fiducial JSON     Fiducial params
  --out PATH          Output .npz file
  --synthetic         Use synthetic data
  --validate / --no-validate  Validate vs full likelihood (default: yes)
```

**Note**: MOPED compresses RAW Gaussian likelihood ($\chi^2 = \delta^T C^{-1} \delta$), NOT the analytically marginalized SN likelihood. The skill file documents this caveat.

### 5.9 Importance Resampling

```bash
pramana reweight run [OPTIONS]
  --old-chain PATH    Original chain .npz file
  --model TEXT        Model: lcdm, wcdm, cpl
  --sn-data PATH      New SN data .dat file
  --sn-cov PATH       New covariance .cov file
  --new-prior JSON    New priors: '{"Om": [0.1, 0.5]}'
  --out PATH          Output .npz file
  --n-samples INT     Resampled chain size
  --plot              Generate corner plot
```

Reweights existing chain to new likelihood/prior without re-running sampler.

### 5.10 Tension Analysis

```bash
pramana tension h0 [OPTIONS]
  --list              List H₀ measurements
  --tension A B       Compute tension between A and B
  --plot              Generate whisker plot
  --out PATH          Output plot file
```

```bash
pramana tension s8 [OPTIONS]
  --list              List S₈ measurements
  --tension A B       Compute tension between A and B
  --plot              Generate whisker plot
  --out PATH          Output plot file
```

```bash
pramana tension append-sn [OPTIONS]
  --base-data PATH    Base Pantheon+ data .npz or files
  --base-cov PATH     Base covariance (if not in .npz)
  --z-new TEXT        New SN redshifts (comma-separated)
  --mb-new TEXT       New SN magnitudes (comma-separated)
  --mb-err-new TEXT   New SN magnitude errors (comma-separated)
  --out PATH          Output .npz file
```

### 5.11 Diagnostics

```bash
pramana diagnose chain [OPTIONS]
  --chain PATH        Chain .npz file
  --model TEXT        Model name (if not in chain)
  --burn-in FLOAT     Burn-in fraction (default: 0.3)
  --gelman-rubin PATH Additional chains for R-hat
  --plot              Generate corner plot
  --out PATH          Output plot file
```

```bash
pramana diagnose compare [OPTIONS]
  --chains PATHS      Chain .npz files to compare
  --labels TEXT       Labels for each chain
  --out PATH          Output triangle plot
```

### 5.12 Data Exploration

```bash
pramana data pantheon [OPTIONS]
  --data PATH         Pantheon+ .dat file
  --cov PATH          Covariance .cov file
  --validate / --no-validate  Validate format (default: yes)
  --stats             Print statistics (default: yes)
  --synthetic         Generate synthetic data
  --out PATH          Save as .npz
```

```bash
pramana data desi [OPTIONS]
  --table / --no-table  Show DESI DR2 BAO table (default: yes)
  --validate          Validate built-in table
```

```bash
pramana data h0 [OPTIONS]
  --list / --no-list  List H₀ measurements (default: yes)
```

```bash
pramana data s8 [OPTIONS]
  --list / --no-list  List S₈ measurements (default: yes)
```

---

## 6. Web Interface

### 6.1 Launch

```bash
streamlit run -m pramana.web.app
```

### 6.2 Pages

| Page | Purpose |
|------|---------|
| **Home** | Overview, quick start, feature summary |
| **Data Explorer** | Load/validate Pantheon+, DESI BAO, view H₀/S₈ measurements |
| **Single-Probe Fit** | Interactive MCMC/Nested/NUTS/Profile/SBI on SN data |
| **Joint Fit** | Combined SN+BAO with per-probe χ² breakdown |
| **Model Comparison** | Bayes factors, evidence, posterior overlays |
| **Forecasting** | Fisher matrix + MCMC validation ellipses, FoM |
| **Emulation** | GP training, validation, speed benchmarks |
| **Tension Analysis** | H₀/S₈ whisker plots, append JWST high-z SNe |

### 6.3 Workflow Example

1. **Data Explorer** → Load Pantheon+ data
2. **Single-Probe Fit** → Select model, method, run fit
3. **Joint Fit** → Enable BAO, run joint fit
4. **Model Comparison** → Compare LCDM vs CPL with Bayes factors
5. **Forecasting** → Fisher forecast + MCMC comparison
6. **Tension Analysis** → View H₀/S₈ whisker plots

---

## 7. Inference Methods

### 7.1 MCMC (emcee)
- **Algorithm**: Affine-invariant ensemble sampler
- **Best for**: Low-dimensional (≤5 params), simple posteriors
- **Output**: Weighted chain, diagnostics (τ, acceptance)
- **Diagnostics**: `summarize()`, `gelman_rubin()`

### 7.2 Nested Sampling (dynesty)
- **Algorithm**: Static nested sampling
- **Best for**: Model comparison (Bayesian evidence), multimodal posteriors
- **Output**: Evidence $\ln Z$, weighted posterior samples
- **Model comparison**: `bayes_factor()` with Jeffreys scale

### 7.3 HMC/NUTS (numpyro)
- **Algorithm**: No-U-Turn Sampler with autodiff gradients
- **Best for**: High-dimensional (≥8 params), strong degeneracies
- **Requires**: JAX (CPU/GPU), differentiable models
- **Output**: Multi-chain samples, automatic R-hat

### 7.4 Profile Likelihood
- **Algorithm**: L-BFGS-B optimization per scan point
- **Best for**: Frequentist cross-check, prior-independent CIs
- **Output**: Profile scan, Wilks' theorem CIs (Δχ² thresholds)

### 7.5 SBI (Neural Posterior Estimation)
- **Algorithm**: Neural Posterior Estimator (normalizing flows)
- **Best for**: Intractable likelihoods, complex systematics
- **Output**: Amortized posterior, fast sampling for new data
- **Validation**: `validate_on_synthetic()` coverage test

### 7.6 Fisher Forecast
- **Algorithm**: $F_{ij} = J^T C^{-1} J$ (Gaussian approximation)
- **Best for**: Survey design, quick sensitivity estimates
- **Output**: Marginalized errors, FoM, confidence ellipses
- **Validation**: `compare_to_mcmc()` flags non-Gaussianity

### 7.7 GP Emulation
- **Algorithm**: Gaussian Process (sklearn, RBF kernel)
- **Best for**: Expensive theory (CAMB), MCMC acceleration
- **Output**: Fast surrogate, uncertainty estimates
- **Validation**: Leave-out test, calibration check

### 7.8 MOPED Compression
- **Algorithm**: Optimal linear compression (Heavens+2000)
- **Best for**: High-dimensional data (CMB $C_\ell$), speed
- **Output**: $P$ compressed numbers ($P$ = parameters)
- **Caveat**: Valid for RAW Gaussian likelihood only

### 7.9 Importance Resampling
- **Algorithm**: $w_i = \mathcal{L}_{\text{new}}/\mathcal{L}_{\text{old}}$
- **Best for**: "What if" questions, quick approximate updates
- **Diagnostic**: Effective sample size (ESS) — trust if ESS > 5% of chain

---

## 8. Models & Priors

### 8.1 Default Priors (MODEL_REGISTRY)

| Model | Parameter | Prior Range | LaTeX Label |
|-------|-----------|-------------|-------------|
| lcdm | $\Omega_m$ | [0.05, 0.6] | $\Omega_m$ |
| wcdm | $\Omega_m$ | [0.05, 0.6] | $\Omega_m$ |
| wcdm | $w$ | [-3.0, 0.0] | $w$ |
| cpl | $\Omega_m$ | [0.05, 0.6] | $\Omega_m$ |
| cpl | $w_0$ | [-3.0, 1.0] | $w_0$ |
| cpl | $w_a$ | [-3.0, 2.0] | $w_a$ |

### 8.2 Adding New Models

Edit `src/pramana/core/models.py`:
1. Add `e_of_z_newmodel(z, *params)` function
2. Add `distance_modulus_newmodel(z, *params, H0=70.0)` wrapper
3. Register in `MODEL_REGISTRY` with params, priors, labels

All downstream modules (likelihood, MCMC, diagnostics, plotting) auto-discover.

### 8.3 Prior Modification

**CLI**: `--new-prior '{"Om": [0.1, 0.5]}'` (reweight command)

**Web**: Prior Editor panel in Single-Probe Fit

**Python**:
```python
from pramana.core.models import MODEL_REGISTRY
MODEL_REGISTRY["cpl"]["priors"]["wa"] = (-2.0, 1.0)  # Tighten
```

---

## 9. Advanced Workflows

### 9.1 Full Analysis Pipeline

```bash
# 1. Explore data
pramana data pantheon --data data/pantheon/Pantheon+SH0ES.dat --cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov

# 2. Single-probe fits for each model
for m in lcdm wcdm cpl; do
    pramana fit mcmc --model $m --sn-data ... --sn-cov ... --out ${m}_chain.npz
done

# 3. Model comparison (Bayes factors)
pramana joint compare --model lcdm --sn-data ... --sn-cov ...

# 4. Best model joint fit with BAO
pramana joint fit --model cpl --sn-data ... --sn-cov ... --bao --out joint_cpl.npz

# 5. Fisher forecast for future survey
pramana forecast run --model cpl --sn-data ... --sn-cov ... --fiducial '{"Om":0.3,"w0":-1,"wa":0}'

# 6. Tension analysis
pramana tension h0 --plot --out h0_tension.png
pramana tension s8 --plot --out s8_tension.png
```

### 9.2 Emulator-Accelerated MCMC

```bash
# 1. Train GP emulator on model predictions
pramana emulate train --model cpl --sn-data ... --sn-cov ... --n-train 500

# 2. Use emulator in MCMC (requires code modification)
#    Replace model_func with emulator.predict() in likelihood
```

### 9.3 MOPED-Compressed Joint Analysis

```bash
# 1. Compress SN data
pramana compress run --model cpl --sn-data ... --sn-cov ... --out moped.npz

# 2. Use compressed likelihood (1000x faster for CMB-scale data)
#    Requires matching RAW Gaussian likelihood
```

### 9.4 Posterior Reweighting

```bash
# 1. Run baseline MCMC
pramana fit mcmc --model cpl --sn-data ... --sn-cov ... --out baseline.npz

# 2. Reweight to different prior
pramana reweight run --old-chain baseline.npz --model cpl --new-prior '{"wa": [-1, 1]}' --out reweighted.npz

# 3. Reweight to new dataset (when available)
pramana reweight run --old-chain baseline.npz --model cpl --sn-data new_data.dat --sn-cov new_cov.cov --out updated.npz
```

### 9.5 JWST High-z SN Extension

```bash
# Append new JWST SNe to Pantheon+
pramana tension append-sn \
    --base-data data/pantheon.npz \
    --z-new "1.5,1.8,2.1" \
    --mb-new "26.5,27.2,27.8" \
    --mb-err-new "0.15,0.18,0.20" \
    --out extended.npz

# Use extended data in fits
pramana fit mcmc --model cpl --sn-data extended.npz --sn-cov extended.npz ...
```

---

## 10. Troubleshooting

### 10.1 Common Issues

| Issue | Solution |
|-------|----------|
| `ImportError: CAMB not installed` | `uv add camb` |
| `ImportError: act_dr6_lenslike not installed` | Download from LAMBDA, install locally |
| `JAX backend error` | Set `--jax-backend cpu` or install CUDA `jaxlib` |
| `FileNotFoundError: Pantheon+SH0ES.dat` | Download data, place in `data/pantheon/` |
| `Covariance shape mismatch` | Ensure same calibrator rows dropped from data & cov |
| `MCMC acceptance > 0.6` | Increase initial spread: `--nwalkers` or check priors |
| `Chain < 50×τ` | Increase `--nsteps` (try 20000+) |
| `R-hat > 1.01` | Run more chains, increase steps |
| `SBI training slow` | Reduce `--sims`, use GPU |
| `GP emulator overconfident` | Increase `alpha` (WhiteKernel noise) |

### 10.2 Performance Tips

- **MCMC**: Use `nwalkers ≈ 2-4×ndim`, `nsteps > 50×τ`
- **NUTS**: Use `num_chains ≥ 2`, `num_warmup ≈ num_samples`
- **Nested**: `nlive ≥ 500` for evidence, `1000+` for posteriors
- **Fisher**: Validate with `compare_to_mcmc` before trusting
- **GP**: Use Latin hypercube design, validate on held-out points
- **MOPED**: Only for RAW Gaussian likelihood (not marginalized SN)

### 10.3 Debug Mode

```bash
# Verbose output
PYTHONWARNINGS=always pramana fit mcmc --model lcdm --synthetic 2>&1 | tee log.txt

# Python interactive debugging
python -c "
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import make_synthetic_dataset
from pramana.core.likelihood import log_likelihood
import numpy as np
z, mb, cov = make_synthetic_dataset(50)
spec = MODEL_REGISTRY['lcdm']
ll = log_likelihood([0.3], z, mb, np.linalg.inv(cov), spec['func'], spec['params'])
print('Log-likelihood:', ll)
"
```

---

## 11. API Reference

### 11.1 Core Modules

```python
# Models
from pramana.core.models import MODEL_REGISTRY, distance_modulus_lcdm, distance_modulus_wcdm, distance_modulus_cpl

# Data I/O
from pramana.core.data_io import load_pantheon, make_synthetic_dataset

# Likelihood
from pramana.core.likelihood import log_likelihood, log_prior, log_probability

# MCMC
from pramana.core.mcmc import run_fit

# Diagnostics
from pramana.core.diagnostics import summarize, gelman_rubin

# Plotting
from pramana.core.plotting import corner_plot, getdist_triangle, compare_hubble_diagram

# BAO
from pramana.core.bao_desi import DESI_DR2_BAO_TABLE, build_data_vector_and_cov, sound_horizon_rd, log_likelihood_bao

# JWST
from pramana.core.jwst_probes import append_supernovae, H0_MEASUREMENTS, S8_MEASUREMENTS, h0_tension_sigma, s8_tension_sigma, plot_h0_whisker, plot_s8_whisker

# Nested Sampling
from pramana.core.nested_sampling import run_nested, equal_weight_posterior, bayes_factor

# Fisher
from pramana.core.fisher_forecast import fisher_matrix_gaussian, forecast_errors, figure_of_merit, fisher_ellipse, compare_to_mcmc

# Profile Likelihood
from pramana.core.profile_likelihood import profile_scan, confidence_interval_from_profile, global_best_fit

# GP Emulator
from pramana.core.gp_emulator import latin_hypercube_design, train_emulator, emulate, validate_emulator

# SBI
from pramana.core.sbi_inference import make_simulator, train_npe, sample_posterior, validate_on_synthetic

# Importance Resampling
from pramana.core.importance_resampling import importance_weights, effective_sample_size, reweight_chain, weighted_quantiles, resample_to_equal_weight

# MOPED
from pramana.core.data_compression import moped_vectors, compress, compressed_log_likelihood, compare_compressed_vs_full

# Joint Likelihood
from pramana.core.joint_likelihood import build_joint_log_probability, per_probe_chi2

# Differentiable Models (JAX)
from pramana.core.differentiable_models import JAX_MODEL_REGISTRY, check_gradient

# HMC/NUTS
from pramana.core.hmc_numpyro import build_sn_model, run_nuts, samples_to_flat_chain

# CMB Theory
from pramana.core.camb_theory import get_cmb_theory, sound_horizon_camb, hubble_of_z_camb

# ACT DR6
from pramana.core.cmb_act import act_dr6_lensing_loglike, act_dr6_lensing_load_data, build_act_cmbonly_model, act_cmbonly_loglike
```

### 11.2 Utilities

```python
from pramana.utils.jax_config import configure_jax, get_jax_backend
from pramana.utils.optional_imports import get_camb, get_act_lenslike, get_act_cmbonly, get_cobaya, get_sbi, get_dynesty, get_emcee, get_numpyro, get_corner, get_getdist
from pramana.utils.validators import validate_pantheon_data, validate_pantheon_cov, validate_desi_bao_file, validate_act_data_dir
```

### 11.3 Example: Custom MCMC

```python
import numpy as np
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon
from pramana.core.likelihood import log_probability
import emcee

# Load data
z, mb_obs, cov, _ = load_pantheon("data/pantheon/Pantheon+SH0ES.dat", "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov")
cov_inv = np.linalg.inv(cov)

# Model
spec = MODEL_REGISTRY["cpl"]
param_names = spec["params"]
priors = spec["priors"]
ndim = len(param_names)

# Initial positions
rng = np.random.default_rng(42)
p0_center = np.array([np.mean(priors[p]) for p in param_names])
p0_spread = np.array([(priors[p][1] - priors[p][0]) * 0.05 for p in param_names])
p0 = p0_center + p0_spread * rng.normal(size=(32, ndim))

# Run
sampler = emcee.EnsembleSampler(32, ndim, log_probability,
    args=(z, mb_obs, cov_inv, spec["func"], param_names, priors))
sampler.run_mcmc(p0, 8000, progress=True)

# Analyze
from pramana.core.diagnostics import summarize
flat_chain, tau = summarize(sampler, param_names)
```

### 11.4 Example: Joint SN+BAO

```python
from pramana.core.joint_likelihood import build_joint_log_probability, per_probe_chi2
from pramana.core.data_io import load_pantheon
import numpy as np
import emcee

# Load data
z, mb_obs, cov, _ = load_pantheon("data/pantheon/Pantheon+SH0ES.dat", "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov")

# Probes
probes = [
    {"kind": "sn", "z": z, "mb_obs": mb_obs, "cov_inv": np.linalg.inv(cov)},
    {"kind": "bao", "H0": 70.0, "rd_mode": "eh98"},
]

# Joint log-probability
log_prob, param_names = build_joint_log_probability("cpl", probes)

# Run MCMC
spec = MODEL_REGISTRY["cpl"]
ndim = len(param_names)
rng = np.random.default_rng(42)
p0 = np.array([np.mean(spec["priors"][p]) for p in param_names]) + \
     np.array([(spec["priors"][p][1] - spec["priors"][p][0]) * 0.05 for p in param_names]) * \
     rng.normal(size=(32, ndim))

sampler = emcee.EnsembleSampler(32, ndim, log_prob)
sampler.run_mcmc(p0, 8000, progress=True)

# Per-probe χ² at best fit
best = np.median(sampler.get_chain(discard=2400, flat=True), axis=0)
per_probe_chi2("cpl", best, probes)
```

---

## Appendix: Validation Results

The suite includes validated cross-checks (from skill file):

| Test | Expected | Verified |
|------|----------|----------|
| MCMC vs NUTS consistency | Ωₘ=0.30-0.32, w₀≈-1 | ✅ |
| Nested sampling Occam penalty | ln K = 2.00 ± 0.28 (ΛCDM vs CPL) | ✅ |
| Fisher vs MCMC non-Gaussianity | wₐ ratio ~3.7× flagged | ✅ |
| GP emulator calibration | pred σ/residual ~2.5 | ✅ |
| MOPED accuracy | max diff < 0.1 vs full likelihood | ✅ |
| SBI coverage | true θ in 68% CI | ✅ |
| Joint SN+BAO degeneracy breaking | σ(Ωₘ): 0.100→0.013 | ✅ |

Run validation: `pytest tests/test_validation.py -v`

---

## License

MIT License — see LICENSE file.

---

## Citation

If you use PRAMANA in research, please cite the original PRAMANA skill methodology and the data sources:
- Pantheon+SH0ES: Scolnic+2022, Brout+2022
- DESI DR2 BAO: Abdul-Karim+2025
- ACT DR6: ACT Collaboration
- JWST H₀: Riess+2025, Freedman+2025
- JWST S₈: DES, KiDS, HSC collaborations

---

*Last updated: 2026 | PRAMANA v2.0.0 · Developed by Ayushman*