# PRAMANA — Quick Reference Card

## CLI Commands

```bash
# Single-probe fits
pramana fit mcmc     --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--nwalkers N] [--nsteps N] [--out FILE] [--plot]
pramana fit nested   --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--nlive N] [--out FILE]
pramana fit nuts     --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--warmup N] [--samples N] [--chains N] [--out FILE] [--plot]
pramana fit profile  --model MODEL --sn-data FILE --sn-cov FILE --param PARAM [--synthetic] [--points N] [--out FILE]
pramana fit sbi      --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--sims N] [--samples N] [--out FILE]

# Joint fits
pramana joint fit    --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--bao/--no-bao] [--bao-H0 FLOAT] [--rd-mode MODE] [--nwalkers N] [--nsteps N] [--out FILE] [--plot] [--per-probe/--no-per-probe]
pramana joint compare --model MODEL --sn-data FILE --sn-cov FILE [--out FILE]

# Forecasting
pramana forecast run   --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--fiducial JSON] [--out FILE] [--compare-mcmc FILE]
pramana forecast ellipse --fisher FILE --param-i NAME --param-j NAME [--out FILE]

# Emulation
pramana emulate train  --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--n-train N] [--n-test N] [--out FILE]
pramana emulate predict --emulator FILE --theta JSON

# Compression
pramana compress run   --model MODEL --sn-data FILE --sn-cov FILE [--synthetic] [--fiducial JSON] [--out FILE] [--validate/--no-validate]

# Reweighting
pramana reweight run   --old-chain FILE --model MODEL [--sn-data FILE] [--sn-cov FILE] [--new-prior JSON] [--out FILE] [--n-samples N] [--plot]

# Tension
pramana tension h0     [--list] [--tension A B] [--plot] [--out FILE]
pramana tension s8     [--list] [--tension A B] [--plot] [--out FILE]
pramana tension append-sn --base-data FILE --z-new "z1,z2" --mb-new "m1,m2" --mb-err-new "e1,e2" [--out FILE]

# Diagnostics
pramana diagnose chain   --chain FILE [--model MODEL] [--burn-in FLOAT] [--gelman-rubin FILES] [--plot] [--out FILE]
pramana diagnose compare --chains FILES [--labels LABELS] [--out FILE]

# Data
pramana data pantheon --data FILE --cov FILE [--validate/--no-validate] [--synthetic] [--out FILE]
pramana data desi     [--table/--no-table] [--validate]
pramana data h0       [--list/--no-list]
pramana data s8       [--list/--no-list]

# Global
pramana --jax-backend auto|cpu|cuda|metal COMMAND
```

## Models & Parameters

| Model | Parameters | Priors |
|-------|------------|--------|
| lcdm | Ωₘ | [0.05, 0.6] |
| wcdm | Ωₘ, w | [0.05, 0.6], [-3.0, 0.0] |
| cpl | Ωₘ, w₀, wₐ | [0.05, 0.6], [-3.0, 1.0], [-3.0, 2.0] |

## Data Files

```
data/pantheon/
├── Pantheon+SH0ES.dat              # Download from GitHub
└── Pantheon+SH0ES_STAT+SYS.cov     # Download from GitHub

data/act/ (optional)
├── act_dr6_lenslike/
└── act_dr6_cmbonly/
```

## Web UI

```bash
streamlit run -m pramana.web.app
# http://localhost:8501
```

Pages: Home, Data Explorer, Single-Probe Fit, Joint Fit, Model Comparison, Forecasting, Emulation, Tension Analysis

## Key Theory

**Distance Modulus**: μ(z) = 25 + 5 log₁₀(d_L/Mpc)

**ΛCDM**: E(z)² = Ωₘ(1+z)³ + (1-Ωₘ)

**wCDM**: E(z)² = Ωₘ(1+z)³ + (1-Ωₘ)(1+z)³⁽¹⁺ʷ⁾

**CPL**: w(a) = w₀ + wₐ(1-a), E(z)² = Ωₘ(1+z)³ + (1-Ωₘ)(1+z)³⁽¹⁺ʷ⁰⁺ʷᵃ⁾ exp(-3wₐz/(1+z))

**SN Likelihood**: χ²_marg = δᵀC⁻¹δ - (1ᵀC⁻¹δ)²/(1ᵀC⁻¹1)  [M_B/H₀ marginalized]

**BAO**: D_M/r_d, D_H/r_d, D_V/r_d from DESI DR2 table (13 points)

**Joint**: ln L = ln L_SN + ln L_BAO + ln L_CMB  [H₀ caveat: SN marginalized, BAO sensitive]

## GPU Support

```bash
# NVIDIA
uv pip install "jaxlib==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Apple Metal
uv pip install jax-metal
```

## Common Workflows

```bash
# Quick test
pramana fit mcmc --model lcdm --synthetic

# Full analysis
pramana fit mcmc --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/... --nwalkers 32 --nsteps 8000
pramana joint fit --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/... --bao
pramana forecast run --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/... --fiducial '{"Om":0.3,"w0":-1,"wa":0}'
pramana tension h0 --plot --out h0.png
pramana tension s8 --plot --out s8.png
```

## Validation

```bash
pytest tests/test_validation.py -v
```

All 12 tests should pass.