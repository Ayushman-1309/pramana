"""PRAMANA Web UI — Home page."""
import streamlit as st


def render():
    st.title("🔭 PRAMANA — Unified Cosmological Inference Suite")
    st.markdown("""
    **Sanskrit *pramāṇa* (प्रमाण)**: a means of valid knowledge — the epistemological question of
    how you actually justify that something is true.

    PRAMANA is a comprehensive cosmological inference toolkit spanning:
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
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models", "3", "ΛCDM, wCDM, CPL")
        st.metric("Probes", "4", "SN, BAO, CMB, JWST")
        st.metric("Inference Methods", "9", "MCMC, Nested, NUTS, Profile, SBI, Fisher, GP, MOPED, Reweight")

    with col2:
        st.markdown("### 🚀 Quick Start")
        st.code("""
# CLI usage
pramana fit --method mcmc --model lcdm \
    --sn-data data/pantheon/Pantheon+SH0ES.dat \
    --sn-cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov

# Joint SN+BAO fit
pramana joint --model cpl --sn-data ... --sn-cov ... --bao
        """, language="bash")

    with col3:
        st.markdown("### 📖 Key Features")
        st.markdown("""
        - **Validated pipelines** — cross-checked methods (MCMC vs NUTS, Fisher vs MCMC, etc.)
        - **Real data** — DESI DR2 BAO table built-in, Pantheon+ loaders
        - **Official CMB wrappers** — ACT DR6 via act_dr6_lenslike/act_dr6_cmbonly
        - **JWST tension** — H₀/S₈ whisker plots with latest measurements
        - **Web + CLI** — interactive UI and scriptable command line
        """)

    st.markdown("---")
    st.markdown("### 📋 Navigation")
    st.markdown("""
    | Page | Purpose |
    |------|---------|
    | **Data Explorer** | Load/validate Pantheon+, DESI BAO, view H₀/S₈ measurements |
    | **Single-Probe Fit** | Run MCMC/Nested/NUTS/Profile/SBI on SN data |
    | **Joint Fit** | Combined SN+BAO(+CMB) with per-probe χ² breakdown |
    | **Model Comparison** | Bayes factors, evidence, posterior overlays |
    | **Forecasting** | Fisher matrix + MCMC validation ellipses |
    | **Emulation** | GP training, validation, speed benchmarks |
    | **Tension Analysis** | H₀/S₈ whisker plots, append JWST high-z SNe |
    """)