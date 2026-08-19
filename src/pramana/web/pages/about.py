"""PRAMANA Web UI — About page."""
import streamlit as st
from pathlib import Path
from pramana.web.components.ui import render_status_bar, VERSION, DEVELOPER, SUITE_NAME


def render():
    render_status_bar()
    st.title("About PRAMANA")

    tab_manual, tab_theory, tab_dev = st.tabs([
        "📖 User Manual", "📐 Theory & Formulas", "👨‍💻 Developer & Credits"
    ])

    # ─── User Manual ───
    with tab_manual:
        st.markdown("### PRAMANA User Manual (GUI / Web Interface)")
        manual_path = Path(__file__).parent.parent.parent.parent / "USER_MANUAL.md"
        if manual_path.exists():
            manual_text = manual_path.read_text(encoding="utf-8")
            st.markdown(manual_text)
        else:
            st.warning("USER_MANUAL.md not found in project root.")
        st.markdown("---")
        st.download_button(
            "⬇ Download Full Manual (.md)",
            data=manual_path.read_bytes() if manual_path.exists() else b"",
            file_name="USER_MANUAL.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ─── Theory & Formulas ───
    with tab_theory:
        st.markdown("### Theoretical Framework & Key Equations")
        st.markdown("""
**PRAMANA** implements a unified cosmological inference suite spanning Type Ia supernovae (SN Ia), 
Baryon Acoustic Oscillations (BAO), Cosmic Microwave Background (CMB), JWST-era probes, 
and weak gravitational lensing (Euclid, Rubin/LSST). The following summarizes the core mathematical 
foundations.

#### 1. Cosmic Background Expansion
**Friedmann Equation** (flat $\\Lambda$CDM / $w$CDM / CPL):
$$
H(z) = H_0 \\sqrt{\\Omega_m (1+z)^3 + \\Omega_r (1+z)^4 + \\Omega_k (1+z)^2 + \\Omega_{\\rm DE} \\exp\\left(3\\int_0^z \\frac{1+w(z')}{1+z'} dz'\\right)}
$$

**CPL Dark Energy Equation of State** (Chevallier–Polarski–Linder):
$$
w(a) = w_0 + w_a (1-a) = w_0 + w_a \\frac{z}{1+z}
$$

**Comoving Distance**:
$$
\\chi(z) = \\frac{c}{H_0} \\int_0^z \\frac{dz'}{E(z')}
$$

**Angular Diameter Distance**: $D_A = \\chi/(1+z)$

**Luminosity Distance**: $D_L = (1+z) \\chi$

**Distance Modulus**: $\\mu = 25 + 5 \\log_{10}(D_L/{\\rm Mpc})$

#### 2. Supernova Ia Likelihood (Pantheon+)
Analytic marginalization over absolute magnitude $M_B$ (Conley et al. 2011):
$$
\\mathcal{L}(\\theta) = \\frac{1}{\\sqrt{(2\\pi)^N \\det C}} 
\\exp\\left[-\\frac{1}{2} (\\Delta - A/C \\cdot \\mathbf{1})^T C^{-1} (\\Delta - A/C \\cdot \\mathbf{1})\\right]
$$
where $\\Delta = m_{B,\\rm obs} - \\mu_{\\rm model}(\\theta)$, 
$A = \\mathbf{1}^T C^{-1} \\Delta$, $C = \\mathbf{1}^T C^{-1} \\mathbf{1}$.

#### 3. BAO Likelihood (DESI DR2)
Distance ratios $D_M/r_d$, $D_H/r_d$, $D_V/r_d$ at effective redshifts.
Block-diagonal covariance from published DESI DR2 compressed data vector (13 points, 7 tracers).
Model predictions via $E(z)$ and sound horizon $r_d$ (EH98 or CAMB).

#### 4. CMB Likelihood (ACT DR6)
Official wrapper around `act_dr6_lenslike` (lensing) and `act_dr6_cmbonly` (primary via Cobaya).
Theory spectra from CAMB: $C_\\ell^{TT,EE,TE,BB,\\kappa\\kappa}$ in $\\mu$K$^2$.

#### 5. Weak Lensing / Cosmic Shear (Euclid / Rubin LSST)
**Limber Approximation** for tomographic shear cross-spectra:
$$
C_\\ell^{ij} = \\int_0^{\\chi_H} \\frac{d\\chi}{\\chi^2} \\, W_i(\\chi) W_j(\\chi) \\, 
P_{\\delta}\\left(k=\\frac{\\ell+1/2}{\\chi}, z(\\chi)\\right)
$$

**Lensing Efficiency Kernel**:
$$
W_i(\\chi) = \\frac{3}{2} \\Omega_m \\left(\\frac{H_0}{c}\\right)^2 \\chi \\, (1+z) 
\\int_\\chi^{\\chi_H} d\\chi' \\, n_i(\\chi') \\frac{\\chi'-\\chi}{\\chi'}
$$

**Intrinsic Alignment (NLA model)**:
$$
F(z) = -A_{\\rm IA} C_1 \\rho_{\\rm crit} \\frac{\\Omega_m}{D(z)} 
\\left(\\frac{1+z}{1+z_{\\rm pivot}}\\right)^{\\eta_{\\rm IA}}
$$

**Gaussian Covariance (Knox formula)**:
$$
{\\rm Cov}[C_\\ell^{ij}, C_\\ell^{kl}] = 
\\frac{C_\\ell^{ik} C_\\ell^{jl} + C_\\ell^{il} C_\\ell^{jk}}
{(2\\ell+1) \\Delta\\ell \\, f_{\\rm sky}}
$$

**Survey Presets**: Euclid (10 bins, 30 gal/arcmin$^2$), LSST Y1 (5 bins, 10 gal/arcmin$^2$), 
LSST Y10 (5 bins, 27 gal/arcmin$^2$).

#### 6. Statistical Methods
- **MCMC** (emcee): Affine-invariant ensemble sampler
- **Nested Sampling** (dynesty): Bayesian evidence $Z$, model comparison via Bayes factors
- **HMC/NUTS** (numpyro/JAX): Gradient-based sampling with auto-diff
- **Profile Likelihood**: Frequentist confidence intervals via Wilks' theorem
- **Fisher Forecasting**: $F_{ij} = J_i^T C^{-1} J_j$, FoM = $1/\\sqrt{\\det \\Sigma_{w_0,w_a}}$
- **GP Emulation** (sklearn): Matérn-5/2 kernel surrogate for expensive theory
- **SBI** (sbi/NPE): Likelihood-free neural posterior estimation
- **MOPED Compression**: Optimal linear compression to $P$ numbers per parameter
- **Importance Resampling**: Reweight chains to new priors/likelihoods
""")
        st.markdown("---")
        theory_path = Path(__file__).parent.parent.parent.parent / "docs" / "PRAMANA_Theory.md"
        if theory_path.exists():
            st.download_button(
                "⬇ Download Full Theory Document (.md)",
                data=theory_path.read_bytes(),
                file_name="PRAMANA_Theory.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("Full theory document will be available at docs/PRAMANA_Theory.md")

    # ─── Developer & Credits ───
    with tab_dev:
        st.markdown("### Developer & Credits")
        st.markdown(f"""
**Suite Name**: {SUITE_NAME}  
**Version**: v{VERSION}  
**Developer**: {DEVELOPER}  
**GitHub**: [https://github.com/Ayushman-1309/pramana.git](https://github.com/Ayushman-1309/pramana.git)  
**License**: MIT  

#### Data Release Attributions
- **Pantheon+SH0ES**: Scolnic et al. 2022, [GitHub DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease)
- **DESI DR2 BAO**: Abdul-Karim et al. 2025, [DESI Data](https://data.desi.lbl.gov/doc/releases/dr2/)
- **ACT DR6**: [NASA LAMBDA](https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/)
- **Euclid**: ESA Euclid Consortium, [Euclid EC](https://www.euclid-ec.org/)
- **Rubin/LSST**: LSST DESC, [LSST](https://www.lsst.org/)

#### Core Dependencies
- numpy, scipy, pandas — numerical computing
- emcee, dynesty — MCMC & nested sampling
- jax, jaxlib, numpyro — differentiable models & HMC/NUTS
- scikit-learn — GP emulation
- torch, sbi — simulation-based inference
- camb, cobaya — CMB theory (optional)
- corner, getdist — posterior plotting
- streamlit, plotly — web UI
- typer, rich — CLI
""")
        st.markdown("---")
        st.markdown("""
#### PRAMANA Design Principles
1. **No bundled observational data** — every probe loaded via manual download or synthetic generation.
2. **Validated cross-checks** — each inference method verified against at least one other.
3. **Official wrappers** — ACT/JWST likelihoods wrap official packages, not reimplementations.
4. **Forward-looking infrastructure** — Weak lensing (Euclid/LSST) validated on synthetic data only.
5. **Reproducible** — versioned, git-tracked, with persistent diagnostics and footer attribution.
""")