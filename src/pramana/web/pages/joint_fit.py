"""PRAMANA Web UI — Joint Fit page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pramana.core.models import MODEL_REGISTRY
from pramana.core.joint_likelihood import build_joint_log_probability, per_probe_chi2
from pramana.core.mcmc import run_fit as run_mcmc
import emcee
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.ui import plotly_template, render_status_bar, plot_export_controls, export_downloads


def render():
    render_status_bar()
    st.title("Joint Multi-Probe Fit")

    # Data loading using shared component
    pantheon_loader(key="pantheon_data", show_instructions=False)

    if "pantheon_data" not in st.session_state:
        st.info("Please load SN data first using the Data Explorer or the section above.")
        return

    data = st.session_state["pantheon_data"]
    z, mb_obs, cov = data["z"], data["mb_obs"], data["cov"]

    # Probe selection
    st.subheader("Probe Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_sn = st.checkbox("SN (Pantheon+)", value=True)
    with col2:
        use_bao = st.checkbox("BAO (DESI DR2)", value=True)
    with col3:
        cmb_ready = "cmb_data" in st.session_state
        use_cmb = st.checkbox("CMB (ACT DR6)", value=False,
                              disabled=not cmb_ready,
                              help="Load/generate CMB data in the Data Hub first (ACT DR6)")

    if use_bao and "bao_data" not in st.session_state:
        st.info("Load or generate BAO data in the **Data Hub** to use it in the joint fit. "
                "The shipped DESI DR2 reference table will be used as fallback.")
    if use_cmb and not cmb_ready:
        st.info("CMB requires ACT DR6 data — load or generate it in the **Data Hub**.")

    # Model selection
    model = st.selectbox("Model", list(MODEL_REGISTRY.keys()), format_func=lambda x: x.upper())
    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]

    # BAO options
    if use_bao:
        with st.expander("BAO Options"):
            bao_H0 = st.number_input("H₀ for BAO (if not fitting)", 50.0, 90.0, 70.0)
            rd_mode = st.selectbox("Sound horizon mode", ["eh98", "free", "planck_prior"])

    # MCMC settings
    col1, col2 = st.columns(2)
    with col1:
        nwalkers = st.number_input("Walkers", 16, 200, 32)
    with col2:
        nsteps = st.slider("Steps", 1000, 50000, 8000, step=1000)

    run_btn = st.button("🚀 Run Joint Fit", type="primary")
    if run_btn:
        probes = []
        if use_sn:
            probes.append({"kind": "sn", "z": z, "mb_obs": mb_obs, "cov_inv": np.linalg.inv(cov)})
        if use_bao:
            bao_probe = {"kind": "bao", "H0": bao_H0, "rd_mode": rd_mode}
            if "bao_data" in st.session_state:
                bd = st.session_state["bao_data"]
                bao_probe.update({
                    "labels": bd["labels"], "z_arr": bd["z_arr"],
                    "data": bd["data"], "cov": bd["cov"],
                })
            probes.append(bao_probe)
        if use_cmb:
            st.warning("CMB joint fitting requires the ACT DR6 likelihood packages "
                       "(act_dr6_lenslike / cobaya) which are not installed — "
                       "CMB probe skipped for this run.")

        log_prob, pnames = build_joint_log_probability(model, probes, priors)
        ndim = len(pnames)

        with st.spinner("Running joint MCMC..."):
            rng = np.random.default_rng(42)
            p0_center = np.array([np.mean(priors[p]) for p in pnames])
            p0_spread = np.array([(priors[p][1] - priors[p][0]) * 0.05 for p in pnames])
            p0 = p0_center + p0_spread * rng.normal(size=(nwalkers, ndim))

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
            sampler.run_mcmc(p0, nsteps, progress=True)

            burn_in = int(nsteps * 0.3)
            flat_chain = sampler.get_chain(discard=burn_in, flat=True)

            st.session_state["joint_chain"] = flat_chain
            st.session_state["joint_model"] = model
            st.session_state["joint_probes"] = probes
            st.session_state["joint_param_names"] = pnames
            st.success(f"✅ Done! {flat_chain.shape[0]} samples")

    # Results
    if "joint_chain" in st.session_state:
        chain = st.session_state["joint_chain"]
        model_name = st.session_state["joint_model"]
        probes = st.session_state["joint_probes"]
        pnames = st.session_state["joint_param_names"]
        spec = MODEL_REGISTRY[model_name]

        st.markdown("---")
        st.subheader("Results")

        # Parameter table
        cols = st.columns(len(pnames))
        for i, p in enumerate(pnames):
            med = np.median(chain[:, i])
            lo, hi = np.percentile(chain[:, i], [16, 84])
            cols[i].metric(p, f"{med:.4f}", f"+{hi-med:.4f}/-{med-lo:.4f}")

        # Export chain
        chain_df = pd.DataFrame(chain, columns=pnames)
        export_downloads(chain_df, f"joint_chain_{model_name}")

        # Per-probe chi2
        if st.button("Show Per-Probe χ²"):
            best = np.median(chain, axis=0)
            # Get per-probe chi2
            chi2_results = _get_per_probe_chi2(model_name, best, probes)
            if chi2_results:
                st.subheader("Per-Probe χ² Breakdown")
                for probe_name, chi2_val, n_data in chi2_results:
                    st.metric(probe_name, f"χ² = {chi2_val:.2f}", f"{n_data} data points")
                chi2_df = pd.DataFrame(chi2_results, columns=["Probe", "χ²", "N_data"])
                export_downloads(chi2_df, f"joint_chi2_{model_name}")

        # Corner plot
        if st.button("Generate Corner Plot"):
            with st.spinner("Generating..."):
                from pramana.core.plotting import corner_plot
                fig = corner_plot(chain, pnames, spec["labels"])
                st.pyplot(fig)

        # 1D marginals
        st.subheader("1D Marginals")
        for i, p in enumerate(pnames):
            fig = go.Figure()
            hist, bins = np.histogram(chain[:, i], bins=50, density=True)
            fig.add_trace(go.Bar(x=(bins[:-1]+bins[1:])/2, y=hist, name=p, opacity=0.7))
            fig.update_layout(title=f"{p} marginal", height=200, showlegend=False, template=plotly_template())
            plot_export_controls(fig, f"joint_marginal_{p}_{model_name}")
            st.plotly_chart(fig, use_container_width=True)


def _get_per_probe_chi2(model_name: str, theta: np.ndarray, probes: list[dict]) -> list[tuple[str, float, int]]:
    """Get per-probe chi2 values as a list of (name, chi2, n_data)."""
    from pramana.core.models import MODEL_REGISTRY
    from pramana.core.likelihood import log_likelihood as sn_log_likelihood
    from pramana.core.bao_desi import log_likelihood_bao
    import numpy as np
    
    spec = MODEL_REGISTRY[model_name]
    param_names = spec["params"]
    
    results = []
    for probe in probes:
        if probe["kind"] == "sn":
            ll = sn_log_likelihood(theta, probe["z"], probe["mb_obs"], probe["cov_inv"], spec["func"], param_names)
            n_data = len(probe["z"])
            name = "SN (Pantheon+)"
        elif probe["kind"] == "bao":
            bao_kwargs = {"H0": probe.get("H0", 70.0), "rd_mode": probe.get("rd_mode", "eh98")}
            if "labels" in probe:
                bao_kwargs.update({
                    "labels": probe["labels"], "z_arr": probe["z_arr"],
                    "data": probe["data"], "cov": probe["cov"],
                })
            ll = log_likelihood_bao(theta, spec["e_of_z"], param_names, **bao_kwargs)
            n_data = len(bao_kwargs.get("data", np.arange(13)))
            name = f"BAO (DESI DR2, rd={probe.get('rd_mode', 'eh98')})"
        else:
            continue
        chi2 = -2 * ll
        results.append((name, chi2, n_data))
    return results