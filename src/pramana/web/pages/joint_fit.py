"""PRAMANA Web UI — Joint Fit page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon
from pramana.core.joint_likelihood import build_joint_log_probability, per_probe_chi2
from pramana.core.mcmc import run_fit as run_mcmc
import emcee


def render():
    st.title("🔗 Joint Multi-Probe Fit")

    # Data loading
    with st.expander("📁 Data Setup", expanded='pantheon_data' not in st.session_state):
        col1, col2 = st.columns(2)
        with col1:
            data_file = st.text_input("Pantheon+ data (.dat)", "data/pantheon/Pantheon+SH0ES.dat")
        with col2:
            cov_file = st.text_input("Covariance (.cov)", "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov")

        if st.button("Load SN Data"):
            with st.spinner("Loading..."):
                try:
                    z, mb_obs, cov, _ = load_pantheon(data_file, cov_file)
                    st.session_state['pantheon_data'] = {'z': z, 'mb_obs': mb_obs, 'cov': cov}
                    st.success(f"Loaded {len(z)} SNe")
                except Exception as e:
                    st.error(f"Error: {e}")

    if 'pantheon_data' not in st.session_state:
        st.info("Please load SN data first.")
        return

    data = st.session_state['pantheon_data']
    z, mb_obs, cov = data['z'], data['mb_obs'], data['cov']

    # Probe selection
    st.subheader("Probe Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_sn = st.checkbox("SN (Pantheon+)", value=True)
    with col2:
        use_bao = st.checkbox("BAO (DESI DR2)", value=True)
    with col3:
        use_cmb = st.checkbox("CMB (ACT DR6)", value=False, disabled=True, help="Requires ACT data download")

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
            probes.append({"kind": "bao", "H0": bao_H0, "rd_mode": rd_mode})

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

            st.session_state['joint_chain'] = flat_chain
            st.session_state['joint_model'] = model
            st.session_state['joint_probes'] = probes
            st.session_state['joint_param_names'] = pnames
            st.success(f"Done! {flat_chain.shape[0]} samples")

    # Results
    if 'joint_chain' in st.session_state:
        chain = st.session_state['joint_chain']
        model_name = st.session_state['joint_model']
        probes = st.session_state['joint_probes']
        pnames = st.session_state['joint_param_names']
        spec = MODEL_REGISTRY[model_name]

        st.markdown("---")
        st.subheader("Results")

        # Parameter table
        cols = st.columns(len(pnames))
        for i, p in enumerate(pnames):
            med = np.median(chain[:, i])
            lo, hi = np.percentile(chain[:, i], [16, 84])
            cols[i].metric(p, f"{med:.4f}", f"+{hi-med:.4f}/-{med-lo:.4f}")

        # Per-probe chi2
        if st.button("Show Per-Probe χ²"):
            best = np.median(chain, axis=0)
            per_probe_chi2(model_name, best, probes)

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
            fig.update_layout(title=f"{p} marginal", height=200, showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)