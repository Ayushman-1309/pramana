"""PRAMANA Web UI — Forecasting page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.fisher_forecast import (
    fisher_matrix_gaussian, forecast_errors, figure_of_merit,
    fisher_ellipse, compare_to_mcmc
)


def render():
    st.title("📈 Fisher Forecasting")

    # Data loading
    with st.expander("📁 Data Setup", expanded='pantheon_data' not in st.session_state):
        col1, col2 = st.columns(2)
        with col1:
            data_file = st.text_input("Pantheon+ data (.dat)", "data/pantheon/Pantheon+SH0ES.dat")
        with col2:
            cov_file = st.text_input("Covariance (.cov)", "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov")
        use_synthetic = st.checkbox("Use synthetic data", value='pantheon_data' not in st.session_state)

        if st.button("Load Data"):
            with st.spinner("Loading..."):
                try:
                    if use_synthetic:
                        z, mb_obs, cov = make_synthetic_dataset()
                        st.session_state['pantheon_data'] = {'z': z, 'mb_obs': mb_obs, 'cov': cov}
                        st.success("Synthetic data loaded")
                    else:
                        z, mb_obs, cov, _ = load_pantheon(data_file, cov_file)
                        st.session_state['pantheon_data'] = {'z': z, 'mb_obs': mb_obs, 'cov': cov}
                        st.success(f"Loaded {len(z)} SNe")
                except Exception as e:
                    st.error(f"Error: {e}")

    if 'pantheon_data' not in st.session_state:
        st.info("Please load data first.")
        return

    data = st.session_state['pantheon_data']
    z, mb_obs, cov = data['z'], data['mb_obs'], data['cov']
    cov_inv = np.linalg.inv(cov)

    # Model and fiducial
    model = st.selectbox("Model", list(MODEL_REGISTRY.keys()), format_func=lambda x: x.upper())
    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]

    st.subheader("Fiducial Parameters")
    fiducial = {}
    for p in param_names:
        lo, hi = priors[p]
        fiducial[p] = st.slider(f"{p}", float(lo), float(hi), float((lo+hi)/2))

    run_btn = st.button("🚀 Run Fisher Forecast", type="primary")
    if run_btn:
        theta_fid = np.array([fiducial[p] for p in param_names])

        def model_predictions(theta):
            params = dict(zip(param_names, theta))
            return spec["func"](z, **params)

        with st.spinner("Computing Fisher matrix..."):
            fisher = fisher_matrix_gaussian(model_predictions, theta_fid, cov_inv)
            errs, cov_mat = forecast_errors(fisher, param_names)

            st.session_state['fisher'] = fisher
            st.session_state['fisher_cov'] = cov_mat
            st.session_state['fisher_errs'] = errs
            st.session_state['fisher_theta_fid'] = theta_fid
            st.session_state['fisher_param_names'] = param_names

        # FoM
        if "w0" in param_names and "wa" in param_names:
            i, j = param_names.index("w0"), param_names.index("wa")
            fom = figure_of_merit(fisher, i, j)
            st.metric("FoM (w0-wa)", f"{fom:.2f}")

    # Results
    if 'fisher' in st.session_state:
        fisher = st.session_state['fisher']
        cov_mat = st.session_state['fisher_cov']
        errs = st.session_state['fisher_errs']
        theta_fid = st.session_state['fisher_theta_fid']
        param_names = st.session_state['fisher_param_names']

        st.markdown("---")
        st.subheader("Forecast Errors")
        for p in param_names:
            st.metric(p, f"σ = {errs[p]:.4g}")

        # Fisher ellipse
        st.subheader("Fisher Confidence Ellipses")
        if len(param_names) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                p_i = st.selectbox("Parameter X", param_names, key="fx")
            with col2:
                p_j = st.selectbox("Parameter Y", param_names, index=1 if len(param_names)>1 else 0, key="fy")

            if p_i != p_j:
                i, j = param_names.index(p_i), param_names.index(p_j)
                x, y = fisher_ellipse(cov_mat, i, j, (theta_fid[i], theta_fid[j]))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='1σ Fisher',
                                         line=dict(color='red', width=2)))
                fig.update_layout(xaxis_title=p_i, yaxis_title=p_j,
                                  title=f"Fisher 1σ Ellipse: {p_i} vs {p_j}",
                                  template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Compare to MCMC
        st.subheader("Fisher vs MCMC Comparison")
        mcmc_file = st.text_input("MCMC chain file (.npz) for comparison", "")
        if mcmc_file and st.button("Compare"):
            try:
                chain_data = np.load(mcmc_file)
                mcmc_chain = chain_data["chain"]
                compare_to_mcmc(errs, mcmc_chain, param_names)
            except Exception as e:
                st.error(f"Error: {e}")