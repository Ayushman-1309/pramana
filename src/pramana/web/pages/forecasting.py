"""PRAMANA Web UI — Forecasting page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.fisher_forecast import (
    fisher_matrix_gaussian, forecast_errors, figure_of_merit,
    fisher_ellipse, compare_to_mcmc
)
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.ui import plotly_template, render_status_bar


def render():
    render_status_bar()
    st.title("Fisher Forecasting")

    # Data loading using shared component
    pantheon_loader(key="pantheon_data", show_instructions=False)

    if "pantheon_data" not in st.session_state:
        st.info("Please load data first using the Data Explorer or the section above.")
        return

    data = st.session_state["pantheon_data"]
    z, mb_obs, cov = data["z"], data["mb_obs"], data["cov"]
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

            st.session_state["fisher"] = fisher
            st.session_state["fisher_cov"] = cov_mat
            st.session_state["fisher_errs"] = errs
            st.session_state["fisher_theta_fid"] = theta_fid
            st.session_state["fisher_param_names"] = param_names

        # FoM
        if "w0" in param_names and "wa" in param_names:
            i, j = param_names.index("w0"), param_names.index("wa")
            fom = figure_of_merit(fisher, i, j)
            st.metric("FoM (w0-wa)", f"{fom:.2f}")

    # Results
    if "fisher" in st.session_state:
        fisher = st.session_state["fisher"]
        cov_mat = st.session_state["fisher_cov"]
        errs = st.session_state["fisher_errs"]
        theta_fid = st.session_state["fisher_theta_fid"]
        param_names = st.session_state["fisher_param_names"]

        st.markdown("---")
        st.subheader("Forecast Errors")
        
        # Display as table
        err_data = []
        for p in param_names:
            err_data.append({"Parameter": p, "σ (Fisher)": f"{errs[p]:.4g}"})
        st.dataframe(pd.DataFrame(err_data), use_container_width=True, hide_index=True)

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
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="1σ Fisher",
                                         line=dict(color="#ff7f0e", width=2)))
                fig.update_layout(xaxis_title=p_i, yaxis_title=p_j,
                                  title=f"Fisher 1σ Ellipse: {p_i} vs {p_j}",
                                  template=plotly_template(), height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Compare to MCMC
        st.subheader("Fisher vs MCMC Comparison")
        st.markdown("Upload an MCMC chain (.npz) to compare Fisher predictions with actual MCMC errors.")
        
        mcmc_file = st.file_uploader("MCMC chain file (.npz)", type=["npz"], key="fisher_mcmc_upload")
        if mcmc_file is not None:
            try:
                chain_data = np.load(mcmc_file)
                mcmc_chain = chain_data["chain"]
                # Get comparison results
                comp_results = _compare_to_mcmc(errs, mcmc_chain, param_names)
                
                if comp_results:
                    st.markdown("**Comparison Results:**")
                    comp_data = []
                    for name, f_err, mcmc_err, ratio in comp_results:
                        comp_data.append({
                            "Parameter": name,
                            "Fisher σ": f"{f_err:.4g}",
                            "MCMC σ": f"{mcmc_err:.4g}",
                            "Ratio (F/M)": f"{ratio:.2f}"
                        })
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
                    
                    # Flag non-Gaussianity
                    for name, f_err, mcmc_err, ratio in comp_results:
                        if ratio > 2.0:
                            st.warning(f"⚠️ {name}: Fisher underestimates error by {ratio:.1f}× — non-Gaussian posterior!")
            except Exception as e:
                st.error(f"Error: {e}")


def _compare_to_mcmc(fisher_errs: dict, mcmc_chain: np.ndarray, param_names: list) -> list:
    """Compare Fisher errors to MCMC errors, returning list of (name, f_err, mcmc_err, ratio)."""
    results = []
    for i, name in enumerate(param_names):
        f_err = fisher_errs[name]
        mcmc_err = np.percentile(mcmc_chain[:, i], 84) - np.percentile(mcmc_chain[:, i], 16)
        mcmc_err /= 2  # convert 68% interval to 1-sigma equivalent
        ratio = f_err / mcmc_err if mcmc_err > 0 else np.inf
        results.append((name, f_err, mcmc_err, ratio))
    return results