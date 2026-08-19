"""PRAMANA Web UI — MOPED Compression page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_compression import moped_vectors, compress, compressed_log_likelihood
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.ui import plotly_template, render_status_bar, plot_export_controls, export_downloads


def render():
    render_status_bar()
    st.title("MOPED Compression")
    st.markdown("""
    **MOPED** (Multiple Optimized Parameter Estimation Data) compression reduces high-dimensional data
    to a small set of optimal linear combinations that preserve Fisher information about parameters.
    
    ⚠️ **Important caveat**: MOPED here compresses the **RAW Gaussian likelihood** (χ² = δᵀC⁻¹δ), 
    NOT the analytically marginalized SN likelihood (which marginalizes over M_B/H₀). 
    The validation compares against the RAW likelihood only.
    """)

    # Load data
    pantheon_loader(key="compression_data", show_instructions=False)

    if "compression_data" not in st.session_state:
        st.info("Please load data first using the Data Explorer or the section above.")
        return

    data = st.session_state["compression_data"]
    z, mb_obs, cov = data["z"], data["mb_obs"], data["cov"]

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

    theta_fid = np.array([fiducial[p] for p in param_names])

    # Compression
    run_btn = st.button("🚀 Compute MOPED Compression", type="primary")
    if run_btn:
        with st.spinner("Computing MOPED vectors..."):
            cov_inv = np.linalg.inv(cov)
            
            def model_predictions(theta):
                return spec["func"](z, *theta)
            
            B = moped_vectors(model_predictions, theta_fid, cov_inv)
            
            # Compressed data: B @ mb_obs
            y_compressed = compress(B, mb_obs)
            
            st.session_state["moped_B"] = B
            st.session_state["moped_y_compressed"] = y_compressed
            st.session_state["moped_model"] = model
            st.session_state["moped_param_names"] = param_names
            st.session_state["moped_fiducial"] = theta_fid
            st.session_state["moped_z"] = z
            st.session_state["moped_cov"] = cov
            st.session_state["moped_mb_obs"] = mb_obs
            
            st.success(f"✅ Compressed {len(z)} data points → {len(param_names)} MOPED coefficients")
            
            # Show compression vectors
            st.subheader("MOPED Vectors (B matrix)")
            B_df = pd.DataFrame(B, index=[f"y_{i}" for i in range(len(z))], columns=param_names)
            st.dataframe(B_df, use_container_width=True)
            export_downloads(B_df, f"moped_B_matrix_{model}")
            
            # Show compressed data
            st.subheader("Compressed Data")
            comp_df = pd.DataFrame({"Parameter": param_names, "Compressed y": y_compressed})
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            export_downloads(comp_df, f"moped_compressed_data_{model}")

    # Validation
    if "moped_B" in st.session_state:
        B = st.session_state["moped_B"]
        y_compressed = st.session_state["moped_y_compressed"]
        model = st.session_state["moped_model"]
        param_names = st.session_state["moped_param_names"]
        theta_fid = st.session_state["moped_fiducial"]
        z = st.session_state["moped_z"]
        cov = st.session_state["moped_cov"]
        mb_obs = st.session_state["moped_mb_obs"]
        spec = MODEL_REGISTRY[model]

        st.markdown("---")
        st.subheader("Validation: MOPED vs Full Likelihood")
        st.caption("Comparing compressed likelihood (RAW) against full RAW Gaussian likelihood")
        
        validate_btn = st.button("Validate Compression")
        if validate_btn:
            with st.spinner("Validating..."):
                # Generate test points around fiducial
                rng = np.random.default_rng(42)
                n_test = 50
                test_thetas = []
                for _ in range(n_test):
                    theta_test = theta_fid + 0.02 * rng.normal(size=len(theta_fid))
                    # Keep within priors
                    for i, p in enumerate(param_names):
                        lo, hi = spec["priors"][p]
                        theta_test[i] = np.clip(theta_test[i], lo, hi)
                    test_thetas.append(theta_test)
                test_thetas = np.array(test_thetas)
                
                # Compare at test points
                cov_inv = np.linalg.inv(cov)
                diffs = []
                for theta in test_thetas:
                    # Full RAW likelihood
                    mu = spec["func"](z, *theta)
                    delta = mb_obs - mu
                    ll_full = -0.5 * delta @ cov_inv @ delta
                    
                    # Compressed likelihood
                    ll_comp = compressed_log_likelihood(theta, B, y_compressed, model_predictions)
                    
                    diffs.append(abs(ll_comp - ll_full))
                
                diffs = np.array(diffs)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Max |Δ log L|", f"{diffs.max():.4f}")
                col2.metric("Mean |Δ log L|", f"{diffs.mean():.4f}")
                col3.metric("Median |Δ log L|", f"{np.median(diffs):.4f}")
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=diffs, nbinsx=20, name="|Δ log L|", opacity=0.7))
                fig.update_layout(
                    title="Distribution of Log-Likelihood Differences (MOPED vs Full RAW)",
                    xaxis_title="|Δ log L|", yaxis_title="Count",
                    template=plotly_template(), height=300
                )
                plot_export_controls(fig, f"moped_validation_{model}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Export validation data
                val_df = pd.DataFrame({"diff_logL": diffs})
                export_downloads(val_df, f"moped_validation_{model}")
                
                # Overall assessment
                if diffs.max() < 0.1:
                    st.success("✅ Excellent compression — max difference < 0.1")
                elif diffs.max() < 1.0:
                    st.warning("⚠️ Acceptable compression — max difference < 1.0")
                else:
                    st.error("❌ Poor compression — max difference ≥ 1.0")

        # Parameter estimation from compressed data
        st.markdown("---")
        st.subheader("Parameter Estimation from Compressed Data")
        st.markdown("Run MCMC using the compressed likelihood (much faster for high-dimensional data).")
        
        if st.button("Run MCMC on Compressed Data"):
            with st.spinner("Running MCMC on compressed data..."):
                import emcee
                from pramana.core.likelihood import log_prior
                
                def compressed_log_prob(theta):
                    lp = log_prior(theta, param_names, priors)
                    if not np.isfinite(lp):
                        return -np.inf
                    ll = compressed_log_likelihood(theta, B, y_compressed, model_predictions)
                    return lp + ll
                
                ndim = len(param_names)
                rng = np.random.default_rng(42)
                p0_center = np.array([np.mean(priors[p]) for p in param_names])
                p0_spread = np.array([(priors[p][1] - priors[p][0]) * 0.05 for p in param_names])
                p0 = p0_center + p0_spread * rng.normal(size=(32, ndim))
                
                sampler = emcee.EnsembleSampler(32, ndim, compressed_log_prob)
                sampler.run_mcmc(p0, 4000, progress=True)
                
                burn_in = 1200
                flat_chain = sampler.get_chain(discard=burn_in, flat=True)
                
                st.session_state["moped_chain"] = flat_chain
                st.success(f"✅ Compressed MCMC done! {flat_chain.shape[0]} samples")

        # Show results
        if "moped_chain" in st.session_state:
            chain = st.session_state["moped_chain"]
            
            st.subheader("Compressed MCMC Results")
            cols = st.columns(len(param_names))
            for i, p in enumerate(param_names):
                med = np.median(chain[:, i])
                lo, hi = np.percentile(chain[:, i], [16, 84])
                cols[i].metric(p, f"{med:.4f}", f"+{hi-med:.4f}/-{med-lo:.4f}")
            
            # Export compressed chain
            chain_df = pd.DataFrame(chain, columns=param_names)
            export_downloads(chain_df, f"moped_chain_{model}")

            # Compare with full likelihood if available
            if "last_chain" in st.session_state and st.session_state.get("last_model") == model:
                st.subheader("Comparison: Compressed vs Full Likelihood")
                full_chain = st.session_state["last_chain"]
                
                comp_data = []
                for i, p in enumerate(param_names):
                    med_comp = np.median(chain[:, i])
                    med_full = np.median(full_chain[:, i])
                    comp_data.append({
                        "Parameter": p,
                        "Compressed": f"{med_comp:.4f}",
                        "Full": f"{med_full:.4f}",
                        "Diff": f"{abs(med_comp - med_full):.4f}"
                    })
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                export_downloads(comp_df, f"moped_comparison_{model}")