"""PRAMANA Web UI — Importance Reweighting page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pramana.core.models import MODEL_REGISTRY
from pramana.core.importance_resampling import (
    importance_weights, effective_sample_size, reweight_chain,
    weighted_quantiles, resample_to_equal_weight
)
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.ui import plotly_template, render_status_bar


def render():
    render_status_bar()
    st.title("Importance Reweighting")
    st.markdown("""
    Reweight an existing MCMC chain to a new likelihood or prior without re-running the sampler.
    Useful for: testing different priors, adding new data, or "what-if" scenarios.
    """)

    # Load original chain
    st.subheader("1. Original Chain")
    chain_file = st.file_uploader("MCMC chain (.npz from CLI or saved results)", type=["npz"], key="reweight_chain_upload")
    
    if chain_file is not None:
        try:
            chain_data = np.load(chain_file)
            original_chain = chain_data["chain"]
            st.success(f"✅ Loaded chain: {original_chain.shape[0]} samples, {original_chain.shape[1]} params")
            
            # Show parameter names if available
            if "params" in chain_data:
                param_names = list(chain_data["params"])
                st.info(f"Parameters: {', '.join(param_names)}")
            else:
                param_names = [f"p{i}" for i in range(original_chain.shape[1])]
                st.warning("No parameter names in file. Using generic names.")
        except Exception as e:
            st.error(f"Error loading chain: {e}")
            return
    else:
        # Option to use last chain from session
        if "last_chain" in st.session_state:
            if st.button("Use last chain from Single-Probe Fit"):
                original_chain = st.session_state["last_chain"]
                param_names = st.session_state["last_model"]
                spec = MODEL_REGISTRY[st.session_state["last_model"]]
                param_names = spec["params"]
                st.success(f"Using chain from {st.session_state['last_model'].upper()} fit")
        else:
            st.info("Upload a chain file or run a fit in Single-Probe Fit first.")
            return

    if "original_chain" not in locals():
        return

    # Model selection (for likelihood evaluation)
    model = st.selectbox("Model", list(MODEL_REGISTRY.keys()), format_func=lambda x: x.upper())
    spec = MODEL_REGISTRY[model]
    priors = spec["priors"]

    # Load new data (optional)
    st.subheader("2. New Data (Optional)")
    pantheon_loader(key="reweight_new_data", show_instructions=False)
    
    new_data = None
    if "reweight_new_data" in st.session_state:
        new_data = st.session_state["reweight_new_data"]
        st.info(f"Using new data: {len(new_data['z'])} SNe")
    else:
        st.caption("No new data loaded — will only reweight priors")

    # New priors
    st.subheader("3. New Priors")
    new_priors = {}
    for p in param_names:
        lo, hi = priors[p]
        new_priors[p] = st.slider(f"{p} prior range", float(lo), float(hi), (float(lo), float(hi)))

    # Run reweighting
    run_btn = st.button("🚀 Reweight Chain", type="primary")
    if run_btn:
        with st.spinner("Computing importance weights..."):
            # Compute log-likelihood for each sample under old and new
            if new_data is not None:
                new_z, new_mb, new_cov = new_data["z"], new_data["mb_obs"], new_data["cov"]
                new_cov_inv = np.linalg.inv(new_cov)
            else:
                new_z, new_mb, new_cov_inv = None, None, None

            # Evaluate old and new log-probabilities
            from pramana.core.likelihood import log_likelihood, log_prior
            
            old_log_probs = []
            new_log_probs = []
            
            for theta in original_chain:
                # Old log-prob: prior + likelihood (if original data available)
                # NOTE: We don't have the original data that produced this chain.
                # The weights will be computed using ONLY the prior for the "old" log-prob,
                # which means the reweighting is effectively: prior_new/prior_old * likelihood_new
                # This is valid ONLY if the original chain was generated with a flat/uninformative prior.
                # If the original chain used informative priors, results will be biased.
                # For rigorous reweighting, the user must provide the original data.
                lp_old = log_prior(theta, param_names, priors)
                if np.isfinite(lp_old):
                    # Original likelihood unavailable — using prior only
                    pass
                
                # New log-prob: prior_new + likelihood_new (if new data provided)
                lp_new = log_prior(theta, param_names, new_priors)
                if np.isfinite(lp_new) and new_data is not None:
                    lp_new += log_likelihood(theta, new_z, new_mb, new_cov_inv, spec["func"], param_names)
                
                old_log_probs.append(lp_old)
                new_log_probs.append(lp_new)

            old_log_probs = np.array(old_log_probs)
            new_log_probs = np.array(new_log_probs)

            # Compute weights — correct signature: (chain, log_prob_old, log_prob_new)
            weights = importance_weights(original_chain, old_log_probs, new_log_probs)
            n_eff = effective_sample_size(weights)
            frac = n_eff / len(original_chain)

            st.session_state["reweight_weights"] = weights
            st.session_state["reweight_chain"] = original_chain
            st.session_state["reweight_param_names"] = param_names
            st.session_state["reweight_n_eff"] = n_eff
            st.session_state["reweight_frac"] = frac

            st.success(f"✅ Reweighting done! ESS = {n_eff:.1f} / {len(original_chain)} ({frac:.1%})")
            
            if frac < 0.05:
                st.warning("⚠️ Low effective sample size (<5%). Results may be unreliable.")
            
            st.info("""
            **Caveat:** The "old" log-probability used here is the PRIOR ONLY (no likelihood),
            because the original data that generated this chain is not available.
            This is mathematically correct ONLY if the original chain was generated with
            a flat/uninformative prior. If the original chain used informative priors,
            the reweighted posterior will be biased. For rigorous reweighting, provide
            the original data alongside the chain.
            """)

    # Results
    if "reweight_weights" in st.session_state:
        weights = st.session_state["reweight_weights"]
        chain = st.session_state["reweight_chain"]
        param_names = st.session_state["reweight_param_names"]
        n_eff = st.session_state["reweight_n_eff"]
        frac = st.session_state["reweight_frac"]

        st.markdown("---")
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Original samples", len(chain))
        col2.metric("Effective sample size (ESS)", f"{n_eff:.1f}")
        col3.metric("ESS fraction", f"{frac:.1%}")

        # Resample to equal weight
        if st.button("Resample to Equal Weight"):
            resampled = resample_to_equal_weight(chain, weights)
            st.session_state["reweight_resampled"] = resampled
            st.success(f"✅ Resampled to {len(resampled)} equal-weight samples")

        # Show parameter constraints
        if "reweight_resampled" in st.session_state:
            chain_display = st.session_state["reweight_resampled"]
        else:
            chain_display = chain  # show weighted percentiles

        st.subheader("Parameter Constraints (Weighted)")
        for i, p in enumerate(param_names):
            if "reweight_resampled" in st.session_state:
                med = np.median(chain_display[:, i])
                lo, hi = np.percentile(chain_display[:, i], [16, 84])
            else:
                med = weighted_quantiles(chain_display[:, i], weights, (0.16, 0.5, 0.84))[1]
                lo = weighted_quantiles(chain_display[:, i], weights, (0.16, 0.5, 0.84))[0]
                hi = weighted_quantiles(chain_display[:, i], weights, (0.16, 0.5, 0.84))[2]
            st.metric(p, f"{med:.4f}", f"+{hi-med:.4f}/-{med-lo:.4f}")

        # Corner plot
        if st.button("Generate Corner Plot"):
            with st.spinner("Generating corner plot..."):
                from pramana.core.plotting import corner_plot
                spec = MODEL_REGISTRY[model]
                if "reweight_resampled" in st.session_state:
                    fig = corner_plot(chain_display, param_names, spec["labels"])
                else:
                    st.info("Corner plot requires resampled equal-weight chain. Click 'Resample to Equal Weight' first.")
                if "reweight_resampled" in st.session_state:
                    st.pyplot(fig)

        # 1D marginals
        st.subheader("1D Marginals")
        for i, p in enumerate(param_names):
            fig = go.Figure()
            if "reweight_resampled" in st.session_state:
                hist, bins = np.histogram(chain_display[:, i], bins=50, density=True)
            else:
                hist, bins = np.histogram(chain_display[:, i], bins=50, density=True, weights=weights)
            fig.add_trace(go.Bar(x=(bins[:-1]+bins[1:])/2, y=hist, name=p, opacity=0.7))
            fig.update_layout(title=f"{p} marginal (weighted)", height=200, showlegend=False, template=plotly_template())
            st.plotly_chart(fig, use_container_width=True)