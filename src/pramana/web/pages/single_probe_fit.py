"""PRAMANA Web UI — Single-Probe Fit page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.mcmc import run_fit as run_mcmc
from pramana.core.diagnostics import summarize
from pramana.core.plotting import corner_plot
from pramana.core.nested_sampling import run_nested, equal_weight_posterior
from pramana.core.profile_likelihood import profile_scan, confidence_interval_from_profile
from pramana.core.likelihood import log_likelihood
from pramana.core.hmc_numpyro import build_sn_model, run_nuts, samples_to_flat_chain
from pramana.core.sbi_inference import make_simulator, train_npe, sample_posterior
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.ui import plotly_template


def render():
    st.title("🎯 Single-Probe Fit")

    # Data loading using shared component
    pantheon_loader(key="pantheon_data", show_instructions=True)

    if "pantheon_data" not in st.session_state:
        st.info("Please load data first using the section above.")
        return

    data = st.session_state["pantheon_data"]
    z, mb_obs, cov = data["z"], data["mb_obs"], data["cov"]

    # Model and method selection
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Model", list(MODEL_REGISTRY.keys()),
                             format_func=lambda x: x.upper())
    with col2:
        method = st.selectbox("Method", [
            "MCMC (emcee)", "Nested Sampling (dynesty)",
            "NUTS/HMC (numpyro)", "Profile Likelihood",
            "SBI (Neural Posterior Estimation)"
        ])
    with col3:
        n_walkers = st.number_input("Walkers / Live points", 16, 200, 32)

    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]

    # Prior editor
    with st.expander("🔧 Prior Editor"):
        prior_vals = {}
        for p in param_names:
            lo, hi = priors[p]
            prior_vals[p] = st.slider(f"{p}", float(lo), float(hi), (float(lo), float(hi)))
        if st.button("Update Priors"):
            for p in param_names:
                priors[p] = prior_vals[p]
            st.success("Priors updated for this session")

    # Method-specific parameters
    if method == "MCMC (emcee)":
        n_steps = st.slider("MCMC steps", 1000, 50000, 4000, step=1000)
        run_btn = st.button("🚀 Run MCMC", type="primary")
        if run_btn:
            progress_bar = st.progress(0, text="Running MCMC...")
            with st.spinner("Running MCMC..."):
                sampler = run_mcmc(model, z, mb_obs, cov, nwalkers=n_walkers, nsteps=n_steps)
                # Note: emcee doesn't support easy progress callbacks in this version
                progress_bar.progress(1.0, text="MCMC complete!")
                burn_in = int(n_steps * 0.3)
                flat_chain = sampler.get_chain(discard=burn_in, flat=True)
                st.session_state["last_chain"] = flat_chain
                st.session_state["last_model"] = model
                st.success(f"✅ Done! {flat_chain.shape[0]} samples")

    elif method == "Nested Sampling (dynesty)":
        n_live = st.slider("Live points", 100, 2000, n_walkers)
        run_btn = st.button("🚀 Run Nested Sampling", type="primary")
        if run_btn:
            with st.spinner("Running nested sampling..."):
                cov_inv = np.linalg.inv(cov)
                def loglike(theta):
                    return log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], param_names)
                results = run_nested(loglike, param_names, priors, nlive=n_live)
                posterior = equal_weight_posterior(results)
                st.session_state["last_chain"] = posterior
                st.session_state["last_model"] = model
                st.success(f"✅ Done! ln(Z) = {results.logz[-1]:.2f} ± {results.logzerr[-1]:.2f}")

    elif method == "NUTS/HMC (numpyro)":
        col1, col2 = st.columns(2)
        with col1:
            n_warmup = st.number_input("Warmup steps", 100, 5000, 1000)
            n_samples = st.number_input("Samples per chain", 100, 10000, 2000)
        with col2:
            n_chains = st.number_input("Chains", 1, 8, 2)
        run_btn = st.button("🚀 Run NUTS", type="primary")
        if run_btn:
            with st.spinner("Running NUTS..."):
                cov_inv = np.linalg.inv(cov)
                model_numpyro, pnames = build_sn_model(z, mb_obs, cov_inv, model, priors)
                mcmc = run_nuts(model_numpyro, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
                flat_chain = samples_to_flat_chain(mcmc, pnames)
                st.session_state["last_chain"] = flat_chain
                st.session_state["last_model"] = model
                st.success(f"✅ Done! {flat_chain.shape[0]} samples")

    elif method == "Profile Likelihood":
        param_of_interest = st.selectbox("Parameter to profile", param_names)
        n_points = st.slider("Scan points", 10, 100, 30)
        run_btn = st.button("🚀 Run Profile Likelihood", type="primary")
        if run_btn:
            with st.spinner("Running profile likelihood..."):
                cov_inv = np.linalg.inv(cov)
                def neg_log_likelihood(theta):
                    return -log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], param_names)
                bounds = {p: priors[p] for p in param_names}
                scan_vals = np.linspace(bounds[param_of_interest][0], bounds[param_of_interest][1], n_points)
                scan_vals, profile_nll, _ = profile_scan(neg_log_likelihood, param_names, param_of_interest, scan_vals, bounds)
                best, lo, hi = confidence_interval_from_profile(scan_vals, profile_nll)
                st.session_state["profile_result"] = {
                    "scan_vals": scan_vals, "profile_nll": profile_nll,
                    "param": param_of_interest, "best": best, "lo": lo, "hi": hi
                }
                st.success(f"Best {param_of_interest} = {best:.4f}, 68% CI: [{lo:.4f}, {hi:.4f}]")

    elif method == "SBI (Neural Posterior Estimation)":
        n_sims = st.number_input("Training simulations", 500, 50000, 2000, step=500)
        n_post_samples = st.number_input("Posterior samples", 100, 20000, 5000, step=500)
        run_btn = st.button("🚀 Train & Sample SBI", type="primary")
        if run_btn:
            with st.spinner("Building simulator..."):
                # mb_err = sqrt of diagonal of covariance (per-SN magnitude error)
                mb_err = np.sqrt(np.diag(cov))
                simulator = make_simulator(spec["func"], z, mb_err)
            with st.spinner(f"Training NPE with {n_sims} simulations..."):
                posterior = train_npe(simulator, priors, param_names, n_simulations=n_sims)
            with st.spinner(f"Sampling {n_post_samples} posterior samples..."):
                # Use fiducial for observed data
                theta_fid = np.array([np.mean(priors[p]) for p in param_names])
                x_o = simulator(theta_fid)
                post_samples = sample_posterior(posterior, x_o, n_samples=n_post_samples)
                st.session_state["last_chain"] = post_samples
                st.session_state["last_model"] = model
                st.success(f"✅ Done! {post_samples.shape[0]} posterior samples")

    # Results display
    if "last_chain" in st.session_state:
        chain = st.session_state["last_chain"]
        model_name = st.session_state["last_model"]
        spec = MODEL_REGISTRY[model_name]

        st.markdown("---")
        st.subheader("Results")

        # Parameter table
        cols = st.columns(len(param_names))
        for i, p in enumerate(param_names):
            med = np.median(chain[:, i])
            lo, hi = np.percentile(chain[:, i], [16, 84])
            cols[i].metric(p, f"{med:.4f}", f"+{hi-med:.4f}/-{med-lo:.4f}")

        # Corner plot (static matplotlib)
        if st.button("Generate Corner Plot"):
            with st.spinner("Generating corner plot..."):
                fig = corner_plot(chain, param_names, spec["labels"])
                st.pyplot(fig)

        # 1D marginal plots (plotly)
        st.subheader("1D Marginals")
        for i, p in enumerate(param_names):
            fig = go.Figure()
            hist, bins = np.histogram(chain[:, i], bins=50, density=True)
            fig.add_trace(go.Bar(x=(bins[:-1]+bins[1:])/2, y=hist, name=p, opacity=0.7))
            fig.update_layout(title=f"{p} marginal", height=200, showlegend=False, template=plotly_template())
            st.plotly_chart(fig, use_container_width=True)

    if "profile_result" in st.session_state:
        res = st.session_state["profile_result"]
        st.markdown("---")
        st.subheader(f"Profile Likelihood: {res['param']}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["scan_vals"], y=res["profile_nll"], mode="lines", name="Profile"))
        fig.add_vline(x=res["best"], line_dash="dash", line_color="red", annotation_text=f"Best: {res['best']:.4f}")
        fig.add_vline(x=res["lo"], line_dash="dot", line_color="orange", annotation_text="68% CI")
        fig.add_vline(x=res["hi"], line_dash="dot", line_color="orange")
        fig.update_layout(xaxis_title=res["param"], yaxis_title="-log L", template=plotly_template())
        st.plotly_chart(fig, use_container_width=True)