"""PRAMANA Web UI — Model Comparison page."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pramana.core.models import MODEL_REGISTRY
from pramana.core.mcmc import run_fit as run_mcmc
from pramana.core.nested_sampling import run_nested, equal_weight_posterior, bayes_factor
from pramana.core.likelihood import log_likelihood
from pramana.core.plotting import getdist_triangle
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.ui import plotly_template, plot_export_controls, export_downloads


def render():
    st.title("⚖️ Model Comparison")

    # Data loading using shared component
    pantheon_loader(key="pantheon_data", show_instructions=False)

    if "pantheon_data" not in st.session_state:
        st.info("Please load data first using the Data Explorer or the section above.")
        return

    data = st.session_state["pantheon_data"]
    z, mb_obs, cov = data["z"], data["mb_obs"], data["cov"]
    cov_inv = np.linalg.inv(cov)

    # Model selection
    models_to_compare = st.multiselect(
        "Models to compare",
        list(MODEL_REGISTRY.keys()),
        default=["lcdm", "wcdm", "cpl"],
        format_func=lambda x: x.upper()
    )

    method = st.radio("Method", ["MCMC (emcee)", "Nested Sampling (dynesty)"])

    run_btn = st.button("🚀 Run Comparison", type="primary")
    if run_btn:
        results = {}
        chains = {}
        param_names_dict = {}

        with st.spinner(f"Running {method} for {len(models_to_compare)} models..."):
            for model in models_to_compare:
                spec = MODEL_REGISTRY[model]
                pnames = spec["params"]
                priors = spec["priors"]

                if method == "MCMC (emcee)":
                    sampler = run_mcmc(model, z, mb_obs, cov, nwalkers=32, nsteps=4000)
                    burn_in = 1200
                    flat_chain = sampler.get_chain(discard=burn_in, flat=True)
                else:
                    def loglike(theta):
                        return log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], pnames)
                    res = run_nested(loglike, pnames, priors, nlive=500)
                    flat_chain = equal_weight_posterior(res)
                    results[model] = res

                chains[model] = flat_chain
                param_names_dict[model] = pnames

        st.session_state["comparison_chains"] = chains
        st.session_state["comparison_param_names"] = param_names_dict
        st.session_state["comparison_results"] = results
        st.session_state["comparison_method"] = method
        st.success("✅ Comparison complete!")

    # Results
    if "comparison_chains" in st.session_state:
        chains = st.session_state["comparison_chains"]
        param_names_dict = st.session_state["comparison_param_names"]
        results = st.session_state.get("comparison_results", {})
        method = st.session_state["comparison_method"]

        st.markdown("---")
        st.subheader("Bayes Factors (Nested Sampling)")
        if results:
            model_names = list(results.keys())
            bf_data = []
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i+1:]:
                    lnK, err = bayes_factor(results[m1], results[m2], m1.upper(), m2.upper())
                    bf_data.append({"Model A": m1.upper(), "Model B": m2.upper(), "ln(K)": f"{lnK:.2f} ± {err:.2f}"})
            if bf_data:
                bf_df = pd.DataFrame(bf_data)
                st.dataframe(bf_df, use_container_width=True, hide_index=True)
                export_downloads(bf_df, "model_comparison_bayes_factors")

        st.subheader("Posterior Comparison")
        # Triangle plot overlay
        if st.button("Generate Triangle Plot"):
            with st.spinner("Generating triangle plot..."):
                fig = getdist_triangle(chains, param_names_dict)
                st.pyplot(fig)

        # Parameter summary table
        st.subheader("Parameter Constraints")
        all_params = set()
        for pnames in param_names_dict.values():
            all_params.update(pnames)

        for p in sorted(all_params):
            cols = st.columns(len(chains) + 1)
            cols[0].write(f"**{p}**")
            for i, (model, chain) in enumerate(chains.items()):
                pnames = param_names_dict[model]
                if p in pnames:
                    idx = pnames.index(p)
                    med = np.median(chain[:, idx])
                    lo, hi = np.percentile(chain[:, idx], [16, 84])
                    cols[i+1].metric(model.upper(), f"{med:.4f}", f"+{hi-med:.4f}/-{med-lo:.4f}")
                else:
                    cols[i+1].write("—")

        # Export parameter constraints table
        constraint_rows = []
        for p in sorted(all_params):
            row = {"Parameter": p}
            for model, chain in chains.items():
                pnames = param_names_dict[model]
                if p in pnames:
                    idx = pnames.index(p)
                    med = np.median(chain[:, idx])
                    lo, hi = np.percentile(chain[:, idx], [16, 84])
                    row[model.upper()] = f"{med:.4f} (+{hi-med:.4f}/-{med-lo:.4f})"
                else:
                    row[model.upper()] = "—"
            constraint_rows.append(row)
        if constraint_rows:
            export_downloads(pd.DataFrame(constraint_rows), "model_comparison_constraints")

        # 1D marginals overlay
        st.subheader("1D Marginal Overlays")
        for p in sorted(all_params):
            fig = go.Figure()
            for model, chain in chains.items():
                pnames = param_names_dict[model]
                if p in pnames:
                    idx = pnames.index(p)
                    hist, bins = np.histogram(chain[:, idx], bins=50, density=True)
                    fig.add_trace(go.Bar(x=(bins[:-1]+bins[1:])/2, y=hist, name=model.upper(), opacity=0.5))
            fig.update_layout(title=f"{p} marginal comparison", barmode="overlay",
                              height=300, template=plotly_template())
            plot_export_controls(fig, f"model_comparison_marginal_{p}")
            st.plotly_chart(fig, use_container_width=True)