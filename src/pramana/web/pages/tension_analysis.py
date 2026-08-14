"""PRAMANA Web UI — Tension Analysis page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pramana.core.jwst_probes import (
    H0_MEASUREMENTS, S8_MEASUREMENTS,
    h0_tension_sigma, s8_tension_sigma,
    plot_h0_whisker, plot_s8_whisker,
    append_supernovae
)


def render():
    st.title("⚡ Tension Analysis")

    tab1, tab2, tab3 = st.tabs(["H₀ Tension", "S₈ Tension", "Append JWST SNe"])

    with tab1:
        st.subheader("H₀ Tension Measurements")

        # Table
        h0_data = []
        for name, d in H0_MEASUREMENTS.items():
            h0_data.append({"Measurement": name, "H₀": d["H0"], "Error": d["err"], "Family": d["family"]})
        st.dataframe(h0_data, use_container_width=True, hide_index=True)

        # Tension calculator
        col1, col2 = st.columns(2)
        with col1:
            h0_a = st.selectbox("Measurement A", list(H0_MEASUREMENTS.keys()), key="h0a")
        with col2:
            h0_b = st.selectbox("Measurement B", list(H0_MEASUREMENTS.keys()), index=2, key="h0b")

        if st.button("Compute Tension", key="h0t"):
            sigma = h0_tension_sigma(h0_a, h0_b)
            st.metric("Tension", f"{sigma:.2f} σ")

        # Whisker plot
        if st.button("Generate Whisker Plot", key="h0w"):
            fig = go.Figure()
            for i, (name, d) in enumerate(H0_MEASUREMENTS.items()):
                color = "#1f5fa8" if "early" in d["family"] else "#a83232"
                fig.add_trace(go.Scatter(
                    x=[d["H0"]], y=[i], mode='markers',
                    error_x=dict(type='data', array=[d["err"]], color=color),
                    marker=dict(size=12, color=color), name=name, showlegend=False
                ))
            fig.update_layout(
                title="H₀ Measurements",
                yaxis=dict(tickmode='array', tickvals=list(range(len(H0_MEASUREMENTS))),
                           ticktext=list(H0_MEASUREMENTS.keys())),
                xaxis_title="H₀ [km/s/Mpc]", height=400, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("S₈ Tension Measurements")

        s8_data = []
        for name, d in S8_MEASUREMENTS.items():
            s8_data.append({"Measurement": name, "S₈": d["S8"], "Error": d["err"], "Family": d["family"]})
        st.dataframe(s8_data, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            s8_a = st.selectbox("Measurement A", list(S8_MEASUREMENTS.keys()), key="s8a")
        with col2:
            s8_b = st.selectbox("Measurement B", list(S8_MEASUREMENTS.keys()), index=1, key="s8b")

        if st.button("Compute Tension", key="s8t"):
            sigma = s8_tension_sigma(s8_a, s8_b)
            st.metric("Tension", f"{sigma:.2f} σ")

        if st.button("Generate Whisker Plot", key="s8w"):
            fig = go.Figure()
            for i, (name, d) in enumerate(S8_MEASUREMENTS.items()):
                color = "#1f5fa8" if "early" in d["family"] else "#2f9e44"
                fig.add_trace(go.Scatter(
                    x=[d["S8"]], y=[i], mode='markers',
                    error_x=dict(type='data', array=[d["err"]], color=color),
                    marker=dict(size=12, color=color), name=name, showlegend=False
                ))
            fig.update_layout(
                title="S₈ Measurements",
                yaxis=dict(tickmode='array', tickvals=list(range(len(S8_MEASUREMENTS))),
                           ticktext=list(S8_MEASUREMENTS.keys())),
                xaxis_title="S₈ = σ₈√(Ωₘ/0.3)", height=400, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Append High-z SNe (JWST Discoveries)")
        st.markdown("Add new high-redshift SNe to the Pantheon+ Hubble diagram.")

        # Load base data
        if 'pantheon_data' in st.session_state:
            data = st.session_state['pantheon_data']
            z_base, mb_base, cov_base = data['z'], data['mb_obs'], data['cov']
            st.info(f"Using loaded Pantheon+ data: {len(z_base)} SNe")
        else:
            st.warning("No base data loaded. Go to Data Explorer or Single-Probe Fit to load data first.")
            if st.button("Use Synthetic Base Data"):
                from pramana.core.data_io import make_synthetic_dataset
                z_base, mb_base, cov_base = make_synthetic_dataset()
                st.session_state['pantheon_data'] = {'z': z_base, 'mb_obs': mb_base, 'cov': cov_base}
                st.rerun()

        if 'pantheon_data' in st.session_state:
            st.markdown("**New SNe (comma-separated values):**")
            col1, col2, col3 = st.columns(3)
            with col1:
                z_new_str = st.text_area("Redshifts (z)", "1.5, 1.8, 2.1", height=80)
            with col2:
                mb_new_str = st.text_area("Magnitudes (m_b)", "26.5, 27.2, 27.8", height=80)
            with col3:
                mb_err_str = st.text_area("Errors (σ_m)", "0.15, 0.18, 0.20", height=80)

            if st.button("Append SNe", type="primary"):
                try:
                    z_new = np.array([float(x.strip()) for x in z_new_str.split(",")])
                    mb_new = np.array([float(x.strip()) for x in mb_new_str.split(",")])
                    mb_err_new = np.array([float(x.strip()) for x in mb_err_str.split(",")])

                    z_out, mb_out, cov_out = append_supernovae(z_base, mb_base, cov_base, z_new, mb_new, mb_err_new)

                    st.session_state['pantheon_data'] = {'z': z_out, 'mb_obs': mb_out, 'cov': cov_out}
                    st.success(f"Appended {len(z_new)} SNe. Total: {len(z_out)} SNe")

                    # Show updated Hubble diagram
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=z_base, y=mb_base, mode='markers',
                                             marker=dict(size=4, color='gray'), name='Original'))
                    fig.add_trace(go.Scatter(x=z_new, y=mb_new, mode='markers',
                                             marker=dict(size=8, color='red'), name='New (JWST)',
                                             error_y=dict(type='data', array=mb_err_new, visible=True)))
                    fig.update_layout(xaxis_title="z", yaxis_title="m_b,corr", template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")