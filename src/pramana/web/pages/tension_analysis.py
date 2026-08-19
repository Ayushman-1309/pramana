"""PRAMANA Web UI — Tension Analysis page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pramana.core.jwst_probes import (
    H0_MEASUREMENTS, S8_MEASUREMENTS,
    h0_tension_sigma, s8_tension_sigma,
    plot_h0_whisker, plot_s8_whisker,
    append_supernovae
)
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.data_hub import dataset_loader
from pramana.web.components.ui import plotly_template, render_status_bar, plot_export_controls, export_downloads


def _hub_h0_table():
    """H0 table from the Data Hub if loaded, else a one-off loader."""
    if "jwst_data" in st.session_state and st.session_state["jwst_data"].get("h0_table"):
        return st.session_state["jwst_data"]["h0_table"]
    return H0_MEASUREMENTS


def _hub_s8_table():
    if "jwst_data" in st.session_state and st.session_state["jwst_data"].get("s8_table"):
        return st.session_state["jwst_data"]["s8_table"]
    return S8_MEASUREMENTS


def render():
    render_status_bar()
    st.title("Tension Analysis")

    # Allow loading JWST-era data (manual download or synthetic) right here
    dataset_loader("jwst", key="jwst_data", show_instructions=False)

    h0_table = _hub_h0_table()
    s8_table = _hub_s8_table()

    tab1, tab2, tab3 = st.tabs(["H₀ Tension", "S₈ Tension", "Append JWST SNe"])

    with tab1:
        st.subheader("H₀ Tension Measurements")

        # Table
        h0_data = []
        for name, d in h0_table.items():
            h0_data.append({"Measurement": name, "H₀": d["H0"], "Error": d["err"], "Family": d["family"]})
        h0_df = pd.DataFrame(h0_data)
        st.dataframe(h0_df, use_container_width=True, hide_index=True)
        export_downloads(h0_df, "tension_H0_measurements")

        # Tension calculator
        col1, col2 = st.columns(2)
        with col1:
            h0_a = st.selectbox("Measurement A", list(h0_table.keys()), key="h0a")
        with col2:
            h0_b = st.selectbox("Measurement B", list(h0_table.keys()), index=min(2, len(h0_table) - 1), key="h0b")

        if st.button("Compute Tension", key="h0t"):
            sigma = h0_tension_sigma(h0_a, h0_b, table=h0_table)
            st.metric("Tension", f"{sigma:.2f} σ")

        # Whisker plot
        if st.button("Generate Whisker Plot", key="h0w"):
            fig = go.Figure()
            for i, (name, d) in enumerate(h0_table.items()):
                color = "#1f5fa8" if "early" in d["family"] else "#a83232"
                fig.add_trace(go.Scatter(
                    x=[d["H0"]], y=[i], mode="markers",
                    error_x=dict(type="data", array=[d["err"]], color=color),
                    marker=dict(size=12, color=color), name=name, showlegend=False
                ))
            fig.update_layout(
                title="H₀ Measurements",
                yaxis=dict(tickmode="array", tickvals=list(range(len(h0_table))),
                           ticktext=list(h0_table.keys())),
                xaxis_title="H₀ [km/s/Mpc]", height=400, template=plotly_template()
            )
            plot_export_controls(fig, "tension_H0_whisker")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("S₈ Tension Measurements")

        s8_data = []
        for name, d in s8_table.items():
            s8_data.append({"Measurement": name, "S₈": d["S8"], "Error": d["err"], "Family": d["family"]})
        s8_df = pd.DataFrame(s8_data)
        st.dataframe(s8_df, use_container_width=True, hide_index=True)
        export_downloads(s8_df, "tension_S8_measurements")

        col1, col2 = st.columns(2)
        with col1:
            s8_a = st.selectbox("Measurement A", list(s8_table.keys()), key="s8a")
        with col2:
            s8_b = st.selectbox("Measurement B", list(s8_table.keys()), index=min(1, len(s8_table) - 1), key="s8b")

        if st.button("Compute Tension", key="s8t"):
            sigma = s8_tension_sigma(s8_a, s8_b, table=s8_table)
            st.metric("Tension", f"{sigma:.2f} σ")

        if st.button("Generate Whisker Plot", key="s8w"):
            fig = go.Figure()
            for i, (name, d) in enumerate(s8_table.items()):
                color = "#1f5fa8" if "early" in d["family"] else "#2f9e44"
                fig.add_trace(go.Scatter(
                    x=[d["S8"]], y=[i], mode="markers",
                    error_x=dict(type="data", array=[d["err"]], color=color),
                    marker=dict(size=12, color=color), name=name, showlegend=False
                ))
            fig.update_layout(
                title="S₈ Measurements",
                yaxis=dict(tickmode="array", tickvals=list(range(len(s8_table))),
                           ticktext=list(s8_table.keys())),
                xaxis_title="S₈ = σ₈√(Ωₘ/0.3)", height=400, template=plotly_template()
            )
            plot_export_controls(fig, "tension_S8_whisker")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Append High-z SNe (JWST Discoveries)")
        st.markdown("Add new high-redshift SNe to the Pantheon+ Hubble diagram.")

        # Load base data using shared loader (but only show if not already loaded)
        if "pantheon_data" not in st.session_state:
            st.info("No base data loaded yet. Load data using the section below or go to Data Explorer.")
            pantheon_loader(key="pantheon_data", show_instructions=True)
        else:
            data = st.session_state["pantheon_data"]
            z_base, mb_base, cov_base = data["z"], data["mb_obs"], data["cov"]
            st.info(f"Using loaded Pantheon+ data: {len(z_base)} SNe")
            
            if st.button("Use Synthetic Base Data Instead"):
                from pramana.core.data_io import make_synthetic_dataset
                z_base, mb_base, cov_base = make_synthetic_dataset()
                st.session_state["pantheon_data"] = {"z": z_base, "mb_obs": mb_base, "cov": cov_base, "df": None}
                st.rerun()

        if "pantheon_data" in st.session_state:
            data = st.session_state["pantheon_data"]
            z_base, mb_base, cov_base = data["z"], data["mb_obs"], data["cov"]
            
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

                    st.session_state["pantheon_data"] = {"z": z_out, "mb_obs": mb_out, "cov": cov_out, "df": None}
                    st.success(f"✅ Appended {len(z_new)} SNe. Total: {len(z_out)} SNe")

                    # Show updated Hubble diagram
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=z_base, y=mb_base, mode="markers",
                                             marker=dict(size=4, color="gray"), name="Original"))
                    fig.add_trace(go.Scatter(x=z_new, y=mb_new, mode="markers",
                                             marker=dict(size=8, color="#ff7f0e"), name="New (JWST)",
                                             error_y=dict(type="data", array=mb_err_new, visible=True)))
                    fig.update_layout(xaxis_title="z", yaxis_title="m_b,corr", template=plotly_template(), height=400)
                    plot_export_controls(fig, "tension_appended_hubble_diagram")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export appended SNe data
                    appended_df = pd.DataFrame({"z": z_new, "mb": mb_new, "err": mb_err_new})
                    export_downloads(appended_df, "tension_appended_SNe")

                except Exception as e:
                    st.error(f"Error: {e}")