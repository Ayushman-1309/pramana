"""PRAMANA Web UI — Data Explorer page."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.bao_desi import DESI_DR2_BAO_TABLE, build_data_vector_and_cov
from pramana.core.jwst_probes import H0_MEASUREMENTS, S8_MEASUREMENTS
from pramana.utils.validators import validate_pantheon_data, validate_pantheon_cov


def render():
    st.title("📊 Data Explorer")

    tab1, tab2, tab3, tab4 = st.tabs(["Pantheon+ SN", "DESI DR2 BAO", "H₀ Measurements", "S₈ Measurements"])

    with tab1:
        st.subheader("Pantheon+SH0ES Supernova Data")
        col1, col2 = st.columns(2)
        with col1:
            data_file = st.text_input("Data file (.dat)", "data/pantheon/Pantheon+SH0ES.dat")
        with col2:
            cov_file = st.text_input("Covariance file (.cov)", "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov")

        use_synthetic = st.checkbox("Use synthetic data for testing", value=False)

        if st.button("Load Data", type="primary"):
            with st.spinner("Loading..."):
                try:
                    if use_synthetic:
                        z, mb_obs, cov = make_synthetic_dataset()
                        df = None
                        st.success(f"Generated synthetic data: {len(z)} SNe")
                    else:
                        z, mb_obs, cov, df = load_pantheon(data_file, cov_file)
                        st.success(f"Loaded {len(z)} SNe from {data_file}")

                        # Validation
                        val = validate_pantheon_data(data_file)
                        val_cov = validate_pantheon_cov(cov_file, len(z))

                        col1, col2, col3 = st.columns(3)
                        col1.metric("N SNe", len(z))
                        col2.metric("z range", f"{z.min():.3f} – {z.max():.3f}")
                        col3.metric("Cov condition", f"{val_cov['condition_number']:.2e}")

                    # Store in session state
                    st.session_state['pantheon_data'] = {
                        'z': z, 'mb_obs': mb_obs, 'cov': cov, 'df': df
                    }

                except Exception as e:
                    st.error(f"Error loading data: {e}")

        if 'pantheon_data' in st.session_state:
            data = st.session_state['pantheon_data']
            z, mb_obs = data['z'], data['mb_obs']

            # Hubble diagram plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z, y=mb_obs, mode='markers',
                                     marker=dict(size=4, opacity=0.5, color='gray'),
                                     name='SN data'))
            fig.update_layout(title="Pantheon+ Hubble Diagram",
                              xaxis_title="Redshift z", yaxis_title="m_b,corr",
                              height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean m_b", f"{mb_obs.mean():.3f}")
            col2.metric("Std m_b", f"{mb_obs.std():.3f}")
            col3.metric("Min z", f"{z.min():.4f}")
            col4.metric("Max z", f"{z.max():.4f}")

            if data['df'] is not None:
                with st.expander("View DataFrame (first 20 rows)"):
                    st.dataframe(data['df'].head(20))

    with tab2:
        st.subheader("DESI DR2 BAO Measurements (Built-in)")

        labels, z_arr, data_vec, cov = build_data_vector_and_cov()

        # Table
        table_data = []
        for tracer, d in DESI_DR2_BAO_TABLE.items():
            if "DV_rd" in d:
                table_data.append({
                    "Tracer": tracer, "z": d["z"],
                    "Observable": "DV/rd", "Value": d["DV_rd"], "Error": d["DV_rd_err"]
                })
            else:
                table_data.append({
                    "Tracer": tracer, "z": d["z"],
                    "Observable": "DM/rd", "Value": d["DM_rd"], "Error": d["DM_rd_err"]
                })
                table_data.append({
                    "Tracer": "", "z": "",
                    "Observable": "DH/rd", "Value": d["DH_rd"], "Error": d["DH_rd_err"]
                })
                table_data.append({
                    "Tracer": "", "z": "",
                    "Observable": "ρ(DM,DH)", "Value": d["rho_MH"], "Error": ""
                })

        df_bao = pd.DataFrame(table_data)
        st.dataframe(df_bao, use_container_width=True, hide_index=True)

        # Covariance heatmap
        fig = go.Figure(data=go.Heatmap(z=cov, colorscale='RdBu', zmid=0))
        fig.update_layout(title="DESI DR2 BAO Covariance Matrix",
                          height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Data points", len(data_vec))
        col2.metric("Cov condition", f"{np.linalg.cond(cov):.2e}")

    with tab3:
        st.subheader("H₀ Tension Measurements")

        h0_data = []
        for name, d in H0_MEASUREMENTS.items():
            h0_data.append({
                "Measurement": name, "H₀": d["H0"], "Error": d["err"], "Family": d["family"]
            })
        df_h0 = pd.DataFrame(h0_data)
        st.dataframe(df_h0, use_container_width=True, hide_index=True)

        # Whisker plot
        fig = go.Figure()
        for i, (name, d) in enumerate(H0_MEASUREMENTS.items()):
            color = "#1f5fa8" if "early" in d["family"] else "#a83232"
            fig.add_trace(go.Scatter(
                x=[d["H0"]], y=[i], mode='markers',
                error_x=dict(type='data', array=[d["err"]], color=color),
                marker=dict(size=10, color=color), name=name, showlegend=False
            ))
        fig.update_layout(
            title="H₀ Measurements (Whisker Plot)",
            yaxis=dict(tickmode='array', tickvals=list(range(len(H0_MEASUREMENTS))),
                       ticktext=list(H0_MEASUREMENTS.keys())),
            xaxis_title="H₀ [km/s/Mpc]", height=400, template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("S₈ Tension Measurements")

        s8_data = []
        for name, d in S8_MEASUREMENTS.items():
            s8_data.append({
                "Measurement": name, "S₈": d["S8"], "Error": d["err"], "Family": d["family"]
            })
        df_s8 = pd.DataFrame(s8_data)
        st.dataframe(df_s8, use_container_width=True, hide_index=True)

        # Whisker plot
        fig = go.Figure()
        for i, (name, d) in enumerate(S8_MEASUREMENTS.items()):
            color = "#1f5fa8" if "early" in d["family"] else "#2f9e44"
            fig.add_trace(go.Scatter(
                x=[d["S8"]], y=[i], mode='markers',
                error_x=dict(type='data', array=[d["err"]], color=color),
                marker=dict(size=10, color=color), name=name, showlegend=False
            ))
        fig.update_layout(
            title="S₈ Measurements (Whisker Plot)",
            yaxis=dict(tickmode='array', tickvals=list(range(len(S8_MEASUREMENTS))),
                       ticktext=list(S8_MEASUREMENTS.keys())),
            xaxis_title="S₈ = σ₈√(Ωₘ/0.3)", height=400, template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)