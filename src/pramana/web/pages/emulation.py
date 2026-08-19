"""PRAMANA Web UI — Emulation page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.gp_emulator import latin_hypercube_design, train_emulator, emulate, validate_emulator
from pramana.web.components.data_loader import pantheon_loader
from pramana.web.components.ui import plotly_template, plot_export_controls


def render():
    st.title("🤖 GP Emulation")

    # Data loading using shared component
    pantheon_loader(key="pantheon_data", show_instructions=False)

    if "pantheon_data" not in st.session_state:
        st.info("Please load data first using the Data Explorer or the section above.")
        return

    data = st.session_state["pantheon_data"]
    z = data["z"]

    # Model selection
    model = st.selectbox("Model", list(MODEL_REGISTRY.keys()), format_func=lambda x: x.upper())
    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]

    bounds = [priors[p] for p in param_names]

    # Training settings
    col1, col2 = st.columns(2)
    with col1:
        n_train = st.number_input("Training points", 50, 2000, 200)
    with col2:
        n_test = st.number_input("Test points", 20, 500, 50)

    train_btn = st.button("🚀 Train Emulator", type="primary")
    if train_btn:
        with st.spinner("Generating training design..."):
            theta_train = latin_hypercube_design(bounds, n_train)

        with st.spinner("Evaluating model at training points..."):
            y_train = np.array([spec["func"](z, *theta) for theta in theta_train])

        with st.spinner("Training GP..."):
            emulator = train_emulator(theta_train, y_train)

        with st.spinner("Validating on test points..."):
            theta_test = latin_hypercube_design(bounds, n_test, seed=123)
            y_test = np.array([spec["func"](z, *theta) for theta in theta_test])
            rel_err = validate_emulator(emulator, theta_test, y_test)
            
            # Proper calibration: compute predicted std on test points and compare to residuals
            y_pred, y_std = emulate(emulator, theta_test, return_std=True)
            residual = y_test - y_pred
            calibration = np.mean(y_std / (np.abs(residual) + 1e-10))

        st.session_state["emulator"] = emulator
        st.session_state["emulator_param_names"] = param_names
        st.session_state["emulator_z"] = z
        st.session_state["emulator_model"] = model
        st.session_state["emulator_bounds"] = bounds
        
        # Show validation results in UI
        st.success("✅ Emulator trained and validated!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Max rel. error", f"{rel_err.max():.2%}")
        col2.metric("Mean rel. error", f"{rel_err.mean():.2%}")
        col3.metric("Calibration (pred σ / |residual|)", f"{calibration:.2f}")

    # Prediction
    if "emulator" in st.session_state:
        emulator = st.session_state["emulator"]
        param_names = st.session_state["emulator_param_names"]
        z = st.session_state["emulator_z"]
        model = st.session_state["emulator_model"]
        bounds = st.session_state["emulator_bounds"]

        st.markdown("---")
        st.subheader("Evaluate Emulator")

        theta_input = {}
        for p in param_names:
            lo, hi = bounds[param_names.index(p)]
            theta_input[p] = st.slider(f"{p}", float(lo), float(hi), float((lo+hi)/2), key=f"emu_{p}")

        if st.button("Predict"):
            theta_arr = np.array([theta_input[p] for p in param_names])
            y_pred, y_std = emulate(emulator, theta_arr, return_std=True)

            # Plot prediction vs true model
            y_true = MODEL_REGISTRY[model]["func"](z, **theta_input)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z, y=y_true, mode="lines", name="True Model", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=z, y=y_pred, mode="lines", name="GP Emulator", line=dict(color="#ff7f0e", dash="dash")))
            fig.add_trace(go.Scatter(x=z, y=y_pred + y_std, mode="lines", name="+1σ", line=dict(color="#ff7f0e", width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=z, y=y_pred - y_std, mode="lines", name="-1σ", line=dict(color="#ff7f0e", width=0), fill="tonexty", fillcolor="rgba(255,127,14,0.1)", showlegend=False))
            fig.update_layout(xaxis_title="z", yaxis_title="μ(z)", template=plotly_template(), height=400)
            plot_export_controls(fig, f"emulation_prediction_{model}")
            st.plotly_chart(fig, use_container_width=True)

            # Show numeric results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max |y_true - y_pred|", f"{np.max(np.abs(y_true - y_pred)):.6f}")
            with col2:
                st.metric("Mean |y_true - y_pred|", f"{np.mean(np.abs(y_true - y_pred)):.6f}")

        # Speed benchmark
        if st.button("⚡ Benchmark Speed"):
            import time
            n_trials = 100
            thetas = latin_hypercube_design(bounds, n_trials, seed=999)

            # True model
            start = time.time()
            for theta in thetas:
                _ = MODEL_REGISTRY[model]["func"](z, *theta)
            true_time = time.time() - start

            # Emulator
            start = time.time()
            for theta in thetas:
                _ = emulate(emulator, theta)
            emu_time = time.time() - start

            col1, col2, col3 = st.columns(3)
            col1.metric("True model", f"{true_time*1000/n_trials:.2f} ms/call")
            col2.metric("Emulator", f"{emu_time*1000/n_trials:.2f} ms/call")
            col3.metric("Speedup", f"{true_time/emu_time:.1f}×")