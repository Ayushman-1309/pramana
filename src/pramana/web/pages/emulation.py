"""PRAMANA Web UI — Emulation page."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.gp_emulator import latin_hypercube_design, train_emulator, emulate, validate_emulator


def render():
    st.title("🤖 GP Emulation")

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
    z = data['z']

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

        st.session_state['emulator'] = emulator
        st.session_state['emulator_param_names'] = param_names
        st.session_state['emulator_z'] = z
        st.success("Emulator trained and validated!")

    # Prediction
    if 'emulator' in st.session_state:
        emulator = st.session_state['emulator']
        param_names = st.session_state['emulator_param_names']

        st.markdown("---")
        st.subheader("Evaluate Emulator")

        theta_input = {}
        for p in param_names:
            lo, hi = MODEL_REGISTRY[param_names[0]]["priors"].get(p, (0, 1)) if p in MODEL_REGISTRY else (0, 1)
            # Use prior bounds from model registry
            spec = MODEL_REGISTRY[list(MODEL_REGISTRY.keys())[0]]
            if p in spec["priors"]:
                lo, hi = spec["priors"][p]
            else:
                lo, hi = 0.0, 1.0
            theta_input[p] = st.slider(f"{p}", float(lo), float(hi), float((lo+hi)/2), key=f"emu_{p}")

        if st.button("Predict"):
            theta_arr = np.array([theta_input[p] for p in param_names])
            y_pred, y_std = emulate(emulator, theta_arr, return_std=True)

            st.write("**Prediction (model at data points):**")
            st.write(y_pred)
            st.write("**Uncertainty (1σ):**")
            st.write(y_std)

            # Plot prediction vs true model
            y_true = MODEL_REGISTRY[list(MODEL_REGISTRY.keys())[0]]["func"](z, **theta_input) if list(MODEL_REGISTRY.keys())[0] == list(MODEL_REGISTRY.keys())[0] else None
            # Actually use the selected model
            y_true = MODEL_REGISTRY[model]["func"](z, **theta_input)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z, y=y_true, mode='lines', name='True Model', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=z, y=y_pred, mode='lines', name='GP Emulator', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=z, y=y_pred + y_std, mode='lines', name='+1σ', line=dict(color='red', width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=z, y=y_pred - y_std, mode='lines', name='-1σ', line=dict(color='red', width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', showlegend=False))
            fig.update_layout(xaxis_title="z", yaxis_title="μ(z)", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

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