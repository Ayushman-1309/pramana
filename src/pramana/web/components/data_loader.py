"""Shared data-loading component for Pantheon+ SN data."""
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.utils.validators import validate_pantheon_data, validate_pantheon_cov
from pramana.web.components.ui import plotly_template


@st.cache_data(show_spinner=False)
def _load_pantheon_cached(data_bytes: bytes | None, cov_bytes: bytes | None, data_path: str | None, cov_path: str | None):
    """Cached loader. Exactly one of (data_bytes, cov_bytes) or (data_path, cov_path) must be provided."""
    if data_bytes is not None and cov_bytes is not None:
        import io
        df = pd.read_csv(io.BytesIO(data_bytes), sep=r"\s+")
        keep_idx = np.arange(len(df))
        if "IS_CALIBRATOR" in df.columns:
            mask = df["IS_CALIBRATOR"].values == 0
            df = df[mask].reset_index(drop=True)
            keep_idx = keep_idx[mask]
        z = df["zHD"].values
        mb = df["m_b_corr"].values
        
        # Parse covariance file: first line is N, then N*N values (row-major)
        cov_text = cov_bytes.decode()
        lines = cov_text.strip().splitlines()
        n = int(lines[0].strip())
        # Join remaining lines and load with np.loadtxt
        cov_data = "\n".join(lines[1:])
        cov = np.loadtxt(io.StringIO(cov_data), dtype=float).reshape((n, n))
        
        if keep_idx is not None:
            cov = cov[np.ix_(keep_idx, keep_idx)]
        return z, mb, cov, df
    elif data_path is not None and cov_path is not None:
        return load_pantheon(data_path, cov_path)
    else:
        raise ValueError("Must provide either file bytes or file paths")


def _validate_files(data_path: str, cov_path: str, n_data: int) -> dict:
    """Validate Pantheon+ data and covariance files."""
    val = validate_pantheon_data(data_path)
    val_cov = validate_pantheon_cov(cov_path, n_data)
    return {"data": val, "cov": val_cov}


def _auto_detect_pantheon_files() -> tuple[Path | None, Path | None]:
    """Auto-detect Pantheon+ files in common locations."""
    search_dirs = [
        Path.cwd() / "data" / "pantheon",
        Path(__file__).parents[3] / "data" / "pantheon",
        Path.home() / "data" / "pantheon",
    ]
    for d in search_dirs:
        dat_files = list(d.glob("*.dat"))
        cov_files = list(d.glob("*.cov"))
        if dat_files and cov_files:
            return dat_files[0], cov_files[0]
    return None, None


def pantheon_loader(key: str = "pantheon_data", show_instructions: bool = True) -> dict | None:
    """
    Unified Pantheon+ data loader component.
    
    Returns dict with keys: z, mb_obs, cov, df (or None for synthetic).
    Stores result in st.session_state[key].
    """
    st.subheader("📁 Pantheon+ Data")
    
    if show_instructions:
        with st.expander("📋 Instructions & Help", expanded=False):
            st.markdown("""
            **Required files from [Pantheon+SH0ES Data Release](https://github.com/PantheonPlusSH0ES/DataRelease):**
            - `Pantheon+SH0ES.dat` — whitespace-delimited ASCII with header
            - `Pantheon+SH0ES_STAT+SYS.cov` — covariance matrix (first line N, then N×N values)
            
            **Required columns:** `zHD`, `m_b_corr`, `IS_CALIBRATOR`
            
            **Tip:** Place files in `data/pantheon/` for auto-detection, or use the upload tab below.
            """)
    
    # Auto-detect
    auto_dat, auto_cov = _auto_detect_pantheon_files()
    if auto_dat and auto_cov:
        st.success(f"🔍 Auto-detected: `{auto_dat.name}` & `{auto_cov.name}`")
    
    # Tabs
    tab_upload, tab_path, tab_synthetic = st.tabs(["📤 Upload", "📂 File Path", "🧪 Synthetic"])
    
    with tab_upload:
        col1, col2 = st.columns(2)
        with col1:
            dat_file = st.file_uploader("Pantheon+ data (.dat)", type=["dat"], key=f"{key}_dat_upload")
        with col2:
            cov_file = st.file_uploader("Covariance (.cov)", type=["cov"], key=f"{key}_cov_upload")
        
        if st.button("Load Uploaded Files", type="primary", key=f"{key}_load_upload"):
            if dat_file is None or cov_file is None:
                st.error("Please upload both .dat and .cov files")
            else:
                with st.spinner("Loading uploaded files..."):
                    try:
                        dat_bytes = dat_file.read()
                        cov_bytes = cov_file.read()
                        z, mb_obs, cov, df = _load_pantheon_cached(dat_bytes, cov_bytes, None, None)
                        st.session_state[key] = {"z": z, "mb_obs": mb_obs, "cov": cov, "df": df}
                        st.success(f"✅ Loaded {len(z)} SNe from uploaded files")
                    except Exception as e:
                        st.error(f"Error loading files: {e}")
    
    with tab_path:
        col1, col2 = st.columns(2)
        with col1:
            default_dat = str(auto_dat) if auto_dat else "data/pantheon/Pantheon+SH0ES.dat"
            data_path = st.text_input("Data file (.dat)", default_dat, key=f"{key}_dat_path")
        with col2:
            default_cov = str(auto_cov) if auto_cov else "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov"
            cov_path = st.text_input("Covariance file (.cov)", default_cov, key=f"{key}_cov_path")
        
        if st.button("Load from Path", type="primary", key=f"{key}_load_path"):
            dp = Path(data_path)
            cp = Path(cov_path)
            if not dp.exists():
                st.error(f"Data file not found: `{dp}`. Check path or use Upload tab.")
            elif not cp.exists():
                st.error(f"Covariance file not found: `{cp}`. Check path or use Upload tab.")
            else:
                with st.spinner("Loading from path..."):
                    try:
                        z, mb_obs, cov, df = _load_pantheon_cached(None, None, data_path, cov_path)
                        val = _validate_files(data_path, cov_path, len(z))
                        st.session_state[key] = {"z": z, "mb_obs": mb_obs, "cov": cov, "df": df}
                        st.success(f"✅ Loaded {len(z)} SNe")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("N SNe", len(z))
                        col2.metric("z range", f"{z.min():.3f} – {z.max():.3f}")
                        col3.metric("Cov condition", f"{val['cov']['condition_number']:.2e}")
                    except Exception as e:
                        st.error(f"Error loading files: {e}")
    
    with tab_synthetic:
        st.info("Generate synthetic Pantheon+ data for testing (no files needed)")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_syn = st.number_input("Number of SNe", 50, 2000, 300, key=f"{key}_n_syn")
        with col2:
            om_syn = st.number_input("True Ωₘ", 0.05, 0.6, 0.3, key=f"{key}_om_syn")
        with col3:
            seed_syn = st.number_input("Random seed", 0, 9999, 42, key=f"{key}_seed_syn")
        
        if st.button("Generate Synthetic Data", type="primary", key=f"{key}_gen_syn"):
            with st.spinner("Generating synthetic data..."):
                z, mb_obs, cov = make_synthetic_dataset(n=n_syn, seed=seed_syn, Om_true=om_syn)
                st.session_state[key] = {"z": z, "mb_obs": mb_obs, "cov": cov, "df": None}
                st.success(f"✅ Generated {len(z)} synthetic SNe")
    
    # Return loaded data if available
    if key in st.session_state:
        return st.session_state[key]
    return None


def render_data_summary(key: str = "pantheon_data"):
    """Render a summary of loaded Pantheon+ data."""
    if key not in st.session_state:
        return
    data = st.session_state[key]
    z, mb_obs, cov, df = data["z"], data["mb_obs"], data["cov"], data["df"]
    
    st.markdown("---")
    st.subheader("Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N SNe", len(z))
    col2.metric("z range", f"{z.min():.3f} – {z.max():.3f}")
    col3.metric("Mean m_b", f"{mb_obs.mean():.3f}")
    col4.metric("Cov condition", f"{np.linalg.cond(cov):.2e}")
    
    if df is not None:
        with st.expander("View DataFrame (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    
    # Hubble diagram
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=mb_obs, mode="markers",
                              marker=dict(size=4, opacity=0.5, color="#ff7f0e"),
                              name="SN data"))
    fig.update_layout(title="Pantheon+ Hubble Diagram",
                      xaxis_title="Redshift z", yaxis_title="m_b,corr",
                      height=350, template=plotly_template())
    st.plotly_chart(fig, use_container_width=True)