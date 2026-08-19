"""PRAMANA Streamlit Web UI — Main entry point."""
import streamlit as st

st.set_page_config(
    page_title="PRAMANA — The Unified Cosmological Inference Suite",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import UI helpers
from pramana.web.components.ui import inject_global_css, VERSION, DEVELOPER, SUITE_NAME

# Import all page render functions
from pramana.web.pages import (
    home,
    data_explorer,
    single_probe_fit,
    joint_fit,
    model_comparison,
    forecasting,
    emulation,
    reweighting,
    compression,
    tension_analysis,
    about,
)

# Inject global CSS (fonts, theme, components)
inject_global_css()

# ─── Native Navigation with st.Page (using callables) ───
pages = {
    "Home": st.Page(
        home.render,
        title="Home",
        icon=":material/home:",
        url_path="home",
        default=True,
    ),
    "Data Explorer": st.Page(
        data_explorer.render,
        title="Data Explorer",
        icon=":material/analytics:",
        url_path="data-explorer",
    ),
    "Single-Probe Fit": st.Page(
        single_probe_fit.render,
        title="Single-Probe Fit",
        icon=":material/tune:",
        url_path="single-probe-fit",
    ),
    "Joint Fit": st.Page(
        joint_fit.render,
        title="Joint Fit",
        icon=":material/merge:",
        url_path="joint-fit",
    ),
    "Model Comparison": st.Page(
        model_comparison.render,
        title="Model Comparison",
        icon=":material/balance:",
        url_path="model-comparison",
    ),
    "Forecasting": st.Page(
        forecasting.render,
        title="Forecasting",
        icon=":material/trending_up:",
        url_path="forecasting",
    ),
    "Emulation": st.Page(
        emulation.render,
        title="Emulation",
        icon=":material/psychology:",
        url_path="emulation",
    ),
    "Importance Reweighting": st.Page(
        reweighting.render,
        title="Importance Reweighting",
        icon=":material/swap_horiz:",
        url_path="reweighting",
    ),
    "MOPED Compression": st.Page(
        compression.render,
        title="MOPED Compression",
        icon=":material/compress:",
        url_path="compression",
    ),
    "Tension Analysis": st.Page(
        tension_analysis.render,
        title="Tension Analysis",
        icon=":material/warning:",
        url_path="tension-analysis",
    ),
    "About": st.Page(
        about.render,
        title="About",
        icon=":material/info:",
        url_path="about",
    ),
}

# Build navigation
pg = st.navigation({
    "PRAMANA-UCIS": [
        pages["Home"],
        pages["Data Explorer"],
        pages["Single-Probe Fit"],
        pages["Joint Fit"],
        pages["Model Comparison"],
    ],
    "Inference": [
        pages["Forecasting"],
        pages["Emulation"],
        pages["Importance Reweighting"],
        pages["MOPED Compression"],
    ],
    "Tension": [
        pages["Tension Analysis"],
    ],
    "About": [
        pages["About"],
    ],
})
pg.run()

# ─── Sidebar: PRAMANA-UCIS wordmark + Quick Actions ───
st.sidebar.markdown(f"""
<div style="text-align:center; padding:1rem 0.5rem; border-bottom:1px solid var(--pr-border); margin-bottom:0.5rem;">
    <div style="font-family:'Source Serif 4',Georgia,serif; font-size:1.25rem; font-weight:600; color:var(--pr-primary);">
        PRAMANA-UCIS
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.7rem; color:var(--pr-text-muted); margin-top:0.25rem;">
        Unified Cosmological Inference Suite
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions**")
if st.sidebar.button("🔄 Clear Cache", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
if st.sidebar.button("🗑️ Reset Session", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"v{VERSION} · Developed by {DEVELOPER}")