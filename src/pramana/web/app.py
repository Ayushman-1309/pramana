"""PRAMANA Streamlit Web UI — Main entry point."""
import streamlit as st

st.set_page_config(
    page_title="PRAMANA — Cosmological Inference Suite",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("🔭 PRAMANA")
st.sidebar.caption("Unified Cosmological Inference Suite")
st.sidebar.markdown("---")

pages = {
    "🏠 Home": "home",
    "📊 Data Explorer": "data_explorer",
    "🎯 Single-Probe Fit": "single_probe_fit",
    "🔗 Joint Fit": "joint_fit",
    "⚖️ Model Comparison": "model_comparison",
    "📈 Forecasting": "forecasting",
    "🤖 Emulation": "emulation",
    "⚡ Tension Analysis": "tension_analysis",
}

selection = st.sidebar.radio("Navigate", list(pages.keys()))
page = pages[selection]

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions**")
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Route to pages
if page == "home":
    from pramana.web.pages.home import render
    render()
elif page == "data_explorer":
    from pramana.web.pages.data_explorer import render
    render()
elif page == "single_probe_fit":
    from pramana.web.pages.single_probe_fit import render
    render()
elif page == "joint_fit":
    from pramana.web.pages.joint_fit import render
    render()
elif page == "model_comparison":
    from pramana.web.pages.model_comparison import render
    render()
elif page == "forecasting":
    from pramana.web.pages.forecasting import render
    render()
elif page == "emulation":
    from pramana.web.pages.emulation import render
    render()
elif page == "tension_analysis":
    from pramana.web.pages.tension_analysis import render
    render()