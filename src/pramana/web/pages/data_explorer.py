"""PRAMANA Web UI — Data Hub (Data Explorer).

Every PRAMANA dataset family (SN Ia, BAO, CMB, JWST-era) is loaded here
with two options: manual download (official links + upload/path) and
synthetic generation. NO real data is bundled with the app — nothing is
loaded until you download it or generate a synthetic version.
"""
import streamlit as st
from pramana.web.components.data_hub import (
    dataset_loader,
    render_family_summary,
)
from pramana.web.components.ui import render_status_bar


def render():
    render_status_bar()
    st.title("Data Hub")

    st.markdown("""
    Load every PRAMANA dataset family here. **No observational data is
    bundled** with the app — each family supports:

    - **📥 Download & Upload** — fetch the official release, then upload it
    - **📂 File Path** — point at files already on disk
    - **🧪 Synthetic** — generate realistic mock data for testing

    Data loaded here is stored in session state and reused by the fit,
    tension, forecast, compression, and reweighting pages.
    """)

    # A compact status overview
    col1, col2, col3, col4 = st.columns(4)
    for col, fam in zip([col1, col2, col3, col4], ["pantheon", "bao", "cmb", "jwst"]):
        key = {
            "pantheon": "pantheon_data",
            "bao": "bao_data",
            "cmb": "cmb_data",
            "jwst": "jwst_data",
        }[fam]
        loaded = key in st.session_state
        src = st.session_state[key].get("source", "?") if loaded else ""
        tag = "✅" if loaded else "—"
        label = {"pantheon": "SN Ia", "bao": "BAO", "cmb": "CMB", "jwst": "JWST"}[fam]
        col.metric(label, tag, f"{src}" if loaded else "not loaded")

    st.markdown("---")

    # --- SN Ia ---
    dataset_loader("pantheon", key="pantheon_data", show_instructions=False)
    render_family_summary("pantheon", key="pantheon_data")

    # --- BAO ---
    dataset_loader("bao", key="bao_data", show_instructions=False)
    render_family_summary("bao", key="bao_data")

    # --- CMB ---
    dataset_loader("cmb", key="cmb_data", show_instructions=False)
    render_family_summary("cmb", key="cmb_data")

    # --- JWST-era ---
    dataset_loader("jwst", key="jwst_data", show_instructions=False)
    render_family_summary("jwst", key="jwst_data")