"""UI design tokens and helpers for PRAMANA v2.0.0."""
import streamlit as st


# Version & attribution
VERSION = "2.0.0"
DEVELOPER = "Ayushman"

# Color palette (scientific, not commercial)
# Near-black background, muted spectral-blue accent, thin gray borders
THEME = {
    "primary": "#5FB3B3",          # muted spectral cyan (accent)
    "primary_muted": "#4A9E9E",    # slightly darker for hover
    "bg": "#0D1117",               # near-black (GitHub dark-like)
    "bg_panel": "#161B22",         # panel/card background
    "text": "#E6E8EB",             # off-white soft text
    "text_muted": "#8B949E",       # muted secondary text
    "border": "#2A2E37",           # thin borders
    "success": "#3FB950",          # convergence success
    "warning": "#D29922",          # warnings
    "error": "#F85149",            # errors
}


def inject_global_css() -> None:
    """Inject global CSS: fonts, typography, layout, component styling."""
    css = f"""
    <style>
    /* ─── Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ─── CSS Variables ─── */
    :root {{
        --pr-primary: {THEME['primary']};
        --pr-primary-muted: {THEME['primary_muted']};
        --pr-bg: {THEME['bg']};
        --pr-bg-panel: {THEME['bg_panel']};
        --pr-text: {THEME['text']};
        --pr-text-muted: {THEME['text_muted']};
        --pr-border: {THEME['border']};
        --pr-success: {THEME['success']};
        --pr-warning: {THEME['warning']};
        --pr-error: {THEME['error']};
    }}

    /* ─── Base Typography ─── */
    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--pr-bg) !important;
        color: var(--pr-text) !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 500 !important;
        color: var(--pr-text) !important;
        line-height: 1.3 !important;
    }}

    h1 {{ font-size: 2rem !important; }}
    h2 {{ font-size: 1.5rem !important; }}
    h3 {{ font-size: 1.25rem !important; }}
    h4 {{ font-size: 1.1rem !important; }}

    code, kbd, pre, .stCodeBlock {{
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.9em !important;
    }}

    /* ─── Layout: capped content width ─── */
    .main .block-container {{
        max-width: 1280px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important; /* space for footer */
    }}

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {{
        background-color: var(--pr-bg-panel) !important;
        border-right: 1px solid var(--pr-border) !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--pr-text) !important;
    }}

    /* ─── Metric Cards ─── */
    [data-testid="stMetric"] {{
        background: var(--pr-bg-panel) !important;
        border: 1px solid var(--pr-border) !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        box-shadow: none !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--pr-text-muted) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        color: var(--pr-text) !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }}

    /* ─── Buttons ─── */
    .stButton > button {{
        background: var(--pr-bg-panel) !important;
        border: 1px solid var(--pr-border) !important;
        color: var(--pr-text) !important;
        border-radius: 4px !important;
        transition: all 0.15s ease !important;
    }}
    .stButton > button:hover {{
        border-color: var(--pr-primary) !important;
        background: var(--pr-primary) !important;
        color: var(--pr-bg) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: var(--pr-primary) !important;
        border-color: var(--pr-primary) !important;
        color: var(--pr-bg) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: var(--pr-primary-muted) !important;
        border-color: var(--pr-primary-muted) !important;
    }}

    /* ─── Inputs ─── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiselect > div > div > div {{
        background: var(--pr-bg) !important;
        border: 1px solid var(--pr-border) !important;
        color: var(--pr-text) !important;
        border-radius: 4px !important;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: var(--pr-primary) !important;
        box-shadow: 0 0 0 2px {THEME['primary']}40 !important;
    }}

    /* ─── Expander ─── */
    .streamlit-expanderHeader {{
        background: var(--pr-bg-panel) !important;
        border: 1px solid var(--pr-border) !important;
        border-radius: 4px !important;
        color: var(--pr-text) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }}
    .streamlit-expanderContent {{
        border: 1px solid var(--pr-border) !important;
        border-top: none !important;
        border-radius: 0 0 4px 4px !important;
        background: var(--pr-bg) !important;
    }}

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem !important;
        background: transparent !important;
        border-bottom: 1px solid var(--pr-border) !important;
        padding-bottom: 0 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border: none !important;
        color: var(--pr-text-muted) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--pr-primary) !important;
        border-bottom-color: var(--pr-primary) !important;
    }}

    /* ─── Tables / DataFrames ─── */
    .stDataFrame {{
        border: 1px solid var(--pr-border) !important;
        border-radius: 4px !important;
        overflow: hidden !important;
    }}
    .stDataFrame [data-testid="stTable"] th {{
        background: var(--pr-bg-panel) !important;
        color: var(--pr-text) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid var(--pr-border) !important;
    }}
    .stDataFrame [data-testid="stTable"] td {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        border-bottom: 1px solid var(--pr-border) !important;
    }}

    /* ─── Code Blocks ─── */
    .stCodeBlock {{
        background: #080C12 !important;
        border: 1px solid var(--pr-border) !important;
        border-radius: 6px !important;
    }}
    .stCodeBlock pre {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        line-height: 1.6 !important;
    }}

    /* ─── Plotly charts: ensure dark bg ─── */
    .js-plotly-plot .plotly {{
        background-color: transparent !important;
    }}

    /* ─── Custom Status Bar ─── */
    .pr-status-bar {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--pr-border);
        font-size: 0.85rem;
        font-family: 'Inter', sans-serif;
    }}
    .pr-status-item {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}
    .pr-status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }}
    .pr-status-dot.loaded {{ background: var(--pr-success); }}
    .pr-status-dot.missing {{ background: var(--pr-warning); }}
    .pr-status-dot.neutral {{ background: var(--pr-text-muted); }}

    /* ─── Diagnostics Panel ─── */
    .pr-diagnostics {{
        background: var(--pr-bg-panel);
        border: 1px solid var(--pr-border);
        border-radius: 6px;
        padding: 1rem;
        margin-top: 1rem;
    }}
    .pr-diagnostics h4 {{
        margin: 0 0 0.75rem 0;
        font-size: 0.95rem;
        color: var(--pr-text);
    }}
    .pr-diagnostics .row {{
        display: flex;
        gap: 1.5rem;
        margin-bottom: 0.5rem;
    }}
    .pr-diagnostics .metric {{
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }}
    .pr-diagnostics .metric-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--pr-text-muted);
    }}
    .pr-diagnostics .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: var(--pr-text);
    }}
    .pr-diagnostics .metric-value.ok {{ color: var(--pr-success); }}
    .pr-diagnostics .metric-value.warn {{ color: var(--pr-warning); }}
    .pr-diagnostics .metric-value.bad {{ color: var(--pr-error); }}

    /* ─── Footer ─── */
    .pr-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--pr-bg-panel);
        border-top: 1px solid var(--pr-border);
        padding: 0.5rem 1rem;
        text-align: center;
        font-size: 0.7rem;
        color: var(--pr-text-muted);
        z-index: 100;
        font-family: 'Inter', sans-serif;
    }}
    .pr-footer a {{
        color: var(--pr-primary);
        text-decoration: none;
    }}
    .pr-footer a:hover {{
        text-decoration: underline;
    }}

    /* ─── Remove default Streamlit padding bottom ─── */
    .main .block-container {{
        padding-bottom: 4.5rem !important;
    }}

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--pr-bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--pr-border); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--pr-text-muted); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def plotly_template() -> str:
    """
    Return the appropriate Plotly template name based on Streamlit's active theme.
    """
    # Streamlit doesn't expose theme directly, but we can infer from config
    # For dark theme (our default), use plotly_dark
    # Could be enhanced with st.get_option("theme.base") when available
    return "plotly_dark"


def render_status_bar() -> None:
    """Render the persistent status bar showing loaded probes, model, and fit status."""
    probes = []
    # Pantheon+
    if "pantheon_data" in st.session_state:
        n = len(st.session_state["pantheon_data"]["z"])
        probes.append(f'<span class="pr-status-item"><span class="pr-status-dot loaded"></span>Pantheon+ ({n} SNe)</span>')
    else:
        probes.append('<span class="pr-status-item"><span class="pr-status-dot missing"></span>Pantheon+ (not loaded)</span>')
    # DESI (built-in)
    probes.append('<span class="pr-status-item"><span class="pr-status-dot loaded"></span>DESI DR2 (built-in)</span>')
    # ACT
    probes.append('<span class="pr-status-item"><span class="pr-status-dot neutral"></span>ACT DR6 (not loaded)</span>')
    
    model = st.session_state.get("active_model", "—")
    
    st.markdown(f"""
    <div class="pr-status-bar">
        {''.join(probes)}
        <span class="pr-status-item" style="margin-left: auto;">
            Model: <code style="font-family: 'JetBrains Mono', monospace;">{model}</code>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_diagnostics_panel(summary: dict) -> None:
    """
    Render a standardized diagnostics panel (R-hat, ESS, acceptance).
    `summary` should contain: rhat, ess, acceptance, tau
    """
    rhat = summary.get("rhat", 1.0)
    ess = summary.get("ess", 0)
    acc = summary.get("acceptance", 0.0)
    tau = summary.get("tau", 0.0)
    
    rhat_class = "ok" if rhat < 1.01 else ("warn" if rhat < 1.1 else "bad")
    ess_class = "ok" if ess > 1000 else ("warn" if ess > 100 else "bad")
    acc_class = "ok" if 0.2 <= acc <= 0.5 else "warn"
    
    st.markdown(f"""
    <div class="pr-diagnostics">
        <h4>Convergence Diagnostics</h4>
        <div class="row">
            <div class="metric">
                <span class="metric-label">R-hat (max)</span>
                <span class="metric-value {rhat_class}">{rhat:.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">ESS (min)</span>
                <span class="metric-value {ess_class}">{ess:.0f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Acceptance</span>
                <span class="metric-value {acc_class}">{acc:.2%}</span>
            </div>
            <div class="metric">
                <span class="metric-label">τ (max)</span>
                <span class="metric-value">{tau:.1f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer() -> None:
    """Render the fixed footer with version and developer credit."""
    st.markdown(f"""
    <div class="pr-footer">
        PRAMANA v{VERSION} · Developed by {DEVELOPER} ·
        <a href="https://github.com/PantheonPlusSH0ES/DataRelease" target="_blank">Pantheon+SH0ES</a> ·
        <a href="https://www.desi.lbl.gov/" target="_blank">DESI DR2</a> ·
        <a href="https://lambda.gsfc.nasa.gov/" target="_blank">ACT DR6</a>
    </div>
    """, unsafe_allow_html=True)