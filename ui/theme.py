import streamlit as st


APP_CSS = """
<style>
    :root {
        --app-ink: #202326;
        --app-muted: #687078;
        --app-border: #dfe3e6;
        --app-surface: #ffffff;
        --app-canvas: #f5f6f7;
        --app-blue: #245fce;
        --app-green: #18794e;
        --app-amber: #a15c00;
        --app-red: #b4232f;
    }

    [data-testid="stAppViewContainer"] { background: var(--app-canvas); }
    [data-testid="stHeader"] { background: rgba(245, 246, 247, 0.94); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] { background: #202326; }
    [data-testid="stSidebar"] * { color: #f4f5f6 !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        border-radius: 6px;
        padding: 0.45rem 0.55rem;
    }
    [data-testid="stSidebar"] [aria-checked="true"] {
        background: #32373b;
    }
    [data-testid="stSidebar"] hr { border-color: #3c4247; }
    [data-testid="stSidebar"] .stCaption { color: #b9c0c5; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #f4f5f6 !important;
    }

    [data-testid="stMain"] { color: var(--app-ink); }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        color: var(--app-ink) !important;
        letter-spacing: 0;
    }
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
        color: var(--app-ink) !important;
    }
    h1, h2, h3 { letter-spacing: 0; }
    h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
    h2 { font-size: 1.35rem; font-weight: 650; }
    h3 { font-size: 1.05rem; font-weight: 650; }
    p, label, [data-testid="stCaptionContainer"] { letter-spacing: 0; }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--app-surface);
        border: 1px dashed #aeb6bd;
        border-radius: 6px;
        min-height: 126px;
    }
    [data-testid="stExpander"] {
        background: var(--app-surface);
        border: 1px solid var(--app-border);
        border-radius: 6px;
    }
    [data-testid="stMetric"] {
        background: var(--app-surface);
        border: 1px solid var(--app-border);
        border-radius: 6px;
        padding: 0.75rem 0.9rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    [data-testid="stExpander"] [data-testid="stMetric"] {
        background: transparent;
        border: 0;
        padding: 0.35rem 0;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--app-border); }

    .stButton > button, .stDownloadButton > button {
        border-radius: 6px;
        min-height: 2.5rem;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: var(--app-blue);
        border-color: var(--app-blue);
    }
    .stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div {
        border-radius: 6px;
    }
    .stButton > button p, .stDownloadButton > button p, [data-testid="stLinkButton"] p {
        overflow-wrap: anywhere;
        white-space: normal;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }

    .app-kicker {
        color: var(--app-muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .app-subtitle { color: var(--app-muted); margin: 0 0 1.25rem 0; }
    .status-row {
        align-items: center;
        display: flex;
        font-size: 0.84rem;
        gap: 0.45rem;
        margin: 0.25rem 0;
    }
    .status-dot { border-radius: 50%; display: inline-block; height: 8px; width: 8px; }
    .status-ok { background: #38a169; }
    .status-warn { background: #d69e2e; }
    .status-off { background: #d64550; }
    .result-meta { color: var(--app-muted); font-size: 0.82rem; }
    .queue-name { color: var(--app-ink); font-weight: 600; }

    @media (max-width: 780px) {
        [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 3rem; }
        h1 { font-size: 1.65rem; }
        [data-testid="stMetricValue"] { font-size: 1.2rem; }
    }
</style>
"""


def apply_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
