
import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Customer Value & Growth | Success Intelligence",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_semrush_churn_dataset_v10_csm.csv"
MODEL_PATH = BASE_DIR / "best_churn_model_semrush_v10_csm.pkl"

EXCLUDED_COLUMNS = [
    "account_id", "account_name", "churn_label", "synthetic_true_churn_probability",
    # CSM assignment fields are operational / UX fields, not model inputs.
    "csm_name", "csm_region", "csm_timezone", "csm_focus_area", "csm_seniority",
    "csm_manager", "book_tier", "csm_portfolio_size", "csm_assigned_account_rank",
    "target_book_size",
]


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg-0: #091523;
            --bg-1: #0D1B2F;
            --bg-2: #102842;
            --ink: #0D1726;
            --ink-soft: #2A3850;
            --muted: #627188;
            --muted-2: #8492A8;
            --line: rgba(145, 161, 185, 0.28);
            --line-strong: rgba(97, 116, 141, 0.32);
            --surface: rgba(255, 255, 255, 0.92);
            --surface-2: rgba(255, 255, 255, 0.72);
            --surface-dark: rgba(9, 21, 35, 0.80);
            --blue: #4D7BF3;
            --blue-2: #2A5CEB;
            --teal: #1E9AA7;
            --teal-2: #137A8A;
            --violet: #6F52ED;
            --green: #1E9B5D;
            --amber: #D38A1D;
            --red: #D95967;
            --shadow-lg: 0 22px 60px rgba(7, 20, 39, 0.12);
            --shadow-md: 0 14px 34px rgba(7, 20, 39, 0.08);
            --radius-xl: 26px;
            --radius-lg: 20px;
            --radius-md: 16px;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 0% 0%, rgba(77, 123, 243, 0.10), transparent 22%),
                radial-gradient(circle at 100% 0%, rgba(30, 154, 167, 0.12), transparent 24%),
                linear-gradient(180deg, #F4F8FC 0%, #EEF4FB 42%, #F9FBFD 100%);
        }
        .main .block-container {
            max-width: 1460px;
            padding-top: 1.35rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: var(--ink);
            letter-spacing: -0.03em;
            font-family: 'Space Grotesk', 'Inter', sans-serif;
        }
        p, li, label, .stMarkdown, .stCaption, .stText, .st-emotion-cache-10trblm {
            color: var(--ink-soft);
        }
        small, .small-muted { color: var(--muted); }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 100% 0%, rgba(77,123,243,.20), transparent 24%),
                linear-gradient(180deg, rgba(255,255,255,.96) 0%, rgba(245,249,253,.98) 100%);
            border-right: 1px solid rgba(148,163,184,.22);
            box-shadow: inset -1px 0 0 rgba(255,255,255,.65);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 1rem;
            font-family: 'Space Grotesk', 'Inter', sans-serif;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }
        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
        section[data-testid="stSidebar"] [data-testid="stNumberInputRootElement"] {
            background: rgba(255,255,255,.86);
            border-radius: 16px;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-testid="stTextInputRootElement"] > div,
        section[data-testid="stSidebar"] [data-testid="stNumberInputRootElement"] > div {
            border: 1px solid rgba(148,163,184,.28) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 24px rgba(15,23,42,.04);
            background: rgba(255,255,255,.88);
        }
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
            font-size: 12px;
            font-weight: 800;
            color: var(--muted-2);
            letter-spacing: .02em;
            text-transform: uppercase;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(148,163,184,.18);
            margin: 1rem 0;
        }

        /* Hero */
        .hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 88% 18%, rgba(77,123,243,.46), transparent 18%),
                radial-gradient(circle at 15% 0%, rgba(255,255,255,.12), transparent 14%),
                linear-gradient(140deg, #091523 0%, #10253F 42%, #123D57 76%, #16697A 100%);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 34px;
            padding: 36px 40px 38px;
            color: white;
            box-shadow: 0 24px 70px rgba(7, 20, 39, .24);
            margin-bottom: 22px;
        }
        .hero:before {
            content: "";
            position: absolute;
            inset: auto -8% -38% auto;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(255,255,255,.18), transparent 64%);
            filter: blur(8px);
        }
        .hero-eyebrow {
            display:inline-flex;
            align-items:center;
            gap: 8px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing:.14em;
            font-weight: 900;
            padding: 8px 14px;
            border-radius:999px;
            background: rgba(255,255,255,.10);
            border:1px solid rgba(255,255,255,.18);
            margin-bottom:16px;
            backdrop-filter: blur(12px);
        }
        .hero-title {
            font-size: 46px;
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            font-weight: 700;
            line-height:1.02;
            letter-spacing: -0.045em;
            margin-bottom: 12px;
        }
        .hero-sub {
            font-size: 16px;
            line-height: 1.62;
            max-width: 1080px;
            color: rgba(255,255,255,.90);
        }
        .hero-sub strong { color: white; }

        /* Generic cards */
        .metric-card, .panel-card, .playbook-card, .signal-card, .story-card, .routing-card, .ux-card,
        .scenario-card, .story-bar-list, .signal-section, .method-card, .queue-card {
            background: linear-gradient(180deg, rgba(255,255,255,.94) 0%, rgba(255,255,255,.88) 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: var(--shadow-md);
            padding: 20px;
            backdrop-filter: blur(10px);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        .metric-card:hover, .story-card:hover, .routing-card:hover, .queue-card:hover, .scenario-card:hover, .signal-brief-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 22px 44px rgba(7,20,39,.10);
            border-color: rgba(77,123,243,.28);
        }
        .metric-label, .playbook-label, .queue-label, .scenario-label, .signal-label, .story-kicker, .routing-kicker, .ux-kicker {
            color: var(--muted-2);
            font-size: 11px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: .11em;
            margin-bottom: 8px;
        }
        .metric-value, .queue-value, .ux-big {
            color: var(--ink);
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            font-size: 34px;
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: .98;
            margin-bottom: 8px;
        }
        .metric-note, .queue-note, .playbook-body, .ux-body, .routing-body, .story-body, .cta-body, .section-sub {
            color: var(--ink-soft);
            font-size: 13px;
            line-height: 1.55;
        }
        .section-head {
            margin: 28px 0 14px;
            padding-top: 4px;
        }
        .section-title {
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            font-size: 24px;
            font-weight: 700;
            color: var(--ink);
            display:flex;
            align-items:center;
            gap:10px;
            letter-spacing: -0.03em;
        }
        .section-title::before {
            content:"";
            display:inline-block;
            width:14px;
            height:14px;
            border-radius:999px;
            background: linear-gradient(135deg,var(--blue),var(--teal));
            box-shadow: 0 0 0 6px rgba(77,123,243,.10);
        }
        .mini-pill, .driver-chip, .ux-owner {
            display:inline-flex;
            align-items:center;
            padding: 6px 10px;
            border-radius:999px;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: .01em;
            border: 1px solid rgba(148,163,184,.16);
            margin: 3px;
        }
        .mini-pill { background: rgba(77,123,243,.08); color: #2550CC; }
        .driver-chip { background: rgba(241,245,249,.95); color:#415066; }
        .ux-owner { background: #EFF6FF; color:#2550CC; }
        .ux-owner.sales { background:#F3E8FF; color:#6E33D8; }
        .ux-owner.product { background:#DFF7F5; color:#0B7C79; }
        .ux-owner.csm { background:#F1F5F9; color:#334155; }
        .ux-owner.support { background:#E0EEFF; color:#1D4ED8; }

        .risk-badge, .signal-status, .urgency-pill, .ux-impact {
            display:inline-flex;
            align-items:center;
            gap: 6px;
            border-radius:999px;
            padding:7px 12px;
            font-size:11px;
            font-weight:900;
            letter-spacing:.02em;
        }
        .risk-high, .signal-status.risk, .urgency-high { background:#FDE8EB; color:#A12B3C; border:1px solid #F7C6CD; }
        .risk-medium, .signal-status.watch, .urgency-medium { background:#FFF2DD; color:#A35F00; border:1px solid #F4D49A; }
        .risk-low, .signal-status.good, .urgency-low, .ux-impact.high { background:#E5F8EE; color:#157347; border:1px solid #C5EDD7; }
        .ux-impact.medium { background:#FFF2DD; color:#A35F00; border:1px solid #F4D49A; }
        .ux-impact.low { background:#F2F5F8; color:#54657B; border:1px solid #E2E8F0; }

        .cta-box, .briefing-note, .story-note, .handoff-box, .routing-note, .signal-cta, .story-callout {
            background: linear-gradient(135deg, rgba(77,123,243,.06) 0%, rgba(30,154,167,.04) 100%);
            border:1px solid rgba(116, 169, 233, .34);
            border-radius: 20px;
            padding: 16px 18px;
            color:#334155;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.7);
        }
        .cta-title { color:#154F9B; font-weight:900; font-size:18px; margin-bottom:8px; }

        .bar-wrap, .story-track, .signal-progress {
            background: #E6EDF5;
            border-radius: 999px;
            overflow: hidden;
            box-shadow: inset 0 1px 2px rgba(7,20,39,.04);
        }
        .bar-wrap { height: 12px; margin: 6px 0 12px; }
        .story-track { height: 12px; }
        .signal-progress { height: 9px; margin: 8px 0 10px; }
        .bar-fill, .story-fill, .signal-progress-fill {
            background: linear-gradient(90deg, var(--blue), var(--teal));
            border-radius: 999px;
        }
        .bar-fill { height:12px; }
        .story-fill { height:12px; }
        .signal-progress-fill { height:9px; }
        .bar-fill.warn, .story-fill.hot, .signal-progress-fill.risk { background: linear-gradient(90deg, #F0A936, #D95967); }
        .story-fill.medium, .signal-progress-fill.watch { background: linear-gradient(90deg, #7EB8FA, #4D7BF3); }
        .signal-progress-fill.good { background: linear-gradient(90deg, #5CCB93, #1E9B5D); }

        .scenario-hero {
            background:
                radial-gradient(circle at 100% 0%, rgba(30,154,167,.16), transparent 24%),
                linear-gradient(135deg, rgba(255,255,255,.95), rgba(240,249,252,.94));
            border: 1px solid rgba(30,154,167,.24);
            border-radius: 26px;
            padding: 22px;
            box-shadow: var(--shadow-md);
            margin: 12px 0 16px;
        }
        .scenario-card.best { border-top:5px solid var(--green); }
        .scenario-card.medium { border-top:5px solid var(--amber); }
        .scenario-card.low { border-top:5px solid #64748B; }
        .signal-brief-card {
            border:1px solid rgba(148,163,184,.22);
            border-radius: 20px;
            padding: 16px;
            background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(255,255,255,.90));
            min-height: 210px;
            position: relative;
            box-shadow: 0 10px 28px rgba(15,23,42,.05);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        .signal-brief-card.good { border-top:5px solid var(--green); }
        .signal-brief-card.watch { border-top:5px solid var(--amber); }
        .signal-brief-card.risk { border-top:5px solid var(--red); }
        .signal-name { color:var(--ink); font-size:15px; font-weight:900; margin-bottom:7px; display:flex; align-items:center; gap:7px; }
        .signal-value-big { color:var(--ink); font-family:'Space Grotesk','Inter',sans-serif; font-size:25px; font-weight:700; margin-bottom:8px; }

        .routing-card.support { border-top:5px solid var(--blue); }
        .routing-card.sales { border-top:5px solid var(--violet); }
        .routing-card.product { border-top:5px solid var(--teal); }
        .routing-card.csm { border-top:5px solid #64748B; }
        .routing-team, .playbook-title, .ux-title, .signal-section-title, .story-headline {
            color: var(--ink);
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        .routing-team { font-size: 18px; margin-bottom: 8px; }
        .signal-section-title { font-size: 20px; margin-bottom:4px; }
        .playbook-title { font-size: 16px; margin-bottom: 6px; }
        .story-headline { font-size: 20px; line-height: 1.18; margin-bottom: 8px; }

        /* Dataframes & metrics */
        div[data-testid="stDataFrame"], [data-testid="stMetric"] {
            border-radius: 18px;
        }
        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border:1px solid rgba(148,163,184,.20);
            box-shadow: 0 10px 24px rgba(15,23,42,.05);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,255,255,.94), rgba(255,255,255,.88));
            border: 1px solid var(--line);
            padding: 16px 18px;
            box-shadow: var(--shadow-md);
        }
        [data-testid="stMetricLabel"] { color: var(--muted-2); font-weight: 800; }
        [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', 'Inter', sans-serif; font-weight: 700; }

        /* Form controls */
        [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255,255,255,.54);
            padding: 6px;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.20);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.8);
            margin-bottom: .6rem;
        }
        button[data-baseweb="tab"] {
            height: 42px;
            padding: 0 16px;
            border-radius: 14px !important;
            color: var(--muted);
            font-weight: 800;
            border: 1px solid transparent !important;
            background: transparent !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(77,123,243,.12), rgba(30,154,167,.10)) !important;
            color: var(--ink) !important;
            border-color: rgba(77,123,243,.18) !important;
            box-shadow: 0 8px 20px rgba(77,123,243,.08);
        }
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            border: 1px solid rgba(77,123,243,.18) !important;
            border-radius: 16px !important;
            background: linear-gradient(135deg, #12345C 0%, #2A5CEB 52%, #1E9AA7 100%) !important;
            color: white !important;
            font-weight: 800 !important;
            padding: .66rem 1rem !important;
            box-shadow: 0 14px 28px rgba(18,52,92,.16) !important;
            transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            filter: saturate(1.02);
            box-shadow: 0 20px 32px rgba(18,52,92,.20) !important;
        }
        .stButton > button[kind="secondary"],
        .stDownloadButton > button[kind="secondary"] {
            background: linear-gradient(180deg, rgba(255,255,255,.95), rgba(246,249,252,.95)) !important;
            color: var(--ink) !important;
            border: 1px solid rgba(148,163,184,.28) !important;
            box-shadow: 0 8px 20px rgba(15,23,42,.06) !important;
        }
        [data-baseweb="select"] > div,
        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInputRootElement"] > div,
        [data-testid="stDateInputField"] {
            border-radius: 16px !important;
            border-color: rgba(148,163,184,.30) !important;
            background: rgba(255,255,255,.92) !important;
            box-shadow: 0 6px 18px rgba(15,23,42,.04);
        }
        [data-testid="stTextArea"] textarea {
            min-height: 110px;
            padding-top: 10px;
        }
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stTextArea"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stDateInput"] label {
            color: var(--muted-2) !important;
            font-size: 12px !important;
            font-weight: 800 !important;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        details {
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.22);
            background: rgba(255,255,255,.72);
            box-shadow: var(--shadow-md);
            padding: 2px 6px;
        }
        details summary {
            font-weight: 800;
            color: var(--ink);
        }
        .tooltip-wrap { position:relative; display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; border-radius:999px; background:#EAF2FF; color:#2550CC; font-size:12px; font-weight:950; cursor:help; }
        .tooltip-wrap .tooltip-box { visibility:hidden; opacity:0; position:absolute; z-index:9999; left:0; top:25px; min-width:285px; max-width:340px; background:#0B1727; color:white; border-radius:14px; padding:12px 13px; box-shadow:0 18px 45px rgba(7,20,39,.25); transition:opacity .15s ease; font-size:12px; line-height:1.45; font-weight:500; }
        .tooltip-wrap:hover .tooltip-box { visibility:visible; opacity:1; }
        .tooltip-box strong { color:#A7F3D0; }

        /* Responsive */
        @media (max-width: 1200px) {
            .playbook-grid, .story-grid, .routing-grid, .queue-grid, .ux-grid-3, .signal-brief-grid { grid-template-columns: 1fr 1fr; }
            .scenario-metrics { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 820px) {
            .hero-title { font-size: 34px; }
            .playbook-grid, .story-grid, .routing-grid, .queue-grid, .ux-grid-3, .ux-grid-2, .signal-brief-grid { grid-template-columns: 1fr; }
            .scenario-metrics { grid-template-columns: 1fr; }
            .story-row { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-head"><div class="section-title">{html.escape(title)}</div><div class="section-sub">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{html.escape(str(label))}</div><div class="metric-value">{html.escape(str(value))}</div><div class="metric-note">{html.escape(str(note))}</div></div>', unsafe_allow_html=True)


def metric_card_html(label, value, note=""):
    """Compatibility KPI renderer for the existing CSM Book view."""
    st.markdown(
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(str(label))}</div>'
        f'<div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-note">{html.escape(str(note))}</div>'
        '</div>', unsafe_allow_html=True
    )


def risk_level(prob):
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def risk_badge(level):
    css = "risk-high" if level == "High" else "risk-medium" if level == "Medium" else "risk-low"
    return f'<span class="risk-badge {css}">{html.escape(level)} risk</span>'


def render_static_bar_chart(series, title, y_label="", as_percent=False):
    """Render a static Matplotlib bar chart so the dashboard does not zoom/resize on scroll."""
    clean = series.dropna()
    labels = [str(x) for x in clean.index]
    values = clean.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, values, color="#2563EB", width=0.58)

    ax.set_title(title, fontsize=11, fontweight="bold", color="#071427", pad=14)
    ax.set_ylabel(y_label, fontsize=9, color="#475569")
    ax.tick_params(axis="x", labelsize=9, colors="#475569", rotation=0)
    ax.tick_params(axis="y", labelsize=8, colors="#64748B")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    if len(values) > 0:
        ymax = max(values) * 1.25 if max(values) > 0 else 1
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, values):
            label = f"{value:.1%}" if as_percent else f"{int(value):,}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155",
                fontweight="bold",
            )

    if as_percent:
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True, clear_figure=True)


def render_static_horizontal_bar_chart(series, title, x_label="", as_percent=False):
    """Render a static Matplotlib horizontal bar chart so the dashboard remains fixed while scrolling."""
    clean = series.dropna().sort_values(ascending=True)
    labels = [str(x) for x in clean.index]
    values = clean.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(labels, values, color="#2563EB", height=0.56)

    ax.set_title(title, fontsize=11, fontweight="bold", color="#071427", pad=14)
    ax.set_xlabel(x_label, fontsize=9, color="#475569")
    ax.tick_params(axis="x", labelsize=8, colors="#64748B")
    ax.tick_params(axis="y", labelsize=9, colors="#475569")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    if len(values) > 0:
        xmax = max(values) * 1.25 if max(values) > 0 else 1
        ax.set_xlim(0, xmax)
        for bar, value in zip(bars, values):
            label = f"{value:.1%}" if as_percent else f"{int(value):,}"
            ax.text(
                value + xmax * 0.018,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=8,
                color="#334155",
                fontweight="bold",
            )

    if as_percent:
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True, clear_figure=True)


def driver_story_copy(family_name: str, score: float) -> str:
    """Plain-English explanations for portfolio-level risk families."""
    name = str(family_name)
    if name == "Renewal & commercial":
        return "Commercial pressure is the strongest theme: renewal timing, pricing objections, grace period or weak intent are creating the clearest retention risk."
    if name == "Feedback & roadmap":
        return "Customers are signalling unmet product needs. This is a Product/CS alignment topic, not only a support follow-up."
    if name == "Sentiment":
        return "Tone and relationship signals are weakening. CSMs should validate expectations and look for frustration before it becomes renewal risk."
    if name == "Usage & adoption":
        return "Customers are not using enough of the product to clearly realise value. Adoption recovery should be prioritised."
    if name == "Support experience":
        return "Support friction is visible: reopened issues, slow resolution or lower CSAT may be eroding confidence."
    if name == "Bugs & reliability":
        return "Reliability risk exists when bugs, stuck issues or data-trust problems reduce confidence in the platform."
    if name == "Limit friction":
        return "Some customers are close to package limits. This can be healthy usage, but it needs expectation management or upgrade fit review."
    return "This signal family contributes to portfolio risk and should be reviewed together with account-level context."


def plan_family_story_copy(plan_family: str, probability: float) -> str:
    """CSM-friendly explanation for average churn probability by package family."""
    name = str(plan_family)
    if probability >= 0.50:
        level = "high"
    elif probability >= 0.40:
        level = "elevated"
    else:
        level = "contained"

    if "Local" in name:
        action = "Review location activation, frozen locations, listings sync, review workflows and local rank tracking adoption."
    elif "Traffic" in name:
        action = "Validate whether market and competitor insights are being used in recurring business reviews and reporting workflows."
    elif "AI Visibility" in name:
        action = "Run an AI Visibility workflow session: prompts, competitors, AI Visibility Score and recurring AI search review."
    elif "SEO" in name:
        action = "Review Site Audit, Position Tracking, crawl limits, keyword tracking usage, My Reports and advanced value workflows."
    elif "Content" in name:
        action = "Check whether content workflows are connected to measurable outputs: briefs, pages, rankings and reporting."
    elif "Semrush One" in name:
        action = "Use broader suite adoption to reinforce value across SEO, AI, Traffic and reporting workflows."
    else:
        action = "Review whether the package is aligned with the customer’s main business goals and active workflows."

    return f"Risk is {level} for this package family. Recommended CSM focus: {action}"


def render_story_summary_cards(title_1, value_1, body_1, title_2, value_2, body_2, title_3, value_3, body_3):
    st.markdown(
        f"""
        <div class="story-grid">
            <div class="story-card">
                <div class="story-kicker">Primary theme</div>
                <div class="story-headline">{html.escape(str(title_1))}</div>
                <div class="metric-value" style="font-size:28px;">{html.escape(str(value_1))}</div>
                <div class="story-body">{html.escape(str(body_1))}</div>
            </div>
            <div class="story-card">
                <div class="story-kicker">Secondary signal</div>
                <div class="story-headline">{html.escape(str(title_2))}</div>
                <div class="metric-value" style="font-size:28px;">{html.escape(str(value_2))}</div>
                <div class="story-body">{html.escape(str(body_2))}</div>
            </div>
            <div class="story-card">
                <div class="story-kicker">Lowest pressure</div>
                <div class="story-headline">{html.escape(str(title_3))}</div>
                <div class="metric-value" style="font-size:28px;">{html.escape(str(value_3))}</div>
                <div class="story-body">{html.escape(str(body_3))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ranked_story_bars(df, label_col, value_col, value_formatter, max_value=None, hot_threshold=None):
    view = df[[label_col, value_col]].copy().dropna().sort_values(value_col, ascending=False)
    if view.empty:
        st.info("No data available for this section.")
        return

    if max_value is None:
        raw_max = float(view[value_col].max())
        max_value = raw_max if raw_max > 0 else 1.0

    rows = []
    for _, r in view.iterrows():
        label = html.escape(str(r[label_col]))
        val = float(r[value_col])
        width = max(2, min(100, val / max_value * 100))
        css = "hot" if hot_threshold is not None and val >= hot_threshold else "medium" if width >= 60 else ""
        formatted_value = html.escape(str(value_formatter(val)))

        # Keep this HTML as one compact string.
        # If it is indented in a triple-quoted string, Streamlit/Markdown may render it as code.
        rows.append(
            f'<div class="story-row">'
            f'<div class="story-name">{label}</div>'
            f'<div class="story-track"><div class="story-fill {css}" style="width:{width:.1f}%;"></div></div>'
            f'<div class="story-value">{formatted_value}</div>'
            f'</div>'
        )

    html_block = f'<div class="story-bar-list">{"".join(rows)}</div>'
    st.markdown(html_block, unsafe_allow_html=True)


def render_driver_story_section(avg_scores: pd.DataFrame):
    view = avg_scores.copy().sort_values("Average score", ascending=False).reset_index(drop=True)
    if view.empty:
        st.info("No risk driver data available.")
        return
    top = view.iloc[0]
    second = view.iloc[1] if len(view) > 1 else view.iloc[0]
    lowest = view.iloc[-1]

    render_story_summary_cards(
        top["Risk family"], f"{top['Average score']:.1f}/100", driver_story_copy(top["Risk family"], top["Average score"]),
        second["Risk family"], f"{second['Average score']:.1f}/100", driver_story_copy(second["Risk family"], second["Average score"]),
        lowest["Risk family"], f"{lowest['Average score']:.1f}/100", "This is currently the least visible source of risk in the selected portfolio view. Keep monitoring, but prioritise stronger signals first.",
    )
    render_ranked_story_bars(
        view,
        "Risk family",
        "Average score",
        lambda v: f"{v:.1f}/100",
        max_value=max(35.0, float(view["Average score"].max())),
        hot_threshold=25,
    )
    st.markdown(
        f"""
        <div class="story-note">
            <strong>How to read this:</strong> these are not separate churn probabilities. They are explainability scores that show
            where risk is building. In this view, the strongest portfolio story is <strong>{html.escape(str(top['Risk family']))}</strong>.
            CSMs should start by reviewing accounts where this theme appears together with high churn probability or near-term renewal.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_commercial_exposure_story_section(by_family: pd.Series):
    view = by_family.sort_values(ascending=False).reset_index()
    view.columns = ["Plan family", "Average churn probability"]
    if view.empty:
        st.info("No commercial exposure data available.")
        return
    top = view.iloc[0]
    second = view.iloc[1] if len(view) > 1 else view.iloc[0]
    lowest = view.iloc[-1]

    render_story_summary_cards(
        top["Plan family"], f"{top['Average churn probability']:.1%}", plan_family_story_copy(top["Plan family"], top["Average churn probability"]),
        second["Plan family"], f"{second['Average churn probability']:.1%}", plan_family_story_copy(second["Plan family"], second["Average churn probability"]),
        lowest["Plan family"], f"{lowest['Average churn probability']:.1%}", "This package family currently shows the lowest average predicted churn probability in the selected view. Use it as a benchmark for healthy adoption patterns.",
    )
    render_ranked_story_bars(
        view,
        "Plan family",
        "Average churn probability",
        lambda v: f"{v:.1%}",
        max_value=max(0.60, float(view["Average churn probability"].max())),
        hot_threshold=0.50,
    )
    st.markdown(
        f"""
        <div class="story-note">
            <strong>CSM takeaway:</strong> package-level risk should not be read as “this product is bad”. It tells us where to inspect the customer journey first.
            For <strong>{html.escape(str(top['Plan family']))}</strong>, the next step is to connect the package-specific usage pattern with account-level drivers:
            adoption, limits, feedback, bugs, sentiment and renewal timing.
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    package = joblib.load(path)
    if isinstance(package, dict):
        return package["model"], package["columns"], package
    return package, None, {}


def prepare_features(df, model_columns=None):
    X = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns], errors="ignore").copy()
    X = pd.get_dummies(X, drop_first=True)
    if model_columns is not None:
        X = X.reindex(columns=model_columns, fill_value=0)
    return X


def predict_for_df(model, model_columns, df):
    X = prepare_features(df, model_columns)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def yesno(v):
    return "Yes" if int(v) == 1 else "No"


def family_risk_scores(row):
    scores = {}
    scores["Usage & adoption"] = np.clip(
        0.32 * max(0, 65 - row.get("tool_activity_score", 0)) +
        0.28 * max(0, 60 - row.get("feature_adoption_score", 0)) +
        0.25 * max(0, -row.get("usage_change_vs_prev_quarter", 0)) +
        0.20 * max(0, row.get("days_since_last_login", 0) - 14), 0, 100)
    scores["Renewal & commercial"] = np.clip(
        30 * (row.get("renewal_due_days", 999) < 45) +
        30 * (row.get("renewal_intent", "positive") == "negative") +
        16 * (row.get("renewal_intent", "positive") == "neutral") +
        22 * row.get("pricing_objection_flag", 0) +
        26 * row.get("downgrade_risk_flag", 0) +
        20 * row.get("in_grace_period", 0), 0, 100)
    scores["Support experience"] = np.clip(
        9 * row.get("reopened_tickets", 0) +
        3 * row.get("avg_resolution_days", 0) +
        25 * max(0, 3.5 - row.get("csat_support", 5)), 0, 100)
    scores["Feedback & roadmap"] = np.clip(
        10 * row.get("critical_feedback_count_90d", 0) +
        7 * row.get("unresolved_feedback_count", 0) +
        0.35 * row.get("avg_unresolved_feedback_age_days", 0) +
        0.55 * max(0, 55 - row.get("roadmap_alignment_score", 100)) +
        24 * row.get("feedback_linked_to_renewal", 0) +
        20 * row.get("competitor_mentioned_in_feedback", 0), 0, 100)
    scores["Bugs & reliability"] = np.clip(
        14 * row.get("critical_bugs_reported_90d", 0) +
        18 * row.get("stuck_bugs_count", 0) +
        0.45 * row.get("oldest_open_bug_days", 0) +
        28 * row.get("data_accuracy_bug_flag", 0) +
        24 * row.get("bug_linked_to_renewal_flag", 0) -
        10 * row.get("workaround_available_flag", 0), 0, 100)
    scores["Sentiment"] = np.clip(
        40 * max(0, -row.get("sentiment_csm", 0)) +
        35 * max(0, -row.get("sentiment_support", 0)) +
        30 * max(0, -row.get("feedback_sentiment_score", 0)) +
        25 * max(0, -row.get("bug_sentiment_score", 0)), 0, 100)
    scores["Limit friction"] = np.clip(
        0.75 * max(0, row.get("crawl_budget_usage_pct", 0) - 80) +
        0.75 * max(0, row.get("keyword_tracking_usage_pct", 0) - 80) +
        0.80 * max(0, row.get("ai_visibility_prompt_usage_pct", 0) - 80) +
        18 * row.get("limit_upgrade_recommended_flag", 0), 0, 100)
    return {k: round(float(v), 1) for k, v in scores.items()}


def top_drivers(row, limit=6):
    drivers = []
    if row.get("tool_activity_score", 100) < 40: drivers.append("Low product activity")
    if row.get("feature_adoption_score", 100) < 45: drivers.append("Weak feature adoption")
    if row.get("usage_change_vs_prev_quarter", 0) < -15: drivers.append("Usage decline")
    if row.get("crawl_budget_usage_pct", 0) > 85: drivers.append("Crawl budget saturation")
    if row.get("keyword_tracking_usage_pct", 0) > 85: drivers.append("Keyword tracking limit pressure")
    if row.get("ai_visibility_active", 0) and row.get("ai_visibility_prompts_tracked", 0) < 10: drivers.append("AI Visibility under-adoption")
    if row.get("roadmap_alignment_score", 100) < 40: drivers.append("Roadmap mismatch")
    if row.get("feedback_linked_to_renewal", 0): drivers.append("Feedback linked to renewal")
    if row.get("critical_feedback_count_90d", 0) > 0: drivers.append("Critical unresolved product feedback")
    if row.get("data_accuracy_bug_flag", 0): drivers.append("Data accuracy trust issue")
    if row.get("stuck_bugs_count", 0) > 0: drivers.append("Stuck bugs")
    if row.get("workaround_available_flag", 1) == 0: drivers.append("No workaround available")
    if row.get("renewal_due_days", 999) < 45 and row.get("renewal_intent", "positive") != "positive": drivers.append("Near-term renewal risk")
    if row.get("sentiment_csm", 0) < -0.2: drivers.append("Negative CSM sentiment")
    return drivers[:limit]



def _is_flag_on(row, key):
    try:
        return int(row.get(key, 0)) == 1
    except Exception:
        return False


def _pct(row, key, default=0):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def build_department_routing(row):
    """Return cross-functional teams that should be involved for this account.

    The goal is CSM usability: explain who should help, why they are needed,
    and what the handoff should ask for.
    """
    routing = []

    support_reasons = []
    support_actions = []

    if _is_flag_on(row, "data_accuracy_bug_flag"):
        support_reasons.append("data accuracy discrepancy")
        support_actions.append("validate the reported discrepancy, confirm expected behaviour vs defect, and prepare a customer-facing explanation")

    if _is_flag_on(row, "site_audit_bug_flag"):
        support_reasons.append("Site Audit issue")
        support_actions.append("review crawl settings, crawl limits, excluded pages, audit frequency and expected behaviour")

    if _is_flag_on(row, "position_tracking_bug_flag"):
        support_reasons.append("Position Tracking issue")
        support_actions.append("check campaign setup, keywords, location, device, search engine and tracking configuration")

    if _is_flag_on(row, "ai_visibility_bug_flag"):
        support_reasons.append("AI Visibility issue")
        support_actions.append("review prompt setup, tracked competitors, prompt limits and reporting interpretation")

    if _is_flag_on(row, "local_listing_sync_bug_flag"):
        support_reasons.append("Local listing sync issue")
        support_actions.append("validate Listing Management sync status, frozen locations, duplicate suppression and location-level setup")

    if _pct(row, "crawl_budget_usage_pct") > 85:
        support_reasons.append("crawl budget limit friction")
        support_actions.append("clarify monthly/per-audit crawl limits and help the customer verify current usage")

    if _pct(row, "keyword_tracking_usage_pct") > 85:
        support_reasons.append("keyword tracking limit friction")
        support_actions.append("clarify keyword tracking limits, campaign allocation and user/team allocation where relevant")

    if _pct(row, "ai_visibility_prompt_usage_pct") > 85:
        support_reasons.append("AI Visibility prompt limit friction")
        support_actions.append("clarify prompt tracking limits, current prompt usage and whether the customer needs a package review")

    if _pct(row, "tool_limit_feedback_count") >= 1:
        support_reasons.append("tool limit misunderstanding or friction")
        support_actions.append("check whether the customer is facing a product limit, user allocation issue or configuration issue")

    if _pct(row, "technical_tickets_last_quarter") >= 4:
        support_reasons.append("high technical support volume")
        support_actions.append("review repeated technical topics and provide a clear navigation or troubleshooting summary")

    hard_block = _pct(row, "stuck_bugs_count") >= 1 and _is_flag_on(row, "workaround_available_flag") is False
    if hard_block:
        support_reasons.append("hard block without workaround")
        support_actions.append("treat as blocked workflow, confirm severity, document customer impact and define update cadence")

    if support_reasons:
        routing.append({
            "team": "Customer Support",
            "type": "support",
            "why": ", ".join(support_reasons[:4]),
            "action": "; ".join(support_actions[:3]),
            "urgency": "High" if hard_block or _is_flag_on(row, "data_accuracy_bug_flag") else "Medium",
        })

    ae_reasons = []
    ae_actions = []

    if _is_flag_on(row, "pricing_objection_flag"):
        ae_reasons.append("pricing objection")
        ae_actions.append("review pricing concern, commercial options and value narrative")

    if _is_flag_on(row, "limit_upgrade_recommended_flag"):
        ae_reasons.append("plan or limit upgrade may be needed")
        ae_actions.append("assess whether current package limits fit actual usage needs")

    if _is_flag_on(row, "downgrade_risk_flag"):
        ae_reasons.append("downgrade risk")
        ae_actions.append("prepare downgrade prevention or right-sizing conversation")

    if _is_flag_on(row, "expansion_opportunity_flag") and row.get("risk_level", "") != "High":
        ae_reasons.append("possible expansion opportunity")
        ae_actions.append("review whether additional toolkit, limits or seats would unlock more value")

    if _pct(row, "renewal_due_days", 999) < 45 and str(row.get("renewal_intent", "positive")) in ["neutral", "negative"]:
        ae_reasons.append("commercially sensitive renewal")
        ae_actions.append("confirm contract terms, renewal options, package scope, purchased add-ons and commercial commitments before customer follow-up")

    if _pct(row, "billing_tickets_last_quarter") >= 2:
        ae_reasons.append("billing or contract questions")
        ae_actions.append("validate contract details, invoice-related questions and purchased package scope")

    if ae_reasons:
        routing.append({
            "team": "Account Executive / Sales",
            "type": "sales",
            "why": ", ".join(ae_reasons[:4]),
            "action": "; ".join(ae_actions[:3]),
            "urgency": "High" if _pct(row, "renewal_due_days", 999) < 30 and str(row.get("renewal_intent", "positive")) != "positive" else "Medium",
        })

    product_reasons = []
    product_actions = []

    if _pct(row, "critical_feedback_count_90d") >= 1 and _pct(row, "roadmap_alignment_score", 100) < 45:
        product_reasons.append("critical product feedback with low roadmap alignment")
        product_actions.append("confirm roadmap status, alternative workflow or workaround")

    if _is_flag_on(row, "data_accuracy_bug_flag") and _pct(row, "stuck_bugs_count") >= 1:
        product_reasons.append("stuck data accuracy bug")
        product_actions.append("confirm investigation status, customer impact and expected update timeline")

    if _is_flag_on(row, "feedback_linked_to_renewal"):
        product_reasons.append("product feedback linked to renewal")
        product_actions.append("provide roadmap or workaround context before renewal conversation")

    if _pct(row, "oldest_open_bug_days") > 45 and _pct(row, "open_bugs_count") > 0:
        product_reasons.append("old unresolved bug")
        product_actions.append("review bug status and decide whether an escalation path or workaround is needed")

    if product_reasons:
        routing.append({
            "team": "Product / Engineering",
            "type": "product",
            "why": ", ".join(product_reasons[:4]),
            "action": "; ".join(product_actions[:3]),
            "urgency": "High" if _is_flag_on(row, "feedback_linked_to_renewal") or _is_flag_on(row, "data_accuracy_bug_flag") else "Medium",
        })

    if not routing:
        routing.append({
            "team": "CSM only",
            "type": "csm",
            "why": "no cross-functional dependency detected",
            "action": "continue proactive monitoring, value reinforcement and adoption coaching",
            "urgency": "Low",
        })

    return routing


def department_routing_html(row):
    routing = build_department_routing(row)
    cards = []
    for item in routing:
        urgency = str(item.get("urgency", "Medium"))
        urgency_class = "urgency-high" if urgency == "High" else "urgency-medium" if urgency == "Medium" else "urgency-low"
        card_type = html.escape(str(item.get("type", "csm")))
        cards.append(
            f'<div class="routing-card {card_type}">'
            f'<div class="urgency-pill {urgency_class}">{html.escape(urgency)} urgency</div>'
            f'<div class="routing-team">{html.escape(str(item.get("team", "")))}</div>'
            f'<div class="routing-kicker">Why involve them</div>'
            f'<div class="routing-body">{html.escape(str(item.get("why", "")))}</div>'
            f'<div class="routing-kicker">CSM handoff ask</div>'
            f'<div class="routing-body">{html.escape(str(item.get("action", "")))}</div>'
            f'</div>'
        )
    return '<div class="routing-grid">' + ''.join(cards) + '</div>'


def build_handoff_message(row):
    routing = build_department_routing(row)
    account = row.get("account_name", row.get("account_id", "this account"))
    risk = f"{row.get('churn_probability', 0):.1%}" if "churn_probability" in row else "elevated"
    drivers = ", ".join(top_drivers(row, 5)) or "general churn risk"
    lines = [
        f"Account: {account} ({row.get('account_id', 'N/A')})",
        f"Context: predicted churn risk is {risk}. Main visible drivers: {drivers}.",
        "",
        "Requested support:",
    ]
    for item in routing:
        if item.get("team") == "CSM only":
            continue
        lines.append(f"- {item['team']}: {item['action']}")
    if len(lines) == 4:
        lines.append("- CSM: continue proactive monitoring and value reinforcement; no cross-functional dependency detected yet.")
    lines.append("")
    lines.append("Goal: give the customer a clear next step, reduce friction before the next touchpoint, and protect renewal confidence.")
    return "\n".join(lines)


def route_filter_matches(row, routing_filter):
    if routing_filter == "All":
        return True
    teams = {item.get("team") for item in build_department_routing(row)}
    if routing_filter == "Needs Customer Support":
        return "Customer Support" in teams
    if routing_filter == "Needs Account Executive / Sales":
        return "Account Executive / Sales" in teams
    if routing_filter == "Needs Product / Engineering":
        return "Product / Engineering" in teams
    if routing_filter == "CSM only":
        return teams == {"CSM only"}
    return True


def routing_summary_counts(df):
    counts = {
        "Customer Support": 0,
        "Account Executive / Sales": 0,
        "Product / Engineering": 0,
        "CSM only": 0,
    }
    for _, r in df.iterrows():
        teams = {item.get("team") for item in build_department_routing(r)}
        for k in counts:
            if k in teams:
                counts[k] += 1
    return counts


def routing_summary_html(df):
    counts = routing_summary_counts(df)
    total = max(len(df), 1)
    def card(label, value, note):
        pct = value / total
        return (
            f'<div class="queue-card">'
            f'<div class="queue-label">{html.escape(label)}</div>'
            f'<div class="queue-value">{value}</div>'
            f'<div class="queue-note">{pct:.1%} of current portfolio · {html.escape(note)}</div>'
            f'</div>'
        )
    return '<div class="queue-grid">' + ''.join([
        card("Needs Support", counts["Customer Support"], "tool issues, limits, navigation or hard blocks"),
        card("Needs AE / Sales", counts["Account Executive / Sales"], "pricing, package fit or contract questions"),
        card("Needs Product", counts["Product / Engineering"], "roadmap, stuck bugs or product gaps"),
        card("CSM only", counts["CSM only"], "no cross-functional dependency detected"),
    ]) + '</div>'


def semrush_csm_cta(row):
    actions = []
    owners = []

    if row.get("data_accuracy_bug_flag", 0) and row.get("stuck_bugs_count", 0) >= 1:
        actions.append("Escalate as a data-trust risk, not a regular bug. Confirm data impact, request Product/Support ETA, document workaround and set a customer update cadence.")
        owners += ["Support Lead", "Product Owner", "CSM"]
    if row.get("crawl_budget_usage_pct", 0) > 85 and row.get("site_audit_projects_active", 0) > 0:
        actions.append("Run a Site Audit limit-fit review: confirm crawl scope, audit frequency, pages-per-audit setup, excluded pages and whether the current plan limits fit website size.")
        owners += ["CSM", "Technical Support", "Account Owner"]
    if row.get("keyword_tracking_usage_pct", 0) > 85:
        actions.append("Review Position Tracking setup: keyword allocation, inactive campaigns, devices/locations, and whether additional tracking capacity or a higher tier is justified.")
        owners += ["CSM", "SEO Specialist", "Account Owner"]
    if row.get("ai_visibility_active", 0) and row.get("ai_visibility_prompts_tracked", 0) < 10:
        actions.append("Book an AI Visibility workflow session. Help the customer define priority prompts, select competitors, review AI Visibility Score and schedule a recurring AI search review.")
        owners += ["CSM", "AI Visibility Specialist"]
    if row.get("local_toolkit_active", 0) and (row.get("frozen_locations_count", 0) > 0 or row.get("duplicate_suppression_issues", 0) > 2):
        actions.append("Run a Local Toolkit cleanup session: review frozen locations, duplicate suppression, listing sync status, review workflows and location ownership.")
        owners += ["CSM", "Local Specialist", "Support"]
    if row.get("feedback_linked_to_renewal", 0) and row.get("roadmap_alignment_score", 100) < 45:
        actions.append("Treat the product gap as a renewal dependency. Align with Product on roadmap status, prepare expectation management and capture workaround options before the renewal conversation.")
        owners += ["CSM", "Product Owner", "Account Owner"]
    if row.get("plan_tier", "") in ["Guru", "Business", "Pro+", "Advanced"] and row.get("tool_activity_score", 100) < 45:
        actions.append("Run an adoption recovery session focused on paid-but-underused workflows: Site Audit, Position Tracking, competitor analysis, My Reports, AI Visibility or API/Share of Voice depending on entitlement.")
        owners += ["CSM", "Account Owner"]
    if row.get("api_access_available", 0) and not row.get("api_access_used", 0):
        actions.append("For this advanced-plan account, validate whether API access could automate reporting or reduce manual work. Offer an API/value discovery conversation.")
        owners += ["CSM", "Solutions Consultant"]
    if row.get("renewal_due_days", 999) < 30 and row.get("renewal_intent", "positive") != "positive":
        actions.append("Trigger renewal-risk motion: confirm decision criteria, agree recovery actions, involve commercial owner and set a next-step date before renewal.")
        owners += ["CSM", "Account Owner", "Renewals"]
    if not actions:
        actions.append("Maintain proactive value reinforcement. Review adoption goals, confirm active workflows, and identify one expansion or enablement opportunity for the next customer touchpoint.")
        owners += ["CSM"]

    # de-duplicate owners while preserving order
    seen = set(); owners_unique=[]
    for o in owners:
        if o not in seen:
            seen.add(o); owners_unique.append(o)

    urgency = "High" if row["churn_probability"] >= 0.70 or row.get("renewal_due_days", 999) < 30 else "Medium" if row["churn_probability"] >= 0.40 else "Low"
    return actions[:3], owners_unique[:5], urgency


def playbook_html(row):
    actions, owners, urgency = semrush_csm_cta(row)
    driver_text = ", ".join(top_drivers(row, 4)) or "No major risk concentration"
    cards = [
        ("Risk reason", driver_text, "What is creating the strongest visible risk for this account."),
        ("Business impact", f"MRR ${row.get('mrr',0):,.0f} · Renewal in {int(row.get('renewal_due_days',0))} days", "Use this to decide whether to escalate or monitor."),
        ("Recommended CSM action", actions[0], "Primary operational next step."),
        ("Owners to involve", ", ".join(owners), "Recommended cross-functional support."),
        ("Urgency", urgency, "Based on churn probability, renewal timing and severity of risk signals."),
    ]
    html_cards = []
    for label, title, body in cards:
        html_cards.append(f'<div class="playbook-card"><div class="playbook-label">{html.escape(label)}</div><div class="playbook-title">{html.escape(str(title))}</div><div class="playbook-body">{html.escape(str(body))}</div></div>')
    if len(actions) > 1:
        html_cards.append(f'<div class="playbook-card"><div class="playbook-label">Secondary action</div><div class="playbook-title">{html.escape(actions[1])}</div><div class="playbook-body">Use this if the primary action is already in motion.</div></div>')
    return '<div class="playbook-grid">' + ''.join(html_cards) + '</div>'


def bar(label, value):
    warn = "warn" if value >= 65 else ""
    return f'<div class="small-muted"><strong>{html.escape(label)}</strong> · {value:.1f}/100</div><div class="bar-wrap"><div class="bar-fill {warn}" style="width:{min(100, max(0, value)):.1f}%"></div></div>'


def simulate_scenarios(row, model, model_columns):
    base = pd.DataFrame([row.drop(labels=[c for c in ["churn_probability", "risk_level", "priority"] if c in row.index])])
    current = float(row["churn_probability"])
    scenarios = []

    def add(name, description, changes):
        simulated = base.copy()
        for k, v in changes.items():
            if k in simulated.columns:
                simulated.loc[simulated.index[0], k] = v(simulated.loc[simulated.index[0], k]) if callable(v) else v
        new_prob = float(predict_for_df(model, model_columns, simulated)[0])
        scenarios.append({
            "Scenario": name,
            "What changes": description,
            "Current probability": current,
            "Simulated probability": new_prob,
            "Change": new_prob - current,
            "New risk level": risk_level(new_prob),
        })

    add("Improve Site Audit & usage", "Increase activity/adoption and reduce inactivity", {
        "tool_activity_score": lambda x: min(88, max(x, 72)),
        "feature_adoption_score": lambda x: min(85, max(x, 70)),
        "days_since_last_login": lambda x: min(x, 5),
        "site_audit_last_run_days": lambda x: min(x, 7),
    })
    add("Resolve limit friction", "Reduce crawl/keyword/prompt saturation and mark upgrade-fit action", {
        "crawl_budget_usage_pct": lambda x: min(x, 72),
        "keyword_tracking_usage_pct": lambda x: min(x, 72),
        "ai_visibility_prompt_usage_pct": lambda x: min(x, 72),
        "limit_upgrade_recommended_flag": 0,
    })
    add("Strengthen renewal signal", "Move renewal intent to positive and remove grace/pricing/downgrade risk", {
        "renewal_intent": "positive",
        "in_grace_period": 0,
        "pricing_objection_flag": 0,
        "downgrade_risk_flag": 0,
    })
    add("Resolve product feedback", "Close critical feedback and improve roadmap alignment/sentiment", {
        "critical_feedback_count_90d": 0,
        "unresolved_feedback_count": lambda x: max(0, min(x, 1)),
        "avg_unresolved_feedback_age_days": lambda x: min(x, 14),
        "roadmap_alignment_score": lambda x: max(x, 75),
        "feedback_sentiment_score": lambda x: max(x, 0.35),
        "feedback_linked_to_renewal": 0,
    })
    add("Resolve bug reliability issues", "Remove stuck/data-trust bugs and set workaround available", {
        "critical_bugs_reported_90d": 0,
        "stuck_bugs_count": 0,
        "open_bugs_count": lambda x: max(0, min(x, 1)),
        "oldest_open_bug_days": lambda x: min(x, 7),
        "data_accuracy_bug_flag": 0,
        "workaround_available_flag": 1,
        "bug_linked_to_renewal_flag": 0,
        "bug_sentiment_score": lambda x: max(x, 0.35),
    })
    add("Best-case recovery package", "Combine adoption, renewal, feedback and bug recovery", {
        "tool_activity_score": lambda x: min(90, max(x, 78)),
        "feature_adoption_score": lambda x: min(90, max(x, 76)),
        "days_since_last_login": lambda x: min(x, 4),
        "renewal_intent": "positive",
        "in_grace_period": 0,
        "pricing_objection_flag": 0,
        "unresolved_feedback_count": 0,
        "critical_feedback_count_90d": 0,
        "roadmap_alignment_score": lambda x: max(x, 80),
        "stuck_bugs_count": 0,
        "data_accuracy_bug_flag": 0,
        "workaround_available_flag": 1,
        "bug_linked_to_renewal_flag": 0,
    })
    out = pd.DataFrame(scenarios)
    out["Current probability"] = out["Current probability"].map(lambda x: f"{x:.1%}")
    out["Simulated probability"] = out["Simulated probability"].map(lambda x: f"{x:.1%}")
    out["Change"] = out["Change"].map(lambda x: f"{x*100:+.1f} pp")
    return out



def _impact_level(delta_pp: float) -> str:
    improvement = abs(float(delta_pp))
    if improvement >= 15:
        return "high"
    if improvement >= 6:
        return "medium"
    return "low"


def _owner_class(owner: str) -> str:
    o = owner.lower()
    if "support" in o:
        return "support"
    if "sales" in o or "account executive" in o or "ae" in o:
        return "sales"
    if "product" in o or "engineering" in o:
        return "product"
    return "csm"


def _tooltip(title: str, what: str, problem: str, action: str, owner: str) -> str:
    content = (
        f"<strong>{html.escape(title)}</strong><br>"
        f"<strong>What it means:</strong> {html.escape(what)}<br>"
        f"<strong>When it is a problem:</strong> {html.escape(problem)}<br>"
        f"<strong>Suggested action:</strong> {html.escape(action)}<br>"
        f"<strong>Who may help:</strong> {html.escape(owner)}"
    )
    return f'<span class="tooltip-wrap">? <span class="tooltip-box">{content}</span></span>'


def _status_from_threshold(value, risk_threshold, watch_threshold=None, inverse=False):
    """Return visual status for numeric indicators.

    inverse=False: higher is worse. inverse=True: lower is worse.
    """
    try:
        v = float(value)
    except Exception:
        return "watch", "Review"
    if watch_threshold is None:
        watch_threshold = risk_threshold * 0.75
    if inverse:
        if v <= risk_threshold:
            return "risk", "Risk"
        if v <= watch_threshold:
            return "watch", "Watch"
        return "good", "Healthy"
    if v >= risk_threshold:
        return "risk", "Risk"
    if v >= watch_threshold:
        return "watch", "Watch"
    return "good", "Healthy"


def _value_pct(row, key):
    try:
        return max(0.0, min(100.0, float(row.get(key, 0))))
    except Exception:
        return 0.0


def _render_signal_card(title, value, status_class, status_label, progress_pct, what, problem, action, owner, note=""):
    owner_html = "".join([f'<span class="ux-owner {_owner_class(x)}">{html.escape(x)}</span>' for x in owner.split(" + ")])
    tooltip = _tooltip(title, what, problem, action, owner)
    progress = max(0, min(100, float(progress_pct)))
    body_note = html.escape(note) if note else html.escape(what)
    return (
        f'<div class="signal-brief-card {status_class}">'
        f'<div class="signal-name">{html.escape(title)} {tooltip}</div>'
        f'<div class="signal-status {status_class}">{html.escape(status_label)}</div>'
        f'<div class="signal-value-big">{html.escape(str(value))}</div>'
        f'<div class="signal-progress"><div class="signal-progress-fill {status_class}" style="width:{progress:.1f}%;"></div></div>'
        f'<div class="ux-body">{body_note}</div>'
        f'<div class="signal-cta"><strong>Suggested action:</strong> {html.escape(action)}<br>{owner_html}</div>'
        f'</div>'
    )


def render_signal_briefing(row):
    """Human-readable Semrush signal briefing for non-technical CSM users."""
    usage_cards = []

    # Site Audit activity: lower last-run days is better; 999 means not active.
    last_run = float(row.get("site_audit_last_run_days", 999))
    if last_run > 60:
        st_cls, st_label, progress = "risk", "Inactive", 15
    elif last_run > 21:
        st_cls, st_label, progress = "watch", "Needs review", 55
    else:
        st_cls, st_label, progress = "good", "Active", 90
    usage_cards.append(_render_signal_card(
        "Site Audit activity",
        f"{int(last_run)} days ago" if last_run < 900 else "Not recently used",
        st_cls, st_label, progress,
        "Shows when the customer last ran Site Audit. Regular audits help the customer see technical SEO value.",
        "If Site Audit has not run recently, the customer may not be getting ongoing technical SEO insights.",
        "Ask whether Site Audit is part of their workflow. If not, run a setup/value session around audit scope, frequency and reporting.",
        "CSM + Customer Support",
        "Checks whether technical SEO monitoring is actually part of the customer workflow."
    ))

    pt_campaigns = int(row.get("position_tracking_campaigns_active", 0))
    if pt_campaigns == 0:
        st_cls, st_label, progress = "risk", "Not active", 10
    elif pt_campaigns <= 1:
        st_cls, st_label, progress = "watch", "Limited", 50
    else:
        st_cls, st_label, progress = "good", "Active", 85
    usage_cards.append(_render_signal_card(
        "Position Tracking setup",
        f"{pt_campaigns} campaigns",
        st_cls, st_label, progress,
        "Shows whether the customer is tracking rankings through active campaigns.",
        "No or very few campaigns can mean the customer is not monitoring the keyword outcomes they care about.",
        "Review campaign setup: location, device, search engine, competitors and keyword list. Confirm it matches their business goals.",
        "CSM + Customer Support",
        "Shows whether ranking monitoring is configured and connected to customer goals."
    ))

    ai_active = int(row.get("ai_visibility_active", 0)) == 1
    ai_prompts = int(row.get("ai_visibility_prompts_tracked", 0))
    ai_limit = max(1, int(row.get("ai_visibility_prompt_limit", 0)))
    ai_pct = ai_prompts / ai_limit * 100 if ai_active else 0
    if not ai_active:
        st_cls, st_label, progress = "good", "Not purchased", 0
        value = "Not active"
        action = "No action unless AI Visibility is part of the customer goals. If relevant, discuss fit with the commercial owner."
        owner = "CSM + Account Executive / Sales"
    elif ai_prompts < 10:
        st_cls, st_label, progress = "risk", "Underused", max(8, ai_pct)
        value = f"{ai_prompts}/{ai_limit} prompts"
        action = "Book an AI Visibility workflow session: define priority prompts, select competitors and set a recurring AI search review."
        owner = "CSM"
    else:
        st_cls, st_label, progress = "good", "In use", min(100, ai_pct)
        value = f"{ai_prompts}/{ai_limit} prompts"
        action = "Use current prompt tracking as a value proof point in the next business review."
        owner = "CSM"
    usage_cards.append(_render_signal_card(
        "AI Visibility adoption",
        value,
        st_cls, st_label, progress,
        "Shows whether the customer is using prompt tracking and AI search visibility workflows.",
        "If AI Visibility is active but prompt tracking is underused, the customer may not see enough value from the toolkit.",
        action,
        owner,
        "Checks whether AI Visibility is becoming a real workflow or only a purchased feature."
    ))

    local_active = int(row.get("local_locations_active", 0))
    local_target = max(0, int(row.get("local_locations_target", 0)))
    if local_target == 0:
        st_cls, st_label, progress = "good", "Not applicable", 0
        value = "Not a local setup"
    else:
        local_pct = local_active / max(1, local_target) * 100
        if local_pct < 50:
            st_cls, st_label, progress = "risk", "Low activation", local_pct
        elif local_pct < 80:
            st_cls, st_label, progress = "watch", "Partial", local_pct
        else:
            st_cls, st_label, progress = "good", "Healthy", local_pct
        value = f"{local_active}/{local_target} locations"
    usage_cards.append(_render_signal_card(
        "Local locations activation",
        value,
        st_cls, st_label, progress,
        "Shows whether purchased/target local locations are active and being managed.",
        "Low activation can mean the customer is paying for local workflows that are not fully live.",
        "Review inactive or frozen locations, confirm Listing Management setup and activate Review Management workflows.",
        "CSM + Customer Support",
        "Highlights whether local value is actually activated, not just contracted."
    ))

    limit_cards = []
    crawl_pct = _value_pct(row, "crawl_budget_usage_pct")
    cls, label = _status_from_threshold(crawl_pct, 85, 65)
    limit_cards.append(_render_signal_card(
        "Crawl budget usage",
        f"{crawl_pct:.1f}%",
        cls, label, crawl_pct,
        "Shows how much of the account's available Site Audit crawl capacity is being used.",
        "High usage can be healthy adoption, but it may also create limit frustration if the customer cannot audit enough pages.",
        "Review Site Audit scope, excluded pages, audit frequency and whether the current plan limit fits website size.",
        "Customer Support + Account Executive / Sales",
        "Useful to distinguish healthy usage from plan-limit friction."
    ))

    kw_pct = _value_pct(row, "keyword_tracking_usage_pct")
    cls, label = _status_from_threshold(kw_pct, 85, 65)
    limit_cards.append(_render_signal_card(
        "Keyword tracking usage",
        f"{kw_pct:.1f}%",
        cls, label, kw_pct,
        "Shows how much of the account's Position Tracking keyword capacity is being used.",
        "High usage may indicate that the customer is constrained by limits or needs help prioritising campaigns/keywords.",
        "Review keyword allocation, inactive campaigns, locations/devices and whether the current tier fits tracking needs.",
        "Customer Support + Account Executive / Sales",
        "Helps detect when the customer is blocked by tracking capacity rather than lack of interest."
    ))

    prompt_pct = _value_pct(row, "ai_visibility_prompt_usage_pct")
    cls, label = _status_from_threshold(prompt_pct, 85, 65)
    if not ai_active:
        cls, label, prompt_pct = "good", "Not applicable", 0
    limit_cards.append(_render_signal_card(
        "AI prompt limit usage",
        f"{prompt_pct:.1f}%" if ai_active else "Not active",
        cls, label, prompt_pct,
        "Shows how much of the AI Visibility prompt tracking allowance is being used.",
        "Near-limit usage can create confusion if the customer expects to track more prompts than contracted.",
        "Clarify prompt limits, current usage and whether the customer needs a workflow redesign or package review.",
        "Customer Support + Account Executive / Sales",
        "Important for explaining AI Visibility limits in plain customer terms."
    ))

    api_pct = _value_pct(row, "api_units_usage_pct") if "api_units_usage_pct" in row.index else 0
    api_available = int(row.get("api_access_available", 0)) == 1
    api_used = int(row.get("api_access_used", 0)) == 1
    if not api_available:
        cls, label, progress, value = "good", "Not included", 0, "Not included"
        api_action = "No action unless API is part of the customer requirements. If needed, involve Sales to review package fit."
        api_owner = "CSM + Account Executive / Sales"
    elif not api_used:
        cls, label, progress, value = "watch", "Available but unused", 35, "Not used"
        api_action = "Explore whether API access could automate reporting or reduce manual work for this account."
        api_owner = "CSM + Solutions Consultant"
    else:
        cls, label = _status_from_threshold(api_pct, 85, 65)
        progress, value = api_pct, f"{api_pct:.1f}%"
        api_action = "If usage is high, help the customer understand API unit consumption and involve Sales if additional capacity is needed."
        api_owner = "Customer Support + Account Executive / Sales"
    limit_cards.append(_render_signal_card(
        "API access / units",
        value,
        cls, label, progress,
        "Shows whether API access is available and whether usage may be creating capacity pressure.",
        "Unused API access can be missed value; saturated API usage can become a limit or pricing conversation.",
        api_action,
        api_owner,
        "Useful for advanced accounts where automation and reporting workflows matter."
    ))

    trust_cards = []
    roadmap = float(row.get("roadmap_alignment_score", 100))
    cls, label = _status_from_threshold(roadmap, 40, 60, inverse=True)
    trust_cards.append(_render_signal_card(
        "Product fit / roadmap alignment",
        f"{roadmap:.1f}/100",
        cls, label, roadmap,
        "Shows whether the customer's needs appear aligned with the product direction or roadmap.",
        "Low alignment means the customer may feel the product is not evolving in the direction they need.",
        "Clarify workarounds, manage expectations and involve Product if the feedback is renewal-critical.",
        "CSM + Product / Engineering",
        "Shows whether churn risk is coming from product gap, not just service experience."
    ))

    feedback = int(row.get("unresolved_feedback_count", 0))
    critical_feedback = int(row.get("critical_feedback_count_90d", 0))
    if critical_feedback > 0 or feedback >= 3:
        cls, label, progress = "risk", "Open friction", min(100, 25 + feedback * 12 + critical_feedback * 20)
    elif feedback > 0:
        cls, label, progress = "watch", "Some friction", min(100, 25 + feedback * 12)
    else:
        cls, label, progress = "good", "Clear", 5
    trust_cards.append(_render_signal_card(
        "Product feedback pressure",
        f"{feedback} unresolved · {critical_feedback} critical",
        cls, label, progress,
        "Shows whether the customer has unresolved requests, feature gaps or recalculation needs.",
        "Repeated or critical feedback can become renewal risk if there is no workaround or roadmap clarity.",
        "Separate support issue from product gap. Confirm workaround, roadmap status and renewal dependency.",
        "CSM + Product / Engineering",
        "Keeps product gaps visible without overloading the CSM with raw feedback records."
    ))

    open_bugs = int(row.get("open_bugs_count", 0))
    stuck_bugs = int(row.get("stuck_bugs_count", 0))
    data_bug = int(row.get("data_accuracy_bug_flag", 0)) == 1
    if data_bug or stuck_bugs > 0:
        cls, label, progress = "risk", "Trust risk", min(100, 45 + stuck_bugs * 25 + (25 if data_bug else 0))
    elif open_bugs > 0:
        cls, label, progress = "watch", "Needs follow-up", min(100, 25 + open_bugs * 12)
    else:
        cls, label, progress = "good", "No visible blocker", 5
    trust_cards.append(_render_signal_card(
        "Bugs / data trust",
        f"{open_bugs} open · {stuck_bugs} stuck",
        cls, label, progress,
        "Shows whether open or stuck bugs are affecting product reliability or data confidence.",
        "Data accuracy issues can damage trust faster than normal support friction, especially near renewal.",
        "Escalate as a trust-risk issue. Confirm impact, ETA, workaround and customer update cadence.",
        "Customer Support + Product / Engineering",
        "Highlights whether the customer can trust the product enough to stay."
    ))

    workaround = int(row.get("workaround_available_flag", 1)) == 1
    if workaround:
        cls, label, progress, value = "good", "Available", 85, "Yes"
    elif stuck_bugs > 0 or data_bug:
        cls, label, progress, value = "risk", "Missing", 15, "No"
    else:
        cls, label, progress, value = "watch", "Check", 50, "Unknown/No"
    trust_cards.append(_render_signal_card(
        "Workaround availability",
        value,
        cls, label, progress,
        "Shows whether there is a practical temporary path while a bug or product gap is unresolved.",
        "No workaround makes a support or product issue much more urgent for the customer.",
        "Ask Support/Product to document the workaround or confirm that none exists, then set clear customer expectations.",
        "Customer Support + Product / Engineering",
        "Turns a technical issue into a clear expectation-management action."
    ))

    def section(title, sub, cards):
        st.markdown(
            f'<div class="signal-section"><div class="signal-section-title">{html.escape(title)}</div>'
            f'<div class="signal-section-sub">{html.escape(sub)}</div>'
            f'<div class="signal-brief-grid">{"".join(cards)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="briefing-note"><strong>How to read this section:</strong> these cards translate Semrush-like fields into plain CSM language. Hover over the <strong>?</strong> icon on each card to understand the metric, why it matters, what the problem may be, and which team can help.</div>',
        unsafe_allow_html=True,
    )
    section("Usage & adoption", "Is the customer actively using the workflows they pay for?", usage_cards)
    section("Limits & friction", "Is the customer blocked by package limits, allocations or configuration expectations?", limit_cards)
    section("Product fit & trust", "Does the customer believe the product fits their needs and can be trusted?", trust_cards)


def simulate_scenarios_raw(row, model, model_columns):
    base = pd.DataFrame([row.drop(labels=[c for c in ["churn_probability", "risk_level", "priority"] if c in row.index])])
    current = float(row["churn_probability"])
    scenarios = []

    def add(name, lever, description, changes, cta, owners, tooltip_title, tooltip_what, tooltip_problem):
        simulated = base.copy()
        for k, v in changes.items():
            if k in simulated.columns:
                simulated.loc[simulated.index[0], k] = v(simulated.loc[simulated.index[0], k]) if callable(v) else v
        new_prob = float(predict_for_df(model, model_columns, simulated)[0])
        delta = new_prob - current
        scenarios.append({
            "Scenario": name,
            "Lever": lever,
            "What changes": description,
            "Current probability": current,
            "Simulated probability": new_prob,
            "Change": delta,
            "New risk level": risk_level(new_prob),
            "CTA": cta,
            "Owners": owners,
            "Tooltip title": tooltip_title,
            "Tooltip what": tooltip_what,
            "Tooltip problem": tooltip_problem,
        })

    add(
        "Improve Site Audit & usage",
        "Adoption recovery",
        "Increase activity/adoption and reduce inactivity",
        {
            "tool_activity_score": lambda x: min(88, max(x, 72)),
            "feature_adoption_score": lambda x: min(85, max(x, 70)),
            "days_since_last_login": lambda x: min(x, 5),
            "site_audit_last_run_days": lambda x: min(x, 7),
        },
        "Run a workflow recovery session focused on Site Audit, Position Tracking and reporting habits.",
        "CSM + Customer Support",
        "Improve usage and adoption",
        "Tests how risk might change if the customer starts using the core workflows more consistently.",
        "If this has the highest impact, churn risk is likely driven by low value realization rather than only bugs or pricing."
    )
    add(
        "Resolve limit friction",
        "Limits and package fit",
        "Reduce crawl/keyword/prompt saturation and mark upgrade-fit action",
        {
            "crawl_budget_usage_pct": lambda x: min(x, 72),
            "keyword_tracking_usage_pct": lambda x: min(x, 72),
            "ai_visibility_prompt_usage_pct": lambda x: min(x, 72),
            "limit_upgrade_recommended_flag": 0,
        },
        "Clarify limits, allocation and package fit. Involve Sales if the issue is actually capacity, not configuration.",
        "Customer Support + Account Executive / Sales",
        "Resolve limit friction",
        "Tests the impact of reducing saturation around crawl, keyword tracking or AI prompt limits.",
        "If this matters, the customer may be active but frustrated by plan limits or allocation expectations."
    )
    add(
        "Strengthen renewal signal",
        "Commercial confidence",
        "Move renewal intent to positive and remove grace/pricing/downgrade risk",
        {
            "renewal_intent": "positive",
            "in_grace_period": 0,
            "pricing_objection_flag": 0,
            "downgrade_risk_flag": 0,
        },
        "Align with the commercial owner before the next customer conversation and clarify decision criteria.",
        "CSM + Account Executive / Sales",
        "Strengthen renewal signal",
        "Tests how risk might change if the commercial conversation becomes positive.",
        "If this has high impact, urgency is tied to renewal timing, pricing, contract fit or downgrade risk."
    )
    add(
        "Resolve product feedback",
        "Product fit",
        "Close critical feedback and improve roadmap alignment/sentiment",
        {
            "critical_feedback_count_90d": 0,
            "unresolved_feedback_count": lambda x: max(0, min(x, 1)),
            "avg_unresolved_feedback_age_days": lambda x: min(x, 14),
            "roadmap_alignment_score": lambda x: max(x, 75),
            "feedback_sentiment_score": lambda x: max(x, 0.35),
            "feedback_linked_to_renewal": 0,
        },
        "Confirm workaround, roadmap status and expectation-management message before renewal.",
        "CSM + Product / Engineering",
        "Resolve product feedback",
        "Tests the impact of reducing unresolved product gaps and improving roadmap fit.",
        "If this is important, the risk is probably about unmet product needs, not just usage or support quality."
    )
    add(
        "Resolve bug reliability issues",
        "Trust recovery",
        "Remove stuck/data-trust bugs and set workaround available",
        {
            "critical_bugs_reported_90d": 0,
            "stuck_bugs_count": 0,
            "open_bugs_count": lambda x: max(0, min(x, 1)),
            "oldest_open_bug_days": lambda x: min(x, 7),
            "data_accuracy_bug_flag": 0,
            "workaround_available_flag": 1,
            "bug_linked_to_renewal_flag": 0,
            "bug_sentiment_score": lambda x: max(x, 0.35),
        },
        "Escalate as trust-risk: confirm impact, workaround, ETA and update cadence with Support/Product.",
        "Customer Support + Product / Engineering",
        "Resolve reliability issues",
        "Tests how risk might change if stuck bugs and data-trust problems are solved.",
        "If this has high impact, the customer may trust the platform less because of reliability or data accuracy concerns."
    )
    add(
        "Best-case recovery package",
        "Combined action plan",
        "Combine adoption, renewal, feedback and bug recovery",
        {
            "tool_activity_score": lambda x: min(90, max(x, 78)),
            "feature_adoption_score": lambda x: min(90, max(x, 76)),
            "days_since_last_login": lambda x: min(x, 4),
            "renewal_intent": "positive",
            "in_grace_period": 0,
            "pricing_objection_flag": 0,
            "unresolved_feedback_count": 0,
            "critical_feedback_count_90d": 0,
            "roadmap_alignment_score": lambda x: max(x, 80),
            "stuck_bugs_count": 0,
            "data_accuracy_bug_flag": 0,
            "workaround_available_flag": 1,
            "bug_linked_to_renewal_flag": 0,
        },
        "Create a coordinated recovery plan with CSM, Support, Sales and Product focused on value, renewal and blockers.",
        "CSM + Support + Sales + Product",
        "Best-case recovery package",
        "Shows the potential impact of acting across multiple risk families at once.",
        "This is not a promise. It helps the CSM explain which combination of actions might create the strongest recovery path."
    )
    return pd.DataFrame(scenarios)


def render_scenario_story(row, model, model_columns):
    raw_scenarios = simulate_scenarios_raw(row, model, model_columns)
    if raw_scenarios.empty:
        st.info("No scenario data available for this account.")
        return

    raw_scenarios["Improvement pp"] = (raw_scenarios["Current probability"] - raw_scenarios["Simulated probability"]) * 100
    best = raw_scenarios.sort_values("Improvement pp", ascending=False).iloc[0]
    current = float(best["Current probability"])
    after = float(best["Simulated probability"])
    improvement = float(best["Improvement pp"])
    impact_class = _impact_level(improvement)

    owners_html = "".join([f'<span class="ux-owner {_owner_class(x.strip())}">{html.escape(x.strip())}</span>' for x in str(best["Owners"]).split("+")])
    st.markdown(
        f'<div class="scenario-hero">'
        f'<div class="ux-kicker">Best lever to reduce risk</div>'
        f'<div class="ux-title">{html.escape(str(best["Scenario"]))}</div>'
        f'<div class="ux-body">{html.escape(str(best["Tooltip problem"]))}</div>'
        f'<div class="scenario-metrics">'
        f'<div class="scenario-metric"><div class="scenario-label">Current risk</div><div class="scenario-value">{current:.1%}</div></div>'
        f'<div class="scenario-metric"><div class="scenario-label">Potential risk</div><div class="scenario-value">{after:.1%}</div></div>'
        f'<div class="scenario-metric"><div class="scenario-label">Potential impact</div><div class="scenario-value">-{improvement:.1f} pp</div></div>'
        f'<div class="scenario-metric"><div class="scenario-label">New level</div><div class="scenario-value">{html.escape(str(best["New risk level"]))}</div></div>'
        f'</div>'
        f'<div class="signal-cta" style="margin-top:14px;"><strong>Suggested CSM action:</strong> {html.escape(str(best["CTA"]))}<br>{owners_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cards = []
    for _, r in raw_scenarios.sort_values("Improvement pp", ascending=False).iterrows():
        imp = max(0, float(r["Improvement pp"]))
        cls = _impact_level(imp)
        owners = "".join([f'<span class="ux-owner {_owner_class(x.strip())}">{html.escape(x.strip())}</span>' for x in str(r["Owners"]).split("+")])
        tooltip = _tooltip(
            str(r["Tooltip title"]),
            str(r["Tooltip what"]),
            str(r["Tooltip problem"]),
            str(r["CTA"]),
            str(r["Owners"]),
        )
        cards.append(
            f'<div class="scenario-card {"best" if cls == "high" else cls}">'
            f'<div class="ux-kicker">{html.escape(str(r["Lever"]))}</div>'
            f'<div class="ux-title">{html.escape(str(r["Scenario"]))} {tooltip}</div>'
            f'<span class="ux-impact {cls}">{"High" if cls == "high" else "Medium" if cls == "medium" else "Low"} potential impact</span>'
            f'<div class="ux-big">{float(r["Simulated probability"]):.1%}</div>'
            f'<div class="ux-body"><strong>Change:</strong> {float(r["Change"])*100:+.1f} pp from current {float(r["Current probability"]):.1%}. New level: <strong>{html.escape(str(r["New risk level"]))}</strong>.</div>'
            f'<div class="signal-cta"><strong>CTA:</strong> {html.escape(str(r["CTA"]))}<br>{owners}</div>'
            f'</div>'
        )
    st.markdown('<div class="ux-grid-3">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="briefing-note"><strong>How to use this:</strong> scenarios are not guarantees. They help the CSM compare which lever may reduce risk fastest and which departments should be involved before the next customer touchpoint.</div>',
        unsafe_allow_html=True,
    )


def simulate_scenarios(row, model, model_columns):
    """Formatted version kept for compatibility and export/debugging."""
    out = simulate_scenarios_raw(row, model, model_columns).copy()
    out["Current probability"] = out["Current probability"].map(lambda x: f"{x:.1%}")
    out["Simulated probability"] = out["Simulated probability"].map(lambda x: f"{x:.1%}")
    out["Change"] = out["Change"].map(lambda x: f"{x*100:+.1f} pp")
    return out[["Scenario", "What changes", "Current probability", "Simulated probability", "Change", "New risk level", "CTA", "Owners"]]


# Load data and model
inject_css()

if st.session_state.get("active_view", "Mission Control") == "Mission Control":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow"><span class="status-pulse"></span> CUSTOMER SUCCESS / INTELLIGENCE PLATFORM</div>
          <div class="hero-title">Customer Value &amp; Growth <span class="hero-highlight">Copilot</span></div>
          <div class="hero-sub">One connected workspace for retention, customer outcomes, growth opportunities and accountable execution. Move from signals to decisions — and from decisions to measurable results.</div>
          <div class="hero-foot"><span>PORTFOLIO INTELLIGENCE</span><span>ACCOUNT DECISIONS</span><span>CROSS-FUNCTIONAL EXECUTION</span></div>
        </div>
        """, unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="v171-mini-brand"><span class="v171-brand-mark">◆</span>'
        '<span>CUSTOMER VALUE <em>&</em> GROWTH</span>'
        '<span class="v171-brand-context">Customer Success intelligence</span></div>',
        unsafe_allow_html=True,
    )

missing = [str(p.name) for p in [DATA_PATH, MODEL_PATH] if not p.exists()]
if missing:
    st.error("Missing required file(s): " + ", ".join(missing) + ". Place them in the same folder as this app.")
    st.stop()

raw = load_data(DATA_PATH)
model, model_columns, model_package = load_model(MODEL_PATH)
df = raw.copy()
df["churn_probability"] = predict_for_df(model, model_columns, df)
df["risk_level"] = df["churn_probability"].apply(risk_level)
df["priority"] = np.select(
    [
        (df["risk_level"] == "High") & (df["mrr"] >= 1500),
        (df["risk_level"] == "High"),
        (df["risk_level"] == "Medium") & (df["renewal_due_days"] < 60),
        (df["risk_level"] == "Medium"),
    ],
    ["P1", "P2", "P2", "P3"],
    default="Monitor",
)
df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

# Global navigation always appears before portfolio filters.
# Screen identifiers stay unchanged to preserve existing deep links and business logic.
st.session_state.setdefault("active_view", "Mission Control")
st.session_state.setdefault("queue_filter", "All accounts")
st.sidebar.markdown(
    '<div class="v171-sidebar-brand"><div class="v171-sidebar-monogram">CV</div>'
    '<div><strong>Customer Value</strong><small>INTELLIGENCE WORKSPACE</small></div></div>',
    unsafe_allow_html=True,
)

NAV_GROUPS = {
    "WORKSPACE": [
        ("Mission Control", "◉  Mission Control"),
        ("Priority Queue", "⚑  Priority Queue"),
        ("Account Briefing", "◎  Account 360°"),
        ("My CSM Book", "▦  My CSM Book"),
    ],
    "CUSTOMER VALUE & REVENUE": [
        ("Real-data Pilot (MVP 2.0)", "◇  Customer Value"),
        ("Growth & Expansion (MVP 2.1)", "↗  Growth Intelligence"),
        ("End-to-End Flow (MVP 2.1.1)", "⇄  Revenue Lifecycle"),
    ],
    "ANALYSIS & RESOURCES": [
        ("Portfolio Insights", "◫  Portfolio Insights"),
        ("Scenario Simulator", "≋  Scenario Simulator"),
        ("CSM Playbook", "☷  CS Playbook"),
        ("Methodology & Data Mapping", "▤  Data & Methodology"),
    ],
}

def v171_navigate(view):
    if view == "Account Briefing" and st.session_state.get("selected_account_id"):
        st.session_state["briefing_account"] = st.session_state["selected_account_id"]
    st.session_state["active_view"] = view

for group_name, destinations in NAV_GROUPS.items():
    with st.sidebar.container():
        st.sidebar.markdown(f'<div class="v171-nav-group">{html.escape(group_name)}</div>', unsafe_allow_html=True)
        for destination, label in destinations:
            selected = st.session_state.get("active_view") == destination
            st.sidebar.button(
                label, key="v171_nav_" + re.sub(r"[^a-z0-9]", "_", destination.lower()),
                on_click=v171_navigate, args=(destination,),
                use_container_width=True, type="primary" if selected else "secondary",
            )

segments = ["All"] + sorted(df["segment"].dropna().unique().tolist())
plan_families = ["All"] + sorted(df["plan_family"].dropna().unique().tolist())
plans = ["All"] + sorted(df["plan_tier"].dropna().unique().tolist())
risks = ["All", "High", "Medium", "Low"]
regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
csm_owners = ["All"] + sorted(df["csm_name"].dropna().unique().tolist()) if "csm_name" in df.columns else ["All"]
csm_managers = ["All"] + sorted(df["csm_manager"].dropna().unique().tolist()) if "csm_manager" in df.columns else ["All"]
csm_focus_areas = ["All"] + sorted(df["csm_focus_area"].dropna().unique().tolist()) if "csm_focus_area" in df.columns else ["All"]
book_tiers = ["All"] + sorted(df["book_tier"].dropna().unique().tolist()) if "book_tier" in df.columns else ["All"]
routing_options = ["All", "Needs Customer Support", "Needs Account Executive / Sales", "Needs Product / Engineering", "CSM only"]

FILTER_KEYS = [
    "v171_segment", "v171_family", "v171_plan", "v171_risk", "v171_region",
    "v171_csm", "v171_manager", "v171_focus", "v171_tier", "v171_routing",
    "v171_only_renewal", "v171_only_limit", "v171_only_bugs",
]

def v171_clear_filters():
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)

active_filters = sum(
    bool(st.session_state.get(key, False)) if key.startswith("v171_only_")
    else st.session_state.get(key, "All") != "All"
    for key in FILTER_KEYS
)
st.sidebar.markdown('<div class="v171-filter-intro">PORTFOLIO SCOPE</div>', unsafe_allow_html=True)
with st.sidebar.expander(f"Portfolio filters · {active_filters} active", expanded=False):
    st.caption("These filters apply to the synthetic portfolio, not the authorized customer workspace.")
    st.button("Clear all filters", key="v171_clear_filters", use_container_width=True, on_click=v171_clear_filters)
    segment_filter = st.selectbox("Segment", segments, key="v171_segment")
    family_filter = st.selectbox("Plan family", plan_families, key="v171_family")
    plan_filter = st.selectbox("Plan tier", plans, key="v171_plan")
    risk_filter = st.selectbox("Risk level", risks, key="v171_risk")
    region_filter = st.selectbox("Region", regions, key="v171_region")
    st.divider()
    st.caption("BOOK OWNERSHIP")
    csm_filter = st.selectbox("CSM owner", csm_owners, key="v171_csm")
    csm_manager_filter = st.selectbox("CSM manager", csm_managers, key="v171_manager")
    csm_focus_filter = st.selectbox("CSM focus area", csm_focus_areas, key="v171_focus")
    book_tier_filter = st.selectbox("Book tier", book_tiers, key="v171_tier")
    st.divider()
    st.caption("SIGNALS & ROUTING")
    only_renewal = st.checkbox("Renewal due < 60 days", key="v171_only_renewal")
    only_limit = st.checkbox("Limit pressure detected", key="v171_only_limit")
    only_bugs = st.checkbox("Bug/data trust issues", key="v171_only_bugs")
    routing_filter = st.selectbox("Action routing", routing_options, key="v171_routing")

with st.sidebar.expander("Queue focus", expanded=False):
    queue_options = ["All accounts", "P1/P2 accounts", "High-risk accounts", "MRR at risk", "Renewal risk", "Adoption gap", "Product friction", "Cross-functional routing", "Needs Customer Support", "Needs Account Executive / Sales", "Needs Product / Engineering"]
    current_queue = st.session_state.get("queue_filter", "All accounts")
    # A selectbox callback updates queue focus without overwriting decisions made by cards/CTAs.
    if st.session_state.get("v171_queue_widget_synced") != current_queue:
        st.session_state["v171_queue_choice"] = current_queue
        st.session_state["v171_queue_widget_synced"] = current_queue
    def v171_queue_change():
        st.session_state["queue_filter"] = st.session_state["v171_queue_choice"]
        st.session_state["v171_queue_widget_synced"] = st.session_state["v171_queue_choice"]
    st.selectbox("Queue view", queue_options, key="v171_queue_choice", on_change=v171_queue_change)

st.sidebar.caption("Synthetic portfolio indicators are separate from source-referenced customer records.")

filtered = df.copy()
if segment_filter != "All": filtered = filtered[filtered["segment"] == segment_filter]
if family_filter != "All": filtered = filtered[filtered["plan_family"] == family_filter]
if plan_filter != "All": filtered = filtered[filtered["plan_tier"] == plan_filter]
if risk_filter != "All": filtered = filtered[filtered["risk_level"] == risk_filter]
if region_filter != "All": filtered = filtered[filtered["region"] == region_filter]
if csm_filter != "All" and "csm_name" in filtered.columns: filtered = filtered[filtered["csm_name"] == csm_filter]
if csm_manager_filter != "All" and "csm_manager" in filtered.columns: filtered = filtered[filtered["csm_manager"] == csm_manager_filter]
if csm_focus_filter != "All" and "csm_focus_area" in filtered.columns: filtered = filtered[filtered["csm_focus_area"] == csm_focus_filter]
if book_tier_filter != "All" and "book_tier" in filtered.columns: filtered = filtered[filtered["book_tier"] == book_tier_filter]
if only_renewal: filtered = filtered[filtered["renewal_due_days"] < 60]
if only_limit: filtered = filtered[(filtered["crawl_budget_usage_pct"] > 85) | (filtered["keyword_tracking_usage_pct"] > 85) | (filtered["ai_visibility_prompt_usage_pct"] > 85)]
if only_bugs: filtered = filtered[(filtered["stuck_bugs_count"] > 0) | (filtered["data_accuracy_bug_flag"] == 1)]
if routing_filter != "All":
    filtered = filtered[filtered.apply(lambda r: route_filter_matches(r, routing_filter), axis=1)]
filtered = filtered.sort_values("churn_probability", ascending=False).reset_index(drop=True)

if filtered.empty:
    st.warning("No portfolio accounts match the current filters. Open **Portfolio filters** in the sidebar and select **Clear all filters** to restore the full account view.")
    st.stop()


# =========================
# V4 CSM Mission Control UX
# =========================

avg_churn = filtered["churn_probability"].mean()
high_count = int((filtered["risk_level"] == "High").sum())
medium_count = int((filtered["risk_level"] == "Medium").sum())
low_count = int((filtered["risk_level"] == "Low").sum())
mrr_at_risk = filtered.loc[filtered["risk_level"] == "High", "mrr"].sum()
renewal_pressure = int(((filtered["renewal_due_days"] < 60) & (filtered["renewal_intent"] != "positive")).sum())
renewal_pressure_30 = int(((filtered["renewal_due_days"] < 30) & (filtered["renewal_intent"] != "positive")).sum())
limit_pressure = int(((filtered["crawl_budget_usage_pct"] > 85) | (filtered["keyword_tracking_usage_pct"] > 85) | (filtered["ai_visibility_prompt_usage_pct"] > 85)).sum())
low_usage_count = int(((filtered["tool_activity_score"] < 45) | (filtered["feature_adoption_score"] < 45)).sum())
product_friction_count = int(((filtered["stuck_bugs_count"] > 0) | (filtered["data_accuracy_bug_flag"] == 1) | (filtered["unresolved_feedback_count"] >= 3) | (filtered["roadmap_alignment_score"] < 40)).sum())
support_needed_count = int(filtered.apply(lambda r: route_filter_matches(r, "Needs Customer Support"), axis=1).sum())
sales_needed_count = int(filtered.apply(lambda r: route_filter_matches(r, "Needs Account Executive / Sales"), axis=1).sum())
product_needed_count = int(filtered.apply(lambda r: route_filter_matches(r, "Needs Product / Engineering"), axis=1).sum())
csm_only_count = int(filtered.apply(lambda r: route_filter_matches(r, "CSM only"), axis=1).sum())

family_scores_df = pd.DataFrame([family_risk_scores(r) for _, r in filtered.iterrows()])
avg_family_scores = family_scores_df.mean().sort_values(ascending=False) if not family_scores_df.empty else pd.Series(dtype=float)
primary_theme = avg_family_scores.index[0] if len(avg_family_scores) else "No theme detected"
secondary_theme = avg_family_scores.index[1] if len(avg_family_scores) > 1 else "No secondary theme"
primary_theme_score = float(avg_family_scores.iloc[0]) if len(avg_family_scores) else 0.0
highest_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False).index[0] if len(filtered) else "N/A"
highest_plan_family = filtered.groupby("plan_family")["churn_probability"].mean().sort_values(ascending=False).index[0] if len(filtered) else "N/A"


def _fmt_pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "N/A"


def _fmt_money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "N/A"


def urgency_label(row):
    if row.get("risk_level") == "High" and row.get("renewal_due_days", 999) < 30:
        return "Act today"
    if row.get("risk_level") == "High":
        return "Act this week"
    if row.get("risk_level") == "Medium" and row.get("renewal_due_days", 999) < 60:
        return "Review this week"
    if row.get("risk_level") == "Medium":
        return "Monitor closely"
    return "Monitor"


def account_story_summary(row):
    drivers = top_drivers(row, 3)
    if not drivers:
        return "No major visible risk driver stands out. Continue regular value reinforcement."
    if len(drivers) == 1:
        driver_text = drivers[0]
    elif len(drivers) == 2:
        driver_text = f"{drivers[0]} and {drivers[1]}"
    else:
        driver_text = f"{drivers[0]}, {drivers[1]} and {drivers[2]}"
    return f"This account needs attention because the strongest visible signals are {driver_text}."


def snapshot_card_html(kicker, title, value, body, cta, tone="blue"):
    tone_color = {"blue":"#2563EB", "teal":"#0F766E", "red":"#DC2626", "amber":"#D97706", "purple":"#6D28D9"}.get(tone, "#2563EB")
    return (
        f'<div class="mission-card" style="border-top:5px solid {tone_color};">'
        f'<div class="mission-kicker">{html.escape(str(kicker))}</div>'
        f'<div class="mission-title">{html.escape(str(title))}</div>'
        f'<div class="mission-value">{html.escape(str(value))}</div>'
        f'<div class="mission-body">{html.escape(str(body))}</div>'
        f'<div class="mission-cta">{html.escape(str(cta))}</div>'
        f'</div>'
    )


def mission_guide_html():
    steps = [
        ("01", "Understand exposure", "Identify the accounts, value gaps and critical renewal windows."),
        ("02", "Choose a priority", "Focus on the P1/P2 queue or a specific problem signal."),
        ("03", "Open the account", "Review evidence, drivers, owner and customer context."),
        ("04", "Decide the next move", "Compare interventions and choose the most relevant action."),
        ("05", "Coordinate & measure", "Engage the right team and verify the outcome."),
    ]
    cards = "".join([
        f'<div class="guide-step"><div class="guide-number">{n}</div><div class="guide-title">{html.escape(t)}</div><div class="guide-body">{html.escape(b)}</div></div>'
        for n,t,b in steps
    ])
    return f'<div class="guide-grid">{cards}</div>'


def daily_brief_html():
    if high_count > 0:
        opening = f"Today, start with {high_count} high-risk accounts."
    elif medium_count > 0:
        opening = f"No high-risk accounts in this view, but {medium_count} medium-risk accounts deserve monitoring."
    else:
        opening = "This portfolio view currently looks stable."
    body = (
        f"{opening} The clearest portfolio theme is {primary_theme} ({primary_theme_score:.1f}/100), "
        f"with {secondary_theme} as the secondary signal. Commercial exposure is {_fmt_money(mrr_at_risk)} in high-risk MRR. "
        f"CSMs should first review P1/P2 accounts with renewal pressure, product friction, or cross-functional blockers."
    )
    return (
        f'<div class="daily-brief"><div class="daily-kicker">PRIORITY INTELLIGENCE</div>'
        f'<div class="daily-title">Your next moves, in order of urgency.</div>'
        f'<div class="daily-body">{html.escape(body)}</div></div>'
    )


def mission_snapshot_grid_html():
    cards = [
        snapshot_card_html("Portfolio risk", "How healthy is my book?", f"{avg_churn:.1%}", f"Average churn probability across {len(filtered):,} selected accounts. Main risk theme: {primary_theme}.", "Open Priority Queue", "blue"),
        snapshot_card_html("Revenue & renewal", "Where is commercial risk?", _fmt_money(mrr_at_risk), f"{renewal_pressure_30} accounts renew in <30 days with weak or non-positive intent.", "Open renewal-risk details", "red"),
        snapshot_card_html("Adoption gap", "Who pays but underuses?", f"{low_usage_count}", "Accounts with low product activity or shallow feature adoption.", "Open adoption recovery list", "teal"),
        snapshot_card_html("Product friction", "Where is trust breaking?", f"{product_friction_count}", "Accounts with stuck bugs, data accuracy flags, unresolved feedback or roadmap mismatch.", "Open product friction cases", "amber"),
        snapshot_card_html("Team dependencies", "Who needs to help?", f"{support_needed_count + sales_needed_count + product_needed_count}", "Accounts where Support, Sales or Product may need to help the CSM unblock risk.", "Review cross-functional routing", "purple"),
    ]
    return f'<div class="mission-grid">{"".join(cards)}</div>'


def render_account_card(row, compact=False):
    actions, owners, urgency = semrush_csm_cta(row)
    routing = build_department_routing(row)
    owner_line = " + ".join([x.get("team", "") for x in routing[:3]]) if routing else "CSM"
    drivers = top_drivers(row, 3)
    chips = "".join([f'<span class="driver-chip">{html.escape(str(d))}</span>' for d in drivers])
    urgency_text = urgency_label(row)
    risk_css = "risk-high" if row["risk_level"] == "High" else "risk-medium" if row["risk_level"] == "Medium" else "risk-low"
    height = "min-height:300px;" if not compact else "min-height:245px;"
    return (
        f'<div class="account-queue-card" style="{height}">'
        f'<div class="account-queue-top"><span class="priority-pill">{html.escape(str(row["priority"]))}</span><span class="urgency-mini {risk_css}">{html.escape(urgency_text)}</span></div>'
        f'<div class="account-queue-title">{html.escape(str(row["account_name"]))}</div>'
        f'<div class="account-queue-sub">{html.escape(str(row["account_id"]))} · {html.escape(str(row["segment"]))} · {html.escape(str(row["plan_family"]))} · {html.escape(str(row["plan_tier"]))}</div>'
        f'<div class="account-risk-line"><span>{risk_badge(row["risk_level"])}</span><span class="account-risk-value">{row["churn_probability"]:.1%}</span></div>'
        f'<div class="account-queue-sub">MRR {_fmt_money(row["mrr"])} · Renewal in {int(row["renewal_due_days"])} days · Intent: {html.escape(str(row["renewal_intent"]))}</div>'
        f'<div style="margin:10px 0 8px;">{chips}</div>'
        f'<div class="account-action"><strong>CSM action:</strong> {html.escape(str(actions[0] if actions else "Review account health and confirm next best action."))}</div>'
        f'<div class="account-owner"><strong>Teams:</strong> {html.escape(owner_line)}</div>'
        f'</div>'
    )


def render_priority_cards(dataframe, max_cards=6):
    if dataframe.empty:
        st.info("No accounts in this queue for the current filters.")
        return
    cards = "".join([render_account_card(r, compact=True) for _, r in dataframe.head(max_cards).iterrows()])
    st.markdown(f'<div class="account-card-grid">{cards}</div>', unsafe_allow_html=True)


def three_question_html(row):
    usage_score = (float(row.get("tool_activity_score", 0)) + float(row.get("feature_adoption_score", 0))) / 2
    usage_status = "Yes" if usage_score >= 65 else "Partially" if usage_score >= 40 else "Not enough"
    usage_body = "The customer shows healthy product activity." if usage_score >= 65 else "Usage or adoption needs review before value is clear." if usage_score >= 40 else "The customer may not be realizing enough product value."

    fit_score = (float(row.get("roadmap_alignment_score", 50)) + max(0, 100 - float(row.get("unresolved_feedback_count", 0)) * 15)) / 2
    fit_status = "Mostly" if fit_score >= 65 else "Not fully" if fit_score >= 40 else "No clear fit"
    fit_body = "Current needs appear mostly aligned." if fit_score >= 65 else "Product feedback or roadmap gaps may need expectation management." if fit_score >= 40 else "Unmet product needs may be a major churn reason."

    trust_score = 100
    trust_score -= 25 if int(row.get("data_accuracy_bug_flag", 0)) == 1 else 0
    trust_score -= min(40, int(row.get("stuck_bugs_count", 0)) * 20)
    trust_score -= 20 if float(row.get("csat_support", 5)) < 3.2 else 0
    trust_status = "Yes" if trust_score >= 70 else "At risk" if trust_score >= 40 else "Weak"
    trust_body = "No major trust blocker is visible." if trust_score >= 70 else "Support, bug or data-trust signals need attention." if trust_score >= 40 else "Trust may be materially damaged by unresolved friction."

    items = [
        ("Is the customer using the product?", usage_status, usage_body, usage_score),
        ("Does the product meet their needs?", fit_status, fit_body, fit_score),
        ("Can they trust the product enough to stay?", trust_status, trust_body, trust_score),
    ]
    cards = "".join([
        f'<div class="question-card"><div class="question-title">{html.escape(q)}</div><div class="question-answer">{html.escape(ans)}</div><div class="question-body">{html.escape(body)}</div><div class="bar-wrap"><div class="bar-fill" style="width:{max(0,min(100,score)):.1f}%;"></div></div></div>'
        for q, ans, body, score in items
    ])
    return f'<div class="question-grid">{cards}</div>'


def render_selected_account_briefing(row, include_scenario=False):
    drivers = top_drivers(row)
    chips = ''.join([f'<span class="driver-chip">{html.escape(str(d))}</span>' for d in drivers])
    st.markdown(
        f'<div class="account-brief-hero">'
        f'<div class="metric-label">{html.escape(str(row["account_id"]))} · {html.escape(str(row["plan_family"]))} · {html.escape(str(row["plan_tier"]))}</div>'
        f'<div class="account-brief-title">{html.escape(str(row["account_name"]))}</div>'
        f'<div class="account-brief-sub">{html.escape(str(row["segment"]))} · {html.escape(str(row["industry"]))} · {html.escape(str(row["region"]))} · MRR {_fmt_money(row["mrr"])}</div>'
        f'<div class="account-risk-line"><span>{risk_badge(row["risk_level"])}</span><span class="account-risk-value">{row["churn_probability"]:.1%}</span></div>'
        f'<div class="account-story">{html.escape(account_story_summary(row))}</div>'
        f'<div>{chips}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_header("Three-question briefing", "A simple way for any CSM or stakeholder to read the account story.")
    st.markdown(three_question_html(row), unsafe_allow_html=True)

    section_header("Risk family breakdown", "What kind of risk is driving this account: adoption, renewal, support, feedback, bugs or limits.")
    scores = family_risk_scores(row)
    cols = st.columns(2)
    for idx, (label, value) in enumerate(scores.items()):
        with cols[idx % 2]:
            st.markdown(bar(label, value), unsafe_allow_html=True)

    section_header("Structured CSM action plan", "Direct instructions for the CSM and cross-functional owners.")
    st.markdown(playbook_html(row), unsafe_allow_html=True)

    section_header("Who needs to be involved?", "Cross-functional routing for blockers that the CSM should not solve alone.")
    st.markdown(department_routing_html(row), unsafe_allow_html=True)
    with st.expander("Suggested handoff message", expanded=False):
        st.code(build_handoff_message(row), language="text")

    section_header("Account signal briefing", "Plain-English evidence grouped by Usage, Limits, and Product trust.")
    render_signal_briefing(row)

    if include_scenario:
        section_header("Scenario Simulator", "What could reduce the risk fastest, and which team should help?")
        render_scenario_story(row, model, model_columns)
        with st.expander("View scenario calculation table", expanded=False):
            scenarios = simulate_scenarios(row, model, model_columns)
            st.dataframe(scenarios, use_container_width=True, hide_index=True)
            st.caption("Scenario results are model-based simulations on synthetic data. They are decision-support estimates, not guaranteed outcomes.")

    with st.expander("Supporting account data", expanded=False):
        key_cols = [
            "account_id","account_name","segment","region","plan_family","plan_tier","mrr","churn_probability","risk_level",
            "renewal_due_days","renewal_intent","tool_activity_score","feature_adoption_score","crawl_budget_usage_pct",
            "keyword_tracking_usage_pct","ai_visibility_prompt_usage_pct","roadmap_alignment_score","unresolved_feedback_count",
            "stuck_bugs_count","data_accuracy_bug_flag","csat_support","sentiment_csm"
        ]
        available_cols = [c for c in key_cols if c in row.index]
        evidence = pd.DataFrame({"Field": available_cols, "Value": [str(row[c]) for c in available_cols]})
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    with st.expander("Raw account data", expanded=False):
        raw_df = pd.DataFrame({"Field": row.index, "Value": [str(v) for v in row.values]})
        st.dataframe(raw_df, use_container_width=True, hide_index=True)


# =========================
# Account 360° immersive briefing layer (V8)
# =========================

def _status_from_score(score: float, inverse: bool = False) -> tuple[str, str]:
    score = float(score)
    if inverse:
        if score >= 70:
            return "Healthy", "low"
        if score >= 40:
            return "Needs review", "medium"
        return "At risk", "high"
    if score >= 70:
        return "High pressure", "high"
    if score >= 40:
        return "Needs review", "medium"
    return "Controlled", "low"


def dominant_account_theme(row: pd.Series) -> str:
    """Return the strongest visible risk family for a single account.

    Used by the Account 360° cockpit to summarize the account story in one clear theme.
    """
    scores = family_risk_scores(row)
    if not scores:
        return "General health risk"

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_theme, top_score = sorted_scores[0]

    if float(top_score) <= 0:
        return "General health risk"

    return str(top_theme)


def account_360_core_html(row: pd.Series) -> str:
    actions, owners, urgency = semrush_csm_cta(row)
    drivers = top_drivers(row, 5)
    driver_chips = "".join([f'<span class="driver-chip">{html.escape(str(d))}</span>' for d in drivers])
    routing = build_department_routing(row)
    teams = " + ".join([r.get("team", "") for r in routing[:3]]) if routing else "CSM"
    main_theme = dominant_account_theme(row)
    return f'''
    <div class="account360-shell">
        <div class="account360-core-card">
            <div class="account360-eyebrow">Account 360° cockpit</div>
            <div class="account360-title">{html.escape(str(row["account_name"]))}</div>
            <div class="account360-sub">{html.escape(str(row["account_id"]))} · {html.escape(str(row["segment"]))} · {html.escape(str(row["plan_family"]))} · {html.escape(str(row["plan_tier"]))}</div>
            <div class="account360-risk-row"><span>{risk_badge(row["risk_level"])}</span><span class="account360-prob">{float(row["churn_probability"]):.1%}</span></div>
            <div class="account360-story"><strong>Main account story:</strong> {html.escape(account_story_summary(row))}</div>
            <div>{driver_chips}</div>
        </div>
        <div class="account360-priority-card">
            <div class="account360-kicker">First action to take</div>
            <div class="account360-action-title">{html.escape(actions[0] if actions else "Review account health and confirm the next best action.")}</div>
            <div class="account360-action-meta">Main theme: <strong>{html.escape(main_theme)}</strong> · Urgency: <strong>{html.escape(urgency)}</strong></div>
            <div class="account360-action-meta">Teams likely needed: <strong>{html.escape(teams)}</strong></div>
        </div>
        <div class="account360-commercial-card">
            <div class="account360-kicker">Commercial context</div>
            <div class="account360-metric-row"><span>MRR</span><strong>{_fmt_money(row.get("mrr", 0))}</strong></div>
            <div class="account360-metric-row"><span>Renewal due</span><strong>{int(row.get("renewal_due_days", 0))} days</strong></div>
            <div class="account360-metric-row"><span>Renewal intent</span><strong>{html.escape(str(row.get("renewal_intent", "unknown")))}</strong></div>
            <div class="account360-metric-row"><span>Priority</span><strong>{html.escape(str(row.get("priority", "Monitor")))}</strong></div>
        </div>
    </div>
    '''


def account_360_radar_html(row: pd.Series) -> str:
    scores = family_risk_scores(row)
    ordered = [
        ("Usage & adoption", "Is the customer actively using the product?", "CSM"),
        ("Renewal & commercial", "Is the commercial timing sensitive?", "CSM + AE"),
        ("Support experience", "Is service friction weakening trust?", "Customer Support"),
        ("Feedback & roadmap", "Are unmet needs creating product-fit risk?", "CSM + Product"),
        ("Bugs & reliability", "Can the customer trust the product output?", "Support + Product"),
        ("Limit friction", "Is the customer hitting plan or tool limits?", "Support + AE"),
    ]
    positions = [
        ("50%", "5%", "#0F766E"),
        ("86%", "28%", "#DC2626"),
        ("78%", "72%", "#2563EB"),
        ("50%", "93%", "#6D28D9"),
        ("18%", "72%", "#D97706"),
        ("14%", "28%", "#0891B2"),
    ]
    nodes = []
    details = []
    for (label, meaning, owner), (left, top, color) in zip(ordered, positions):
        value = float(scores.get(label, 0))
        status, css = _status_from_score(value)
        nodes.append(
            f'<div class="account360-node {css}" style="left:{left}; top:{top}; --node-color:{color};" title="{html.escape(meaning)} Suggested owner: {html.escape(owner)}.">'
            f'<span class="account360-node-label">{html.escape(label)}</span>'
            f'<span class="account360-node-value">{value:.0f}</span>'
            f'</div>'
        )
        details.append(
            f'<div class="account360-radar-row">'
            f'<div><strong>{html.escape(label)}</strong><br><span>{html.escape(meaning)}</span></div>'
            f'<div class="account360-status {css}">{html.escape(status)}</div>'
            f'<div class="account360-mini-bar"><div style="width:{max(0,min(100,value)):.1f}%; background:{color};"></div></div>'
            f'<div class="account360-owner">{html.escape(owner)}</div>'
            f'</div>'
        )
    center = float(row.get("churn_probability", 0)) * 100
    return f'''
    <div class="account360-radar-shell">
        <div class="account360-radar-stage">
            <div class="account360-ring ring-a"></div>
            <div class="account360-ring ring-b"></div>
            <div class="account360-ring ring-c"></div>
            <div class="account360-center">
                <div class="core-label">Account risk</div>
                <div class="core-value">{center:.0f}%</div>
                <div class="core-note">{html.escape(str(row.get("risk_level", "")))}</div>
            </div>
            {''.join(nodes)}
        </div>
        <div class="account360-radar-details">
            <div class="account360-kicker">Risk dimensions</div>
            <div class="account360-details-title">Where pressure is coming from</div>
            {''.join(details)}
        </div>
    </div>
    '''


def account_360_questions_html(row: pd.Series) -> str:
    usage_score = (float(row.get("tool_activity_score", 0)) + float(row.get("feature_adoption_score", 0))) / 2
    usage_status = "Healthy" if usage_score >= 65 else "Partial" if usage_score >= 40 else "Weak"
    usage_cta = "Run an adoption recovery session around core workflows." if usage_score < 50 else "Reinforce value and identify the next adoption milestone."

    fit_score = (float(row.get("roadmap_alignment_score", 50)) + max(0, 100 - float(row.get("unresolved_feedback_count", 0)) * 15)) / 2
    fit_status = "Aligned" if fit_score >= 65 else "Needs expectation management" if fit_score >= 40 else "Product-fit risk"
    fit_cta = "Clarify roadmap fit, workaround options and renewal dependency." if fit_score < 55 else "Keep validating product needs and future roadmap alignment."

    trust_score = 100
    trust_score -= 25 if int(row.get("data_accuracy_bug_flag", 0)) == 1 else 0
    trust_score -= min(40, int(row.get("stuck_bugs_count", 0)) * 20)
    trust_score -= 20 if float(row.get("csat_support", 5)) < 3.2 else 0
    trust_status = "Stable" if trust_score >= 70 else "At risk" if trust_score >= 40 else "Weak"
    trust_cta = "Escalate data-trust or bug issues with Support/Product and set an update cadence." if trust_score < 55 else "Maintain proactive support follow-up and value reinforcement."

    items = [
        ("Is the customer using the product?", usage_status, usage_score, usage_cta, "Usage & adoption"),
        ("Does the product meet their needs?", fit_status, fit_score, fit_cta, "Product fit"),
        ("Can they trust the product enough to stay?", trust_status, trust_score, trust_cta, "Trust & reliability"),
    ]
    cards = []
    for question, status, score, cta, lens in items:
        _, css = _status_from_score(score, inverse=True)
        cards.append(
            f'<div class="account360-lens-card {css}">'
            f'<div class="account360-kicker">{html.escape(lens)}</div>'
            f'<div class="account360-lens-question">{html.escape(question)}</div>'
            f'<div class="account360-lens-status">{html.escape(status)}</div>'
            f'<div class="account360-mini-bar"><div style="width:{max(0,min(100,score)):.1f}%;"></div></div>'
            f'<div class="account360-lens-cta"><strong>Suggested move:</strong> {html.escape(cta)}</div>'
            f'</div>'
        )
    return f'<div class="account360-lens-grid">{"".join(cards)}</div>'


def action_360_panel_html(row: pd.Series) -> str:
    actions, owners, urgency = semrush_csm_cta(row)
    routing = build_department_routing(row)
    route_cards = []
    for item in routing[:3]:
        team = item.get("team", "CSM")
        team_css = "support" if "Support" in team else "sales" if "Sales" in team or "Executive" in team else "product" if "Product" in team else "csm"
        route_cards.append(
            f'<div class="account360-route-card {team_css}">'
            f'<div class="account360-route-team">{html.escape(team)}</div>'
            f'<div class="account360-route-label">Why needed</div><div class="account360-route-body">{html.escape(str(item.get("why", "")))}</div>'
            f'<div class="account360-route-label">Ask them to</div><div class="account360-route-body">{html.escape(str(item.get("action", "")))}</div>'
            f'</div>'
        )
    if not route_cards:
        route_cards.append('<div class="account360-route-card csm"><div class="account360-route-team">CSM only</div><div class="account360-route-body">No cross-functional blocker detected.</div></div>')
    return f'''
    <div class="account360-action-shell">
        <div class="account360-action-main">
            <div class="account360-kicker">Action 360°</div>
            <div class="account360-action-head">Recommended next move</div>
            <div class="account360-action-text">{html.escape(actions[0] if actions else "Review account health and validate next best action.")}</div>
            <div class="account360-action-meta">Urgency: <strong>{html.escape(urgency)}</strong> · Owners: <strong>{html.escape(", ".join(owners))}</strong></div>
        </div>
        <div class="account360-route-grid">{''.join(route_cards)}</div>
    </div>
    '''


def render_selected_account_briefing(row, include_scenario=False):
    st.markdown(account_360_core_html(row), unsafe_allow_html=True)
    actions, owners, urgency = semrush_csm_cta(row)
    owner_text = ", ".join(owners) if owners else "CSM"
    st.markdown(
        '<div class="v17-decision-strip">'
        '<div><div class="v17-ds-label">RECOMMENDED NEXT MOVE</div>'
        f'<div class="v17-ds-action">{html.escape(str(actions[0] if actions else "Review account signals and confirm the priority."))}</div>'
        f'<div class="v17-ds-meta">Owner: {html.escape(owner_text)} · Priority: {html.escape(urgency)}</div></div>'
        '<div class="v17-ds-tag">DECISION READY</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="v171-account-insight">'
        '<div><span>WHY THIS MATTERS</span><strong>'
        + html.escape(str(dominant_account_theme(row))) + '</strong>'
        + '<p>' + html.escape(str(account_story_summary(row))) + '</p></div>'
        + '<div class="v171-account-impact">'
        + '<small>RENEWAL</small><strong>' + html.escape(str(int(row.get('renewal_due_days', 0)))) + ' days</strong>'
        + '<small>MONTHLY REVENUE</small><strong>' + html.escape(_fmt_money(row.get('mrr', 0))) + '</strong></div>'
        + '</div>', unsafe_allow_html=True,
    )
    st.caption("Decision support uses synthetic account signals. Verify customer context before taking action.")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↗ Compare interventions", key="v17_account_compare", use_container_width=True):
            st.session_state["selected_account_id"] = row["account_id"]
            go_to("Scenario Simulator")
    with c2:
        if st.button("◎ Review priority queue", key="v17_account_queue", use_container_width=True):
            go_to("Priority Queue")
    with c3:
        if st.button("⇄ Customer value workspace", key="v17_account_value", use_container_width=True):
            go_to("Real-data Pilot (MVP 2.0)")

    section_header("Account 360° risk map", "See the strongest risk signals together. Start with the largest unresolved issue and validate the supporting evidence.")
    st.markdown(account_360_radar_html(row), unsafe_allow_html=True)

    section_header("Three-question CSM lens", "Three questions connect observed usage, unmet needs and product trust to a clear customer conversation.")
    st.markdown(account_360_questions_html(row), unsafe_allow_html=True)

    section_header("Action 360°", "Translate account signals into a clear next move, with accountable owners and a documented handoff.")
    st.markdown(action_360_panel_html(row), unsafe_allow_html=True)
    with st.expander("Suggested handoff message", expanded=False):
        st.code(build_handoff_message(row), language="text")

    section_header("Scenario Simulator", "Compare hypothetical interventions before choosing a recovery plan. Simulations are not forecasts of actual intervention success.")
    render_scenario_story(row, model, model_columns)
    with st.expander("View scenario calculation table", expanded=False):
        scenarios = simulate_scenarios(row, model, model_columns)
        st.dataframe(scenarios, use_container_width=True, hide_index=True)
        st.caption("Scenario results are model-based simulations on synthetic data. They are decision-support estimates, not guaranteed outcomes.")

    section_header("Account signal briefing", "Explore usage, capacity and trust in plain English. Each signal includes context and practical follow-up guidance.")
    render_signal_briefing(row)

    section_header("Structured CSM action plan", "A repeatable plan for coordination and follow-up, grounded in the customer signals shown above.")
    st.markdown(playbook_html(row), unsafe_allow_html=True)

    section_header("Supporting evidence", "Review underlying fields and their values when you need to investigate or document a decision.")
    with st.expander("Key account evidence", expanded=False):
        key_cols = [
            "account_id","account_name","segment","region","plan_family","plan_tier","mrr","churn_probability","risk_level",
            "renewal_due_days","renewal_intent","tool_activity_score","feature_adoption_score","crawl_budget_usage_pct",
            "keyword_tracking_usage_pct","ai_visibility_prompt_usage_pct","roadmap_alignment_score","unresolved_feedback_count",
            "stuck_bugs_count","data_accuracy_bug_flag","csat_support","sentiment_csm"
        ]
        available_cols = [c for c in key_cols if c in row.index]
        evidence = pd.DataFrame({"Field": available_cols, "Value": [str(row[c]) for c in available_cols]})
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    with st.expander("Raw account data", expanded=False):
        raw_df = pd.DataFrame({"Field": row.index, "Value": [str(v) for v in row.values]})
        st.dataframe(raw_df, use_container_width=True, hide_index=True)


# Additional CSS for guided V4 UX
st.markdown(
    '<style>'
    '.guide-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:12px 0 22px;}'
    '.guide-step{background:white;border:1px solid var(--border);border-radius:20px;padding:16px;box-shadow:0 8px 22px rgba(15,23,42,.045);}'
    '.guide-number{width:28px;height:28px;border-radius:999px;background:linear-gradient(135deg,#2563EB,#0F766E);color:white;display:flex;align-items:center;justify-content:center;font-weight:950;margin-bottom:10px;}'
    '.guide-title{color:var(--navy);font-size:14px;font-weight:950;margin-bottom:6px;}'
    '.guide-body{color:#64748B;font-size:12px;line-height:1.35;}'
    '.daily-brief{background:radial-gradient(circle at 100% 0%, rgba(15,118,110,.20), transparent 28%),linear-gradient(135deg,#FFFFFF,#F0FDFA);border:1px solid #99F6E4;border-radius:24px;padding:22px;box-shadow:0 12px 30px rgba(15,23,42,.06);margin:10px 0 18px;}'
    '.daily-kicker{color:#0F766E;font-size:12px;font-weight:950;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;}'
    '.daily-title{color:var(--navy);font-size:24px;font-weight:950;margin-bottom:8px;}'
    '.daily-body{color:#334155;font-size:15px;line-height:1.55;}'
    '.mission-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:12px 0 22px;}'
    '.mission-card{background:white;border:1px solid var(--border);border-radius:22px;padding:18px;box-shadow:0 10px 28px rgba(15,23,42,.055);min-height:235px;}'
    '.mission-kicker{color:var(--muted);font-size:11px;font-weight:950;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;}'
    '.mission-title{color:var(--navy);font-size:16px;font-weight:950;line-height:1.2;margin-bottom:10px;}'
    '.mission-value{color:var(--navy);font-size:31px;font-weight:950;line-height:1;margin-bottom:10px;}'
    '.mission-body{color:#475569;font-size:13px;line-height:1.4;margin-bottom:12px;}'
    '.mission-cta{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:10px;color:#334155;font-size:12px;font-weight:850;}'
    '.account-card-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0 22px;}'
    '.account-queue-card{background:white;border:1px solid var(--border);border-radius:22px;padding:19px;box-shadow:0 10px 28px rgba(15,23,42,.055);}'
    '.account-queue-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}'
    '.priority-pill{display:inline-block;border-radius:999px;padding:6px 10px;background:#EEF2FF;color:#4338CA;font-size:12px;font-weight:950;}'
    '.urgency-mini{display:inline-block;border-radius:999px;padding:5px 8px;font-size:11px;font-weight:950;}'
    '.account-queue-title{color:var(--navy);font-size:19px;font-weight:950;line-height:1.2;margin-bottom:5px;}'
    '.account-queue-sub{color:#64748B;font-size:12px;line-height:1.35;margin-bottom:6px;}'
    '.account-risk-line{display:flex;align-items:center;gap:10px;margin:12px 0;}'
    '.account-risk-value{color:var(--navy);font-size:28px;font-weight:950;}'
    '.account-action{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:11px;color:#334155;font-size:12px;line-height:1.4;margin-top:10px;}'
    '.account-owner{color:#475569;font-size:12px;margin-top:8px;}'
    '.account-brief-hero{background:white;border:1px solid var(--border);border-radius:24px;padding:24px;box-shadow:0 12px 30px rgba(15,23,42,.06);margin:12px 0 20px;}'
    '.account-brief-title{color:var(--navy);font-size:32px;font-weight:950;line-height:1.1;margin-bottom:7px;}'
    '.account-brief-sub{color:#64748B;font-size:14px;margin-bottom:12px;}'
    '.account-story{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:13px;color:#334155;font-size:14px;line-height:1.45;margin:12px 0;}'
    '.question-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0 20px;}'
    '.question-card{background:white;border:1px solid var(--border);border-radius:22px;padding:19px;box-shadow:0 10px 28px rgba(15,23,42,.055);min-height:175px;}'
    '.question-title{color:#475569;font-size:12px;font-weight:950;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;}'
    '.question-answer{color:var(--navy);font-size:24px;font-weight:950;margin-bottom:8px;}'
    '.question-body{color:#475569;font-size:13px;line-height:1.4;margin-bottom:8px;}'
    '.return-panel{background:linear-gradient(135deg,#FFFFFF,#EFF6FF);border:1px solid #BFDBFE;border-radius:22px;padding:17px 20px;box-shadow:0 10px 28px rgba(15,23,42,.055);margin:8px 0 12px;}'
    '.return-eyebrow{color:#2563EB;font-size:11px;font-weight:950;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;}'
    '.return-title{color:var(--navy);font-size:20px;font-weight:950;line-height:1.2;margin-bottom:4px;}'
    '.return-body{color:#475569;font-size:13px;line-height:1.45;}'
    '.mission-card{transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease;}'
    '.mission-card:hover{transform:translateY(-3px);box-shadow:0 18px 42px rgba(15,23,42,.10);border-color:#BFDBFE;}'
    'div[data-testid="stButton"] > button{background:linear-gradient(135deg,#071427 0%,#2563EB 58%,#0F766E 100%) !important;color:#FFFFFF !important;border:0 !important;border-radius:16px !important;min-height:46px !important;padding:0.65rem 1rem !important;font-weight:950 !important;box-shadow:0 10px 24px rgba(37,99,235,.22) !important;transition:transform .12s ease, box-shadow .12s ease, filter .12s ease !important;}'
    'div[data-testid="stButton"] > button:hover{transform:translateY(-2px) !important;box-shadow:0 14px 30px rgba(37,99,235,.30) !important;filter:brightness(1.05) !important;}'
    'div[data-testid="stButton"] > button:active{transform:translateY(0px) !important;box-shadow:0 8px 18px rgba(37,99,235,.20) !important;}'
    'div[data-testid="stButton"] > button p{color:#FFFFFF !important;font-weight:950 !important;}'
    '.portfolio360-action-intro{margin:14px 0 10px;padding:13px 16px;border:1px solid #D8E7FF;border-radius:17px;background:linear-gradient(135deg,#FFFFFF,#F1F7FF);color:#334155;font-size:13px;line-height:1.45;}'
    '.immediate-account-card{background:linear-gradient(155deg,#FFFFFF 0%,#FFF7F7 100%);border:1px solid #FECACA;border-top:5px solid #E05D5D;border-radius:22px;padding:17px;min-height:218px;box-shadow:0 14px 30px rgba(127,29,29,.08);margin-bottom:10px;}'
    '.immediate-account-top{display:flex;align-items:center;gap:7px;color:#B91C1C;font-size:11px;font-weight:950;letter-spacing:.08em;}'
    '.immediate-risk-dot{width:9px;height:9px;border-radius:50%;background:#E05D5D;box-shadow:0 0 0 4px rgba(224,93,93,.13);}'
    '.immediate-account-name{font-size:18px;font-weight:950;color:#0B102F;line-height:1.15;margin-top:13px;min-height:40px;}'
    '.immediate-account-id{font-size:12px;color:#64748B;margin-top:4px;}'
    '.immediate-account-metric{font-size:31px;font-weight:950;color:#B91C1C;line-height:1;margin-top:17px;}'
    '.immediate-account-sub{font-size:12px;color:#475569;line-height:1.45;margin-top:8px;}'
    '.immediate-account-theme{display:inline-block;margin-top:11px;padding:6px 9px;border-radius:999px;background:#FFF1F2;color:#9F1239;font-size:11px;font-weight:850;}'
    '@media (max-width:1100px){.mission-grid,.guide-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.account-card-grid,.question-grid{grid-template-columns:1fr;}}'
    '</style>',
    unsafe_allow_html=True,
)


# =========================
# Guided navigation with real CTAs
# =========================

VIEW_OPTIONS = [
    "Mission Control",
    "Real-data Pilot (MVP 2.0)",
    "Growth & Expansion (MVP 2.1)",
    "End-to-End Flow (MVP 2.1.1)",
    "Priority Queue",
    "My CSM Book",
    "Portfolio Insights",
    "Account Briefing",
    "Scenario Simulator",
    "CSM Playbook",
    "Methodology & Data Mapping",
]

NAV_LABELS = {
    "Mission Control": "01  ·  Mission Control",
    "Real-data Pilot (MVP 2.0)": "02  ·  Customer Value & Actions",
    "Growth & Expansion (MVP 2.1)": "03  ·  Growth Intelligence",
    "End-to-End Flow (MVP 2.1.1)": "04  ·  Revenue Lifecycle",
    "Priority Queue": "05  ·  Priority Queue",
    "My CSM Book": "06  ·  My CSM Book",
    "Portfolio Insights": "07  ·  Portfolio Insights",
    "Account Briefing": "08  ·  Account 360°",
    "Scenario Simulator": "09  ·  Scenario Simulator",
    "CSM Playbook": "10  ·  CS Playbook",
    "Methodology & Data Mapping": "11  ·  Data & Methodology",
}

QUEUE_FILTER_OPTIONS = [
    "All accounts",
    "P1/P2 accounts",
    "High-risk accounts",
    "MRR at risk",
    "Renewal risk",
    "Adoption gap",
    "Product friction",
    "Cross-functional routing",
    "Needs Customer Support",
    "Needs Account Executive / Sales",
    "Needs Product / Engineering",
]

if "active_view" not in st.session_state:
    st.session_state["active_view"] = "Mission Control"
if "queue_filter" not in st.session_state:
    st.session_state["queue_filter"] = "All accounts"


def go_to(view: str, queue_filter: str | None = None):
    """Navigate without discarding current portfolio scope or selected account."""
    st.session_state["active_view"] = view
    if view == "Account Briefing" and st.session_state.get("selected_account_id"):
        st.session_state["briefing_account"] = st.session_state["selected_account_id"]
    if queue_filter is not None:
        st.session_state["queue_filter"] = queue_filter
    st.rerun()


def apply_queue_filter(dataframe: pd.DataFrame, mode: str) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    if mode == "P1/P2 accounts":
        return dataframe[dataframe["priority"].isin(["P1", "P2"])]

    if mode == "High-risk accounts":
        return dataframe[dataframe["risk_level"] == "High"]

    if mode == "MRR at risk":
        return dataframe[dataframe["risk_level"] == "High"].sort_values("mrr", ascending=False)

    if mode == "Renewal risk":
        return dataframe[
            (dataframe["renewal_due_days"] < 60)
            & (dataframe["renewal_intent"] != "positive")
        ]

    if mode == "Adoption gap":
        return dataframe[
            (dataframe["tool_activity_score"] < 45)
            | (dataframe["feature_adoption_score"] < 45)
            | (dataframe["site_audit_last_run_days"] > 60)
            | (dataframe["position_tracking_campaigns_active"] == 0)
            | ((dataframe["ai_visibility_active"] == 1) & (dataframe["ai_visibility_prompts_tracked"] < 10))
        ]

    if mode == "Product friction":
        return dataframe[
            (dataframe["stuck_bugs_count"] > 0)
            | (dataframe["data_accuracy_bug_flag"] == 1)
            | (dataframe["unresolved_feedback_count"] >= 3)
            | (dataframe["roadmap_alignment_score"] < 40)
            | (dataframe["workaround_available_flag"] == 0)
        ]

    if mode == "Cross-functional routing":
        return dataframe[
            dataframe.apply(lambda r: not route_filter_matches(r, "CSM only"), axis=1)
        ]

    if mode in ["Needs Customer Support", "Needs Account Executive / Sales", "Needs Product / Engineering"]:
        return dataframe[dataframe.apply(lambda r: route_filter_matches(r, mode), axis=1)]

    return dataframe


def queue_context(mode: str) -> tuple[str, str]:
    contexts = {
        "All accounts": (
            "All accounts in the current portfolio view",
            "Use this when you want the complete book after applying sidebar filters. Start with P1/P2 before moving to Monitor accounts.",
        ),
        "P1/P2 accounts": (
            "Priority queue · P1/P2 accounts",
            "Start here. These are the accounts with the clearest risk, revenue impact or renewal sensitivity.",
        ),
        "High-risk accounts": (
            "Immediate-action queue · High-risk accounts",
            "Accounts already above the high-risk threshold. Review the top accounts first, then open Account 360° to confirm the recommended action and any cross-functional handoff.",
        ),
        "MRR at risk": (
            "Revenue-protection queue · High-risk MRR",
            "High-risk accounts ordered by monthly recurring revenue. Use this view to protect the largest commercial exposure first.",
        ),
        "Renewal risk": (
            "Renewal-risk queue",
            "Accounts where commercial timing, renewal intent, pricing, downgrade or grace-period signals need CSM + AE attention.",
        ),
        "Adoption gap": (
            "Adoption recovery queue",
            "Accounts that may be paying for value they are not fully realizing yet: weak usage, shallow feature adoption or inactive workflows.",
        ),
        "Product friction": (
            "Product friction queue",
            "Accounts where trust may be breaking because of stuck bugs, data accuracy flags, unresolved feedback or roadmap mismatch.",
        ),
        "Cross-functional routing": (
            "Cross-functional dependency queue",
            "Accounts where the CSM likely needs Customer Support, Account Executive / Sales, or Product / Engineering to unblock risk.",
        ),
        "Needs Customer Support": (
            "Customer Support handoff queue",
            "Accounts that need technical clarification, tool troubleshooting, limit explanation, navigation guidance or hard-block support.",
        ),
        "Needs Account Executive / Sales": (
            "Account Executive / Sales handoff queue",
            "Accounts where pricing, package fit, contract scope, renewal terms or commercial options should be reviewed.",
        ),
        "Needs Product / Engineering": (
            "Product / Engineering handoff queue",
            "Accounts where risk depends on roadmap fit, product gaps, stuck bugs or data-trust issues that CSM/Support cannot solve alone.",
        ),
    }
    return contexts.get(mode, contexts["All accounts"])




def account_format_label(dataframe: pd.DataFrame, account_id: str) -> str:
    """Readable label for account search widgets."""
    try:
        r = dataframe.loc[dataframe["account_id"] == account_id].iloc[0]
        return f"{r['account_id']} · {r['account_name']} · {r.get('csm_name', 'No CSM')} · {r['plan_family']} · {r['risk_level']} · {r['churn_probability']:.1%}"
    except Exception:
        return str(account_id)


def render_global_account_finder(all_accounts_df: pd.DataFrame):
    """One discoverable Account Finder, shared by every synthetic portfolio view."""
    if all_accounts_df.empty:
        return
    st.markdown('<div class="v171-finder-kicker">SEARCH & OPEN ACCOUNT</div>', unsafe_allow_html=True)
    with st.expander("⌕  Account Finder  ·  Search any account", expanded=False):
        st.caption("Search by company, ID, toolkit, CSM, segment or region. Current portfolio filters do not restrict account search.")
        c1, c2, c3 = st.columns([1.35, 1.65, 0.75])

        with c1:
            query = st.text_input(
                "Search account",
                placeholder="Example: SEM-03142, Signal Global, AI Visibility, Agency...",
                key="visible_account_query",
                label_visibility="collapsed",
            ).strip()

        search_base = all_accounts_df.copy()
        if query:
            q = query.lower()
            searchable_cols = [
                "account_id", "account_name", "plan_family", "plan_tier",
                "segment", "region", "industry", "customer_type", "risk_level", "priority", "csm_name", "csm_manager", "csm_focus_area", "book_tier"
            ]
            mask = pd.Series(False, index=search_base.index)
            for col in searchable_cols:
                if col in search_base.columns:
                    mask = mask | search_base[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
            matches = search_base[mask].sort_values("churn_probability", ascending=False).reset_index(drop=True)
        else:
            matches = search_base.sort_values("churn_probability", ascending=False).head(75).reset_index(drop=True)

        with c2:
            if matches.empty:
                st.selectbox(
                    "Matching accounts",
                    ["No matching account found"],
                    key="visible_account_empty",
                    label_visibility="collapsed",
                )
                selected_account = None
            else:
                options = matches["account_id"].tolist()[:200]
                current_selected = st.session_state.get("selected_account_id", options[0])
                selected_index = options.index(current_selected) if current_selected in options else 0
                if st.session_state.get("visible_account_select") not in options:
                    st.session_state["visible_account_select"] = options[selected_index]
                selected_account = st.selectbox(
                    "Matching accounts",
                    options,
                    index=selected_index,
                    format_func=lambda x: account_format_label(search_base, x),
                    key="visible_account_select",
                    label_visibility="collapsed",
                )

        with c3:
            if st.button("Open 360°", key="visible_open_account_360", use_container_width=True):
                if selected_account:
                    st.session_state["selected_account_id"] = selected_account
                    st.session_state["briefing_account"] = selected_account
                    st.session_state["active_view"] = "Account Briefing"
                    st.rerun()

        if query and not matches.empty:
            st.caption(f"{len(matches)} matching account(s). Select one and open Account 360°.")
        elif query and matches.empty:
            st.caption("No account matched your search. Try account ID, company name, package family, segment or region.")



def render_csm_book_overview(book_df: pd.DataFrame, all_df: pd.DataFrame):
    """CSM book-of-business command view."""
    if book_df.empty:
        st.info("No accounts match the selected CSM filters.")
        return

    selected_csm = book_df["csm_name"].iloc[0] if "csm_name" in book_df.columns and book_df["csm_name"].nunique() == 1 else "Selected CSM book"
    owner_context = ""
    if "csm_name" in book_df.columns and book_df["csm_name"].nunique() == 1:
        r = book_df.iloc[0]
        owner_context = (
            f'<span class="csm-owner-chip">{html.escape(str(r.get("csm_region", "")))}</span>'
            f'<span class="csm-owner-chip">{html.escape(str(r.get("csm_timezone", "")))}</span>'
            f'<span class="csm-owner-chip">Focus: {html.escape(str(r.get("csm_focus_area", "")))}</span>'
            f'<span class="csm-owner-chip">{html.escape(str(r.get("book_tier", "")))}</span>'
            f'<span class="csm-owner-chip">Manager: {html.escape(str(r.get("csm_manager", "")))}</span>'
        )
    else:
        owner_context = f'<span class="csm-owner-chip">{book_df["csm_name"].nunique() if "csm_name" in book_df.columns else 0} CSMs selected</span>'

    avg = book_df["churn_probability"].mean()
    high = int((book_df["risk_level"] == "High").sum())
    p1p2 = int(book_df["priority"].isin(["P1", "P2"]).sum())
    mrr = book_df.loc[book_df["risk_level"] == "High", "mrr"].sum()
    renewals = int(((book_df["renewal_due_days"] < 60) & (book_df["renewal_intent"] != "positive")).sum())
    needs_support = int(book_df.apply(lambda r: route_filter_matches(r, "Needs Customer Support"), axis=1).sum())
    needs_sales = int(book_df.apply(lambda r: route_filter_matches(r, "Needs Account Executive / Sales"), axis=1).sum())
    needs_product = int(book_df.apply(lambda r: route_filter_matches(r, "Needs Product / Engineering"), axis=1).sum())

    fam = pd.DataFrame([family_risk_scores(r) for _, r in book_df.iterrows()])
    theme = fam.mean().sort_values(ascending=False).index[0] if not fam.empty else "General health risk"

    st.markdown(
        f'<div class="csm-book-hero">'
        f'<div class="csm-book-title">{html.escape(str(selected_csm))} · Book of Business</div>'
        f'<div>{owner_context}</div>'
        f'<div class="csm-book-sub">This view turns the selected CSM portfolio into a guided daily brief: where risk is concentrated, which accounts should be reviewed first, and which teams may need to help unblock customer risk.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card_html("Assigned accounts", f"{len(book_df):,}", f"Average churn probability: {avg:.1%}")
    with k2: metric_card_html("P1/P2 accounts", f"{p1p2}", "Start here for daily action review.")
    with k3: metric_card_html("MRR at risk", _fmt_money(mrr), "High-risk revenue in this book.")
    with k4: metric_card_html("Renewal pressure", f"{renewals}", "Renewals <60 days with weak or non-positive intent.")

    section_header("Today’s CSM focus", "A simple story for the selected book before drilling into accounts.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="csm-action-card"><div class="csm-action-title">Main risk theme</div><div class="csm-action-body">{html.escape(str(theme))} is the clearest pressure point in this book. Review accounts where this theme overlaps with P1/P2 priority.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="csm-action-card"><div class="csm-action-title">Cross-functional blockers</div><div class="csm-action-body">Support needed: {needs_support}. Sales needed: {needs_sales}. Product/Engineering needed: {needs_product}. Use routing cards before customer follow-up.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="csm-action-card"><div class="csm-action-title">Recommended next step</div><div class="csm-action-body">Open the Priority Queue filtered by this book, start with P1/P2 accounts, then use Account 360° to prepare the CSM action and internal handoff.</div></div>', unsafe_allow_html=True)

    section_header("CSM priority accounts", "The highest-priority accounts in the selected book.")
    render_priority_cards(book_df.sort_values(["priority", "churn_probability"], ascending=[True, False]), max_cards=8)

    with st.expander("Manager view: risk distribution by CSM", expanded=False):
        if "csm_name" in all_df.columns:
            manager_df = all_df.groupby("csm_name").agg(
                assigned_accounts=("account_id", "count"),
                avg_churn_probability=("churn_probability", "mean"),
                high_risk_accounts=("risk_level", lambda s: int((s == "High").sum())),
                mrr_at_risk=("mrr", lambda s: float(all_df.loc[s.index][all_df.loc[s.index, "risk_level"] == "High"]["mrr"].sum())),
            ).reset_index().sort_values(["high_risk_accounts", "avg_churn_probability"], ascending=False)
            display = manager_df.copy()
            display["avg_churn_probability"] = display["avg_churn_probability"].map(lambda x: f"{x:.1%}")
            display["mrr_at_risk"] = display["mrr_at_risk"].map(lambda x: f"${x:,.0f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

def render_return_control(current_view: str, current_queue_filter: str, current_queue_title: str):
    """Compact breadcrumbs and a meaningful back action, without duplicated introductions."""
    if current_view == "Mission Control":
        return
    current_label = {
        "Account Briefing": "Account 360°",
        "Real-data Pilot (MVP 2.0)": "Customer Value",
        "Growth & Expansion (MVP 2.1)": "Growth Intelligence",
        "End-to-End Flow (MVP 2.1.1)": "Revenue Lifecycle",
    }.get(current_view, current_view)
    trail = ["Mission Control"]
    if current_view == "Account Briefing":
        trail.append(current_queue_filter if current_queue_filter != "All accounts" else "Portfolio")
        aid = st.session_state.get("selected_account_id")
        if aid:
            trail.append(str(aid))
    elif current_view == "Priority Queue":
        trail.append(current_queue_filter)
    else:
        trail.append(current_label)
    st.markdown(
        '<div class="v171-breadcrumb"><span>WORKSPACE</span> ' +
        ' <span class="v171-crumb-sep">/</span> '.join(html.escape(x) for x in trail) +
        '</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.35, 4.2])
    with col1:
        if st.button('← Mission Control', key='v171_back_'+re.sub(r'[^a-z0-9]','_', current_view.lower()), use_container_width=True):
            go_to('Mission Control')
    with col2:
        if current_view == "Account Briefing":
            st.caption("Your portfolio scope and queue focus stay available when you return.")
        elif current_view == "Priority Queue":
            st.caption(f"Queue focus: {current_queue_filter} · Choose an account to investigate.")
        else:
            st.caption("Follow the customer journey and keep supporting evidence linked to each decision.")


def portfolio_360_html():
    """Immersive but readable portfolio 360 visual for Mission Control."""
    # Normalize values for the radial bars. These are visual intensities, not new model outputs.
    renewal_intensity = min(100, max(0, primary_theme_score))
    adoption_intensity = min(100, max(0, low_usage_count / max(len(filtered), 1) * 100))
    friction_intensity = min(100, max(0, product_friction_count / max(len(filtered), 1) * 100))
    dependency_intensity = min(100, max(0, (support_needed_count + sales_needed_count + product_needed_count) / max(len(filtered), 1) * 100))
    revenue_intensity = min(100, max(0, mrr_at_risk / max(filtered["mrr"].sum(), 1) * 100))

    def node(label, value, left, top, color, queue):
        return (
            f'<div class="orbit-node" style="left:{left}; top:{top}; --node-color:{color};" '
            f'title="Use the matching action button below to open {html.escape(queue)}.">' 
            f'<span class="orbit-dot"></span><span class="orbit-label">{html.escape(label)}</span>'
            f'<span class="orbit-value">{html.escape(value)}</span></div>'
        )

    nodes = "".join([
        node("Renewal", f"{renewal_pressure_30}", "64%", "13%", "#DC2626", "Renewal risk"),
        node("Adoption", f"{low_usage_count}", "76%", "55%", "#0F766E", "Adoption gap"),
        node("Friction", f"{product_friction_count}", "41%", "76%", "#D97706", "Product friction"),
        node("Teams", f"{support_needed_count + sales_needed_count + product_needed_count}", "15%", "53%", "#6D28D9", "Cross-functional routing"),
        node("MRR", _fmt_money(mrr_at_risk), "24%", "14%", "#2563EB", "P1/P2 accounts"),
    ])

    return f"""
    <div class="portfolio-360-shell">
        <div class="portfolio-360-copy">
            <div class="immersive-eyebrow">Portfolio 360°</div>
            <div class="immersive-title">A visual command hub for what needs action first.</div>
            <div class="immersive-body">
                The center shows the overall portfolio risk. The surrounding orbit highlights where the pressure is coming from:
                commercial timing, adoption gaps, product friction, revenue exposure and cross-functional blockers.
            </div>
            <div class="immersive-takeaway"><strong>Recommended starting point:</strong> review P1/P2 accounts where high risk overlaps with renewal pressure, product friction or another department dependency.</div>
        </div>
        <div class="portfolio-360-stage">
            <div class="orbit-ring ring-1"></div>
            <div class="orbit-ring ring-2"></div>
            <div class="orbit-ring ring-3"></div>
            <div class="risk-core">
                <div class="core-label">Average risk</div>
                <div class="core-value">{avg_churn:.1%}</div>
                <div class="core-note">{len(filtered):,} accounts</div>
            </div>
            {nodes}
        </div>
        <div class="portfolio-360-legend">
            <div class="legend-row"><span class="legend-chip red"></span><span>Renewal pressure</span><strong>{renewal_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip teal"></span><span>Adoption gap</span><strong>{adoption_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip amber"></span><span>Product friction</span><strong>{friction_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip purple"></span><span>Team dependencies</span><strong>{dependency_intensity:.0f}/100</strong></div>
            <div class="legend-row"><span class="legend-chip blue"></span><span>MRR at risk share</span><strong>{revenue_intensity:.0f}/100</strong></div>
        </div>
    </div>
    """


def render_portfolio_360_actions():
    """Real Streamlit CTAs paired with the Portfolio 360° visual nodes."""
    team_count = support_needed_count + sales_needed_count + product_needed_count
    high_risk_count = int((filtered["risk_level"] == "High").sum())

    st.markdown(
        '<div class="portfolio360-action-intro"><strong>Open a work queue directly from the 360° view.</strong> '
        'Each action applies the relevant filter and keeps the CSM workflow focused.</div>',
        unsafe_allow_html=True,
    )

    actions = [
        ("🚨 High-risk accounts", f"{high_risk_count} need immediate review", "High-risk accounts", "portfolio360_high_risk"),
        ("💼 Renewal", f"{renewal_pressure_30} with renewal pressure", "Renewal risk", "portfolio360_renewal"),
        ("📈 Adoption", f"{low_usage_count} need value recovery", "Adoption gap", "portfolio360_adoption"),
        ("🛠 Friction", f"{product_friction_count} show trust friction", "Product friction", "portfolio360_friction"),
        ("🤝 Teams", f"{team_count} may need another team", "Cross-functional routing", "portfolio360_teams"),
        ("💰 MRR at risk", _fmt_money(mrr_at_risk), "MRR at risk", "portfolio360_mrr"),
    ]

    cols = st.columns(6)
    for col, (label, caption, queue, key) in zip(cols, actions):
        with col:
            st.caption(caption)
            if st.button(label, key=key, use_container_width=True, help=f"Open the {queue} queue"):
                go_to("Priority Queue", queue)


def render_immediate_action_accounts():
    """Visible individual-account launchers for the most urgent high-risk accounts."""
    high_risk = filtered[filtered["risk_level"] == "High"].sort_values(
        ["churn_probability", "mrr"], ascending=[False, False]
    )

    section_header(
        "Immediate-action accounts",
        "Start with individual high-risk accounts. Open Account 360° for the full story, recommended actions and required handoffs.",
    )

    top = high_risk.head(4)
    left, right = st.columns([3, 1])
    with left:
        if high_risk.empty:
            st.info("No accounts are currently classified as high risk in the selected portfolio view.")
        else:
            st.markdown(
                f'<div class="briefing-note"><strong>{len(high_risk)} high-risk accounts are visible in this portfolio.</strong> '
                'The cards below show the most urgent ones by churn probability and revenue exposure.</div>',
                unsafe_allow_html=True,
            )
    with right:
        if st.button("View all high-risk accounts", key="view_all_high_risk_from_mission", use_container_width=True):
            go_to("Priority Queue", "High-risk accounts")

    if top.empty:
        return

    cards = st.columns(len(top))
    for col, (_, row) in zip(cards, top.iterrows()):
        account_name = html.escape(str(row.get("account_name", row.get("account_id", "Account"))))
        account_id = html.escape(str(row.get("account_id", "")))
        main_theme = html.escape(str(dominant_account_theme(row)))
        with col:
            st.markdown(
                f'<div class="immediate-account-card">'
                f'<div class="immediate-account-top"><span class="immediate-risk-dot"></span><span>HIGH RISK</span></div>'
                f'<div class="immediate-account-name">{account_name}</div>'
                f'<div class="immediate-account-id">{account_id}</div>'
                f'<div class="immediate-account-metric">{float(row.get("churn_probability", 0)):.1%}</div>'
                f'<div class="immediate-account-sub">{_fmt_money(row.get("mrr", 0))} MRR · renewal in {int(row.get("renewal_due_days", 0))} days</div>'
                f'<div class="immediate-account-theme">{main_theme}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Open Account 360°", key=f"mission_high_risk_{row['account_id']}", use_container_width=True):
                st.session_state["selected_account_id"] = row["account_id"]
                go_to("Account Briefing")



def immersive_priority_landscape_html(dataframe):
    """Priority lanes with a 3D-lite look. Keeps full account cards available below."""
    lanes = []
    lane_data = [
        ("P1", "Act today", "Front lane: immediate attention", "#DC2626"),
        ("P2", "Review this week", "Second lane: active follow-up", "#D97706"),
        ("P3", "Stabilize", "Third lane: monitor and prevent escalation", "#2563EB"),
        ("Monitor", "Regular cadence", "Back lane: keep value visible", "#0F766E"),
    ]
    for priority, label, note, color in lane_data:
        subset = dataframe[dataframe["priority"] == priority]
        top = subset.sort_values("churn_probability", ascending=False).head(1)
        if top.empty:
            headline = "No accounts"
            details = "Nothing in this lane for the selected view."
            risk = "—"
        else:
            r = top.iloc[0]
            headline = str(r.get("account_name", r.get("account_id", "Account")))
            details = f'{r.get("plan_family", "")} · renewal in {int(r.get("renewal_due_days", 0))} days · {_fmt_money(r.get("mrr", 0))} MRR'
            risk = f'{float(r.get("churn_probability", 0)):.1%}'
        lanes.append(
            f'<div class="priority-3d-lane" style="--lane-color:{color};">'
            f'<div class="lane-top"><span>{html.escape(priority)}</span><strong>{len(subset)}</strong></div>'
            f'<div class="lane-label">{html.escape(label)}</div>'
            f'<div class="lane-risk">{html.escape(risk)}</div>'
            f'<div class="lane-account">{html.escape(headline)}</div>'
            f'<div class="lane-details">{html.escape(details)}</div>'
            f'<div class="lane-note">{html.escape(note)}</div>'
            f'</div>'
        )
    return f'<div class="priority-3d-board">{"".join(lanes)}</div>'


def render_clickable_mission_cards():
    cards = [
        {
            "kicker": "Portfolio risk",
            "title": "How healthy is my book?",
            "value": f"{avg_churn:.1%}",
            "body": f"Average churn probability across {len(filtered):,} selected accounts. Main risk theme: {primary_theme}.",
            "button": "🚀 Open Priority Queue",
            "tone": "blue",
            "queue": "P1/P2 accounts",
            "icon": "◎",
        },
        {
            "kicker": "Revenue & renewal",
            "title": "Where is commercial risk?",
            "value": _fmt_money(mrr_at_risk),
            "body": f"{renewal_pressure_30} accounts renew in <30 days with weak or non-positive intent.",
            "button": "💼 Open renewal-risk details",
            "tone": "red",
            "queue": "Renewal risk",
            "icon": "$",
        },
        {
            "kicker": "Adoption gap",
            "title": "Who pays but underuses?",
            "value": f"{low_usage_count}",
            "body": "Accounts with low product activity, shallow feature adoption or inactive Semrush-like workflows.",
            "button": "📈 Open adoption recovery list",
            "tone": "teal",
            "queue": "Adoption gap",
            "icon": "↗",
        },
        {
            "kicker": "Product friction",
            "title": "Where is trust breaking?",
            "value": f"{product_friction_count}",
            "body": "Accounts with stuck bugs, data accuracy flags, unresolved feedback or roadmap mismatch.",
            "button": "🛠️ Open product friction cases",
            "tone": "amber",
            "queue": "Product friction",
            "icon": "⚙",
        },
        {
            "kicker": "Team dependencies",
            "title": "Who needs to help?",
            "value": f"{support_needed_count + sales_needed_count + product_needed_count}",
            "body": "Accounts where Support, Sales or Product may need to help the CSM unblock risk.",
            "button": "🤝 Review cross-functional routing",
            "tone": "purple",
            "queue": "Cross-functional routing",
            "icon": "◌",
        },
    ]

    tone_color = {"blue": "#2563EB", "teal": "#0F766E", "red": "#DC2626", "amber": "#D97706", "purple": "#6D28D9"}
    cols = st.columns(5)
    for i, card in enumerate(cards):
        color = tone_color.get(card["tone"], "#2563EB")
        with cols[i]:
            st.markdown(
                f'<div class="mission-card immersive-card" style="--accent:{color}; border-top:5px solid {color};">'
                f'<div class="card-depth-glow"></div>'
                f'<div class="mission-icon-orb">{html.escape(card["icon"])}</div>'
                f'<div class="mission-kicker">{html.escape(card["kicker"])}</div>'
                f'<div class="mission-title">{html.escape(card["title"])}</div>'
                f'<div class="mission-value">{html.escape(card["value"])}</div>'
                f'<div class="mission-body">{html.escape(card["body"])}</div>'
                f'<div class="card-direction">Focus queue → {html.escape(card["queue"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(card["button"], key=f"mission_cta_{i}", use_container_width=True):
                go_to("Priority Queue", card["queue"])



# Immersive V7 corporate 3D-lite visual layer
st.markdown(
    """
    <style>
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.18);
        box-shadow: 0 28px 70px rgba(7,20,39,.28);
    }
    .hero:after {
        content:"";
        position:absolute;
        right:-90px;
        top:-90px;
        width:310px;
        height:310px;
        border-radius:999px;
        background: radial-gradient(circle, rgba(167,243,208,.30), rgba(37,99,235,.08) 45%, transparent 70%);
        filter: blur(.2px);
    }
    .mission-card.immersive-card {
        position:relative;
        overflow:hidden;
        transform-style: preserve-3d;
        box-shadow: 0 20px 45px rgba(15,23,42,.10), inset 0 1px 0 rgba(255,255,255,.8);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        min-height: 270px;
    }
    .mission-card.immersive-card:hover {
        transform: perspective(900px) rotateX(2deg) rotateY(-2deg) translateY(-6px);
        box-shadow: 0 30px 70px rgba(15,23,42,.17), 0 0 0 1px rgba(37,99,235,.08);
    }
    .card-depth-glow {
        position:absolute;
        right:-55px;
        top:-55px;
        width:145px;
        height:145px;
        border-radius:999px;
        background: radial-gradient(circle, color-mix(in srgb, var(--accent), white 18%), transparent 62%);
        opacity:.25;
    }
    .mission-icon-orb {
        width:42px;
        height:42px;
        border-radius:16px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-size:20px;
        font-weight:950;
        background: linear-gradient(135deg, var(--accent), #071427);
        box-shadow: 0 12px 25px color-mix(in srgb, var(--accent), transparent 70%);
        margin-bottom:12px;
    }
    .card-direction {
        margin-top:12px;
        background:linear-gradient(135deg,#F8FAFC,#EFF6FF);
        border:1px solid #DBEAFE;
        color:#334155;
        padding:9px 10px;
        border-radius:14px;
        font-size:12px;
        font-weight:850;
    }
    .portfolio-360-shell {
        display:grid;
        grid-template-columns: 1.05fr 1.25fr .88fr;
        gap:18px;
        align-items:center;
        background: radial-gradient(circle at 50% 40%, rgba(37,99,235,.10), transparent 34%), linear-gradient(135deg, #FFFFFF, #F8FAFC 48%, #ECFEFF);
        border:1px solid #BFE7F6;
        border-radius:30px;
        padding:24px;
        box-shadow:0 26px 65px rgba(15,23,42,.10);
        margin: 14px 0 22px;
        overflow:hidden;
        position:relative;
    }
    .portfolio-360-shell:before {
        content:"";
        position:absolute;
        inset:-2px;
        background:linear-gradient(120deg, rgba(37,99,235,.10), transparent, rgba(15,118,110,.12));
        pointer-events:none;
    }
    .portfolio-360-copy, .portfolio-360-stage, .portfolio-360-legend { position:relative; z-index:1; }
    .immersive-eyebrow { color:#0F766E; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.10em; margin-bottom:8px; }
    .immersive-title { color:#071427; font-size:26px; font-weight:950; line-height:1.08; margin-bottom:10px; }
    .immersive-body { color:#475569; font-size:14px; line-height:1.55; margin-bottom:12px; }
    .immersive-takeaway { background:#FFFFFF; border:1px solid #DDEBF7; border-radius:18px; padding:13px 14px; color:#334155; font-size:13px; line-height:1.45; }
    .portfolio-360-stage {
        height:330px;
        min-width:360px;
        perspective: 1000px;
        transform-style: preserve-3d;
    }
    .orbit-ring {
        position:absolute;
        left:50%;
        top:50%;
        transform:translate(-50%, -50%) rotateX(63deg) rotateZ(-12deg);
        border:1px solid rgba(37,99,235,.23);
        border-radius:50%;
        box-shadow: 0 0 38px rgba(37,99,235,.08);
    }
    .ring-1 { width:145px; height:145px; }
    .ring-2 { width:230px; height:230px; border-color:rgba(15,118,110,.25); }
    .ring-3 { width:320px; height:320px; border-color:rgba(109,40,217,.14); }
    .risk-core {
        position:absolute;
        left:50%;
        top:50%;
        transform:translate(-50%, -50%);
        width:145px;
        height:145px;
        border-radius:50%;
        background: radial-gradient(circle at 35% 30%, #FFFFFF, #EFF6FF 52%, #DBEAFE);
        border:1px solid #BFDBFE;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        box-shadow: 0 22px 45px rgba(37,99,235,.18);
    }
    .core-label { color:#64748B; font-size:11px; font-weight:950; text-transform:uppercase; letter-spacing:.07em; }
    .core-value { color:#071427; font-size:31px; font-weight:950; line-height:1; margin:6px 0; }
    .core-note { color:#64748B; font-size:12px; font-weight:800; }
    .orbit-node {
        position:absolute;
        border:0;
        background:white;
        border-radius:18px;
        padding:10px 12px;
        min-width:116px;
        text-align:left;
        box-shadow: 0 14px 34px rgba(15,23,42,.13);
        border:1px solid #E2E8F0;
        transform:translate(-50%, -50%);
    }
    .orbit-dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:var(--node-color); margin-right:6px; box-shadow:0 0 0 5px color-mix(in srgb, var(--node-color), transparent 82%); }
    .orbit-label { color:#334155; font-size:12px; font-weight:900; }
    .orbit-value { display:block; margin-top:5px; color:#071427; font-size:18px; font-weight:950; }
    .portfolio-360-legend { background:white; border:1px solid #E2E8F0; border-radius:22px; padding:16px; box-shadow:0 14px 34px rgba(15,23,42,.06); }
    .legend-row { display:grid; grid-template-columns:18px 1fr auto; gap:8px; align-items:center; padding:10px 0; border-bottom:1px solid #EEF2F7; color:#475569; font-size:13px; }
    .legend-row:last-child { border-bottom:0; }
    .legend-row strong { color:#071427; }
    .legend-chip { width:12px; height:12px; border-radius:999px; display:inline-block; }
    .legend-chip.red { background:#DC2626; } .legend-chip.teal { background:#0F766E; } .legend-chip.amber { background:#D97706; } .legend-chip.purple { background:#6D28D9; } .legend-chip.blue { background:#2563EB; }
    .priority-3d-board {
        display:grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap:16px;
        margin: 14px 0 22px;
        perspective: 1200px;
    }
    .priority-3d-lane {
        position:relative;
        background: linear-gradient(145deg,#FFFFFF,#F8FAFC);
        border:1px solid #E2E8F0;
        border-top:5px solid var(--lane-color);
        border-radius:24px;
        padding:18px;
        min-height:245px;
        box-shadow:0 22px 52px rgba(15,23,42,.11);
        transform: rotateX(1.5deg);
        overflow:hidden;
    }
    .priority-3d-lane:after {
        content:"";
        position:absolute;
        width:120px;
        height:120px;
        right:-45px;
        top:-45px;
        border-radius:999px;
        background: radial-gradient(circle, color-mix(in srgb, var(--lane-color), white 12%), transparent 68%);
        opacity:.20;
    }
    .lane-top { display:flex; justify-content:space-between; align-items:center; color:#64748B; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px; }
    .lane-top strong { color:var(--lane-color); font-size:24px; }
    .lane-label { color:#071427; font-size:20px; font-weight:950; margin-bottom:12px; }
    .lane-risk { color:#071427; font-size:34px; font-weight:950; line-height:1; margin-bottom:10px; }
    .lane-account { color:#334155; font-size:14px; font-weight:900; line-height:1.25; margin-bottom:6px; }
    .lane-details, .lane-note { color:#64748B; font-size:12px; line-height:1.38; }
    .lane-note { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px; padding:9px 10px; margin-top:10px; }
    .account-queue-card, .question-card, .scenario-card, .routing-card, .signal-brief-card {
        transition: transform .16s ease, box-shadow .16s ease;
    }
    .account-queue-card:hover, .question-card:hover, .scenario-card:hover, .routing-card:hover, .signal-brief-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 22px 52px rgba(15,23,42,.12);
    }
    @media (max-width:1100px){
        .portfolio-360-shell{grid-template-columns:1fr;}
        .portfolio-360-stage{min-width:0;}
        .priority-3d-board{grid-template-columns:1fr 1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# V8 Account 360° immersive CSS layer
st.markdown(
    """
    <style>
    .account360-shell { display:grid; grid-template-columns: 1.25fr 1.05fr .85fr; gap:16px; margin: 14px 0 22px; perspective:1200px; }
    .account360-core-card, .account360-priority-card, .account360-commercial-card {
        background: radial-gradient(circle at 100% 0%, rgba(15,118,110,.12), transparent 28%), linear-gradient(145deg,#FFFFFF,#F8FAFC);
        border:1px solid #E2E8F0; border-radius:28px; padding:24px; box-shadow:0 24px 58px rgba(15,23,42,.10); position:relative; overflow:hidden;
    }
    .account360-core-card { transform: perspective(900px) rotateX(.8deg); }
    .account360-priority-card { border-top:5px solid #2563EB; }
    .account360-commercial-card { border-top:5px solid #0F766E; }
    .account360-eyebrow, .account360-kicker { color:#0F766E; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.09em; margin-bottom:8px; }
    .account360-title { color:#071427; font-size:34px; font-weight:950; line-height:1.08; margin-bottom:6px; }
    .account360-sub { color:#64748B; font-size:14px; line-height:1.4; margin-bottom:12px; }
    .account360-risk-row { display:flex; align-items:center; gap:12px; margin: 10px 0 12px; }
    .account360-prob { color:#071427; font-size:36px; font-weight:950; line-height:1; }
    .account360-story { background:#FFFFFF; border:1px solid #DDEBF7; border-radius:18px; padding:14px; color:#334155; font-size:14px; line-height:1.5; margin:12px 0; }
    .account360-action-title { color:#071427; font-size:20px; font-weight:950; line-height:1.25; margin-bottom:12px; }
    .account360-action-meta { color:#475569; font-size:13px; line-height:1.45; margin-top:8px; }
    .account360-metric-row { display:flex; justify-content:space-between; align-items:center; padding:11px 0; border-bottom:1px solid #EEF2F7; color:#64748B; font-size:13px; }
    .account360-metric-row:last-child { border-bottom:0; }
    .account360-metric-row strong { color:#071427; font-size:15px; }
    .account360-radar-shell { display:grid; grid-template-columns: 1.05fr 1fr; gap:18px; align-items:center; background:linear-gradient(135deg,#FFFFFF,#F8FAFC 50%,#ECFEFF); border:1px solid #BFE7F6; border-radius:30px; padding:24px; box-shadow:0 26px 65px rgba(15,23,42,.10); margin: 14px 0 22px; }
    .account360-radar-stage { height:390px; min-width:390px; position:relative; perspective:1000px; }
    .account360-ring { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%) rotateX(62deg) rotateZ(-9deg); border-radius:50%; border:1px solid rgba(37,99,235,.24); box-shadow:0 0 38px rgba(37,99,235,.08); }
    .account360-ring.ring-a { width:150px; height:150px; }
    .account360-ring.ring-b { width:250px; height:250px; border-color:rgba(15,118,110,.25); }
    .account360-ring.ring-c { width:350px; height:350px; border-color:rgba(109,40,217,.18); }
    .account360-center { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:150px; height:150px; border-radius:50%; background:radial-gradient(circle at 35% 28%,#FFFFFF,#EFF6FF 55%,#DBEAFE); border:1px solid #BFDBFE; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 22px 45px rgba(37,99,235,.18); }
    .account360-node { position:absolute; transform:translate(-50%,-50%); background:white; border:1px solid #E2E8F0; border-top:4px solid var(--node-color); border-radius:18px; min-width:126px; padding:10px 11px; box-shadow:0 16px 34px rgba(15,23,42,.13); }
    .account360-node.high { box-shadow:0 16px 38px rgba(220,38,38,.18); } .account360-node.medium { box-shadow:0 16px 38px rgba(217,119,6,.16); }
    .account360-node-label { display:block; color:#334155; font-size:11px; font-weight:950; line-height:1.2; }
    .account360-node-value { display:block; color:#071427; font-size:22px; font-weight:950; margin-top:5px; }
    .account360-radar-details { background:white; border:1px solid #E2E8F0; border-radius:24px; padding:20px; box-shadow:0 14px 34px rgba(15,23,42,.06); }
    .account360-details-title { color:#071427; font-size:22px; font-weight:950; margin-bottom:12px; }
    .account360-radar-row { display:grid; grid-template-columns: 1.6fr .9fr 1fr .9fr; gap:10px; align-items:center; padding:11px 0; border-bottom:1px solid #EEF2F7; color:#475569; font-size:12px; }
    .account360-radar-row:last-child { border-bottom:0; }
    .account360-radar-row strong { color:#071427; font-size:13px; }
    .account360-status { display:inline-block; text-align:center; border-radius:999px; padding:6px 8px; font-size:11px; font-weight:950; }
    .account360-status.high { background:#FEE2E2; color:#991B1B; } .account360-status.medium { background:#FEF3C7; color:#92400E; } .account360-status.low { background:#DCFCE7; color:#166534; }
    .account360-mini-bar { height:11px; border-radius:999px; background:#E2E8F0; overflow:hidden; }
    .account360-mini-bar div { height:11px; border-radius:999px; background:linear-gradient(90deg,#2563EB,#0F766E); }
    .account360-owner { color:#64748B; font-size:11px; font-weight:850; }
    .account360-lens-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:12px 0 22px; }
    .account360-lens-card { background:white; border:1px solid #E2E8F0; border-radius:24px; padding:20px; box-shadow:0 18px 42px rgba(15,23,42,.08); border-top:5px solid #0F766E; min-height:225px; transition:transform .16s ease, box-shadow .16s ease; }
    .account360-lens-card:hover { transform:translateY(-5px); box-shadow:0 28px 62px rgba(15,23,42,.13); }
    .account360-lens-card.high { border-top-color:#DC2626; } .account360-lens-card.medium { border-top-color:#D97706; } .account360-lens-card.low { border-top-color:#0F766E; }
    .account360-lens-question { color:#475569; font-size:13px; font-weight:950; text-transform:uppercase; letter-spacing:.05em; line-height:1.25; margin-bottom:10px; }
    .account360-lens-status { color:#071427; font-size:25px; font-weight:950; margin-bottom:10px; }
    .account360-lens-cta { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:16px; padding:12px; color:#334155; font-size:13px; line-height:1.45; margin-top:12px; }
    .account360-action-shell { display:grid; grid-template-columns: .9fr 1.5fr; gap:16px; margin:12px 0 22px; }
    .account360-action-main { background:radial-gradient(circle at 100% 0%,rgba(37,99,235,.16),transparent 30%),linear-gradient(145deg,#FFFFFF,#EFF6FF); border:1px solid #BFDBFE; border-radius:26px; padding:22px; box-shadow:0 18px 42px rgba(15,23,42,.08); }
    .account360-action-head { color:#071427; font-size:26px; font-weight:950; line-height:1.12; margin-bottom:12px; }
    .account360-action-text { color:#334155; font-size:14px; line-height:1.55; background:white; border:1px solid #DBEAFE; border-radius:18px; padding:14px; }
    .account360-route-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    .account360-route-card { background:white; border:1px solid #E2E8F0; border-radius:22px; padding:18px; box-shadow:0 14px 34px rgba(15,23,42,.07); border-top:5px solid #64748B; }
    .account360-route-card.support { border-top-color:#2563EB; } .account360-route-card.sales { border-top-color:#7C3AED; } .account360-route-card.product { border-top-color:#0F766E; }
    .account360-route-team { color:#071427; font-size:17px; font-weight:950; margin-bottom:9px; }
    .account360-route-label { color:#64748B; font-size:10px; font-weight:950; text-transform:uppercase; letter-spacing:.08em; margin:9px 0 4px; }
    .account360-route-body { color:#475569; font-size:12px; line-height:1.42; }
    .account360-search-panel { background:linear-gradient(135deg,#FFFFFF,#F0FDFA); border:1px solid #99F6E4; border-radius:20px; padding:15px; margin:10px 0 16px; box-shadow:0 10px 26px rgba(15,23,42,.06); }
    .account360-search-title { color:#0F766E; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
    .account360-search-body { color:#475569; font-size:12px; line-height:1.35; }
    @media (max-width:1100px){ .account360-shell, .account360-radar-shell, .account360-action-shell { grid-template-columns:1fr; } .account360-lens-grid, .account360-route-grid { grid-template-columns:1fr; } .account360-radar-stage { min-width:0; } }
    </style>
    """,
    unsafe_allow_html=True,
)



# V9 visible global account finder CSS
st.markdown(
    """
    <style>
    .global-finder-shell {
        margin: 14px 0 12px;
        padding: 18px 20px;
        border-radius: 24px;
        border: 1px solid rgba(37,99,235,.18);
        background:
            radial-gradient(circle at 98% 0%, rgba(37,99,235,.16), transparent 26%),
            linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 52%, #F0FDFA 100%);
        box-shadow: 0 18px 44px rgba(15, 23, 42, .08);
    }
    .global-finder-kicker {
        color: #0F766E;
        font-size: 11px;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .10em;
        margin-bottom: 5px;
    }
    .global-finder-title {
        color: #071427;
        font-size: 21px;
        line-height: 1.15;
        font-weight: 950;
        margin-bottom: 6px;
    }
    .global-finder-body {
        color: #475569;
        font-size: 13px;
        line-height: 1.45;
        max-width: 1180px;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 16px !important;
        border: 1px solid #CBD5E1 !important;
        background: #FFFFFF !important;
        min-height: 48px;
        font-weight: 750;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 16px !important;
        min-height: 48px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("""
<style>
/* V17.1 – task-first navigation and decision hierarchy */
.v171-sidebar-brand {display:flex; align-items:center; gap:11px; margin: 4px 0 18px; padding:12px 11px; border:1px solid rgba(45,94,156,.18); background:linear-gradient(135deg,#0E2138,#183F60); color:white; border-radius:17px; box-shadow:0 10px 28px rgba(9,26,43,.17);}
.v171-sidebar-brand strong {display:block; font-family:'Space Grotesk',Inter,sans-serif; font-size:15px; color:#F4F8FD; letter-spacing:-.025em;}
.v171-sidebar-brand small {display:block; font-size:9px; letter-spacing:.13em; font-weight:800; color:#ADD6E0; padding-top:2px;}
.v171-sidebar-monogram {width:34px; height:34px; display:flex; align-items:center; justify-content:center; border-radius:11px; background:linear-gradient(135deg,#66D1C6,#5384E9); color:#071427; font-size:12px; font-weight:950;}
.v171-nav-group,.v171-filter-intro {color:#6D7E95; font-size:10px; font-weight:900; letter-spacing:.16em; margin:13px 0 5px;}
section[data-testid="stSidebar"] .stButton>button {min-height:37px; padding:7px 13px!important; font-size:12px!important; text-align:left!important; border-radius:11px!important; justify-content:flex-start!important; box-shadow:none!important;}
section[data-testid="stSidebar"] .stButton>button[kind="secondary"] {background:transparent!important; color:#43536B!important; border:1px solid transparent!important;}
section[data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {background:#EDF3FC!important; border:1px solid #D5E3F7!important; color:#102840!important;}
section[data-testid="stSidebar"] .stButton>button[kind="primary"] {background:linear-gradient(135deg,#142D4A,#215F88)!important; color:white!important; box-shadow:0 5px 15px rgba(20,45,74,.14)!important; border-color:transparent!important;}
section[data-testid="stSidebar"] [data-testid="stExpander"] {border:1px solid #DDE6F0; border-radius:14px; overflow:hidden; margin-top:7px; background:rgba(255,255,255,.67);}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {font-size:12px; color:#30435B;}
.v171-mini-brand {display:flex; align-items:center; gap:12px; color:#14314E; border:1px solid rgba(74,108,149,.18); border-radius:17px; background:linear-gradient(110deg,rgba(255,255,255,.96),rgba(241,248,252,.88)); padding:12px 16px; margin:0 0 12px; box-shadow:0 9px 26px rgba(13,33,57,.05);}
.v171-mini-brand >span:nth-child(2) {font-weight:900; font-size:12px; letter-spacing:.13em;}
.v171-mini-brand em {font-style:normal; color:#189EA2;}
.v171-brand-mark {font-size:19px; color:#1A99A0;}
.v171-brand-context {margin-left:auto; font-size:11px; font-weight:700; color:#7B8EA1;}
.v171-breadcrumb {display:flex; gap:9px; align-items:center; flex-wrap:wrap; font-size:12px; color:#64748B; font-weight:800; margin:0 0 8px;}
.v171-breadcrumb span:first-child {font-size:10px; color:#8291A7; font-weight:900; letter-spacing:.11em;}
.v171-crumb-sep {color:#B8C5D4;}
.v171-finder-kicker,.v171-stage-label {font-size:10px; letter-spacing:.16em; color:#7F91A5; font-weight:900; margin:12px 0 3px;}
[data-testid="stExpander"]:has(.v171-finder-shell) {border:1px solid #D5E4F0; border-radius:18px;}
.v171-mission-heading {display:flex; align-items:center; justify-content:space-between; gap:16px; margin:13px 0 16px;}
.v171-mission-heading span {font-size:10px; color:#277D89; font-weight:900; letter-spacing:.15em;}
.v171-mission-heading h3 {font-size:25px; font-family:'Space Grotesk',Inter,sans-serif; color:#102640; margin:5px 0;}
.v171-mission-heading p {font-size:13px; color:#657589; margin:0;}
.v171-mission-status {white-space:nowrap; font-size:10px; font-weight:900; letter-spacing:.10em; padding:9px 13px; border:1px solid #BDE2D8; color:#106B62; border-radius:999px; background:#E8FAF5;}
.v171-status-dot {display:inline-block; width:7px; height:7px; border-radius:50%; background:#19A76B; margin-right:6px;}
.v171-decision-kpi {background:linear-gradient(155deg,#FFFFFF,#F6FAFD); border:1px solid #DAE5F0; border-radius:20px; padding:18px 21px; box-shadow:0 10px 30px rgba(15,35,58,.055); min-height:134px; margin-bottom:12px;}
.v171-decision-kpi >span {display:block; font-size:10px; letter-spacing:.12em; font-weight:900; color:#74839B;}
.v171-decision-kpi strong {display:block; font-size:30px; font-family:'Space Grotesk',Inter,sans-serif; color:#10243C; margin:7px 0 5px; letter-spacing:-.04em;}
.v171-decision-kpi small {display:block; font-size:12px; color:#697D93;}
.v171-account-insight {display:grid; grid-template-columns:minmax(0,1fr) 180px; gap:18px; border:1px solid #C5DCEA; border-radius:22px; background:linear-gradient(120deg,#F7FCFD,#EDF6FA); margin:11px 0 13px; padding:18px 21px; box-shadow:0 9px 25px rgba(17,54,79,.045);}
.v171-account-insight span {font-size:10px; color:#177E88; font-weight:900; letter-spacing:.13em;}
.v171-account-insight strong {display:block; color:#122D43; font-size:20px; margin:6px 0; font-family:'Space Grotesk',Inter,sans-serif;}
.v171-account-insight p {margin:0; color:#475F74; font-size:13px; line-height:1.5;}
.v171-account-impact {border-left:1px solid #CBDEE9; padding-left:19px; display:flex; flex-direction:column; gap:4px; justify-content:center;}
.v171-account-impact small {color:#71849A; font-size:10px; font-weight:900; letter-spacing:.1em;}
.v171-account-impact strong {font-size:17px; margin:0 0 6px;}
.v171-active-stage {background:linear-gradient(125deg,#F4F9FD,#EAF5F6); border:1px solid #CDE2EA; border-radius:18px; padding:16px 19px; margin:13px 0;}
.v171-active-stage span {display:block; color:#2B7F89; font-size:10px; letter-spacing:.14em; font-weight:900;}
.v171-active-stage strong {display:block; font-family:'Space Grotesk',Inter,sans-serif; font-size:20px; color:#102C45; margin:3px 0;}
.v171-active-stage p {font-size:13px; color:#51667C; margin:2px 0;}
.stButton>button:focus-visible {outline:3px solid rgba(31,130,187,.45)!important; outline-offset:2px!important;}
@media(max-width:950px) {
 .v171-mission-heading {display:block;}
 .v171-mission-status {display:inline-block; margin-top:10px;}
 .v171-account-insight {grid-template-columns:1fr;}
 .v171-account-impact {border-left:0; border-top:1px solid #CBDEE9; padding-left:0; padding-top:12px;}
 .v171-brand-context {display:none;}
}
</style>
""", unsafe_allow_html=True)


WORKSPACE_STORY = {
    "Mission Control": ("PORTFOLIO / DECISIONS", "Your priorities, one clear path forward", "See where retention risk, revenue exposure and unresolved customer needs intersect. Start with an account; leave with an action.", "Portfolio analysis · synthetic data"),
    "Account Briefing": ("ACCOUNT / 360°", "One account. One clear decision.", "Understand product adoption, renewal exposure, customer friction and the people who can resolve the next blocker.", "Account analysis · synthetic data"),
    "End-to-End Flow (MVP 2.1.1)": ("REVENUE / LIFECYCLE", "From customer value to verified outcome", "Qualify genuine customer needs, record accountable approval, align with Sales and track the outcome with evidence.", "Workspace · source-referenced inputs"),
    "Growth & Expansion (MVP 2.1)": ("GROWTH / INTELLIGENCE", "Find growth that serves the customer", "Surface unmet needs, confirm product fit and qualify opportunities only when customer outcomes support the case.", "Workspace · source-referenced inputs"),
    "Real-data Pilot (MVP 2.0)": ("CUSTOMER / OUTCOMES", "Make customer value measurable", "Track agreed goals, guide the next best action and record what changed after intervention.", "Workspace · source-referenced inputs"),
    "Priority Queue": ("PORTFOLIO / ACTIONS", "Focus on the accounts that matter now", "Review the right accounts in priority order and open their full context before making decisions.", "Portfolio analysis · synthetic data"),
}

def render_workspace_story(view):
    """Consistent information scent on high-value screens; view names stay stable."""
    if view not in WORKSPACE_STORY:
        return
    eyebrow, title, summary, provenance = WORKSPACE_STORY[view]
    st.markdown(
        '<section class="v17-workspace-head" aria-label="Workspace overview">'
        f'<div class="v17-workspace-eyebrow">{html.escape(eyebrow)}</div>'
        '<div class="v17-workspace-top">'
        f'<div><h2 class="v17-workspace-title">{html.escape(title)}</h2>'
        f'<p class="v17-workspace-sub">{html.escape(summary)}</p></div>'
        f'<div class="v17-data-chip">◉ {html.escape(provenance)}</div>'
        '</div></section>', unsafe_allow_html=True)

# Navigation is rendered first in the sidebar, before the contextual portfolio filters.

active_view = st.session_state["active_view"]
active_queue_filter = st.session_state["queue_filter"]
queue_filtered = apply_queue_filter(filtered, active_queue_filter).sort_values("churn_probability", ascending=False).reset_index(drop=True)
queue_title, queue_description = queue_context(active_queue_filter)

# A concise decision header precedes the detailed workspace.
render_workspace_story(active_view)

# Show a clear way back to the main panel whenever the user drills down from a Mission Control CTA.
render_return_control(active_view, active_queue_filter, queue_title)

# Persistent visible account search, placed near the top so CSMs can jump into any customer quickly.
if active_view not in ("Real-data Pilot (MVP 2.0)","Growth & Expansion (MVP 2.1)","End-to-End Flow (MVP 2.1.1)"):
    render_global_account_finder(df)


# =========================
# MVP 2.0: real-data, evidence-led pilot for up to three accounts
# Data never gets committed or written to disk by this module.
# Synthetic ML predictions remain isolated in the existing demo views.
# =========================

import io
import os
import uuid
from datetime import date, datetime, timezone

PILOT_COLUMNS = [
    'account_id', 'account_name', 'csm_name', 'plan_family', 'mrr_eur',
    'renewal_date', 'goal_name', 'goal_metric', 'goal_direction', 'goal_unit',
    'goal_baseline', 'goal_target', 'goal_current', 'measurement_as_of',
    'active_users_30d', 'licensed_users', 'open_blockers',
    'critical_tickets_30d', 'source_system', 'source_reference',
    'validated_by', 'validation_date',
]
PILOT_ACTION_COLUMNS = [
    'action_id', 'account_id', 'account_name', 'csm_name', 'proposed_at',
    'action_type', 'recommendation_reason', 'description', 'requested_team',
    'approval_status', 'approved_by', 'approved_at', 'rejection_reason',
    'execution_status', 'due_date', 'completed_at', 'before_goal_value',
    'before_active_users', 'before_licensed_users', 'before_measurement_as_of',
    'after_goal_value', 'after_active_users', 'after_licensed_users',
    'after_measurement_as_of', 'outcome_evidence_reference', 'result_notes',
]
PILOT_NUMERIC = [
    'mrr_eur', 'goal_baseline', 'goal_target', 'goal_current',
    'active_users_30d', 'licensed_users', 'open_blockers', 'critical_tickets_30d',
]


def pilot_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def pilot_empty_accounts():
    return pd.DataFrame(columns=PILOT_COLUMNS)


def pilot_empty_actions():
    return pd.DataFrame(columns=PILOT_ACTION_COLUMNS)


def pilot_safe_number(v):
    return pd.to_numeric(v, errors='coerce')


def pilot_clean_accounts(input_df):
    """Strictly map user-provided CSV to the pilot schema without inventing values."""
    if input_df is None or input_df.empty:
        return pilot_empty_accounts(), ['No accounts have been entered.']
    if len(input_df) > 3:
        return pilot_empty_accounts(), ['This workspace is configured for up to three accounts.']
    df = input_df.copy()
    for col in PILOT_COLUMNS:
        if col not in df:
            df[col] = pd.NA
    df = df[PILOT_COLUMNS]
    errors = []
    for col in ['account_id', 'account_name', 'csm_name']:
        df[col] = df[col].fillna('').astype(str).str.strip()
        if df[col].eq('').any():
            errors.append(f'Missing required field: {col}.')
    if df['account_id'].duplicated().any():
        errors.append('account_id must be unique across pilot accounts.')
    for col in PILOT_NUMERIC:
        provided = df[col].notna() & df[col].astype(str).str.strip().ne('')
        converted = pd.to_numeric(df[col], errors='coerce')
        if (provided & converted.isna()).any():
            errors.append(f'{col} must contain a number or remain blank.')
        df[col] = converted
    for col in ['mrr_eur', 'active_users_30d', 'licensed_users', 'open_blockers', 'critical_tickets_30d']:
        if df[col].dropna().lt(0).any():
            errors.append(f'{col} cannot be negative.')
    if ((df['licensed_users'] > 0) & (df['active_users_30d'] > df['licensed_users'])).any():
        errors.append('active_users_30d cannot exceed licensed_users (check your definition).')
    dirs = df['goal_direction'].fillna('').astype(str).str.strip().str.lower()
    if (~dirs.isin(['', 'increase', 'decrease'])).any():
        errors.append('goal_direction must be increase or decrease.')
    df['goal_direction'] = dirs
    for col in ['renewal_date', 'measurement_as_of', 'validation_date']:
        # Preserve as YYYY-MM-DD for clean CSV roundtrips and portability.
        parsed = pd.to_datetime(df[col], errors='coerce')
        invalid = df[col].notna() & df[col].astype(str).str.strip().ne('') & parsed.isna()
        if invalid.any():
            errors.append(f'Invalid date in {col}; use YYYY-MM-DD.')
        df[col] = parsed.dt.strftime('%Y-%m-%d').fillna('')
    return df, errors


def pilot_progress_pct(row):
    """Achievement of the primary outcome goal, not a churn probability or ROI."""
    baseline = pilot_safe_number(row.get('goal_baseline'))
    target = pilot_safe_number(row.get('goal_target'))
    current = pilot_safe_number(row.get('goal_current'))
    direction = str(row.get('goal_direction', '')).lower()
    if any(pd.isna(x) for x in [baseline, target, current]):
        return None
    if direction == 'increase' and target > baseline:
        return float(np.clip((current - baseline) / (target - baseline) * 100, 0, 100))
    if direction == 'decrease' and target < baseline:
        return float(np.clip((baseline - current) / (baseline - target) * 100, 0, 100))
    return None


def pilot_adoption_pct(row):
    active = pilot_safe_number(row.get('active_users_30d'))
    licenses = pilot_safe_number(row.get('licensed_users'))
    if pd.isna(active) or pd.isna(licenses) or licenses <= 0:
        return None
    return float(active / licenses * 100)


def pilot_evidence_check(row):
    """Documented/attested evidence only; does NOT independently verify its truth."""
    required = ['source_system', 'source_reference', 'validated_by', 'validation_date',
                'measurement_as_of', 'goal_name', 'goal_metric', 'goal_direction',
                'goal_baseline', 'goal_target', 'goal_current']
    missing = [c for c in required if pd.isna(row.get(c, None)) or not str(row.get(c, '')).strip()]
    if missing:
        return 'Missing evidence: ' + ', '.join(missing)
    if pilot_progress_pct(row) is None:
        return 'Invalid outcome definition: target must differ from baseline in the intended direction'
    measurement = pd.to_datetime(row.get('measurement_as_of'), errors='coerce')
    validation = pd.to_datetime(row.get('validation_date'), errors='coerce')
    if pd.isna(measurement) or pd.isna(validation):
        return 'Invalid evidence dates'
    if measurement.date() > date.today() or validation.date() > date.today():
        return 'Future evidence date – review'
    age = (date.today() - measurement.date()).days
    if age > 30:
        return f'Stale evidence ({age} days old)'
    return 'Evidence documented (not independently verified)'


def pilot_renewal_days(row):
    dt = pd.to_datetime(row.get('renewal_date'), errors='coerce')
    return None if pd.isna(dt) else int((dt.date() - date.today()).days)


def pilot_suggestions(row):
    """Deterministic explainable NBA: rules, not machine learning or causal estimates."""
    suggestions = []
    account_id = str(row['account_id'])
    progress = pilot_progress_pct(row)
    adoption = pilot_adoption_pct(row)
    renewal_days = pilot_renewal_days(row)
    blockers = pilot_safe_number(row.get('open_blockers'))
    tickets = pilot_safe_number(row.get('critical_tickets_30d'))
    evidence = pilot_evidence_check(row)

    def add(priority, action, team, why):
        suggestions.append(dict(account_id=account_id, account_name=str(row['account_name']),
                                priority=priority, action_type=action,
                                requested_team=team, reason=why))

    if evidence != 'Evidence documented (not independently verified)':
        add('P1', 'Refresh evidence & validate account data', 'CSM / Data', evidence)
    if pd.notna(blockers) and blockers > 0:
        add('P1', 'Resolve outcome-blocking product issue', 'Customer Support / Product',
            f'{int(blockers)} open blocker(s) affecting customer workflows.')
    elif pd.notna(tickets) and tickets >= 2:
        add('P2', 'Review recurrent critical support issues', 'Customer Support',
            f'{int(tickets)} critical support tickets in 30 days.')
    if progress is None:
        add('P1', 'Agree measurable success outcome with customer', 'CSM',
            'Baseline, direction, target or current outcome is unavailable or inconsistent.')
    elif progress < 50:
        add('P1', 'Run outcome recovery / value review', 'CSM',
            f'Only {progress:.0f}% of stated outcome progress achieved.')
    elif progress < 80:
        add('P2', 'Optimize customer outcome plan', 'CSM',
            f'{progress:.0f}% progress toward primary outcome goal.')
    if adoption is not None and adoption < 40:
        add('P2', 'Targeted adoption enablement session', 'CSM / Support',
            f'{adoption:.0f}% of licensed users active in 30 days (coverage proxy).')
    if renewal_days is not None and renewal_days <= 60:
        add('P1' if renewal_days <= 30 else 'P2', 'Renewal value / stakeholder review', 'CSM / Sales',
            (f'Renewal overdue by {abs(renewal_days)} days' if renewal_days < 0 else f'Renewal in {renewal_days} days')
            + '; validate outcome evidence before commercial decision.')
    if not suggestions:
        add('P3', 'Document success and plan next milestone', 'CSM',
            'No critical conditions detected in the currently available account signals.')
    priority_rank = {'P1': 0, 'P2': 1, 'P3': 2}
    return sorted(suggestions, key=lambda x: priority_rank[x['priority']])


def pilot_action_impact(action, account):
    if str(action.get('execution_status')) != 'Completed':
        return ('Pending completion', None, None)
    before = pilot_safe_number(action.get('before_goal_value'))
    after = pilot_safe_number(action.get('after_goal_value'))
    after_date = pd.to_datetime(action.get('after_measurement_as_of'), errors='coerce')
    approval = pd.to_datetime(action.get('approved_at'), errors='coerce', utc=True)
    reference = str(action.get('outcome_evidence_reference') or '').strip()
    if pd.isna(before) or pd.isna(after) or pd.isna(after_date) or not reference:
        return ('Measurement pending / evidence missing', None, None)
    if pd.notna(approval) and after_date.date() < approval.date():
        return ('Post-measurement predates approval', None, None)
    direction = str(account.get('goal_direction', '')).lower()
    outcome_change = (after - before) if direction == 'increase' else ((before - after) if direction == 'decrease' else None)
    before_active = pilot_safe_number(action.get('before_active_users'))
    before_seats = pilot_safe_number(action.get('before_licensed_users'))
    after_active = pilot_safe_number(action.get('after_active_users'))
    after_seats = pilot_safe_number(action.get('after_licensed_users'))
    adoption_delta = None
    if not any(pd.isna(x) for x in [before_active, before_seats, after_active, after_seats]) and before_seats > 0 and after_seats > 0:
        adoption_delta = float(after_active / after_seats * 100 - before_active / before_seats * 100)
    return ('Observed change · NOT causal attribution', outcome_change, adoption_delta)


def pilot_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def pilot_import_actions(uploaded_df, accounts):
    required = {'action_id', 'account_id', 'approval_status', 'execution_status'}
    if not required.issubset(uploaded_df.columns):
        return pilot_empty_actions(), ['Action log is missing: ' + ', '.join(sorted(required - set(uploaded_df.columns)))]
    out = uploaded_df.copy()
    for col in PILOT_ACTION_COLUMNS:
        if col not in out.columns:
            out[col] = ''
    out = out[PILOT_ACTION_COLUMNS].fillna('')
    if out['action_id'].astype(str).duplicated().any():
        return pilot_empty_actions(), ['Duplicate action_id values in action log.']
    valid_accounts = set(accounts['account_id'].astype(str))
    if not set(out['account_id'].astype(str)).issubset(valid_accounts):
        return pilot_empty_actions(), ['The action log includes accounts not present in the current pilot.']
    if not out['approval_status'].isin(['Proposed', 'Approved', 'Rejected']).all():
        return pilot_empty_actions(), ['Invalid approval_status value.']
    if not out['execution_status'].isin(['Not started', 'In progress', 'Completed', 'Cancelled']).all():
        return pilot_empty_actions(), ['Invalid execution_status value.']
    return out, []


def pilot_render():
    st.markdown('## Customer Value & Actions')
    st.caption('Understand customer outcomes · prioritize interventions · measure changes after action')
    st.warning('**Restricted customer data.** Never upload customer, contract or employee information to a public Streamlit app. '
               'Use a locally hosted session or an IT-approved access-controlled internal deployment. '
               'The toggle below is an acknowledgement, **not authentication**.')
    if os.environ.get('SEMRUSH_INTERNAL_PILOT', '0') != '1':
        st.info('Customer data import is disabled in public sessions. For an approved internal environment, start the app using '
                '`SEMRUSH_INTERNAL_PILOT=1 python -m streamlit run streamlit_app_semrush_v3.py`. '
                'Do not configure this flag on a public deployment.')
        return
    if not st.checkbox('I confirm that this is a restricted environment and that the account data is authorized for use.', key='mvp20_authorized'):
        st.info('Confirm the approved environment to enable account imports.')
        return

    if 'mvp20_accounts' not in st.session_state:
        st.session_state['mvp20_accounts'] = pilot_empty_accounts()
    if 'mvp20_actions' not in st.session_state:
        st.session_state['mvp20_actions'] = pilot_empty_actions()

    if st.button('Continue to Growth & Expansion Intelligence →',
                 key='mvp20_to_growth',
                 disabled=st.session_state['mvp20_accounts'].empty):
        st.session_state['active_view'] = 'Growth & Expansion (MVP 2.1)'
        st.rerun()

    tabs = st.tabs(['1 · Value Intelligence', '2 · Next Best Action', '3 · CS Impact', 'Data governance'])
    with tabs[0]:
        st.markdown('### Connect customer accounts')
        st.write('The account workspace uses 22 source-referenced fields for outcome tracking. The synthetic portfolio model is not applied to imported customer records.')
        t = pd.DataFrame([{c: '' for c in PILOT_COLUMNS} for _ in range(3)])
        st.download_button('Download blank CSV · 3 accounts', data=pilot_csv_bytes(t),
                           file_name='mvp20_pilot_accounts_blank.csv', mime='text/csv', key='mvp20_blank')
        source = st.file_uploader('Import the completed account CSV', type=['csv'], key='mvp20_accounts_file',
                                  help='Keep this data on an approved internal/local deployment only.')
        if source is not None and st.button('Validate and load accounts', key='mvp20_load'):
            try:
                incoming = pd.read_csv(source, dtype={'account_id': str})
                incoming = incoming.dropna(how='all')
                incoming = incoming[incoming['account_id'].notna() & incoming['account_id'].astype(str).str.strip().ne('')]
                clean, errors = pilot_clean_accounts(incoming)
                if errors:
                    for err in errors: st.error(err)
                else:
                    existing_actions = st.session_state['mvp20_actions']
                    new_ids=set(clean['account_id'].astype(str))
                    legacy_ids=set(existing_actions['account_id'].astype(str)) if not existing_actions.empty else set()
                    growth_ids=set(st.session_state['mvp21_signals']['account_id'].astype(str)) if 'mvp21_signals' in st.session_state and not st.session_state['mvp21_signals'].empty else set()
                    opportunity_ids=set(st.session_state['mvp21_opps']['account_id'].astype(str)) if 'mvp21_opps' in st.session_state and not st.session_state['mvp21_opps'].empty else set()
                    if not (legacy_ids | growth_ids | opportunity_ids).issubset(new_ids):
                        st.error('Existing actions or growth records reference other accounts. Export them before replacing the account set.')
                    else:
                        st.session_state['mvp20_accounts'] = clean
                        st.success(f'{len(clean)} account(s) loaded. No churn model has been applied.')
            except Exception as exc:
                st.error('Could not read CSV. Check column headers and formatting: ' + str(exc))
        with st.expander('Add one account manually', expanded=False):
            current_count = len(st.session_state['mvp20_accounts'])
            if current_count >= 3:
                st.info('Three accounts are already loaded. Export first if you need to replace them.')
            else:
                with st.form('mvp20_manual_account_form', clear_on_submit=True):
                    a, b = st.columns(2)
                    with a:
                        account_id = st.text_input('Internal account ID *')
                        account_name = st.text_input('Account display name *')
                        csm_name = st.text_input('Assigned CSM *')
                        plan_family = st.text_input('Subscribed toolkit / package')
                        mrr_eur = st.text_input('MRR, EUR (optional)', value='', placeholder='2500')
                        renewal_date = st.text_input('Renewal date (YYYY-MM-DD)', placeholder='2027-01-15')
                        goal_name = st.text_input('Primary customer outcome', placeholder='Improve AI brand visibility')
                        goal_metric = st.text_input('Outcome metric', placeholder='AI mentions across monitored prompts')
                        goal_direction = st.selectbox('Expected direction', ['increase', 'decrease'])
                        goal_unit = st.text_input('Metric unit', placeholder='mentions')
                        goal_baseline = st.text_input('Outcome baseline', value='', placeholder='10')
                    with b:
                        goal_target = st.text_input('Outcome target', value='', placeholder='100')
                        goal_current = st.text_input('Current measured value', value='', placeholder='42')
                        measurement_as_of = st.text_input('Measurement date (YYYY-MM-DD)', value=str(date.today()))
                        active_users = st.text_input('Active users last 30 days (optional)', value='', placeholder='4')
                        licensed_users = st.text_input('Licensed users (optional)', value='', placeholder='10')
                        open_blockers = st.text_input('Open outcome blockers (optional)', value='', placeholder='0')
                        critical_tickets = st.text_input('Critical tickets in last 30 days (optional)', value='', placeholder='0')
                        source_system = st.text_input('Evidence source system', placeholder='CRM / product analytics')
                        source_reference = st.text_input('Evidence reference / record ID', placeholder='internal dashboard reference')
                        validated_by = st.text_input('Validated by (owner name)')
                        validation_date = st.text_input('Validation date (YYYY-MM-DD)', value=str(date.today()))
                    submitted = st.form_submit_button('Add account', use_container_width=True)
                if submitted:
                    record = dict(zip(PILOT_COLUMNS, [account_id, account_name, csm_name, plan_family, mrr_eur,
                         renewal_date, goal_name, goal_metric, goal_direction, goal_unit, goal_baseline, goal_target,
                         goal_current, measurement_as_of, active_users, licensed_users, open_blockers,
                         critical_tickets, source_system, source_reference, validated_by, validation_date]))
                    clean, errors = pilot_clean_accounts(pd.concat([st.session_state['mvp20_accounts'], pd.DataFrame([record])], ignore_index=True))
                    if errors:
                        for err in errors: st.error(err)
                    else:
                        st.session_state['mvp20_accounts'] = clean
                        st.success('Account connected to the workspace.')
                        st.rerun()
        accounts = st.session_state['mvp20_accounts']
        if accounts.empty:
            st.info('Add a real account to show value metrics. No actual customer information is included in this package.')
        else:
            summaries = []
            for _, row in accounts.iterrows():
                prog = pilot_progress_pct(row)
                adoption = pilot_adoption_pct(row)
                summaries.append({'Account': row['account_name'], 'CSM': row['csm_name'],
                   'Outcome achievement (%)': None if prog is None else round(prog, 1),
                   '30d active / licensed (%)': None if adoption is None else round(adoption, 1),
                   'Open blockers': row['open_blockers'], 'Renewal (days)': pilot_renewal_days(row),
                   'Evidence': pilot_evidence_check(row)})
            c1, c2, c3 = st.columns(3)
            c1.metric('Connected accounts', len(accounts), 'Max 3')
            values = [pilot_progress_pct(row) for _, row in accounts.iterrows()]
            values = [v for v in values if v is not None]
            c2.metric('Avg outcome progress', f'{np.mean(values):.1f}%' if values else 'N/A', 'Descriptive only')
            c3.metric('Evidence documented', f'{sum(pilot_evidence_check(row).startswith("Evidence documented") for _, row in accounts.iterrows())}/{len(accounts)}', 'Attested, not audited')
            st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)
            chosen = st.selectbox('Inspect account outcomes', accounts['account_id'].tolist(),
                                  format_func=lambda x: str(accounts.loc[accounts.account_id == x, 'account_name'].iloc[0]),
                                  key='mvp20_value_account')
            row = accounts.loc[accounts.account_id == chosen].iloc[0]
            prog = pilot_progress_pct(row)
            adoption = pilot_adoption_pct(row)
            st.markdown(f'#### {html.escape(str(row["account_name"]))} · Customer value 360°')
            p1, p2, p3 = st.columns(3)
            p1.metric('Primary goal progress', 'N/A' if prog is None else f'{prog:.1f}%',
                      'From recorded baseline to target')
            p2.metric('30-day usage coverage', 'N/A' if adoption is None else f'{adoption:.1f}%',
                      'Active users ÷ licensed users')
            days = pilot_renewal_days(row)
            p3.metric('Days to renewal', 'N/A' if days is None else str(days), 'Not a churn forecast')
            st.write('**Customer outcome:**', str(row.get('goal_name', '') or 'Not documented'))
            st.write('**Metric / current / target:**', str(row.get('goal_metric','')),
                     f'{row.get("goal_current", "—")} → {row.get("goal_target", "—")} {row.get("goal_unit", "")}' )
            st.caption('Value achievement is normalized progress toward one declared goal. Usage coverage is a user-count proxy, not product-depth adoption or ROI. Both need source evidence.')
            st.info('Evidence status: ' + pilot_evidence_check(row))
            with st.expander('View exact source values'):
                st.dataframe(pd.DataFrame([row]), hide_index=True, use_container_width=True)
            st.download_button('Export current accounts CSV', pilot_csv_bytes(accounts),
                               file_name='mvp20_pilot_accounts_working.csv', mime='text/csv', key='mvp20_export_accounts')

    with tabs[1]:
        st.markdown('### Next Best Action · explainable recommendations')
        accounts = st.session_state['mvp20_accounts']
        if accounts.empty:
            st.info('Load accounts first in Value Intelligence.')
        else:
            proposed = []
            for _, row in accounts.iterrows():
                proposed.extend(pilot_suggestions(row))
            ranked = pd.DataFrame(proposed)
            st.dataframe(ranked.rename(columns={'priority':'Priority', 'account_name':'Account',
              'action_type':'Recommended action', 'requested_team':'Suggested owner(s)', 'reason':'Why'}).drop(columns=['account_id']),
              use_container_width=True, hide_index=True)
            st.caption('Rankings are transparent rules over uploaded signals. No ML risk score or predicted action uplift is used for these real accounts.')
            accid = st.selectbox('Choose account for action proposal', accounts['account_id'].tolist(), key='mvp20_nba_account',
                format_func=lambda x: str(accounts.loc[accounts.account_id == x,'account_name'].iloc[0]))
            choices = [s for s in proposed if s['account_id'] == accid]
            with st.form('mvp20_propose_form'):
                action_choice = st.selectbox('Recommended intervention', range(len(choices)),
                    format_func=lambda i: f'{choices[i]["priority"]} · {choices[i]["action_type"]}')
                suggestion = choices[action_choice]
                team = st.text_input('Suggested supporting team(s)', value=suggestion['requested_team'])
                why = st.text_area('Evidence / reason', value=suggestion['reason'])
                desc = st.text_area('CSM action plan (editable)', value=suggestion['action_type'])
                due_date = st.text_input('Target date (YYYY-MM-DD)', placeholder='2026-11-15')
                clicked = st.form_submit_button('Propose action for CSM approval')
            if clicked:
                acc = accounts.loc[accounts.account_id == accid].iloc[0]
                if not desc.strip():
                    st.error('Action description is required.')
                elif due_date and pd.isna(pd.to_datetime(due_date, errors='coerce')):
                    st.error('Due date must be YYYY-MM-DD.')
                else:
                    item = {col: '' for col in PILOT_ACTION_COLUMNS}
                    item.update({'action_id': uuid.uuid4().hex[:12], 'account_id':accid, 'account_name':acc['account_name'],
                      'csm_name':acc['csm_name'], 'proposed_at':pilot_now(), 'action_type':suggestion['action_type'],
                      'recommendation_reason':why, 'description':desc.strip(), 'requested_team':team.strip(),
                      'approval_status':'Proposed', 'execution_status':'Not started', 'due_date':due_date})
                    st.session_state['mvp20_actions'] = pd.concat([st.session_state['mvp20_actions'], pd.DataFrame([item])], ignore_index=True)
                    st.success('Action proposed; it cannot start until the assigned CSM approves.')
            actions = st.session_state['mvp20_actions']
            pending = actions[actions['approval_status'] == 'Proposed'] if not actions.empty else actions
            if not pending.empty:
                st.markdown('#### CSM approval queue')
                with st.form('mvp20_approval_form'):
                    sel = st.selectbox('Pending action', pending['action_id'].tolist(),
                       format_func=lambda x: f'{x} · {pending.loc[pending.action_id==x,"account_name"].iloc[0]} · {pending.loc[pending.action_id==x,"description"].iloc[0][:70]}')
                    a = pending.loc[pending.action_id == sel].iloc[0]
                    st.caption(f'Assigned CSM: {a["csm_name"]} · {a["description"]}')
                    approver = st.text_input('CSM approver name (must match assigned owner)')
                    decision = st.radio('Decision', ['Approve', 'Reject'], horizontal=True)
                    reason = st.text_input('Reason / comment (required if rejecting)')
                    submitted = st.form_submit_button('Record CSM decision')
                if submitted:
                    if approver.strip().casefold() != str(a['csm_name']).strip().casefold():
                        st.error('Only the assigned CSM name can be entered as the decision owner. Identity must still be verified by company SSO in production.')
                    elif decision == 'Reject' and not reason.strip():
                        st.error('Provide a reason for rejection.')
                    else:
                        idx = actions.index[actions.action_id == sel][0]
                        actions.loc[idx, 'approval_status'] = 'Approved' if decision=='Approve' else 'Rejected'
                        actions.loc[idx, 'approved_by'] = approver.strip()
                        actions.loc[idx, 'approved_at'] = pilot_now()
                        if decision == 'Reject':
                            actions.loc[idx, 'rejection_reason'] = reason.strip()
                            actions.loc[idx, 'execution_status'] = 'Cancelled'
                        else:
                            acc = accounts.loc[accounts.account_id == a['account_id']].iloc[0]
                            for target, source in [('before_goal_value','goal_current'), ('before_active_users','active_users_30d'),
                                                   ('before_licensed_users','licensed_users'), ('before_measurement_as_of','measurement_as_of')]:
                                actions.loc[idx,target] = acc[source]
                        st.session_state['mvp20_actions'] = actions.copy()
                        st.success(f'Action {decision.lower()} decision recorded (self-attested, not authenticated).')
                        st.rerun()
            if not actions.empty:
                st.markdown('#### Action tracker')
                visible = ['action_id','account_name','csm_name','description','approval_status','execution_status','due_date','approved_by']
                st.dataframe(actions[visible], use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown('### CS Impact Tracking · before / after measurements')
        st.caption('Track what changed after a CSM-approved action. These are descriptive observations, **not proof the CSM caused the change**.')
        actions = st.session_state['mvp20_actions']
        accounts = st.session_state['mvp20_accounts']
        approved = actions[actions['approval_status'] == 'Approved'] if not actions.empty else actions
        if approved.empty:
            st.info('Record CSM approval on an action first.')
        else:
            with st.form('mvp20_completion_form'):
                aid = st.selectbox('Approved action to update', approved['action_id'].tolist(),
                    format_func=lambda x: f'{x} · {approved.loc[approved.action_id == x,"account_name"].iloc[0]} · {approved.loc[approved.action_id == x,"action_type"].iloc[0]}')
                selected = approved.loc[approved.action_id == aid].iloc[0]
                st.info(f'Before snapshot: outcome {selected["before_goal_value"]}, active users {selected["before_active_users"]}, '
                        f'measurement date {selected["before_measurement_as_of"]}. Stored at CSM approval.')
                newstatus = st.selectbox('Execution status', ['Not started','In progress','Completed','Cancelled'],
                     index=['Not started','In progress','Completed','Cancelled'].index(selected['execution_status'])
                     if selected['execution_status'] in ['Not started','In progress','Completed','Cancelled'] else 0)
                observed = st.text_input('Post-action measurement date (YYYY-MM-DD)',
                     value=str(selected['after_measurement_as_of'] or ''))
                goal_after_text = st.text_input('Post-action outcome value (leave blank if not measured)',
                     value=str(selected['after_goal_value'] if str(selected['after_goal_value']) not in ('','nan') else ''))
                active_after_text = st.text_input('Active users last 30 days at post-measurement (optional)',
                     value=str(selected['after_active_users'] if str(selected['after_active_users']) not in ('','nan') else ''))
                licensed_after_text = st.text_input('Licensed users at post-measurement (optional)',
                     value=str(selected['after_licensed_users'] if str(selected['after_licensed_users']) not in ('','nan') else ''))
                evidence_ref = st.text_input('Post-action measurement evidence ID / source reference',
                     value=str(selected['outcome_evidence_reference'] or ''))
                note = st.text_area('Observed result / limitations', value=str(selected['result_notes'] or ''))
                saved = st.form_submit_button('Save action status and measurement')
            if saved:
                numerics = {}
                invalid_num = False
                for key, raw_value in [('after_goal_value',goal_after_text),('after_active_users',active_after_text),('after_licensed_users',licensed_after_text)]:
                    val = str(raw_value).strip()
                    numerics[key] = '' if not val else pilot_safe_number(val)
                    if val and pd.isna(numerics[key]): invalid_num = True
                date_val = pd.to_datetime(observed, errors='coerce') if observed.strip() else pd.NaT
                if invalid_num:
                    st.error('Post-measurement values must be numeric if provided.')
                elif observed.strip() and (pd.isna(date_val) or date_val.date() > date.today()):
                    st.error('Post-measurement date must be a valid date that is not in the future.')
                else:
                    idx = actions.index[actions.action_id == aid][0]
                    previous = str(actions.loc[idx,'execution_status'])
                    actions.loc[idx, 'execution_status'] = newstatus
                    if newstatus == 'Completed' and previous != 'Completed':
                        actions.loc[idx, 'completed_at'] = pilot_now()
                    for col,val in numerics.items(): actions.loc[idx,col] = val
                    actions.loc[idx, 'after_measurement_as_of'] = date_val.strftime('%Y-%m-%d') if pd.notna(date_val) else ''
                    actions.loc[idx, 'outcome_evidence_reference'] = evidence_ref.strip()
                    actions.loc[idx, 'result_notes'] = note.strip()
                    st.session_state['mvp20_actions'] = actions.copy()
                    st.success('Observation recorded. Do not interpret before/after change as proven incremental CS impact.')
                    st.rerun()
            st.markdown('#### Impact evidence board')
            impact_rows = []
            for _, a in approved.iterrows():
                matching = accounts.loc[accounts.account_id == a['account_id']]
                if matching.empty: continue
                acc = matching.iloc[0]
                status, outcome, adoption = pilot_action_impact(a, acc)
                impact_rows.append({'Account':a['account_name'], 'Action':a['action_type'],
                                    'CSM':a['csm_name'], 'Execution':a['execution_status'],
                                    'Evidence status':status,
                                    'Outcome movement (favorable units)':outcome,
                                    'Usage coverage change (pp)':None if adoption is None else round(adoption,1),
                                    'Measurement evidence':a['outcome_evidence_reference']})
            st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)
            verified_obs = [x for x in impact_rows if x['Evidence status'].startswith('Observed change')]
            k1,k2,k3 = st.columns(3)
            k1.metric('Approved actions', len(approved))
            k2.metric('Completed actions', int((approved['execution_status']=='Completed').sum()))
            k3.metric('Actions with before/after evidence', len(verified_obs))
            st.caption('Never sum outcome units across accounts unless metrics and units are truly comparable. No retained-ARR or revenue-attribution claim is calculated.')
        if not actions.empty:
            st.download_button('Export action log CSV · SAVE THIS TO KEEP YOUR WORK', pilot_csv_bytes(actions),
                         file_name='mvp20_csm_actions_and_impact.csv', mime='text/csv', key='mvp20_export_actions')

    with tabs[3]:
        st.markdown('### Data quality, privacy and operating model')
        st.markdown('''
**Required to interpret account metrics:** account identifier, assigned CSM, measurable outcome,
metric direction, baseline, current value, target, measurement date, data source and source reference.
Missing values stay missing; we do not infer synthetic values or treat them as zero.

**Evidence rule:** *Evidence documented* means that a source reference and an owner-attested validation
are present and the measurement is no more than 30 days old. It does not mean this application has independently audited the evidence.

**Approval rule:** a recommended action is only a proposal. The assigned CSM must approve/reject it before tracking execution.
Approval names are currently self-attested; company SSO, permissions and an auditable record are required for verified approvals.

**Impact rule:** comparisons are observational. Establishing incremental retention or revenue uplift requires a suitable
counterfactual, experimental evaluation or stronger research design.

**Persistence:** uploads/actions live in your browser session only. **Export both account and action CSVs** before closing,
then restore them when needed. Files are not automatically sent to external APIs or saved to a server database.

**Security:** do not enable this workflow in a public app. An environment flag is a feature switch, not a security control.
Use an IT-approved internal deployment, restrict application access, apply data minimization and retention controls.
''')
        if not st.session_state['mvp20_accounts'].empty:
            st.download_button('Export account snapshot', pilot_csv_bytes(st.session_state['mvp20_accounts']),
                               'mvp20_account_snapshot.csv', mime='text/csv', key='mvp20_gov_accounts')
        if not st.session_state['mvp20_actions'].empty:
            st.download_button('Export action log', pilot_csv_bytes(st.session_state['mvp20_actions']),
                               'mvp20_action_log.csv', mime='text/csv', key='mvp20_gov_actions')
        old_log = st.file_uploader('Restore previously exported action log (CSV)', type='csv', key='mvp20_old_actions')
        if old_log is not None and st.button('Validate and restore action log', key='mvp20_restore_actions'):
            try:
                loaded = pd.read_csv(old_log, dtype=str, keep_default_na=False)
                restored, errors = pilot_import_actions(loaded, st.session_state['mvp20_accounts'])
                if errors:
                    for err in errors: st.error(err)
                else:
                    st.session_state['mvp20_actions'] = restored
                    st.success(f'{len(restored)} actions restored.')
            except Exception as exc:
                st.error('Invalid action log: ' + str(exc))


# =========================
# MVP 2.1 – Growth & Expansion Intelligence (authorized 1–3 account pilot)
# Separate, additive workspace. Rules do not train a model or infer causal impact.
# =========================

GROWTH_COLUMNS = [
    'account_id', 'customer_need', 'need_category', 'candidate_toolkit', 'existing_toolkits',
    'need_source_system', 'need_evidence_reference', 'need_evidence_date', 'need_validated_by',
    'customer_confirmed_need', 'product_fit_confirmed', 'expansion_requested_by_customer',
    'value_dependency_confirmed', 'capacity_usage_pct', 'additional_teams_request_count',
    'feature_requests_90d', 'expansion_mrr_estimate_eur', 'estimate_basis', 'ae_owner',
]
GROWTH_BOOLEAN = ['customer_confirmed_need', 'product_fit_confirmed',
                  'expansion_requested_by_customer', 'value_dependency_confirmed']
GROWTH_NUMERIC = ['capacity_usage_pct', 'additional_teams_request_count',
                  'feature_requests_90d', 'expansion_mrr_estimate_eur']
GROWTH_OPP_COLUMNS = [
    'opportunity_id','account_id','account_name','csm_name','created_at','candidate_toolkit',
    'customer_need','need_evidence_reference','readiness_at_proposal', 'review_status',
    'reviewed_by','reviewed_at','review_note','handoff_status','ae_owner','handoff_at',
    'handoff_evidence_reference','stage','closed_at','contracted_incremental_mrr_eur',
    'contract_evidence_reference','outcome_note',
]
GROWTH_REVIEW = ['Pending CSM', 'Approved for discovery', 'Rejected', 'On hold']
GROWTH_HANDOFF = ['Not sent', 'Sent to AE', 'Accepted by AE', 'Declined by AE']
GROWTH_STAGES = ['Discovery', 'Need validated', 'Demo completed', 'Proposal', 'Contracted', 'Lost']


def growth_empty_signals():
    return pd.DataFrame(columns=GROWTH_COLUMNS)


def growth_empty_opportunities():
    return pd.DataFrame(columns=GROWTH_OPP_COLUMNS)


def growth_yes(row, col):
    return str(row.get(col, '')).strip().lower() == 'yes'


def growth_clean_signals(incoming, accounts):
    """One evidence-linked growth hypothesis per pilot account; blank != zero/no."""
    if incoming is None or incoming.empty:
        return growth_empty_signals(), ['No growth-signal rows provided.']
    result = incoming.copy()
    for col in GROWTH_COLUMNS:
        if col not in result.columns:
            result[col] = ''
    result = result[GROWTH_COLUMNS].copy()
    issues = []
    for col in GROWTH_COLUMNS:
        if col not in GROWTH_NUMERIC:
            result[col] = result[col].fillna('').astype(str).str.strip()
    if result['account_id'].eq('').any():
        issues.append('account_id is required for every row.')
    if result['account_id'].duplicated().any():
        issues.append('Use at most one growth-signal row per account for this pilot.')
    allowed = set(accounts['account_id'].astype(str)) if accounts is not None and not accounts.empty else set()
    if not set(result['account_id']).issubset(allowed):
        issues.append('Growth signals must reference connected customer accounts.')
    if len(result) > 3:
        issues.append('Up to three growth-signal rows are supported in this workspace.')
    for col in GROWTH_BOOLEAN:
        result[col] = result[col].str.lower()
        if not result[col].isin(['', 'yes', 'no']).all():
            issues.append(f'{col}: allowed values are yes, no, or blank (unknown).')
    for col in GROWTH_NUMERIC:
        raw = result[col].fillna('').astype(str).str.strip()
        numeric = pd.to_numeric(raw.replace('', np.nan), errors='coerce')
        if (raw.ne('') & numeric.isna()).any():
            issues.append(f'{col}: enter a number or leave blank.')
        if numeric.dropna().lt(0).any():
            issues.append(f'{col}: cannot be negative.')
        result[col] = numeric
    if result['capacity_usage_pct'].dropna().gt(300).any():
        issues.append('capacity_usage_pct exceeds the 300% pilot validation limit; check units (90 not 0.90).')
    for col in ['additional_teams_request_count','feature_requests_90d']:
        vals = result[col].dropna()
        if ((vals % 1) != 0).any():
            issues.append(f'{col}: count must be an integer.')
    raw_date = result['need_evidence_date'].copy()
    parsed = pd.to_datetime(raw_date, errors='coerce', format='%Y-%m-%d')
    if (raw_date.ne('') & parsed.isna()).any():
        issues.append('need_evidence_date must be YYYY-MM-DD.')
    if (parsed.dt.date > date.today()).fillna(False).any():
        issues.append('need_evidence_date cannot be in the future.')
    result['need_evidence_date'] = parsed.dt.strftime('%Y-%m-%d').fillna('')
    return result, issues


def growth_evaluate(growth, account):
    """Transparent evidence-and-value gates; never a machine-learning likelihood."""
    value = pilot_progress_pct(account)
    adoption = pilot_adoption_pct(account)
    blockers = pilot_safe_number(account.get('open_blockers'))
    evidence = pilot_evidence_check(account)
    need = str(growth.get('customer_need', '')).strip()
    product = str(growth.get('candidate_toolkit', '')).strip()
    ref = str(growth.get('need_evidence_reference', '')).strip()
    source = str(growth.get('need_source_system', '')).strip()
    validated = str(growth.get('need_validated_by', '')).strip()
    when = pd.to_datetime(growth.get('need_evidence_date'), errors='coerce')
    fresh = pd.notna(when) and 0 <= (date.today()-when.date()).days <= 45
    need_attested = bool(need and product and ref and source and validated and fresh)
    confirmed = growth_yes(growth,'customer_confirmed_need')
    fit = growth_yes(growth,'product_fit_confirmed')
    requested = growth_yes(growth,'expansion_requested_by_customer')
    dependent = growth_yes(growth,'value_dependency_confirmed')
    pressure = pilot_safe_number(growth.get('capacity_usage_pct'))
    team_reqs = pilot_safe_number(growth.get('additional_teams_request_count'))
    requests = pilot_safe_number(growth.get('feature_requests_90d'))
    activity = ((pd.notna(pressure) and pressure >= 85) or
                (pd.notna(team_reqs) and team_reqs > 0) or
                (pd.notna(requests) and requests > 0))
    signals = []
    if pd.notna(pressure) and pressure >= 85: signals.append(f'Capacity usage {pressure:.0f}%')
    if pd.notna(team_reqs) and team_reqs > 0: signals.append(f'{int(team_reqs)} additional team requests')
    if pd.notna(requests) and requests > 0: signals.append(f'{int(requests)} relevant feature requests / 90d')
    if requested: signals.append('Customer explicitly requested expansion discussion')
    if not need_attested:
        return {'readiness':'Validate evidence','gate':'Need, candidate toolkit, source, reference, validator and evidence dated ≤45 days are required.',
                'signals':signals,'ready':False,'value_progress':value}
    if evidence != 'Evidence documented (not independently verified)' or value is None:
        return {'readiness':'Validate account value','gate':'Current customer-outcome evidence is incomplete or stale.',
                'signals':signals,'ready':False,'value_progress':value}
    friction = pd.notna(blockers) and blockers > 0
    low_usage = adoption is not None and adoption < 40
    low_value = value < 80 or friction or low_usage
    if low_value:
        if dependent and confirmed and fit:
            return {'readiness':'Value-dependent review','gate':'Customer value is below the pilot threshold or blocked. CSM must first document why the proposed solution is necessary for recovery; no automatic sales handoff.',
                    'signals':signals,'ready':False,'value_progress':value}
        return {'readiness':'Value recovery first','gate':'Value progress <80%, unresolved blockers, or low user coverage. Resolve value issues before commercial handoff.',
                'signals':signals,'ready':False,'value_progress':value}
    if not confirmed or not fit:
        return {'readiness':'Explore need','gate':'Customer need and product fit must both be confirmed explicitly.',
                'signals':signals,'ready':False,'value_progress':value}
    if not (requested or activity):
        return {'readiness':'Explore need','gate':'Need is confirmed, but no documented growth trigger or customer expansion request is recorded.',
                'signals':signals,'ready':False,'value_progress':value}
    return {'readiness':'Ready for CSM assessment',
            'gate':'Documented value, customer-confirmed need, product fit and current signal. CSM approval still required.',
            'signals':signals,'ready':True,'value_progress':value}


def growth_metrics(opps):
    """Revenue counts exclusively from contracted, evidence-supported, CSM/AE-reviewed deals."""
    if opps is None or opps.empty:
        return {'proposed':0,'csm_approved':0,'handoffs':0,'contracted':0,'verified_incremental_mrr_eur':0.0}
    approved = opps['review_status'] == 'Approved for discovery'
    sent = opps['handoff_status'].isin(['Sent to AE','Accepted by AE','Declined by AE'])
    amount = pd.to_numeric(opps['contracted_incremental_mrr_eur'], errors='coerce')
    ref = opps['contract_evidence_reference'].fillna('').astype(str).str.strip().ne('')
    qualified = (approved & (opps['handoff_status'] == 'Accepted by AE') &
                 (opps['stage'] == 'Contracted') & amount.notna() & amount.gt(0) & ref)
    return {'proposed':len(opps),'csm_approved':int(approved.sum()),
            'handoffs':int((approved & sent).sum()), 'contracted':int(qualified.sum()),
            'verified_incremental_mrr_eur':float(amount.loc[qualified].sum())}


def growth_import_opps(frame, accounts):
    if frame is None or frame.empty:
        return growth_empty_opportunities(), []
    if not set(GROWTH_OPP_COLUMNS).issubset(set(frame.columns)):
        return growth_empty_opportunities(), ['Opportunity register is missing required fields. Restore a complete workspace export.']
    result = frame[GROWTH_OPP_COLUMNS].fillna('').astype(str).copy()
    if result['opportunity_id'].eq('').any() or result['opportunity_id'].duplicated().any():
        return growth_empty_opportunities(), ['Opportunity IDs must be present and unique.']
    valid_accounts = set(accounts['account_id'].astype(str)) if accounts is not None and not accounts.empty else set()
    if not set(result['account_id']).issubset(valid_accounts):
        return growth_empty_opportunities(), ['Opportunity log contains accounts outside the loaded pilot.']
    for col, options in [('review_status',GROWTH_REVIEW),('handoff_status',GROWTH_HANDOFF),('stage',GROWTH_STAGES)]:
        if not result[col].isin(options).all():
            return growth_empty_opportunities(), [f'Invalid {col} in imported opportunity log.']
    for _, o in result.iterrows():
        expected_csm = str(accounts.loc[accounts['account_id'].astype(str)==o['account_id'], 'csm_name'].iloc[0])
        if o['csm_name'] != expected_csm:
            return growth_empty_opportunities(), ['CSM ownership mismatch in opportunity log.']
        if o['review_status'] == 'Approved for discovery' and (o['reviewed_by'] != expected_csm or not o['reviewed_at']):
            return growth_empty_opportunities(), ['Missing CSM attestation in approved opportunity.']
        if o['handoff_status'] != 'Not sent' and (o['review_status'] != 'Approved for discovery' or not o['ae_owner']):
            return growth_empty_opportunities(), ['Sales handoff requires CSM approval and AE owner.']
        if o['stage'] == 'Contracted':
            amount = pd.to_numeric(o['contracted_incremental_mrr_eur'], errors='coerce')
            if not (o['handoff_status'] == 'Accepted by AE' and pd.notna(amount) and amount > 0 and o['contract_evidence_reference']):
                return growth_empty_opportunities(), ['Contracted opportunity requires AE acceptance, positive incremental MRR and documentary reference.']
    return result, []


def growth_opportunity_brief(account, signal, assessment):
    """Single customer-led diagnostic tied directly to imported source fields."""
    readiness = str(assessment.get('readiness', 'Evidence needed'))
    cls = 'ready' if assessment.get('ready') else 'hold'
    value_pct = assessment.get('value_progress')
    goal = 'Not measured' if value_pct is None else f'{value_pct:.0f}%'
    blockers = pilot_safe_number(account.get('open_blockers'))
    blocker_text = 'Unreported' if pd.isna(blockers) else str(int(blockers))
    return (
        f'<div class="v17-growth-brief {cls}">'
        '<div class="v17-growth-header"><div><span class="v17-growth-kicker">ACCOUNT OPPORTUNITY REVIEW</span>'
        f'<h3>{html.escape(str(account.get("account_name", "Account")))}</h3></div>'
        f'<div class="v17-growth-badge">{html.escape(readiness)}</div></div>'
        '<div class="v17-growth-kpis">'
        f'<div><small>GOAL PROGRESS</small><strong>{html.escape(goal)}</strong></div>'
        f'<div><small>OPEN BLOCKERS</small><strong>{html.escape(blocker_text)}</strong></div>'
        f'<div><small>CANDIDATE SOLUTION</small><strong>{html.escape(str(signal.get("candidate_toolkit") or "Unspecified"))}</strong></div>'
        '</div>'
        '<div class="v17-growth-details"><span>DOCUMENTED CUSTOMER NEED</span>'
        f'<p>{html.escape(str(signal.get("customer_need") or "Customer need not documented"))}</p></div>'
        '</div>'
    )

def growth_render():
    st.markdown('## Growth Intelligence')
    st.caption('Discover customer-led growth opportunities · validate product fit · align CS and Sales · follow the commercial result')
    if st.button('Open Revenue Lifecycle →',key='growth_open_lifecycle'):
        st.session_state['active_view']='End-to-End Flow (MVP 2.1.1)'
        st.rerun()
    st.warning('**Restricted data workspace.** Customer information must remain in an approved internal environment. Typed names are acknowledgements, not identity verification.')
    if os.environ.get('SEMRUSH_INTERNAL_PILOT','0') != '1':
        st.info('Customer data operations are unavailable in public sessions. Open the app in an approved internal environment to use this workspace.')
        return
    if not st.checkbox('I confirm that this environment and the account data are company-authorized.',key='mvp21_authorized'):
        st.info('Confirm the controlled environment to load account signals.')
        return
    if 'mvp20_accounts' not in st.session_state: st.session_state['mvp20_accounts'] = pilot_empty_accounts()
    if 'mvp21_signals' not in st.session_state: st.session_state['mvp21_signals'] = growth_empty_signals()
    if 'mvp21_opps' not in st.session_state: st.session_state['mvp21_opps'] = growth_empty_opportunities()
    accounts = st.session_state['mvp20_accounts']
    if accounts.empty:
        st.info('No accounts are connected. Open **Customer Value & Actions** to load your authorized account data, then return to Growth Intelligence.')
        if st.button('Open Customer Value', key='mvp21_goto_pilot'):
            st.session_state['active_view']='Real-data Pilot (MVP 2.0)'
            st.rerun()
        return
    tabs=st.tabs(['Growth Radar','Signal & Product Fit','CSM → Sales (guided)','Expansion Outcome (guided)','Governance & Exports'])
    with tabs[0]:
        st.markdown('### Growth opportunity radar')
        st.caption('Prioritize confirmed needs and healthy customer outcomes. Recovery needs stay visible rather than being treated as expansion targets.')
        signals = st.session_state['mvp21_signals']
        frame=[]
        for _, a in accounts.iterrows():
            matched=signals.loc[signals['account_id']==a['account_id']]
            if matched.empty:
                frame.append({'Account':a['account_name'],'CSM':a['csm_name'],'Candidate toolkit':'—',
                              'Readiness':'No signal submitted','Goal progress %':pilot_progress_pct(a),
                              'Signal evidence':'—','Unresolved blockers':a['open_blockers']})
                continue
            g=matched.iloc[0]; assessment=growth_evaluate(g,a)
            frame.append({'Account':a['account_name'],'CSM':a['csm_name'],
                          'Candidate toolkit':g['candidate_toolkit'] or '—','Readiness':assessment['readiness'],
                          'Goal progress %':None if assessment['value_progress'] is None else round(assessment['value_progress'],1),
                          'Signal evidence':g['need_evidence_reference'] or '—',
                          'Unresolved blockers':a['open_blockers']})
        snapshot=pd.DataFrame(frame)
        c1,c2,c3=st.columns(3)
        c1.metric('Accounts considered',len(snapshot),'Max 3')
        c2.metric('Ready for CSM review',int((snapshot['Readiness']=='Ready for CSM assessment').sum()),'Not sales-qualified yet')
        c3.metric('Value recovery first',int((snapshot['Readiness']=='Value recovery first').sum()),'Customer success before expansion')
        ready_rows = snapshot.loc[snapshot['Readiness'] == 'Ready for CSM assessment']
        recovery_rows = snapshot.loc[snapshot['Readiness'].isin(['Value recovery first', 'Value-dependent review'])]
        evidence_rows = snapshot.drop(ready_rows.index.union(recovery_rows.index))
        st.markdown('#### Ready for CSM qualification')
        if ready_rows.empty:
            st.info('No accounts meet the current value and evidence requirements. Confirm customer outcomes before advancing.')
        else:
            st.dataframe(ready_rows, use_container_width=True, hide_index=True)
        st.markdown('#### Customer value recovery and review')
        if recovery_rows.empty:
            st.success('No accounts currently require a recovery-first review in this selection.')
        else:
            st.dataframe(recovery_rows, use_container_width=True, hide_index=True)
        if not evidence_rows.empty:
            with st.expander('Needs more evidence', expanded=False):
                st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
        st.caption('Readiness is a transparent workflow eligibility rule, not a purchase-probability score. Unresolved value gaps remain visible.')
        if not signals.empty:
            st.markdown('#### Account opportunity briefing')
            selected=st.selectbox('Inspect growth hypothesis',signals['account_id'].tolist(),key='mvp21_inspect')
            g=signals.loc[signals.account_id==selected].iloc[0]
            a=accounts.loc[accounts.account_id==selected].iloc[0]
            assessment=growth_evaluate(g,a)
            st.markdown(growth_opportunity_brief(a,g,assessment),unsafe_allow_html=True)
            st.write('**Customer need:**', g['customer_need'] or 'Not documented')
            st.write('**Candidate solution:**', g['candidate_toolkit'] or 'Not documented')
            st.write('**Evidence:**', g['need_source_system'] or '—', ' / ', g['need_evidence_reference'] or '—')
            st.write('**Observed signals:**', '; '.join(assessment['signals']) if assessment['signals'] else 'No strong growth trigger recorded')
            st.info(f'**{assessment["readiness"]}** — {assessment["gate"]}')
            st.markdown('**Next best action**')
            if assessment['ready']:
                st.success('CSM: run needs discovery and verify customer interest. Only propose an AE handoff after CSM approval.')
            elif assessment['readiness'].startswith('Value'):
                st.warning('CSM / Support: focus on outcomes and resolve blockers; do not automatically upsell an unresolved problem.')
            else:
                st.write('CSM: document the customer need, solution fit and evidence; revisit when evidence is sufficient.')
    with tabs[1]:
        st.markdown('### Map growth signals to explicit, evidenced customer needs')
        st.write('The mapping is **product-neutral**: enter the candidate toolkit only if a verified need indicates an appropriate solution. No automatic cross-sell assumptions.')
        template=pd.DataFrame([{c:'' for c in GROWTH_COLUMNS} for _ in range(3)])
        st.download_button('Download blank growth-signal CSV',pilot_csv_bytes(template),'mvp21_growth_signals_blank.csv',mime='text/csv',key='mvp21_blank')
        upl=st.file_uploader('Import growth signals (one row per existing account)',type='csv',key='mvp21_growth_upload')
        if upl is not None and st.button('Validate & load growth signals',key='mvp21_load_growth'):
            try:
                incoming=pd.read_csv(upl,dtype={'account_id':str}).dropna(how='all')
                incoming=incoming[incoming['account_id'].notna() & incoming['account_id'].str.strip().ne('')]
                cleaned,errors=growth_clean_signals(incoming,accounts)
                if errors:
                    for e in errors: st.error(e)
                else:
                    opps=st.session_state['mvp21_opps']
                    removed=set(opps['account_id'].astype(str))-set(cleaned['account_id'].astype(str)) if not opps.empty else set()
                    if removed: st.error('Previously created opportunities refer to removed signal accounts. Export the log first; keep rows when reimporting.')
                    else:
                        st.session_state['mvp21_signals']=cleaned
                        st.success(f'{len(cleaned)} verified-format signal rows loaded. Evidence is attested, not independently audited.')
                        st.rerun()
            except Exception as ex:
                st.error(f'Unable to parse growth signals: {ex}')
        signals=st.session_state['mvp21_signals']
        if not signals.empty:
            visible=['account_id','customer_need','candidate_toolkit','customer_confirmed_need','product_fit_confirmed',
                    'capacity_usage_pct','need_evidence_date','need_evidence_reference']
            st.dataframe(signals[visible],hide_index=True,use_container_width=True)
            st.download_button('Export current growth signals',pilot_csv_bytes(signals),'mvp21_growth_signals_working.csv',mime='text/csv',key='mvp21_export_signals')
    with tabs[2]:
        st.markdown('### CSM qualification & Sales handoff')
        st.info('The Revenue Lifecycle workspace sequences CSM approval, AE ownership, evidence and commercial updates, keeping qualification and Sales handoff aligned.')
        if st.button('Open guided CSM → Sales flow',key='growth_goto_governed_sales'):
            st.session_state['active_view']='End-to-End Flow (MVP 2.1.1)'
            st.rerun()
        opps=st.session_state['mvp21_opps']
        if not opps.empty:
            st.dataframe(opps[['opportunity_id','account_name','candidate_toolkit','review_status','handoff_status','stage']],use_container_width=True,hide_index=True)
    with tabs[3]:
        st.markdown('### Expansion results & ongoing customer value')
        st.info('Record signed incremental monthly revenue only after AE acceptance; then add a documented subsequent customer value measurement in the guided workflow.')
        if st.button('Open guided Expansion Outcome',key='growth_goto_governed_outcome'):
            st.session_state['active_view']='End-to-End Flow (MVP 2.1.1)'
            st.rerun()
        stats=growth_metrics(st.session_state['mvp21_opps'])
        aa,bb,cc=st.columns(3)
        aa.metric('Proposed',stats['proposed'])
        bb.metric('CSM approved',stats['csm_approved'])
        cc.metric('Documented contracted MRR',f'€{stats["verified_incremental_mrr_eur"]:,.0f}/mo')
        st.caption('Not an audited revenue report and not evidence of causal CS impact.')
    with tabs[4]:
        st.markdown('### Controls & export / restore')
        st.markdown('''**Expansion policy:** documented customer outcome must take precedence over revenue potential. The radar does not suggest automatic selling to underperforming accounts.

**Signal confidence:** evidence must identify a system, internal source reference, owner and date within 45 days. Values are attested by the user, not independently checked against source systems.

**CSM approval:** assigned CSM name is checked as typed text only. **Not identity verification.** AE statuses are also self-reported; no CRM integration.

**Revenue recognition:** the workspace counts only incremental *monthly* contracted amount with a documented agreement and accepted AE handoff. This is user-declared, not audited revenue, and not proof CS caused that expansion.

**Revenue Lifecycle:** opportunity creation, CSM decision, AE acceptance, contract and post-sale value measurement follow a controlled sequence in **Revenue Lifecycle**.

**Security / retention:** do not enable on a publicly accessible app. No enterprise RBAC or persistent server-side storage. Export files must stay in approved internal storage. No real account data is included in the downloadable demo files.''')
        signals=st.session_state['mvp21_signals']; opps=st.session_state['mvp21_opps']
        if not signals.empty:
            st.download_button('Export growth evidence CSV',pilot_csv_bytes(signals),
                'mvp21_growth_signals_working.csv',mime='text/csv',key='mvp21_gov_signals')
        if not opps.empty:
            st.download_button('Export opportunity log CSV · SAVE YOUR WORK',pilot_csv_bytes(opps),
                'mvp21_growth_opportunities_working.csv',mime='text/csv',key='mvp21_gov_opps')
        st.info('For a full restore including approvals, sales decisions, post-value measurements and the activity log, use **Revenue Lifecycle** → **Evidence & activity** → **Restore snapshot ZIP**. Legacy opportunity CSV restore is disabled to prevent loss of lifecycle fields.')


# =========================
# MVP 2.1.1 – E2E CUSTOMER VALUE → GROWTH → CSM → SALES → OUTCOME
# Reuses the original pilot dataframes; governed actions can ONLY advance through
# the explicit transitions below. Session log is NOT an immutable audit system.
# =========================
import zipfile

FLOW_EXTRA_COLUMNS = [
    'proposal_customer_benefit', 'ae_decision_by', 'ae_decision_at',
    'ae_decision_note', 'stage_reference', 'stage_updated_at',
    'post_goal_value', 'post_measurement_as_of', 'post_value_evidence_reference',
    'post_validated_by', 'post_value_note',
]
FLOW_EVENT_COLUMNS = ['event_id','timestamp_utc','account_id','opportunity_id',
                      'step','transition','actor_attested','reference','detail']
FLOW_STAGE_SEQUENCE = ['Discovery', 'Need validated', 'Demo completed', 'Proposal']
FLOW_STEPS = ['Customer Value', 'Growth Signal', 'CSM Approval', 'Sales Handoff', 'Expansion Outcome']


def flow_expand_opps(frame):
    """Non-destructive upgrade of the existing V14 session opportunity schema."""
    out = frame.copy()
    for col in FLOW_EXTRA_COLUMNS:
        if col not in out.columns:
            out[col] = ''
    return out


def flow_evidence_status(account):
    state = pilot_evidence_check(account)
    goal = pilot_progress_pct(account)
    adoption = pilot_adoption_pct(account)
    blockers = pilot_safe_number(account.get('open_blockers'))
    enough = (state == 'Evidence documented (not independently verified)' and goal is not None)
    viable = enough and goal >= 80 and not (pd.notna(blockers) and blockers > 0) and (adoption is None or adoption >= 40)
    if not enough:
        return {'documented':False, 'commercial_ready':False, 'message':state,
                'goal':goal,'adoption':adoption}
    if not viable:
        return {'documented':True, 'commercial_ready':False,
                'message':'Value recovery required: pilot gate expects ≥80% goal progress, no blockers and ≥40% active-seat coverage if measured.',
                'goal':goal,'adoption':adoption}
    return {'documented':True,'commercial_ready':True,'message':'Outcome evidence attested and pilot value gate satisfied.',
            'goal':goal,'adoption':adoption}


def flow_step_status(account, signal=None, opp=None):
    """A display-only stage map: no probabilistic predictions and no hardcoded success."""
    value = flow_evidence_status(account)
    growth = growth_evaluate(signal, account) if signal is not None else None
    stages = ['waiting'] * 5
    stages[0] = 'complete' if value['documented'] else 'attention'
    stages[1] = ('complete' if growth and growth['ready'] else 'attention'
                 if signal is not None else 'waiting')
    if opp is None:
        return stages
    review = str(opp.get('review_status',''))
    handoff = str(opp.get('handoff_status',''))
    stage = str(opp.get('stage',''))
    stages[2] = ('complete' if review == 'Approved for discovery' else 'attention'
                 if review in ['Rejected','On hold'] else 'in progress')
    if review == 'Approved for discovery':
        stages[3] = ('complete' if handoff == 'Accepted by AE' else 'attention'
                     if handoff == 'Declined by AE' else 'in progress')
    if handoff == 'Accepted by AE':
        stages[4] = ('complete' if stage == 'Lost' or (stage == 'Contracted' and str(opp.get('post_value_evidence_reference','')).strip()) else 'in progress')
    return stages


def flow_transition(row, transition, account, growth, actor, reference='', notes='',
                    ae_owner='', amount=None, goal_value=None, observed_at=''):
    """Pure, strict transition validator returning a changed opportunity dict.

    Actor names are only self-attested in this MVP: never advertise as SSO/RBAC.
    """
    new = dict(row)
    for col in FLOW_EXTRA_COLUMNS:
        new.setdefault(col, '')
    review = str(new.get('review_status',''))
    handoff = str(new.get('handoff_status',''))
    stage = str(new.get('stage',''))
    actor, reference, notes = str(actor).strip(), str(reference).strip(), str(notes).strip()
    now = pilot_now()
    if not notes:
        raise ValueError('Document a decision rationale or an outcome note.')
    if transition in ['approve', 'hold', 'reject']:
        if review != 'Pending CSM':
            raise ValueError('Only a pending proposal can receive its first CSM decision.')
        if actor != str(account['csm_name']).strip():
            raise ValueError('CSM name must exactly match the account owner (self-attested).')
        if transition == 'approve' and (growth is None or not growth_evaluate(growth, account)['ready']):
            raise ValueError('Cannot approve: current value / growth readiness evidence fails its gate.')
        new['review_status'] = {'approve':'Approved for discovery','hold':'On hold','reject':'Rejected'}[transition]
        new['reviewed_by'], new['reviewed_at'], new['review_note'] = actor, now, notes
    elif transition == 'send_to_ae':
        if review != 'Approved for discovery' or handoff != 'Not sent':
            raise ValueError('Sales handoff requires CSM approval and must first be marked Sent to AE.')
        if growth is None or not growth_evaluate(growth, account)['ready']:
            raise ValueError('Current value / growth evidence must still satisfy the eligibility gate before Sales handoff.')
        ae_owner = str(ae_owner).strip()
        if not ae_owner or not reference:
            raise ValueError('AE owner and CRM/handoff reference are required.')
        if actor != str(account['csm_name']).strip():
            raise ValueError('Handoff sender must match assigned CSM (self-attested).')
        new.update(handoff_status='Sent to AE', ae_owner=ae_owner,
                   handoff_evidence_reference=reference, handoff_at=now)
    elif transition in ['accept_ae','decline_ae']:
        if review != 'Approved for discovery' or handoff != 'Sent to AE':
            raise ValueError('An AE can only accept or decline an opportunity already sent to Sales.')
        if not actor or actor != str(new.get('ae_owner','')).strip():
            raise ValueError('AE name must exactly match the assigned Sales owner (self-attested).')
        if not reference:
            raise ValueError('An AE decision requires a CRM/source reference.')
        new['handoff_status'] = 'Accepted by AE' if transition == 'accept_ae' else 'Declined by AE'
        new.update(ae_decision_by=actor, ae_decision_at=now, ae_decision_note=notes,
                   handoff_evidence_reference=reference)
    elif transition.startswith('stage:'):
        next_stage = transition.split(':',1)[1]
        if review != 'Approved for discovery' or handoff != 'Accepted by AE':
            raise ValueError('Commercial pipeline stages require CSM approval and AE acceptance.')
        if next_stage not in FLOW_STAGE_SEQUENCE:
            raise ValueError('Invalid pipeline stage.')
        if stage not in FLOW_STAGE_SEQUENCE or FLOW_STAGE_SEQUENCE.index(next_stage) < FLOW_STAGE_SEQUENCE.index(stage):
            raise ValueError('Pipeline stages cannot move backwards.')
        if not actor or actor != str(new.get('ae_owner','')).strip() or not reference:
            raise ValueError('Commercial stage update requires named AE and CRM evidence reference.')
        new.update(stage=next_stage, stage_reference=reference, stage_updated_at=now)
    elif transition in ['contracted','lost']:
        if review != 'Approved for discovery' or handoff != 'Accepted by AE':
            raise ValueError('A commercial outcome requires approved CSM handoff and AE acceptance.')
        if stage in ['Contracted','Lost']:
            raise ValueError('This opportunity is already closed. Closed deals cannot be overwritten.')
        if not actor or actor != str(new.get('ae_owner','')).strip():
            raise ValueError('Commercial close must be attested by the assigned AE.')
        if not reference:
            raise ValueError('Closure requires CRM / contract / billing evidence reference.')
        if transition == 'contracted':
            actual = pd.to_numeric(amount, errors='coerce')
            if pd.isna(actual) or actual <= 0 or not np.isfinite(actual):
                raise ValueError('Actual *incremental monthly* contracted MRR must be positive.')
            new.update(stage='Contracted',contracted_incremental_mrr_eur=float(actual),
                       contract_evidence_reference=reference)
        else:
            new.update(stage='Lost',contracted_incremental_mrr_eur='',contract_evidence_reference='')
        new['outcome_note'], new['closed_at'] = notes, now
    elif transition == 'post_value':
        if stage != 'Contracted':
            raise ValueError('Post-expansion customer value can be recorded only after a contracted outcome.')
        if str(new.get('post_value_evidence_reference','')).strip():
            raise ValueError('A post-expansion measurement is already recorded. Export and review before changing it.')
        val = pd.to_numeric(goal_value, errors='coerce')
        when = pd.to_datetime(observed_at, errors='coerce')
        baseline_dt = pd.to_datetime(account.get('measurement_as_of'), errors='coerce')
        close_dt = pd.to_datetime(new.get('closed_at'), errors='coerce', utc=True)
        if pd.isna(val) or not np.isfinite(val) or pd.isna(when):
            raise ValueError('A numerical outcome value and valid YYYY-MM-DD measurement date are required.')
        if when.date() > date.today():
            raise ValueError('Do not use future measurement dates.')
        if pd.notna(baseline_dt) and when.date() < baseline_dt.date():
            raise ValueError('Post-value measurement cannot predate the original account measurement.')
        if pd.notna(close_dt) and when.date() < close_dt.date():
            raise ValueError('Post-value measurement cannot predate the commercial close.')
        if not reference or not actor:
            raise ValueError('Outcome measurement needs a source reference and named validator.')
        new.update(post_goal_value=float(val),post_measurement_as_of=when.strftime('%Y-%m-%d'),
                   post_value_evidence_reference=reference,post_validated_by=actor,post_value_note=notes)
    else:
        raise ValueError('Unknown transition: '+str(transition))
    return new


def flow_record_transition(opp_id, transition, account, signal=None, **kwargs):
    """Update session state atomically after validation and append an event row."""
    opps = flow_expand_opps(st.session_state['mvp21_opps'])
    matches = opps.index[opps['opportunity_id'] == opp_id]
    if len(matches) != 1:
        raise ValueError('Opportunity is missing or duplicated.')
    i = matches[0]
    before = opps.loc[i].to_dict()
    if str(before['account_id']) != str(account['account_id']):
        raise ValueError('Opportunity account mismatch.')
    updated = flow_transition(before,transition,account,signal,**kwargs)
    for field, value in updated.items():
        opps.at[i,field] = value
    st.session_state['mvp21_opps'] = opps
    flow_event(before['account_id'],opp_id,transition,kwargs.get('actor',''),
               kwargs.get('reference',''),kwargs.get('notes',''))


def flow_event(account_id, opp_id, transition, actor, reference='', detail=''):
    if 'mvp211_events' not in st.session_state:
        st.session_state['mvp211_events'] = pd.DataFrame(columns=FLOW_EVENT_COLUMNS)
    step = ('CSM Approval' if transition in ['propose','approve','hold','reject'] else
            'Sales Handoff' if transition in ['send_to_ae','accept_ae','decline_ae'] or transition.startswith('stage:') else
            'Expansion Outcome')
    event = {'event_id':'EV-'+uuid.uuid4().hex[:12].upper(), 'timestamp_utc':pilot_now(),
             'account_id':account_id,'opportunity_id':opp_id,'step':step,
             'transition':transition,'actor_attested':actor,'reference':reference,'detail':detail}
    st.session_state['mvp211_events'] = pd.concat([st.session_state['mvp211_events'],pd.DataFrame([event])],ignore_index=True)


def flow_post_value_delta(opp, account):
    """Observed outcome-unit change, not causal impact or an ROI percentage."""
    if str(opp.get('stage','')) != 'Contracted' or not str(opp.get('post_value_evidence_reference','')).strip():
        return None
    pre = pilot_safe_number(account.get('goal_current'))
    post = pilot_safe_number(opp.get('post_goal_value'))
    if pd.isna(pre) or pd.isna(post):
        return None
    direction = str(account.get('goal_direction','')).lower()
    return float(post-pre if direction=='increase' else pre-post if direction=='decrease' else np.nan)


def flow_export_zip():
    buf=io.BytesIO()
    state = [('01_pilot_accounts.csv', st.session_state.get('mvp20_accounts',pilot_empty_accounts())),
             ('02_growth_signals.csv',st.session_state.get('mvp21_signals',growth_empty_signals())),
             ('03_opportunities.csv',flow_expand_opps(st.session_state.get('mvp21_opps',growth_empty_opportunities()))),
             ('04_activity_log.csv',st.session_state.get('mvp211_events',pd.DataFrame(columns=FLOW_EVENT_COLUMNS))),
             ('05_cs_impact_actions.csv',st.session_state.get('mvp20_actions',pilot_empty_actions()))]
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name, frame in state:
            z.writestr(name,pilot_csv_bytes(frame))
        z.writestr('README.txt','Customer Value & Growth workspace snapshot. May contain confidential customer data. Store only in approved internal locations. Names are self-attested, with no SSO or tamper-resistant audit. Never upload to a public repository.\n')
    return buf.getvalue()


def flow_restore_snapshot(zipped_bytes):
    """Validate all 4 files before replacing any current in-session pilot state."""
    if len(zipped_bytes) > 8_000_000:
        raise ValueError('Workspace snapshot exceeds the 8 MB upload limit.')
    wanted=['01_pilot_accounts.csv','02_growth_signals.csv',
            '03_opportunities.csv','04_activity_log.csv']
    try:
        with zipfile.ZipFile(io.BytesIO(zipped_bytes),'r') as z:
            if not all(name in z.namelist() for name in wanted):
                raise ValueError('This is not a complete workspace snapshot.')
            if any(z.getinfo(name).file_size > 2_000_000 for name in wanted):
                raise ValueError('Unexpectedly large CSV in the workspace snapshot.')
            frames={name:pd.read_csv(io.BytesIO(z.read(name)),dtype=str,
                                      keep_default_na=False,encoding='utf-8-sig') for name in wanted}
    except (zipfile.BadZipFile,KeyError,UnicodeError) as exc:
        raise ValueError(f'Cannot read snapshot: {exc}') from exc
    raw_accounts=frames[wanted[0]]
    raw_accounts=raw_accounts.loc[raw_accounts['account_id'].astype(str).str.strip().ne('')] if 'account_id' in raw_accounts else raw_accounts
    if raw_accounts.empty:
        raise ValueError('Workspace snapshot must contain at least one account.')
    accounts,errors=pilot_clean_accounts(raw_accounts)
    if errors:
        raise ValueError('Account validation: '+'; '.join(errors))
    raw_signals=frames[wanted[1]]
    if not raw_signals.empty:
        if 'account_id' not in raw_signals:
            raise ValueError('Signal CSV missing account_id.')
        raw_signals=raw_signals.loc[raw_signals['account_id'].astype(str).str.strip().ne('')]
    if raw_signals.empty:
        signals=growth_empty_signals()
    else:
        signals,errors=growth_clean_signals(raw_signals,accounts)
        if errors: raise ValueError('Growth signal validation: '+'; '.join(errors))
    raw_opps=frames[wanted[2]]
    if not raw_opps.empty:
        if 'opportunity_id' not in raw_opps:
            raise ValueError('Opportunity CSV missing opportunity_id.')
        raw_opps=raw_opps.loc[raw_opps['opportunity_id'].astype(str).str.strip().ne('')]
    if raw_opps.empty:
        opportunities=flow_expand_opps(growth_empty_opportunities())
    else:
        _,errors=growth_import_opps(raw_opps,accounts)
        if errors: raise ValueError('Opportunity validation: '+'; '.join(errors))
        if not set(raw_opps['account_id'].astype(str)).issubset(set(signals['account_id'].astype(str))):
            raise ValueError('Opportunity lacks its matching growth signal.')
        opportunities=flow_expand_opps(raw_opps)
    raw_events=frames[wanted[3]]
    if raw_events.empty:
        events=pd.DataFrame(columns=FLOW_EVENT_COLUMNS)
    else:
        if not set(FLOW_EVENT_COLUMNS).issubset(raw_events.columns):
            raise ValueError('Activity log lacks required columns.')
        events=raw_events[FLOW_EVENT_COLUMNS].copy()
        if events['event_id'].duplicated().any() or events['event_id'].eq('').any():
            raise ValueError('Activity log contains invalid or repeated event IDs.')
        if not set(events['account_id']).issubset(set(accounts['account_id'])):
            raise ValueError('Activity log refers to unknown account IDs.')
        if not set(events['opportunity_id']).issubset(set(opportunities['opportunity_id'])):
            raise ValueError('Activity log refers to unknown opportunity IDs.')
    # A fifth file preserves the older MVP 2.0 Next Best Action / CS Impact work.
    # It is optional for snapshots created before this addition.
    impact_actions=pilot_empty_actions()
    try:
        with zipfile.ZipFile(io.BytesIO(zipped_bytes),'r') as z:
            if '05_cs_impact_actions.csv' in z.namelist():
                if z.getinfo('05_cs_impact_actions.csv').file_size > 2_000_000:
                    raise ValueError('Customer Success action records exceed the current workspace limit.')
                raw_actions=pd.read_csv(io.BytesIO(z.read('05_cs_impact_actions.csv')),
                                        dtype=str,keep_default_na=False,encoding='utf-8-sig')
                if not raw_actions.empty:
                    impact_actions,action_issues=pilot_import_actions(raw_actions,accounts)
                    if action_issues:
                        raise ValueError('CS Impact action validation: '+'; '.join(action_issues))
    except (zipfile.BadZipFile,KeyError,UnicodeError) as exc:
        raise ValueError(f'Cannot read action log: {exc}') from exc
    return accounts,signals,opportunities,events,impact_actions


def flow_demo_seed():
    """The fictional 3-account demo is never loaded into public-only mode."""
    base=Path(__file__).resolve().parent
    a_path=base/'mvp21_DEMO_FAKE_accounts.csv'
    g_path=base/'mvp21_DEMO_FAKE_growth_signals.csv'
    if not a_path.exists() or not g_path.exists():
        raise ValueError('The example account files are missing. Extract the complete application bundle to open the example journeys.')
    a=pd.read_csv(a_path,dtype={'account_id':str})
    g=pd.read_csv(g_path,dtype={'account_id':str})
    today=date.today().isoformat()
    # Dates of the synthetic simulation are scenario dates, NOT verified customer data.
    for col in ['measurement_as_of','validation_date']:
        a[col]=today
    g['need_evidence_date']=today
    a,ae=pilot_clean_accounts(a)
    g,ge=growth_clean_signals(g,a)
    if ae or ge:
        raise ValueError('; '.join(ae+ge))
    st.session_state.update(mvp20_accounts=a, mvp20_actions=pilot_empty_actions(),
                            mvp21_signals=g, mvp21_opps=growth_empty_opportunities(),
                            mvp211_events=pd.DataFrame(columns=FLOW_EVENT_COLUMNS))



LIFECYCLE_STAGE_EXPLAINERS = [
    ("Customer value", "Verify progress against an agreed, measurable outcome."),
    ("Growth signal", "Validate customer demand, supporting evidence and solution fit."),
    ("CSM decision", "Record the assigned CSM’s decision and business rationale."),
    ("Sales handoff", "Share context, assign ownership and record the AE response."),
    ("Expansion outcome", "Document the commercial outcome and measure post-sale value."),
]

def render_lifecycle_stepper(stages):
    """Accessible static status indicator, with no reliance on motion or color alone."""
    states = {"complete": ("Complete", "✓"), "in progress": ("In progress", "→"),
              "attention": ("Needs attention", "!"), "waiting": ("Not started", "·")}
    cards = []
    for i, ((title, explanation), current) in enumerate(zip(LIFECYCLE_STAGE_EXPLAINERS, stages)):
        status, symbol = states.get(current, states["waiting"])
        cards.append(
            f'<div class="v17-flow-stage is-{current.replace(" ", "-")}">'
            f'<div class="v17-flow-top"><span class="v17-flow-num">{i+1:02d}</span>'
            f'<span class="v17-flow-state">{html.escape(symbol)} {html.escape(status)}</span></div>'
            f'<div class="v17-flow-title">{html.escape(title)}</div>'
            f'<div class="v17-flow-desc">{html.escape(explanation)}</div></div>'
        )
    return '<div class="v17-flow-steps">' + ''.join(cards) + '</div>'


def lifecycle_next_action(account, signal, opp, value, readiness):
    """Only returns supported guidance derived from current lifecycle status."""
    if not value['documented']:
        return ('Document the customer outcome', 'Confirm the goal, measurement date, data source and source reference before opening a growth opportunity.')
    if not value['commercial_ready']:
        return ('Recover customer value first', 'Resolve blockers or missing adoption outcomes, then reassess the opportunity with the customer.')
    if signal is None:
        return ('Capture a growth signal', 'Record the customer need, evidence source, appropriate solution and evidence date in Growth Intelligence.')
    if readiness is None or not readiness['ready']:
        return ('Validate the growth hypothesis', 'Check need confirmation, product fit, current evidence and the readiness gate before CSM review.')
    if opp is None:
        return ('Submit for CSM review', 'Create a customer-benefit proposal, then obtain an explicit decision from the assigned CSM.')
    if str(opp.get('review_status')) == 'Pending CSM':
        return ('Record the CSM decision', 'Approve, hold or reject the proposal with a clear rationale tied to customer value.')
    if str(opp.get('review_status')) != 'Approved for discovery':
        return ('Review the CSM decision', 'The opportunity cannot advance. Reassess the customer need before creating another proposal.')
    handoff = str(opp.get('handoff_status'))
    if handoff == 'Not sent':
        return ('Prepare the Sales handoff', 'Assign an AE, include the customer-benefit narrative and record the CRM reference.')
    if handoff == 'Sent to AE':
        return ('Request the AE response', 'Sales must accept or decline the handoff and provide a decision reference.')
    if handoff != 'Accepted by AE':
        return ('Close the handoff decision', 'Sales declined the opportunity. Document why and protect the customer outcome.')
    stage = str(opp.get('stage'))
    if stage == 'Lost':
        return ('Capture the learning', 'Review the reason for loss and update the customer plan. No incremental revenue is counted.')
    if stage == 'Contracted':
        if str(opp.get('post_value_evidence_reference', '')).strip():
            return ('Review post-expansion value', 'The commercial outcome and subsequent customer result have been recorded. Use the activity log for evidence.')
        return ('Measure value after expansion', 'Record the observed customer outcome after contract signature with a dated source reference.')
    return ('Advance the opportunity responsibly', 'Keep the CRM stage current and only record contracted incremental MRR after a completed agreement.')

def v171_recommended_flow_stage(account, signal, opp, value, readiness):
    """Identify the next actionable stage from recorded evidence; never bypass workflow gates."""
    if not value['documented'] or not value['commercial_ready']:
        return 0
    if signal is None or readiness is None or not readiness['ready']:
        return 1
    if opp is None or str(opp.get('review_status', '')) != 'Approved for discovery':
        return 2
    if str(opp.get('handoff_status', '')) != 'Accepted by AE':
        return 3
    return 4


def flow_render():
    st.markdown('## Revenue Lifecycle')
    st.caption('Customer outcomes → qualified growth signals → CSM decision → Sales coordination → documented commercial result')
    st.warning('**Restricted customer data.** This workspace requires an approved internal environment. Approvals are self-attested; SSO, role-based permissions and an immutable audit log are not connected.')
    if os.environ.get('SEMRUSH_INTERNAL_PILOT','0')!='1':
        st.info('Customer lifecycle records are available only in an approved internal environment. The public portfolio view remains separate.')
        return
    if not st.checkbox('I confirm authorization to work with these customer records in this environment.',key='mvp211_authorized'):
        st.info('Confirm authorized access to continue to the lifecycle workspace.')
        return
    if 'mvp20_accounts' not in st.session_state: st.session_state['mvp20_accounts']=pilot_empty_accounts()
    if 'mvp21_signals' not in st.session_state: st.session_state['mvp21_signals']=growth_empty_signals()
    if 'mvp21_opps' not in st.session_state: st.session_state['mvp21_opps']=growth_empty_opportunities()
    if 'mvp211_events' not in st.session_state: st.session_state['mvp211_events']=pd.DataFrame(columns=FLOW_EVENT_COLUMNS)
    st.session_state['mvp21_opps']=flow_expand_opps(st.session_state['mvp21_opps'])
    accounts=st.session_state['mvp20_accounts']
    if accounts.empty:
        st.markdown('### Open a customer journey')
        st.write('Explore three synthetic customer journeys: recovery, value qualification and a growth opportunity. No real customer information is included.')
        if st.button('Explore three example accounts',key='flow_seed_demo',type='primary'):
            try:
                flow_demo_seed()
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if st.button('Connect authorized accounts',key='flow_to_real'):
            st.session_state['active_view']='Real-data Pilot (MVP 2.0)'
            st.rerun()
        return
    if st.session_state['mvp21_signals'].empty:
        st.info('Accounts are connected, but growth signals are missing. Open **Growth Intelligence** → **Signal & Product Fit** to import supporting evidence.')
        if st.button('Open signal import',key='flow_signal_import'):
            st.session_state['active_view']='Growth & Expansion (MVP 2.1)'
            st.rerun()
    account_ids=accounts['account_id'].astype(str).tolist()
    aid=st.selectbox('Select the account to follow through the 5 stages',account_ids,
                     format_func=lambda x: str(accounts.loc[accounts['account_id']==x,'account_name'].iloc[0]),key='mvp211_account')
    account=accounts.loc[accounts['account_id']==aid].iloc[0]
    signals=st.session_state['mvp21_signals']
    matched=signals.loc[signals['account_id'].astype(str)==aid] if not signals.empty else signals
    signal=matched.iloc[0] if not matched.empty else None
    opps=st.session_state['mvp21_opps']
    customer_opps=opps.loc[opps['account_id'].astype(str)==aid] if not opps.empty else opps
    selected_opp=None
    if not customer_opps.empty:
        oid=st.selectbox('Opportunity record',customer_opps['opportunity_id'].tolist(),key='mvp211_selected_opp')
        selected_opp=customer_opps.loc[customer_opps['opportunity_id']==oid].iloc[0]
    value=flow_evidence_status(account)
    readiness=growth_evaluate(signal,account) if signal is not None else None
    steps=flow_step_status(account,signal,selected_opp)
    st.markdown('### Opportunity journey')
    st.markdown(render_lifecycle_stepper(steps), unsafe_allow_html=True)
    completed = sum(stage == 'complete' for stage in steps)
    st.progress(completed/5, text=f'{completed} of 5 checkpoints completed')
    next_step, next_detail = lifecycle_next_action(account, signal, selected_opp, value, readiness)
    st.markdown(
        '<div class="v17-next-action">'
        '<div><span class="v17-next-label">YOUR NEXT BEST STEP</span>'
        f'<strong>{html.escape(next_step)}</strong><p>{html.escape(next_detail)}</p></div>'
        '<div class="v17-next-symbol">↗</div></div>',
        unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Goal achieved', '—' if value['goal'] is None else f'{value["goal"]:.0f}%')
    k2.metric('Active-seat coverage','—' if value['adoption'] is None else f'{value["adoption"]:.0f}%')
    k3.metric('Growth gate',readiness['readiness'] if readiness else 'No signal')
    k4.metric('Monthly contracted expansion', f'€{growth_metrics(customer_opps)["verified_incremental_mrr_eur"]:,.0f}')
    st.caption('Goal achievement reflects progress against the agreed target. Readiness is a rules-based eligibility check, not a probability of purchase. Contracted revenue reflects recorded amounts and requires source verification.')
    with st.expander('Understand the journey metrics & decision gates', expanded=False):
        st.markdown('''**Goal achieved** compares the measured outcome with the agreed baseline and target; it is not a measure of customer satisfaction.

**Active-seat coverage** is active users divided by provisioned seats, when available. It is a usage proxy, not proof of customer value.

**Growth readiness** is a transparent qualification rule: sufficient documented value, no outstanding blockers, and supported evidence of the need and solution fit. It is not a purchase probability.

**Contracted expansion** sums recorded *incremental monthly* MRR for opportunities marked contracted after CSM approval and AE acceptance, with a source reference. Entries are not verified against billing.

**Journey status** represents documented workflow checkpoints. An approved decision records an attestation in this workspace, not an authenticated corporate approval.''')
    recommendation = v171_recommended_flow_stage(account, signal, selected_opp, value, readiness)
    if st.session_state.get('v171_flow_account') != aid:
        st.session_state['v171_flow_account'] = aid
        st.session_state['v171_flow_stage'] = recommendation
    selected_stage = int(st.session_state.get('v171_flow_stage', recommendation))
    selected_stage = max(0, min(selected_stage, 5))
    st.markdown('<div class="v171-stage-label">JOURNEY NAVIGATION · CHOOSE A STAGE</div>', unsafe_allow_html=True)
    stage_labels = ['01 · Customer value', '02 · Growth signal', '03 · CSM decision',
                    '04 · Sales handoff', '05 · Expansion outcome', '▤ · Activity & evidence']
    stage_columns = st.columns(5)
    for idx, col in enumerate(stage_columns):
        with col:
            badge = ('✓' if idx < 5 and steps[idx] == 'complete'
                     else '●' if idx < 5 and steps[idx] == 'in progress'
                     else '!' if idx < 5 and steps[idx] == 'attention' else '·')
            if st.button(badge + '  ' + stage_labels[idx], key=f'v171_flow_select_{idx}',
                         type='primary' if selected_stage == idx else 'secondary',
                         use_container_width=True,
                         help='View evidence and actions for this stage. Workflow approvals and gates still apply.'):
                st.session_state['v171_flow_stage'] = idx
                st.rerun()
    if st.button('▤  Activity, evidence & snapshots', key='v171_flow_activity', use_container_width=False):
        st.session_state['v171_flow_stage'] = 5
        st.rerun()
    st.markdown('<div class="v171-active-stage"><span>IN FOCUS</span><strong>'
                +html.escape(stage_labels[selected_stage])+'</strong><p>'
                +html.escape(LIFECYCLE_STAGE_EXPLAINERS[selected_stage][1] if selected_stage < 5
                    else 'Inspect the decision log and export or restore source-referenced records.')
                +'</p></div>', unsafe_allow_html=True)
    if selected_stage < 4 and steps[selected_stage] == 'complete':
        if st.button('Continue → ' + stage_labels[selected_stage + 1],
                     key='v171_flow_continue', type='primary'):
            st.session_state['v171_flow_stage'] = selected_stage + 1
            st.rerun()
    if selected_stage == 0:
        st.markdown('#### Customer objectives & observed value')
        st.write('**Objective:**',str(account.get('goal_name','') or 'Not entered'))
        st.write('**Measure:**',str(account.get('goal_metric','') or 'Not entered'))
        st.write('**Baseline → Current → Target:**', f'{account.get("goal_baseline","—")} → {account.get("goal_current","—")} → {account.get("goal_target","—")} {account.get("goal_unit","")}')
        st.write('**Evidence:**',str(account.get('source_system','') or '—'), ' / ', str(account.get('source_reference','') or '—'))
        st.write('**Measured on:**',str(account.get('measurement_as_of','') or '—'))
        st.info(value['message']) if value['documented'] else st.warning(value['message'])
        if not value['commercial_ready']:
            st.warning('Customer value comes first. Record outcomes and resolve blockers before initiating expansion discovery. **No commercial approval** is possible while the readiness gate is blocked.')
        if st.button('Open customer value workspace',key='flow_open_value'):
            st.session_state['active_view']='Real-data Pilot (MVP 2.0)'
            st.rerun()
    if selected_stage == 1:
        st.markdown('#### Unmet needs & solution fit')
        if signal is None:
            st.warning('No growth signal loaded for this account. Use the existing signal CSV importer.')
        else:
            st.write('**Customer need:**',signal.get('customer_need') or 'Not documented')
            st.write('**Candidate solution:**',signal.get('candidate_toolkit') or 'Not documented')
            st.write('**Confirmed need:**',signal.get('customer_confirmed_need') or 'Unknown')
            st.write('**Solution fit:**',signal.get('product_fit_confirmed') or 'Unknown')
            st.write('**Source / document:**',signal.get('need_source_system') or '—',' / ',signal.get('need_evidence_reference') or '—')
            st.write('**Evidence date:**',signal.get('need_evidence_date') or '—')
            st.info(f'**{readiness["readiness"]}** — {readiness["gate"]}')
            if readiness['signals']:
                st.write('**Signals:**','; '.join(readiness['signals']))
            if not readiness['ready']:
                st.warning('Growth discovery and commercial handoff are blocked. Address the documented reason before proposing.')
        if st.button('Open growth evidence importer',key='flow_open_signal'):
            st.session_state['active_view']='Growth & Expansion (MVP 2.1)'
            st.rerun()
    if selected_stage == 2:
        st.markdown('#### CSM qualification & decision')
        if readiness is None or not readiness['ready']:
            st.info('No proposal can be created unless the value and growth checks pass.')
        elif selected_opp is None:
            st.success('Evidence supports a CSM review. This does NOT authorize a sale.')
            with st.form('flow_propose_form'):
                reason=st.text_area('Explain customer benefit and discovery question',placeholder='The customer requested... An additional capability could help achieve...')
                create=st.form_submit_button('Create proposal for CSM review')
            if create:
                if not reason.strip(): st.error('Describe the concrete customer benefit first.')
                else:
                    entry=dict.fromkeys(GROWTH_OPP_COLUMNS+FLOW_EXTRA_COLUMNS,'')
                    entry.update(opportunity_id='GR-'+uuid.uuid4().hex[:10].upper(),account_id=aid,
                                 account_name=account['account_name'],csm_name=account['csm_name'],
                                 created_at=pilot_now(),candidate_toolkit=signal['candidate_toolkit'],
                                 customer_need=signal['customer_need'],need_evidence_reference=signal['need_evidence_reference'],
                                 readiness_at_proposal=readiness['readiness'],review_status='Pending CSM',
                                 review_note=reason.strip(),proposal_customer_benefit=reason.strip(),
                                 handoff_status='Not sent',stage='Discovery')
                    st.session_state['mvp21_opps']=pd.concat([opps,pd.DataFrame([entry])],ignore_index=True)
                    flow_event(aid,entry['opportunity_id'],'propose',account['csm_name'],signal['need_evidence_reference'],reason.strip())
                    st.success('Proposal created. Approval is still required.')
                    st.rerun()
        elif selected_opp['review_status']=='Pending CSM':
            st.info('Proposal submitted; awaiting the assigned CSM decision.')
            st.write('**Customer benefit:**',selected_opp.get('proposal_customer_benefit') or selected_opp.get('review_note'))
            with st.form('flow_csm_form'):
                decision=st.selectbox('CSM decision',['Approve for discovery','On hold','Reject'])
                actor=st.text_input('Assigned CSM name',placeholder=str(account['csm_name']))
                comment=st.text_area('Evidence and reason for this decision')
                send=st.form_submit_button('Record CSM decision')
            if send:
                mapping={'Approve for discovery':'approve','On hold':'hold','Reject':'reject'}
                try:
                    flow_record_transition(selected_opp['opportunity_id'],mapping[decision],account,signal,
                                           actor=actor,notes=comment)
                    st.rerun()
                except ValueError as e: st.error(str(e))
        else:
            st.write('**CSM decision:**',selected_opp.get('review_status'))
            st.write('**Recorded by:**',selected_opp.get('reviewed_by') or '—')
            st.write('**Reason:**',selected_opp.get('review_note') or '—')
            if selected_opp['review_status'] != 'Approved for discovery':
                st.warning('Opportunity is on hold or rejected. Create a new supported proposal after circumstances change; do not bypass this decision.')
    if selected_stage == 3:
        st.markdown('#### Sales ownership & decision')
        if selected_opp is None or selected_opp['review_status']!='Approved for discovery':
            st.info('Waiting for the assigned CSM to approve the opportunity for discovery.')
        elif selected_opp['handoff_status']=='Not sent':
            with st.form('flow_handoff_form'):
                csm=st.text_input('CSM sender (must match owner)')
                ae=st.text_input('Assigned Account Executive',value=str(signal.get('ae_owner') or '') if signal is not None else '')
                crm=st.text_input('CRM record / handoff reference')
                note=st.text_area('Discovery context, customer benefit and next action')
                sent=st.form_submit_button('Send to Sales — record handoff')
            if sent:
                try:
                    flow_record_transition(selected_opp['opportunity_id'],'send_to_ae',account,signal,
                                           actor=csm,ae_owner=ae,reference=crm,notes=note)
                    st.rerun()
                except ValueError as e: st.error(str(e))
        elif selected_opp['handoff_status']=='Sent to AE':
            st.info('Handoff sent. Waiting for Sales acceptance.')
            st.write('**Handoff reference:**',selected_opp.get('handoff_evidence_reference') or '—')
            with st.form('flow_ae_form'):
                response=st.radio('AE review',['Accept','Decline'],horizontal=True)
                actor=st.text_input('AE name',placeholder=str(selected_opp.get('ae_owner','')))
                reference=st.text_input('CRM decision reference',value=str(selected_opp.get('handoff_evidence_reference') or ''))
                note=st.text_area('AE decision rationale')
                apply=st.form_submit_button('Record Sales decision')
            if apply:
                try:
                    flow_record_transition(selected_opp['opportunity_id'],'accept_ae' if response=='Accept' else 'decline_ae',
                                           account,signal,actor=actor,reference=reference,notes=note)
                    st.rerun()
                except ValueError as e: st.error(str(e))
        else:
            st.write('**Sales decision:**',selected_opp.get('handoff_status'))
            st.write('**AE owner:**',selected_opp.get('ae_owner') or '—')
            st.write('**Decision note:**',selected_opp.get('ae_decision_note') or '—')
            if selected_opp['handoff_status']=='Declined by AE':
                st.warning('Sales declined. No contracted revenue can be recorded.')
    if selected_stage == 4:
        st.markdown('#### Commercial outcome + post-expansion customer value')
        if selected_opp is None or selected_opp['handoff_status']!='Accepted by AE':
            st.info('The commercial outcome is locked until Sales explicitly accepts the CSM-approved handoff.')
        else:
            stage=str(selected_opp['stage'])
            st.write('**Commercial pipeline stage:**',stage)
            if stage not in ['Contracted','Lost']:
                with st.expander('Optional · Update discovery / proposal stage'):
                    with st.form('flow_stage_form'):
                        next_stage=st.selectbox('Pipeline stage',FLOW_STAGE_SEQUENCE,
                                                index=FLOW_STAGE_SEQUENCE.index(stage) if stage in FLOW_STAGE_SEQUENCE else 0)
                        ae=st.text_input('AE updating stage')
                        reference=st.text_input('CRM opportunity reference')
                        reason=st.text_area('What changed?')
                        update=st.form_submit_button('Save commercial stage')
                    if update:
                        try:
                            flow_record_transition(selected_opp['opportunity_id'],'stage:'+next_stage,
                                                   account,signal,actor=ae,reference=reference,notes=reason)
                            st.rerun()
                        except ValueError as e: st.error(str(e))
                with st.form('flow_close_form'):
                    final=st.radio('Final commercial outcome',['Contracted','Lost'],horizontal=True)
                    ae=st.text_input('AE confirming outcome')
                    crm=st.text_input('Contract/billing reference (Contracted) or CRM loss reference (Lost)')
                    mrr=st.number_input('Actual incremental *monthly* MRR (€)',min_value=0.0,value=0.0,step=50.0,
                                        disabled=False, help='Only actual contracted increments count; not forecast or full account MRR.')
                    note=st.text_area('Evidence and deal outcome notes')
                    closed=st.form_submit_button('Record commercial outcome')
                if closed:
                    try:
                        flow_record_transition(selected_opp['opportunity_id'],final.lower(),account,signal,
                                               actor=ae,reference=crm,notes=note,amount=mrr if final=='Contracted' else None)
                        st.rerun()
                    except ValueError as e: st.error(str(e))
            elif stage=='Lost':
                st.warning('Closed Lost. No incremental MRR credited.')
                st.write('**Loss evidence:**',selected_opp.get('outcome_note') or '—')
            else:
                st.success('Contracted expansion recorded with documentary reference (self-attested).')
                st.metric('Incremental contracted MRR',f'€{float(pd.to_numeric(selected_opp["contracted_incremental_mrr_eur"],errors="coerce")):,.0f} / month')
                st.write('**Contract/billing reference:**',selected_opp.get('contract_evidence_reference') or '—')
                delta=flow_post_value_delta(selected_opp,account)
                if delta is not None:
                    st.metric('Observed improvement in primary outcome',f'{delta:+,.2f} {account.get("goal_unit", "units")}')
                    st.caption('Measured before/after difference, NOT proof that the expansion or CSM intervention caused the improvement.')
                    st.write('**Post-measurement source:**',selected_opp.get('post_value_evidence_reference'))
                else:
                    st.info('Customer-value follow-up pending. A signed contract does not prove the customer achieved better results.')
                    with st.form('flow_post_value_form'):
                        post=st.number_input('Measured primary goal value AFTER expansion',value=0.0,format='%.2f')
                        dt=st.date_input('Post-expansion observation date',value=date.today())
                        validator=st.text_input('Owner validating measurement')
                        source=st.text_input('Telemetry / measurement source reference')
                        note=st.text_area('What did we observe? Any other causes?')
                        record=st.form_submit_button('Document observed customer outcome')
                    if record:
                        try:
                            flow_record_transition(selected_opp['opportunity_id'],'post_value',account,signal,
                                                   actor=validator,reference=source,notes=note,
                                                   goal_value=post,observed_at=dt.isoformat())
                            st.rerun()
                        except ValueError as e: st.error(str(e))
    if selected_stage == 5:
        st.markdown('#### Activity log · self-attested, session-only')
        events=st.session_state['mvp211_events']
        current=events.loc[events['account_id'].astype(str)==aid] if not events.empty else events
        if current.empty: st.info('No workflow actions logged yet for this account.')
        else: st.dataframe(current.sort_values('timestamp_utc',ascending=False),hide_index=True,use_container_width=True)
        st.caption('Activity is recorded in this session. Entries are self-attested, with no independent identity verification or tamper-resistant storage.')
        st.download_button('Export workspace snapshot (.zip)',flow_export_zip(),
                           file_name='mvp211_internal_pilot_snapshot.zip',mime='application/zip',key='flow_snapshot')
        st.caption('The export includes account data, evidence, commercial decisions, activity and impact actions. Store it securely; no server-side database persistence is configured.')
        with st.expander('Restore a previously exported workspace snapshot'):
            up=st.file_uploader('Select a secure workspace snapshot ZIP',type=['zip'],key='flow_restore_zip')
            confirm=st.checkbox('I understand this will REPLACE the current in-session accounts, signals, opportunities and activity log.',key='flow_confirm_restore')
            if st.button('Validate and restore snapshot',disabled=up is None or not confirm,key='flow_restore_btn'):
                try:
                    restored=flow_restore_snapshot(up.getvalue())
                    st.session_state.update(mvp20_accounts=restored[0],mvp21_signals=restored[1],
                                            mvp21_opps=restored[2],mvp211_events=restored[3],
                                            mvp20_actions=restored[4])
                    st.success('Snapshot loaded into session. All files passed structural checks. This is NOT cryptographic or identity verification.')
                    st.rerun()
                except Exception as exc:
                    st.error(f'Restore rejected: {exc}')

        st.markdown('**Decision controls:** current source dates, named owners and record references are required; Sales handoff follows CSM approval, and estimated MRR is excluded from contracted totals. Authentication and internal-system verification are separate integration requirements.')


if active_view == "Real-data Pilot (MVP 2.0)":
    pilot_render()

elif active_view == "Growth & Expansion (MVP 2.1)":
    growth_render()

elif active_view == "End-to-End Flow (MVP 2.1.1)":
    flow_render()

elif active_view == "Mission Control":
    st.markdown(
        '<div class="v171-mission-heading"><div><span>DAILY DECISION DESK</span>'
        '<h3>Act on what matters. Then measure the outcome.</h3>'
        '<p>Start with urgent retention exposure, then move through customer value and qualified growth.</p></div>'
        '<div class="v171-mission-status"><span class="v171-status-dot"></span> SYNTHETIC PORTFOLIO</div></div>',
        unsafe_allow_html=True,
    )
    d1,d2,d3 = st.columns(3)
    with d1:
        st.markdown('<div class="v171-decision-kpi"><span>HIGH-RISK ACCOUNTS</span><strong>'+str(high_count)+'</strong><small>Prioritize the highest-risk customers</small></div>',unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="v171-decision-kpi"><span>MRR EXPOSURE</span><strong>'+html.escape(_fmt_money(mrr_at_risk))+'</strong><small>High-risk revenue, not forecast loss</small></div>',unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="v171-decision-kpi"><span>RENEWAL PRESSURE</span><strong>'+str(renewal_pressure)+'</strong><small>Renewals requiring review within 60 days</small></div>',unsafe_allow_html=True)
    mission_shortcuts = st.columns([1,1,1,1])
    with mission_shortcuts[0]:
        if st.button("◉  Review P1 / P2", key="v17_mission_p1", use_container_width=True, type="primary"):
            go_to("Priority Queue", "P1/P2 accounts")
    with mission_shortcuts[1]:
        if st.button("↗  Growth opportunities", key="v17_mission_growth", use_container_width=True):
            go_to("Growth & Expansion (MVP 2.1)")
    with mission_shortcuts[2]:
        if st.button("⇄  Revenue lifecycle", key="v17_mission_flow", use_container_width=True):
            go_to("End-to-End Flow (MVP 2.1.1)")
    with mission_shortcuts[3]:
        if st.button("▦  My CSM book", key="v17_mission_book", use_container_width=True):
            go_to("My CSM Book")
    with st.expander("How Mission Control prioritizes the portfolio", expanded=False):
        st.markdown(daily_brief_html(), unsafe_allow_html=True)
        st.markdown(mission_guide_html(), unsafe_allow_html=True)
    render_immediate_action_accounts()
    st.markdown(portfolio_360_html(), unsafe_allow_html=True)
    render_portfolio_360_actions()
    with st.expander("Explore all portfolio risk and opportunity signals", expanded=False):
        render_clickable_mission_cards()
    section_header('Customer-led growth', 'Explore documented customer needs, solution fit and the decisions required before a Sales handoff.')
    st.caption('Growth assessments use a separate evidence-led workspace. The portfolio view uses synthetic data and is not a source of real account forecasts.')
    if st.button('Open Growth Intelligence →', key='mission_to_growth'):
        st.session_state['active_view'] = 'Growth & Expansion (MVP 2.1)'
        st.rerun()

    st.markdown(
        '<div class="briefing-note"><strong>How to use the cards:</strong> click any snapshot CTA to open the Priority Queue with the right filter already applied. The goal is to move from portfolio signal → focused queue → account briefing → handoff or recovery action.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Understand the portfolio indicators", expanded=False):
        st.markdown("""**Average risk:** mean model-estimated churn probability across accounts in the selected synthetic portfolio. Individual predictions can differ significantly.

**High-risk accounts:** accounts with model-estimated churn probability of 70% or higher. The thresholds are configurable business rules, not validated company policy.

**MRR at risk:** recurring revenue associated with accounts classified as high risk; it is exposure, **not** forecasted revenue loss.

**Renewal pressure:** approaching renewal with weak or non-positive renewal intent.

**Adoption gap:** accounts where activity or feature-adoption scores are low.

**Product friction:** unresolved bugs, data-quality issues, feedback volume or product-fit concerns.

**Team dependencies:** potential handoffs to Support, Sales or Product based on account signals. The same account may need more than one team.

All values on Mission Control are calculated from **synthetic account data**, not real Semrush accounts.""")
    with st.expander("View details behind Portfolio Risk", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            render_static_bar_chart(filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int), "Accounts by risk level", "Accounts", False)
        with c2:
            by_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False)
            render_static_bar_chart(by_segment, "Average churn probability by segment", "Churn probability", True)

    with st.expander("View details behind Revenue & Renewal Exposure", expanded=False):
        renewal_cols = ["priority", "account_id", "account_name", "segment", "plan_family", "mrr", "churn_probability", "renewal_due_days", "renewal_intent", "pricing_objection_flag", "downgrade_risk_flag"]
        renewal_view = filtered[(filtered["renewal_due_days"] < 60) | (filtered["risk_level"] == "High")][renewal_cols].copy().head(50)
        renewal_view["mrr"] = renewal_view["mrr"].map(lambda x: f"${x:,.0f}")
        renewal_view["churn_probability"] = renewal_view["churn_probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(renewal_view, use_container_width=True, hide_index=True)

    with st.expander("View details behind Adoption Gap", expanded=False):
        adoption_view = filtered[(filtered["tool_activity_score"] < 45) | (filtered["feature_adoption_score"] < 45)][["account_id", "account_name", "plan_family", "tool_activity_score", "feature_adoption_score", "site_audit_last_run_days", "position_tracking_campaigns_active", "ai_visibility_prompts_tracked", "churn_probability"]].head(50).copy()
        adoption_view["churn_probability"] = adoption_view["churn_probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(adoption_view, use_container_width=True, hide_index=True)

    with st.expander("View details behind Product Friction & Trust Risk", expanded=False):
        friction_view = filtered[(filtered["stuck_bugs_count"] > 0) | (filtered["data_accuracy_bug_flag"] == 1) | (filtered["unresolved_feedback_count"] >= 3) | (filtered["roadmap_alignment_score"] < 40)][["account_id", "account_name", "plan_family", "stuck_bugs_count", "data_accuracy_bug_flag", "unresolved_feedback_count", "roadmap_alignment_score", "workaround_available_flag", "churn_probability"]].head(50).copy()
        friction_view["churn_probability"] = friction_view["churn_probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(friction_view, use_container_width=True, hide_index=True)

    with st.expander("View cross-functional dependency details", expanded=False):
        st.markdown(routing_summary_html(filtered), unsafe_allow_html=True)


elif active_view == "Priority Queue":
    section_header("CSM Priority Queue", "A guided work queue. Use the focus selector or Mission Control CTAs to choose the right account set.")
    st.markdown(
        f'<div class="daily-brief"><div class="daily-kicker">Current queue focus</div><div class="daily-title">{html.escape(queue_title)}</div><div class="daily-body">{html.escape(queue_description)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(immersive_priority_landscape_html(queue_filtered), unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        metric_card("Accounts in this queue", len(queue_filtered), "After sidebar filters and queue focus")
    with q2:
        metric_card("Average churn probability", _fmt_pct(queue_filtered["churn_probability"].mean()) if len(queue_filtered) else "N/A", "Mean predicted risk in this queue")
    with q3:
        metric_card("MRR in queue", _fmt_money(queue_filtered["mrr"].sum()) if len(queue_filtered) else "$0", "Commercial exposure represented here")
    with q4:
        metric_card("High-risk accounts", int((queue_filtered["risk_level"] == "High").sum()) if len(queue_filtered) else 0, "Accounts already over the high-risk threshold")

    st.markdown('<div class="briefing-note"><strong>How to use this queue:</strong> start with P1/P2 cards. If this queue came from a Mission Control CTA, it is already filtered for the relevant risk theme. Open Account Briefing for the selected account when you need evidence, scenario simulation or handoff details.</div>', unsafe_allow_html=True)

    if queue_filtered.empty:
        st.info("No accounts match this queue focus with the current sidebar filters.")
    else:
        for priority, title, expanded in [("P1", "P1 — Act first", True), ("P2", "P2 — Review this week", True), ("P3", "P3 — Monitor closely", False), ("Monitor", "Monitor — Regular cadence", False)]:
            subset = queue_filtered[queue_filtered["priority"] == priority]
            if not subset.empty:
                with st.expander(title, expanded=expanded):
                    render_priority_cards(subset, max_cards=9)

        with st.expander("Full operational account table for this queue", expanded=False):
            cols = ["priority", "account_id", "account_name", "segment", "region", "plan_family", "plan_tier", "mrr", "churn_probability", "risk_level", "renewal_due_days", "renewal_intent", "tool_activity_score", "feature_adoption_score", "crawl_budget_usage_pct", "keyword_tracking_usage_pct", "roadmap_alignment_score", "stuck_bugs_count", "data_accuracy_bug_flag"]
            table = queue_filtered[cols].copy()
            table["mrr"] = table["mrr"].map(lambda x: f"${x:,.0f}")
            table["churn_probability"] = table["churn_probability"].map(lambda x: f"{x:.1%}")
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button("Download this queue as CSV", table.to_csv(index=False), f"semrush_copilot_{active_queue_filter.lower().replace(' ', '_').replace('/', '_')}.csv", "text/csv")



elif active_view == "My CSM Book":
    section_header("My CSM Book", "Review one CSM portfolio as a daily command center.")
    if "csm_name" not in df.columns:
        st.warning("This dataset does not include CSM assignment fields.")
    else:
        # If a specific CSM is selected in the sidebar, use the filtered book. Otherwise show all selected CSM filters.
        render_csm_book_overview(filtered, df)

elif active_view == "Portfolio Insights":
    section_header("Portfolio Insights", "A guided view of where risk is building before looking at account-level evidence.")
    c1, c2 = st.columns(2)
    with c1:
        render_static_bar_chart(filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int), "Accounts by risk level", "Accounts", False)
    with c2:
        by_segment = filtered.groupby("segment")["churn_probability"].mean().sort_values(ascending=False)
        render_static_bar_chart(by_segment, "Average churn probability by segment", "Churn probability", True)

    section_header("Where is risk building?", "Risk family score explained as a CSM story, not a raw table.")
    avg_scores = avg_family_scores.reset_index()
    avg_scores.columns = ["Risk family", "Average score"]
    render_driver_story_section(avg_scores)

    section_header("Which package families need attention?", "A CSM-friendly view of risk concentration by Semrush-like package family.")
    by_family = filtered.groupby("plan_family")["churn_probability"].mean().sort_values(ascending=False)
    render_commercial_exposure_story_section(by_family)

    section_header("Cross-functional action routing", "Which teams are needed to unblock retention risk in the current portfolio view.")
    st.markdown(routing_summary_html(filtered), unsafe_allow_html=True)


elif active_view == "Account Briefing":
    section_header("Account 360° Briefing", "Search any customer, understand the full account story, then move from risk evidence to action.")
    briefing_source = queue_filtered if active_queue_filter != "All accounts" and not queue_filtered.empty else filtered
    if briefing_source.empty:
        st.info("No accounts available for the current filters.")
    else:
        default_selected = st.session_state.get("selected_account_id", None)
        # If the global finder selected an account outside the current sidebar filters, temporarily use the full dataset.
        if default_selected is not None and default_selected in df["account_id"].values and default_selected not in briefing_source["account_id"].values:
            briefing_source = df.copy().sort_values("churn_probability", ascending=False).reset_index(drop=True)
            st.info("Showing an account opened from global search. Some sidebar filters may not apply to this account.")

        account_options = briefing_source["account_id"].tolist()
        default_selected = st.session_state.get("selected_account_id", account_options[0])
        default_index = account_options.index(default_selected) if default_selected in account_options else 0
        if st.session_state.get("selected_account_id") != st.session_state.get("briefing_account"):
            st.session_state["briefing_account"] = default_selected
        def v171_account_selection_changed():
            st.session_state["selected_account_id"] = st.session_state["briefing_account"]
        selected = st.selectbox(
            "Account in focus",
            account_options,
            index=default_index,
            format_func=lambda x: account_format_label(briefing_source, x),
            key="briefing_account",
            on_change=v171_account_selection_changed,
        )
        st.session_state["selected_account_id"] = selected
        row = briefing_source.loc[briefing_source["account_id"] == selected].iloc[0]
        st.caption(f"Briefing source: {active_queue_filter}. Change Priority Queue focus in the sidebar if you want a different account set.")
        b1, b2, b3 = st.columns([1, 1, 3])
        with b1:
            if st.button("Open simulator for this account", key="briefing_to_sim", use_container_width=True):
                st.session_state["selected_account_id"] = selected
                go_to("Scenario Simulator", active_queue_filter)
        with b2:
            if st.button("Back to queue", key="briefing_back_queue", use_container_width=True):
                go_to("Priority Queue", active_queue_filter)
        render_selected_account_briefing(row, include_scenario=False)


elif active_view == "Scenario Simulator":
    section_header("Scenario Simulator", "Compare which action could reduce churn probability fastest before deciding the recovery plan.")
    sim_source = queue_filtered if active_queue_filter != "All accounts" and not queue_filtered.empty else filtered
    if sim_source.empty:
        st.info("No accounts available for simulation with the current filters.")
    else:
        sim_options = sim_source["account_id"].tolist()
        default_sim = st.session_state.get("selected_account_id", sim_options[0])
        default_sim_index = sim_options.index(default_sim) if default_sim in sim_options else 0
        selected_sim = st.selectbox(
            "Search / select account for simulation",
            sim_options,
            index=default_sim_index,
            format_func=lambda x: f"{x} · {sim_source.loc[sim_source['account_id']==x, 'account_name'].iloc[0]}",
            key="simulator_account",
        )
        st.session_state["selected_account_id"] = selected_sim
        sim_row = sim_source.loc[sim_source["account_id"] == selected_sim].iloc[0]
        st.caption(f"Simulation source: {active_queue_filter}. Change Priority Queue focus in the sidebar if you want to simulate a different queue.")
        st.markdown(account_360_core_html(sim_row), unsafe_allow_html=True)
        render_scenario_story(sim_row, model, model_columns)
        with st.expander("View scenario calculation table", expanded=False):
            scenarios = simulate_scenarios(sim_row, model, model_columns)
            st.dataframe(scenarios, use_container_width=True, hide_index=True)
            st.caption("Scenario results are model-based simulations on synthetic data. They are decision-support estimates, not guaranteed outcomes.")


elif active_view == "CSM Playbook":
    section_header("CSM Playbook", "Rules that translate model signals into direct customer-success actions.")
    st.markdown('<div class="briefing-note"><strong>How to use this:</strong> the playbook is not another dashboard. It explains why the Copilot recommends involving CSM, Support, Sales or Product for specific account patterns.</div>', unsafe_allow_html=True)

    section_header("Cross-functional routing rules", "When the CSM should involve Customer Support, Account Executive / Sales, or Product / Engineering.")
    routing_rules = pd.DataFrame([
        ["Customer Support", "Tool discrepancy or bug", "data_accuracy_bug_flag, site_audit_bug_flag, position_tracking_bug_flag, ai_visibility_bug_flag, local_listing_sync_bug_flag", "Validate issue, confirm expected behaviour vs defect, document workaround and customer-facing explanation."],
        ["Customer Support", "Limit misunderstanding", "crawl_budget_usage_pct > 85, keyword_tracking_usage_pct > 85, ai_visibility_prompt_usage_pct > 85, tool_limit_feedback_count >= 1", "Clarify account limits, user/team allocation, current usage and navigation path to verify limits."],
        ["Customer Support", "Navigation / troubleshooting help", "technical_tickets_last_quarter >= 4 or repeated tool-specific issues", "Provide precise navigation explanation for Site Audit, Position Tracking, backlinks, prompt tracking or listing setup."],
        ["Customer Support", "Hard block", "stuck_bugs_count >= 1 and workaround_available_flag = 0", "Treat as blocked workflow, confirm severity, document impact and define update cadence."],
        ["Account Executive / Sales", "Pricing or package fit", "pricing_objection_flag, downgrade_risk_flag, limit_upgrade_recommended_flag", "Review commercial options, upgrade/downgrade fit, package scope and value narrative."],
        ["Account Executive / Sales", "Contract questions", "renewal_due_days < 45 and weak renewal intent, billing tickets, purchased add-ons uncertainty", "Confirm contract terms, purchased packages, renewal options and commercial commitments before customer follow-up."],
        ["Product / Engineering", "Roadmap or product gap", "critical_feedback_count_90d >= 1 and roadmap_alignment_score < 45", "Confirm roadmap status, workaround or alternative workflow before renewal conversation."],
        ["Product / Engineering", "Stuck data-trust issue", "data_accuracy_bug_flag = 1 and stuck_bugs_count >= 1", "Confirm investigation status, customer impact, ETA and next update path."],
    ], columns=["Team", "Trigger", "Signals", "CSM handoff ask"])
    st.dataframe(routing_rules, use_container_width=True, hide_index=True)

    section_header("Semrush-specific CSM action rules", "Rules that translate model signals into direct customer-success actions.")
    rules = pd.DataFrame([
        ["Crawl budget saturation", "crawl_budget_usage_pct > 85", "Run a Site Audit limit-fit review: crawl scope, audit frequency, excluded pages and plan limit fit."],
        ["Keyword tracking limit pressure", "keyword_tracking_usage_pct > 85", "Review Position Tracking keyword allocation, inactive campaigns, devices/locations and upgrade fit."],
        ["AI Visibility under-adoption", "ai_visibility_active = 1 and prompts tracked < 10", "Book an AI Visibility workflow session: define prompts, competitors and recurring AI search review."],
        ["Product gap linked to renewal", "feedback_linked_to_renewal = 1 and roadmap_alignment_score < 45", "Treat as renewal dependency. Align Product roadmap status and workaround before renewal conversation."],
        ["Data trust bug", "data_accuracy_bug_flag = 1 and stuck_bugs_count >= 1", "Escalate as a trust-risk issue. Confirm impact, ETA, workaround and update cadence."],
        ["Advanced plan under-adoption", "Business / Pro+ / Advanced and activity < 45", "Run adoption recovery focused on paid-but-underused workflows and executive reporting."],
        ["Active but frustrated", "High activity and negative sentiment", "Do not treat as disengagement; run value-and-friction review to remove blockers."],
    ], columns=["Risk pattern", "Condition", "CSM instruction"])
    st.dataframe(rules, use_container_width=True, hide_index=True)


elif active_view == "Methodology & Data Mapping":
    section_header("Methodology & Data Mapping", "Keep the details available, but separate from the daily CSM workflow.")
    st.markdown(
        '<div class="method-card"><strong>What is predicted?</strong><br>For each synthetic account, the model estimates <code>P(churn = 1 | account signals)</code>. The portfolio KPI is the average of those account-level probabilities.</div>'
        '<div class="method-card"><strong>What goes into the model?</strong><br>Account profile, plan family, toolkit adoption, usage limits, Site Audit and Position Tracking usage, AI Visibility adoption, Local Toolkit activity, support friction, feedback, bugs, sentiment and renewal context.</div>'
        '<div class="method-card"><strong>What is not used live?</strong><br><code>account_id</code>, <code>account_name</code>, <code>churn_label</code>, and <code>synthetic_true_churn_probability</code> are excluded from predictive features. The churn label is used only during training.</div>'
        '<div class="method-card"><strong>Cross-functional routing</strong><br>The routing layer is rule-based. It does not change the churn probability; it translates visible risk patterns into CSM handoff guidance for Support, Account Executive / Sales, or Product / Engineering.</div>'
        '<div class="method-card"><strong>Internal navigation</strong><br>Mission Control cards route the CSM into the correct Priority Queue filter. This keeps the first view simple while preserving all detailed account evidence in drill-down sections. CSM assignment fields are used for filtering and workflow ownership, not as predictive model inputs.</div>'
        '<div class="method-card"><strong>Important disclaimer</strong><br>This is synthetic demo data inspired by public Semrush product information. It is not internal Semrush customer data and should be retrained on authorized historical company data before production use.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Suggested company data mapping", expanded=False):
        mapping = pd.DataFrame([
            ["login_count_30d / tool_activity_score", "Product analytics", "User activity, sessions, workflow events"],
            ["crawl_budget_usage_pct", "Product usage / Site Audit", "Pages crawled, monthly limits, audit settings"],
            ["keyword_tracking_usage_pct", "Position Tracking / subscription limits", "Tracked keywords, locations, devices, keyword allowance"],
            ["ai_visibility_prompt_usage_pct", "AI Visibility Toolkit", "Prompts tracked, prompt limit, prompt research usage"],
            ["tickets_last_quarter / csat_support", "Customer Support platform", "Support volume, reopenings, resolution time, CSAT"],
            ["feedback_count_90d / roadmap_alignment_score", "Feedback / Product Ops", "Feature requests, unresolved feedback, roadmap status"],
            ["stuck_bugs_count / data_accuracy_bug_flag", "Bug tracker / Support escalation", "Open bugs, severity, data trust issues, workaround"],
            ["renewal_due_days / renewal_intent", "CRM / CS platform", "Renewal date, forecast, risk notes, opportunity context"],
        ], columns=["Copilot signal", "Likely source", "What to map"])
        st.dataframe(mapping, use_container_width=True, hide_index=True)
    st.write("Model package:", model_package.get("model_type", "Unknown"), "· Training columns:", len(model_columns or []))


# =========================
# V11 Harmonized UI polish
# Soft enterprise palette, balanced mission widgets, refined CTAs and sidebar design.
# This block intentionally comes last so it overrides earlier visual styles without changing logic.
# =========================
st.markdown(
    """
    <style>
    :root {
        --v11-navy: #0A1633;
        --v11-ink: #111827;
        --v11-muted: #667085;
        --v11-border: #D9E2EF;
        --v11-bg: #F3F7FC;
        --v11-card: #FFFFFF;
        --v11-blue: #4A78D4;
        --v11-blue-dark: #1E3A8A;
        --v11-teal: #2C8FA3;
        --v11-teal-dark: #0F766E;
        --v11-coral: #D95F66;
        --v11-amber: #C98223;
        --v11-purple: #7556C8;
    }

    /* Page canvas */
    .stApp {
        background:
            radial-gradient(circle at 88% 3%, rgba(92, 151, 191, 0.16), transparent 28%),
            linear-gradient(180deg, #F7FAFD 0%, #EEF5FC 100%) !important;
        color: var(--v11-navy) !important;
    }

    /* Sidebar: no more flat white. Make it feel like part of the product. */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 20% 0%, rgba(74, 120, 212, 0.16), transparent 30%),
            linear-gradient(180deg, #F7FBFF 0%, #EAF2FB 52%, #E2EEF9 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.38) !important;
        box-shadow: inset -1px 0 0 rgba(255,255,255,.65), 14px 0 35px rgba(15, 23, 42, 0.04) !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.7rem !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #1F2937 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        letter-spacing: -0.02em !important;
    }

    /* Sidebar controls */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background: rgba(255, 255, 255, 0.78) !important;
        border: 1px solid rgba(148, 163, 184, 0.42) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045), inset 0 1px 0 rgba(255,255,255,.75) !important;
        min-height: 46px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
    section[data-testid="stSidebar"] input:hover,
    section[data-testid="stSidebar"] textarea:hover {
        border-color: rgba(74, 120, 212, 0.55) !important;
        box-shadow: 0 10px 26px rgba(74, 120, 212, 0.10), inset 0 1px 0 rgba(255,255,255,.80) !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.35) !important;
    }

    /* Main hero is softer and more corporate */
    .hero {
        background:
            radial-gradient(circle at 100% 0%, rgba(95, 165, 182, .26), transparent 28%),
            linear-gradient(135deg, #08182F 0%, #21427A 54%, #4F94A5 100%) !important;
        box-shadow: 0 26px 64px rgba(10, 22, 51, .24) !important;
        border: 1px solid rgba(255,255,255,.18) !important;
    }

    /* Mission widgets: same anatomy, same vertical rhythm, softer colors. */
    .mission-grid {
        gap: 18px !important;
        align-items: stretch !important;
        margin-top: 18px !important;
    }

    .mission-card,
    .mission-card.immersive-card {
        min-height: 300px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        padding: 22px 22px 20px !important;
        border: 1px solid rgba(217, 226, 239, 0.95) !important;
        border-top: 6px solid var(--accent, #4A78D4) !important;
        border-radius: 26px !important;
        background:
            radial-gradient(circle at 100% 0%, color-mix(in srgb, var(--accent, #4A78D4), white 86%) 0%, rgba(255,255,255,0) 44%),
            linear-gradient(180deg, rgba(255,255,255,.98) 0%, rgba(248,251,255,.96) 100%) !important;
        box-shadow: 0 18px 40px rgba(15, 23, 42, .075), inset 0 1px 0 rgba(255,255,255,.88) !important;
        overflow: hidden !important;
    }

    .mission-card.immersive-card:hover,
    .mission-card:hover {
        transform: perspective(900px) rotateX(1.2deg) rotateY(-1.2deg) translateY(-5px) !important;
        box-shadow: 0 26px 58px rgba(15, 23, 42, .13), 0 0 0 1px color-mix(in srgb, var(--accent, #4A78D4), white 70%) !important;
    }

    .mission-icon-orb {
        width: 54px !important;
        height: 54px !important;
        border-radius: 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: linear-gradient(135deg, color-mix(in srgb, var(--accent, #4A78D4), #0A1633 42%), var(--accent, #4A78D4)) !important;
        color: #FFFFFF !important;
        font-size: 22px !important;
        box-shadow: 0 14px 28px color-mix(in srgb, var(--accent, #4A78D4), transparent 72%) !important;
        margin-bottom: 18px !important;
    }

    .mission-kicker {
        min-height: 18px !important;
        color: #6B778A !important;
        font-size: 11px !important;
        font-weight: 950 !important;
        letter-spacing: .09em !important;
        margin-bottom: 10px !important;
    }

    .mission-title {
        min-height: 46px !important;
        color: var(--v11-navy) !important;
        font-size: 18px !important;
        font-weight: 950 !important;
        line-height: 1.18 !important;
        margin-bottom: 12px !important;
    }

    .mission-value {
        min-height: 44px !important;
        color: #061126 !important;
        font-size: 35px !important;
        font-weight: 950 !important;
        letter-spacing: -0.035em !important;
        margin-bottom: 12px !important;
    }

    .mission-body {
        min-height: 70px !important;
        color: #4B5565 !important;
        font-size: 13.5px !important;
        line-height: 1.48 !important;
        margin-bottom: 16px !important;
    }

    .card-direction,
    .mission-cta {
        margin-top: auto !important;
        background: rgba(255,255,255,0.72) !important;
        border: 1px solid rgba(217, 226, 239, 0.95) !important;
        border-radius: 16px !important;
        padding: 12px 13px !important;
        color: #344054 !important;
        font-size: 12.5px !important;
        font-weight: 900 !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.85) !important;
    }

    /* Softer, more integrated CTA buttons under mission cards. */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #173A7A 0%, #4A78D4 55%, #5FA5B6 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,.20) !important;
        border-radius: 18px !important;
        min-height: 54px !important;
        padding: 0.78rem 1.05rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 14px 30px rgba(74, 120, 212, .20), inset 0 1px 0 rgba(255,255,255,.25) !important;
        transition: transform .14s ease, box-shadow .14s ease, filter .14s ease !important;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        filter: brightness(1.035) saturate(.98) !important;
        box-shadow: 0 18px 38px rgba(74, 120, 212, .26), inset 0 1px 0 rgba(255,255,255,.30) !important;
    }

    div[data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 10px 22px rgba(74, 120, 212, .20) !important;
    }

    div[data-testid="stButton"] > button p {
        color: #FFFFFF !important;
        font-weight: 900 !important;
    }

    /* Reduce the visual dominance of warning tones while keeping category identity. */
    .mission-card.immersive-card[style*="#DC2626"],
    .mission-card[style*="#DC2626"] {
        --accent: #D95F66 !important;
    }
    .mission-card.immersive-card[style*="#D97706"],
    .mission-card[style*="#D97706"] {
        --accent: #C98223 !important;
    }
    .mission-card.immersive-card[style*="#6D28D9"],
    .mission-card[style*="#6D28D9"] {
        --accent: #7556C8 !important;
    }
    .mission-card.immersive-card[style*="#2563EB"],
    .mission-card[style*="#2563EB"] {
        --accent: #4A78D4 !important;
    }
    .mission-card.immersive-card[style*="#0F766E"],
    .mission-card[style*="#0F766E"] {
        --accent: #2C8FA3 !important;
    }

    /* Small polish for guide cards so they match the new softer sidebar / mission system. */
    .guide-step,
    .metric-card,
    .panel-card,
    .queue-card,
    .story-card,
    .playbook-card,
    .signal-card,
    .ux-card,
    .routing-card,
    .account-queue-card,
    .question-card {
        border-color: rgba(217,226,239,.95) !important;
        box-shadow: 0 14px 34px rgba(15, 23, 42, .06) !important;
    }

    @media (max-width: 1400px) {
        .mission-grid { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    }
    @media (max-width: 980px) {
        .mission-grid { grid-template-columns: repeat(1, minmax(0, 1fr)) !important; }
        .mission-card, .mission-card.immersive-card { min-height: auto !important; }
        .mission-title, .mission-body { min-height: auto !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Customer Value & Growth — polished operator experience, final visual layer.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
    --v17-ink: #102237; --v17-muted: #687B90; --v17-blue: #2666DA;
    --v17-green: #0D8E80; --v17-violet: #6951C7; --v17-amber: #CE8A26;
    --v17-stroke: #D9E2EC; --v17-border: #CBD8E5;
    --navy: #102237; --border: #D9E2EC; --muted: #687B90;
}
.stApp { background: linear-gradient(140deg,#F3F7FB,#EDF3F8 48%,#F8FAFC) !important; }
html,body,p,button,input,textarea,.stApp,[class*="css"] { font-family: 'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif; }
h1,h2,h3,h4,.hero-title,.section-title,.mission-title,.metric-value,.v17-workspace-title { font-family:'Space Grotesk','DM Sans',sans-serif !important; }
.main .block-container { max-width: 1510px !important; padding-top: 1.15rem !important; }
.hero { border-radius: 25px !important; padding: 32px 40px !important; margin-bottom: 22px !important;
background: linear-gradient(117deg,#081629 0%,#123B58 65%,#0E6679 100%) !important;
box-shadow:0 14px 34px rgba(7,26,44,.20) !important; }
.hero-title { font-size:44px; font-weight:700 !important; letter-spacing:-.05em; }
.hero-highlight { color:#72DDD6; }
.hero-eyebrow { font-weight:800; font-size:10px; letter-spacing:.16em; }
.status-pulse { width:7px;height:7px;display:inline-block;border-radius:50%;background:#55E3B1;box-shadow:0 0 0 4px rgba(85,227,177,.13); }
.hero-foot { margin-top:22px;display:flex;gap:10px;flex-wrap:wrap; }
.hero-foot span { font-size:10px; font-weight:800; letter-spacing:.10em; color:#DDEDF3; border:1px solid rgba(255,255,255,.20);background:rgba(255,255,255,.06);border-radius:8px;padding:7px 10px; }
.v17-workspace-head { margin: 6px 0 20px; padding: 2px 0 12px; border-bottom:1px solid #D8E2EC; }
.v17-workspace-eyebrow { color:#286F93;font-size:10px;font-weight:900;letter-spacing:.16em;margin-bottom:10px; }
.v17-workspace-top { display:flex;align-items:end;justify-content:space-between;gap:25px; }
.v17-workspace-title { color:#102237;font-size:27px;letter-spacing:-.035em;font-weight:700;margin:0 0 8px; }
.v17-workspace-sub { font-size:14px;color:#52647B;line-height:1.55;max-width:860px;margin:0; }
.v17-data-chip { display:inline-flex;flex-shrink:0;white-space:nowrap;border-radius:9px;padding:10px 12px;background:#E8F0F7;border:1px solid #D0DFEB;color:#496A82;font-size:11px;font-weight:800;letter-spacing:.01em; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#FAFCFE,#EDF4F8) !important; border-right:1px solid #D4DFEA !important; }
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p { color:#495E76 !important; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] label { padding:4px 0 !important;font-weight:750; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {font-size:12px;}
.section-title {font-size:22px !important;font-weight:700 !important;}
.section-title::before { width:9px !important;height:9px !important;border-radius:3px !important;box-shadow:none !important; }
.mission-grid { gap:13px !important; }
.mission-card,.mission-card.immersive-card { border-radius:17px !important;min-height:275px !important;padding:20px !important;
border:1px solid #D9E2EC !important;border-top:3px solid var(--accent,#2666DA) !important;
box-shadow:0 6px 20px rgba(15,39,70,.055) !important;
background:linear-gradient(180deg,#FFFFFF,#F9FBFD) !important; }
.mission-card:hover,.mission-card.immersive-card:hover { transform:translateY(-3px) !important;box-shadow:0 14px 32px rgba(15,39,70,.10) !important; }
.mission-icon-orb {height:42px !important;width:42px !important;border-radius:12px !important;box-shadow:none !important;margin-bottom:12px !important;font-size:18px !important; }
.mission-title {font-size:16px !important;min-height:38px !important;line-height:1.2 !important;color:#152A42 !important;}
.mission-kicker {color:#708399 !important;font-size:10px !important;}
.mission-value {font-size:31px !important;min-height:35px !important;font-weight:700 !important;}
.mission-body {min-height:64px !important;font-size:12.5px !important;line-height:1.5 !important;}
.card-direction,.mission-cta {border-radius:10px !important;padding:9px 10px !important;background:#EEF4FA !important;color:#42647D !important;border:1px solid #DFE8F1 !important;font-weight:800 !important;}
.immediate-account-card,.account-queue-card,.question-card,.story-card,.signal-brief-card,
.metric-card,.panel-card,.queue-card,.routing-card,.guide-step,.playbook-card,.ux-card,.scenario-card {
border:1px solid #DAE3ED !important;border-radius:17px !important;box-shadow:0 6px 18px rgba(15,39,70,.05) !important; }
.guide-grid {gap:9px;}.guide-step{min-height:122px;padding:14px;}
.guide-number{width:28px;height:28px;border-radius:8px !important;font-size:12px;box-shadow:none;}
.guide-title{font-size:13px;}.guide-body{font-size:11px;line-height:1.4;}
.daily-brief{ background:linear-gradient(120deg,#FFFFFF,#EAF5F5) !important;border:1px solid #D2E6E7 !important;border-left:4px solid #168E8A !important;border-radius:17px !important;box-shadow:0 5px 20px rgba(15,39,70,.05) !important;padding:20px 22px !important; }
.daily-kicker{font-size:10px !important;letter-spacing:.13em !important;}
.daily-title{font-size:22px !important;font-weight:700 !important;}
.daily-body{font-size:13px !important;}
.risk-core{box-shadow:0 18px 42px rgba(18,44,78,.19) !important;}
.stButton > button,div[data-testid="stButton"] > button,
.stDownloadButton > button,.stFormSubmitButton > button {background:#173A5F !important;color:#FFF !important;
border-radius:11px !important;min-height:44px !important;border:1px solid #173A5F !important;padding:.57rem .86rem !important;
font-weight:750 !important;font-size:12px !important;box-shadow:0 5px 12px rgba(23,58,95,.11) !important;}
.stButton > button:hover,div[data-testid="stButton"] > button:hover{background:#1E5278 !important;transform:translateY(-1px) !important;box-shadow:0 10px 19px rgba(23,58,95,.16) !important;}
.stButton > button[kind="secondary"],div[data-testid="stButton"] > button[kind="secondary"] {
background:#FFFFFF !important;color:#173A5F !important;border:1px solid #B9CADA !important;box-shadow:0 4px 12px rgba(23,58,95,.05) !important;}
div[data-testid="stButton"] > button[kind="secondary"] p,.stButton > button[kind="secondary"] p {color:#173A5F !important;}
.stButton > button:focus-visible,.stDownloadButton > button:focus-visible {outline:3px solid #3F84DE !important;outline-offset:2px !important;}
[data-baseweb="tab-list"]{background:rgba(255,255,255,.72) !important;border:1px solid #D9E3ED !important;border-radius:12px !important;padding:5px !important;}
button[data-baseweb="tab"]{border-radius:8px !important;font-size:12px !important;font-weight:800 !important;}
button[data-baseweb="tab"][aria-selected="true"] {background:#E7F1FA !important;color:#17477A !important;border-color:#C8DEEF !important;}
.v17-decision-strip{display:flex;align-items:center;justify-content:space-between;gap:18px;
background:linear-gradient(105deg,#0E243C,#174A60);border-radius:17px;padding:19px 23px;margin:18px 0 13px;color:white;}
.v17-ds-label{font-weight:800;color:#94D4D1;font-size:10px;letter-spacing:.14em;margin-bottom:9px;}
.v17-ds-action{color:white;font-family:'Space Grotesk',sans-serif;font-size:20px;line-height:1.35;max-width:1080px;}
.v17-ds-meta{color:#C5D9E7;font-size:12px;margin-top:8px;}
.v17-ds-tag{border-radius:7px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.18);padding:9px 12px;color:#D9F4E9;white-space:nowrap;font-weight:800;font-size:10px;letter-spacing:.10em;}
.v17-flow-steps{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:12px 0 16px;}
.v17-flow-stage{background:linear-gradient(180deg,#FFF,#FAFCFE);border:1px solid #D5E1EC;border-top:4px solid #A6B6C8;border-radius:15px;padding:15px;min-height:154px;}
.v17-flow-stage.is-complete{border-top-color:#0C9A87;background:linear-gradient(180deg,#F7FFFD,#FFFFFF);}
.v17-flow-stage.is-in-progress{border-top-color:#377BDD;background:linear-gradient(180deg,#F5F9FF,#FFFFFF);}
.v17-flow-stage.is-attention{border-top-color:#D99A37;background:linear-gradient(180deg,#FFF9F0,#FFFFFF);}
.v17-flow-top{display:flex;align-items:center;justify-content:space-between;gap:4px;margin-bottom:13px;}
.v17-flow-num{font-family:'Space Grotesk',sans-serif;color:#96A8B8;font-weight:700;font-size:19px;}
.v17-flow-state{font-size:10px;font-weight:800;color:#516A80;text-align:right;}
.v17-flow-title{color:#10293E;font-size:14px;font-weight:800;margin-bottom:7px;letter-spacing:-.01em;}
.v17-flow-desc{color:#667D90;font-size:11px;line-height:1.5;}
.v17-next-action{background:linear-gradient(110deg,#E9F6F5,#EFF4FA);border:1px solid #C9E5E5;border-radius:16px;padding:18px 21px;margin:16px 0;display:flex;justify-content:space-between;align-items:center;gap:12px;}
.v17-next-action strong{color:#123C55;display:block;font-size:17px;font-family:'Space Grotesk',sans-serif;margin-top:5px;}
.v17-next-action p{color:#4A6375;margin:6px 0 0;font-size:12.5px;line-height:1.5;}
.v17-next-label{color:#13877F;font-weight:900;font-size:10px;letter-spacing:.12em;}
.v17-next-symbol{color:#13877F;font-size:30px;}
@media(max-width:1200px){.v17-flow-steps{grid-template-columns:repeat(3,minmax(0,1fr));}.v17-workspace-top{align-items:start;}}
@media(max-width:840px){.v17-flow-steps{grid-template-columns:1fr 1fr;}.v17-workspace-top{display:block;}.v17-data-chip{margin-top:12px;}.hero{padding:26px!important;}.hero-title{font-size:33px!important;}.v17-decision-strip{display:block;}.v17-ds-tag{display:none;}}
@media(max-width:520px){.v17-flow-steps{grid-template-columns:1fr;}}
</style>
""", unsafe_allow_html=True)

# Targeted opportunity briefing styles.
st.markdown("""
<style>
.v17-growth-brief {border:1px solid #D6E2EB;border-top:4px solid #D09A47;border-radius:17px;padding:21px 22px;margin:12px 0 18px;background:linear-gradient(120deg,#FFF,#F8FBFD);box-shadow:0 7px 22px rgba(17,41,66,.055);}
.v17-growth-brief.ready {border-top-color:#0A9A87;}
.v17-growth-header {display:flex;align-items:start;justify-content:space-between;gap:16px;margin-bottom:18px;}
.v17-growth-header h3 {font-size:21px;line-height:1.2;margin:7px 0 0;color:#102237;}
.v17-growth-kicker{font-size:10px;font-weight:900;color:#4F7A8F;letter-spacing:.13em;}
.v17-growth-badge{border-radius:9px;background:#EEF4F7;padding:9px 12px;border:1px solid #DDE6EF;font-size:11px;font-weight:800;color:#275A76;}
.v17-growth-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-bottom:14px;}
.v17-growth-kpis>div{background:#F3F8FC;border:1px solid #E1E9F0;border-radius:11px;padding:13px 14px;min-height:78px;}
.v17-growth-kpis small{display:block;font-size:10px;color:#70859A;letter-spacing:.10em;font-weight:800;margin-bottom:8px;}
.v17-growth-kpis strong{display:block;color:#18364E;font-size:16px;line-height:1.25;overflow-wrap:anywhere;}
.v17-growth-details{background:#F7FAFC;border:1px solid #E3EAF0;border-radius:11px;padding:13px 15px;}
.v17-growth-details span{display:block;font-size:10px;color:#547A91;font-weight:900;letter-spacing:.09em;}
.v17-growth-details p{color:#263E52;line-height:1.5;font-size:13px;margin:6px 0 0;}
@media(max-width:700px){.v17-growth-kpis{grid-template-columns:1fr;}.v17-growth-header{display:block;}.v17-growth-badge{margin-top:12px;display:inline-block;}}
</style>
""", unsafe_allow_html=True)
