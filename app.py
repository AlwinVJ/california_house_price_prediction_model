import os
import warnings
import joblib
import pandas as pd
import streamlit as st

# Suppress warnings and set CPU count for clean execution
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# Session State Initialization (Theme, Inputs, Prediction)
# ---------------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = None

if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

# Default values for inputs matching the reference application
if "val_longitude" not in st.session_state:
    st.session_state.val_longitude = -118.24

if "val_latitude" not in st.session_state:
    st.session_state.val_latitude = 34.05

if "val_housing_median_age" not in st.session_state:
    st.session_state.val_housing_median_age = 41.0

if "val_total_rooms" not in st.session_state:
    st.session_state.val_total_rooms = 6.984

if "val_total_bedrooms" not in st.session_state:
    st.session_state.val_total_bedrooms = 1.023

if "val_population" not in st.session_state:
    st.session_state.val_population = 322.0

if "val_households" not in st.session_state:
    st.session_state.val_households = 126.0

if "val_median_income" not in st.session_state:
    st.session_state.val_median_income = 8.3252

if "val_ocean_proximity" not in st.session_state:
    st.session_state.val_ocean_proximity = "<1H OCEAN"


# ---------------------------------------------------------
# Load Model (Cached)
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.joblib")


model = load_model()


# ---------------------------------------------------------
# Theme-based Design System & Colors (Matching Reference Image)
# ---------------------------------------------------------

is_dark = st.session_state.theme == "dark"

if is_dark:
    # DARK THEME PALETTE (Right half of reference image)
    bg_app = "#090d16"
    bg_card = "#121824"
    bg_card_secondary = "#172033"
    border_color = "#1f2a3c"
    border_focus = "#a855f7"
    text_primary = "#f8fafc"
    text_secondary = "#cbd5e1"
    text_muted = "#8092aa"
    text_icon = "#c084fc"
    accent_color = "#a855f7"
    accent_hover = "#9333ea"
    accent_light = "rgba(168, 85, 247, 0.12)"
    accent_glow = "rgba(168, 85, 247, 0.28)"
    input_bg = "#1c2434"
    input_border = "#2a374c"
    input_text = "#f8fafc"
    shadow_card = "0 4px 20px rgba(0, 0, 0, 0.45)"
    pred_card_bg = "linear-gradient(180deg, #1c1836 0%, #17152b 40%, #131422 70%, #2e1d4d 100%)"
    pred_card_border = "#372b54"
    pred_val_color = "#c084fc"
    pred_wave_color = "#3d2266"
    pred_circle_bg = "#271b42"
    pred_circle_border = "#433268"
    btn_gradient = "linear-gradient(135deg, #7c3aed 0%, #9333ea 100%)"
    btn_hover = "linear-gradient(135deg, #6d28d9 0%, #7e22ce 100%)"
else:
    # LIGHT THEME PALETTE (Left half of reference image)
    bg_app = "#f0f3f8"
    bg_card = "#ffffff"
    bg_card_secondary = "#f8fafc"
    border_color = "#e2e8f0"
    border_focus = "#2563eb"
    text_primary = "#0f172a"
    text_secondary = "#475569"
    text_muted = "#94a3b8"
    text_icon = "#2563eb"
    accent_color = "#2563eb"
    accent_hover = "#1d4ed8"
    accent_light = "rgba(37, 99, 235, 0.08)"
    accent_glow = "rgba(37, 99, 235, 0.2)"
    input_bg = "#f8fafc"
    input_border = "#e2e8f0"
    input_text = "#0f172a"
    shadow_card = "0 4px 20px rgba(0, 0, 0, 0.04)"
    pred_card_bg = "linear-gradient(180deg, #dbeafe 0%, #edf5ff 40%, #ffffff 75%, #dbeafe 100%)"
    pred_card_border = "#bfdbfe"
    pred_val_color = "#1d4ed8"
    pred_wave_color = "#93c5fd"
    pred_circle_bg = "#ffffff"
    pred_circle_border = "#bfdbfe"
    btn_gradient = "linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%)"
    btn_hover = "linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%)"

css_rules = f"""<style>
/* --------------------------------------------------
   CSS VARIABLES & APP THEME TOKENS
   -------------------------------------------------- */
:root {{
    --bg-app: {bg_app};
    --bg-card: {bg_card};
    --bg-card-secondary: {bg_card_secondary};
    --border-color: {border_color};
    --border-focus: {border_focus};
    --text-primary: {text_primary};
    --text-secondary: {text_secondary};
    --text-muted: {text_muted};
    --text-icon: {text_icon};
    --accent: {accent_color};
    --accent-hover: {accent_hover};
    --accent-light: {accent_light};
    --accent-glow: {accent_glow};
    --input-bg: {input_bg};
    --input-border: {input_border};
    --input-text: {input_text};
    --shadow-card: {shadow_card};
    --pred-card-bg: {pred_card_bg};
    --pred-card-border: {pred_card_border};
    --pred-val-color: {pred_val_color};
    --pred-wave-color: {pred_wave_color};
    --pred-circle-bg: {pred_circle_bg};
    --pred-circle-border: {pred_circle_border};
    --btn-gradient: {btn_gradient};
    --btn-hover: {btn_hover};
}}

/* Hide Default Streamlit Chrome for Standalone Web App Look */
header[data-testid="stHeader"] {{
    display: none !important;
    height: 0 !important;
}}
footer {{
    display: none !important;
}}
#MainMenu {{
    display: none !important;
}}
div[data-testid="stToolbar"] {{
    display: none !important;
}}
div[data-testid="stDecoration"] {{
    display: none !important;
}}
div[data-testid="stStatusWidget"] {{
    display: none !important;
}}

/* Application Shell Background & Typography */
.stApp {{
    background-color: var(--bg-app) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}}

/* Desktop Viewport Optimization: No Vertical Scrolling */
.block-container {{
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1180px !important;
}}

/* Spacing between vertical blocks */
div[data-testid="stVerticalBlock"] > div {{
    gap: 0.22rem !important;
}}

/* Top Header Bar */
.top-header-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.15rem 0 0.35rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 0.35rem;
}}

.brand-wrapper {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
}}

.brand-icon {{
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-primary);
}}

.brand-title {{
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
}}

/* Theme Toggle Button in Header (Exact Square Pill from Reference Image) */
div[data-testid="column"]:nth-child(2) button[kind="secondary"] {{
    width: 36px !important;
    height: 36px !important;
    min-height: 36px !important;
    border-radius: 9px !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-size: 1.1rem !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
    float: right !important;
}}

div[data-testid="column"]:nth-child(2) button[kind="secondary"]:hover {{
    border-color: var(--accent) !important;
    transform: translateY(-1px);
}}

/* Hero Heading Section (Matching Reference Image) */
.hero-heading-container {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.1rem 0 0.25rem 0;
    margin-bottom: 0.3rem;
}}

.hero-text-group {{
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}}

.hero-title {{
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0 !important;
    line-height: 1.2 !important;
}}

.hero-title-accent {{
    color: var(--accent) !important;
}}

.hero-subtitle {{
    font-size: 0.84rem;
    color: var(--text-muted);
    margin: 0 !important;
}}

.hero-illustration-box {{
    display: flex;
    align-items: center;
    justify-content: center;
}}

/* Left Card: House Details Form Wrapper */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow-card) !important;
    padding: 0.95rem 1.15rem !important;
}}

/* Card Header Title */
.card-header-bar {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.45rem;
    padding-bottom: 0.3rem;
}}

.card-header-icon {{
    display: flex;
    align-items: center;
    color: var(--text-icon);
}}

.card-header-title {{
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text-primary);
}}

/* Input Row Layout: Label on Left, Input on Right */
.param-row-label {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
    white-space: nowrap;
    user-select: none;
    height: 34px;
}}

.param-icon {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    color: var(--text-secondary);
}}

.param-info-tag {{
    font-size: 0.7rem;
    color: var(--text-muted);
    cursor: help;
    margin-left: 0.15rem;
}}

/* Input Fields Overrides: NO Dark Mode Leak into Light Mode */
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"] {{
    margin-bottom: 0 !important;
}}

div[data-baseweb="input"],
div[data-baseweb="base-input"] {{
    background-color: var(--input-bg) !important;
    border-color: var(--input-border) !important;
}}

div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    min-height: 32px !important;
    height: 32px !important;
    max-height: 32px !important;
    border-radius: 7px !important;
    border: 1px solid var(--input-border) !important;
    background-color: var(--input-bg) !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div {{
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}}

div[data-testid="stNumberInput"] input {{
    color: var(--input-text) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    height: 30px !important;
    padding: 0 0.5rem !important;
    background-color: transparent !important;
    text-align: left !important;
}}

div[data-testid="stNumberInput"] button {{
    height: 30px !important;
    width: 20px !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
}}

div[data-testid="stNumberInput"] button:hover {{
    color: var(--accent) !important;
}}

div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {{
    color: var(--input-text) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    line-height: 30px !important;
}}

div[data-testid="stSelectbox"] svg {{
    fill: var(--text-secondary) !important;
}}

/* Dropdown Menu Popover Options Contrast */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow-card) !important;
}}

li[role="option"] {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
}}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {{
    background-color: var(--accent-light) !important;
    color: var(--accent) !important;
}}

/* Predict Button (Exact Full Width Button from Reference Image) */
button[kind="primary"] {{
    background: var(--btn-gradient) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    height: 40px !important;
    min-height: 40px !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 10px var(--accent-glow) !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}}

button[kind="primary"]:hover {{
    background: var(--btn-hover) !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
    transform: translateY(-1px);
}}

/* Right Card: Prediction Result Card (Matching Reference Image) */
.prediction-card-box {{
    background: var(--pred-card-bg);
    border: 1px solid var(--pred-card-border);
    border-radius: 16px;
    padding: 1.4rem 1.2rem 0 1.2rem;
    box-shadow: var(--shadow-card);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    min-height: 380px;
    margin-bottom: 0.45rem;
    position: relative;
    overflow: hidden;
}}

.pred-circle-icon {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: var(--pred-circle-bg);
    border: 1.5px solid var(--pred-circle-border);
    color: var(--accent);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    margin-top: 0.5rem;
}}

.pred-center-group {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    margin: 1.2rem 0 1rem 0;
}}

.pred-label {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}}

.pred-price {{
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--pred-val-color);
}}

.pred-price-placeholder {{
    color: var(--text-muted);
    font-size: 2.8rem;
    font-weight: 700;
}}

.pred-subtext {{
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
}}

.pred-bottom-wave {{
    width: 100%;
    margin-bottom: -4px;
    line-height: 0;
}}

/* Model Info Popover Button */
div[data-testid="stPopover"] > button {{
    height: 35px !important;
    min-height: 35px !important;
    border-radius: 9px !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stPopover"] > button:hover {{
    color: var(--text-primary) !important;
    border-color: var(--accent) !important;
    background: var(--bg-card-secondary) !important;
}}

/* Popover Floating Body */
div[data-testid="stPopoverBody"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-card) !important;
    padding: 0.85rem !important;
    color: var(--text-primary) !important;
    min-width: 270px !important;
}}

.popover-header {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border-color);
}}

.popover-title {{
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-primary);
}}

.popover-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}}

.popover-table td {{
    padding: 0.38rem 0.4rem;
    border-bottom: 1px solid var(--border-color);
}}

.popover-key {{
    font-weight: 600;
    color: var(--text-secondary);
}}

.popover-val {{
    font-weight: 500;
    color: var(--text-primary);
    text-align: right;
}}

/* Responsive Media Queries */
@media (max-width: 860px) {{
    .block-container {{
        padding: 0.75rem 0.9rem !important;
    }}
    .hero-heading-container {{
        flex-direction: row;
    }}
    .prediction-card-box {{
        min-height: 280px;
    }}
}}
</style>"""

st.markdown(css_rules, unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Header Section: Brand and Theme Toggle (Matching Reference)
# ---------------------------------------------------------

header_col1, header_col2 = st.columns([0.9, 0.1], vertical_alignment="center")

with header_col1:
    header_html = """<div class="brand-wrapper">
<div class="brand-icon">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9.5z"></path>
<polyline points="9 21 9 12 15 12 15 21"></polyline>
</svg>
</div>
<span class="brand-title">House Price Prediction</span>
</div>"""
    st.markdown(header_html, unsafe_allow_html=True)

with header_col2:
    toggle_icon = "☀️" if is_dark else "🌙"
    if st.button(toggle_icon, key="theme_toggle_btn", use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

# Subtle Header Divider
st.markdown('<div style="border-bottom: 1px solid var(--border-color); margin-bottom: 0.35rem;"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. Main Heading Section with Cozy House Illustration
# ---------------------------------------------------------

if is_dark:
    # Nighttime house illustration for dark mode
    illustration_svg = """<svg class="hero-illustration-svg" width="128" height="64" viewBox="0 0 128 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<ellipse cx="64" cy="56" rx="58" ry="8" fill="#17142b"/>
<circle cx="28" cy="42" r="14" fill="#241b3d"/>
<circle cx="98" cy="38" r="16" fill="#211938"/>
<circle cx="108" cy="44" r="12" fill="#291e45"/>
<circle cx="20" cy="46" r="11" fill="#1d1633"/>
<rect x="42" y="26" width="44" height="30" rx="2" fill="#2e264a"/>
<path d="M38 27L64 8L90 27H38Z" fill="#1e1833" stroke="#544378" stroke-width="1.5" stroke-linejoin="round"/>
<rect x="74" y="12" width="6" height="12" fill="#1e1833"/>
<rect x="49" y="32" width="10" height="10" rx="1" fill="#fbbf24"/>
<rect x="67" y="32" width="10" height="10" rx="1" fill="#fbbf24"/>
<line x1="54" y1="32" x2="54" y2="42" stroke="#2e264a" stroke-width="1"/>
<line x1="49" y1="37" x2="59" y2="37" stroke="#2e264a" stroke-width="1"/>
<line x1="72" y1="32" x2="72" y2="42" stroke="#2e264a" stroke-width="1"/>
<line x1="67" y1="37" x2="77" y2="37" stroke="#2e264a" stroke-width="1"/>
<rect x="58" y="44" width="8" height="12" fill="#1a142e"/>
<circle cx="88" cy="14" r="5" fill="#fef08a" opacity="0.6"/>
</svg>"""
else:
    # Daytime house illustration for light mode
    illustration_svg = """<svg class="hero-illustration-svg" width="128" height="64" viewBox="0 0 128 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<ellipse cx="64" cy="56" rx="58" ry="8" fill="#e0f2fe"/>
<circle cx="28" cy="42" r="14" fill="#6ee7b7"/>
<circle cx="98" cy="38" r="16" fill="#34d399"/>
<circle cx="108" cy="44" r="12" fill="#10b981"/>
<circle cx="20" cy="46" r="11" fill="#059669"/>
<rect x="42" y="26" width="44" height="30" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
<path d="M38 27L64 8L90 27H38Z" fill="#3b82f6" stroke="#2563eb" stroke-width="1.5" stroke-linejoin="round"/>
<rect x="74" y="12" width="6" height="12" fill="#1e40af"/>
<rect x="49" y="32" width="10" height="10" rx="1" fill="#bae6fd"/>
<rect x="67" y="32" width="10" height="10" rx="1" fill="#bae6fd"/>
<line x1="54" y1="32" x2="54" y2="42" stroke="#0284c7" stroke-width="1"/>
<line x1="49" y1="37" x2="59" y2="37" stroke="#0284c7" stroke-width="1"/>
<line x1="72" y1="32" x2="72" y2="42" stroke="#0284c7" stroke-width="1"/>
<line x1="67" y1="37" x2="77" y2="37" stroke="#0284c7" stroke-width="1"/>
<rect x="58" y="44" width="8" height="12" fill="#1e3a8a"/>
<circle cx="104" cy="14" r="6" fill="#fef08a" opacity="0.8"/>
</svg>"""

hero_html = f"""<div class="hero-heading-container">
<div class="hero-text-group">
<h1 class="hero-title">House Price <span class="hero-title-accent">Prediction</span></h1>
<p class="hero-subtitle">Estimate house prices in California using a machine learning model.</p>
</div>
<div class="hero-illustration-box">
{illustration_svg}
</div>
</div>"""

st.markdown(hero_html, unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. Two-Column Dashboard Layout
# ---------------------------------------------------------

col_left, col_right = st.columns([1.18, 0.92], gap="medium")


# ---------------------------------------------------------
# Left Column: House Details Card (Matching Reference Image)
# ---------------------------------------------------------

with col_left:
    with st.container(border=True):
        card_header_html = """<div class="card-header-bar">
<div class="card-header-icon">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 2l7 4v6c0 5-3.5 9-7 10-3.5-1-7-5-7-10V6l7-4z"></path>
<path d="M9 12l2 2 4-4"></path>
</svg>
</div>
<span class="card-header-title">House Details</span>
</div>"""
        st.markdown(card_header_html, unsafe_allow_html=True)

        # Row 1: Longitude
        r1_lbl, r1_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r1_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 21s-8-7.5-8-12a8 8 0 1 1 16 0c0 4.5-8 12-8 12z"></path>
<circle cx="12" cy="9" r="2.5"></circle>
</svg>
</span>
<span>Longitude</span>
<span class="param-info-tag">ⓘ</span>
</div>""", unsafe_allow_html=True)
        with r1_inp:
            longitude = st.number_input(
                "Longitude",
                format="%.2f",
                value=float(st.session_state.val_longitude),
                key="num_longitude",
                label_visibility="collapsed"
            )
            st.session_state.val_longitude = longitude

        # Row 2: Latitude
        r2_lbl, r2_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r2_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="9"></circle>
<line x1="12" y1="3" x2="12" y2="21"></line>
<line x1="3" y1="12" x2="21" y2="12"></line>
</svg>
</span>
<span>Latitude</span>
</div>""", unsafe_allow_html=True)
        with r2_inp:
            latitude = st.number_input(
                "Latitude",
                format="%.2f",
                value=float(st.session_state.val_latitude),
                key="num_latitude",
                label_visibility="collapsed"
            )
            st.session_state.val_latitude = latitude

        # Row 3: Housing Median Age
        r3_lbl, r3_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r3_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
<polyline points="9 22 9 12 15 12 15 22"></polyline>
</svg>
</span>
<span>Housing Median Age (years)</span>
</div>""", unsafe_allow_html=True)
        with r3_inp:
            housing_median_age = st.number_input(
                "Housing Median Age",
                min_value=1.0,
                step=1.0,
                format="%.1f",
                value=float(st.session_state.val_housing_median_age),
                key="num_housing_median_age",
                label_visibility="collapsed"
            )
            st.session_state.val_housing_median_age = housing_median_age

        # Row 4: Total Rooms
        r4_lbl, r4_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r4_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<rect x="3" y="3" width="18" height="18" rx="2"></rect>
<line x1="3" y1="12" x2="21" y2="12"></line>
<line x1="12" y1="3" x2="12" y2="21"></line>
</svg>
</span>
<span>Total Rooms</span>
</div>""", unsafe_allow_html=True)
        with r4_inp:
            total_rooms = st.number_input(
                "Total Rooms",
                min_value=1.0,
                step=1.0,
                format="%.3f",
                value=float(st.session_state.val_total_rooms),
                key="num_total_rooms",
                label_visibility="collapsed"
            )
            st.session_state.val_total_rooms = total_rooms

        # Row 5: Total Bedrooms
        r5_lbl, r5_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r5_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M2 4v16"></path>
<path d="M2 8h18a2 2 0 0 1 2 2v10"></path>
<path d="M2 17h20"></path>
<circle cx="6" cy="12" r="2"></circle>
</svg>
</span>
<span>Total Bedrooms</span>
</div>""", unsafe_allow_html=True)
        with r5_inp:
            total_bedrooms = st.number_input(
                "Total Bedrooms",
                min_value=1.0,
                step=1.0,
                format="%.3f",
                value=float(st.session_state.val_total_bedrooms),
                key="num_total_bedrooms",
                label_visibility="collapsed"
            )
            st.session_state.val_total_bedrooms = total_bedrooms

        # Row 6: Population
        r6_lbl, r6_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r6_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 21s-8-7.5-8-12a8 8 0 1 1 16 0c0 4.5-8 12-8 12z"></path>
</svg>
</span>
<span>Population</span>
</div>""", unsafe_allow_html=True)
        with r6_inp:
            population = st.number_input(
                "Population",
                min_value=0.0,
                step=1.0,
                format="%.1f",
                value=float(st.session_state.val_population),
                key="num_population",
                label_visibility="collapsed"
            )
            st.session_state.val_population = population

        # Row 7: Households
        r7_lbl, r7_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r7_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
<circle cx="9" cy="7" r="4"></circle>
<path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
<path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
</svg>
</span>
<span>Households</span>
</div>""", unsafe_allow_html=True)
        with r7_inp:
            households = st.number_input(
                "Households",
                min_value=0.0,
                step=1.0,
                format="%.1f",
                value=float(st.session_state.val_households),
                key="num_households",
                label_visibility="collapsed"
            )
            st.session_state.val_households = households

        # Row 8: Median Income
        r8_lbl, r8_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r8_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<line x1="18" y1="20" x2="18" y2="10"></line>
<line x1="12" y1="20" x2="12" y2="4"></line>
<line x1="6" y1="20" x2="6" y2="14"></line>
</svg>
</span>
<span>Median Income (×$10,000)</span>
</div>""", unsafe_allow_html=True)
        with r8_inp:
            median_income = st.number_input(
                "Median Income",
                min_value=0.0,
                step=0.1,
                format="%.4f",
                value=float(st.session_state.val_median_income),
                key="num_median_income",
                label_visibility="collapsed"
            )
            st.session_state.val_median_income = median_income

        # Row 9: Ocean Proximity
        r9_lbl, r9_inp = st.columns([1.55, 1.05], vertical_alignment="center")
        with r9_lbl:
            st.markdown("""<div class="param-row-label">
<span class="param-icon">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M2 12c2.5-3 5-3 7.5 0s5 3 7.5 0 5-3 7 0"></path>
<path d="M2 17c2.5-3 5-3 7.5 0s5 3 7.5 0 5-3 7 0"></path>
</svg>
</span>
<span>Ocean Proximity</span>
</div>""", unsafe_allow_html=True)
        with r9_inp:
            ocean_options = [
                "<1H OCEAN",
                "INLAND",
                "ISLAND",
                "NEAR BAY",
                "NEAR OCEAN"
            ]
            curr_ocean = st.session_state.val_ocean_proximity
            ocean_idx = ocean_options.index(curr_ocean) if curr_ocean in ocean_options else 0
            ocean_proximity = st.selectbox(
                "Ocean Proximity",
                ocean_options,
                index=ocean_idx,
                key="sel_ocean_proximity",
                label_visibility="collapsed"
            )
            st.session_state.val_ocean_proximity = ocean_proximity

        # Predict Button with Play triangle icon
        predict_clicked = st.button("▷  Predict House Price", use_container_width=True, type="primary", key="btn_predict")


# ---------------------------------------------------------
# Prediction Logic (Exact Backend ML Logic Preserved)
# ---------------------------------------------------------

if predict_clicked:
    if ocean_proximity == "None" or ocean_proximity == "Select the proximity level from the drop down list":
        ocean_value = None
    else:
        ocean_value = ocean_proximity

    new_row = pd.DataFrame(
        [{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": int(housing_median_age),
            "total_rooms": int(total_rooms),
            "total_bedrooms": int(total_bedrooms),
            "population": int(population),
            "households": int(households),
            "median_income": median_income,
            "ocean_proximity": ocean_value
        }]
    )

    try:
        prediction = model.predict(new_row)[0]
        # Target is expressed in units of $100,000
        predicted_price = prediction * 100000
        st.session_state.predicted_price = predicted_price
        st.session_state.prediction_error = None
    except Exception as e:
        st.session_state.prediction_error = str(e)
        st.session_state.predicted_price = None


# ---------------------------------------------------------
# Right Column: Prediction Result Card & Model Info Popover
# ---------------------------------------------------------

with col_right:
    price_val = st.session_state.get("predicted_price", None)
    err_val = st.session_state.get("prediction_error", None)

    if price_val is not None:
        price_display = f"${price_val:,.0f}"
        price_class = "pred-price"
        subtext_display = "Estimated Median Block Value"
    elif err_val is not None:
        price_display = "Error"
        price_class = "pred-price-placeholder"
        subtext_display = f"Error: {err_val}"
    else:
        price_display = "--"
        price_class = "pred-price pred-price-placeholder"
        subtext_display = "Click 'Predict House Price' to calculate."

    wave_svg = f"""<svg class="pred-bottom-wave" viewBox="0 0 320 80" fill="none" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
<path d="M0 45C60 20 120 60 180 35C240 10 280 40 320 25V80H0V45Z" fill="{pred_wave_color}" opacity="0.35"/>
<path d="M0 55C80 30 160 65 240 40C280 28 300 45 320 38V80H0V55Z" fill="{pred_wave_color}" opacity="0.5"/>
<path d="M0 65C90 45 180 75 270 55C295 50 310 58 320 54V80H0V65Z" fill="{pred_wave_color}" opacity="0.8"/>
</svg>"""

    house_badge_svg = """<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9.5z"></path>
<polyline points="9 21 9 12 15 12 15 21"></polyline>
</svg>"""

    pred_card_html = f"""<div class="prediction-card-box">
<div class="pred-circle-icon">
{house_badge_svg}
</div>
<div class="pred-center-group">
<div class="pred-label">Predicted House Price</div>
<div class="{price_class}">{price_display}</div>
<div class="pred-subtext">{subtext_display}</div>
</div>
{wave_svg}
</div>"""

    st.markdown(pred_card_html, unsafe_allow_html=True)

    # Model Info Popover (Streamlit native popover)
    with st.popover("📊  Model Info", use_container_width=True):
        popover_html = """<div class="popover-wrapper">
<div class="popover-header">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<line x1="18" y1="20" x2="18" y2="10"></line>
<line x1="12" y1="20" x2="12" y2="4"></line>
<line x1="6" y1="20" x2="6" y2="14"></line>
</svg>
<span class="popover-title">Model Info</span>
</div>
<table class="popover-table">
<tbody>
<tr>
<td class="popover-key">Model</td>
<td class="popover-val">Ridge Regression</td>
</tr>
<tr>
<td class="popover-key">Dataset</td>
<td class="popover-val">California Housing</td>
</tr>
<tr>
<td class="popover-key">Target</td>
<td class="popover-val">Median House Value</td>
</tr>
</tbody>
</table>
</div>"""
        st.markdown(popover_html, unsafe_allow_html=True)
