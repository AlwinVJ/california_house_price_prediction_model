import os
import warnings
import joblib
import pandas as pd
import streamlit as st

# Suppress sklearn unpickling and CPU count warnings for clean output
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

# Default values for inputs to persist across theme toggles
if "val_longitude" not in st.session_state:
    st.session_state.val_longitude = -124.35

if "val_latitude" not in st.session_state:
    st.session_state.val_latitude = 32.54

if "val_housing_median_age" not in st.session_state:
    st.session_state.val_housing_median_age = 1

if "val_total_rooms" not in st.session_state:
    st.session_state.val_total_rooms = 1

if "val_total_bedrooms" not in st.session_state:
    st.session_state.val_total_bedrooms = 1

if "val_population" not in st.session_state:
    st.session_state.val_population = 0

if "val_households" not in st.session_state:
    st.session_state.val_households = 0

if "val_median_income" not in st.session_state:
    st.session_state.val_median_income = 0.0

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
# Theme-based CSS Styling
# ---------------------------------------------------------

is_dark = st.session_state.theme == "dark"

if is_dark:
    # Dark Mode Palette: Deep navy/charcoal, purple accent, light gray text
    bg_app = "#090d16"
    bg_card = "#111827"
    bg_card_secondary = "#172033"
    border_color = "#1f2d42"
    border_focus = "#8b5cf6"
    text_primary = "#f8fafc"
    text_secondary = "#94a3b8"
    text_muted = "#64748b"
    accent_color = "#8b5cf6"
    accent_hover = "#7c3aed"
    accent_light = "rgba(139, 92, 246, 0.14)"
    accent_glow = "rgba(139, 92, 246, 0.28)"
    input_bg = "#0c1322"
    input_border = "#223147"
    input_text = "#f8fafc"
    shadow_card = "0 4px 20px rgba(0, 0, 0, 0.4)"
    pred_card_bg = "linear-gradient(145deg, #111827 0%, #1a1733 100%)"
    pred_card_border = "#373254"
    pred_val_color = "#c4b5fd"
    btn_gradient = "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)"
    btn_hover = "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)"
    badge_bg = "rgba(139, 92, 246, 0.15)"
    badge_text = "#c4b5fd"
    badge_border = "rgba(139, 92, 246, 0.3)"
else:
    # Light Mode Palette: Soft white/light blue, blue accent, dark navy text
    bg_app = "#f1f5f9"
    bg_card = "#ffffff"
    bg_card_secondary = "#f8fafc"
    border_color = "#e2e8f0"
    border_focus = "#2563eb"
    text_primary = "#0f172a"
    text_secondary = "#475569"
    text_muted = "#94a3b8"
    accent_color = "#2563eb"
    accent_hover = "#1d4ed8"
    accent_light = "rgba(37, 99, 235, 0.08)"
    accent_glow = "rgba(37, 99, 235, 0.2)"
    input_bg = "#fcfdfd"
    input_border = "#cbd5e1"
    input_text = "#0f172a"
    shadow_card = "0 2px 12px rgba(15, 23, 42, 0.05)"
    pred_card_bg = "linear-gradient(145deg, #ffffff 0%, #f0f7ff 100%)"
    pred_card_border = "#bfdbfe"
    pred_val_color = "#1d4ed8"
    btn_gradient = "linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)"
    btn_hover = "linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)"
    badge_bg = "#eff6ff"
    badge_text = "#1d4ed8"
    badge_border = "#bfdbfe"

css_rules = f"""
<style>
/* CSS Variables */
:root {{
    --bg-app: {bg_app};
    --bg-card: {bg_card};
    --bg-card-secondary: {bg_card_secondary};
    --border-color: {border_color};
    --border-focus: {border_focus};
    --text-primary: {text_primary};
    --text-secondary: {text_secondary};
    --text-muted: {text_muted};
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
    --btn-gradient: {btn_gradient};
    --btn-hover: {btn_hover};
    --badge-bg: {badge_bg};
    --badge-text: {badge_text};
    --badge-border: {badge_border};
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
    max-width: 1220px !important;
}}

/* Spacing between vertical blocks */
div[data-testid="stVerticalBlock"] > div {{
    gap: 0.3rem !important;
}}

/* Top Header Bar */
.top-header-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.15rem 0 0.4rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 0.35rem;
}}

.brand-wrapper {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
}}

.brand-icon-box {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: var(--accent-light);
    color: var(--accent);
    font-size: 1.15rem;
}}

.brand-title {{
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
}}

/* Theme Toggle Button in Header */
div[data-testid="column"]:nth-child(2) button[kind="secondary"] {{
    height: 32px !important;
    min-height: 32px !important;
    border-radius: 7px !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0 0.75rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="column"]:nth-child(2) button[kind="secondary"]:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--bg-card-secondary) !important;
}}

/* Hero Heading Section */
.hero-heading-container {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.1rem 0 0.3rem 0;
    margin-bottom: 0.25rem;
}}

.hero-text-group {{
    display: flex;
    flex-direction: column;
    gap: 0.05rem;
}}

.hero-title {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0 !important;
    line-height: 1.2 !important;
}}

.hero-subtitle {{
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 0 !important;
}}

.hero-illustration-box {{
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.3rem 0.55rem;
    background: var(--accent-light);
    border: 1px solid var(--badge-border);
    border-radius: 9px;
}}

.hero-illustration-svg {{
    width: 42px;
    height: 32px;
}}

/* Left Card: House Details Form Wrapper */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-card) !important;
    padding: 0.75rem 0.95rem !important;
}}

/* Card Header Title */
.card-header-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-color);
}}

.card-title-group {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.card-header-icon {{
    font-size: 0.95rem;
}}

.card-header-title {{
    font-size: 0.94rem;
    font-weight: 700;
    color: var(--text-primary);
}}

.card-header-badge {{
    font-size: 0.7rem;
    font-weight: 600;
    background: var(--badge-bg);
    color: var(--badge-text);
    border: 1px solid var(--badge-border);
    padding: 0.12rem 0.45rem;
    border-radius: 9999px;
}}

/* Input Labels Styling */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {{
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 0.08rem !important;
    padding-bottom: 0 !important;
    min-height: unset !important;
}}

div[data-testid="stNumberInput"] label p,
div[data-testid="stSelectbox"] label p {{
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin: 0 !important;
}}

/* Compact Input Fields: 36px to 38px height */
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    min-height: 36px !important;
    height: 36px !important;
    max-height: 36px !important;
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
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    height: 34px !important;
    padding: 0 0.5rem !important;
    background-color: transparent !important;
}}

div[data-testid="stNumberInput"] button {{
    height: 34px !important;
    width: 24px !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
}}

div[data-testid="stNumberInput"] button:hover {{
    color: var(--accent) !important;
    background: var(--bg-card-secondary) !important;
}}

div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {{
    color: var(--input-text) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    line-height: 34px !important;
}}

div[data-testid="stSelectbox"] svg {{
    fill: var(--text-secondary) !important;
}}

/* Predict Button */
button[kind="primary"] {{
    background: var(--btn-gradient) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    height: 38px !important;
    min-height: 38px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px var(--accent-glow) !important;
    transition: all 0.2s ease !important;
    margin-top: 0.25rem !important;
}}

button[kind="primary"]:hover {{
    background: var(--btn-hover) !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
    transform: translateY(-1px);
}}

/* Right Card: Prediction Result Card */
.prediction-card-box {{
    background: var(--pred-card-bg);
    border: 1px solid var(--pred-card-border);
    border-radius: 12px;
    padding: 1.05rem 1.05rem;
    box-shadow: var(--shadow-card);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 270px;
    margin-bottom: 0.45rem;
    position: relative;
    overflow: hidden;
}}

.prediction-card-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.pred-icon-circle {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: 9px;
    background: var(--accent-light);
    color: var(--accent);
}}

.pred-tag {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: var(--badge-bg);
    color: var(--badge-text);
    border: 1px solid var(--badge-border);
    padding: 0.18rem 0.5rem;
    border-radius: 9999px;
}}

.pred-content-middle {{
    margin: 1.1rem 0 0.7rem 0;
}}

.pred-label {{
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}}

.pred-price {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: var(--pred-val-color);
    margin-bottom: 0.4rem;
}}

.pred-price-placeholder {{
    color: var(--text-muted);
    font-size: 2.1rem;
    font-weight: 700;
}}

.pred-status-row {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.35rem;
}}

.pred-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
}}

.dot-idle {{
    background-color: var(--text-muted);
}}

.dot-success {{
    background-color: #10b981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
}}

.dot-error {{
    background-color: #ef4444;
}}

.pred-status-text {{
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--text-secondary);
}}

.pred-footer-note {{
    font-size: 0.7rem;
    color: var(--text-muted);
    border-top: 1px dashed var(--border-color);
    padding-top: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

/* Model Info Popover Button */
div[data-testid="stPopover"] > button {{
    height: 36px !important;
    min-height: 36px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
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
    min-width: 280px !important;
}}

.popover-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.55rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border-color);
}}

.popover-title {{
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-primary);
}}

.popover-badge {{
    font-size: 0.66rem;
    font-weight: 600;
    background: var(--badge-bg);
    color: var(--badge-text);
    padding: 0.1rem 0.38rem;
    border-radius: 4px;
}}

.popover-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}}

.popover-table th {{
    text-align: left;
    padding: 0.3rem 0.45rem;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border-color);
}}

.popover-table td {{
    padding: 0.4rem 0.45rem;
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

/* Responsive Media Queries for Mobile Screens */
@media (max-width: 860px) {{
    .block-container {{
        padding: 0.75rem 0.9rem !important;
    }}
    .hero-heading-container {{
        flex-direction: row;
    }}
    .prediction-card-box {{
        min-height: 210px;
    }}
}}
</style>
"""

st.markdown(css_rules, unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Header Section: Brand Title and Theme Toggle
# ---------------------------------------------------------

header_col1, header_col2 = st.columns([0.84, 0.16], vertical_alignment="center")

with header_col1:
    st.markdown(
        """
        <div class="brand-wrapper">
            <div class="brand-icon-box">🏠</div>
            <span class="brand-title">House Price Prediction</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_col2:
    toggle_icon = "☀️" if is_dark else "🌙"
    toggle_text = "Light" if is_dark else "Dark"
    if st.button(f"{toggle_icon}  {toggle_text}", key="theme_toggle_btn", use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

# Subtle Header Divider
st.markdown('<div style="border-bottom: 1px solid var(--border-color); margin-bottom: 0.3rem;"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. Main Heading Section with House Illustration
# ---------------------------------------------------------

house_svg = f"""
<svg class="hero-illustration-svg" viewBox="0 0 54 42" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M27 3L4 21H10V39H44V21H50L27 3Z" fill="{accent_light}" stroke="{accent_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <rect x="22" y="25" width="10" height="14" fill="{bg_card}" stroke="{accent_color}" stroke-width="1.8"/>
    <circle cx="24" cy="32" r="1" fill="{accent_color}"/>
    <rect x="13" y="21" width="6" height="7" rx="1" fill="{bg_card}" stroke="{border_color}" stroke-width="1.5"/>
    <rect x="35" y="21" width="6" height="7" rx="1" fill="{bg_card}" stroke="{border_color}" stroke-width="1.5"/>
    <path d="M38 7V12" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

st.markdown(
    f"""
    <div class="hero-heading-container">
        <div class="hero-text-group">
            <h1 class="hero-title">House Price Prediction</h1>
            <p class="hero-subtitle">Estimate house prices in California using a machine learning model.</p>
        </div>
        <div class="hero-illustration-box">
            {house_svg}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# 3. Two-Column Dashboard Layout
# ---------------------------------------------------------

col_left, col_right = st.columns([1.32, 0.88], gap="medium")


# ---------------------------------------------------------
# Left Column: House Details Input Form
# ---------------------------------------------------------

with col_left:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-header-bar">
                <div class="card-title-group">
                    <span class="card-header-icon">🏠</span>
                    <span class="card-header-title">House Details</span>
                </div>
                <span class="card-header-badge">9 Features</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 8 numerical features organized in 4 balanced rows of 2 columns
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            longitude = st.number_input(
                "Longitude",
                format="%.2f",
                value=float(st.session_state.val_longitude),
                key="num_longitude"
            )
            st.session_state.val_longitude = longitude

        with r1_c2:
            latitude = st.number_input(
                "Latitude",
                format="%.2f",
                value=float(st.session_state.val_latitude),
                key="num_latitude"
            )
            st.session_state.val_latitude = latitude

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            housing_median_age = st.number_input(
                "Housing Median Age (years)",
                min_value=1,
                step=1,
                value=int(st.session_state.val_housing_median_age),
                key="num_housing_median_age"
            )
            st.session_state.val_housing_median_age = housing_median_age

        with r2_c2:
            total_rooms = st.number_input(
                "Total Rooms",
                min_value=1,
                step=1,
                value=int(st.session_state.val_total_rooms),
                key="num_total_rooms"
            )
            st.session_state.val_total_rooms = total_rooms

        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            total_bedrooms = st.number_input(
                "Total Bedrooms",
                min_value=1,
                step=1,
                value=int(st.session_state.val_total_bedrooms),
                key="num_total_bedrooms"
            )
            st.session_state.val_total_bedrooms = total_bedrooms

        with r3_c2:
            population = st.number_input(
                "Population",
                min_value=0,
                step=1,
                value=int(st.session_state.val_population),
                key="num_population"
            )
            st.session_state.val_population = population

        r4_c1, r4_c2 = st.columns(2)
        with r4_c1:
            households = st.number_input(
                "Households",
                min_value=0,
                step=1,
                value=int(st.session_state.val_households),
                key="num_households"
            )
            st.session_state.val_households = households

        with r4_c2:
            median_income = st.number_input(
                "Median Income (in $10,000s)",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                value=float(st.session_state.val_median_income),
                key="num_median_income"
            )
            st.session_state.val_median_income = median_income

        # 9th Feature: Ocean Proximity
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
            key="sel_ocean_proximity"
        )
        st.session_state.val_ocean_proximity = ocean_proximity

        # Predict Button
        predict_clicked = st.button("Predict House Price", use_container_width=True, type="primary", key="btn_predict")


# ---------------------------------------------------------
# Prediction Logic (Exact Backend Logic Preserved)
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
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
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
        price_display = f"${price_val:,.2f}"
        price_class = "pred-price"
        dot_class = "dot-success"
        status_text = "Calculated using trained ML model"
    elif err_val is not None:
        price_display = "Error"
        price_class = "pred-price-placeholder"
        dot_class = "dot-error"
        status_text = f"Error: {err_val}"
    else:
        price_display = "--"
        price_class = "pred-price pred-price-placeholder"
        dot_class = "dot-idle"
        status_text = "Click 'Predict House Price' to calculate"

    # Prediction Result Card
    st.markdown(
        f"""
        <div class="prediction-card-box">
            <div class="prediction-card-top">
                <div class="pred-icon-circle">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                </div>
                <span class="pred-tag">Estimate</span>
            </div>
            <div class="pred-content-middle">
                <div class="pred-label">Predicted House Price</div>
                <div class="{price_class}">{price_display}</div>
                <div class="pred-status-row">
                    <span class="pred-dot {dot_class}"></span>
                    <span class="pred-status-text">{status_text}</span>
                </div>
            </div>
            <div class="pred-footer-note">
                <span>California Census Block Group</span>
                <span>USD ($)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Model Info Popover (Streamlit native popover)
    with st.popover("ℹ️  Model Info", use_container_width=True):
        st.markdown(
            """
            <div class="popover-header">
                <span class="popover-title">Model Specifications</span>
                <span class="popover-badge">Production</span>
            </div>
            <table class="popover-table">
                <thead>
                    <tr>
                        <th>Property</th>
                        <th style="text-align: right;">Information</th>
                    </tr>
                </thead>
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
            <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.5rem; text-align: center;">
                Standardized pipeline with imputer, scaler & encoder
            </div>
            """,
            unsafe_allow_html=True
        )
