import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.joblib")


model = load_model()


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #0e1117;
    }

    /* Reduce top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-description {
        font-size: 1.05rem;
        color: #b8c1d1;
        margin-bottom: 1.5rem;
    }

    /* Section cards */
    .section-card {
        background: linear-gradient(
            145deg,
            #111a27,
            #131e2d
        );

        border: 1px solid #26364a;
        border-radius: 14px;

        padding: 1.25rem 1.4rem 1rem 1.4rem;

        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-bottom: 0.15rem;
    }

    .section-description {
        color: #9eabc0;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    /* Input labels */
    label {
        font-weight: 500 !important;
    }

    /* Number inputs */
    div[data-testid="stNumberInput"] input {
        border-radius: 9px;
    }

    /* Select box */
    div[data-baseweb="select"] {
        border-radius: 9px;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        height: 3.2rem;

        border-radius: 10px;
        border: none;

        background: linear-gradient(
            90deg,
            #1677ff,
            #287cff
        );

        color: white;

        font-size: 1.1rem;
        font-weight: 650;

        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    div.stButton > button:hover {
        border: none;
        color: white;
    }

    /* Prediction result */
    .prediction-card {
        background: linear-gradient(
            145deg,
            #0c2923,
            #102e27
        );

        border: 1px solid #195848;
        border-radius: 14px;

        padding: 1.4rem;

        margin-top: 1rem;
    }

    .prediction-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #7ff0c3;
    }

    .prediction-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;

        margin-top: 0.5rem;
    }

    /* Mobile adjustments */
    @media (max-width: 768px) {

        .main-title {
            font-size: 1.8rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .section-card {
            padding: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🏠 California House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">'
    'Enter the housing information below to predict the median house value in California.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Location
# ---------------------------------------------------------

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Location</div>
        <div class="section-description">
            Geographic coordinates of the area
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    longitude = st.number_input(
        "Longitude",
        value=-124.35,
        format="%.2f"
    )

with col2:

    latitude = st.number_input(
        "Latitude",
        value=32.54,
        format="%.2f"
    )


# ---------------------------------------------------------
# Housing Information
# ---------------------------------------------------------

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Housing Information</div>
        <div class="section-description">
            Details about the houses in the area
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    housing_median_age = st.number_input(
        "Housing Median Age (years)",
        min_value=1,
        value=1,
        step=1
    )

with col2:

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        value=1,
        step=1
    )

with col3:

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        value=1,
        step=1
    )


# ---------------------------------------------------------
# Population Information
# ---------------------------------------------------------

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Population Information</div>
        <div class="section-description">
            Population and household details
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    population = st.number_input(
        "Population",
        min_value=0,
        value=0,
        step=1
    )

with col2:

    households = st.number_input(
        "Households",
        min_value=0,
        value=0,
        step=1
    )

with col3:

    median_income = st.number_input(
        "Median Income (in $10,000s)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        format="%.2f"
    )


# ---------------------------------------------------------
# Ocean Proximity
# ---------------------------------------------------------

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Ocean Proximity</div>
        <div class="section-description">
            Distance to the nearest ocean
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "Select the proximity level from the drop down list",
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict House Price", use_container_width=True):

    # Handle the "None" option
    if ocean_proximity == "None":
        ocean_value = None
    else:
        ocean_value = ocean_proximity

    # Create input DataFrame
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

        # Make prediction
        prediction = model.predict(new_row)[0]

        # Target is expressed in units of $100,000
        predicted_price = prediction * 100000

        # Prediction result
        st.success("Prediction completed successfully!")

        st.subheader("📈 Prediction Result")

        st.metric(
            label="Predicted Median House Value",
            value=f"${predicted_price:,.2f}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )