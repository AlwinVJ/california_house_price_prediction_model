import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("house_price_model.joblib")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("California House Price Prediction")

st.write(
    "Enter the housing information below to predict "
    "the median house value."
)


# --------------------------------------------------
# Input fields
# --------------------------------------------------

# --------------------------------------------------
# Location
# --------------------------------------------------

st.subheader("Location")

longitude = st.number_input(
    "Longitude",
    min_value=-124.35,
    value=-124.35,
    step=0.01,
    format="%.2f"
)

latitude = st.number_input(
    "Latitude",
    min_value=32.54,
    value=32.54,
    step=0.01,
    format="%.2f"
)


# --------------------------------------------------
# Housing Information
# --------------------------------------------------

st.subheader("Housing Information")

housing_median_age = st.number_input(
    "Housing Median Age (years)",
    min_value=1,
    value=1,
    step=1
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1,
    value=1,
    step=1
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1,
    value=1,
    step=1
)


# --------------------------------------------------
# Population Information
# --------------------------------------------------

st.subheader("Population Information")

population = st.number_input(
    "Total number of people residing in the area",
    min_value=0,
    value=0,
    step=1
)

households = st.number_input(
    "Total number of people residing in the house",
    min_value=0,
    value=0,
    step=1
)


# --------------------------------------------------
# Income
# --------------------------------------------------

st.subheader("Income")

median_income = st.number_input(
    "Median Income (in $10,000s)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    format="%.2f"
)


# --------------------------------------------------
# Ocean Proximity
# --------------------------------------------------

st.subheader("Location Type")

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict House Price"):

    new_row = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    prediction = model.predict(new_row)[0]

    st.success(
        f"Predicted House Value: ${prediction * 100000:,.2f}"
    )