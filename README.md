California House Price Prediction

A practical end-to-end machine learning project for predicting median
house values in California using demographic, housing, geographic, and
ocean-proximity features.

The project covers the complete regression workflow:

EDA → Data Preprocessing → Baseline Model → Cross-Validation → Model
Selection → Hyperparameter Tuning → Final Evaluation → Inference →
Streamlit Deployment

Project Overview

The model predicts the median_house_value of a California census block
group from features such as location, housing characteristics,
population, household information, income, and proximity to the ocean.

This project was built to practice a production-oriented machine
learning workflow rather than only training a single regression model.

Key objectives

Understand and explore a real-world regression dataset

Handle missing numerical and categorical data

Build reusable preprocessing pipelines

Establish a Linear Regression baseline

Compare multiple regression algorithms using cross-validation

Select the best model using RMSE

Tune the selected model with GridSearchCV

Evaluate the final model on an untouched test set

Save the trained pipeline for inference

Build a Streamlit interface for interactive predictions

Dataset

The project uses the California Housing Prices dataset.

Dataset source: California Housing Prices on
Kaggle

The dataset contains 20,640 observations and includes numerical and
categorical features.

Features

Feature                             Description

longitude                         Geographic longitude

latitude                          Geographic latitude

housing_median_age                Median age of houses in the block
group

total_rooms                       Total number of rooms in the block
group

total_bedrooms                    Total number of bedrooms in the
block group

population                        Total population in the block group

households                        Total number of households in the
block group

median_income                     Median household income, measured
in tens of thousands of USD

Target

median_house_value --- median house value for the block group,
measured in USD.

Important: A row represents a California census block group rather
than one individual house. Therefore, variables such as population,
households, total_rooms, and total_bedrooms describe the block
group.

Machine Learning Workflow

1. Exploratory Data Analysis

The notebook investigates:

Dataset shape and data types

Numerical and categorical features

Missing values

Duplicate rows

Descriptive statistics

Feature distributions

Outliers

Target distribution

Correlations between numerical features

Correlation of numerical features with the target

Important observations include:

ocean_proximity is a categorical feature.

total_bedrooms contains missing values.

median_house_value is right-skewed and capped at an upper value.

Several numerical features contain skew and outliers.

median_income has a strong relationship with the target.

Several housing/population variables exhibit multicollinearity.

2. Train/Test Split

The data is divided into:

80% training data

20% test data

A fixed random state of 42 is used for reproducibility.

The test set is kept separate until final evaluation.

3. Preprocessing

The project uses a ColumnTransformer with separate preprocessing
pipelines.

Numerical features

Median imputation using SimpleImputer

Standardization using StandardScaler

Categorical features

Most-frequent imputation using SimpleImputer

One-hot encoding using OneHotEncoder(handle_unknown="ignore")

The preprocessing and model are combined into a single Scikit-learn
Pipeline.

This helps keep preprocessing consistent during training,
cross-validation, tuning, and inference while reducing the risk of data
leakage.

Models Compared

The following regression models are evaluated using 5-fold
cross-validation:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

HistGradientBoostingRegressor

Model Selection

The primary model-selection metric is:

RMSE --- Root Mean Squared Error

Secondary metrics:

MAE --- Mean Absolute Error

R² --- Coefficient of Determination

Cross-validation is performed using shuffled 5-fold KFold validation
with random_state=42.

The best-performing model based on cross-validation RMSE was:

HistGradientBoostingRegressor

Hyperparameter Tuning

GridSearchCV is used to tune the HistGradientBoostingRegressor.

The search includes:

learning_rate

max_depth

max_leaf_nodes

min_samples_leaf

l2_regularization

The final selected configuration used in the notebook is:

learning_rate = 0.1
max_depth = None
max_leaf_nodes = 63
min_samples_leaf = 20
l2_regularization = 0.1

The final model is retrained on the complete training split using these
parameters.

Evaluation

The final model is evaluated on both the training and test sets using:

RMSE
MAE
R²

Residual analysis is also performed using:

Residuals vs. predictions plot

Residual distribution plot

The notebook intentionally evaluates the final model on the held-out
test set only after model selection and hyperparameter tuning.

Model Artifact

After training the final pipeline, save it using:

import joblib

joblib.dump(hgb_best, "house_price_model.joblib")

The saved artifact contains the preprocessing pipeline and trained
regression model, allowing the application to make predictions without
retraining the model.

Streamlit Application

The project includes a Streamlit frontend that loads the saved model and
accepts housing information from the user.

The application follows this flow:

User Input
    ↓
Streamlit UI
    ↓
Pandas DataFrame
    ↓
Saved Scikit-learn Pipeline
    ↓
Preprocessing
    ↓
HistGradientBoostingRegressor
    ↓
Predicted Median House Value

Run the application

Activate the project environment:

conda activate cali_house

Install dependencies:

pip install -r requirements.txt

Start Streamlit:

python -m streamlit run app.py

The application will open in your browser.

Project Structure

cali_house_price_prediction/
│
├── app.py
├── house_price_prediction.ipynb
├── housing.csv
├── house_price_model.joblib
├── requirements.txt
└── README.md

File descriptions

File                                Purpose

house_price_prediction.ipynb      Complete EDA, preprocessing,
training, evaluation, tuning, and
inference workflow

housing.csv                       California housing dataset

house_price_model.joblib          Saved trained preprocessing + model
pipeline

app.py                            Streamlit prediction interface

requirements.txt                  Python dependencies

README.md                         Project documentation

Installation

1. Clone the repository

git clone <your-repository-url>
cd cali_house_price_prediction

2. Create the Conda environment

conda create -n cali_house python=3.12
conda activate cali_house

3. Install dependencies

pip install -r requirements.txt

A typical requirements.txt for the project is:

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

Running the Notebook

Launch Jupyter:

jupyter notebook

Then open:

house_price_prediction.ipynb

Run the notebook from top to bottom to reproduce the analysis and
training workflow.

After training the final model, generate:

house_price_model.joblib

before running the Streamlit application.

Example Prediction

The notebook includes an inference example using values such as:

example_pred = predict_house_price(
    model=hgb_best,
    longitude=-122.230,
    latitude=37.880,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)

The same input structure is used by the Streamlit application.

Technologies Used

Python

Pandas --- data manipulation

NumPy --- numerical computing

Matplotlib --- visualization

Seaborn --- statistical visualization

Scikit-learn --- preprocessing, model training,
cross-validation, tuning, and evaluation

Joblib --- model serialization

Streamlit --- interactive web application

Jupyter Notebook --- experimentation and analysis

Key Machine Learning Concepts Practiced

This project provides practical exposure to:

Regression problems

Exploratory Data Analysis

Feature/target separation

Train/test splitting

Missing-value imputation

Categorical encoding

Feature scaling

Scikit-learn pipelines

Column transformers

Linear Regression

Ridge and Lasso Regression

Random Forest Regression

Gradient boosting

Cross-validation

Model comparison

Hyperparameter tuning

Grid Search

RMSE, MAE, and R²

Residual analysis

Model serialization

Model inference

Streamlit deployment

Limitations

This project is intended as a machine learning learning/deployment
project and should not be treated as a production-grade real-estate
valuation system.

Potential limitations include:

The dataset represents historical census block-group information.

The target variable is capped at an upper value in the original
dataset.

Predictions depend heavily on the quality and distribution of the
input features.

The model may perform poorly on data that differs substantially from
the training distribution.

Geographic and economic conditions can change over time.

Future Improvements

Possible extensions include:

Add feature engineering such as rooms-per-household and
bedrooms-per-room

Compare additional boosting models

Add automated model/version tracking

Add prediction confidence or uncertainty estimates

Improve Streamlit UI and validation

Add interactive EDA visualizations

Add unit tests

Add CI/CD with GitHub Actions

Containerize the application with Docker

Deploy the Streamlit application publicly

Add monitoring for data drift and model performance

Author

Alwin V J

This project was created as a hands-on machine learning project to
understand the complete lifecycle of a regression model, from raw data
exploration through model training and interactive inference.
