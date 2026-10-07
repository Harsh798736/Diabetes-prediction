import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load("diabetes.pkl")
scaler = joblib.load("Scaled.pkl")
expected_columns = joblib.load("Cols.pkl")

st.title("🩺 Diabetes Prediction System")
st.markdown("Provide the following details to check your diabetes risk:")

# User inputs
pregnancies = st.number_input(
    "Number of Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=250,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.slider(
    "Age",
    10,
    100,
    40
)

if st.button("Predict"):

    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    input_df = pd.DataFrame([input_data])

    # Ensure the same column order used during training
    input_df = input_df[expected_columns]

    # Scale using the scaler fitted on training data
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")