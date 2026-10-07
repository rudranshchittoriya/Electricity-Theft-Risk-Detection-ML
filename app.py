import streamlit as st
import sys
import os
import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"

st.set_page_config(
    page_title="Electricity Theft Risk Detector",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Electricity Theft Risk Detector")
st.caption("Machine Learning based academic risk-classification system")

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Run src/train.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Customer Consumption Profile")

connection = st.selectbox("Connection Type", ["Residential", "Commercial", "Industrial"])
season = st.selectbox("Seasonal Factor", ["Summer", "Winter", "Monsoon", "Spring", "Autumn"])

monthly = st.number_input("Monthly Consumption", min_value=0.0, value=150.0)
previous = st.number_input("Previous Month Consumption", min_value=0.0, value=180.0)
avg6 = st.number_input("Average Consumption (6M)", min_value=0.0, value=175.0)
max6 = st.number_input("Maximum Consumption (6M)", min_value=0.0, value=230.0)
min6 = st.number_input("Minimum Consumption (6M)", min_value=0.0, value=120.0)
voltage = st.number_input("Voltage Fluctuations", min_value=0.0, value=2.0)
ratio = st.number_input("Consumption Ratio", min_value=0.0, value=0.85)

if st.button("🔍 Analyze Customer", use_container_width=True):
    row = pd.DataFrame([{
        "Connection Type": connection,
        "Monthly Consumption": monthly,
        "Previous Month Consumption": previous,
        "Avg Consumption 6M": avg6,
        "Max Consumption 6M": max6,
        "Min Consumption 6M": min6,
        "Voltage Fluctuations": voltage,
        "Seasonal Factor": season,
        "Consumption Ratio": ratio
    }])

    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.50)

    if probability >= 0.75:
        risk = "🔴 HIGH RISK"
    elif probability >= 0.50:
        risk = "🟠 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    st.metric("Theft Probability", f"{probability*100:.1f}%")
    st.success(risk)

    if prediction:
        st.warning("The model detected a suspicious consumption pattern.")
    else:
        st.info("The model did not detect a strong suspicious pattern.")

st.divider()
st.caption("A model prediction is not proof of theft. Human verification is required.")
