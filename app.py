# for running streamlit run app.py
# Local URL: http://localhost:8501
# Network URL: http://10.119.75.119:8501

import streamlit as st
import pandas as pd
import joblib

# === Load Model and Scaler ===
model_bundle = joblib.load("models/fuel_efficiency_model.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
model_columns = scaler.feature_names_in_

# === Streamlit UI ===
st.set_page_config(page_title="Fuel Efficiency Estimator", layout="centered")
st.title("🚗 Fuel Efficiency Estimator for DriveGo Motors")
st.write("Estimate the fuel efficiency (in kmpl) based on vehicle specifications.")

# === Input Fields ===
car_model = st.text_input("Car Model", "DriveX")
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
engine_size_cc = st.number_input("Engine Size (cc)", min_value=500, max_value=5000, value=1600, step=100)
vehicle_weight_kg = st.number_input("Vehicle Weight (kg)", min_value=600, max_value=5000, value=1300, step=50)
num_cylinders = st.selectbox("Number of Cylinders", [3, 4, 6, 8, 12])
turbocharged = st.selectbox("Turbocharged", ["Yes", "No"])
avg_speed_kmph = st.slider("Average Speed (kmph)", 20, 120, 60)
city_drive_ratio = st.slider("City Drive Ratio (0 = All Highway, 1 = All City)", 0.0, 1.0, 0.5)
tire_type = st.selectbox("Tire Type", ["Radial", "Bias"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])

# === Prepare Data ===
if st.button("Estimate Fuel Efficiency"):
    input_data = pd.DataFrame([{
        "car_model": car_model,
        "transmission": transmission,
        "engine_size_cc": engine_size_cc,
        "vehicle_weight_kg": vehicle_weight_kg,
        "num_cylinders": num_cylinders,
        "turbocharged": turbocharged,
        "avg_speed_kmph": avg_speed_kmph,
        "city_drive_ratio": city_drive_ratio,
        "tire_type": tire_type,
        "fuel_type": fuel_type
    }])

    # Encode categorical features
    input_encoded = pd.get_dummies(input_data)

    # Add missing columns (from training) with 0
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match model
    input_encoded = input_encoded[model_columns]

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Predict
    predicted_kmpl = model.predict(input_scaled)[0]
    st.success(f"✅ Estimated Fuel Efficiency: **{predicted_kmpl:.2f} kmpl**")

    st.markdown("---")
    st.info("Model trained with Linear Regression on DriveGo Motors data.")
