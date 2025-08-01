import joblib
import pandas as pd

# === Load the saved model and scaler ===
model_bundle = joblib.load("models/fuel_efficiency_model.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
model_columns = scaler.feature_names_in_

# === New input data to predict ===
new_data = pd.DataFrame([{
    "car_model": "DriveX",
    "transmission": "Manual",
    "engine_size_cc": 1600,
    "vehicle_weight_kg": 1300,
    "num_cylinders": 4,
    "turbocharged": "Yes",
    "avg_speed_kmph": 60,
    "city_drive_ratio": 0.5,
    "tire_type": "Radial",
    "fuel_type": "Petrol"
}])

# === Encode the categorical features just like training ===
new_data_encoded = pd.get_dummies(new_data)

# === Align with training data columns (fill missing with 0) ===
for col in model_columns:
    if col not in new_data_encoded.columns:
        new_data_encoded[col] = 0

# === Reorder columns to match model training ===
new_data_encoded = new_data_encoded[model_columns]

# === Scale the input ===
X_scaled = scaler.transform(new_data_encoded)

# === Predict fuel efficiency ===
predicted_kmpl = model.predict(X_scaled)
print(f"Estimated Fuel Efficiency: {predicted_kmpl[0]:.2f} kmpl")
