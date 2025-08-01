import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# === Configuration ===
DATA_PATH = r"D:\bhargav\internship project\ybi\Fuel Efficiency Estimator for DriveGo Motors.csv"
MODEL_PATH = "models/fuel_efficiency_model.pkl"
TARGET_COLUMN = "estimated_fuel_efficiency_kmpl"

# === Load Data ===
df = pd.read_csv(DATA_PATH)
print(f"Original Data Shape: {df.shape}")
df.dropna(inplace=True)
print(f"After Dropping NA: {df.shape}")
print(f"Available columns: {df.columns.tolist()}")

# === Check for Target Column ===
if TARGET_COLUMN not in df.columns:
    raise Exception(f"Missing target column '{TARGET_COLUMN}' in dataset!")

# === Split Features & Target ===
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# === Encode Categorical Features ===
X = pd.get_dummies(X)

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train Model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# === Save Model & Scaler ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
