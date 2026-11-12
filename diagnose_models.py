
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load validation data
data = pd.read_csv("data/train.csv")
val = data[data["year"] >= 2008].copy()

FEATURES = [
    "state_ut",
    "district",
    "year",
    "ipc_lag_1",
    "ipc_lag_2",
    "historical_avg_2"
]

# Load the best current Random Forest
model = joblib.load("model/random_forest_200_15.joblib")

# Predictions
val["baseline_prediction"] = val["ipc_lag_1"]
val["rf_prediction"] = model.predict(val[FEATURES])

# Calculate errors
val["baseline_error"] = (
    val["total_ipc_crimes"] - val["baseline_prediction"]
).abs()

val["rf_error"] = (
    val["total_ipc_crimes"] - val["rf_prediction"]
).abs()

# Compare performance year by year
print("\n========== YEAR-WISE MODEL COMPARISON ==========")

for year, group in val.groupby("year"):

    baseline_mae = mean_absolute_error(
        group["total_ipc_crimes"],
        group["baseline_prediction"]
    )

    rf_mae = mean_absolute_error(
        group["total_ipc_crimes"],
        group["rf_prediction"]
    )

    print(f"\nYear: {year}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Random Forest MAE: {rf_mae:.2f}")

print("\n========== OVERALL COMPARISON ==========")
print("Baseline MAE:", round(val["baseline_error"].mean(), 2))
print("Random Forest MAE:", round(val["rf_error"].mean(), 2))
