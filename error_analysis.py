
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error

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

# Load Random Forest
model = joblib.load("model/random_forest_200_15.joblib")

# Generate predictions
val["rf_prediction"] = model.predict(val[FEATURES])
val["baseline_prediction"] = val["ipc_lag_1"]

# Calculate signed errors
val["rf_error"] = (
    val["rf_prediction"] - val["total_ipc_crimes"]
)

val["baseline_error"] = (
    val["baseline_prediction"] - val["total_ipc_crimes"]
)

# Divide observations into crime-count quartiles
val["crime_group"] = pd.qcut(
    val["total_ipc_crimes"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"],
    duplicates="drop"
)

print("\n========== ERROR ANALYSIS ==========")

for group, records in val.groupby("crime_group", observed=True):

    print(f"\nCrime group: {group}")
    print("Records:", len(records))

    print(
        "Baseline MAE:",
        round(records["baseline_error"].abs().mean(), 2)
    )

    print(
        "Random Forest MAE:",
        round(records["rf_error"].abs().mean(), 2)
    )

    print(
        "RF average signed error:",
        round(records["rf_error"].mean(), 2)
    )

print("\n========== PREDICTION BIAS ==========")

print(
    "Actual average:",
    round(val["total_ipc_crimes"].mean(), 2)
)

print(
    "RF prediction average:",
    round(val["rf_prediction"].mean(), 2)
)

print(
    "Baseline prediction average:",
    round(val["baseline_prediction"].mean(), 2)
)
