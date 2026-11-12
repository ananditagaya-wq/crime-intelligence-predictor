
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load the same training dataset
data = pd.read_csv("data/train.csv")

# Use exactly the same validation period as train.py
validation_data = data[
    data["year"] >= 2008
].copy()

# Actual values
y_actual = validation_data["total_ipc_crimes"]

# Baseline: predict using previous year's crime count
baseline_predictions = validation_data["ipc_lag_1"]

# Calculate evaluation metrics
mae = mean_absolute_error(
    y_actual, baseline_predictions
)

rmse = mean_squared_error(
    y_actual, baseline_predictions
) ** 0.5

r2 = r2_score(
    y_actual, baseline_predictions
)

print("\n========== BASELINE RESULTS ==========")
print("Method: Previous Year Crime Count")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")
print("Validation records:", len(validation_data))
