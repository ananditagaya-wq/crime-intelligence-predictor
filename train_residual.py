
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

FEATURES = [
    "state_ut",
    "district",
    "year",
    "ipc_lag_1",
    "ipc_lag_2",
    "historical_avg_2"
]

TARGET = "total_ipc_crimes"

CATEGORICAL = ["state_ut", "district"]

NUMERICAL = [
    "year",
    "ipc_lag_1",
    "ipc_lag_2",
    "historical_avg_2"
]

# Load data
data = pd.read_csv("data/train.csv")

# Chronological split
fit_data = data[data["year"] <= 2007].copy()
val_data = data[data["year"] >= 2008].copy()

X_fit = fit_data[FEATURES]
X_val = val_data[FEATURES]

# Learn the change from previous year
y_fit = fit_data[TARGET] - fit_data["ipc_lag_1"]

y_val = val_data[TARGET]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL
        ),
        ("numerical", "passthrough", NUMERICAL)
    ]
)

# Random Forest
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train
model.fit(X_fit, y_fit)
os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/residual_model.joblib"
)

print("Residual model saved successfully!")

# Predict change and reconstruct total
predicted_change = model.predict(X_val)

predictions = (
    val_data["ipc_lag_1"].to_numpy()
    + predicted_change
)

# Crime counts cannot be negative
predictions = predictions.clip(min=0)

# Baseline predictions
baseline = val_data["ipc_lag_1"].to_numpy()

# Evaluation
mae = mean_absolute_error(y_val, predictions)

rmse = mean_squared_error(
    y_val, predictions
) ** 0.5

r2 = r2_score(y_val, predictions)

baseline_mae = mean_absolute_error(y_val, baseline)

baseline_rmse = mean_squared_error(
    y_val, baseline
) ** 0.5

print("\n========== RESIDUAL MODEL RESULTS ==========")
print(f"Residual Model MAE: {mae:.2f}")
print(f"Residual Model RMSE: {rmse:.2f}")
print(f"Residual Model R2: {r2:.4f}")

print("\n========== BASELINE COMPARISON ==========")
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")

print("\nMAE improvement over baseline:",
      round(baseline_mae - mae, 2))
