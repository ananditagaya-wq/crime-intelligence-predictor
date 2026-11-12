
import pandas as pd
import joblib

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

# Load test dataset
test = pd.read_csv("data/test.csv")

X_test = test[FEATURES]
y_test = test["total_ipc_crimes"]

# Load models
original_model = joblib.load(
    "model/random_forest_200_15.joblib"
)

residual_model = joblib.load(
    "model/residual_model.joblib"
)

# Generate predictions
baseline_predictions = test["ipc_lag_1"].to_numpy()

original_predictions = original_model.predict(X_test)

predicted_change = residual_model.predict(X_test)

residual_predictions = (
    test["ipc_lag_1"].to_numpy() + predicted_change
).clip(min=0)


# Evaluation function
def evaluate(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual, predicted
    ) ** 0.5

    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")


print("\n========== FINAL TEST RESULTS ==========")

print("Test records:", len(test))
print("Test years:", sorted(test["year"].unique()))

evaluate(
    "Previous-Year Baseline",
    y_test,
    baseline_predictions
)

evaluate(
    "Original Random Forest",
    y_test,
    original_predictions
)

evaluate(
    "Residual Random Forest",
    y_test,
    residual_predictions
)
