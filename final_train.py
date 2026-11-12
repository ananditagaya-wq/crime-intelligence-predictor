
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "state_ut",
    "district",
    "year",
    "ipc_lag_1",
    "ipc_lag_2",
    "historical_avg_2"
]

CATEGORICAL = ["state_ut", "district"]

NUMERICAL = [
    "year",
    "ipc_lag_1",
    "ipc_lag_2",
    "historical_avg_2"
]

TARGET = "total_ipc_crimes"

# Load complete training data (2003–2009)
data = pd.read_csv("data/train.csv")

X = data[FEATURES]

# Train on change from previous year
y = data[TARGET] - data["ipc_lag_1"]

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

# Selected residual Random Forest
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

# Train on all training records
model.fit(X, y)

# Save production model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.joblib")

print("\n========== FINAL MODEL TRAINED ==========")
print("Training records:", len(data))
print("Training years:", data["year"].min(), "-", data["year"].max())
print("Algorithm: Residual Random Forest")
print("Trees: 200")
print("Maximum depth: 15")
print("Saved: model/model.joblib")
