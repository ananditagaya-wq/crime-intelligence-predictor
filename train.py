
import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

FEATURES = [
    "state_ut",
    "district",
    "year",
    "ipc_lag_1",
    "ipc_lag_2",
    "historical_avg_2"
]

TARGET = "total_ipc_crimes"

CATEGORICAL_FEATURES = [
    "state_ut",
    "district"
]

NUMERICAL_FEATURES = [
    "year",
    "ipc_lag_1",
    "ipc_lag_2",
    "historical_avg_2"
]


# ==========================================
# 2. COMMAND LINE ARGUMENTS
# ==========================================

parser = argparse.ArgumentParser(
    description="Crime Intelligence ML Training"
)

parser.add_argument(
    "--model",
    choices=["random_forest", "gradient_boosting"],
    default="random_forest"
)

parser.add_argument(
    "--n_estimators",
    type=int,
    default=100
)

parser.add_argument(
    "--max_depth",
    type=int,
    default=10
)

args = parser.parse_args()


# ==========================================
# 3. LOAD DATA
# ==========================================

train_data = pd.read_csv("data/train.csv")

print("\n========== MODEL TRAINING ==========")

print("Dataset shape:", train_data.shape)
print("Selected model:", args.model)


# ==========================================
# 4. CHRONOLOGICAL VALIDATION SPLIT
# ==========================================

fit_data = train_data[
    train_data["year"] <= 2007
].copy()

validation_data = train_data[
    train_data["year"] >= 2008
].copy()

X_fit = fit_data[FEATURES]
y_fit = fit_data[TARGET]

X_val = validation_data[FEATURES]
y_val = validation_data[TARGET]

print("\nTraining records:", len(X_fit))
print("Validation records:", len(X_val))


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            "passthrough",
            NUMERICAL_FEATURES
        )
    ]
)


# ==========================================
# 6. SELECT ALGORITHM
# ==========================================

if args.model == "random_forest":

    regressor = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )

else:

    regressor = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )


# ==========================================
# 7. CREATE PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ]
)


# ==========================================
# 8. EXPERIMENT TRACKING WITH MLFLOW
# ==========================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Crime-Intelligence-Predictor"
)

with mlflow.start_run():

    # Train
    model.fit(X_fit, y_fit)

    # Validate
    predictions = model.predict(X_val)

    mae = mean_absolute_error(
        y_val, predictions
    )

    rmse = mean_squared_error(
        y_val, predictions
    ) ** 0.5

    r2 = r2_score(
        y_val, predictions
    )

    # Log parameters
    mlflow.log_param("algorithm", args.model)
    mlflow.log_param(
        "n_estimators", args.n_estimators
    )
    mlflow.log_param(
        "max_depth", args.max_depth
    )
    mlflow.log_param(
        "validation_start_year", 2008
    )

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    # Save model
    os.makedirs("model", exist_ok=True)

    model_path = (
        f"model/{args.model}_"
        f"{args.n_estimators}_"
        f"{args.max_depth}.joblib"
    )

    joblib.dump(model, model_path)

    print("\n========== VALIDATION RESULTS ==========")
    print("Algorithm:", args.model)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    print("\nModel saved:", model_path)

print("\nMLflow run completed successfully!")
