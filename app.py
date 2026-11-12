
import os
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict

# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

MODEL_PATH = os.path.join("model", "model.joblib")

model = joblib.load(MODEL_PATH)

# =====================================
# 2. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Crime Intelligence Predictor",
    description="Predicts total IPC crimes using historical crime data.",
    version="1.0.0"
)
# Serve frontend homepage
@app.get("/")
def home():
    return FileResponse("static/index.html")

# =====================================
# 3. INPUT DATA VALIDATION
# =====================================

class CrimeInput(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    state_ut: str = Field(
        min_length=2,
        max_length=100
    )

    district: str = Field(
        min_length=2,
        max_length=100
    )

    year: int = Field(
        ge=2003,
        le=2100
    )

    ipc_lag_1: float = Field(
        ge=0,
        le=10000000
    )

    ipc_lag_2: float = Field(
        ge=0,
        le=10000000
    )

    historical_avg_2: float = Field(
        ge=0,
        le=10000000
    )


# =====================================
# 4. HEALTH CHECK ENDPOINT
# =====================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# =====================================
# 5. PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
def predict_crime(data: CrimeInput):

    input_data = pd.DataFrame([{
        "state_ut": data.state_ut,
        "district": data.district,
        "year": data.year,
        "ipc_lag_1": data.ipc_lag_1,
        "ipc_lag_2": data.ipc_lag_2,
        "historical_avg_2": data.historical_avg_2
    }])

    predicted_change = model.predict(input_data)[0]

    predicted_total = max(
        0,
        data.ipc_lag_1 + predicted_change
    )

    return {
        "state_ut": data.state_ut,
        "district": data.district,
        "year": data.year,
        "predicted_total_ipc_crimes": round(
            float(predicted_total), 2
        )
    }
