
import os
import joblib
import pandas as pd

from auth.routes import router as auth_router
from auth.database import create_users_table
from fastapi import FastAPI, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from auth.routes import get_current_user

# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

MODEL_PATH = os.path.join("model", "model.joblib")

model = joblib.load(MODEL_PATH)

# =====================================
# LOAD HISTORICAL CRIME DATA
# =====================================

DATA_PATH = os.path.join("data", "crime_dataset.csv")

crime_data = pd.read_csv(DATA_PATH)

crime_data.columns = crime_data.columns.str.strip()

# Identify crime-count columns
KEY_COLUMNS = ["state_ut", "district", "year"]

CRIME_COLUMNS = [
    col for col in crime_data.columns
    if col not in KEY_COLUMNS
]

# Convert crime values to numeric
for col in CRIME_COLUMNS:
    crime_data[col] = pd.to_numeric(
        crime_data[col], errors="coerce"
    ).fillna(0)

# Aggregate duplicate district-year records
crime_data = (
    crime_data
    .groupby(KEY_COLUMNS, as_index=False)[CRIME_COLUMNS]
    .sum()
)

print("Crime dataset loaded:", crime_data.shape)


# =====================================
# 2. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Crime Intelligence Predictor",
    description="Predicts total IPC crimes using historical crime data.",
    version="1.0.0"
)
app.include_router(auth_router)


@app.on_event("startup")
def initialize_database():
    create_users_table()
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
# 5. OVERVIEW DASHBOARD API
# =====================================

@app.get("/api/overview")
def get_overview(
    state_ut: str = Query(default="All"),
    district: str = Query(default="All"),
    start_year: int = Query(default=2001, ge=2001, le=2012),
    end_year: int = Query(default=2012, ge=2001, le=2012),
    current_user: dict = Depends(get_current_user)
):
    if start_year > end_year:
        return {"error": "Start year cannot exceed end year."}

    # Apply filters
    filtered = crime_data[
        (crime_data["year"] >= start_year) &
        (crime_data["year"] <= end_year)
    ].copy()

    if state_ut != "All":
        filtered = filtered[
            filtered["state_ut"].str.casefold() == state_ut.casefold()
        ]

    if district != "All":
        filtered = filtered[
            filtered["district"].str.casefold() == district.casefold()
        ]

    if filtered.empty:
        return {"message": "No records found for the selected filters."}

    # KPI calculations
    total_crimes = int(filtered["total_ipc_crimes"].sum())
    annual_average = round(
        total_crimes / filtered["year"].nunique(), 2
    )

    total_districts = int(filtered["district"].nunique())

    # Yearly crime trend
    trend = (
        filtered.groupby("year")["total_ipc_crimes"]
        .sum()
        .reset_index()
        .to_dict(orient="records")
    )

    # Top states: based on year selection, independent
    # of the selected state/district filter
    national_data = crime_data[
        (crime_data["year"] >= start_year) &
        (crime_data["year"] <= end_year)
    ]

    top_states = (
        national_data.groupby("state_ut")["total_ipc_crimes"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
        .to_dict(orient="records")
    )

    # Crime category counts
    category_totals = (
        filtered[CRIME_COLUMNS]
        .drop(columns=["total_ipc_crimes"])
        .sum()
        .sort_values(ascending=False)
    )

    categories = [
        {"category": name, "count": int(value)}
        for name, value in category_totals.items()
    ]

    top_categories = categories[:6]

    # Automatically generated insights
    peak_year = max(trend, key=lambda x: x["total_ipc_crimes"])
    leading_state = top_states[0]

    insights = [
        f"Highest recorded crime count occurred in {peak_year['year']}.",
        f"{leading_state['state_ut']} recorded the highest total among states in the selected period.",
        f"The selected filters contain {total_districts} distinct districts.",
        f"{top_categories[0]['category'].replace('_', ' ').title()} has the highest recorded category count."
    ]

    return {
        "filters": {
            "state_ut": state_ut,
            "district": district,
            "start_year": start_year,
            "end_year": end_year
        },
        "kpis": {
            "total_ipc_crimes": total_crimes,
            "average_annual_crimes": annual_average,
            "districts_covered": total_districts,
            "crime_categories": len(categories)
        },
        "trend": trend,
        "top_states": top_states,
        "categories": top_categories,
        "insights": insights,
        "data_note": (
            "Crime category counts are reported separately. "
            "Their sum may not equal total IPC crimes."
        )
    }


# =====================================
# 5. PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
def predict_crime(
    data: CrimeInput,
    current_user: dict = Depends(get_current_user)
):

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
