# Crime Intelligence Predictor

An ML-powered crime analytics and prediction platform that transforms historical Indian crime records into actionable insights through interactive visualizations, state-wise crime mapping, and machine learning-based predictions.

## 1. Project Overview

The Crime Intelligence Predictor is a full-stack machine learning application designed to analyze historical Indian IPC crime data and predict future crime counts using historical crime patterns.

The platform combines data preprocessing, machine learning, REST API development, authentication, and interactive visualization into a single web application.

## 2. Key Features

- **Interactive Dashboard:** View crime statistics through KPIs and analytical visualizations.
- **India Crime Map:** Explore state-wise crime intensity using an interactive choropleth-style map.
- **State-wise Analysis:** Examine crime statistics across Indian states and Union Territories.
- **Dynamic Year Filters:** Analyze crime trends across selected historical periods.
- **Crime Prediction:** Generate crime-count predictions using a machine learning model.
- **Model Evaluation:** Compare model performance against a baseline.
- **User Authentication:** Secure signup and login functionality.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Interactive Maps | Leaflet.js |
| Map Data | GeoJSON |
| Authentication | JWT-based authentication |
| Database | SQLite |
| Containerization | Docker |
| Deployment | Render |
| Version Control | Git and GitHub |

## 4. Dataset

The project uses historical district-wise IPC crime records for India.

- **Coverage:** 2001–2012
- **Original dataset:** 9,017 records
- **Processed records:** 9,016 district-year records after duplicate aggregation
- **Geographical level:** State/UT and district
- **Data processing:** Missing numerical crime values are handled during preprocessing.

The dataset is historical and should not be interpreted as representing current crime conditions.

## 5. Machine Learning Methodology

The prediction system uses a Random Forest-based residual learning approach.

Historical crime information is used to construct prediction inputs, including:

- Previous year's crime count.
- Crime count from two years earlier.
- Historical average crime count.

The model predicts a change relative to the previous year's crime count, which is then used to calculate the final predicted crime count.

### Model Performance

| Metric | Residual Random Forest | Baseline |
|---|---:|---:|
| MAE | 465.04 | 483.31 |
| RMSE | 1518.95 | 1707.79 |
| R² | 0.9948 | 0.9934 |

The residual Random Forest model showed improvements of approximately 3.8% in MAE and 11.1% in RMSE over the baseline on the reported test evaluation.

## 6. Application Architecture

The application follows a frontend-backend architecture.

1. The frontend collects user inputs and displays analytical results.
2. FastAPI handles authentication, dashboard requests, and prediction requests.
3. Pandas processes historical crime records.
4. The machine learning model generates predictions.
5. The dashboard displays analytical results and interactive geographical visualizations.

## 7. Installation and Local Setup

### Prerequisites

- Python 3.11
- Git
- Docker (optional)

### Clone the repository

```bash
git clone https://github.com/ananditagaya-wq/crime-intelligence-predictor.git
cd crime-intelligence-predictor
```

### Create a virtual environment

```bash
python -m venv venv
```

### Install dependencies

```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file containing the required application configuration, including a securely generated `SECRET_KEY`.

Never commit `.env` files or authentication secrets to GitHub.

### Run the application

```bash
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

## 8. Deployment

The application is containerized using Docker and deployed through Render.

Live Application:

https://crime-intelligence-predictor.onrender.com/

Source Code:

https://github.com/ananditagaya-wq/crime-intelligence-predictor

## 9. Project Limitations

- The dataset covers only 2001–2012.
- Historical IPC records do not represent present-day crime conditions.
- State boundaries in the map are used for visualization and may differ from historical administrative boundaries.
- The model's predictions are estimates based on historical patterns, not guaranteed future outcomes.
- Recorded crime statistics may not capture every crime incident.

## 10. Future Scope

- Integration of more recent crime datasets.
- District-level geographical analytics.
- Advanced temporal forecasting techniques.
- Additional model comparison and validation.
- Improved geographical and demographic analysis.
- Integration of additional data sources.

## 11. Author

**Anandita Gaya**

B.Tech – Computer Science and Engineering (Data Science)

GitHub: https://github.com/ananditagaya-wq

---

*Crime Intelligence Predictor — Turning historical crime data into meaningful intelligence.*
