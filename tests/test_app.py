
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# Test 1: Health endpoint
def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


# Test 2: Valid crime prediction
def test_valid_prediction():

    payload = {
        "state_ut": "A & N ISLANDS",
        "district": "ANDAMAN",
        "year": 2010,
        "ipc_lag_1": 921,
        "ipc_lag_2": 859,
        "historical_avg_2": 890
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_total_ipc_crimes" in result
    assert result["predicted_total_ipc_crimes"] >= 0


# Test 3: Invalid year
def test_invalid_year():

    payload = {
        "state_ut": "A & N ISLANDS",
        "district": "ANDAMAN",
        "year": 1800,
        "ipc_lag_1": 921,
        "ipc_lag_2": 859,
        "historical_avg_2": 890
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# Test 4: Negative crime count
def test_negative_crime_count():

    payload = {
        "state_ut": "A & N ISLANDS",
        "district": "ANDAMAN",
        "year": 2010,
        "ipc_lag_1": -100,
        "ipc_lag_2": 859,
        "historical_avg_2": 890
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
