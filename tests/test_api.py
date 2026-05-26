from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    payload = {
        "make": "Toyota",
        "model": "Corolla",
        "year": 2020,
        "style": "Sedan",
        "distance": 50000,
        "engine_capacity": 1.6,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "predicted_price" in data