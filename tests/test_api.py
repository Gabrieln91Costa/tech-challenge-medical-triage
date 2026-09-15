import sys
from pathlib import Path

from fastapi.testclient import TestClient


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": "Patient with cardiovascular disease and severe chest pain"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "class_name" in data
    assert "latency_ms" in data

    assert data["prediction"] in [1, 2, 3, 4, 5]
    assert isinstance(data["class_name"], str)
    assert data["latency_ms"] >= 0


def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 200


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code in [200, 307]