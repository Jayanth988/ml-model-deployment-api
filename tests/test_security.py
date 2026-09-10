from fastapi.testclient import TestClient

from app.main import app


def test_missing_api_key_returns_401():
    with TestClient(app) as test_client:
        response = test_client.get("/api/v1/health")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_invalid_api_key_returns_401():
    with TestClient(app) as test_client:
        response = test_client.get(
            "/api/v1/health",
            headers={"X-API-Key": "wrong-key"}
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_unexpected_extra_field_returns_422(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "unexpected_field": "not allowed"
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_empty_string_returns_422(client):
    payload = {
        "sepal_length": "",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_extreme_number_returns_422(client):
    payload = {
        "sepal_length": 1000000,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422