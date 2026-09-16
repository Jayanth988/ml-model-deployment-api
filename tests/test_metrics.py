from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_metrics_endpoint_returns_prometheus_format():
    with TestClient(app) as test_client:
        response = test_client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# HELP" in response.text
    assert "# TYPE" in response.text
    assert "http_requests_total" in response.text


def test_prediction_creates_ml_prediction_metric():
    with TestClient(app) as test_client:
        test_client.headers.update({
            "X-API-Key": settings.API_KEY
        })

        prediction_response = test_client.post(
            "/api/v1/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )

        assert prediction_response.status_code == 200

        prediction_class = prediction_response.json()["prediction"]

        metrics_response = test_client.get("/metrics")

    assert metrics_response.status_code == 200
    assert "ml_predictions_total" in metrics_response.text
    assert (
        f'predicted_class="{prediction_class}"'
        in metrics_response.text
    )