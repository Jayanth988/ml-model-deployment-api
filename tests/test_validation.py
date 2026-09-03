def test_predict_with_missing_field_returns_422(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


def test_predict_with_invalid_value_returns_422(client):
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422