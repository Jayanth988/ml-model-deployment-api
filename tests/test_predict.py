def test_predict_with_valid_input_returns_prediction(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1, 2]
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0