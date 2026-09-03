def test_predict_batch_with_oversized_batch_returns_422(client):
    single_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    payload = {
        "inputs": [single_input] * 101
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 422