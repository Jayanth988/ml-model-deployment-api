def test_v2_predict_with_valid_input_returns_probabilities(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1, 2]
    assert "probabilities" in data
    assert "confidence" not in data

    assert len(data["probabilities"]) == 3

    for probability in data["probabilities"]:
        assert 0 <= probability <= 1

    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0


def test_v1_and_v2_have_different_response_shapes(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=payload
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    assert "confidence" in v1_data
    assert "probabilities" not in v1_data

    assert "probabilities" in v2_data
    assert "confidence" not in v2_data

    assert set(v1_data.keys()) == {
        "prediction",
        "confidence",
        "request_id"
    }

    assert set(v2_data.keys()) == {
        "prediction",
        "probabilities",
        "request_id"
    }

    assert v1_data["prediction"] == v2_data["prediction"]