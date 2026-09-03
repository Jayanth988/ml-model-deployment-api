def test_model_info_returns_expected_metadata(client):
    response = client.get("/api/v1/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model_type" in data
    assert "model_version" in data
    assert "training_date" in data
    assert "feature_names" in data

    assert data["model_type"] == "RandomForestClassifier"
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) == 4