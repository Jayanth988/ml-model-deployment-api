# ML Model Deployment as a Monitored REST API

A production-oriented Machine Learning REST API built with FastAPI, scikit-learn, Pydantic, structured logging, automated testing, Docker, and Docker Compose.

## Project Goal

Build and deploy a Machine Learning model as a reliable REST API with:

- Input validation
- Model loading at application startup
- Prediction and confidence scores
- Batch predictions
- API versioning
- Structured logging
- Request tracking
- Environment-based configuration
- Automated testing
- Docker containerization
- Docker Compose orchestration

## Dataset

Iris Flower Classification Dataset

The model predicts one of three iris flower classes using four numerical measurements.

## Machine Learning Model

- Algorithm: Random Forest Classifier
- Library: scikit-learn
- Model serialization: joblib
- Features:
  - sepal_length
  - sepal_width
  - petal_length
  - petal_width

The trained model is stored in:

```text
ml/saved_model/model.joblib
## Security and Robustness

### API Key Authentication

Protected API endpoints require an `X-API-Key` request header.

The API key is loaded from environment-based configuration using Pydantic Settings and is not hardcoded in the application code.

Missing or invalid API keys return:

```text
401 Unauthorized