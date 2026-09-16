# ML Model Deployment as a Monitored REST API

A production-oriented Machine Learning REST API built with FastAPI and scikit-learn.

This project demonstrates how to take a trained Machine Learning model and expose it through a reliable, validated, monitored, tested, and containerized REST API.

## Project Overview

The project uses the Iris Flower Classification dataset and a Random Forest Classifier.

The trained model is exposed through FastAPI with:

- Input validation using Pydantic
- Model loading during application startup
- Single predictions
- Batch predictions
- API versioning
- Prediction confidence and probabilities
- API key authentication
- Request IDs for tracing
- Structured application logging
- Prometheus metrics
- Environment-based configuration
- Automated pytest testing
- Docker containerization
- Docker Compose
- GitHub Actions CI

## Architecture

```text
                        Client
                          |
                          | HTTP Request
                          v
                 +-------------------+
                 |     FastAPI       |
                 |    Application    |
                 +---------+---------+
                           |
                +----------+----------+
                |                     |
                v                     v
        API Key Security        Pydantic Validation
                |                     |
                +----------+----------+
                           |
                           v
                  Versioned API Router
                   /api/v1 or /api/v2
                           |
                           v
                 +-------------------+
                 |   Loaded ML Model |
                 | Random Forest     |
                 +---------+---------+
                           |
                           v
                    Prediction Result
                           |
             +-------------+-------------+
             |                           |
             v                           v
        Structured Logs          Prometheus Metrics
             |                           |
             v                           v
          logs/app.log               /metrics


                 Docker Container
                        |
                        v
                  Docker Compose