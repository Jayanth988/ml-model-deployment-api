# ML Model Deployment as a Monitored REST API

## Project Goal

Build a production-ready Machine Learning REST API using FastAPI.

## Dataset

Iris Flower Classification Dataset

## Problem Statement

Predict iris flower species using flower measurements.

## API Input

- sepal_length
- sepal_width
- petal_length
- petal_width

## API Output

- prediction
- confidence

## Request Flow

User
→ FastAPI
→ Validation
→ ML Model
→ Prediction
→ JSON Response