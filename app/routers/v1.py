from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import PredictionInput, PredictionOutput


router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)


@router.get("/health")
def health(request: Request):
    model = request.app.state.model

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@router.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(data: PredictionInput, request: Request):

    model = request.app.state.model
    logger = request.app.state.logger
    request_id = request.state.request_id

    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    try:
        prediction = model.predict([features])

        probabilities = model.predict_proba([features])

        confidence = probabilities[0][prediction[0]]

    except Exception as exc:

        logger.error(
            "request_id=%s prediction failed: %s",
            request_id,
            exc,
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    prediction_value = int(prediction[0])
    confidence_value = float(confidence)

    logger.info(
        "request_id=%s prediction succeeded prediction=%s confidence=%.4f",
        request_id,
        prediction_value,
        confidence_value
    )

    return {
        "prediction": prediction_value,
        "confidence": confidence_value,
        "request_id": request_id
    }


# V2 design plan:
# If /api/v2/predict needs an extra response field, I would create
# a separate v2 router and a separate Pydantic response schema.
# The existing v1 schema and endpoint would remain unchanged so
# existing clients continue to work without breaking changes.