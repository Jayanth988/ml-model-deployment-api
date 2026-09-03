from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    PredictionInput,
    PredictionV2Output
)


router = APIRouter(
    prefix="/api/v2",
    tags=["v2"]
)


@router.post(
    "/predict",
    response_model=PredictionV2Output
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

    except Exception as exc:

        logger.error(
            "request_id=%s v2 prediction failed: %s",
            request_id,
            exc,
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail="V2 prediction failed"
        )

    prediction_value = int(prediction[0])

    probability_values = [
        float(probability)
        for probability in probabilities[0]
    ]

    logger.info(
        "request_id=%s v2 prediction succeeded "
        "prediction=%s probabilities=%s",
        request_id,
        prediction_value,
        probability_values
    )

    return {
        "prediction": prediction_value,
        "probabilities": probability_values,
        "request_id": request_id
    }