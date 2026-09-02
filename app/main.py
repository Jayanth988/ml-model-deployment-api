from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import setup_logging
import joblib
import uuid
import time


logger = setup_logging()

model = None


# Custom exception for prediction shape problems
class PredictionShapeError(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    logger.info("Loading model")

    model = joblib.load("ml/saved_model/model.joblib")

    logger.info("Model loaded successfully")

    yield

    logger.info("Application shutdown")


app = FastAPI(lifespan=lifespan)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = None

    try:
        response = await call_next(request)
        return response

    finally:
        duration = time.perf_counter() - start_time

        status_code = (
            response.status_code
            if response is not None
            else 500
        )

        logger.info(
            "request_id=%s method=%s path=%s duration=%.4fs status_code=%s",
            request_id,
            request.method,
            request.url.path,
            duration,
            status_code
        )


# Custom exception handler
@app.exception_handler(PredictionShapeError)
async def prediction_shape_exception_handler(
    request: Request,
    exc: PredictionShapeError
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    logger.error(
        "request_id=%s prediction shape error: %s",
        request_id,
        exc
    )

    return JSONResponse(
        status_code=400,
        content={
            "error": "Prediction input shape is invalid",
            "detail": str(exc)
        }
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(data: PredictionInput, request: Request):

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