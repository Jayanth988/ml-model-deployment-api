from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.models.schemas import PredictionInput, PredictionOutput
import joblib
import uuid


model = None


# Custom exception for prediction shape problems
class PredictionShapeError(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Loading Model...")

    model = joblib.load("ml/saved_model/model.joblib")

    print("Model Loaded Successfully")

    yield

    print("Application Shutdown")


app = FastAPI(lifespan=lifespan)


# Custom exception handler
@app.exception_handler(PredictionShapeError)
async def prediction_shape_exception_handler(
    request: Request,
    exc: PredictionShapeError
):
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


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

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

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = str(uuid.uuid4())

    return {
        "prediction": int(prediction[0]),
        "confidence": float(confidence),
        "request_id": request_id
    }