from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.models.schemas import PredictionInput
import joblib
import uuid


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Loading Model...")

    model = joblib.load("ml/saved_model/model.joblib")

    print("Model Loaded Successfully")

    yield

    print("Application Shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: PredictionInput):
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    prediction = model.predict([features])

    probabilities = model.predict_proba([features])

    confidence = probabilities[0][prediction[0]]

    request_id = str(uuid.uuid4())

    return {
        "prediction": int(prediction[0]),
        "confidence": float(confidence),
        "request_id": request_id
    }