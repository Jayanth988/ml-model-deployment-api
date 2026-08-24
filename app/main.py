from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.models.schemas import PredictionInput
import joblib


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


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict(data: PredictionInput):
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    prediction = model.predict([features])

    return {
        "prediction": int(prediction[0])
    }