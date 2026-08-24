from fastapi import FastAPI
from contextlib import asynccontextmanager
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
def predict():
    features = [5.1, 3.5, 1.4, 0.2]

    prediction = model.predict([features])

    return {
        "prediction": int(prediction[0])
    }