from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.logging_config import setup_logging
from app.routers.v1 import router as v1_router
from app.routers.v2 import router as v2_router

import joblib
import json
import uuid
import time


logger = setup_logging()

model = None
model_metadata = None


class PredictionShapeError(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model
    global model_metadata

    logger.info("Loading model")

    model = joblib.load(
        settings.MODEL_PATH
    )

    with open(
        settings.METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as metadata_file:
        model_metadata = json.load(metadata_file)

    app.state.model = model
    app.state.model_metadata = model_metadata
    app.state.logger = logger

    logger.info("Model loaded successfully")
    logger.info("Model metadata loaded successfully")

    yield

    logger.info("Application shutdown")


app = FastAPI(
    title=settings.API_TITLE,
    lifespan=lifespan
)


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
            "request_id=%s method=%s path=%s "
            "duration=%.4fs status_code=%s",
            request_id,
            request.method,
            request.url.path,
            duration,
            status_code
        )


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


app.include_router(v1_router)

app.include_router(v2_router)