from contextlib import asynccontextmanager
import json
import time
import uuid

import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.logging_config import setup_logging
from app.routers.v1 import router as v1_router
from app.routers.v2 import router as v2_router


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

    model = joblib.load(settings.MODEL_PATH)

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


# Prometheus monitoring
instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app)


allowed_origins = [
    origin.strip()
    for origin in settings.CORS_ORIGINS.split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.exception_handler(PredictionShapeError)
async def prediction_shape_error_handler(
    request: Request,
    exc: PredictionShapeError
):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc)
        }
    )


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next
):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration = time.perf_counter() - start_time

        logger.info(
            "request_id=%s method=%s path=%s status=%s duration=%.4fs",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration
        )

        return response

    except Exception:
        duration = time.perf_counter() - start_time

        logger.exception(
            "request_id=%s method=%s path=%s "
            "unhandled exception duration=%.4fs",
            request_id,
            request.method,
            request.url.path,
            duration
        )

        raise


app.include_router(v1_router)
app.include_router(v2_router)