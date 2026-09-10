from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )


class PredictionInput(StrictBaseModel):

    sepal_length: float = Field(
        ...,
        gt=0,
        le=100,
        description="Sepal length must be between 0 and 100"
    )

    sepal_width: float = Field(
        ...,
        gt=0,
        le=100,
        description="Sepal width must be between 0 and 100"
    )

    petal_length: float = Field(
        ...,
        gt=0,
        le=100,
        description="Petal length must be between 0 and 100"
    )

    petal_width: float = Field(
        ...,
        gt=0,
        le=100,
        description="Petal width must be between 0 and 100"
    )


class PredictionOutput(StrictBaseModel):

    prediction: int
    confidence: float
    request_id: str


class PredictionBatchInput(StrictBaseModel):

    inputs: List[PredictionInput] = Field(
        ...,
        min_length=1,
        description="Batch must contain at least 1 input"
    )

    @field_validator("inputs")
    @classmethod
    def validate_batch_size(cls, value):

        if len(value) > settings.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size cannot exceed {settings.MAX_BATCH_SIZE}"
            )

        return value


class PredictionBatchOutput(StrictBaseModel):

    predictions: List[PredictionOutput]


class ModelInfoOutput(StrictBaseModel):

    model_type: str
    model_version: str
    training_date: str
    feature_names: List[str]


class PredictionV2Output(StrictBaseModel):

    prediction: int
    probabilities: List[float]
    request_id: str