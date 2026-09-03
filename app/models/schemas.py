from typing import List

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class PredictionInput(BaseModel):

    sepal_length: float = Field(
        ...,
        gt=0,
        description="Sepal length must be positive"
    )

    sepal_width: float = Field(
        ...,
        gt=0,
        description="Sepal width must be positive"
    )

    petal_length: float = Field(
        ...,
        gt=0,
        description="Petal length must be positive"
    )

    petal_width: float = Field(
        ...,
        gt=0,
        description="Petal width must be positive"
    )


class PredictionOutput(BaseModel):

    prediction: int
    confidence: float
    request_id: str


class PredictionBatchInput(BaseModel):

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


class PredictionBatchOutput(BaseModel):

    predictions: List[PredictionOutput]


class ModelInfoOutput(BaseModel):

    model_type: str
    model_version: str
    training_date: str
    feature_names: List[str]


class PredictionV2Output(BaseModel):

    prediction: int
    probabilities: List[float]
    request_id: str