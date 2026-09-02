from typing import List
from pydantic import BaseModel, Field


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
        max_length=100,
        description="Batch must contain between 1 and 100 inputs"
    )


class PredictionBatchOutput(BaseModel):
    predictions: List[PredictionOutput]


class ModelInfoOutput(BaseModel):
    model_type: str
    model_version: str
    training_date: str
    feature_names: List[str]