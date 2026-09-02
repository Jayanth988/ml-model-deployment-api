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