from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    url: str = Field(
        ...,
        description="The URL to classify",
        examples=["http://secure-appleid-update.tk/login"],
    )


class PredictResponse(BaseModel):
    url: str
    status: str           
    probability: float   
    model_used: str
    features: dict
