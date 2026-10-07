from typing import Dict, Optional
from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    title: str = Field(..., description="Title of the support ticket")
    description: str = Field(..., description="Detailed description of the issue")


class PredictionResponse(BaseModel):
    category: str = Field(..., description="Predicted ticket category")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Class probabilities across categories"
    )


class HealthCheckResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)