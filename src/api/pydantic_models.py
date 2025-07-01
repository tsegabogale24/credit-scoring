from pydantic import BaseModel
from typing import List

class CustomerData(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    risk_probability: float
