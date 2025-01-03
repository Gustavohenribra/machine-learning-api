from pydantic import BaseModel
from typing import List

class InferenceRequest(BaseModel):
    """Para uma única amostra."""
    features: List[float]

class BatchInferenceRequest(BaseModel):
    """Para batch de amostras."""
    features_batch: List[List[float]]
