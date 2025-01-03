from pydantic import BaseModel
from typing import Optional

class TrainRequest(BaseModel):
    file_name: str
    target_column: str
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42
