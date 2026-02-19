from pydantic import BaseModel
from typing import Optional

class HealthResponse(BaseModel):
    name: Optional[str] = None
    status: str
    version: str
