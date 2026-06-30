"""Pydantic schemas for service health responses."""

from pydantic import BaseModel
from typing import Optional


# Health payloads stay intentionally small so load balancers, Spaces, and CI
# probes can check readiness without invoking model inference.
class HealthResponse(BaseModel):
    """Health response returned by root and health-check endpoints.

    Attributes:
        name: Optional API display name returned by the root endpoint.
        status: Machine-readable service status.
        version: API version string.
    """

    name: Optional[str] = None
    status: str
    version: str
