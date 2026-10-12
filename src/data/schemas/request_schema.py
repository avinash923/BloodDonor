"""
data/schemas/request_schema.py
Pydantic v2 schema definitions for Emergency Blood Requests.
Used by the spatial engine, ranking module, and simulation.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class UrgencyTier(str, Enum):
    """
    Standardized urgency classification for emergency blood requests.
    Determines T_critical (maximum transport horizon) in the feasibility engine.
    """
    CRITICAL = "CRITICAL"          # Level 1 – Life-threatening trauma (e.g., 60-min window)
    URGENT = "URGENT"              # Level 2 – Urgent surgery (e.g., 120-min window)
    SCHEDULED = "SCHEDULED"        # Level 3 – Planned shortage management (e.g., 240-min window)


# Maps each urgency tier to its maximum transport horizon in minutes
URGENCY_HORIZON_MINUTES: dict[str, int] = {
    UrgencyTier.CRITICAL: 60,
    UrgencyTier.URGENT: 120,
    UrgencyTier.SCHEDULED: 240,
}

# Backward-compatible alias used by experiment runners
_URGENCY_HORIZONS = URGENCY_HORIZON_MINUTES

# Maps each urgency tier to its normalized urgency score U(e) in [0, 1]
URGENCY_SCORE: dict[str, float] = {
    UrgencyTier.CRITICAL: 1.0,
    UrgencyTier.URGENT: 0.7,
    UrgencyTier.SCHEDULED: 0.4,
}


class EmergencyRequest(BaseModel):
    """
    Schema for a blood emergency request event.
    Consumed by spatial feasibility engine and UDI ranking.
    """
    request_id: str = Field(..., description="Unique emergency request identifier")
    blood_type_required: str = Field(..., description="Required blood type (e.g., O-, A+)")
    units_required: int = Field(..., ge=1, description="Number of blood units needed")
    urgency: UrgencyTier = Field(..., description="Urgency classification tier")

    # Hospital / collection site coordinates
    site_latitude: float = Field(..., ge=-90.0, le=90.0)
    site_longitude: float = Field(..., ge=-180.0, le=180.0)

    # Temporal context
    request_timestamp: datetime = Field(..., description="Time the emergency request was raised")
    deadline_minutes: Optional[int] = Field(
        None, ge=1,
        description="Override for T_critical; defaults to URGENCY_HORIZON_MINUTES[urgency]"
    )

    def effective_horizon_minutes(self) -> int:
        """Returns the effective maximum transport horizon in minutes."""
        if self.deadline_minutes is not None:
            return self.deadline_minutes
        return URGENCY_HORIZON_MINUTES[self.urgency]

    def urgency_score(self) -> float:
        """Returns the normalized urgency score U(e) for UDI computation."""
        return URGENCY_SCORE[self.urgency]

    class Config:
        use_enum_values = True
