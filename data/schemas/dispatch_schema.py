"""
data/schemas/dispatch_schema.py
Pydantic v2 schema definitions for dispatch events and
simulation state tracking (donor contact, response, arrival).
These schemas are used exclusively by the Dynamic Recruitment
Simulation (DRS) module — SYNTHETIC_SIMULATION stream only.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DonorState(str, Enum):
    """
    Finite state machine states for a donor within a recruitment episode.
    """
    UNCONTACTED = "UNCONTACTED"      # Not yet reached in this episode
    NOTIFIED = "NOTIFIED"            # Alert sent; awaiting response
    ACCEPTED = "ACCEPTED"            # Donor confirmed they will donate
    DECLINED = "DECLINED"            # Donor actively declined the alert
    EXPIRED = "EXPIRED"              # Response timeout exceeded T_critical
    ARRIVED = "ARRIVED"              # Donor physically arrived and donated


class DispatchEvent(BaseModel):
    """
    Records a single dispatch action in the simulation.
    """
    event_id: str
    episode_id: str = Field(..., description="Emergency episode this event belongs to")
    wave_number: int = Field(..., ge=1, description="Recruitment wave index")
    donor_id: str
    dispatched_at_minutes: float = Field(..., ge=0.0, description="Simulation time of dispatch (min)")
    response_latency_minutes: Optional[float] = Field(
        None, ge=0.0, description="Simulated time from dispatch to donor response"
    )
    accepted: Optional[bool] = Field(None, description="True if donor accepted, False if declined")
    arrival_time_minutes: Optional[float] = Field(
        None, ge=0.0, description="Simulated arrival time relative to episode start"
    )
    final_state: DonorState = Field(DonorState.UNCONTACTED)

    class Config:
        use_enum_values = True


class EpisodeResult(BaseModel):
    """
    Aggregated outcome of a single recruitment simulation episode.
    All EpisodeResult objects are tagged SYNTHETIC_SIMULATION and
    must never be reported as real-world clinical evidence.
    """
    episode_id: str
    data_tag: str = Field("SYNTHETIC_SIMULATION", description="Data stream classification — immutable")
    request_id: str
    units_required: int
    units_fulfilled: int = Field(0, ge=0)
    units_fulfilled_on_time: int = Field(0, ge=0)
    quota_met: bool = Field(False)
    quota_met_on_time: bool = Field(False)
    late_arrivals: int = Field(0, ge=0)
    time_to_fulfillment_minutes: Optional[float] = Field(
        None, description="Minutes from episode start to eventual quota fulfillment; None if not met"
    )
    time_to_on_time_fulfillment: Optional[float] = Field(
        None, description="Minutes to quota fulfillment if it happened strictly on time"
    )
    total_notifications_sent: int = Field(0, ge=0)
    notification_to_fulfillment_ratio: Optional[float] = Field(
        None, description="Total alerts sent / units fulfilled; None if quota not met"
    )
    shortfall: int = Field(0, ge=0, description="Units unfulfilled at episode end")
    waves_used: int = Field(0, ge=0)
    dispatch_log: list[DispatchEvent] = Field(default_factory=list)

    def compute_derived_metrics(self) -> None:
        """Compute derived metrics after episode completion."""
        self.shortfall = max(0, self.units_required - self.units_fulfilled)
        self.quota_met = self.units_fulfilled >= self.units_required
        self.quota_met_on_time = self.units_fulfilled_on_time >= self.units_required
        if self.quota_met and self.total_notifications_sent > 0 and self.units_fulfilled > 0:
            self.notification_to_fulfillment_ratio = (
                self.total_notifications_sent / self.units_fulfilled
            )
