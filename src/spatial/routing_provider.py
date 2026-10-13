"""
src/spatial/routing_provider.py
Abstract Routing Provider Interface & Domain Data Models.

Provides unified abstractions for:
- SyntheticTravelTimeProvider: Local deterministic distance and congestion engine.
- GoogleMapsRoutesProvider: Real-time Routes API (computeRouteMatrix & computeRoutes).

DATA STREAM: SPATIAL_ROUTING (Abstract provider layer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Sequence
import numpy as np


class RoutingProviderType(str, Enum):
    """Available routing providers."""
    SYNTHETIC = "synthetic"
    GOOGLE = "google"
    OSRM = "osrm"


class TravelMode(str, Enum):
    """Supported transportation modes for the Routes API."""
    DRIVE = "DRIVE"
    TWO_WHEELER = "TWO_WHEELER"
    BICYCLE = "BICYCLE"
    WALK = "WALK"


class RoutingPreference(str, Enum):
    """Traffic-awareness preferences."""
    TRAFFIC_UNAWARE = "TRAFFIC_UNAWARE"
    TRAFFIC_AWARE = "TRAFFIC_AWARE"
    TRAFFIC_AWARE_OPTIMAL = "TRAFFIC_AWARE_OPTIMAL"


@dataclass(frozen=True)
class LocationCoordinate:
    """Validated geographical point (latitude, longitude)."""
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")

    def to_dict(self) -> dict:
        return {
            "latLng": {
                "latitude": float(self.latitude),
                "longitude": float(self.longitude),
            }
        }


@dataclass(frozen=True)
class RouteEstimate:
    """
    Typed result of a donor-to-hospital routing query.
    
    Fields:
        origin_id: Donor identifier.
        destination_id: Emergency site/hospital identifier.
        distance_meters: Physical road distance in meters (None if infeasible/error).
        duration_seconds: Travel duration in seconds (None if infeasible/error).
        duration_minutes: Derived travel duration in minutes (used directly by UDI engine).
        traffic_duration_seconds: Duration incorporating real-time traffic, if available.
        provider: Provider identifier ('synthetic' or 'google').
        is_success: True if a valid route was found and ETA computed.
        error_code: Optional diagnostic error status (e.g., 'NO_ROUTE_FOUND', 'INVALID_ARGUMENT').
        polyline_encoded: Optional route geometry string for visual map display.
        estimated_at: Timestamp when the estimate was generated.
    """
    origin_id: str
    destination_id: str
    distance_meters: Optional[float] = None
    duration_seconds: Optional[float] = None
    duration_minutes: Optional[float] = None
    traffic_duration_seconds: Optional[float] = None
    provider: str = "synthetic"
    is_success: bool = True
    error_code: Optional[str] = None
    polyline_encoded: Optional[str] = None
    estimated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def failed(
        cls,
        origin_id: str,
        destination_id: str,
        error_code: str,
        provider: str = "unknown",
    ) -> RouteEstimate:
        """Create a failed route estimate where duration_minutes is explicitly None."""
        return cls(
            origin_id=origin_id,
            destination_id=destination_id,
            distance_meters=None,
            duration_seconds=None,
            duration_minutes=None,
            traffic_duration_seconds=None,
            provider=provider,
            is_success=False,
            error_code=error_code,
        )


class BaseRoutingProvider(ABC):
    """Abstract interface that all travel-time routing providers must implement."""

    @abstractmethod
    def get_travel_time_minutes(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> Optional[float]:
        """
        Compute travel time in minutes between a single origin and destination.
        Returns None if no feasible route exists or calculation failed.
        """
        pass

    @abstractmethod
    def compute_donor_eta_batch(
        self,
        donor_locations: dict[str, LocationCoordinate],
        hospital_location: LocationCoordinate,
        hospital_id: str = "SITE_001",
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> dict[str, RouteEstimate]:
        """
        Compute ETAs for multiple donors travelling to a single emergency hospital.
        Returns mapping from donor_id to RouteEstimate.
        Failed/unreachable routes MUST return RouteEstimate.is_success == False
        and duration_minutes == None (NEVER 0.0).
        """
        pass

    @abstractmethod
    def compute_detailed_route(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        origin_id: str = "ORIGIN",
        destination_id: str = "DEST",
        departure_time: Optional[datetime] = None,
    ) -> RouteEstimate:
        """
        Compute a single detailed route including route geometry (encoded polyline).
        """
        pass
