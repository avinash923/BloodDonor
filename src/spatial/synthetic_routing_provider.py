"""
src/spatial/synthetic_routing_provider.py
Synthetic Travel Time Provider.

Wraps the existing deterministic TravelTimeEngine and CongestionModel
under the unified BaseRoutingProvider interface.
Guarantees 100% backward-compatibility with all existing simulation scripts
and historical regression suites.

DATA STREAM: SYNTHETIC_SIMULATION
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import numpy as np

from src.spatial.routing_provider import (
    BaseRoutingProvider,
    LocationCoordinate,
    RouteEstimate,
)
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.spatial.distance_metrics import network_grid_distance_km
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticRoutingProvider(BaseRoutingProvider):
    """
    Local synthetic travel time provider based on distance metrics and congestion profile.
    Requires no API keys, runs offline, and produces deterministic results.
    """

    def __init__(
        self,
        travel_time_engine: Optional[TravelTimeEngine] = None,
        base_speed_kmh: float = 30.0,
        distance_method: str = "network_grid",
        congestion_model: Optional[CongestionModel] = None,
        detour_factor: float = 1.35,
    ) -> None:
        if travel_time_engine is not None:
            self.engine = travel_time_engine
        else:
            self.engine = TravelTimeEngine(
                base_speed_kmh=base_speed_kmh,
                distance_method=distance_method,
                congestion_model=congestion_model,
                detour_factor=detour_factor,
            )

    def get_travel_time_minutes(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> Optional[float]:
        """Compute single travel time using local congestion engine."""
        h = hour_of_day if hour_of_day is not None else (departure_time.hour if departure_time else 10)
        tt = self.engine.travel_time_minutes(
            donor_lat=origin.latitude,
            donor_lon=origin.longitude,
            site_lat=destination.latitude,
            site_lon=destination.longitude,
            hour_of_day=h,
        )
        return float(tt) if np.isfinite(tt) else None

    def compute_donor_eta_batch(
        self,
        donor_locations: dict[str, LocationCoordinate],
        hospital_location: LocationCoordinate,
        hospital_id: str = "SITE_001",
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> dict[str, RouteEstimate]:
        """Compute batch ETAs across all donors in the candidate pool."""
        h = hour_of_day if hour_of_day is not None else (departure_time.hour if departure_time else 10)
        estimates: dict[str, RouteEstimate] = {}

        if not donor_locations:
            return estimates

        donor_ids = list(donor_locations.keys())
        d_lats = np.array([donor_locations[d].latitude for d in donor_ids], dtype=float)
        d_lons = np.array([donor_locations[d].longitude for d in donor_ids], dtype=float)

        # Vectorized engine call
        tt_arr = self.engine.travel_time_minutes(
            donor_lat=d_lats,
            donor_lon=d_lons,
            site_lat=hospital_location.latitude,
            site_lon=hospital_location.longitude,
            hour_of_day=h,
        )

        for did, tt, d_lat, d_lon in zip(donor_ids, tt_arr, d_lats, d_lons):
            if np.isfinite(tt) and tt >= 0:
                # Approximate distance in meters
                dist_km = network_grid_distance_km(
                    d_lat, d_lon, hospital_location.latitude, hospital_location.longitude,
                    detour_factor=self.engine.detour_factor,
                )
                dur_sec = float(tt * 60.0)
                estimates[did] = RouteEstimate(
                    origin_id=did,
                    destination_id=hospital_id,
                    distance_meters=float(dist_km * 1000.0),
                    duration_seconds=dur_sec,
                    duration_minutes=float(tt),
                    traffic_duration_seconds=dur_sec,
                    provider="synthetic",
                    is_success=True,
                )
            else:
                estimates[did] = RouteEstimate.failed(
                    origin_id=did,
                    destination_id=hospital_id,
                    error_code="INFINITE_OR_INVALID_ETA",
                    provider="synthetic",
                )

        return estimates

    def compute_detailed_route(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        origin_id: str = "ORIGIN",
        destination_id: str = "DEST",
        departure_time: Optional[datetime] = None,
    ) -> RouteEstimate:
        """Compute synthetic detailed route with straight-line polyline representation."""
        h = departure_time.hour if departure_time else 10
        tt = self.get_travel_time_minutes(origin, destination, hour_of_day=h)
        if tt is None:
            return RouteEstimate.failed(origin_id, destination_id, "NO_ROUTE_FOUND", provider="synthetic")

        dist_km = network_grid_distance_km(
            origin.latitude, origin.longitude, destination.latitude, destination.longitude,
            detour_factor=self.engine.detour_factor,
        )
        return RouteEstimate(
            origin_id=origin_id,
            destination_id=destination_id,
            distance_meters=float(dist_km * 1000.0),
            duration_seconds=float(tt * 60.0),
            duration_minutes=float(tt),
            traffic_duration_seconds=float(tt * 60.0),
            provider="synthetic",
            is_success=True,
            polyline_encoded=None,
        )
