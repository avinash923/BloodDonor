"""
src/spatial/routing_service.py
Unified Routing Service Factory & Dispatcher.

Manages provider lifecycle and selection based on system configuration:
- ROUTING_PROVIDER=synthetic (default)
- ROUTING_PROVIDER=google
- ROUTING_PROVIDER=osrm (free OpenStreetMap road routing, no key)

Provides clean integration with the recommendation engine and UDI ranker:
Takes candidate donors and emergency hospital coordinates, retrieves ETAs,
and populates DonorCandidate.travel_time_minutes.

DATA STREAM: SPATIAL_ROUTING
"""

from __future__ import annotations

import os
from typing import Optional, Sequence
from datetime import datetime
import numpy as np

from src.spatial.routing_provider import (
    BaseRoutingProvider,
    RoutingProviderType,
    LocationCoordinate,
    RouteEstimate,
    TravelMode,
    RoutingPreference,
)
from src.spatial.synthetic_routing_provider import SyntheticRoutingProvider
from src.spatial.google_routes_provider import GoogleMapsRoutesProvider
from src.spatial.osrm_provider import OSRMRoutingProvider
from src.ranking.udi_engine import DonorCandidate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RoutingService:
    """
    High-level routing coordinator that dispatches routing calculations
    to the active provider (Synthetic or Google Maps).
    """

    def __init__(
        self,
        provider_type: Optional[RoutingProviderType | str] = None,
        google_api_key: Optional[str] = None,
        travel_mode: TravelMode = TravelMode.DRIVE,
        routing_preference: RoutingPreference = RoutingPreference.TRAFFIC_AWARE,
        synthetic_provider: Optional[SyntheticRoutingProvider] = None,
    ) -> None:
        raw_type = (
            provider_type
            or os.getenv("ROUTING_PROVIDER", RoutingProviderType.SYNTHETIC.value)
        )
        self.provider_type = RoutingProviderType(str(raw_type).lower())

        if self.provider_type == RoutingProviderType.GOOGLE:
            logger.info("Initializing Google Maps Routes API provider.")
            self.provider: BaseRoutingProvider = GoogleMapsRoutesProvider(
                api_key=google_api_key,
                travel_mode=travel_mode,
                routing_preference=routing_preference,
            )
        elif self.provider_type == RoutingProviderType.OSRM:
            logger.info("Initializing free OSRM road routing provider (OpenStreetMap).")
            self.provider = OSRMRoutingProvider(fallback=synthetic_provider)
        else:
            logger.info("Initializing Synthetic Routing provider (offline/deterministic).")
            self.provider = synthetic_provider or SyntheticRoutingProvider()

    def get_travel_time_minutes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> Optional[float]:
        """Compute single travel time between coordinates."""
        orig = LocationCoordinate(origin_lat, origin_lon)
        dest = LocationCoordinate(dest_lat, dest_lon)
        return self.provider.get_travel_time_minutes(
            origin=orig,
            destination=dest,
            departure_time=departure_time,
            hour_of_day=hour_of_day,
        )

    def compute_donor_etas(
        self,
        donors: Sequence[tuple[str, float, float]], # list of (donor_id, lat, lon)
        hospital_lat: float,
        hospital_lon: float,
        hospital_id: str = "SITE_001",
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> dict[str, RouteEstimate]:
        """
        Compute ETAs for a collection of donors.
        Validates coordinate bounds, rejecting malformed coordinates safely.
        """
        valid_locations: dict[str, LocationCoordinate] = {}
        failed_estimates: dict[str, RouteEstimate] = {}

        hospital_coord = LocationCoordinate(hospital_lat, hospital_lon)

        for did, lat, lon in donors:
            try:
                coord = LocationCoordinate(lat, lon)
                valid_locations[did] = coord
            except (ValueError, TypeError) as e:
                logger.warning("Donor %s has invalid coordinates: (%s, %s)", did, lat, lon)
                failed_estimates[did] = RouteEstimate.failed(
                    origin_id=did,
                    destination_id=hospital_id,
                    error_code="INVALID_COORDINATES",
                    provider=self.provider_type.value,
                )

        # Query provider for valid locations
        valid_estimates = self.provider.compute_donor_eta_batch(
            donor_locations=valid_locations,
            hospital_location=hospital_coord,
            hospital_id=hospital_id,
            departure_time=departure_time,
            hour_of_day=hour_of_day,
        )

        # Merge results
        return {**failed_estimates, **valid_estimates}

    def populate_candidates_with_etas(
        self,
        candidates: Sequence[DonorCandidate],
        donor_coordinates: dict[str, tuple[float, float]], # did -> (lat, lon)
        hospital_lat: float,
        hospital_lon: float,
        hospital_id: str = "SITE_001",
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> list[DonorCandidate]:
        """
        Attaches calculated route travel times directly onto DonorCandidate objects
        for immediate consumption by UDIEngine.score_and_rank().
        Missing/failed routes explicitly receive travel_time_minutes = None.
        """
        donor_list = [
            (c.donor_id, donor_coordinates[c.donor_id][0], donor_coordinates[c.donor_id][1])
            for c in candidates
            if c.donor_id in donor_coordinates
        ]

        estimates = self.compute_donor_etas(
            donors=donor_list,
            hospital_lat=hospital_lat,
            hospital_lon=hospital_lon,
            hospital_id=hospital_id,
            departure_time=departure_time,
            hour_of_day=hour_of_day,
        )

        updated_candidates = []
        for c in candidates:
            est = estimates.get(c.donor_id)
            if est and est.is_success and est.duration_minutes is not None:
                c.travel_time_minutes = float(est.duration_minutes)
            else:
                c.travel_time_minutes = None
            updated_candidates.append(c)

        return updated_candidates
