"""
src/spatial/osrm_provider.py
Free road routing through an OSRM server (OpenStreetMap data, no API key).

Endpoints used (OSRM HTTP API v1):
  /table/v1/{profile}/{coords}?sources=..&destinations=0&annotations=duration,distance
      -> road durations/distances from many donors to one hospital
  /route/v1/{profile}/{origin};{destination}?overview=full&geometries=polyline
      -> one road route with an encoded polyline (precision 5) for the map

The default base URL is the public OSRM demo server, which is free but
rate-limited and offered without guarantees; set OSRM_BASE_URL to a
self-hosted OSRM for production. OSRM durations are free-flow estimates:
they do not include live traffic.

When OSRM cannot be reached, estimates fall back to the local synthetic
engine and are labelled ``synthetic_fallback`` so callers never present
them as road routes.

DATA STREAM: SPATIAL_ROUTING
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.spatial.routing_provider import BaseRoutingProvider, LocationCoordinate, RouteEstimate
from src.spatial.synthetic_routing_provider import SyntheticRoutingProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OSRM_BASE_URL = "https://router.project-osrm.org"
FALLBACK_PROVIDER = "synthetic_fallback"


def _coord(point: LocationCoordinate) -> str:
    # OSRM expects longitude,latitude.
    return f"{point.longitude:.6f},{point.latitude:.6f}"


class OSRMRoutingProvider(BaseRoutingProvider):
    """Road ETAs and route geometry from an OSRM server, with synthetic fallback."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        profile: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_table_sources: int = 90,
        cache_ttl_seconds: float = 60.0,
        fallback: Optional[BaseRoutingProvider] = None,
        retry_after_failure_seconds: float = 60.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("OSRM_BASE_URL") or DEFAULT_OSRM_BASE_URL).rstrip("/")
        self.profile = profile or os.getenv("OSRM_PROFILE", "driving")
        self.timeout_seconds = float(timeout_seconds or os.getenv("OSRM_TIMEOUT_SECONDS", "8"))
        self.max_table_sources = max_table_sources
        self.cache_ttl_seconds = cache_ttl_seconds
        self.fallback = fallback or SyntheticRoutingProvider()
        self._cache: dict[tuple, tuple[float, RouteEstimate]] = {}
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None
        # Circuit breaker: after a failure, skip OSRM for a while instead of
        # waiting for the timeout on every request.
        self.retry_after_failure_seconds = retry_after_failure_seconds
        self._down_until = 0.0

    # -----------------------------------------------------------------
    # HTTP + cache helpers
    # -----------------------------------------------------------------

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._down_until

    def _get(self, service: str, coords: str, params: dict) -> dict:
        if not self.available:
            raise ConnectionError("OSRM temporarily skipped after a recent failure")
        try:
            return self._request(service, coords, params)
        except Exception:
            self._down_until = time.monotonic() + self.retry_after_failure_seconds
            raise

    def _request(self, service: str, coords: str, params: dict) -> dict:
        url = f"{self.base_url}/{service}/v1/{self.profile}/{coords}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "LifePulse-BloodDonor/1.0 (emergency donor matching)"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("code") != "Ok":
            raise RuntimeError(f"OSRM returned {payload.get('code')}")
        return payload

    def _cache_key(self, kind: str, origin: LocationCoordinate, destination: LocationCoordinate) -> tuple:
        return (kind, round(origin.latitude, 5), round(origin.longitude, 5),
                round(destination.latitude, 5), round(destination.longitude, 5))

    def _cached(self, key: tuple) -> Optional[RouteEstimate]:
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < self.cache_ttl_seconds:
                return hit[1]
        return None

    def _store(self, key: tuple, estimate: RouteEstimate) -> None:
        with self._lock:
            if len(self._cache) > 5000:
                self._cache.clear()
            self._cache[key] = (time.monotonic(), estimate)

    @staticmethod
    def _relabel(estimate: RouteEstimate) -> RouteEstimate:
        return RouteEstimate(**{**estimate.__dict__, "provider": FALLBACK_PROVIDER})

    # -----------------------------------------------------------------
    # BaseRoutingProvider API
    # -----------------------------------------------------------------

    def get_travel_time_minutes(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> Optional[float]:
        estimate = self.compute_detailed_route(origin, destination, departure_time=departure_time)
        return estimate.duration_minutes if estimate.is_success else None

    def compute_donor_eta_batch(
        self,
        donor_locations: dict[str, LocationCoordinate],
        hospital_location: LocationCoordinate,
        hospital_id: str = "SITE_001",
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> dict[str, RouteEstimate]:
        estimates: dict[str, RouteEstimate] = {}
        pending: list[str] = []
        for did, location in donor_locations.items():
            hit = self._cached(self._cache_key("eta", location, hospital_location))
            if hit is not None:
                estimates[did] = RouteEstimate(**{**hit.__dict__, "origin_id": did, "destination_id": hospital_id})
            else:
                pending.append(did)

        for start in range(0, len(pending), self.max_table_sources):
            chunk = pending[start:start + self.max_table_sources]
            coords = ";".join([_coord(hospital_location)] + [_coord(donor_locations[d]) for d in chunk])
            try:
                payload = self._get("table", coords, {
                    "sources": ";".join(str(i) for i in range(1, len(chunk) + 1)),
                    "destinations": "0",
                    "annotations": "duration,distance",
                })
                durations = payload.get("durations") or []
                distances = payload.get("distances") or []
                self.last_error = None
            except Exception as error:  # network, HTTP, JSON or OSRM error
                self.last_error = type(error).__name__
                logger.warning("OSRM table request failed (%s); using synthetic fallback.", self.last_error)
                subset = {d: donor_locations[d] for d in chunk}
                fallback = self.fallback.compute_donor_eta_batch(subset, hospital_location, hospital_id,
                                                                 departure_time, hour_of_day)
                estimates.update({d: self._relabel(e) for d, e in fallback.items()})
                continue

            for index, did in enumerate(chunk):
                seconds = durations[index][0] if index < len(durations) and durations[index] else None
                meters = distances[index][0] if index < len(distances) and distances[index] else None
                if seconds is None:
                    estimate = RouteEstimate.failed(did, hospital_id, "NO_ROUTE_FOUND", provider="osrm")
                else:
                    estimate = RouteEstimate(
                        origin_id=did, destination_id=hospital_id,
                        distance_meters=float(meters) if meters is not None else None,
                        duration_seconds=float(seconds), duration_minutes=float(seconds) / 60.0,
                        provider="osrm", is_success=True,
                    )
                self._store(self._cache_key("eta", donor_locations[did], hospital_location), estimate)
                estimates[did] = estimate
        return estimates

    def compute_detailed_route(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        origin_id: str = "ORIGIN",
        destination_id: str = "DEST",
        departure_time: Optional[datetime] = None,
    ) -> RouteEstimate:
        key = self._cache_key("route", origin, destination)
        hit = self._cached(key)
        if hit is not None:
            return RouteEstimate(**{**hit.__dict__, "origin_id": origin_id, "destination_id": destination_id})
        try:
            payload = self._get("route", f"{_coord(origin)};{_coord(destination)}",
                                {"overview": "full", "geometries": "polyline", "steps": "false"})
            route = payload["routes"][0]
            self.last_error = None
        except Exception as error:
            self.last_error = type(error).__name__
            logger.warning("OSRM route request failed (%s); using synthetic fallback.", self.last_error)
            return self._relabel(self.fallback.compute_detailed_route(
                origin, destination, origin_id, destination_id, departure_time))
        estimate = RouteEstimate(
            origin_id=origin_id, destination_id=destination_id,
            distance_meters=float(route["distance"]), duration_seconds=float(route["duration"]),
            duration_minutes=float(route["duration"]) / 60.0, provider="osrm", is_success=True,
            polyline_encoded=route.get("geometry"),
        )
        self._store(key, estimate)
        return estimate
