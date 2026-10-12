"""
src/spatial/google_routes_provider.py
Google Maps Platform Routes API Routing Provider.

Implements real-time road network routing via:
1. computeRouteMatrix (https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix)
   - Batches candidate donors (up to 100 per call, API element max 625)
   - Evaluates multi-donor origin-to-destination matrices
   - Parses duration strings ("1845s" -> 30.75 minutes)
   - Checks element-level status (code == 0 / condition == ROUTE_EXISTS)
   - Distinguishes traffic-aware durations
2. computeRoutes (https://routes.googleapis.com/directions/v2:computeRoutes)
   - Single origin-destination detailed route with encoded polyline

SECURITY:
- API key read strictly from environment variable or explicit config.
- Never logs API keys, headers, or full sensitive coordinates.
- Safe failure policy: unreachable routes return is_success=False, duration_minutes=None (NEVER 0.0).

DATA STREAM: SPATIAL_ROUTING (Google Maps Routes API)
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional, Sequence

from src.spatial.routing_provider import (
    BaseRoutingProvider,
    LocationCoordinate,
    RouteEstimate,
    TravelMode,
    RoutingPreference,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROUTE_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
COMPUTE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"


def _parse_duration_seconds(duration_str: Optional[str]) -> Optional[float]:
    """
    Parse Protobuf duration string (e.g. '1234s' or '1234.5s') to float seconds.
    """
    if not duration_str:
        return None
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)s$", str(duration_str).strip())
    if match:
        return float(match.group(1))
    return None


class GoogleMapsRoutesProvider(BaseRoutingProvider):
    """
    Adapter for Google Maps Platform Routes API v2.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        travel_mode: TravelMode = TravelMode.DRIVE,
        routing_preference: RoutingPreference = RoutingPreference.TRAFFIC_AWARE,
        timeout_seconds: float = 10.0,
        max_batch_size: int = 100,
        max_retries: int = 2,
    ) -> None:
        # Read API key from parameter or environment
        key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not key or not key.strip():
            raise ValueError(
                "Google Maps API key is required when ROUTING_PROVIDER=google. "
                "Set the GOOGLE_MAPS_API_KEY environment variable or pass api_key."
            )
        self.api_key = key.strip()
        self.travel_mode = travel_mode
        self.routing_preference = routing_preference
        self.timeout_seconds = float(timeout_seconds)
        self.max_batch_size = min(int(max_batch_size), 100) # Keep well below 625 element limit
        self.max_retries = int(max_retries)

    def _execute_http_post(
        self,
        url: str,
        payload: dict,
        field_mask: str,
    ) -> list[dict] | dict:
        """
        Executes HTTP POST request with exponential backoff on transient errors.
        Does NOT leak credentials or coordinates in logs.
        """
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_data = resp.read().decode("utf-8")
                    return json.loads(resp_data)
            except urllib.error.HTTPError as e:
                last_err = e
                # 4xx errors (e.g. 400 Invalid Argument, 403 Forbidden) are client errors - do not retry
                if 400 <= e.code < 500:
                    err_msg = f"Routes API HTTP error {e.code}: {e.reason}"
                    try:
                        detail = e.read().decode("utf-8")
                        err_obj = json.loads(detail)
                        if "error" in err_obj:
                            err_msg += f" - {err_obj['error'].get('message', '')}"
                    except Exception:
                        pass
                    logger.error("Google Routes API client error: %s", err_msg)
                    raise RuntimeError(f"Google Maps API error ({e.code}): {err_msg}") from e

                # 5xx server errors - retry with backoff
                if attempt < self.max_retries:
                    sleep_time = (2 ** attempt) * 0.5
                    logger.warning("Routes API transient 5xx error (%d), retrying in %.2fs...", e.code, sleep_time)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Google Maps API server error ({e.code}) after {self.max_retries} retries.") from e

            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                if attempt < self.max_retries:
                    sleep_time = (2 ** attempt) * 0.5
                    logger.warning("Routes API network/timeout error, retrying in %.2fs...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Google Maps API network failure after {self.max_retries} retries: {str(e)}") from e

        raise RuntimeError(f"Routes API execution failed: {last_err}")

    def compute_donor_eta_batch(
        self,
        donor_locations: dict[str, LocationCoordinate],
        hospital_location: LocationCoordinate,
        hospital_id: str = "SITE_001",
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> dict[str, RouteEstimate]:
        """
        Batch compute ETAs from candidate donors to emergency hospital via computeRouteMatrix.
        """
        estimates: dict[str, RouteEstimate] = {}
        if not donor_locations:
            return estimates

        donor_ids = list(donor_locations.keys())
        dest_waypoint = {"waypoint": {"location": hospital_location.to_dict()}}

        # Process in chunks of max_batch_size
        for i in range(0, len(donor_ids), self.max_batch_size):
            chunk_ids = donor_ids[i : i + self.max_batch_size]
            origins = [
                {"waypoint": {"location": donor_locations[did].to_dict()}}
                for did in chunk_ids
            ]

            payload: dict = {
                "origins": origins,
                "destinations": [dest_waypoint],
                "travelMode": self.travel_mode.value,
                "routingPreference": self.routing_preference.value,
            }

            if departure_time is not None:
                # Format ISO 8601 UTC
                dt_utc = departure_time.astimezone(timezone.utc) if departure_time.tzinfo else departure_time
                payload["departureTime"] = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            field_mask = (
                "originIndex,destinationIndex,status,condition,distanceMeters,duration,staticDuration"
            )

            try:
                matrix_resp = self._execute_http_post(ROUTE_MATRIX_URL, payload, field_mask)
            except Exception as e:
                logger.error("Failed batch Route Matrix call for %d donors: %s", len(chunk_ids), str(e))
                for did in chunk_ids:
                    estimates[did] = RouteEstimate.failed(
                        did, hospital_id, error_code=f"BATCH_API_ERROR: {type(e).__name__}", provider="google"
                    )
                continue

            # Process response elements
            # computeRouteMatrix returns an array of element objects:
            # [{"originIndex": 0, "destinationIndex": 0, "status": {}, "condition": "ROUTE_EXISTS", ...}]
            parsed_chunk: dict[int, dict] = {}
            if isinstance(matrix_resp, list):
                for elem in matrix_resp:
                    orig_idx = elem.get("originIndex")
                    if orig_idx is not None and 0 <= orig_idx < len(chunk_ids):
                        parsed_chunk[orig_idx] = elem

            for idx, did in enumerate(chunk_ids):
                elem = parsed_chunk.get(idx)
                if not elem:
                    estimates[did] = RouteEstimate.failed(
                        did, hospital_id, error_code="MISSING_ELEMENT_IN_RESPONSE", provider="google"
                    )
                    continue

                # Check status and condition
                status = elem.get("status", {})
                condition = elem.get("condition", "")
                status_code = status.get("code", 0)

                if status_code != 0 or condition == "ROUTE_NOT_FOUND":
                    err_msg = status.get("message") or condition or "ROUTE_FAILED"
                    estimates[did] = RouteEstimate.failed(
                        did, hospital_id, error_code=err_msg, provider="google"
                    )
                    continue

                # Parse distance and durations
                dist_m = elem.get("distanceMeters")
                dur_str = elem.get("duration")
                static_dur_str = elem.get("staticDuration")

                dur_sec = _parse_duration_seconds(dur_str)
                static_dur_sec = _parse_duration_seconds(static_dur_str)

                if dur_sec is not None and dur_sec >= 0:
                    dur_min = float(dur_sec / 60.0)
                    estimates[did] = RouteEstimate(
                        origin_id=did,
                        destination_id=hospital_id,
                        distance_meters=float(dist_m) if dist_m is not None else None,
                        duration_seconds=dur_sec,
                        duration_minutes=dur_min,
                        traffic_duration_seconds=dur_sec if self.routing_preference != RoutingPreference.TRAFFIC_UNAWARE else static_dur_sec,
                        provider="google",
                        is_success=True,
                    )
                else:
                    estimates[did] = RouteEstimate.failed(
                        did, hospital_id, error_code="UNPARSEABLE_OR_NEGATIVE_DURATION", provider="google"
                    )

        return estimates

    def get_travel_time_minutes(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        departure_time: Optional[datetime] = None,
        hour_of_day: Optional[int] = None,
    ) -> Optional[float]:
        """Compute single travel time via compute_donor_eta_batch."""
        res = self.compute_donor_eta_batch(
            donor_locations={"SINGLE": origin},
            hospital_location=destination,
            departure_time=departure_time,
            hour_of_day=hour_of_day,
        )
        est = res.get("SINGLE")
        if est and est.is_success and est.duration_minutes is not None:
            return est.duration_minutes
        return None

    def compute_detailed_route(
        self,
        origin: LocationCoordinate,
        destination: LocationCoordinate,
        origin_id: str = "ORIGIN",
        destination_id: str = "DEST",
        departure_time: Optional[datetime] = None,
    ) -> RouteEstimate:
        """
        Compute detailed route via computeRoutes (including encoded polyline for map display).
        """
        payload = {
            "origin": {"location": origin.to_dict()},
            "destination": {"location": destination.to_dict()},
            "travelMode": self.travel_mode.value,
            "routingPreference": self.routing_preference.value,
        }
        if departure_time is not None:
            dt_utc = departure_time.astimezone(timezone.utc) if departure_time.tzinfo else departure_time
            payload["departureTime"] = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        field_mask = "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"

        try:
            resp = self._execute_http_post(COMPUTE_ROUTES_URL, payload, field_mask)
        except Exception as e:
            logger.error("computeRoutes failed: %s", str(e))
            return RouteEstimate.failed(origin_id, destination_id, f"API_ERROR: {type(e).__name__}", provider="google")

        routes = resp.get("routes", []) if isinstance(resp, dict) else []
        if not routes:
            return RouteEstimate.failed(origin_id, destination_id, "NO_ROUTE_FOUND", provider="google")

        first_route = routes[0]
        dur_str = first_route.get("duration")
        dist_m = first_route.get("distanceMeters")
        polyline = first_route.get("polyline", {}).get("encodedPolyline")

        dur_sec = _parse_duration_seconds(dur_str)
        if dur_sec is not None:
            return RouteEstimate(
                origin_id=origin_id,
                destination_id=destination_id,
                distance_meters=float(dist_m) if dist_m is not None else None,
                duration_seconds=dur_sec,
                duration_minutes=float(dur_sec / 60.0),
                traffic_duration_seconds=dur_sec,
                provider="google",
                is_success=True,
                polyline_encoded=polyline,
            )
        return RouteEstimate.failed(origin_id, destination_id, "INVALID_DURATION", provider="google")
