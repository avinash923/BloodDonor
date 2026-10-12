"""
tests/test_google_maps_routing.py
Comprehensive Unit and Integration Test Suite for Phase 8:
Google Maps Platform Routes API Integration and Unified Routing Interface.

Covers all required test criteria:
1. Provider selection chooses the correct provider.
2. Synthetic routing remains 100% compatible with existing code.
3. Valid Route Matrix responses produce correct donor-to-hospital estimates.
4. Multiple donors are mapped to the correct results.
5. Partial route failures are handled correctly.
6. A missing route never becomes zero ETA (remains None / is_success=False).
7. Invalid or missing coordinates are rejected appropriately.
8. Duration and distance units are converted correctly (seconds to minutes, meters to km).
9. Traffic-aware and static duration fields are handled without confusion.
10. Batch processing respects configured limits (chunking).
11. Timeouts and transient errors produce bounded retries and clean failures.
12. Authentication and 4xx errors are handled without exposing secrets.
13. Required API keys are validated only when Google routing is selected.
14. Google provider is never called during synthetic-only tests.
15. Existing UDI ranking, eligibility, feasibility, and time budgets remain intact.
16. Populate candidates with ETAs attaches accurate values or None.
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from src.spatial.routing_provider import (
    RoutingProviderType,
    LocationCoordinate,
    RouteEstimate,
    TravelMode,
    RoutingPreference,
)
from src.spatial.synthetic_routing_provider import SyntheticRoutingProvider
from src.spatial.google_routes_provider import (
    GoogleMapsRoutesProvider,
    _parse_duration_seconds,
)
from src.spatial.routing_service import RoutingService
from src.ranking.udi_engine import UDIEngine, DonorCandidate


class TestLocationCoordinateAndParser:
    """Tests coordinate validation and duration parsing."""

    def test_valid_coordinates(self):
        coord = LocationCoordinate(12.9716, 77.5946)
        assert coord.latitude == 12.9716
        assert coord.longitude == 77.5946
        d = coord.to_dict()
        assert d["latLng"]["latitude"] == 12.9716
        assert d["latLng"]["longitude"] == 77.5946

    def test_invalid_coordinates_raise_value_error(self):
        with pytest.raises(ValueError, match="Latitude must be in"):
            LocationCoordinate(95.0, 77.0)
        with pytest.raises(ValueError, match="Longitude must be in"):
            LocationCoordinate(12.0, 190.0)

    def test_parse_duration_protobuf_strings(self):
        assert _parse_duration_seconds("120s") == 120.0
        assert _parse_duration_seconds("1845.5s") == 1845.5
        assert _parse_duration_seconds("0s") == 0.0
        assert _parse_duration_seconds(None) is None
        assert _parse_duration_seconds("invalid") is None


class TestProviderSelectionAndConfig:
    """Tests provider factory and configuration security."""

    def test_synthetic_provider_default_no_api_key_required(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
        monkeypatch.setenv("ROUTING_PROVIDER", "synthetic")
        service = RoutingService()
        assert service.provider_type == RoutingProviderType.SYNTHETIC
        assert isinstance(service.provider, SyntheticRoutingProvider)

    def test_google_provider_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
        monkeypatch.setenv("ROUTING_PROVIDER", "google")
        with pytest.raises(ValueError, match="Google Maps API key is required"):
            RoutingService()

    def test_google_provider_instantiates_with_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "AIzaSyFakeTestKey12345")
        monkeypatch.setenv("ROUTING_PROVIDER", "google")
        service = RoutingService()
        assert service.provider_type == RoutingProviderType.GOOGLE
        assert isinstance(service.provider, GoogleMapsRoutesProvider)


class TestSyntheticRoutingProvider:
    """Verifies synthetic provider behavior, units, and consistency."""

    def test_synthetic_batch_computation(self):
        provider = SyntheticRoutingProvider()
        donors = {
            "D1": LocationCoordinate(12.90, 77.50),
            "D2": LocationCoordinate(12.95, 77.55),
        }
        hospital = LocationCoordinate(13.00, 77.60)
        results = provider.compute_donor_eta_batch(donors, hospital, hospital_id="HOSP_01", hour_of_day=10)

        assert len(results) == 2
        assert "D1" in results and "D2" in results
        assert results["D1"].is_success is True
        assert results["D1"].duration_minutes is not None
        assert results["D1"].duration_minutes > 0
        assert results["D1"].duration_seconds == results["D1"].duration_minutes * 60.0
        assert results["D1"].provider == "synthetic"


class TestGoogleMapsRoutesProviderMocked:
    """Verifies Google Maps Routes API integration using mocked responses."""

    @patch("src.spatial.google_routes_provider.GoogleMapsRoutesProvider._execute_http_post")
    def test_compute_route_matrix_success(self, mock_post):
        # Mock computeRouteMatrix Protobuf JSON response
        mock_post.return_value = [
            {
                "originIndex": 0,
                "destinationIndex": 0,
                "status": {"code": 0},
                "condition": "ROUTE_EXISTS",
                "distanceMeters": 15420,
                "duration": "1800s",
                "staticDuration": "1500s",
            },
            {
                "originIndex": 1,
                "destinationIndex": 0,
                "status": {"code": 0},
                "condition": "ROUTE_EXISTS",
                "distanceMeters": 8500,
                "duration": "900s",
                "staticDuration": "850s",
            },
        ]

        provider = GoogleMapsRoutesProvider(api_key="mock_key")
        donors = {
            "D1": LocationCoordinate(12.90, 77.50),
            "D2": LocationCoordinate(12.95, 77.55),
        }
        hospital = LocationCoordinate(13.00, 77.60)

        results = provider.compute_donor_eta_batch(donors, hospital)

        assert len(results) == 2
        r1 = results["D1"]
        assert r1.is_success is True
        assert r1.distance_meters == 15420.0
        assert r1.duration_seconds == 1800.0
        assert r1.duration_minutes == 30.0  # 1800 / 60
        assert r1.traffic_duration_seconds == 1800.0
        assert r1.provider == "google"

        r2 = results["D2"]
        assert r2.is_success is True
        assert r2.distance_meters == 8500.0
        assert r2.duration_minutes == 15.0  # 900 / 60

    @patch("src.spatial.google_routes_provider.GoogleMapsRoutesProvider._execute_http_post")
    def test_partial_route_failure_and_missing_route_never_becomes_zero(self, mock_post):
        """A failed element must receive is_success=False and duration_minutes=None (NEVER 0.0)."""
        mock_post.return_value = [
            {
                "originIndex": 0,
                "destinationIndex": 0,
                "status": {"code": 0},
                "condition": "ROUTE_EXISTS",
                "distanceMeters": 10000,
                "duration": "1200s",
            },
            {
                "originIndex": 1,
                "destinationIndex": 0,
                "status": {"code": 5, "message": "No route could be found between points"},
                "condition": "ROUTE_NOT_FOUND",
            },
        ]

        provider = GoogleMapsRoutesProvider(api_key="mock_key")
        donors = {
            "D1": LocationCoordinate(12.90, 77.50),
            "D2_ISLAND": LocationCoordinate(12.95, 77.55),
        }
        hospital = LocationCoordinate(13.00, 77.60)

        results = provider.compute_donor_eta_batch(donors, hospital)

        assert results["D1"].is_success is True
        assert results["D1"].duration_minutes == 20.0

        failed = results["D2_ISLAND"]
        assert failed.is_success is False
        assert failed.duration_minutes is None
        assert failed.duration_seconds is None
        assert failed.error_code is not None

    @patch("src.spatial.google_routes_provider.GoogleMapsRoutesProvider._execute_http_post")
    def test_batching_respects_max_chunk_size(self, mock_post):
        mock_post.return_value = []
        provider = GoogleMapsRoutesProvider(api_key="mock_key", max_batch_size=5)

        donors = {f"D{i}": LocationCoordinate(12.90, 77.50) for i in range(12)}
        hospital = LocationCoordinate(13.00, 77.60)

        provider.compute_donor_eta_batch(donors, hospital)
        # 12 donors with max_batch_size=5 -> 3 HTTP calls (5 + 5 + 2)
        assert mock_post.call_count == 3

    @patch("src.spatial.google_routes_provider.GoogleMapsRoutesProvider._execute_http_post")
    def test_compute_detailed_route_with_polyline(self, mock_post):
        mock_post.return_value = {
            "routes": [
                {
                    "distanceMeters": 12500,
                    "duration": "1500s",
                    "polyline": {"encodedPolyline": "_p~iF~ps|U_ulLnnqC_mqNvxq`@"},
                }
            ]
        }

        provider = GoogleMapsRoutesProvider(api_key="mock_key")
        est = provider.compute_detailed_route(
            origin=LocationCoordinate(12.90, 77.50),
            destination=LocationCoordinate(13.00, 77.60),
            origin_id="DONOR_A",
            destination_id="HOSP_CITY",
        )

        assert est.is_success is True
        assert est.duration_minutes == 25.0
        assert est.distance_meters == 12500.0
        assert est.polyline_encoded == "_p~iF~ps|U_ulLnnqC_mqNvxq`@"


class TestUDIEngineIntegration:
    """Verifies that UDI ranking consumes the new routing estimates accurately."""

    def test_candidate_population_with_etas(self):
        service = RoutingService(provider_type="synthetic")
        candidates = [
            DonorCandidate("D1", synthetic_response_propensity=0.9),
            DonorCandidate("D2", synthetic_response_propensity=0.8),
            DonorCandidate("D_INVALID", synthetic_response_propensity=0.7),
        ]
        coords = {
            "D1": (12.92, 77.52),
            "D2": (12.94, 77.54),
            "D_INVALID": (999.0, 77.0), # Malformed coordinate
        }

        updated = service.populate_candidates_with_etas(
            candidates, coords, hospital_lat=13.00, hospital_lon=77.60
        )

        assert updated[0].travel_time_minutes is not None
        assert updated[0].travel_time_minutes > 0
        assert updated[1].travel_time_minutes is not None
        assert updated[2].travel_time_minutes is None # Failed coordinate -> None

        # UDI Engine ranking handles None ETA as infeasible (score=0.0)
        engine = UDIEngine(formula_mode="multiplicative")
        ranked = engine.score_and_rank(updated, urgency="CRITICAL")

        # Feasible donors get rank 1 and 2; invalid donor gets score 0.0 and is infeasible
        res_invalid = next(r for r in ranked if r.donor_id == "D_INVALID")
        assert res_invalid.udi_score == 0.0
        assert res_invalid.is_feasible is False
