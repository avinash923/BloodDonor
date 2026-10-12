"""
src/spatial/__init__.py
Spatial package exports.
"""

from src.spatial.distance_metrics import (
    haversine_distance_km,
    euclidean_distance_km,
    network_grid_distance_km,
    pairwise_distance_matrix,
)
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.spatial.feasibility_filter import FeasibilityFilter, DonorFeasibilityResult, FeasibilityFilterResult
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
from src.spatial.routing_service import RoutingService

__all__ = [
    "haversine_distance_km",
    "euclidean_distance_km",
    "network_grid_distance_km",
    "pairwise_distance_matrix",
    "TravelTimeEngine",
    "CongestionModel",
    "FeasibilityFilter",
    "DonorFeasibilityResult",
    "FeasibilityFilterResult",
    "BaseRoutingProvider",
    "RoutingProviderType",
    "LocationCoordinate",
    "RouteEstimate",
    "TravelMode",
    "RoutingPreference",
    "SyntheticRoutingProvider",
    "GoogleMapsRoutesProvider",
    "RoutingService",
]
