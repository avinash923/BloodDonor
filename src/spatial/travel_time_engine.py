"""
src/spatial/travel_time_engine.py
Congestion-adjusted travel time computation engine.

Core formula:
    τ_ij = d_ij / (v_base × κ_traffic(t))

where:
    d_ij         = road-network distance in km (from distance_metrics)
    v_base       = base urban road speed (km/h) from config
    κ_traffic(t) = time-of-day congestion coefficient in (0, 1]

Outputs travel times in MINUTES for direct comparison against
urgency horizon T_critical (also in minutes).

DATA STREAM: SYNTHETIC_SIMULATION (spatial computations operate on
synthetic urban coordinates only in Phase 3).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.spatial.distance_metrics import (
    haversine_distance_km,
    euclidean_distance_km,
    network_grid_distance_km,
    pairwise_distance_matrix,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------
# Default speed configuration (can be overridden via spatial_config)
# -----------------------------------------------------------------
_DEFAULT_SPEED_KMH: float = 30.0          # Urban collector road default

# 24-hour congestion profile (index = hour of day 0–23)
# κ = 1.0 → free flow; lower values → congestion-induced slowdown
_DEFAULT_CONGESTION_PROFILE: list[float] = [
    1.00, 1.00, 1.00, 1.00, 0.95, 0.85,   # 00–05
    0.70, 0.55, 0.45, 0.60, 0.70, 0.75,   # 06–11
    0.75, 0.80, 0.80, 0.75, 0.60, 0.45,   # 12–17
    0.55, 0.70, 0.85, 0.90, 0.95, 1.00,   # 18–23
]


# =================================================================
# Congestion Multiplier
# =================================================================

class CongestionModel:
    """
    Retrieves the time-of-day traffic congestion multiplier κ_traffic(t).

    Args:
        hourly_profile: List of 24 float values in (0, 1], one per hour.
    """

    def __init__(self, hourly_profile: list[float] | None = None) -> None:
        self._profile = hourly_profile or _DEFAULT_CONGESTION_PROFILE
        if len(self._profile) != 24:
            raise ValueError(
                f"Congestion profile must have exactly 24 entries (got {len(self._profile)})."
            )
        if any(k <= 0 or k > 1 for k in self._profile):
            raise ValueError("All congestion multipliers must be in (0, 1].")

    def kappa(self, hour_of_day: int) -> float:
        """
        Return the congestion multiplier for a given hour.

        Args:
            hour_of_day: Integer in [0, 23].

        Returns:
            κ ∈ (0, 1].
        """
        if not 0 <= hour_of_day <= 23:
            raise ValueError(f"hour_of_day must be in [0, 23], got {hour_of_day}.")
        return self._profile[hour_of_day]

    def effective_speed_kmh(self, hour_of_day: int, base_speed_kmh: float) -> float:
        """
        Compute effective travel speed after applying congestion discount.

        v_effective = v_base × κ(t)

        Args:
            hour_of_day:    Integer in [0, 23].
            base_speed_kmh: Base free-flow urban speed in km/h.

        Returns:
            Effective speed in km/h.
        """
        return base_speed_kmh * self.kappa(hour_of_day)

    @property
    def peak_hours(self) -> list[int]:
        """Return hours with κ ≤ 0.55 (severe congestion)."""
        return [h for h, k in enumerate(self._profile) if k <= 0.55]

    @property
    def offpeak_hours(self) -> list[int]:
        """Return hours with κ ≥ 0.95 (near free-flow)."""
        return [h for h, k in enumerate(self._profile) if k >= 0.95]


# =================================================================
# Travel Time Engine
# =================================================================

class TravelTimeEngine:
    """
    Computes congestion-adjusted travel times between donor locations
    and emergency blood request sites.

    τ_ij (minutes) = (d_ij / v_effective(t)) × 60

    Args:
        base_speed_kmh:    Default urban road speed in km/h.
        distance_method:   'euclidean', 'haversine', or 'network_grid'.
        congestion_model:  CongestionModel instance (default profile used if None).
        detour_factor:     Grid circuity multiplier for 'network_grid' method.
    """

    def __init__(
        self,
        base_speed_kmh: float = _DEFAULT_SPEED_KMH,
        distance_method: str = "network_grid",
        congestion_model: CongestionModel | None = None,
        detour_factor: float = 1.35,
    ) -> None:
        if base_speed_kmh <= 0:
            raise ValueError("base_speed_kmh must be positive.")
        self.base_speed_kmh = base_speed_kmh
        self.distance_method = distance_method
        self.congestion_model = congestion_model or CongestionModel()
        self.detour_factor = detour_factor

    def travel_time_minutes(
        self,
        donor_lat: float | np.ndarray,
        donor_lon: float | np.ndarray,
        site_lat: float | np.ndarray,
        site_lon: float | np.ndarray,
        hour_of_day: int = 10,
    ) -> float | np.ndarray:
        """
        Compute travel time(s) from donor location(s) to a single site.

        Args:
            donor_lat/lon:  Donor coordinates (scalar or array).
            site_lat/lon:   Site coordinates (scalar or array).
            hour_of_day:    Request hour for congestion lookup [0, 23].

        Returns:
            Travel time in minutes (scalar or ndarray matching donor inputs).
        """
        # Compute distance
        if self.distance_method == "euclidean":
            dist_km = euclidean_distance_km(donor_lat, donor_lon, site_lat, site_lon)
        elif self.distance_method == "haversine":
            dist_km = haversine_distance_km(donor_lat, donor_lon, site_lat, site_lon)
        elif self.distance_method == "network_grid":
            dist_km = network_grid_distance_km(
                donor_lat, donor_lon, site_lat, site_lon,
                detour_factor=self.detour_factor,
            )
        else:
            raise ValueError(f"Unknown distance method: '{self.distance_method}'")

        # Effective speed accounting for congestion
        v_eff = self.congestion_model.effective_speed_kmh(hour_of_day, self.base_speed_kmh)
        if v_eff <= 0:
            raise RuntimeError(f"Effective speed is zero or negative at hour {hour_of_day}.")

        return (dist_km / v_eff) * 60.0   # convert hours → minutes

    def pairwise_travel_time_matrix(
        self,
        donor_lats: np.ndarray,
        donor_lons: np.ndarray,
        site_lats: np.ndarray,
        site_lons: np.ndarray,
        hour_of_day: int = 10,
    ) -> np.ndarray:
        """
        Compute the full (N donors × M sites) travel time matrix in minutes.

        Args:
            donor_lats/lons: Donor coordinates, shape (N,).
            site_lats/lons:  Site coordinates, shape (M,).
            hour_of_day:     Request hour for congestion lookup.

        Returns:
            Travel time matrix of shape (N, M) in minutes.
        """
        dist_matrix = pairwise_distance_matrix(
            donor_lats, donor_lons,
            site_lats, site_lons,
            method=self.distance_method,
            detour_factor=self.detour_factor,
        )
        v_eff = self.congestion_model.effective_speed_kmh(hour_of_day, self.base_speed_kmh)
        return (dist_matrix / v_eff) * 60.0

    def timeliness_score(
        self,
        travel_time_minutes: float | np.ndarray,
        t_critical_minutes: float,
    ) -> float | np.ndarray:
        """
        Compute the normalised timeliness component of the UDI:

            timeliness = 1 - (τ_ij / T_critical)

        Bounded to [0, 1]; donors arriving after T_critical return 0.

        Args:
            travel_time_minutes: Computed travel time(s) in minutes.
            t_critical_minutes:  Urgency horizon T_critical in minutes.

        Returns:
            Timeliness score(s) in [0, 1].
        """
        score = 1.0 - (np.asarray(travel_time_minutes) / t_critical_minutes)
        return np.clip(score, 0.0, 1.0)
