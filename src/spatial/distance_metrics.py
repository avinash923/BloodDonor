"""
src/spatial/distance_metrics.py
Geodesic and planar distance computation utilities.

Implements three methods compared in EXP-2:
  1. Euclidean distance  — planar approximation (baseline)
  2. Haversine distance  — geodesic great-circle distance (primary)
  3. Network grid factor — Haversine scaled by Manhattan detour ratio

All functions operate on arrays for vectorized batch evaluation.
Units: distances returned in KILOMETRES.

NOTE: These functions are used by the synthetic spatial engine only.
      No real GPS donor data is processed here in Phase 3.
"""

from __future__ import annotations

import math
import numpy as np

# Earth mean radius (km) — WGS-84 approximation
_EARTH_RADIUS_KM: float = 6371.0088


# =================================================================
# 1. Euclidean Distance (planar approximation)
# =================================================================

def euclidean_distance_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Planar Euclidean approximation treating lat/lon degrees as a
    flat 2-D coordinate system scaled to kilometres.

    Approximation factor (1 degree lat ≈ 111.32 km):
      - Δlat_km = Δlat * 111.32
      - Δlon_km = Δlon * 111.32 * cos(mean_lat)

    CAUTION: Accuracy degrades at large distances (>50 km) and high
    latitudes. Use Haversine for primary analysis.

    Args:
        lat1, lon1: Origin coordinates (decimal degrees).
        lat2, lon2: Destination coordinates (decimal degrees).

    Returns:
        Distance in kilometres (scalar or ndarray).
    """
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    dlat_km = (lat2 - lat1) * 111.32
    dlon_km = (lon2 - lon1) * 111.32 * np.cos(mean_lat_rad)
    return np.sqrt(dlat_km ** 2 + dlon_km ** 2)


# =================================================================
# 2. Haversine Distance (geodesic great-circle)
# =================================================================

def haversine_distance_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Haversine formula for great-circle distance between two points
    on a sphere of radius _EARTH_RADIUS_KM.

    Formula:
        a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
        d = 2R·arcsin(√a)

    where φ = latitude, λ = longitude (both in radians).

    Args:
        lat1, lon1: Origin coordinates (decimal degrees).
        lat2, lon2: Destination coordinates (decimal degrees).

    Returns:
        Great-circle distance in kilometres (scalar or ndarray).
    """
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    # Clamp for numerical safety (floating-point rounding may push a slightly > 1)
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# =================================================================
# 3. Network Grid Distance (Haversine × Manhattan detour factor)
# =================================================================

# Empirically derived urban detour ratio for a grid-like road network.
# A value of 1.35 means actual road distance ≈ 35% longer than straight line.
# This is consistent with published urban network detour studies for
# medium-density grid cities (Circuity factor; Levinson & El-Geneidy, 2009).
_DEFAULT_DETOUR_FACTOR: float = 1.35


def network_grid_distance_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
    detour_factor: float = _DEFAULT_DETOUR_FACTOR,
) -> float | np.ndarray:
    """
    Approximate road-network distance using Haversine scaled by a
    constant Manhattan-grid detour factor.

    d_network ≈ detour_factor × d_haversine

    This provides a closer approximation to actual driving distance in
    urban grid-layout cities without requiring a full routing graph.

    Args:
        lat1, lon1:     Origin coordinates (decimal degrees).
        lat2, lon2:     Destination coordinates (decimal degrees).
        detour_factor:  Circuity multiplier (default 1.35).

    Returns:
        Estimated road distance in kilometres (scalar or ndarray).
    """
    return detour_factor * haversine_distance_km(lat1, lon1, lat2, lon2)


# =================================================================
# Batch Pairwise Distance Matrix
# =================================================================

def pairwise_distance_matrix(
    donor_lats: np.ndarray,
    donor_lons: np.ndarray,
    site_lats: np.ndarray,
    site_lons: np.ndarray,
    method: str = "haversine",
    detour_factor: float = _DEFAULT_DETOUR_FACTOR,
) -> np.ndarray:
    """
    Compute the full pairwise distance matrix between N donors and M sites.

    Args:
        donor_lats:    Array of donor latitudes, shape (N,).
        donor_lons:    Array of donor longitudes, shape (N,).
        site_lats:     Array of site latitudes, shape (M,).
        site_lons:     Array of site longitudes, shape (M,).
        method:        Distance method: 'euclidean', 'haversine', or 'network_grid'.
        detour_factor: Detour factor for network_grid method.

    Returns:
        Distance matrix of shape (N, M) in kilometres.
    """
    _methods = {
        "euclidean": euclidean_distance_km,
        "haversine": haversine_distance_km,
        "network_grid": lambda la1, lo1, la2, lo2: network_grid_distance_km(
            la1, lo1, la2, lo2, detour_factor=detour_factor
        ),
    }
    if method not in _methods:
        raise ValueError(
            f"Unknown distance method '{method}'. "
            f"Choose from: {list(_methods.keys())}"
        )

    dist_fn = _methods[method]
    n_donors = len(donor_lats)
    n_sites = len(site_lats)

    # Broadcast: donors (N,1), sites (1,M) → matrix (N,M)
    lat1 = donor_lats[:, np.newaxis]   # (N, 1)
    lon1 = donor_lons[:, np.newaxis]   # (N, 1)
    lat2 = site_lats[np.newaxis, :]    # (1, M)
    lon2 = site_lons[np.newaxis, :]    # (1, M)

    return dist_fn(lat1, lon1, lat2, lon2)


def great_circle_bearing_degrees(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute the initial bearing (azimuth) from point 1 to point 2
    on the great-circle path. Useful for directional analysis.

    Returns:
        Bearing in degrees [0, 360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0
