"""
src/spatial/feasibility_filter.py
Urgency-isochrone feasibility filter for blood donor candidate selection.

A donor i is FEASIBLE for emergency e if and only if:
    τ_ij ≤ T_critical(e) × (1 - safety_margin)

where:
    τ_ij           = congestion-adjusted travel time (minutes)
    T_critical(e)  = urgency horizon for tier e (minutes)
    safety_margin  = fraction reserved for on-site donation processing

Donors failing this hard constraint are EXCLUDED from ranking, regardless
of their response probability or UDI score.

DATA STREAM: SYNTHETIC_SIMULATION (Phase 3 spatial evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Urgency horizon map (minutes) — mirrors request_schema.py
_URGENCY_HORIZONS: dict[str, int] = {
    "CRITICAL": 60,
    "URGENT": 120,
    "SCHEDULED": 240,
}


# =================================================================
# Feasibility Result
# =================================================================

@dataclass
class DonorFeasibilityResult:
    """Outcome of feasibility evaluation for one donor-request pair."""
    donor_id: str
    donor_lat: float
    donor_lon: float
    travel_time_minutes: float
    t_critical_minutes: float
    effective_horizon_minutes: float    # T_critical × (1 - safety_margin)
    is_feasible: bool
    timeliness_score: float             # ∈ [0, 1]; 0 if infeasible


@dataclass
class FeasibilityFilterResult:
    """
    Aggregated output of the feasibility filter for a single emergency scenario.
    """
    request_id: str
    urgency: str
    hour_of_day: int
    site_lat: float
    site_lon: float
    t_critical_minutes: float
    effective_horizon_minutes: float
    n_candidates_total: int
    n_feasible: int
    feasible_results: list[DonorFeasibilityResult] = field(default_factory=list)
    infeasible_results: list[DonorFeasibilityResult] = field(default_factory=list)

    @property
    def feasibility_rate(self) -> float:
        """Fraction of candidates passing the isochrone cutoff."""
        if self.n_candidates_total == 0:
            return 0.0
        return self.n_feasible / self.n_candidates_total

    @property
    def mean_feasible_travel_time(self) -> float | None:
        if not self.feasible_results:
            return None
        return float(np.mean([r.travel_time_minutes for r in self.feasible_results]))


# =================================================================
# Feasibility Filter
# =================================================================

class FeasibilityFilter:
    """
    Applies the urgency-isochrone hard constraint to eliminate donors
    who cannot physically reach the blood collection site within the
    emergency deadline.

    Args:
        travel_time_engine:    Configured TravelTimeEngine instance.
        safety_margin_fraction: Fraction of T_critical reserved for
                                on-site processing (default 0.15).
                                Effective horizon = T_critical × (1 - margin).
        urgency_horizons:      Override urgency tier → horizon mapping (minutes).
        warn_below_pool_size:  Emit a warning if feasible pool drops below this.
    """

    def __init__(
        self,
        travel_time_engine: TravelTimeEngine | None = None,
        safety_margin_fraction: float = 0.15,
        urgency_horizons: dict[str, int] | None = None,
        warn_below_pool_size: int = 3,
    ) -> None:
        self.engine = travel_time_engine or TravelTimeEngine()
        if not 0.0 <= safety_margin_fraction < 1.0:
            raise ValueError("safety_margin_fraction must be in [0, 1).")
        self.safety_margin = safety_margin_fraction
        self.horizons = urgency_horizons or _URGENCY_HORIZONS
        self.warn_below_pool_size = warn_below_pool_size

    def effective_horizon(self, urgency: str) -> float:
        """
        Compute the effective feasibility horizon with safety margin applied.

        T_eff = T_critical × (1 - safety_margin)
        """
        t_crit = self.horizons.get(urgency)
        if t_crit is None:
            raise ValueError(
                f"Unknown urgency tier '{urgency}'. "
                f"Valid: {list(self.horizons.keys())}"
            )
        return float(t_crit) * (1.0 - self.safety_margin)

    def filter_donors(
        self,
        donor_ids: list[str],
        donor_lats: np.ndarray,
        donor_lons: np.ndarray,
        site_lat: float,
        site_lon: float,
        urgency: str,
        hour_of_day: int = 10,
        request_id: str = "REQ_000",
    ) -> FeasibilityFilterResult:
        """
        Filter donor candidates against the urgency isochrone cutoff.

        Args:
            donor_ids:   List of donor identifiers.
            donor_lats:  Donor latitude array, shape (N,).
            donor_lons:  Donor longitude array, shape (N,).
            site_lat:    Emergency site latitude.
            site_lon:    Emergency site longitude.
            urgency:     Urgency tier string ('CRITICAL', 'URGENT', 'SCHEDULED').
            hour_of_day: Hour of day for congestion lookup [0, 23].
            request_id:  Emergency request identifier for logging.

        Returns:
            FeasibilityFilterResult with feasible and infeasible donor lists.
        """
        t_crit = float(self.horizons[urgency])
        t_eff = self.effective_horizon(urgency)
        n = len(donor_ids)

        # Vectorized travel time computation
        travel_times = self.engine.travel_time_minutes(
            donor_lats, donor_lons,
            site_lat, site_lon,
            hour_of_day=hour_of_day,
        )

        timeliness = self.engine.timeliness_score(travel_times, t_crit)

        feasible, infeasible = [], []
        for i, did in enumerate(donor_ids):
            tt = float(travel_times[i])
            tl = float(timeliness[i])
            is_feasible = tt <= t_eff
            record = DonorFeasibilityResult(
                donor_id=did,
                donor_lat=float(donor_lats[i]),
                donor_lon=float(donor_lons[i]),
                travel_time_minutes=tt,
                t_critical_minutes=t_crit,
                effective_horizon_minutes=t_eff,
                is_feasible=is_feasible,
                timeliness_score=tl if is_feasible else 0.0,
            )
            (feasible if is_feasible else infeasible).append(record)

        result = FeasibilityFilterResult(
            request_id=request_id,
            urgency=urgency,
            hour_of_day=hour_of_day,
            site_lat=site_lat,
            site_lon=site_lon,
            t_critical_minutes=t_crit,
            effective_horizon_minutes=t_eff,
            n_candidates_total=n,
            n_feasible=len(feasible),
            feasible_results=feasible,
            infeasible_results=infeasible,
        )

        if len(feasible) < self.warn_below_pool_size:
            logger.warning(
                "[%s] Urgency=%s | Only %d feasible donors found (threshold=%d). "
                "Recruitment may fail.",
                request_id, urgency, len(feasible), self.warn_below_pool_size,
            )
        else:
            logger.debug(
                "[%s] Urgency=%s | Hour=%02d | Feasible: %d/%d (%.1f%%) | "
                "T_eff=%.0f min | Mean τ_feasible=%.1f min",
                request_id, urgency, hour_of_day,
                len(feasible), n, result.feasibility_rate * 100,
                t_eff,
                result.mean_feasible_travel_time or 0.0,
            )

        return result

    def batch_filter(
        self,
        donor_ids: list[str],
        donor_lats: np.ndarray,
        donor_lons: np.ndarray,
        scenarios: list[dict],
    ) -> list[FeasibilityFilterResult]:
        """
        Apply feasibility filtering across multiple emergency scenarios.

        Args:
            donor_ids/lats/lons: Donor pool.
            scenarios: List of dicts, each with keys:
                       'request_id', 'site_lat', 'site_lon',
                       'urgency', 'hour_of_day'.

        Returns:
            List of FeasibilityFilterResult, one per scenario.
        """
        results = []
        for s in scenarios:
            r = self.filter_donors(
                donor_ids=donor_ids,
                donor_lats=donor_lats,
                donor_lons=donor_lons,
                site_lat=s["site_lat"],
                site_lon=s["site_lon"],
                urgency=s["urgency"],
                hour_of_day=s.get("hour_of_day", 10),
                request_id=s.get("request_id", "REQ_?"),
            )
            results.append(r)
        return results
