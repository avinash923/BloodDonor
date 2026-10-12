"""
src/data/synthetic_generator.py
Reproducible synthetic urban donor pool and emergency scenario generator.

DATA STREAM TAG: SYNTHETIC_SIMULATION
All data produced by this module is synthetic / in-silico.
It is NEVER to be reported as real-world evidence.

Generates:
  - SyntheticDonorPool: N donors placed on a bounded urban lat/lon grid
    with assigned blood types, eligibility flags, and RFMT-derived
    response probability stubs.
  - SyntheticEmergencyScenario: M emergency blood request events with
    urgency tier, blood type, site coordinates, and request timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
BLOOD_TYPE_DISTRIBUTION = [0.36, 0.06, 0.27, 0.06, 0.03, 0.01, 0.38, 0.07]
# Normalise to sum to 1
_BT_PROBS = np.array(BLOOD_TYPE_DISTRIBUTION)
_BT_PROBS /= _BT_PROBS.sum()

URGENCY_TIERS = ["CRITICAL", "URGENT", "SCHEDULED"]
URGENCY_TIER_DIST = [0.15, 0.40, 0.45]  # realistic emergency tier ratios

DATA_TAG = "SYNTHETIC_SIMULATION"


# =================================================================
# Donor Pool
# =================================================================

@dataclass
class SyntheticDonorRecord:
    """Single synthetic donor record."""
    donor_id: str
    data_tag: str = DATA_TAG            # Immutable stream classification

    # Spatial
    latitude: float = 0.0
    longitude: float = 0.0

    # Demographics
    blood_type: str = "O+"
    age_years: int = 30

    # Behavioral proxies (RFMT-inspired, synthetic values)
    recency_months: float = 6.0         # Months since last donation
    frequency_donations: int = 3        # Total donations
    time_months: float = 24.0           # Months since first donation

    # Eligibility (8-week deferral respected)
    is_eligible: bool = True

    # Response propensity used purely for synthetic in-silico generation
    synthetic_response_propensity: float = 0.30

    # Alert fatigue counter
    recent_alert_count: int = 0


@dataclass
class SyntheticDonorPool:
    """Collection of synthetic donors with convenience accessors."""
    records: list[SyntheticDonorRecord] = field(default_factory=list)
    data_tag: str = DATA_TAG

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])

    def eligible_donors(self) -> list[SyntheticDonorRecord]:
        return [r for r in self.records if r.is_eligible]

    def filter_blood_type(self, blood_type: str) -> list[SyntheticDonorRecord]:
        compatible = _compatible_blood_types(blood_type)
        return [r for r in self.records if r.blood_type in compatible and r.is_eligible]

    @property
    def n_donors(self) -> int:
        return len(self.records)


def _compatible_blood_types(required: str) -> list[str]:
    """
    Return a list of donor blood types compatible with the required type.
    Follows universal donor / compatibility rules (simplified).
    """
    compatibility: dict[str, list[str]] = {
        "A+":  ["A+", "A-", "O+", "O-"],
        "A-":  ["A-", "O-"],
        "B+":  ["B+", "B-", "O+", "O-"],
        "B-":  ["B-", "O-"],
        "AB+": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],  # universal recipient
        "AB-": ["A-", "B-", "AB-", "O-"],
        "O+":  ["O+", "O-"],
        "O-":  ["O-"],   # universal donor
    }
    return compatibility.get(required, [required])


# =================================================================
# Emergency Scenario
# =================================================================

@dataclass
class SyntheticEmergencyScenario:
    """A single synthetic blood emergency request event."""
    request_id: str
    data_tag: str = DATA_TAG

    # Blood requirement
    blood_type_required: str = "O+"
    units_required: int = 2
    urgency: str = "URGENT"

    # Site (hospital / blood bank)
    site_latitude: float = 0.0
    site_longitude: float = 0.0
    site_id: str = "SITE_00"

    # Timing
    request_timestamp: datetime = field(default_factory=datetime.now)
    hour_of_day: int = 10              # Hour extracted for congestion lookup


# =================================================================
# Generator
# =================================================================

class SyntheticUrbanGenerator:
    """
    Generates reproducible synthetic donor pools and emergency scenarios
    on a bounded urban lat/lon grid.

    All output is tagged DATA_TAG = 'SYNTHETIC_SIMULATION'.

    Args:
        lat_min, lat_max: Latitude bounds of the simulated urban area.
        lon_min, lon_max: Longitude bounds of the simulated urban area.
        seed:             Random seed for full reproducibility.
    """

    def __init__(
        self,
        lat_min: float = 12.800,
        lat_max: float = 13.100,
        lon_min: float = 77.450,
        lon_max: float = 77.750,
        seed: int = 42,
    ) -> None:
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.rng = np.random.default_rng(seed)

    def generate_donor_pool(self, n: int = 1000) -> SyntheticDonorPool:
        """
        Generate a synthetic donor pool of N donors.

        Attributes generated per donor:
          - GPS coordinates: Uniform over bounding box
          - Blood type: Weighted by realistic ABO/Rh distribution
          - Age: Uniform integer in [18, 65]
          - Recency: Exponential (mean 8 months)
          - Frequency: Poisson(λ=4) clamped to [1, 50]
          - Eligibility: False if recency < 2 (within 8-week deferral window)
          - Base response probability: Beta(2, 5) ~ mean ≈ 0.29
          - Recent alert count: Poisson(λ=1)

        Args:
            n: Number of donors to generate.

        Returns:
            SyntheticDonorPool instance.
        """
        logger.info("[SYNTHETIC_SIMULATION] Generating %d synthetic donors ...", n)

        lats = self.rng.uniform(self.lat_min, self.lat_max, size=n)
        lons = self.rng.uniform(self.lon_min, self.lon_max, size=n)
        blood_types = self.rng.choice(BLOOD_TYPES, size=n, p=_BT_PROBS)
        ages = self.rng.integers(18, 66, size=n)
        recency = self.rng.exponential(scale=8.0, size=n)
        frequency = np.clip(self.rng.poisson(lam=4, size=n), 1, 50)
        time_months = recency + self.rng.exponential(scale=20.0, size=n)
        eligibility = recency >= 2.0   # 8-week (≈2-month) deferral rule
        propensities = self.rng.beta(a=2.0, b=5.0, size=n)
        alert_counts = self.rng.poisson(lam=1, size=n)

        records = []
        for i in range(n):
            records.append(SyntheticDonorRecord(
                donor_id=f"SYN_D_{i:05d}",
                latitude=float(lats[i]),
                longitude=float(lons[i]),
                blood_type=str(blood_types[i]),
                age_years=int(ages[i]),
                recency_months=float(recency[i]),
                frequency_donations=int(frequency[i]),
                time_months=float(time_months[i]),
                is_eligible=bool(eligibility[i]),
                synthetic_response_propensity=float(propensities[i]),
                recent_alert_count=int(alert_counts[i]),
            ))

        pool = SyntheticDonorPool(records=records)
        logger.info(
            "[SYNTHETIC_SIMULATION] Pool generated: %d donors, %d eligible (%.1f%%)",
            n, len(pool.eligible_donors()),
            100.0 * len(pool.eligible_donors()) / n,
        )
        return pool

    def generate_sites(self, n_sites: int = 5) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Generate hospital / blood bank site coordinates within the urban grid.

        Returns:
            Tuple of (site_lats, site_lons, site_ids).
        """
        lats = self.rng.uniform(self.lat_min, self.lat_max, size=n_sites)
        lons = self.rng.uniform(self.lon_min, self.lon_max, size=n_sites)
        ids = [f"SITE_{i:02d}" for i in range(n_sites)]
        return lats, lons, ids

    def generate_emergency_scenarios(
        self,
        n: int = 100,
        site_lats: np.ndarray | None = None,
        site_lons: np.ndarray | None = None,
        site_ids: list[str] | None = None,
        base_datetime: datetime | None = None,
    ) -> list[SyntheticEmergencyScenario]:
        """
        Generate N synthetic emergency blood request scenarios.

        Args:
            n:              Number of scenarios.
            site_lats/lons: Pre-generated site coordinates (generated internally if None).
            site_ids:       Site identifier strings.
            base_datetime:  Base timestamp; scenarios spread over 30-day window.

        Returns:
            List of SyntheticEmergencyScenario instances.
        """
        if site_lats is None:
            site_lats, site_lons, site_ids = self.generate_sites()

        if base_datetime is None:
            base_datetime = datetime(2024, 1, 1, 8, 0, 0)

        logger.info("[SYNTHETIC_SIMULATION] Generating %d emergency scenarios ...", n)

        scenarios = []
        for i in range(n):
            site_idx = self.rng.integers(0, len(site_lats))
            urgency = self.rng.choice(URGENCY_TIERS, p=URGENCY_TIER_DIST)
            blood_type = self.rng.choice(BLOOD_TYPES, p=_BT_PROBS)

            # Units required: CRITICAL=1-2, URGENT=1-4, SCHEDULED=1-6
            units_range = {"CRITICAL": (1, 2), "URGENT": (1, 4), "SCHEDULED": (1, 6)}
            lo, hi = units_range[urgency]
            units = int(self.rng.integers(lo, hi + 1))

            # Timestamp: random offset within 30-day window
            offset_hours = float(self.rng.uniform(0, 30 * 24))
            ts = base_datetime + timedelta(hours=offset_hours)
            hour = ts.hour

            scenarios.append(SyntheticEmergencyScenario(
                request_id=f"SYN_REQ_{i:05d}",
                blood_type_required=blood_type,
                units_required=units,
                urgency=urgency,
                site_latitude=float(site_lats[site_idx]),
                site_longitude=float(site_lons[site_idx]),
                site_id=site_ids[site_idx],
                request_timestamp=ts,
                hour_of_day=hour,
            ))

        tier_counts = {t: sum(1 for s in scenarios if s.urgency == t) for t in URGENCY_TIERS}
        logger.info("[SYNTHETIC_SIMULATION] Scenarios: %s", tier_counts)
        return scenarios
