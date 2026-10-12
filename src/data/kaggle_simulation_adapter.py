"""
src/data/kaggle_simulation_adapter.py
Phase 3 — Kaggle Dataset to Simulation Adapter

Converts cleaned Kaggle donor profiles into simulation-ready DonorProfile
objects compatible with the existing schemas, ranking engine, and simulator.

DATA PROVENANCE:
  - Dataset-derived fields: city, blood_group, months_since_first_donation,
    number_of_donation, pints_donated, availability (historical label).
  - Synthetic fields: donor_id, latitude, longitude, is_eligible,
    synthetic_response_propensity, recent_alert_count,
    simulated_current_availability.

IMPORTANT:
  - The source `availability` field is a historical classification label.
    It does NOT represent current operational availability or emergency
    response probability. Phase 2 confirmed near-chance classifier
    performance (CV ROC-AUC = 0.4959, Test ROC-AUC = 0.4731).
  - Synthetic coordinates are generated per-city using representative
    bounding boxes. They are NOT real donor locations.
  - synthetic_response_propensity is drawn from Beta(2, 5) and is
    independent of the Kaggle availability label.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.kaggle_loader import load_kaggle_dataframe
from src.data.schemas.donor_schema import (
    DonorProfile,
    DonorStream,
    BloodType,
    KaggleFeatures,
)
from src.data.synthetic_generator import (
    SyntheticUrbanGenerator,
    SyntheticEmergencyScenario,
    BLOOD_TYPES,
    _BT_PROBS,
    _compatible_blood_types,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------
# City-to-synthetic-region mapping (Australian capital cities)
# These are approximate urban bounding boxes. Coordinates are
# SYNTHETIC and do NOT represent actual donor locations.
# ---------------------------------------------------------------
CITY_BOUNDING_BOXES: dict[str, dict[str, float]] = {
    "Sydney":    {"lat_min": -33.950, "lat_max": -33.800, "lon_min": 151.100, "lon_max": 151.300},
    "Melbourne": {"lat_min": -37.850, "lat_max": -37.700, "lon_min": 144.900, "lon_max": 145.050},
    "Brisbane":  {"lat_min": -27.520, "lat_max": -27.400, "lon_min": 153.000, "lon_max": 153.150},
    "Perth":     {"lat_min": -32.000, "lat_max": -31.850, "lon_min": 115.800, "lon_max": 115.950},
    "Adelaide":  {"lat_min": -35.000, "lat_max": -34.850, "lon_min": 138.550, "lon_max": 138.700},
    "Hobart":    {"lat_min": -42.920, "lat_max": -42.830, "lon_min": 147.280, "lon_max": 147.380},
    "Darwin":    {"lat_min": -12.480, "lat_max": -12.400, "lon_min": 130.830, "lon_max": 130.910},
    "Canberra":  {"lat_min": -35.330, "lat_max": -35.220, "lon_min": 149.060, "lon_max": 149.180},
}

# Map dataset blood_group strings to BloodType enum
_BLOOD_GROUP_MAP: dict[str, BloodType] = {
    "A+": BloodType.A_POS, "A-": BloodType.A_NEG,
    "B+": BloodType.B_POS, "B-": BloodType.B_NEG,
    "AB+": BloodType.AB_POS, "AB-": BloodType.AB_NEG,
    "O+": BloodType.O_POS, "O-": BloodType.O_NEG,
}

# Expected dataset columns (must match kaggle_loader expectations)
_REQUIRED_COLS = {"city", "blood_group", "months_since_first_donation",
                  "number_of_donation", "pints_donated", "availability"}

# PII / forbidden columns — adapter must never expose these
_FORBIDDEN_COLS = {"name", "email", "password", "contact_number", "donor_id", "created_at"}


class KaggleSimulationAdapter:
    """
    Converts cleaned Kaggle donor profiles to simulation-ready DonorProfile
    objects with synthetic spatial and behavioural fields.

    Args:
        data_path: Path to blood_donor_dataset_modeling.csv.
        seed: Master random seed for all synthetic generation.
        expected_checksum: SHA256 checksum for dataset validation.
    """

    def __init__(
        self,
        data_path: str | Path = "data/raw/blood_donor_dataset_modeling.csv",
        seed: int = 42,
        expected_checksum: str = "d3d1a7e301de160b547816e41fcf61880817e5125bbc4b4364e49adcc3eb02a3",
    ) -> None:
        self.data_path = Path(data_path)
        self.seed = seed
        self.expected_checksum = expected_checksum
        self.rng = np.random.default_rng(seed)
        self._df: Optional[pd.DataFrame] = None
        self._profiles: Optional[list[DonorProfile]] = None

    def load_and_validate(self) -> pd.DataFrame:
        """Load the dataset, validate checksum and columns, return DataFrame."""
        # Checksum validation
        raw_bytes = self.data_path.read_bytes()
        actual_checksum = hashlib.sha256(raw_bytes).hexdigest()
        if actual_checksum != self.expected_checksum:
            raise ValueError(
                f"Dataset checksum mismatch.\n"
                f"  Expected: {self.expected_checksum}\n"
                f"  Actual:   {actual_checksum}"
            )
        logger.info("Dataset checksum verified: %s", actual_checksum[:16] + "...")

        # Load via existing loader (applies column and target validation)
        self._df = load_kaggle_dataframe(self.data_path)

        # Double-check no forbidden columns leaked
        forbidden_found = _FORBIDDEN_COLS & set(self._df.columns)
        if forbidden_found:
            raise ValueError(f"Forbidden PII columns found: {forbidden_found}")

        required_missing = _REQUIRED_COLS - set(self._df.columns)
        if required_missing:
            raise ValueError(f"Required columns missing: {required_missing}")

        logger.info(
            "Dataset loaded: %d rows, %d columns. Availability: 1=%d, 0=%d",
            len(self._df), len(self._df.columns),
            int((self._df["availability"] == 1).sum()),
            int((self._df["availability"] == 0).sum()),
        )
        return self._df

    def build_simulation_profiles(
        self,
        n_donors: Optional[int] = None,
        sample_seed: Optional[int] = None,
    ) -> list[DonorProfile]:
        """
        Convert dataset rows into DonorProfile objects for simulation.

        Args:
            n_donors: If specified, sample this many rows (without replacement).
                      If None, use the full dataset.
            sample_seed: Seed for sampling. Defaults to self.seed + 1.

        Returns:
            List of DonorProfile with synthetic spatial and propensity fields.
        """
        if self._df is None:
            self.load_and_validate()

        df = self._df.copy()

        # Sample if requested
        if n_donors is not None and n_donors < len(df):
            ss = sample_seed if sample_seed is not None else (self.seed + 1)
            df = df.sample(n=n_donors, random_state=ss, replace=False).reset_index(drop=True)
            logger.info("Sampled %d donors from %d total.", n_donors, len(self._df))

        profiles = []
        rng = np.random.default_rng(self.seed)

        for idx, row in df.iterrows():
            # --- Synthetic donor ID (no dataset identifiers) ---
            donor_id = f"KAG_D_{idx:05d}"

            # --- Map blood group ---
            bg_str = str(row["blood_group"]).strip()
            blood_type = _BLOOD_GROUP_MAP.get(bg_str, BloodType.UNKNOWN)

            # --- Synthetic coordinates from city bounding box ---
            city = str(row["city"]).strip()
            bbox = CITY_BOUNDING_BOXES.get(city)
            if bbox is not None:
                lat = rng.uniform(bbox["lat_min"], bbox["lat_max"])
                lon = rng.uniform(bbox["lon_min"], bbox["lon_max"])
            else:
                # Fallback: use Sydney as default with a warning
                logger.warning("Unknown city '%s', using Sydney bounding box.", city)
                bbox = CITY_BOUNDING_BOXES["Sydney"]
                lat = rng.uniform(bbox["lat_min"], bbox["lat_max"])
                lon = rng.uniform(bbox["lon_min"], bbox["lon_max"])

            # --- Kaggle-derived features (preserved verbatim) ---
            kaggle_features = KaggleFeatures(
                city=city,
                months_since_first_donation=int(row["months_since_first_donation"]),
                number_of_donation=int(row["number_of_donation"]),
                pints_donated=int(row["pints_donated"]),
                availability=int(row["availability"]),
            )

            # --- Synthetic eligibility ---
            # Use 8-week deferral heuristic: if months_since_first_donation < 2
            # and they have donated before, they might be in deferral.
            # For simplicity, generate probabilistically (90% eligible).
            is_eligible = bool(rng.random() < 0.90)

            # --- Synthetic response propensity ---
            # Beta(2, 5) distribution ~ mean ≈ 0.29, independent of
            # the Kaggle availability label. This is used ONLY as
            # simulation ground truth, never as a ranking input.
            synthetic_propensity = float(rng.beta(a=2.0, b=5.0))

            # --- Synthetic current availability ---
            # Generated independently from historical availability.
            # 70% chance of being currently available in any given scenario.
            simulated_availability = int(rng.random() < 0.70)

            # --- Synthetic alert fatigue ---
            alert_count = int(rng.poisson(lam=1))

            profile = DonorProfile(
                donor_id=donor_id,
                stream=DonorStream.KAGGLE_PILOT,
                kaggle_features=kaggle_features,
                latitude=float(lat),
                longitude=float(lon),
                blood_type=blood_type,
                is_eligible=is_eligible,
                recent_alert_count=alert_count,
                synthetic_response_propensity=synthetic_propensity,
                simulated_current_availability=simulated_availability,
                # Explicitly NOT setting response_probability or
                # donation_behavior_score — those require validated models.
            )
            profiles.append(profile)

        self._profiles = profiles
        logger.info(
            "Built %d simulation profiles. Eligible: %d (%.1f%%)",
            len(profiles),
            sum(1 for p in profiles if p.is_eligible),
            100.0 * sum(1 for p in profiles if p.is_eligible) / len(profiles),
        )
        return profiles

    def profiles_to_dataframe(self, profiles: Optional[list[DonorProfile]] = None) -> pd.DataFrame:
        """
        Export profiles to a DataFrame with explicit provenance columns.

        Columns are prefixed:
          - dataset_*  : derived from Kaggle CSV
          - synthetic_* : generated for simulation
        """
        if profiles is None:
            profiles = self._profiles
        if profiles is None:
            raise ValueError("No profiles built. Call build_simulation_profiles() first.")

        records = []
        for p in profiles:
            rec = {
                # Synthetic identifier
                "donor_id": p.donor_id,
                "stream": p.stream,
                # Dataset-derived
                "dataset_city": p.kaggle_features.city if p.kaggle_features else None,
                "dataset_blood_group": p.blood_type,
                "dataset_months_since_first_donation": (
                    p.kaggle_features.months_since_first_donation if p.kaggle_features else None
                ),
                "dataset_number_of_donation": (
                    p.kaggle_features.number_of_donation if p.kaggle_features else None
                ),
                "dataset_pints_donated": (
                    p.kaggle_features.pints_donated if p.kaggle_features else None
                ),
                "dataset_historical_availability": (
                    p.kaggle_features.availability if p.kaggle_features else None
                ),
                # Synthetic spatial
                "synthetic_latitude": p.latitude,
                "synthetic_longitude": p.longitude,
                # Synthetic operational
                "synthetic_is_eligible": p.is_eligible,
                "synthetic_response_propensity": p.synthetic_response_propensity,
                "synthetic_current_availability": p.simulated_current_availability,
                "synthetic_alert_count": p.recent_alert_count,
            }
            records.append(rec)
        return pd.DataFrame(records)

    def get_provenance_summary(self) -> dict:
        """Return a summary of field provenance for documentation."""
        return {
            "dataset_derived_fields": [
                "city", "blood_group", "months_since_first_donation",
                "number_of_donation", "pints_donated", "availability (historical label)",
            ],
            "synthetic_fields": [
                "donor_id", "latitude", "longitude", "is_eligible",
                "synthetic_response_propensity", "simulated_current_availability",
                "recent_alert_count",
            ],
            "excluded_pii_fields": list(_FORBIDDEN_COLS),
            "city_coordinate_mapping": {
                city: f"lat:[{bb['lat_min']},{bb['lat_max']}], lon:[{bb['lon_min']},{bb['lon_max']}]"
                for city, bb in CITY_BOUNDING_BOXES.items()
            },
            "availability_semantics": (
                "The source 'availability' field is a historical general-willingness label "
                "from the Kaggle dataset. It does NOT represent current operational availability "
                "or time-pressured emergency recruitment probability. "
                "Phase 2 classifier achieved near-chance performance (CV ROC-AUC=0.4959, "
                "Test ROC-AUC=0.4731), confirming limited predictive discrimination. "
                "A separate 'simulated_current_availability' field is generated synthetically "
                "for each simulation scenario."
            ),
            "response_propensity_semantics": (
                "synthetic_response_propensity is drawn from Beta(2,5), independent of "
                "the Kaggle availability label. It is used ONLY as hidden simulation "
                "ground truth for the StochasticResponseModel, never as a ranking input."
            ),
        }
