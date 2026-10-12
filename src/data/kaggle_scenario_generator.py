"""
src/data/kaggle_scenario_generator.py
Phase 3 — Integrated Scenario Generator for Kaggle-based Simulation

Creates reproducible scenario sets combining Kaggle donor profiles
with synthetic emergency requests, for use across all ranking policies
in subsequent comparative experiments.

DATA PROVENANCE:
  - Donor pools are sampled from the Kaggle dataset via KaggleSimulationAdapter.
  - Emergency scenarios are generated synthetically.
  - All coordinates are synthetic (see adapter for city-to-bbox mapping).
  - Response propensities are synthetic and independent of Kaggle availability.

REPRODUCIBILITY:
  - All random operations use the provided seed or deterministic sub-seeds.
  - Scenarios are serialized to JSON/CSV for exact reuse across policies.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.kaggle_simulation_adapter import (
    KaggleSimulationAdapter,
    CITY_BOUNDING_BOXES,
)
from src.data.schemas.donor_schema import DonorProfile
from src.data.synthetic_generator import (
    SyntheticEmergencyScenario,
    BLOOD_TYPES,
    _BT_PROBS,
    URGENCY_TIERS,
    URGENCY_TIER_DIST,
)
from src.ranking.udi_engine import DonorCandidate
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScenarioManifest:
    """Metadata for a generated scenario set."""
    run_id: str
    created_at: str
    seed: int
    n_donors_total: int
    n_donors_sampled: int
    n_scenarios: int
    dataset_path: str
    dataset_checksum: str
    schema_version: str = "1.0.0"
    dataset_derived_fields: list[str] = field(default_factory=list)
    synthetic_fields: list[str] = field(default_factory=list)
    city_to_region_mapping: dict[str, str] = field(default_factory=dict)
    generation_parameters: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Scenario manifest saved to %s", path)


class KaggleScenarioGenerator:
    """
    Generates integrated simulation scenarios from the Kaggle donor pool.

    Creates:
      - A donor pool sampled from the cleaned dataset
      - Emergency request scenarios with synthetic sites in donor cities
      - DonorCandidate objects with computed travel times
      - A scenario manifest with full provenance

    Args:
        data_path: Path to the cleaned Kaggle CSV.
        seed: Master random seed.
        n_donors: Number of donors to sample (None = use all).
        n_scenarios: Number of emergency scenarios to generate.
        base_speed_kmh: Base urban travel speed.
        distance_method: Distance computation method.
    """

    def __init__(
        self,
        data_path: str | Path = "data/raw/blood_donor_dataset_modeling.csv",
        seed: int = 42,
        n_donors: Optional[int] = None,
        n_scenarios: int = 20,
        base_speed_kmh: float = 30.0,
        distance_method: str = "network_grid",
    ) -> None:
        self.data_path = Path(data_path)
        self.seed = seed
        self.n_donors = n_donors
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(seed)

        self.adapter = KaggleSimulationAdapter(data_path=data_path, seed=seed)
        self.travel_engine = TravelTimeEngine(
            base_speed_kmh=base_speed_kmh,
            distance_method=distance_method,
        )

        self._profiles: Optional[list[DonorProfile]] = None
        self._scenarios: Optional[list[SyntheticEmergencyScenario]] = None
        self._manifest: Optional[ScenarioManifest] = None

    def generate(self) -> dict:
        """
        Generate the full integrated scenario set.

        Returns:
            Dict with keys: 'profiles', 'scenarios', 'manifest',
            'donor_candidates_by_scenario'.
        """
        # Step 1: Load and validate dataset
        df = self.adapter.load_and_validate()

        # Step 2: Build donor profiles
        self._profiles = self.adapter.build_simulation_profiles(
            n_donors=self.n_donors,
            sample_seed=self.seed + 1,
        )

        # Verify uniqueness of donor IDs
        donor_ids = [p.donor_id for p in self._profiles]
        if len(set(donor_ids)) != len(donor_ids):
            raise ValueError("Duplicate donor IDs detected in profile generation.")

        # Step 3: Generate emergency scenarios in donor cities
        self._scenarios = self._generate_emergency_scenarios()

        # Step 4: Build DonorCandidate objects per scenario with travel times
        candidates_by_scenario = self._build_candidates_per_scenario()

        # Step 5: Create manifest
        provenance = self.adapter.get_provenance_summary()
        self._manifest = ScenarioManifest(
            run_id=f"P3_{uuid.uuid4().hex[:8]}",
            created_at=datetime.now().isoformat(),
            seed=self.seed,
            n_donors_total=len(df),
            n_donors_sampled=len(self._profiles),
            n_scenarios=len(self._scenarios),
            dataset_path=str(self.data_path),
            dataset_checksum=self.adapter.expected_checksum,
            dataset_derived_fields=provenance["dataset_derived_fields"],
            synthetic_fields=provenance["synthetic_fields"],
            city_to_region_mapping=provenance["city_coordinate_mapping"],
            generation_parameters={
                "seed": self.seed,
                "n_donors": self.n_donors,
                "n_scenarios": self.n_scenarios,
                "base_speed_kmh": self.travel_engine.base_speed_kmh,
                "distance_method": self.travel_engine.distance_method,
                "detour_factor": self.travel_engine.detour_factor,
                "eligibility_rate": sum(1 for p in self._profiles if p.is_eligible) / len(self._profiles),
                "response_propensity_distribution": "Beta(2, 5)",
                "simulated_availability_rate": 0.70,
            },
            notes=[
                "All coordinates are SYNTHETIC — not real donor locations.",
                "Response propensities are independent of Kaggle availability.",
                "The Kaggle 'availability' label is preserved as a dataset attribute only.",
                "Phase 2 classifier near-chance (CV ROC-AUC=0.4959) — not used for response.",
                "Emergency sites are placed within city bounding boxes of the donor pool.",
            ],
        )

        logger.info(
            "Phase 3 scenario generation complete: %d profiles, %d scenarios, run_id=%s",
            len(self._profiles), len(self._scenarios), self._manifest.run_id,
        )

        return {
            "profiles": self._profiles,
            "scenarios": self._scenarios,
            "manifest": self._manifest,
            "candidates_by_scenario": candidates_by_scenario,
        }

    def _generate_emergency_scenarios(self) -> list[SyntheticEmergencyScenario]:
        """Generate emergency scenarios with sites in donor-represented cities."""
        rng = np.random.default_rng(self.seed + 100)

        # Determine which cities are represented in the donor pool
        cities_in_pool = set()
        for p in self._profiles:
            if p.kaggle_features:
                cities_in_pool.add(p.kaggle_features.city)
        cities_list = sorted(cities_in_pool)

        if not cities_list:
            raise ValueError("No cities found in donor profiles.")

        base_datetime = datetime(2024, 1, 1, 8, 0, 0)
        scenarios = []

        for i in range(self.n_scenarios):
            # Select a random city for the emergency site
            city = rng.choice(cities_list)
            bbox = CITY_BOUNDING_BOXES[city]

            site_lat = float(rng.uniform(bbox["lat_min"], bbox["lat_max"]))
            site_lon = float(rng.uniform(bbox["lon_min"], bbox["lon_max"]))
            site_id = f"SITE_{city[:3].upper()}_{i:03d}"

            urgency = str(rng.choice(URGENCY_TIERS, p=URGENCY_TIER_DIST))
            blood_type = str(rng.choice(BLOOD_TYPES, p=_BT_PROBS))

            units_range = {"CRITICAL": (1, 2), "URGENT": (1, 4), "SCHEDULED": (1, 6)}
            lo, hi = units_range[urgency]
            units = int(rng.integers(lo, hi + 1))

            offset_hours = float(rng.uniform(0, 30 * 24))
            ts = base_datetime + timedelta(hours=offset_hours)

            scenarios.append(SyntheticEmergencyScenario(
                request_id=f"KAG_REQ_{i:05d}",
                blood_type_required=blood_type,
                units_required=units,
                urgency=urgency,
                site_latitude=site_lat,
                site_longitude=site_lon,
                site_id=site_id,
                request_timestamp=ts,
                hour_of_day=ts.hour,
            ))

        tier_counts = {t: sum(1 for s in scenarios if s.urgency == t) for t in URGENCY_TIERS}
        logger.info("Emergency scenarios generated: %s", tier_counts)
        return scenarios

    def _build_candidates_per_scenario(self) -> dict[str, list[DonorCandidate]]:
        """
        For each scenario, compute travel times and build DonorCandidate objects.

        The candidates dict maps scenario request_id -> list of DonorCandidate.
        Only eligible donors with valid coordinates are included.
        """
        candidates_by_scenario = {}

        for scenario in self._scenarios:
            candidates = []
            for profile in self._profiles:
                if not profile.is_eligible:
                    continue
                if profile.latitude is None or profile.longitude is None:
                    continue

                # Compute travel time
                tt = self.travel_engine.travel_time_minutes(
                    donor_lat=profile.latitude,
                    donor_lon=profile.longitude,
                    site_lat=scenario.site_latitude,
                    site_lon=scenario.site_longitude,
                    hour_of_day=scenario.hour_of_day,
                )
                tt_val = float(tt)

                candidates.append(DonorCandidate(
                    donor_id=profile.donor_id,
                    travel_time_minutes=tt_val,
                    synthetic_response_propensity=profile.synthetic_response_propensity,
                    recent_alert_count=profile.recent_alert_count,
                    is_eligible=profile.is_eligible,
                ))

            candidates_by_scenario[scenario.request_id] = candidates

        logger.info(
            "Built candidates for %d scenarios. Mean candidates/scenario: %.0f",
            len(candidates_by_scenario),
            np.mean([len(v) for v in candidates_by_scenario.values()]),
        )
        return candidates_by_scenario

    def save_outputs(self, output_dir: str | Path = "reports/phase3_dataset_integration") -> dict[str, str]:
        """
        Save all Phase 3 outputs to the specified directory.

        Returns:
            Dict of output file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # 1. Scenario manifest
        manifest_path = output_dir / "scenario_manifest.json"
        self._manifest.save(manifest_path)
        paths["scenario_manifest"] = str(manifest_path)

        # 2. Dataset profile audit
        audit = {
            "n_profiles": len(self._profiles),
            "n_eligible": sum(1 for p in self._profiles if p.is_eligible),
            "n_ineligible": sum(1 for p in self._profiles if not p.is_eligible),
            "blood_type_distribution": {},
            "city_distribution": {},
            "provenance": self.adapter.get_provenance_summary(),
        }
        for p in self._profiles:
            bt = str(p.blood_type)
            audit["blood_type_distribution"][bt] = audit["blood_type_distribution"].get(bt, 0) + 1
            if p.kaggle_features:
                c = p.kaggle_features.city
                audit["city_distribution"][c] = audit["city_distribution"].get(c, 0) + 1

        audit_path = output_dir / "dataset_profile_audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2, default=str)
        paths["dataset_profile_audit"] = str(audit_path)

        # 3. Sample donor profiles CSV
        df_profiles = self.adapter.profiles_to_dataframe(self._profiles)
        profiles_path = output_dir / "sample_donor_profiles.csv"
        df_profiles.to_csv(profiles_path, index=False)
        paths["sample_donor_profiles"] = str(profiles_path)

        # 4. Sample emergency scenarios CSV
        scenario_records = []
        for s in self._scenarios:
            scenario_records.append({
                "request_id": s.request_id,
                "blood_type_required": s.blood_type_required,
                "units_required": s.units_required,
                "urgency": s.urgency,
                "site_latitude": s.site_latitude,
                "site_longitude": s.site_longitude,
                "site_id": s.site_id,
                "request_timestamp": s.request_timestamp.isoformat(),
                "hour_of_day": s.hour_of_day,
                "data_tag": s.data_tag,
            })
        scenarios_path = output_dir / "sample_emergency_scenarios.csv"
        pd.DataFrame(scenario_records).to_csv(scenarios_path, index=False)
        paths["sample_emergency_scenarios"] = str(scenarios_path)

        logger.info("Phase 3 outputs saved to %s", output_dir)
        return paths
