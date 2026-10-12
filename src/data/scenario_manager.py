"""
src/data/scenario_manager.py
Mechanism to save and load synthetic scenarios and donor pools to ensure
provenance and paired testing across experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.data.synthetic_generator import SyntheticDonorPool, SyntheticDonorRecord, SyntheticEmergencyScenario
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioManager:
    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_scenario_set(self, set_id: str, pool: SyntheticDonorPool, scenarios: list[SyntheticEmergencyScenario], seed: int, config: dict):
        """Save a donor pool and list of scenarios to disk."""
        pool_path = self.data_dir / f"{set_id}_pool.csv"
        scenario_path = self.data_dir / f"{set_id}_scenarios.csv"
        meta_path = self.data_dir / f"{set_id}_meta.json"

        # Save Pool
        df_pool = pool.to_dataframe()
        df_pool.to_csv(pool_path, index=False)

        # Save Scenarios
        df_scenarios = pd.DataFrame([s.__dict__ for s in scenarios])
        df_scenarios.to_csv(scenario_path, index=False)

        # Save Meta
        meta = {
            "set_id": set_id,
            "seed": seed,
            "generated_at": datetime.now().isoformat(),
            "config": config,
            "data_source_label": "SYNTHETIC_SIMULATION",
            "schema_version": "1.0"
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
            
        logger.info(f"Saved scenario set {set_id} to {self.data_dir}")

    def load_scenario_set(self, set_id: str) -> tuple[SyntheticDonorPool, list[SyntheticEmergencyScenario], dict]:
        """Load a saved donor pool and list of scenarios from disk."""
        pool_path = self.data_dir / f"{set_id}_pool.csv"
        scenario_path = self.data_dir / f"{set_id}_scenarios.csv"
        meta_path = self.data_dir / f"{set_id}_meta.json"

        if not (pool_path.exists() and scenario_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Scenario set {set_id} not found in {self.data_dir}")

        # Load Meta
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Load Pool
        df_pool = pd.read_csv(pool_path)
        records = []
        for _, row in df_pool.iterrows():
            r_dict = row.to_dict()
            # Handle possible nan values
            r_dict = {k: v if pd.notnull(v) else None for k, v in r_dict.items()}
            records.append(SyntheticDonorRecord(**r_dict))
        pool = SyntheticDonorPool(records=records)

        # Load Scenarios
        df_scenarios = pd.read_csv(scenario_path)
        df_scenarios["request_timestamp"] = pd.to_datetime(df_scenarios["request_timestamp"])
        scenarios = []
        for _, row in df_scenarios.iterrows():
            s_dict = row.to_dict()
            scenarios.append(SyntheticEmergencyScenario(**s_dict))

        logger.info(f"Loaded scenario set {set_id} from {self.data_dir}")
        return pool, scenarios, meta
