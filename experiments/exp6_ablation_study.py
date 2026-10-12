"""
experiments/exp6_ablation_study.py
EXP-6: Component Ablation Study

Evaluates system performance when dropping individual UDI components.
"""

from __future__ import annotations

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.ranking.udi_engine import UDIEngine
from src.evaluation.ablation_runner import AblationRunner
from src.data.synthetic_generator import SyntheticUrbanGenerator
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from experiments.exp3_udi_ranking import build_candidates, simulate_responders

logger = get_logger("exp6_ablation_study")

def run_ablation():
    logger.info("Running EXP-6 Component Ablation...")
    
    with open("configs/udi_weights.yaml") as f:
        cfg = yaml.safe_load(f)
        
    base_engine = UDIEngine(
        w_urgency=cfg["weights"]["w_urgency"],
        w_timeliness=cfg["weights"]["w_timeliness"],
        w_probability=cfg["weights"]["w_probability"],
        formula_mode=cfg.get("formula", {}).get("mode", "multiplicative"),
        probability_floor=cfg.get("formula", {}).get("probability_floor", 0.0),
    )
    
    ablations = {
        "No_Probability": cfg["experimental_configs"]["no_probability"],
        "No_Timeliness": cfg["experimental_configs"]["no_timeliness"]
    }
    
    runner = AblationRunner(base_engine, ablations)
    
    # Generate 1 synthetic scenario for demonstration
    gen = SyntheticUrbanGenerator(seed=42)
    pool = gen.generate_donor_pool(n=200)
    site_lats, site_lons, site_ids = gen.generate_sites(n_sites=1)
    scenario = gen.generate_emergency_scenarios(n=1, site_lats=site_lats, site_lons=site_lons, site_ids=site_ids)[0]
    
    engine_spatial = TravelTimeEngine(distance_method="network_grid", congestion_model=CongestionModel())
    tt = engine_spatial.travel_time_minutes(
        np.array([d.latitude for d in pool.records]),
        np.array([d.longitude for d in pool.records]),
        scenario.site_latitude,
        scenario.site_longitude,
        scenario.hour_of_day
    )
    
    candidates = build_candidates(pool.records, tt, 0.25)
    responders = simulate_responders(candidates, 42)
    
    res = runner.evaluate_configs(candidates, scenario.urgency, responders, k=10)
    
    df = pd.DataFrame([res])
    print("\nEXP-6 ABLATION RESULTS (Precision@10):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_ablation()
