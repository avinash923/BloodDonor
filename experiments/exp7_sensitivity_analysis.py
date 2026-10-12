"""
experiments/exp7_sensitivity_analysis.py
EXP-7: Sensitivity Analysis

Sweeps parameters like alert fatigue decay rate (lambda) to observe
impact on performance metrics.
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
from src.evaluation.sensitivity_runner import SensitivityRunner
from src.data.synthetic_generator import SyntheticUrbanGenerator
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from experiments.exp3_udi_ranking import build_candidates, simulate_responders

logger = get_logger("exp7_sensitivity_analysis")

def run_sensitivity():
    logger.info("Running EXP-7 Sensitivity Analysis...")
    
    with open("configs/udi_weights.yaml") as f:
        cfg = yaml.safe_load(f)
        
    base_engine = UDIEngine(
        w_urgency=cfg["weights"]["w_urgency"],
        w_timeliness=cfg["weights"]["w_timeliness"],
        w_probability=cfg["weights"]["w_probability"],
        formula_mode=cfg.get("formula", {}).get("mode", "multiplicative"),
        probability_floor=cfg.get("formula", {}).get("probability_floor", 0.0),
    )
    
    runner = SensitivityRunner(base_engine)
    lambdas = cfg["sensitivity"]["lambda_fatigue_range"]
    
    # Generate 1 synthetic scenario for demonstration
    gen = SyntheticUrbanGenerator(seed=42)
    pool = gen.generate_donor_pool(n=200)
    
    # Inject some fatigue into the pool for this test
    rng = np.random.default_rng(42)
    for r in pool.records:
        r.recent_alert_count = rng.integers(0, 5)
        
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
    
    res = runner.sweep_fatigue_lambda(candidates, scenario.urgency, lambdas, responders, k=10)
    
    df = pd.DataFrame(list(res.items()), columns=["Fatigue_Lambda", "Precision@10"])
    print("\nEXP-7 SENSITIVITY RESULTS:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_sensitivity()
