"""
experiments/exp8_udi_comparison.py
EXP-8: Additive vs. Multiplicative UDI Formulation Comparison

Compares the two formulations using the identical 500 saved scenarios
and dynamic wave simulation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.data.synthetic_generator import SyntheticUrbanGenerator
from src.data.scenario_manager import ScenarioManager
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.simulation.response_model import StochasticResponseModel
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.simulator import SimulationEpisode
from src.data.schemas.request_schema import _URGENCY_HORIZONS
import yaml

logger = get_logger("exp8_udi_comparison")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_candidates(
    pool_records: list,
    travel_times: np.ndarray,
    fallback_prob: float = 0.25,
) -> tuple[list[DonorCandidate], dict[str, float]]:
    candidates = []
    true_probs = {}
    for record, tt in zip(pool_records, travel_times):
        prob = float(np.clip(
            getattr(record, "synthetic_response_propensity", fallback_prob), 0.0, 1.0
        ))
        true_probs[record.donor_id] = prob
        candidates.append(DonorCandidate(
            donor_id=record.donor_id,
            travel_time_minutes=float(tt) if pd.notnull(tt) else None,
            synthetic_response_propensity=prob,
            recent_alert_count=int(record.recent_alert_count),
            is_eligible=bool(record.is_eligible),
        ))
    return candidates, true_probs

def run_experiment(
    n_scenarios: int = 50, # using small number for testing if needed
    n_donors: int = 500,
    seed: int = 42,
    output_dir: str = "reports",
) -> None:
    t_start = time.perf_counter()
    set_global_seed(seed)

    logger.info("=" * 65)
    logger.info("EXP-8: Additive vs Multiplicative UDI")
    logger.info("=" * 65)

    manager = ScenarioManager()
    set_id = f"EXP8_{seed}_{n_scenarios}_{n_donors}"
    
    try:
        pool, scenarios, meta = manager.load_scenario_set(set_id)
        logger.info(f"Loaded existing scenario set {set_id}")
    except FileNotFoundError:
        logger.info(f"Generating new scenario set {set_id}")
        gen = SyntheticUrbanGenerator(seed=seed)
        pool = gen.generate_donor_pool(n=n_donors)
        site_lats, site_lons, site_ids = gen.generate_sites(n_sites=5)
        scenarios = gen.generate_emergency_scenarios(
            n=n_scenarios, site_lats=site_lats, site_lons=site_lons, site_ids=site_ids,
        )
        manager.save_scenario_set(set_id, pool, scenarios, seed, {"n_scenarios": n_scenarios, "n_donors": n_donors})

    donor_lats = np.array([d.latitude for d in pool.records])
    donor_lons = np.array([d.longitude for d in pool.records])

    engine_spatial = TravelTimeEngine(
        base_speed_kmh=30.0, distance_method="network_grid", congestion_model=CongestionModel()
    )
    
    udi_additive = UDIEngine(w_urgency=0.30, w_timeliness=0.40, w_probability=0.30, formula_mode="additive")
    udi_multiplicative = UDIEngine(w_urgency=0.30, w_timeliness=0.40, w_probability=0.30, formula_mode="multiplicative")
    
    engines = {
        "Additive": udi_additive,
        "Multiplicative": udi_multiplicative
    }
    
    sim_cfg = load_config("configs/simulation_config.yaml")
    
    # We use dynamic wave policy for both
    policy = DispatchCascadePolicy(
        w1_mult=sim_cfg["dispatch"]["wave_1_multiplier"],
        w_sub_mult=sim_cfg["dispatch"]["subsequent_wave_multiplier"]
    )
    
    all_results = []
    
    for scen_idx, scenario in enumerate(scenarios):
        tt = engine_spatial.travel_time_minutes(
            donor_lats, donor_lons, scenario.site_latitude, scenario.site_longitude,
            hour_of_day=scenario.hour_of_day
        )
        
        t_crit = _URGENCY_HORIZONS[scenario.urgency]
        
        for formula_name, engine in engines.items():
            # Reset random seed per formula to guarantee identical response events
            # This ensures paired comparison uses exactly the same stochastic responses
            response_model = StochasticResponseModel(
                mu=sim_cfg["latency"]["mu"],
                sigma=sim_cfg["latency"]["sigma"],
                seed=seed + scen_idx # deterministic per scenario
            )
            
            candidates, true_probs = build_candidates(pool.records, tt)
            ranked_pool = engine.score_and_rank(candidates, urgency=scenario.urgency)
            feasible_pool = [r for r in ranked_pool if r.is_feasible and r.is_eligible]
            
            sim = SimulationEpisode(
                request=scenario,
                candidates=true_probs,
                ranked_pool=feasible_pool,
                t_critical_minutes=t_crit,
                response_model=response_model,
                dispatch_policy=policy,
                wave_timeout=sim_cfg["dispatch"]["wave_timeout_minutes"]
            )
            
            res = sim.run()
            
            all_results.append({
                "Formula": formula_name,
                "Scenario_ID": scenario.request_id,
                "Urgency": scenario.urgency,
                "Units_Req": scenario.units_required,
                "Quota_Met": res.quota_met,
                "Time_To_Fulfill": res.time_to_fulfillment_minutes if res.quota_met else np.nan,
                "Alerts_Sent": res.total_notifications_sent,
                "Invitations_Per_Fulfillment": res.total_notifications_sent / scenario.units_required if res.quota_met else np.nan,
                "Waves": res.waves_used,
                "Shortfall": res.shortfall
            })
            
        if (scen_idx + 1) % 10 == 0:
            logger.info("  ... %d/%d scenarios complete", scen_idx + 1, n_scenarios)

    df = pd.DataFrame(all_results)
    
    agg = df.groupby("Formula").agg({
        "Quota_Met": "mean",
        "Time_To_Fulfill": "mean",
        "Alerts_Sent": "mean",
        "Invitations_Per_Fulfillment": "mean",
        "Shortfall": "mean"
    }).reset_index()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_dir}/exp8_udi_comparison.csv", index=False)

    print("\n" + "=" * 65)
    print("EXP-8 RESULTS: Additive vs Multiplicative")
    print("=" * 65)
    print(agg.to_string(index=False))
    print("=" * 65)

if __name__ == "__main__":
    run_experiment()
