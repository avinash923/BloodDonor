"""
experiments/exp9_phase3_demonstration.py
EXP-9: Phase 3 Simulator Integration Demonstration.
Runs a small set of synthetic scenarios to validate the dynamic simulator,
logging full trace events including re-ranking and fatigue updates.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticUrbanGenerator
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.ranking.udi_engine import UDIEngine
from src.simulation.response_model import StochasticResponseModel
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.simulator import SimulationEpisode
from experiments.exp4_dynamic_recruitment import build_candidates
from src.data.schemas.request_schema import _URGENCY_HORIZONS

def run_demonstration():
    print("="*65)
    print("EXP-9: Phase 3 Dynamic Recruitment Demonstration")
    print("="*65)
    
    seed = 42
    n_scenarios = 3
    n_donors = 50
    out_dir = Path("reports/phase3_simulation_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Synthetic Data
    gen = SyntheticUrbanGenerator(seed=seed)
    pool = gen.generate_donor_pool(n=n_donors)
    site_lats, site_lons, site_ids = gen.generate_sites(n_sites=1)
    scenarios = gen.generate_emergency_scenarios(n=n_scenarios, site_lats=site_lats, site_lons=site_lons, site_ids=site_ids)
    
    engine_spatial = TravelTimeEngine(distance_method="network_grid", congestion_model=CongestionModel())
    
    with open("configs/udi_weights.yaml") as f:
        cfg = yaml.safe_load(f)
        
    udi_engine = UDIEngine(
        w_urgency=cfg["weights"]["w_urgency"],
        w_timeliness=cfg["weights"]["w_timeliness"],
        w_probability=cfg["weights"]["w_probability"],
        formula_mode=cfg.get("formula", {}).get("mode", "multiplicative"),
        probability_floor=cfg.get("formula", {}).get("probability_floor", 0.0),
    )
    
    response_model = StochasticResponseModel(seed=seed)
    dispatch_policy = DispatchCascadePolicy(w1_mult=2.0, w_sub_mult=1.0)
    
    for idx, scenario in enumerate(scenarios):
        trace_file = out_dir / f"trace_{scenario.request_id}.txt"
        with open(trace_file, "w") as f:
            f.write(f"=== Scenario {scenario.request_id} ===\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Urgency: {scenario.urgency}\n")
            f.write(f"Units Required: {scenario.units_required}\n")
            f.write(f"Deadline (T_crit): {_URGENCY_HORIZONS[scenario.urgency]} mins\n\n")
            
            donor_lats = [d.latitude for d in pool.records]
            donor_lons = [d.longitude for d in pool.records]
            tt = engine_spatial.travel_time_minutes(donor_lats, donor_lons, scenario.site_latitude, scenario.site_longitude, scenario.hour_of_day)
            
            candidates, true_probs = build_candidates(pool.records, tt)
            ranked_pool = udi_engine.score_and_rank(candidates, urgency=scenario.urgency)
            feasible_pool = [r for r in ranked_pool if r.is_feasible and r.is_eligible]
            
            f.write("--- Initial Candidate Ranking ---\n")
            for i, r in enumerate(feasible_pool[:5]):
                c = next(c for c in candidates if c.donor_id == r.donor_id)
                f.write(f"{i+1}. {r.donor_id} | UDI: {r.udi_score:.4f} | T_score: {r.timeliness_component:.4f} | P: {r.probability_component:.2f} | Fatigue: {r.fatigue_discount:.2f} | Elig: {r.is_eligible}\n")
            f.write("\n")
            
            sim = SimulationEpisode(
                request=scenario,
                candidates=true_probs,
                ranked_pool=feasible_pool,
                t_critical_minutes=_URGENCY_HORIZONS[scenario.urgency],
                response_model=response_model,
                dispatch_policy=dispatch_policy,
                udi_engine=udi_engine,
                unranked_candidates=candidates
            )
            
            res = sim.run()
            
            f.write("--- Dispatch Log & Re-ranking Events ---\n")
            for ev in res.dispatch_log:
                f.write(f"Wave {ev.wave_number}: Dispatched {ev.donor_id} at {ev.dispatched_at_minutes:.1f}m\n")
                if ev.response_latency_minutes is not None:
                    resp_time = ev.dispatched_at_minutes + ev.response_latency_minutes
                    f.write(f"  -> Response at {resp_time:.1f}m: Accepted={ev.accepted}\n")
                if ev.arrival_time_minutes is not None:
                    f.write(f"  -> Arrived at {ev.arrival_time_minutes:.1f}m\n")
                f.write(f"  -> Final State: {ev.final_state}\n")
            
            f.write("\n--- Final Outcome ---\n")
            f.write(f"Quota Met: {res.quota_met}\n")
            f.write(f"Units Fulfilled: {res.units_fulfilled} / {res.units_required}\n")
            f.write(f"Time to Fulfillment: {res.time_to_fulfillment_minutes}\n")
            f.write(f"Total Waves: {res.waves_used}\n")
            f.write(f"Total Alerts Sent: {res.total_notifications_sent}\n")
            
        print(f"Scenario {scenario.request_id} simulated. Trace saved to {trace_file}")

    print("\nPhase 3 demonstration complete.")

if __name__ == "__main__":
    run_demonstration()
