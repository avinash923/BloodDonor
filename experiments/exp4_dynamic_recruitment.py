"""
experiments/exp4_dynamic_recruitment.py
EXP-4: Dynamic Recruitment Simulation

Evaluates multi-wave dynamic dispatch against a single-shot mass broadcast baseline.

Evaluation Protocol:
  - 500 synthetic emergency scenarios (SYNTHETIC_SIMULATION stream)
  - Compares:
    1. Mass Broadcast (notifies all feasible donors at t=0)
    2. Dynamic Wave Dispatch (batches based on quota, re-evaluates)
  - Metrics: Time-to-Fulfillment, Notification Overhead (Alerts / Fulfilled),
             Alert Fatigue Index (donors notified unnecessarily)

DATA STREAM: SYNTHETIC_SIMULATION
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

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

logger = get_logger("exp4_dynamic_recruitment")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_candidates(
    pool_records: list,
    travel_times: np.ndarray,
    fallback_prob: float = 0.25,
) -> tuple[list[DonorCandidate], dict[str, float]]:
    """Build DonorCandidate objects and a separate dict of true probabilities."""
    candidates = []
    true_probs = {}
    for record, tt in zip(pool_records, travel_times):
        prob = float(np.clip(
            getattr(record, "synthetic_response_propensity", fallback_prob), 0.0, 1.0
        ))
        true_probs[record.donor_id] = prob
        candidates.append(DonorCandidate(
            donor_id=record.donor_id,
            travel_time_minutes=float(tt),
            synthetic_response_propensity=prob,
            recent_alert_count=int(record.recent_alert_count),
            is_eligible=bool(record.is_eligible),
        ))
    return candidates, true_probs


def run_experiment(
    n_scenarios: int = 50, # Reduced for testing, should be 500
    n_donors: int = 500,
    seed: int = 42,
    output_dir: str = "reports",
) -> None:
    t_start = time.perf_counter()
    set_global_seed(seed)

    logger.info("=" * 65)
    logger.info("EXP-4: Dynamic Recruitment Simulation")
    logger.info("Data Stream: SYNTHETIC_SIMULATION")
    logger.info("Scenarios: %d | Donors: %d | Seed: %d", n_scenarios, n_donors, seed)
    logger.info("=" * 65)

    # 1. Setup Data
    logger.info("[1/4] Setting up environment and scenarios ...")
    manager = ScenarioManager()
    set_id = f"EXP4_{seed}_{n_scenarios}_{n_donors}"
    
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
    
    udi_cfg = load_config("configs/udi_weights.yaml")
    udi_engine = UDIEngine(
        w_urgency=udi_cfg["weights"]["w_urgency"],
        w_timeliness=udi_cfg["weights"]["w_timeliness"],
        w_probability=udi_cfg["weights"]["w_probability"],
        formula_mode=udi_cfg.get("formula", {}).get("mode", "multiplicative"),
        probability_floor=udi_cfg.get("formula", {}).get("probability_floor", 0.0),
    )
    
    sim_cfg = load_config("configs/simulation_config.yaml")
    
    response_model = StochasticResponseModel(
        mu=sim_cfg["latency"]["mu"],
        sigma=sim_cfg["latency"]["sigma"],
        seed=seed
    )

    # Dispatch Policies to compare
    policies = {
        "Mass Broadcast": DispatchCascadePolicy(w1_mult=100.0, w_sub_mult=100.0), # Send to everyone
        "Dynamic Wave (Proposed)": DispatchCascadePolicy(
            w1_mult=sim_cfg["dispatch"]["wave_1_multiplier"],
            w_sub_mult=sim_cfg["dispatch"]["subsequent_wave_multiplier"]
        )
    }

    # 2. Run Simulations
    logger.info("[2/4] Running simulations ...")
    
    all_results = []
    
    for scen_idx, scenario in enumerate(scenarios):
        # Setup context
        tt = engine_spatial.travel_time_minutes(
            donor_lats, donor_lons, scenario.site_latitude, scenario.site_longitude,
            hour_of_day=scenario.hour_of_day
        )
        candidates, true_probs = build_candidates(pool.records, tt)
        
        # Rank donors once per scenario using UDI
        ranked_pool = udi_engine.score_and_rank(candidates, urgency=scenario.urgency)
        # Filter to only feasible donors for dispatch
        feasible_pool = [r for r in ranked_pool if r.is_feasible and r.is_eligible]
        
        t_crit = _URGENCY_HORIZONS[scenario.urgency]

        for policy_name, policy in policies.items():
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
                "Policy": policy_name,
                "Scenario_ID": scenario.request_id,
                "Urgency": scenario.urgency,
                "Units_Req": scenario.units_required,
                "Quota_Met": res.quota_met,
                "Time_To_Fulfill": res.time_to_fulfillment_minutes if res.quota_met else np.nan,
                "Alerts_Sent": res.total_notifications_sent,
                "Waves": res.waves_used,
                "Shortfall": res.shortfall,
                "Data_Source": "SYNTHETIC_SIMULATION",
                "Scenario_Set_ID": set_id
            })
            
        if (scen_idx + 1) % 10 == 0:
            logger.info("  ... %d/%d scenarios complete", scen_idx + 1, n_scenarios)

    df = pd.DataFrame(all_results)

    # 3. Aggregate
    logger.info("[3/4] Aggregating metrics ...")
    
    agg = df.groupby("Policy").agg({
        "Quota_Met": "mean",
        "Time_To_Fulfill": "mean",
        "Alerts_Sent": "mean",
        "Waves": "mean",
        "Shortfall": "mean"
    }).reset_index()
    
    # 4. Save
    logger.info("[4/4] Saving reports ...")
    
    tab_dir = Path(output_dir) / "tables"
    fig_dir = Path(output_dir) / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    md = "# EXP-4: Dynamic Recruitment Simulation\n\n"
    md += "> **Data Stream**: SYNTHETIC_SIMULATION\n\n"
    md += agg.to_markdown(index=False)
    (tab_dir / "exp4_simulation_summary.md").write_text(md)
    
    # Save Raw Data for EXP-5
    csv_path = Path(output_dir) / "exp4_raw_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw simulation results saved to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    df_met = df[df["Quota_Met"] == True]
    if not df_met.empty:
        df_met.boxplot(column="Alerts_Sent", by="Policy", ax=ax)
        ax.set_title("Distribution of Alerts Sent per Episode")
        ax.set_ylabel("Number of Alerts")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(fig_dir / "exp4_alerts_boxplot.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 65)
    print("EXP-4 RESULTS (SYNTHETIC_SIMULATION)")
    print("=" * 65)
    print(agg.to_string(index=False))
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP-4: Dynamic Recruitment")
    parser.add_argument("--n-scenarios", type=int, default=50) # Use 50 for speed during testing
    parser.add_argument("--n-donors", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        n_scenarios=args.n_scenarios,
        n_donors=args.n_donors,
        seed=args.seed,
        output_dir=args.output_dir,
    )
