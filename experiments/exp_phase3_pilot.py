"""
experiments/exp_phase3_pilot.py
Phase 3 — Integrated Pilot Run

Runs a small-scale end-to-end simulation using Kaggle-derived donor
profiles with synthetic spatial/response fields.

This script:
  1. Loads and validates the Kaggle dataset
  2. Builds 200 simulation profiles
  3. Generates 20 emergency scenarios
  4. Runs one simulation episode per scenario using UDI ranking
  5. Saves all outputs to reports/phase3_dataset_integration/

DATA STREAM: SYNTHETIC_SIMULATION
All results are synthetic in-silico experiments.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kaggle_scenario_generator import KaggleScenarioGenerator
from src.data.schemas.request_schema import EmergencyRequest, URGENCY_HORIZON_MINUTES
from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.simulation.response_model import StochasticResponseModel
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.simulator import SimulationEpisode
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("reports/phase3_dataset_integration")
DATA_PATH = Path("data/raw/blood_donor_dataset_modeling.csv")
SEED = 42
N_DONORS = 200
N_SCENARIOS = 20


def run_pilot():
    """Execute the Phase 3 pilot simulation."""
    logger.info("=" * 70)
    logger.info("Phase 3 — Kaggle Integration Pilot Run")
    logger.info("=" * 70)

    # Step 1: Generate scenario set
    gen = KaggleScenarioGenerator(
        data_path=DATA_PATH,
        seed=SEED,
        n_donors=N_DONORS,
        n_scenarios=N_SCENARIOS,
    )
    result = gen.generate()
    profiles = result["profiles"]
    scenarios = result["scenarios"]
    manifest = result["manifest"]
    candidates_by_scenario = result["candidates_by_scenario"]

    # Step 2: Initialize simulation components
    udi_engine = UDIEngine(
        formula_mode="multiplicative",
        safety_margin=0.15,
        lambda_fatigue=0.30,
    )
    response_model = StochasticResponseModel(seed=SEED + 200)
    dispatch_policy = DispatchCascadePolicy(w1_mult=3.0, w_sub_mult=2.0)

    # Step 3: Build ground-truth propensity map
    propensity_map = {}
    for p in profiles:
        propensity_map[p.donor_id] = p.synthetic_response_propensity or 0.30

    # Step 4: Run simulation episodes
    episode_results = []
    for scenario in scenarios:
        candidates = candidates_by_scenario[scenario.request_id]
        if not candidates:
            logger.warning("No candidates for %s, skipping.", scenario.request_id)
            continue

        # Build emergency request
        request = EmergencyRequest(
            request_id=scenario.request_id,
            blood_type_required=scenario.blood_type_required,
            units_required=scenario.units_required,
            urgency=scenario.urgency,
            site_latitude=scenario.site_latitude,
            site_longitude=scenario.site_longitude,
            request_timestamp=scenario.request_timestamp,
        )

        t_crit = float(URGENCY_HORIZON_MINUTES.get(scenario.urgency, 120))

        # Rank candidates
        ranked = udi_engine.score_and_rank(candidates, urgency=scenario.urgency)

        # Build propensity dict for this scenario's candidates
        scenario_propensity = {}
        for c in candidates:
            scenario_propensity[c.donor_id] = propensity_map.get(c.donor_id, 0.30)

        # Run simulation
        sim = SimulationEpisode(
            request=request,
            candidates=scenario_propensity,
            ranked_pool=ranked,
            t_critical_minutes=t_crit,
            response_model=StochasticResponseModel(seed=SEED + hash(scenario.request_id) % 10000),
            dispatch_policy=dispatch_policy,
            udi_engine=udi_engine,
            unranked_candidates=candidates,
        )

        episode = sim.run()
        episode_results.append({
            "request_id": scenario.request_id,
            "urgency": scenario.urgency,
            "blood_type_required": scenario.blood_type_required,
            "units_required": scenario.units_required,
            "units_fulfilled": episode.units_fulfilled,
            "units_on_time": episode.units_fulfilled_on_time,
            "quota_met": episode.quota_met,
            "quota_met_on_time": episode.quota_met_on_time,
            "late_arrivals": episode.late_arrivals,
            "notifications_sent": episode.total_notifications_sent,
            "waves_used": episode.waves_used,
            "time_to_fulfillment": episode.time_to_fulfillment_minutes,
            "n_candidates": len(candidates),
            "n_feasible": sum(1 for r in ranked if r.is_feasible),
        })

        logger.info(
            "  %s | %s | Q=%d | Fulfilled=%d | OnTime=%d | Late=%d | Notified=%d | Waves=%d",
            scenario.request_id, scenario.urgency,
            scenario.units_required, episode.units_fulfilled,
            episode.units_fulfilled_on_time, episode.late_arrivals,
            episode.total_notifications_sent, episode.waves_used,
        )

    # Step 5: Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save scenario generator outputs
    gen.save_outputs(output_dir=OUTPUT_DIR)

    # Save pilot results
    df_results = pd.DataFrame(episode_results)
    results_path = OUTPUT_DIR / "pilot_episode_results.csv"
    df_results.to_csv(results_path, index=False)

    # Summary statistics
    summary = {
        "n_scenarios": len(episode_results),
        "n_donors_sampled": N_DONORS,
        "quota_met_rate": float(df_results["quota_met"].mean()) if len(df_results) > 0 else 0.0,
        "quota_met_on_time_rate": float(df_results["quota_met_on_time"].mean()) if len(df_results) > 0 else 0.0,
        "mean_notifications": float(df_results["notifications_sent"].mean()) if len(df_results) > 0 else 0.0,
        "mean_waves": float(df_results["waves_used"].mean()) if len(df_results) > 0 else 0.0,
        "mean_late_arrivals": float(df_results["late_arrivals"].mean()) if len(df_results) > 0 else 0.0,
        "urgency_breakdown": df_results.groupby("urgency")["quota_met"].mean().to_dict() if len(df_results) > 0 else {},
    }

    summary_path = OUTPUT_DIR / "pilot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("Phase 3 Pilot Complete")
    logger.info("  Scenarios run: %d", len(episode_results))
    logger.info("  Quota met rate: %.1f%%", summary["quota_met_rate"] * 100)
    logger.info("  On-time rate: %.1f%%", summary["quota_met_on_time_rate"] * 100)
    logger.info("  Mean notifications: %.1f", summary["mean_notifications"])
    logger.info("  Outputs saved to: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    run_pilot()
