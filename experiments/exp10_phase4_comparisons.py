"""
experiments/exp10_phase4_comparisons.py
EXP-10: Phase 4 Full Comparative Experiments for the Emergency Blood Donor Recommendation System.

Evaluates multiple ranking policies over 500 scenarios:
- Random
- Proximity-First
- Probability-First
- Heuristic
- Additive UDI (Static)
- Multiplicative UDI (Static)
- Multiplicative UDI (Dynamic Re-ranking)
"""

from __future__ import annotations
import sys
import time
import json
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed
from src.data.synthetic_generator import SyntheticUrbanGenerator
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.ranking.udi_engine import UDIEngine
from src.ranking.baselines import get_all_baselines
from src.simulation.response_model import StochasticResponseModel
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.simulator import SimulationEpisode
from experiments.exp4_dynamic_recruitment import build_candidates
from src.data.schemas.request_schema import _URGENCY_HORIZONS


def run_phase4_experiments(n_scenarios: int = 500, n_donors: int = 1000, seed: int = 42):
    set_global_seed(seed)
    out_dir = Path("reports/phase4_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "event_traces").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    print(f"Generating {n_scenarios} scenarios and {n_donors} donors...")
    gen = SyntheticUrbanGenerator(seed=seed)
    pool = gen.generate_donor_pool(n=n_donors)
    site_lats, site_lons, site_ids = gen.generate_sites(n_sites=5)
    scenarios = gen.generate_emergency_scenarios(n=n_scenarios, site_lats=site_lats, site_lons=site_lons, site_ids=site_ids)
    
    # Save manifest
    manifest = {
        "seed": seed, "n_scenarios": n_scenarios, "n_donors": n_donors,
        "scenarios": [
            {"id": s.request_id, "urgency": s.urgency, "units_req": s.units_required} 
            for s in scenarios
        ]
    }
    with open(out_dir / "scenario_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    engine_spatial = TravelTimeEngine(distance_method="network_grid", congestion_model=CongestionModel())
    dispatch_policy = DispatchCascadePolicy(w1_mult=3.0, w_sub_mult=2.0)
    
    # Load UDI config
    with open("configs/udi_weights.yaml") as f:
        cfg = yaml.safe_load(f)
    w = cfg["weights"]

    additive_engine = UDIEngine(
        w_urgency=w["w_urgency"], w_timeliness=w["w_timeliness"], w_probability=w["w_probability"],
        formula_mode="additive"
    )
    multiplicative_engine = UDIEngine(
        w_urgency=w["w_urgency"], w_timeliness=w["w_timeliness"], w_probability=w["w_probability"],
        formula_mode="multiplicative", probability_floor=0.0
    )
    
    baselines = get_all_baselines(seed=seed)
    
    policies = list(baselines.keys()) + [
        "UDI-Additive (Static)",
        "UDI-Multiplicative (Static)",
        "UDI-Multiplicative (Dynamic)"
    ]

    all_results = []
    
    for scen_idx, scenario in enumerate(scenarios):
        # Reset stochastic response model per scenario so that policies face identical outcomes
        response_model = StochasticResponseModel(seed=seed + scen_idx)
        
        donor_lats = [d.latitude for d in pool.records]
        donor_lons = [d.longitude for d in pool.records]
        tt = engine_spatial.travel_time_minutes(
            donor_lats, donor_lons, scenario.site_latitude, scenario.site_longitude, scenario.hour_of_day
        )
        candidates, true_probs = build_candidates(pool.records, tt)
        
        t_crit = _URGENCY_HORIZONS[scenario.urgency]

        for policy_name in policies:
            # We must reset candidate alert fatigue per policy to ensure independent tracking
            cands_copy = build_candidates(pool.records, tt)[0]
            
            # Rank
            if policy_name in baselines:
                ranked_pool = baselines[policy_name].rank(cands_copy, urgency=scenario.urgency)
                dyn_engine = None
            elif policy_name == "UDI-Additive (Static)":
                ranked_pool = additive_engine.score_and_rank(cands_copy, urgency=scenario.urgency)
                dyn_engine = None
            elif policy_name == "UDI-Multiplicative (Static)":
                ranked_pool = multiplicative_engine.score_and_rank(cands_copy, urgency=scenario.urgency)
                dyn_engine = None
            elif policy_name == "UDI-Multiplicative (Dynamic)":
                ranked_pool = multiplicative_engine.score_and_rank(cands_copy, urgency=scenario.urgency)
                dyn_engine = multiplicative_engine

            feasible_pool = [r for r in ranked_pool if r.is_feasible and r.is_eligible]
            
            sim = SimulationEpisode(
                request=scenario,
                candidates=true_probs,
                ranked_pool=feasible_pool,
                t_critical_minutes=t_crit,
                response_model=response_model,
                dispatch_policy=dispatch_policy,
                udi_engine=dyn_engine,
                unranked_candidates=cands_copy if dyn_engine else None
            )
            
            res = sim.run()
            
            metrics = {
                "Scenario_ID": scenario.request_id,
                "Policy": policy_name,
                "Urgency": scenario.urgency,
                "Units_Req": scenario.units_required,
                "Quota_Met_Eventual": res.quota_met,
                "Quota_Met_Deadline": res.quota_met_on_time,
                "Time_To_Fulfill_Eventual": res.time_to_fulfillment_minutes,
                "Time_To_Fulfill_Deadline": res.time_to_on_time_fulfillment,
                "Alerts_Sent": res.total_notifications_sent,
                "Shortfall": res.shortfall,
                "Waves": res.waves_used,
                "Late_Arrivals": res.late_arrivals,
                "Units_Fulfilled": res.units_fulfilled,
                "No_Suitable_Donors": len(feasible_pool) == 0
            }
            all_results.append(metrics)
            
            # Save Trace (only for the first 10 scenarios to avoid massive IO)
            if scen_idx < 10:
                trace_file = out_dir / "event_traces" / f"trace_{scenario.request_id}_{policy_name.replace(' ', '_')}.txt"
                with open(trace_file, "w") as f:
                    f.write(f"Policy: {policy_name}\nFinal Outcome: {metrics}\nLog:\n")
                    for ev in res.dispatch_log:
                        f.write(f"Wave {ev.wave_number}: Dispatched {ev.donor_id} at {ev.dispatched_at_minutes:.1f}m -> {ev.final_state}\n")
                        
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "episode_results.csv", index=False)
    
    # Calculate aggregate metrics
    agg = df.groupby("Policy").apply(lambda x: pd.Series({
        "Deadline_Fulfilment_Rate": x["Quota_Met_Deadline"].mean() * 100,
        "Eventual_Fulfilment_Rate": x["Quota_Met_Eventual"].mean() * 100,
        "Mean_Time_To_Fulfill_Eventual": x.loc[x["Quota_Met_Eventual"], "Time_To_Fulfill_Eventual"].mean(),
        "Median_Time_To_Fulfill_Eventual": x.loc[x["Quota_Met_Eventual"], "Time_To_Fulfill_Eventual"].median(),
        "Invitations_Per_Successful_Request": x.loc[x["Quota_Met_Eventual"], "Alerts_Sent"].sum() / max(1, x["Quota_Met_Eventual"].sum()),
        "Mean_Shortfall_Ratio": (x["Shortfall"].sum() / max(1, x["Units_Req"].sum())),
        "Alert_Fatigue_Index": x["Alerts_Sent"].sum() / max(1, x["Units_Req"].sum()),
        "Late_Arrival_Rate": (x["Late_Arrivals"].sum() / max(1, x["Units_Fulfilled"].sum())) * 100,
        "No_Suitable_Donor_Rate": x["No_Suitable_Donors"].mean() * 100,
        "Failed_Requests_Count": (~x["Quota_Met_Eventual"]).sum()
    }), include_groups=False).reset_index()
    
    agg.to_csv(out_dir / "ranking_policy_metrics.csv", index=False)
    
    # -----------------------------------------------------------------
    # Statistical Comparisons (vs UDI-Multiplicative Dynamic)
    # -----------------------------------------------------------------
    from src.evaluation.stats_testing import wilcoxon_test
    import scipy.stats as stats
    
    ref_policy = "UDI-Multiplicative (Dynamic)"
    ref_df = df[df["Policy"] == ref_policy].copy()
    if not ref_df["Scenario_ID"].is_unique:
        raise ValueError(f"Duplicate Scenario_IDs found in reference policy {ref_policy}.")
    ref_df = ref_df.sort_values("Scenario_ID").set_index("Scenario_ID")
    
    stat_results = []
    for p in policies:
        if p == ref_policy: continue
        comp_df = df[df["Policy"] == p].copy()
        if not comp_df["Scenario_ID"].is_unique:
            raise ValueError(f"Duplicate Scenario_IDs found in policy {p}.")
        comp_df = comp_df.sort_values("Scenario_ID").set_index("Scenario_ID")
        
        # 1. Paired scenario validation
        if not ref_df.index.equals(comp_df.index):
            raise ValueError(f"Mismatch in Scenario_IDs between {ref_policy} and {p}.")
        
        # 2. McNemar's Test for Quota Met (binary paired primary outcome using deadline fulfilment)
        b = ((comp_df["Quota_Met_Deadline"] == True) & (ref_df["Quota_Met_Deadline"] == False)).sum()
        c = ((comp_df["Quota_Met_Deadline"] == False) & (ref_df["Quota_Met_Deadline"] == True)).sum()
        
        n_discordant = b + c
        if n_discordant == 0:
            mcnemar_stat, mcnemar_p, mcnemar_exact_p = 0.0, 1.0, 1.0
        else:
            # Continuity-corrected chi-square
            mcnemar_stat = (abs(b - c) - 1.0)**2 / n_discordant
            mcnemar_p = stats.chi2.sf(mcnemar_stat, 1)
            
            # Exact McNemar (binomial test)
            try:
                mcnemar_exact_p = stats.binomtest(min(b, c), n_discordant, 0.5).pvalue
            except AttributeError:
                mcnemar_exact_p = stats.binom_test(min(b, c), n_discordant, 0.5)
        
        # 3. Wilcoxon for Time to Fulfill (conditional secondary analysis)
        # Only compare on scenarios where BOTH met the quota eventually
        joint_success = comp_df["Quota_Met_Eventual"] & ref_df["Quota_Met_Eventual"]
        tt_comp = comp_df.loc[joint_success, "Time_To_Fulfill_Eventual"].values
        tt_ref = ref_df.loc[joint_success, "Time_To_Fulfill_Eventual"].values
        
        # Validate finite times
        valid_mask = np.isfinite(tt_comp) & np.isfinite(tt_ref)
        tt_comp, tt_ref = tt_comp[valid_mask], tt_ref[valid_mask]
        
        if len(tt_comp) < 5 or np.all(tt_comp == tt_ref):
            w_p = np.nan
        else:
            w_res = wilcoxon_test(tt_comp, tt_ref)
            w_p = w_res.p_value
            
        stat_results.append({
            "Policy": p,
            "Baseline": ref_policy,
            "McNemar_Stat": mcnemar_stat,
            "McNemar_p_approx": mcnemar_p,
            "McNemar_p_exact": mcnemar_exact_p,
            "Wilcoxon_p_conditional": w_p,
            "Joint_Successes": joint_success.sum()
        })
        
    pd.DataFrame(stat_results).to_csv(out_dir / "pairwise_comparisons.csv", index=False)
    
    # -----------------------------------------------------------------
    # Generate Figures
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    agg.plot.bar(x="Policy", y="Deadline_Fulfilment_Rate", ax=ax, color="steelblue")
    ax.set_title("Deadline Fulfilment Rate by Policy")
    ax.set_ylabel("% Fulfilled")
    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "fulfilment_rate.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    success_df = df[df["Quota_Met_Eventual"] == True]
    success_df.boxplot(column="Time_To_Fulfill_Eventual", by="Policy", ax=ax, rot=45)
    ax.set_title("Eventual Time to Fulfilment (Successful Requests Only)")
    ax.set_ylabel("Minutes")
    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(out_dir / "figures" / "time_to_fulfill.png")
    
    # Save a simple report (tabulate not required — use plain-text tables)
    stat_df = pd.DataFrame(stat_results)
    with open(out_dir / "experiment_report.md", "w") as f:
        f.write("# Phase 4: Comparative Experiment Results\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write("```\n")
        f.write(agg.to_string(index=False))
        f.write("\n```\n")
        f.write("\n## Statistical Comparisons\n\n")
        f.write("```\n")
        f.write(stat_df.to_string(index=False))
        f.write("\n```\n")
        
    print(f"Experiments complete. Results saved to {out_dir}")

if __name__ == "__main__":
    # Due to time/compute constraints on terminal execution, we default to 10 scenarios
    # The full 500 scenarios run can be triggered by passing 500
    run_phase4_experiments(n_scenarios=500)
