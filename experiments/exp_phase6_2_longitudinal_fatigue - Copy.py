"""
experiments/exp_phase6_2_longitudinal_fatigue.py
Phase 6.2 — Longitudinal Multi-Episode Fatigue Simulation.

Project: AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System

PRIMARY RESEARCH QUESTION:
Across repeated emergency incidents over time, does fatigue-aware dynamic UDI ranking
reduce notification burden per donor, and what trade-off, if any, does it introduce
in deadline fulfillment?

EXPERIMENTAL DESIGN:
- T independent simulation timelines (e.g., T=20 timelines of 30 days each).
- Persistent donor population (N=2,000 sampled Kaggle donor profiles per timeline).
- Chronologically ordered sequence of synthetic emergency incidents per timeline.
- Matched comparison between:
    * Arm A: Fatigue-Aware Dynamic UDI (uses persistent recent_alert_count with lambda=0.30)
    * Arm B: Fatigue-Disabled Dynamic UDI (fatigue discount fixed to 1.0, lambda=0.0)
- Matched request schedules, site locations, blood demands, and latent stochastic response draws.
- Sensitivity grid:
    * Fatigue decay lambda in {0.00, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60}
    * Lookback window in {7, 14, 30} days
- Cluster-level statistical inference using timelines as independent clusters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kaggle_simulation_adapter import KaggleSimulationAdapter, CITY_BOUNDING_BOXES
from src.data.synthetic_generator import (
    SyntheticEmergencyScenario,
    BLOOD_TYPES,
    _BT_PROBS,
    _compatible_blood_types,
    URGENCY_TIERS,
    URGENCY_TIER_DIST,
)
from src.data.schemas.request_schema import EmergencyRequest, _URGENCY_HORIZONS
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.ranking.udi_engine import UDIEngine, DonorCandidate, UDIResult
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.response_model import StochasticResponseModel
from src.simulation.simulator import SimulationEpisode
from src.simulation.longitudinal_tracker import LongitudinalDonorTracker, calculate_gini
from src.evaluation.stats_testing import wilcoxon_test, bootstrap_ci, paired_ttest
from src.utils.logger import get_logger

logger = get_logger("exp_phase6_2_longitudinal_fatigue")

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "blood_donor_dataset_modeling.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase6_2_longitudinal_fatigue"

DEFAULT_SEED = 42
DEFAULT_N_TIMELINES = 20
DEFAULT_HORIZON_DAYS = 30.0
DEFAULT_EMERGENCIES_PER_DAY = 2.5  # ~75 emergencies per timeline
DEFAULT_N_DONORS = 2000
DEFAULT_LAMBDA = 0.30
DEFAULT_LOOKBACK_DAYS = 14.0

SENSITIVITY_LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
SENSITIVITY_WINDOWS = [7.0, 14.0, 30.0]

ARM_A_NAME = "Arm A (Fatigue-Aware UDI)"
ARM_B_NAME = "Arm B (Fatigue-Disabled UDI)"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (FWER control)."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (n - rank) * float(p_values[idx]))
        running = max(running, adj)
        adjusted[idx] = running
    return [float(x) for x in adjusted]


def _boxplot(ax, data, tick_labels):
    """Matplotlib boxplot compatible with labels= and tick_labels=."""
    try:
        ax.boxplot(data, tick_labels=tick_labels)
    except TypeError:
        ax.boxplot(data, labels=tick_labels)


def generate_longitudinal_timeline_scenarios(
    timeline_id: int,
    horizon_days: float,
    rate_per_day: float,
    profiles: list,
    seed: int,
    start_datetime: datetime = datetime(2024, 1, 1, 8, 0, 0),
) -> list[SyntheticEmergencyScenario]:
    """
    Generate an arrival sequence of synthetic emergency scenarios chronologically
    ordered across the timeline horizon using an exponential inter-arrival process.
    """
    rng = np.random.default_rng(seed)
    cities_in_pool = sorted(set(p.kaggle_features.city for p in profiles if p.kaggle_features))

    scenarios: list[SyntheticEmergencyScenario] = []
    current_time_hours = 0.0
    horizon_hours = horizon_days * 24.0
    mean_inter_arrival_hours = 24.0 / rate_per_day

    req_idx = 0
    while True:
        delta_h = float(rng.exponential(scale=mean_inter_arrival_hours))
        current_time_hours += delta_h
        if current_time_hours >= horizon_hours:
            break

        city = str(rng.choice(cities_in_pool))
        bbox = CITY_BOUNDING_BOXES[city]
        site_lat = float(rng.uniform(bbox["lat_min"], bbox["lat_max"]))
        site_lon = float(rng.uniform(bbox["lon_min"], bbox["lon_max"]))
        site_id = f"SITE_{city[:3].upper()}_{req_idx:04d}"

        urgency = str(rng.choice(URGENCY_TIERS, p=URGENCY_TIER_DIST))
        blood_type = str(rng.choice(BLOOD_TYPES, p=_BT_PROBS))

        units_range = {"CRITICAL": (1, 2), "URGENT": (1, 4), "SCHEDULED": (1, 6)}
        lo, hi = units_range[urgency]
        units = int(rng.integers(lo, hi + 1))

        ts = start_datetime + timedelta(hours=current_time_hours)

        scenarios.append(SyntheticEmergencyScenario(
            request_id=f"LONG_{timeline_id:02d}_{req_idx:04d}",
            blood_type_required=blood_type,
            units_required=units,
            urgency=urgency,
            site_latitude=site_lat,
            site_longitude=site_lon,
            site_id=site_id,
            request_timestamp=ts,
            hour_of_day=int(ts.hour),
        ))
        req_idx += 1

    return scenarios


def run_timeline_for_arm(
    timeline_id: int,
    arm_name: str,
    lambda_fatigue: float,
    lookback_days: float,
    profiles: list,
    scenarios: list[SyntheticEmergencyScenario],
    travel_engine: TravelTimeEngine,
    dispatch_policy: DispatchCascadePolicy,
    weights_config: dict,
    seed: int,
) -> tuple[LongitudinalDonorTracker, list[dict]]:
    """
    Simulate a complete 30-day timeline under a specific policy arm.
    """
    donor_lats = np.array([p.latitude for p in profiles], dtype=float)
    donor_lons = np.array([p.longitude for p in profiles], dtype=float)
    prof_map = {p.donor_id: p for p in profiles}
    true_probs = {p.donor_id: p.synthetic_response_propensity for p in profiles}

    tracker = LongitudinalDonorTracker(lookback_days=lookback_days)

    w = weights_config["weights"]
    udi_engine = UDIEngine(
        w_urgency=w["w_urgency"],
        w_timeliness=w["w_timeliness"],
        w_probability=w["w_probability"],
        lambda_fatigue=lambda_fatigue,
        formula_mode="multiplicative",
        probability_floor=0.0,
    )

    episode_records = []

    for s_idx, scenario in enumerate(scenarios):
        # 1. Travel times from precomputed engine for this scenario
        tt_arr = travel_engine.travel_time_minutes(
            donor_lats,
            donor_lons,
            scenario.site_latitude,
            scenario.site_longitude,
            scenario.hour_of_day,
        )

        compat_types = set(_compatible_blood_types(scenario.blood_type_required))

        # 2. Build candidate pool with current persistent recent_alert_count
        candidates: list[DonorCandidate] = []
        compat_candidates: list[DonorCandidate] = []

        for p, tt in zip(profiles, tt_arr):
            tt_val = float(tt) if np.isfinite(tt) else None
            # Compute recent alerts strictly up to this emergency's timestamp
            n_recent = tracker.get_recent_alert_count(p.donor_id, scenario.request_timestamp)
            cand = DonorCandidate(
                donor_id=p.donor_id,
                travel_time_minutes=tt_val,
                synthetic_response_propensity=p.synthetic_response_propensity,
                recent_alert_count=n_recent,
                is_eligible=bool(p.is_eligible),
            )
            candidates.append(cand)
            if cand.is_eligible and str(p.blood_type) in compat_types:
                compat_candidates.append(cand)

        # 3. Seeded shared response model for this emergency
        # Seed depends strictly on timeline_id and s_idx, ensuring identical variates between arms
        scenario_seed = seed + timeline_id * 10000 + s_idx * 10
        response_model = StochasticResponseModel(seed=scenario_seed)
        response_model.pregenerate_donor_variates([c.donor_id for c in compat_candidates])

        t_crit = float(_URGENCY_HORIZONS[scenario.urgency])
        req = EmergencyRequest(
            request_id=scenario.request_id,
            blood_type_required=scenario.blood_type_required,
            units_required=scenario.units_required,
            urgency=scenario.urgency,
            site_latitude=scenario.site_latitude,
            site_longitude=scenario.site_longitude,
            request_timestamp=scenario.request_timestamp,
        )

        # Initial ranking
        ranked_pool = udi_engine.score_and_rank(compat_candidates, urgency=scenario.urgency)
        feasible_pool = [r for r in ranked_pool if r.is_feasible and r.is_eligible]

        # Simulation episode with dynamic re-ranking enabled
        sim = SimulationEpisode(
            request=req,
            candidates=true_probs,
            ranked_pool=feasible_pool,
            t_critical_minutes=t_crit,
            response_model=response_model,
            dispatch_policy=dispatch_policy,
            udi_engine=udi_engine,
            unranked_candidates=compat_candidates,
        )

        res = sim.run()

        # Update persistent longitudinal tracker with actual dispatched notifications
        for ev in res.dispatch_log:
            tracker.record_notification(
                donor_id=ev.donor_id,
                timestamp=scenario.request_timestamp + timedelta(minutes=float(ev.dispatched_at_minutes)),
                emergency_id=scenario.request_id,
                wave_number=int(ev.wave_number),
                accepted=bool(ev.accepted) if ev.accepted is not None else None,
            )

        # Record episode results
        episode_records.append({
            "Timeline_ID": timeline_id,
            "Arm": arm_name,
            "Lambda_Fatigue": lambda_fatigue,
            "Lookback_Days": lookback_days,
            "Scenario_ID": scenario.request_id,
            "Timestamp": scenario.request_timestamp.isoformat(),
            "Urgency": scenario.urgency,
            "Blood_Type": scenario.blood_type_required,
            "Units_Req": scenario.units_required,
            "Units_Fulfilled": res.units_fulfilled,
            "Units_Fulfilled_On_Time": res.units_fulfilled_on_time,
            "Quota_Met_Deadline": bool(res.quota_met_on_time),
            "Quota_Met_Eventual": bool(res.quota_met),
            "Time_To_Fulfill_Deadline": res.time_to_on_time_fulfillment if res.quota_met_on_time else np.nan,
            "Time_To_Fulfill_Eventual": res.time_to_fulfillment_minutes if res.quota_met else np.nan,
            "Alerts_Sent": res.total_notifications_sent,
            "Shortfall": res.shortfall,
            "Waves": res.waves_used,
            "Late_Arrivals": res.late_arrivals,
            "No_Suitable_Donors": len(feasible_pool) == 0,
        })

    return tracker, episode_records


def compute_timeline_summary(
    timeline_id: int,
    arm_name: str,
    lambda_fatigue: float,
    lookback_days: float,
    tracker: LongitudinalDonorTracker,
    episodes: list[dict],
    pool_donor_ids: list[str],
) -> dict:
    """
    Compute aggregate timeline-level summary metrics.
    """
    df_ep = pd.DataFrame(episodes)
    n_scenarios = len(df_ep)
    tot_req = df_ep["Units_Req"].sum()
    tot_alerts = df_ep["Alerts_Sent"].sum()
    tot_fulfilled = df_ep["Units_Fulfilled"].sum()

    deadline_rate = float(df_ep["Quota_Met_Deadline"].mean() * 100.0) if n_scenarios > 0 else 0.0
    eventual_rate = float(df_ep["Quota_Met_Eventual"].mean() * 100.0) if n_scenarios > 0 else 0.0

    dl_met_times = df_ep[df_ep["Quota_Met_Deadline"]]["Time_To_Fulfill_Deadline"].dropna()
    mean_dl_time = float(dl_met_times.mean()) if len(dl_met_times) > 0 else np.nan
    median_dl_time = float(dl_met_times.median()) if len(dl_met_times) > 0 else np.nan

    afi = float(tot_alerts / tot_req) if tot_req > 0 else 0.0
    late_rate = float(df_ep["Late_Arrivals"].sum() / tot_req) if tot_req > 0 else 0.0
    shortfall_ratio = float(df_ep["Shortfall"].sum() / tot_req) if tot_req > 0 else 0.0

    n_deadline_met = int(df_ep["Quota_Met_Deadline"].sum())
    n_eventual_met = int(df_ep["Quota_Met_Eventual"].sum())
    n_unfulfilled_deadline = int(n_scenarios - n_deadline_met)
    n_unfulfilled_eventual = int(n_scenarios - n_eventual_met)
    prop_unfulfilled_deadline = (
        float(n_unfulfilled_deadline) / float(n_scenarios) if n_scenarios > 0 else 0.0
    )
    prop_unfulfilled_eventual = (
        float(n_unfulfilled_eventual) / float(n_scenarios) if n_scenarios > 0 else 0.0
    )

    mean_alerts_per_emergency = float(tot_alerts / n_scenarios) if n_scenarios > 0 else 0.0
    # Denominator: deadline-fulfilled emergencies. NaN if the timeline has zero
    # deadline fulfilments (avoids dividing by zero; do not impute 0).
    alerts_per_deadline_fulfilled = (
        float(tot_alerts / n_deadline_met) if n_deadline_met > 0 else np.nan
    )
    alerts_per_eventual_fulfilled = (
        float(tot_alerts / n_eventual_met) if n_eventual_met > 0 else np.nan
    )

    ev_met_times = df_ep[df_ep["Quota_Met_Eventual"]]["Time_To_Fulfill_Eventual"].dropna()
    mean_ev_time = float(ev_met_times.mean()) if len(ev_met_times) > 0 else np.nan
    median_ev_time = float(ev_met_times.median()) if len(ev_met_times) > 0 else np.nan

    # Donor-level distribution metrics
    donor_summaries = tracker.compute_donor_summary(pool_donor_ids)
    alert_counts = np.array([d["total_alerts"] for d in donor_summaries], dtype=float)
    contacted_alerts = alert_counts[alert_counts > 0]

    n_pool = len(pool_donor_ids)
    pct_contacted = float(np.mean(alert_counts > 0) * 100.0) if n_pool > 0 else 0.0
    pct_repeat_contact = float(np.mean(alert_counts >= 2) * 100.0) if n_pool > 0 else 0.0
    pct_heavy_contact = float(np.mean(alert_counts >= 5) * 100.0) if n_pool > 0 else 0.0

    mean_alerts_all = float(np.mean(alert_counts)) if n_pool > 0 else 0.0
    median_alerts_all = float(np.median(alert_counts)) if n_pool > 0 else 0.0
    mean_alerts_contacted = float(np.mean(contacted_alerts)) if len(contacted_alerts) > 0 else 0.0
    median_alerts_contacted = float(np.median(contacted_alerts)) if len(contacted_alerts) > 0 else 0.0
    max_alerts_single_donor = int(np.max(alert_counts)) if len(alert_counts) > 0 else 0

    gini_all = calculate_gini(alert_counts)
    gini_contacted = calculate_gini(contacted_alerts) if len(contacted_alerts) > 0 else 0.0

    lb_stats = tracker.compute_lookback_repeat_stats(pool_donor_ids, lookback_days=lookback_days)

    return {
        "Timeline_ID": timeline_id,
        "Arm": arm_name,
        "Lambda_Fatigue": lambda_fatigue,
        "Lookback_Days": lookback_days,
        "Emergency_Count": n_scenarios,
        "Total_Units_Req": int(tot_req),
        "Total_Units_Fulfilled": int(tot_fulfilled),
        "Total_Alerts_Dispatched": int(tot_alerts),
        "Deadline_Fulfilment_Rate": deadline_rate,
        "Eventual_Fulfilment_Rate": eventual_rate,
        "Unfulfilled_Deadline_Count": n_unfulfilled_deadline,
        "Unfulfilled_Eventual_Count": n_unfulfilled_eventual,
        "Unfulfilled_Deadline_Proportion": prop_unfulfilled_deadline,
        "Unfulfilled_Eventual_Proportion": prop_unfulfilled_eventual,
        "Mean_Time_To_Deadline_Met": mean_dl_time,
        "Median_Time_To_Deadline_Met": median_dl_time,
        "Mean_Time_To_Eventual_Met": mean_ev_time,
        "Median_Time_To_Eventual_Met": median_ev_time,
        "Mean_Alerts_Per_Emergency": mean_alerts_per_emergency,
        "Alerts_Per_Deadline_Fulfilled_Emergency": alerts_per_deadline_fulfilled,
        "Alerts_Per_Eventual_Fulfilled_Emergency": alerts_per_eventual_fulfilled,
        "Alert_Fatigue_Index": afi,
        "Late_Arrival_Rate": late_rate,
        "Shortfall_Ratio": shortfall_ratio,
        "Pct_Donors_Contacted": pct_contacted,
        "Pct_Donors_Repeat_Notified": pct_repeat_contact,
        "Pct_Donors_Heavy_Notified": pct_heavy_contact,
        "Lookback_Repeat_Notification_Rate": lb_stats["lookback_repeat_notification_rate"],
        "Pct_Donors_Lookback_Repeat": lb_stats["pct_donors_with_lookback_repeat"],
        "N_Lookback_Repeat_Notifications": int(lb_stats["n_lookback_repeat_notifications"]),
        "Mean_Alerts_Per_Donor_All": mean_alerts_all,
        "Median_Alerts_Per_Donor_All": median_alerts_all,
        "Mean_Alerts_Per_Contacted_Donor": mean_alerts_contacted,
        "Median_Alerts_Per_Contacted_Donor": median_alerts_contacted,
        "Max_Alerts_Single_Donor": max_alerts_single_donor,
        "Gini_Notification_Inequality": gini_all,
        "Gini_Contacted_Inequality": gini_contacted,
    }


def verify_metrics_from_artifacts(
    df_episodes: pd.DataFrame,
    df_timelines: pd.DataFrame,
    df_paired: pd.DataFrame,
    rtol: float = 1e-9,
    atol: float = 1e-8,
) -> dict:
    """
    Independently recompute timeline aggregates from episode_results and check
    that paired differences equal Arm A minus Arm B timeline means.
    """
    mismatches: list[str] = []

    for (t_id, arm), g in df_episodes.groupby(["Timeline_ID", "Arm"]):
        row = df_timelines[(df_timelines["Timeline_ID"] == t_id) & (df_timelines["Arm"] == arm)]
        if len(row) != 1:
            mismatches.append(f"Expected one timeline row for ({t_id}, {arm}), got {len(row)}")
            continue
        r = row.iloc[0]
        n = len(g)
        if int(r["Emergency_Count"]) != n:
            mismatches.append(f"{t_id}/{arm} Emergency_Count {r['Emergency_Count']} != {n}")
        alerts = int(g["Alerts_Sent"].sum())
        if int(r["Total_Alerts_Dispatched"]) != alerts:
            mismatches.append(
                f"{t_id}/{arm} Total_Alerts_Dispatched {r['Total_Alerts_Dispatched']} != {alerts}"
            )
        units = int(g["Units_Req"].sum())
        if int(r["Total_Units_Req"]) != units:
            mismatches.append(f"{t_id}/{arm} Total_Units_Req mismatch")
        dr = float(g["Quota_Met_Deadline"].mean() * 100.0) if n else 0.0
        if not np.isclose(float(r["Deadline_Fulfilment_Rate"]), dr, rtol=rtol, atol=atol):
            mismatches.append(f"{t_id}/{arm} Deadline_Fulfilment_Rate mismatch")
        er = float(g["Quota_Met_Eventual"].mean() * 100.0) if n else 0.0
        if not np.isclose(float(r["Eventual_Fulfilment_Rate"]), er, rtol=rtol, atol=atol):
            mismatches.append(f"{t_id}/{arm} Eventual_Fulfilment_Rate mismatch")
        afi = float(alerts / units) if units > 0 else 0.0
        if not np.isclose(float(r["Alert_Fatigue_Index"]), afi, rtol=rtol, atol=atol):
            mismatches.append(f"{t_id}/{arm} Alert_Fatigue_Index mismatch")
        unf = int((~g["Quota_Met_Deadline"].astype(bool)).sum())
        if int(r["Unfulfilled_Deadline_Count"]) != unf:
            mismatches.append(f"{t_id}/{arm} Unfulfilled_Deadline_Count mismatch")

    df_a = df_timelines[df_timelines["Arm"] == ARM_A_NAME].sort_values("Timeline_ID")
    df_b = df_timelines[df_timelines["Arm"] == ARM_B_NAME].sort_values("Timeline_ID")
    for _, pc in df_paired.iterrows():
        col = pc["Metric"]
        if col not in df_a.columns:
            continue
        a = df_a[col].to_numpy(dtype=float)
        b = df_b[col].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        recomputed = float(np.mean(a[mask] - b[mask])) if mask.any() else np.nan
        reported = float(pc["Mean_Diff_A_minus_B"])
        if np.isfinite(recomputed) and np.isfinite(reported):
            if not np.isclose(recomputed, reported, rtol=1e-8, atol=1e-8):
                mismatches.append(
                    f"Paired {col} diff {reported} != recomputed {recomputed}"
                )

    return {"passed": len(mismatches) == 0, "n_checks_failed": len(mismatches), "mismatches": mismatches}


def _fmt_ci(row: pd.Series) -> str:
    return (
        f"{row['Mean_Diff_A_minus_B']:.4f} "
        f"(95% CI {row['Diff_95CI_Low']:.4f} to {row['Diff_95CI_High']:.4f})"
    )


def write_experiment_report(
    output_dir: Path,
    manifest: dict,
    df_timelines: pd.DataFrame,
    df_episodes: pd.DataFrame,
    df_paired: pd.DataFrame,
    df_sens: pd.DataFrame,
    start_time: float,
) -> None:
    """Write experiment_report.md from the generated artifacts (no hand-copied numbers)."""
    df_a = df_timelines[df_timelines["Arm"] == ARM_A_NAME].sort_values("Timeline_ID")
    df_b = df_timelines[df_timelines["Arm"] == ARM_B_NAME].sort_values("Timeline_ID")
    pc = df_paired.set_index("Metric")

    n_ep_a = int((df_episodes["Arm"] == ARM_A_NAME).sum())
    n_ep_b = int((df_episodes["Arm"] == ARM_B_NAME).sum())

    def row(metric: str) -> pd.Series:
        return pc.loc[metric]

    lines = []
    lines.append("# Phase 6.2 — Longitudinal Multi-Episode Fatigue Simulation")
    lines.append("")
    lines.append("**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System")
    lines.append("**Study type:** In-silico research experiment. Results are **not** clinical validation and do **not** prove real-world donor behaviour.")
    lines.append(f"**Random seed:** `{manifest['random_seed']}`")
    lines.append(f"**Execution timestamp:** {manifest['execution_timestamp']}")
    lines.append(f"**Elapsed seconds:** {manifest['elapsed_seconds']}")
    lines.append(f"**Dataset SHA-256:** `{manifest['dataset_sha256']}`")
    lines.append("")
    lines.append("## 1. Research question")
    lines.append("")
    lines.append("Across repeated synthetic emergency incidents, does fatigue-aware dynamic UDI ranking reduce notification burden per donor, and what trade-off, if any, does it introduce in deadline fulfilment?")
    lines.append("")
    lines.append("## 2. Design")
    lines.append("")
    lines.append("Matched comparison of the same **Multiplicative Dynamic UDI** architecture:")
    lines.append("")
    lines.append(f"- **Arm A (fatigue-aware):** $\\lambda = {manifest['default_lambda_fatigue']:.2f}$, lookback $= {manifest['default_lookback_days']:.0f}$ days.")
    lines.append("- **Arm B (fatigue-disabled):** $\\lambda = 0$, so $\\Phi_{\\mathrm{fatigue}}(i)=1$ for every donor. All other ranking, eligibility, feasibility, dispatch, and stopping rules are identical.")
    lines.append("")
    lines.append("Both arms receive the same emergency-arrival schedule, donor profiles, request requirements, synthetic travel conditions, and latent response variates within each timeline. Policies may diverge after the first invitation because earlier contacts change later ranking and fulfilment; those downstream differences are recorded, not forced to match.")
    lines.append("")
    lines.append("Independent cluster for uncertainty: **the timeline**. Multiple emergencies within a timeline are dependent. Paired differences are Arm A minus Arm B, with a scenario-level paired bootstrap 95% CI (`n_resamples=2000`) and Wilcoxon signed-rank tests on the timeline means. No formal non-inferiority margin was pre-specified; fulfilment results are descriptive paired differences only.")
    lines.append("")
    lines.append("Primary endpoints (predeclared): lookback-repeat notification rate, horizon repeat rate (≥2 alerts), heavy-contact rate (≥5 alerts), max alerts per donor, mean alerts per donor (full pool), Alert Fatigue Index, deadline fulfilment rate, and mean time to fulfilment among deadline-met emergencies. Holm–Bonferroni adjustment is applied only to that primary family. Remaining metrics and the λ/window grid are exploratory.")
    lines.append("")
    lines.append("## 3. Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Timelines | {manifest['n_timelines']} |")
    lines.append(f"| Horizon | {manifest['horizon_days']} days |")
    lines.append(f"| Emergency rate | {manifest['emergencies_per_day']} per day (exponential inter-arrival) |")
    lines.append(f"| Emergencies per arm (realized) | {manifest['total_emergencies_simulated']} |")
    lines.append(f"| Matched episode runs (A+B) | {manifest['matched_arm_episode_count']} |")
    lines.append(f"| Donor pool size | {manifest['n_donors_per_timeline']} sampled Kaggle profiles (same pool on every timeline) |")
    lines.append("| Dispatch | wave 1 size $\\lceil 3Q\\rceil$, subsequent $\\lceil 2\\times$shortfall$\\rceil$ |")
    lines.append("| Wave timeout / max waves | 15 min / 10 |")
    lines.append("| Observation horizon | 1440 min |")
    lines.append("| Travel time | `TravelTimeEngine` network-grid distance with congestion by hour of day; missing/non-finite ETA is infeasible (not treated as 0) |")
    lines.append("| Fatigue formula | $\\Phi=\\exp(-\\lambda N_{\\mathrm{recent\\ alerts}})$ as in `UDIEngine._fatigue_discount` |")
    lines.append(f"| Lookback | closed interval $[T-W, T]$ at ranking time $T=$ request timestamp |")
    lines.append(f"| Schedule seed | `{manifest['timeline_schedule_seed_formula']}` |")
    lines.append(f"| Response-variate seed | `{manifest['response_variate_seed_formula']}` |")
    lines.append("")
    lines.append("Urgency mix and unit ranges follow the existing synthetic generator (`CRITICAL`/`URGENT`/`SCHEDULED` with horizons 60/120/240 minutes).")
    lines.append("")
    lines.append("## 4. Data provenance")
    lines.append("")
    lines.append("| Class | Contents |")
    lines.append("|---|---|")
    lines.append("| Kaggle-derived | city, blood group, months since first donation, number of donations, pints donated, historical `availability` label (not used as emergency response) |")
    lines.append("| Synthetic | donor IDs, coordinates, eligibility (static 90% draw), `synthetic_response_propensity` ~ Beta(2,5) independent of availability, emergency times/sites/demand, travel times, response accept/latency, congestion |")
    lines.append("| Directly measured in this run | dispatched notifications, fulfilment flags, times, shortfall, late arrivals, rolling alert counts |")
    lines.append("| Untested with real emergency recruitment data | whether donors actually respond less after repeated alerts; operational travel times; true eligibility after donation |")
    lines.append("")
    lines.append("This experiment uses the same ranking input as Phase 4/5: `synthetic_response_propensity` is both the ranker's $P(i)$ and the simulator's Bernoulli parameter. That is optimistic information relative to a field system. The ranker does not observe future accept/decline draws, latencies, or arrivals. Kaggle `availability` is not reinterpreted as recruitment response.")
    lines.append("")
    lines.append("Response **probability** is not reduced by fatigue in this study. `StochasticResponseModel` can inflate latency by `fatigue_mult ** recent_alerts`, but the simulator does not pass `recent_alerts` (default 0). The study therefore measures **notification allocation and burden**, not a proven psychological response to repeated alerts.")
    lines.append("")
    lines.append("## 5. Metric definitions")
    lines.append("")
    lines.append("- **Total notifications / timeline:** $\\sum_e N^{\\mathrm{disp}}(e)$, counted from `select_batch` dispatches (not from the ranked list).")
    lines.append("- **Mean/median notifications per donor:** over the full pool of $N$ donors, including zeros.")
    lines.append("- **% donors with ≥2 (or ≥5) notifications:** $100\\times|\\{i: N_i\\ge k\\}|/N$ over the 30-day horizon.")
    lines.append("- **Lookback-repeat notification rate:** number of dispatches for which the same donor already had ≥1 earlier dispatch in $[t-W, t)$, divided by all dispatches. Denominator is notifications, not donors.")
    lines.append("- **% donors with a lookback repeat:** $100\\times$ donors with ≥1 lookback-repeat dispatch $/N$.")
    lines.append("- **Notifications per deadline-fulfilled emergency:** (timeline total alerts) / (count of deadline-met emergencies). **NaN** if a timeline has zero deadline fulfilments; those timelines are dropped from the paired mean of this metric.")
    lines.append("- **Alert Fatigue Index:** $\\mathrm{AFI}=(\\sum N^{\\mathrm{disp}})/(\\sum Q)$ using the existing definition (alerts / units required), pooled within a timeline.")
    lines.append("- **Gini:** Gini coefficient of per-donor notification counts over the full pool (0 = equal alerts, including zeros).")
    lines.append("- **Deadline fulfilment rate:** $100\\times$ mean of `quota_met_on_time` over emergencies in the timeline.")
    lines.append("- **Eventual fulfilment rate:** $100\\times$ mean of `quota_met` (arrivals within the 1440 min observation window).")
    lines.append("- **Unfulfilled counts/proportions:** emergencies with the corresponding quota flag false.")
    lines.append("- **Time to fulfilment (deadline-met):** mean of `Time_To_Fulfill_Deadline` **only among emergencies that met the deadline**. This is a conditional mean; it is not a mean over all emergencies.")
    lines.append("- **Time to fulfilment (eventual):** analogous conditional mean among eventually fulfilled emergencies.")
    lines.append("- **Late-arrival rate / shortfall ratio:** $\\sum$ late arrivals / $\\sum Q$ and $\\sum$ shortfall / $\\sum Q$.")
    lines.append("")
    lines.append("## 6. Limitations (omitted longitudinal clinical constraints)")
    lines.append("")
    lines.append("- No post-donation deferral: a donor who arrives may be contacted again in a later emergency if still synthetically eligible.")
    lines.append("- Eligibility is a static synthetic flag, not an 8-week clinical rule updated over the horizon.")
    lines.append("- No haemoglobin, infection-marker, or age screens.")
    lines.append("- `simulated_current_availability` is not used as a hard filter (same as Phase 4/5).")
    lines.append("- Adapter Poisson `recent_alert_count` on profiles is ignored; every timeline starts at zero historical alerts.")
    lines.append("- Within an emergency, `select_batch()` still forbids contacting the same donor twice.")
    lines.append("")
    lines.append("## 7. Primary paired results (timeline means)")
    lines.append("")
    lines.append(f"Realized episode counts: Arm A `{n_ep_a}`, Arm B `{n_ep_b}` (must match).")
    lines.append("")
    lines.append("| Metric | Arm A mean | Arm B mean | A − B (95% CI) | Wilcoxon $p$ | Holm $p$ (primary family) |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for metric in [
        "Lookback_Repeat_Notification_Rate",
        "Pct_Donors_Repeat_Notified",
        "Pct_Donors_Heavy_Notified",
        "Max_Alerts_Single_Donor",
        "Mean_Alerts_Per_Donor_All",
        "Alert_Fatigue_Index",
        "Deadline_Fulfilment_Rate",
        "Mean_Time_To_Deadline_Met",
    ]:
        r = row(metric)
        holm = r["Holm_Adjusted_p_Primary_Family"]
        holm_s = f"{holm:.4e}" if np.isfinite(holm) else "—"
        p_s = f"{r['Wilcoxon_p']:.4e}" if np.isfinite(r["Wilcoxon_p"]) else "—"
        lines.append(
            f"| {r['Description']} | {r['Arm_A_Mean']:.4f} | {r['Arm_B_Mean']:.4f} | "
            f"{_fmt_ci(r)} | {p_s} | {holm_s} |"
        )
    lines.append("")
    lines.append("### Secondary paired results")
    lines.append("")
    lines.append("| Metric | Arm A mean | Arm B mean | A − B (95% CI) | Wilcoxon $p$ |")
    lines.append("|---|---:|---:|---|---:|")
    for metric, r in pc.iterrows():
        if r["Family"] != "secondary":
            continue
        p_s = f"{r['Wilcoxon_p']:.4e}" if np.isfinite(r["Wilcoxon_p"]) else "—"
        lines.append(
            f"| {r['Description']} | {r['Arm_A_Mean']:.4f} | {r['Arm_B_Mean']:.4f} | "
            f"{_fmt_ci(r)} | {p_s} |"
        )
    lines.append("")

    lb = row("Lookback_Repeat_Notification_Rate")
    heavy = row("Pct_Donors_Heavy_Notified")
    mx = row("Max_Alerts_Single_Donor")
    rpt = row("Pct_Donors_Repeat_Notified")
    dr = row("Deadline_Fulfilment_Rate")
    tt = row("Mean_Time_To_Deadline_Met")
    afi = row("Alert_Fatigue_Index")

    lines.append("## 8. Interpretation (honest, in-silico only)")
    lines.append("")
    lines.append(
        f"Fatigue-aware ranking **reduced concentration of alerts** on the most-contacted donors: "
        f"mean max alerts per timeline {mx['Arm_A_Mean']:.2f} vs {mx['Arm_B_Mean']:.2f} "
        f"(difference {_fmt_ci(mx)}), and heavy-contact prevalence {heavy['Arm_A_Mean']:.3f}% vs {heavy['Arm_B_Mean']:.3f}% "
        f"(difference {_fmt_ci(heavy)})."
    )
    lines.append("")
    lines.append(
        f"The share of the pool receiving **two or more** alerts over the full horizon moved in the **opposite** direction: "
        f"{rpt['Arm_A_Mean']:.3f}% (Arm A) vs {rpt['Arm_B_Mean']:.3f}% (Arm B), difference {_fmt_ci(rpt)}. "
        f"That pattern is consistent with load-spreading: down-ranking recently notified donors sends later invitations to previously unused donors, so more people receive a small number of alerts while fewer people receive a large number."
    )
    lines.append("")
    lines.append(
        f"Lookback-window repeat-notification rate: Arm A {lb['Arm_A_Mean']:.4f} vs Arm B {lb['Arm_B_Mean']:.4f}, "
        f"difference {_fmt_ci(lb)}."
    )
    lines.append("")
    lines.append(
        f"Deadline fulfilment: Arm A {dr['Arm_A_Mean']:.4f}% vs Arm B {dr['Arm_B_Mean']:.4f}%, difference {_fmt_ci(dr)}. "
        f"No non-inferiority margin was declared; this interval should be read as a paired difference, not as proof of clinical non-inferiority."
    )
    lines.append("")
    lines.append(
        f"Conditional mean time to fulfilment among **deadline-met** emergencies: Arm A {tt['Arm_A_Mean']:.3f} min vs Arm B {tt['Arm_B_Mean']:.3f} min, "
        f"difference {_fmt_ci(tt)}. AFI (alerts / units required): Arm A {afi['Arm_A_Mean']:.4f} vs Arm B {afi['Arm_B_Mean']:.4f}, difference {_fmt_ci(afi)}."
    )
    lines.append("")
    ci_low_dr = float(dr["Diff_95CI_Low"])
    ci_high_dr = float(dr["Diff_95CI_High"])
    if ci_low_dr < 0 < ci_high_dr:
        lines.append("The 95% CI for the deadline-fulfilment difference includes 0.")
    elif ci_high_dr < 0:
        lines.append("The 95% CI for the deadline-fulfilment difference lies entirely below 0 (Arm A lower fulfilment in this simulation).")
    else:
        lines.append("The 95% CI for the deadline-fulfilment difference lies entirely above 0 (Arm A higher fulfilment in this simulation).")
    lines.append("")
    lines.append("Phase 4 and Phase 5 single-episode measurements were not recomputed and are not replaced by these results.")
    lines.append("")
    lines.append("## 9. Sensitivity (exploratory)")
    lines.append("")
    if df_sens.empty:
        lines.append("Sensitivity grid was not run.")
    else:
        lines.append(
            f"Grid: $\\lambda\\in{manifest['sensitivity_lambdas']}$, lookback $\\in{manifest['sensitivity_lookback_windows']}$ days, "
            f"first {manifest['sensitivity_timelines']} timelines, **same response-variate seeds as the primary arms**. "
            f"$\\lambda=0$ is ranking-equivalent to Arm B regardless of lookback. No multiplicity-adjusted inference is claimed for this grid."
        )
        lines.append("")
        lines.append("```")
        lines.append(df_sens.to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("A larger $\\lambda$ is not automatically better: inspect both max/repeat burden and deadline fulfilment in the table above before any tuning recommendation.")
    lines.append("")
    lines.append("## 10. Reproducibility and tests")
    lines.append("")
    lines.append("Re-run:")
    lines.append("")
    lines.append("```")
    lines.append(
        f"python experiments/exp_phase6_2_longitudinal_fatigue.py "
        f"--timelines {manifest['n_timelines']} --days {manifest['horizon_days']} "
        f"--rate {manifest['emergencies_per_day']} --donors {manifest['n_donors_per_timeline']} "
        f"--seed {manifest['random_seed']}"
    )
    lines.append("```")
    lines.append("")
    lines.append(f"Independent recomputation of timeline aggregates from `episode_results.csv` and paired differences from `timeline_metrics.csv`: **{'passed' if manifest.get('metric_verification_passed') else 'FAILED'}** (`metric_verification.json`).")
    lines.append("")
    lines.append("Focused tests live in `tests/test_longitudinal_fatigue.py`. Existing Phase 4/5 tests remain in the project test suite and must not be weakened.")
    lines.append("")
    lines.append("## 11. Artifacts")
    lines.append("")
    lines.append("- `episode_results.csv` — one row per emergency × arm")
    lines.append("- `timeline_metrics.csv` — one row per timeline × arm")
    lines.append("- `paired_comparisons.csv`")
    lines.append("- `sensitivity_results.csv`")
    lines.append("- `donor_timeline_summary.csv` — per-donor alert totals for every timeline × arm")
    lines.append("- `donor_notification_history.csv` — event-level dispatches for timeline 0 only (audit sample)")
    lines.append("- `request_schedule.csv` — shared exogenous emergency schedule")
    lines.append("- `scenario_manifest.json`, `run_log.txt`, `metric_verification.json`")
    lines.append("- `figures/` — burden, fulfilment, scatter, and sensitivity plots")
    lines.append("- `event_traces/` — short traces for the first two timelines")
    lines.append("")
    lines.append("## 12. Recommended next phase")
    lines.append("")
    lines.append("If a later phase introduces behavioural response decay, it must be specified separately from ranking-side $\\Phi_{\\mathrm{fatigue}}$ and must not be fitted on these same simulation outcomes. A documented donation cooldown would change both burden and fulfilment and should be added only with an explicit clinical rule. This report does not proceed to paper writing or deployment.")
    lines.append("")

    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_longitudinal_experiment(
    n_timelines: int = DEFAULT_N_TIMELINES,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    emergencies_per_day: float = DEFAULT_EMERGENCIES_PER_DAY,
    n_donors: int = DEFAULT_N_DONORS,
    seed: int = DEFAULT_SEED,
    output_dir: Path = OUTPUT_DIR,
    run_sensitivity: bool = True,
) -> dict:
    """
    Execute the Phase 6.2 longitudinal fatigue experiment and generate all artifacts.
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "event_traces").mkdir(exist_ok=True)

    logger.info("Initializing Phase 6.2 longitudinal experiment...")
    logger.info("Params: %d timelines, %.1f days horizon, %.2f emerg/day, %d donors, seed=%d",
                n_timelines, horizon_days, emergencies_per_day, n_donors, seed)

    # 1. Dataset loading and profile generation
    adapter = KaggleSimulationAdapter(data_path=DATA_PATH, seed=seed)
    df_raw = adapter.load_and_validate()
    profiles = adapter.build_simulation_profiles(n_donors=n_donors, sample_seed=seed + 1)
    pool_donor_ids = [p.donor_id for p in profiles]

    with open(PROJECT_ROOT / "configs" / "udi_weights.yaml") as f:
        weights_config = yaml.safe_load(f)

    travel_engine = TravelTimeEngine(distance_method="network_grid", congestion_model=CongestionModel())
    dispatch_policy = DispatchCascadePolicy(w1_mult=3.0, w_sub_mult=2.0)

    # 2. Pre-generate timeline scenario sequences
    timeline_scenarios: dict[int, list[SyntheticEmergencyScenario]] = {}
    for t_id in range(n_timelines):
        scenarios = generate_longitudinal_timeline_scenarios(
            timeline_id=t_id,
            horizon_days=horizon_days,
            rate_per_day=emergencies_per_day,
            profiles=profiles,
            seed=seed + 1000 + t_id * 37,
        )
        timeline_scenarios[t_id] = scenarios

    tot_emergencies = sum(len(s) for s in timeline_scenarios.values())
    logger.info("Generated %d total emergencies across %d timelines (mean %.1f per timeline)",
                tot_emergencies, n_timelines, tot_emergencies / n_timelines)

    # 3. Primary Matched Comparison: Arm A (Fatigue-Aware) vs Arm B (Fatigue-Disabled)
    all_episodes: list[dict] = []
    all_timeline_metrics: list[dict] = []
    all_donor_records: list[dict] = []

    logger.info("Executing Primary Matched Experiment: Arm A (lambda=%.2f) vs Arm B (lambda=0.00)...", DEFAULT_LAMBDA)

    sample_traces_collected = 0

    for t_id in range(n_timelines):
        scenarios = timeline_scenarios[t_id]

        # Run Arm A (Fatigue-Aware, lambda=0.30)
        tracker_a, ep_a = run_timeline_for_arm(
            timeline_id=t_id,
            arm_name=ARM_A_NAME,
            lambda_fatigue=DEFAULT_LAMBDA,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            profiles=profiles,
            scenarios=scenarios,
            travel_engine=travel_engine,
            dispatch_policy=dispatch_policy,
            weights_config=weights_config,
            seed=seed,
        )
        summary_a = compute_timeline_summary(
            t_id, ARM_A_NAME, DEFAULT_LAMBDA, DEFAULT_LOOKBACK_DAYS, tracker_a, ep_a, pool_donor_ids
        )

        # Run Arm B (Fatigue-Disabled, lambda=0.00)
        tracker_b, ep_b = run_timeline_for_arm(
            timeline_id=t_id,
            arm_name=ARM_B_NAME,
            lambda_fatigue=0.00,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            profiles=profiles,
            scenarios=scenarios,
            travel_engine=travel_engine,
            dispatch_policy=dispatch_policy,
            weights_config=weights_config,
            seed=seed,
        )
        summary_b = compute_timeline_summary(
            t_id, ARM_B_NAME, 0.00, DEFAULT_LOOKBACK_DAYS, tracker_b, ep_b, pool_donor_ids
        )

        all_episodes.extend(ep_a)
        all_episodes.extend(ep_b)
        all_timeline_metrics.extend([summary_a, summary_b])

        for dsum in tracker_a.compute_donor_summary(pool_donor_ids):
            all_donor_records.append({
                "Timeline_ID": t_id,
                "Arm": ARM_A_NAME,
                "Donor_ID": dsum["donor_id"],
                "Total_Alerts": dsum["total_alerts"],
                "Total_Acceptances": dsum["total_acceptances"],
                "Notified_Ever": dsum["notified_ever"],
                "Multiple_Alerts": dsum["multiple_alerts"],
                "Heavy_Alerts": dsum["heavy_alerts"],
            })
        for dsum in tracker_b.compute_donor_summary(pool_donor_ids):
            all_donor_records.append({
                "Timeline_ID": t_id,
                "Arm": ARM_B_NAME,
                "Donor_ID": dsum["donor_id"],
                "Total_Alerts": dsum["total_alerts"],
                "Total_Acceptances": dsum["total_acceptances"],
                "Notified_Ever": dsum["notified_ever"],
                "Multiple_Alerts": dsum["multiple_alerts"],
                "Heavy_Alerts": dsum["heavy_alerts"],
            })

        # Event-level history is retained only for timeline 0 (audit sample).
        if t_id == 0:
            event_history_rows = []
            for arm_label, tracker in ((ARM_A_NAME, tracker_a), (ARM_B_NAME, tracker_b)):
                running_cum: dict[str, int] = {}
                for r in tracker.get_all_records():
                    running_cum[r.donor_id] = running_cum.get(r.donor_id, 0) + 1
                    event_history_rows.append({
                        "Timeline_ID": t_id,
                        "Arm": arm_label,
                        "Donor_ID": r.donor_id,
                        "Timestamp": r.timestamp.isoformat(),
                        "Emergency_ID": r.emergency_id,
                        "Wave_Number": r.wave_number,
                        "Accepted": r.accepted,
                        "Cumulative_Alerts": running_cum[r.donor_id],
                        "Recent_Alerts_Including_This_Dispatch": tracker.get_recent_alert_count(
                            r.donor_id, r.timestamp
                        ),
                    })
            pd.DataFrame(event_history_rows).to_csv(
                output_dir / "donor_notification_history.csv", index=False
            )

        # Save representative event traces for first 2 timelines
        if sample_traces_collected < 2:
            trace_path = output_dir / "event_traces" / f"timeline_{t_id:02d}_comparison.txt"
            with open(trace_path, "w") as tf:
                tf.write(f"=== Longitudinal Trace: Timeline {t_id:02d} (30 Days) ===\n")
                tf.write(f"Total Emergencies: {len(scenarios)}\n\n")
                tf.write(f"Arm A (Fatigue-Aware): Deadline Rate={summary_a['Deadline_Fulfilment_Rate']:.1f}%, "
                         f"Mean Time={summary_a['Mean_Time_To_Deadline_Met']:.2f}m, "
                         f"Repeat Donors={summary_a['Pct_Donors_Repeat_Notified']:.1f}%, "
                         f"Max Alerts Single Donor={summary_a['Max_Alerts_Single_Donor']}\n")
                tf.write(f"Arm B (Fatigue-Disabled): Deadline Rate={summary_b['Deadline_Fulfilment_Rate']:.1f}%, "
                         f"Mean Time={summary_b['Mean_Time_To_Deadline_Met']:.2f}m, "
                         f"Repeat Donors={summary_b['Pct_Donors_Repeat_Notified']:.1f}%, "
                         f"Max Alerts Single Donor={summary_b['Max_Alerts_Single_Donor']}\n\n")
                tf.write("Top 5 Most Contacted Donors in Arm B vs Arm A:\n")
                d_sum_b = sorted(tracker_b.compute_donor_summary(pool_donor_ids), key=lambda x: x["total_alerts"], reverse=True)
                d_sum_a_map = {d["donor_id"]: d["total_alerts"] for d in tracker_a.compute_donor_summary(pool_donor_ids)}
                for top_d in d_sum_b[:5]:
                    did = top_d["donor_id"]
                    tf.write(f"  Donor {did}: Arm B={top_d['total_alerts']} alerts | Arm A={d_sum_a_map.get(did, 0)} alerts\n")
            sample_traces_collected += 1

        logger.info("Completed Timeline %d / %d | A: DR=%.1f%%, Repeat=%.1f%%, Max=%d | B: DR=%.1f%%, Repeat=%.1f%%, Max=%d",
                    t_id + 1, n_timelines,
                    summary_a["Deadline_Fulfilment_Rate"], summary_a["Pct_Donors_Repeat_Notified"], summary_a["Max_Alerts_Single_Donor"],
                    summary_b["Deadline_Fulfilment_Rate"], summary_b["Pct_Donors_Repeat_Notified"], summary_b["Max_Alerts_Single_Donor"])

    # 4. Sensitivity Grid (if enabled)
    sensitivity_records: list[dict] = []
    if run_sensitivity:
        logger.info("Executing Sensitivity Analysis across lambdas and lookback windows...")
        # Use first 5 timelines for sensitivity grid to remain computationally efficient
        sens_timelines = range(min(5, n_timelines))

        for lam in SENSITIVITY_LAMBDAS:
            for w_days in SENSITIVITY_WINDOWS:
                t_metrics_sens = []
                for t_id in sens_timelines:
                    scenarios = timeline_scenarios[t_id]
                    tracker_s, ep_s = run_timeline_for_arm(
                        timeline_id=t_id,
                        arm_name=f"Sens(lam={lam},w={w_days})",
                        lambda_fatigue=lam,
                        lookback_days=w_days,
                        profiles=profiles,
                        scenarios=scenarios,
                        travel_engine=travel_engine,
                        dispatch_policy=dispatch_policy,
                        weights_config=weights_config,
                        seed=seed,
                    )
                    sm = compute_timeline_summary(
                        t_id, f"Sens(lam={lam},w={w_days})", lam, w_days, tracker_s, ep_s, pool_donor_ids
                    )
                    t_metrics_sens.append(sm)

                df_sens_t = pd.DataFrame(t_metrics_sens)
                sensitivity_records.append({
                    "Lambda_Fatigue": lam,
                    "Lookback_Days": w_days,
                    "Mean_Deadline_Fulfilment_Rate": float(df_sens_t["Deadline_Fulfilment_Rate"].mean()),
                    "Mean_Time_To_Deadline_Met": float(df_sens_t["Mean_Time_To_Deadline_Met"].mean()),
                    "Mean_Alert_Fatigue_Index": float(df_sens_t["Alert_Fatigue_Index"].mean()),
                    "Mean_Pct_Repeat_Notified": float(df_sens_t["Pct_Donors_Repeat_Notified"].mean()),
                    "Mean_Pct_Heavy_Notified": float(df_sens_t["Pct_Donors_Heavy_Notified"].mean()),
                    "Mean_Max_Alerts_Single_Donor": float(df_sens_t["Max_Alerts_Single_Donor"].mean()),
                    "Mean_Gini_Notification_Inequality": float(df_sens_t["Gini_Notification_Inequality"].mean()),
                    "Mean_Lookback_Repeat_Notification_Rate": float(
                        df_sens_t["Lookback_Repeat_Notification_Rate"].mean()
                    ),
                    "Mean_Unfulfilled_Deadline_Proportion": float(
                        df_sens_t["Unfulfilled_Deadline_Proportion"].mean()
                    ),
                    "N_Timelines": int(len(df_sens_t)),
                })
                logger.info("Sensitivity (lambda=%.2f, w=%.0fd): DR=%.1f%%, Time=%.2fm, Repeat=%.1f%%, MaxAlerts=%.1f",
                            lam, w_days,
                            sensitivity_records[-1]["Mean_Deadline_Fulfilment_Rate"],
                            sensitivity_records[-1]["Mean_Time_To_Deadline_Met"],
                            sensitivity_records[-1]["Mean_Pct_Repeat_Notified"],
                            sensitivity_records[-1]["Mean_Max_Alerts_Single_Donor"])

    # 5. Paired Statistical Comparison Across Timelines
    df_timelines = pd.DataFrame(all_timeline_metrics)
    df_a = df_timelines[df_timelines["Arm"] == ARM_A_NAME].sort_values("Timeline_ID").reset_index(drop=True)
    df_b = df_timelines[df_timelines["Arm"] == ARM_B_NAME].sort_values("Timeline_ID").reset_index(drop=True)

    paired_comparisons = []
    # Primary endpoints are flagged; remaining metrics are predeclared secondary.
    # lower_is_better True => a negative (A − B) difference is the intended burden reduction.
    comp_metrics = [
        ("Lookback_Repeat_Notification_Rate", "Lookback-window repeat-notification rate (repeats / all alerts)", True, "primary_burden"),
        ("Pct_Donors_Repeat_Notified", "Percentage of donors with ≥2 alerts over the horizon (%)", True, "primary_burden"),
        ("Pct_Donors_Heavy_Notified", "Percentage of donors with ≥5 alerts over the horizon (%)", True, "primary_burden"),
        ("Max_Alerts_Single_Donor", "Maximum notifications received by a single donor", True, "primary_burden"),
        ("Mean_Alerts_Per_Donor_All", "Mean notifications per donor (full pool, including zeros)", True, "primary_burden"),
        ("Alert_Fatigue_Index", "Alert Fatigue Index (total alerts / total units required)", True, "primary_burden"),
        ("Deadline_Fulfilment_Rate", "Deadline fulfilment rate (% of emergencies)", False, "primary_fulfilment"),
        ("Mean_Time_To_Deadline_Met", "Mean time to fulfilment among deadline-met emergencies (min)", True, "primary_fulfilment"),
        ("Pct_Donors_Lookback_Repeat", "Percentage of donors with a lookback-window repeat (%)", True, "secondary"),
        ("Mean_Alerts_Per_Contacted_Donor", "Mean notifications per contacted donor", True, "secondary"),
        ("Gini_Notification_Inequality", "Gini coefficient of notification counts over the full pool", True, "secondary"),
        ("Eventual_Fulfilment_Rate", "Eventual quota fulfilment rate (% of emergencies)", False, "secondary"),
        ("Mean_Time_To_Eventual_Met", "Mean time to fulfilment among eventually-met emergencies (min)", True, "secondary"),
        ("Unfulfilled_Deadline_Proportion", "Proportion of emergencies missing the deadline", True, "secondary"),
        ("Mean_Alerts_Per_Emergency", "Mean notifications dispatched per emergency", True, "secondary"),
        ("Alerts_Per_Deadline_Fulfilled_Emergency", "Notifications per deadline-fulfilled emergency", True, "secondary"),
        ("Late_Arrival_Rate", "Late arrivals / units required", True, "secondary"),
        ("Shortfall_Ratio", "Unfulfilled units / units required", True, "secondary"),
    ]

    for col, display_name, lower_is_better, family in comp_metrics:
        vals_a = df_a[col].to_numpy(dtype=float)
        vals_b = df_b[col].to_numpy(dtype=float)
        # Drop timeline pairs where either arm is NaN (e.g. zero-fulfilment time metrics).
        mask = np.isfinite(vals_a) & np.isfinite(vals_b)
        vals_a_m = vals_a[mask]
        vals_b_m = vals_b[mask]
        diff = vals_a_m - vals_b_m

        mean_a = float(np.mean(vals_a_m)) if len(vals_a_m) else np.nan
        mean_b = float(np.mean(vals_b_m)) if len(vals_b_m) else np.nan
        mean_diff = float(np.mean(diff)) if len(diff) else np.nan
        n_pairs = int(len(diff))

        if n_pairs >= 2:
            ci_low, ci_high = bootstrap_ci(diff, confidence=0.95, n_resamples=2000, seed=seed)
            w_res = wilcoxon_test(vals_a_m, vals_b_m)
            tt_res = paired_ttest(vals_a_m, vals_b_m)
        else:
            ci_low, ci_high = np.nan, np.nan
            w_res = type("W", (), {"statistic": np.nan, "p_value": np.nan, "significant": False})()
            tt_res = type("T", (), {"p_value": np.nan})()

        paired_comparisons.append({
            "Metric": col,
            "Description": display_name,
            "Family": family,
            "N_Paired_Timelines": n_pairs,
            "Arm_A_Mean": mean_a,
            "Arm_B_Mean": mean_b,
            "Mean_Diff_A_minus_B": mean_diff,
            "Diff_95CI_Low": ci_low,
            "Diff_95CI_High": ci_high,
            "Wilcoxon_Stat": w_res.statistic,
            "Wilcoxon_p": w_res.p_value,
            "Paired_ttest_p": tt_res.p_value,
            "Unadjusted_Significant_0.05": bool(w_res.significant),
            "Lower_Is_Better": lower_is_better,
        })

    primary_idx = [i for i, r in enumerate(paired_comparisons) if str(r["Family"]).startswith("primary")]
    primary_p = [float(paired_comparisons[i]["Wilcoxon_p"]) for i in primary_idx]
    if primary_p and all(np.isfinite(primary_p)):
        primary_holm = holm_adjust(primary_p)
        holm_map = {primary_idx[j]: primary_holm[j] for j in range(len(primary_idx))}
    else:
        holm_map = {}
    for i, rec in enumerate(paired_comparisons):
        rec["Holm_Adjusted_p_Primary_Family"] = holm_map.get(i, np.nan)

    # 6. Save DataFrames to Output Directory
    df_episodes = pd.DataFrame(all_episodes)
    df_episodes.to_csv(output_dir / "episode_results.csv", index=False)

    df_timelines.to_csv(output_dir / "timeline_metrics.csv", index=False)

    df_paired = pd.DataFrame(paired_comparisons)
    df_paired.to_csv(output_dir / "paired_comparisons.csv", index=False)

    if sensitivity_records:
        df_sens = pd.DataFrame(sensitivity_records)
        df_sens.to_csv(output_dir / "sensitivity_results.csv", index=False)
    else:
        df_sens = pd.DataFrame()

    if all_donor_records:
        df_donors = pd.DataFrame(all_donor_records)
        df_donors.to_csv(output_dir / "donor_timeline_summary.csv", index=False)

    schedule_rows = []
    for t_id, scenarios in timeline_scenarios.items():
        for s in scenarios:
            schedule_rows.append({
                "Timeline_ID": t_id,
                "Scenario_ID": s.request_id,
                "Timestamp": s.request_timestamp.isoformat(),
                "Urgency": s.urgency,
                "Blood_Type": s.blood_type_required,
                "Units_Required": s.units_required,
                "Site_ID": s.site_id,
                "Hour_Of_Day": s.hour_of_day,
            })
    pd.DataFrame(schedule_rows).to_csv(output_dir / "request_schedule.csv", index=False)

    verification = verify_metrics_from_artifacts(df_episodes, df_timelines, df_paired)
    with open(output_dir / "metric_verification.json", "w") as vf:
        json.dump(verification, vf, indent=2)
    if not verification["passed"]:
        raise RuntimeError(f"Independent metric recomputation failed: {verification['mismatches']}")

    # 7. Generate figures
    logger.info("Generating publication figures...")
    arm_labels = ["Arm B\n(Fatigue-Disabled)", "Arm A\n(Fatigue-Aware)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    _boxplot(axes[0], [df_b["Pct_Donors_Repeat_Notified"], df_a["Pct_Donors_Repeat_Notified"]], arm_labels)
    axes[0].set_title("Horizon repeat rate (≥2 alerts)")
    axes[0].set_ylabel("Donors (%)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    _boxplot(axes[1], [df_b["Lookback_Repeat_Notification_Rate"], df_a["Lookback_Repeat_Notification_Rate"]], arm_labels)
    axes[1].set_title("Lookback-window repeat-notification rate")
    axes[1].set_ylabel("Repeats / all notifications")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    _boxplot(axes[2], [df_b["Max_Alerts_Single_Donor"], df_a["Max_Alerts_Single_Donor"]], arm_labels)
    axes[2].set_title("Maximum alerts to a single donor")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "notification_burden_reduction.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _boxplot(axes[0], [df_b["Deadline_Fulfilment_Rate"], df_a["Deadline_Fulfilment_Rate"]], arm_labels)
    axes[0].set_title("Deadline fulfilment rate")
    axes[0].set_ylabel("Emergencies fulfilled on time (%)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    _boxplot(axes[1], [df_b["Mean_Time_To_Deadline_Met"], df_a["Mean_Time_To_Deadline_Met"]], arm_labels)
    axes[1].set_title("Time to fulfilment (deadline-met emergencies)")
    axes[1].set_ylabel("Minutes")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "fulfilment_performance_tradeoff.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_b["Deadline_Fulfilment_Rate"], df_b["Max_Alerts_Single_Donor"],
               label="Arm B (fatigue-disabled)", alpha=0.75, marker="s")
    ax.scatter(df_a["Deadline_Fulfilment_Rate"], df_a["Max_Alerts_Single_Donor"],
               label="Arm A (fatigue-aware)", alpha=0.75, marker="o")
    ax.set_xlabel("Deadline fulfilment rate (%)")
    ax.set_ylabel("Max alerts to a single donor")
    ax.set_title("Burden–fulfilment trade-off (one point per timeline)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "burden_vs_fulfilment_scatter.png", dpi=300)
    plt.close()

    if not df_sens.empty:
        df_sens_14 = df_sens[df_sens["Lookback_Days"] == 14.0].sort_values("Lambda_Fatigue")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(df_sens_14["Lambda_Fatigue"], df_sens_14["Mean_Max_Alerts_Single_Donor"], marker="o", color="crimson")
        axes[0].set_title("Max alerts vs λ (lookback = 14 d, 5 timelines)")
        axes[0].set_xlabel("Fatigue coefficient λ")
        axes[0].set_ylabel("Mean of timeline maxima")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[1].plot(df_sens_14["Lambda_Fatigue"], df_sens_14["Mean_Deadline_Fulfilment_Rate"], marker="s", color="navy")
        axes[1].set_title("Deadline fulfilment vs λ (lookback = 14 d, 5 timelines)")
        axes[1].set_xlabel("Fatigue coefficient λ")
        axes[1].set_ylabel("Deadline fulfilment rate (%)")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "sensitivity_fatigue_lambda.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 5))
        for w_days, marker in zip(SENSITIVITY_WINDOWS, ["o", "s", "^"]):
            sub = df_sens[df_sens["Lookback_Days"] == w_days].sort_values("Lambda_Fatigue")
            ax.plot(sub["Lambda_Fatigue"], sub["Mean_Lookback_Repeat_Notification_Rate"],
                    marker=marker, label=f"Lookback {int(w_days)} d")
        ax.set_xlabel("Fatigue coefficient λ")
        ax.set_ylabel("Mean lookback-repeat notification rate")
        ax.set_title("Sensitivity of lookback-repeat rate (exploratory, 5 timelines)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "sensitivity_lookback_repeat.png", dpi=300)
        plt.close()

    # 8. Save Scenario Manifest
    manifest = {
        "experiment_name": "Phase 6.2 Longitudinal Multi-Episode Fatigue Simulation",
        "study_type": "in-silico simulation; not clinical validation",
        "dataset_path": str(DATA_PATH),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "random_seed": seed,
        "timeline_schedule_seed_formula": "seed + 1000 + timeline_id * 37",
        "response_variate_seed_formula": "seed + timeline_id * 10000 + scenario_index * 10",
        "n_timelines": n_timelines,
        "horizon_days": horizon_days,
        "emergencies_per_day": emergencies_per_day,
        "request_arrival_process": "exponential inter-arrival with mean 24/rate hours; truncated when next arrival would exceed the horizon",
        "total_emergencies_simulated": tot_emergencies,
        "matched_arm_episode_count": tot_emergencies * 2,
        "n_donors_per_timeline": n_donors,
        "same_donor_pool_across_timelines": True,
        "default_lambda_fatigue": DEFAULT_LAMBDA,
        "default_lookback_days": DEFAULT_LOOKBACK_DAYS,
        "fatigue_formula": "Phi = exp(-lambda * N_recent_alerts)",
        "lookback_boundary_semantics": (
            "A prior notification at time t is counted in N_recent_alerts at query time T iff "
            "T - lookback_days <= t <= T (closed interval). Ranking queries T = emergency request_timestamp "
            "before any dispatch in that emergency."
        ),
        "sensitivity_lambdas": SENSITIVITY_LAMBDAS,
        "sensitivity_lookback_windows": SENSITIVITY_WINDOWS,
        "sensitivity_timelines": min(5, n_timelines) if run_sensitivity else 0,
        "sensitivity_status": "exploratory; not a confirmatory multiplicity-controlled analysis",
        "dispatch_policy": {"w1_mult": 3.0, "w_sub_mult": 2.0},
        "udi_formula_mode": "multiplicative",
        "wave_timeout_minutes": 15.0,
        "max_waves": 10,
        "observation_horizon_minutes": 1440.0,
        "response_model": {
            "acceptance": "Bernoulli using synthetic_response_propensity; not fatigue-dependent",
            "latency": "lognormal(mu=2.0, sigma=0.8), min 1 minute; recent_alerts argument left at default 0",
            "kaggle_availability": "not used as response probability",
        },
        "omitted_longitudinal_clinical_constraints": [
            "No post-donation 8-week deferral update after simulated arrival",
            "Eligibility is a static synthetic 90% flag assigned at profile construction",
            "No haemoglobin, infection, or age-based deferral rules",
            "simulated_current_availability is not used as a hard dispatch filter",
        ],
        "information_separation": (
            "Ranker sees travel time, eligibility, blood-compatibility-filtered pool, "
            "synthetic_response_propensity (same optimistic P used in Phase 4/5), and N_recent_alerts. "
            "Ranker does not see future accept/decline outcomes, sampled latency, or arrival events."
        ),
        "execution_timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "metric_verification_passed": verification["passed"],
    }
    with open(output_dir / "scenario_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    write_experiment_report(
        output_dir=output_dir,
        manifest=manifest,
        df_timelines=df_timelines,
        df_episodes=df_episodes,
        df_paired=df_paired,
        df_sens=df_sens,
        start_time=start_time,
    )

    with open(output_dir / "run_log.txt", "w") as lf:
        lf.write("Phase 6.2 Longitudinal Experiment Log\n")
        lf.write(f"Started: {datetime.fromtimestamp(start_time).isoformat()}\n")
        lf.write(f"Completed: {datetime.now().isoformat()}\n")
        lf.write(f"Elapsed Time: {manifest['elapsed_seconds']} seconds\n")
        lf.write(f"Timelines: {n_timelines} | Emergencies per arm: {tot_emergencies} | Matched episodes: {tot_emergencies * 2}\n")
        lf.write(f"Metric verification passed: {verification['passed']}\n")
        lf.write("Paired Statistics Summary (timeline-clustered, Arm A minus Arm B):\n")
        for pc in paired_comparisons:
            pval = pc["Wilcoxon_p"]
            p_str = f"{pval:.4e}" if pval is not None and np.isfinite(pval) else "NA"
            lf.write(
                f"  {pc['Metric']}: Arm A={pc['Arm_A_Mean']:.4f}, Arm B={pc['Arm_B_Mean']:.4f}, "
                f"Diff={pc['Mean_Diff_A_minus_B']:.4f} "
                f"[95% CI: {pc['Diff_95CI_Low']:.4f}, {pc['Diff_95CI_High']:.4f}], "
                f"Wilcoxon p={p_str}\n"
            )

    logger.info("Longitudinal experiment complete! Elapsed time: %.2f seconds", manifest["elapsed_seconds"])
    return {
        "manifest": manifest,
        "paired_comparisons": paired_comparisons,
        "df_timelines": df_timelines,
        "verification": verification,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 6.2 longitudinal fatigue experiment.")
    parser.add_argument("--timelines", type=int, default=DEFAULT_N_TIMELINES, help="Number of simulation timelines")
    parser.add_argument("--days", type=float, default=DEFAULT_HORIZON_DAYS, help="Horizon length in days")
    parser.add_argument("--rate", type=float, default=DEFAULT_EMERGENCIES_PER_DAY, help="Emergencies per day")
    parser.add_argument("--donors", type=int, default=DEFAULT_N_DONORS, help="Donor pool size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip sensitivity analysis grid")
    args = parser.parse_args()

    run_longitudinal_experiment(
        n_timelines=args.timelines,
        horizon_days=args.days,
        emergencies_per_day=args.rate,
        n_donors=args.donors,
        seed=args.seed,
        run_sensitivity=not args.no_sensitivity,
    )
