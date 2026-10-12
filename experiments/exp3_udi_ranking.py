"""
experiments/exp3_udi_ranking.py
EXP-3: UDI Multi-Criteria Ranking vs Operational Baselines

Compares five ranking strategies across 100 synthetic emergency scenarios
(3 urgency tiers × varied donor pools):
  1. Random Selection
  2. Proximity-First (Nearest donor)
  3. Probability-First (Highest ML propensity)
  4. Heuristic (Proximity × Probability product)
  5. UDI (Proposed multi-criteria ranking)

Evaluation Metrics:
  - Cumulative response yield at k=5, 10, 20 (Precision@k)
  - Mean travel time of top-k candidates (minutes)
  - Mean response probability of top-k candidates
  - Mean UDI score of top-k candidates
  - Kendall's τ rank correlation vs UDI ranking

DATA STREAM: SYNTHETIC_SIMULATION — all results are in-silico.
These are NOT real-world clinical findings.

Usage:
    python experiments/exp3_udi_ranking.py [--n-scenarios 100]
                                           [--n-donors 1000]
                                           [--seed 42]
                                           [--output-dir reports]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

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
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.spatial.feasibility_filter import FeasibilityFilter
from src.ranking.udi_engine import UDIEngine, DonorCandidate, precision_at_k
from src.ranking.baselines import (
    RandomRanking,
    ProximityFirstRanking,
    ProbabilityFirstRanking,
    HeuristicRanking,
)

logger = get_logger("exp3_udi_ranking")

URGENCY_TIERS = ["CRITICAL", "URGENT", "SCHEDULED"]
TOP_K_VALUES = [5, 10, 20]


# =================================================================
# Helpers
# =================================================================

def load_udi_config(config_path: str = "configs/udi_weights.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_candidates(
    pool_records: list,
    travel_times: np.ndarray,
    fallback_prob: float = 0.25,
) -> list[DonorCandidate]:
    """
    Convert synthetic donor records + precomputed travel times to DonorCandidate list.

    Uses synthetic_response_propensity from synthetic generator as the proxy for
    the ML-predicted probability since real recruitment response data is unavailable.
    """
    candidates = []
    for record, tt in zip(pool_records, travel_times):
        prob = float(np.clip(
            getattr(record, "synthetic_response_propensity", fallback_prob), 0.0, 1.0
        ))
        candidates.append(DonorCandidate(
            donor_id=record.donor_id,
            travel_time_minutes=float(tt),
            synthetic_response_propensity=prob,
            recent_alert_count=int(record.recent_alert_count),
            is_eligible=bool(record.is_eligible),
        ))
    return candidates


def rank_with_baseline(ranker, candidates, urgency, safety_margin=0.15):
    return ranker.rank(candidates, urgency=urgency, safety_margin=safety_margin)


def evaluate_ranking(
    ranked_results: list,
    candidates: list[DonorCandidate],
    k_values: list[int],
    simulated_responder_ids: set[str],
) -> dict:
    """
    Compute evaluation metrics for a given ranking result.

    Returns dict with Precision@k, mean travel time, mean probability,
    and mean UDI score for each k.
    """
    metrics = {}
    for k in k_values:
        top_k = ranked_results[:k]
        feasible_top_k = [r for r in top_k if r.is_feasible and r.is_eligible]

        if feasible_top_k:
            cand_map = {c.donor_id: c for c in candidates}
            mean_tt = float(np.mean([
                cand_map[r.donor_id].travel_time_minutes
                for r in feasible_top_k if r.donor_id in cand_map
            ]))
            mean_prob = float(np.mean([
                cand_map[r.donor_id].response_probability
                for r in feasible_top_k if r.donor_id in cand_map
            ]))
        else:
            mean_tt = float("nan")
            mean_prob = float("nan")

        p_at_k = precision_at_k(ranked_results, k, simulated_responder_ids)
        metrics[f"precision@{k}"] = p_at_k
        metrics[f"mean_tt@{k}"] = mean_tt
        metrics[f"mean_prob@{k}"] = mean_prob

    return metrics


def simulate_responders(
    candidates: list[DonorCandidate],
    seed_offset: int,
) -> set[str]:
    """
    Simulate ground-truth responders via Bernoulli trials using
    each candidate's effective_probability. Used for Precision@k evaluation.

    SYNTHETIC_SIMULATION: This is a stochastic simulation, not real outcomes.
    """
    rng = np.random.default_rng(seed_offset)
    responders = set()
    for c in candidates:
        if c.is_eligible and rng.random() < c.effective_probability:
            responders.add(c.donor_id)
    return responders


# =================================================================
# Main Experiment
# =================================================================

def run_experiment(
    n_scenarios: int = 100,
    n_donors: int = 1000,
    seed: int = 42,
    output_dir: str = "reports",
) -> None:
    t_start = time.perf_counter()
    set_global_seed(seed)

    logger.info("=" * 65)
    logger.info("EXP-3: UDI Multi-Criteria Ranking vs Baselines")
    logger.info("Data Stream: SYNTHETIC_SIMULATION (in-silico only)")
    logger.info("Scenarios: %d | Donors: %d | Seed: %d", n_scenarios, n_donors, seed)
    logger.info("=" * 65)

    # -----------------------------------------------------------------
    # 1. Generate synthetic data
    # -----------------------------------------------------------------
    logger.info("[1/5] Generating synthetic environment ...")
    gen = SyntheticUrbanGenerator(seed=seed)
    pool = gen.generate_donor_pool(n=n_donors)
    site_lats, site_lons, site_ids = gen.generate_sites(n_sites=5)
    scenarios = gen.generate_emergency_scenarios(
        n=n_scenarios, site_lats=site_lats, site_lons=site_lons, site_ids=site_ids,
    )

    # -----------------------------------------------------------------
    # 2. Load config and instantiate rankers
    # -----------------------------------------------------------------
    logger.info("[2/5] Instantiating rankers ...")
    cfg = load_udi_config()
    w = cfg["weights"]
    udi_engine = UDIEngine(
        w_urgency=w["w_urgency"],
        w_timeliness=w["w_timeliness"],
        w_probability=w["w_probability"],
        lambda_fatigue=cfg["fatigue_penalty"]["lambda_fatigue"],
        safety_margin=0.15,
        formula_mode=cfg.get("formula", {}).get("mode", "multiplicative"),
        probability_floor=cfg.get("formula", {}).get("probability_floor", 0.0),
    )
    baselines = {
        "Random":           RandomRanking(seed=seed),
        "Proximity-First":  ProximityFirstRanking(),
        "Probability-First": ProbabilityFirstRanking(),
        "Heuristic":        HeuristicRanking(),
    }

    # TravelTimeEngine for computing donor-to-site times
    engine_spatial = TravelTimeEngine(
        base_speed_kmh=30.0,
        distance_method="network_grid",
        congestion_model=CongestionModel(),
    )

    donor_lats = np.array([d.latitude for d in pool.records])
    donor_lons = np.array([d.longitude for d in pool.records])

    # -----------------------------------------------------------------
    # 3. Evaluate across all scenarios
    # -----------------------------------------------------------------
    logger.info("[3/5] Evaluating all rankers across %d scenarios ...", n_scenarios)

    all_rows = []
    for scen_idx, scenario in enumerate(scenarios):
        hour = scenario.hour_of_day
        urgency = scenario.urgency
        site_lat = scenario.site_latitude
        site_lon = scenario.site_longitude

        # Compute travel times for this scenario
        travel_times = engine_spatial.travel_time_minutes(
            donor_lats, donor_lons, site_lat, site_lon, hour_of_day=hour,
        )

        # Build candidate list
        candidates = build_candidates(
            pool.records, travel_times,
            fallback_prob=cfg["defaults"]["fallback_response_probability"],
        )

        # Simulate ground-truth responders
        responders = simulate_responders(candidates, seed_offset=seed + scen_idx)

        # UDI ranking
        udi_results = udi_engine.score_and_rank(candidates, urgency=urgency)

        # Baseline rankings
        baseline_results = {
            name: rank_with_baseline(ranker, candidates, urgency)
            for name, ranker in baselines.items()
        }

        # Evaluate
        for ranker_name, results in [("UDI", udi_results)] + list(baseline_results.items()):
            metrics = evaluate_ranking(results, candidates, TOP_K_VALUES, responders)
            row = {
                "Scenario": scenario.request_id,
                "Urgency": urgency,
                "Hour": hour,
                "Ranker": ranker_name,
                **metrics,
            }
            all_rows.append(row)

        if (scen_idx + 1) % 20 == 0:
            logger.info("  ... completed %d/%d scenarios", scen_idx + 1, n_scenarios)

    df = pd.DataFrame(all_rows)

    # -----------------------------------------------------------------
    # 4. Aggregate and save outputs
    # -----------------------------------------------------------------
    logger.info("[4/5] Aggregating results and saving reports ...")
    fig_dir = Path(output_dir) / "figures"
    tab_dir = Path(output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Summary table: mean metrics per ranker and urgency
    agg = (
        df.groupby(["Ranker", "Urgency"])[
            [f"precision@{k}" for k in TOP_K_VALUES] +
            [f"mean_tt@{k}" for k in TOP_K_VALUES] +
            [f"mean_prob@{k}" for k in TOP_K_VALUES]
        ]
        .mean()
        .round(4)
        .reset_index()
    )

    md = "# EXP-3: UDI Ranking vs Baselines (SYNTHETIC_SIMULATION)\n\n"
    md += "> **Data Stream**: SYNTHETIC_SIMULATION — results are in-silico only.\n\n"
    md += agg.to_markdown(index=False)
    (tab_dir / "udi_ranking_summary.md").write_text(md, encoding="utf-8")
    logger.info("Table saved: %s", tab_dir / "udi_ranking_summary.md")

    # Plots
    _plot_precision_at_k_bars(agg, fig_dir)
    _plot_mean_travel_time(agg, fig_dir)

    # -----------------------------------------------------------------
    # 5. Console summary
    # -----------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 65)
    print("EXP-3 RESULTS — UDI vs Baselines (SYNTHETIC_SIMULATION)")
    print("=" * 65)
    print(f"\n{'Ranker':<26} {'P@5':>8} {'P@10':>8} {'P@20':>8} {'MeanTT@10':>12}")
    print("-" * 68)
    ranker_order = ["UDI", "Proximity-First", "Probability-First", "Heuristic", "Random"]
    for rname in ranker_order:
        sub = df[df["Ranker"] == rname]
        if sub.empty:
            continue
        p5 = sub["precision@5"].mean()
        p10 = sub["precision@10"].mean()
        p20 = sub["precision@20"].mean()
        tt10 = sub["mean_tt@10"].mean()
        print(f"{rname:<26} {p5:>8.4f} {p10:>8.4f} {p20:>8.4f} {tt10:>12.2f}")
    print("=" * 65)
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Outputs: {output_dir}/")
    print("\nNOTE: All results from SYNTHETIC_SIMULATION stream.")
    print("These are NOT real-world clinical observations.")


def _plot_precision_at_k_bars(agg: pd.DataFrame, fig_dir: Path) -> None:
    urgencies = agg["Urgency"].unique()
    ranker_order = ["UDI", "Proximity-First", "Probability-First", "Heuristic", "Random"]
    colors = {"UDI": "#2E86AB", "Proximity-First": "#3BB273",
              "Probability-First": "#F7B731", "Heuristic": "#9B59B6",
              "Random": "#BDC3C7"}

    fig, axes = plt.subplots(1, len(urgencies), figsize=(14, 5), sharey=True)
    if len(urgencies) == 1:
        axes = [axes]

    for ax, urgency in zip(axes, sorted(urgencies)):
        sub = agg[agg["Urgency"] == urgency]
        x = np.arange(len(ranker_order))
        width = 0.25
        for j, k in enumerate(TOP_K_VALUES):
            vals = []
            for rname in ranker_order:
                row = sub[sub["Ranker"] == rname]
                vals.append(float(row[f"precision@{k}"].values[0]) if not row.empty else 0.0)
            ax.bar(x + j * width, vals, width, label=f"P@{k}",
                   color=[colors.get(rname, "#999") for rname in ranker_order],
                   alpha=0.8 - j * 0.15)
        ax.set_xticks(x + width)
        ax.set_xticklabels(ranker_order, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{urgency}", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Precision@k", fontsize=11)
    fig.suptitle("EXP-3: Precision@k by Ranker & Urgency Tier (SYNTHETIC_SIMULATION)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "precision_at_k_exp3.png", dpi=150)
    plt.close(fig)


def _plot_mean_travel_time(agg: pd.DataFrame, fig_dir: Path) -> None:
    """Bar chart of mean travel time of top-10 candidates per ranker (URGENT tier)."""
    sub = agg[agg["Urgency"] == "URGENT"]
    rankers = sub["Ranker"].tolist()
    tt = sub["mean_tt@10"].tolist()
    colors_list = ["#2E86AB", "#3BB273", "#F7B731", "#9B59B6", "#BDC3C7"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(rankers, tt, color=colors_list[:len(rankers)], edgecolor="white")
    ax.set_xlabel("Mean Travel Time of Top-10 Candidates (min)", fontsize=11)
    ax.set_title("EXP-3: Mean Travel Time @ Top-10 — URGENT Tier (SYNTHETIC_SIMULATION)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, tt):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_travel_time_exp3.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-3: UDI Ranking vs Baselines (SYNTHETIC_SIMULATION)"
    )
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--n-donors", type=int, default=1000)
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
