"""
experiments/exp2_spatial_feasibility.py
EXP-2: Spatial Feasibility Evaluation — Travel Time Models

Compares three distance/travel-time approaches:
  1. Euclidean   — planar approximation (baseline)
  2. Haversine   — geodesic great-circle (standard reference)
  3. Network Grid — Haversine × circuity detour factor (urban-realistic)

Evaluation Protocol:
  - 100 synthetic emergency scenarios (SYNTHETIC_SIMULATION stream)
  - 3 urgency tiers × 4 representative hours of day
  - Metrics: feasibility rate, mean travel time, candidate retention rate,
             distance MAE vs Haversine reference, Kendall's τ rank correlation

DATA STREAM: SYNTHETIC_SIMULATION — all scenarios are in-silico.
Results are NOT real-world clinical evidence.

Usage:
    python experiments/exp2_spatial_feasibility.py [--n-scenarios 100]
                                                   [--seed 42]
                                                   [--output-dir reports]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
from src.spatial.distance_metrics import pairwise_distance_matrix
from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
from src.spatial.feasibility_filter import FeasibilityFilter

logger = get_logger("exp2_spatial_feasibility")


# =================================================================
# Helpers
# =================================================================

def kendall_tau(ranks_a: np.ndarray, ranks_b: np.ndarray) -> float:
    """Compute Kendall's τ between two rank orderings."""
    from scipy.stats import kendalltau
    tau, _ = kendalltau(ranks_a, ranks_b)
    return float(tau)


def build_engine(method: str, base_speed: float = 30.0) -> TravelTimeEngine:
    return TravelTimeEngine(
        base_speed_kmh=base_speed,
        distance_method=method,
        congestion_model=CongestionModel(),
    )


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
    logger.info("EXP-2: Spatial Feasibility Evaluation")
    logger.info("Data Stream: SYNTHETIC_SIMULATION (in-silico only)")
    logger.info("Scenarios: %d | Donors: %d | Seed: %d", n_scenarios, n_donors, seed)
    logger.info("=" * 65)

    # -----------------------------------------------------------------
    # 1. Generate synthetic data
    # -----------------------------------------------------------------
    logger.info("[1/5] Generating synthetic urban environment ...")
    gen = SyntheticUrbanGenerator(seed=seed)
    pool = gen.generate_donor_pool(n=n_donors)
    site_lats, site_lons, site_ids = gen.generate_sites(n_sites=5)
    scenarios = gen.generate_emergency_scenarios(
        n=n_scenarios,
        site_lats=site_lats, site_lons=site_lons, site_ids=site_ids,
    )

    donor_lats = np.array([d.latitude for d in pool.eligible_donors()])
    donor_lons = np.array([d.longitude for d in pool.eligible_donors()])
    donor_ids = [d.donor_id for d in pool.eligible_donors()]
    n_eligible = len(donor_ids)
    logger.info("Eligible donors for spatial evaluation: %d", n_eligible)

    # -----------------------------------------------------------------
    # 2. Distance method comparison
    # -----------------------------------------------------------------
    logger.info("[2/5] Computing pairwise distance matrices (3 methods) ...")
    methods = ["euclidean", "haversine", "network_grid"]
    method_labels = {"euclidean": "Euclidean", "haversine": "Haversine",
                     "network_grid": "Network Grid"}

    # Use first site as reference for pairwise distance comparison
    ref_site_lat, ref_site_lon = float(site_lats[0]), float(site_lons[0])

    dist_results = {}
    for method in methods:
        D = pairwise_distance_matrix(
            donor_lats, donor_lons,
            np.array([ref_site_lat]), np.array([ref_site_lon]),
            method=method,
        )
        dist_results[method] = D[:, 0]  # shape (N,)

    # MAE relative to Haversine reference
    hav_dists = dist_results["haversine"]
    for method in ["euclidean", "network_grid"]:
        mae = float(np.mean(np.abs(dist_results[method] - hav_dists)))
        logger.info("  %s vs Haversine MAE: %.4f km", method_labels[method], mae)

    # Rank correlation (Kendall's τ) relative to Haversine ranking
    hav_ranks = np.argsort(np.argsort(hav_dists))
    rank_taus = {}
    for method in ["euclidean", "network_grid"]:
        ranks_m = np.argsort(np.argsort(dist_results[method]))
        rank_taus[method] = kendall_tau(hav_ranks, ranks_m)
    logger.info("  Euclidean rank τ vs Haversine: %.4f", rank_taus["euclidean"])
    logger.info("  Network Grid rank τ vs Haversine: %.4f", rank_taus["network_grid"])

    # -----------------------------------------------------------------
    # 3. Feasibility analysis across methods and hours
    # -----------------------------------------------------------------
    logger.info("[3/5] Feasibility analysis across methods and urgency tiers ...")
    test_hours = [2, 8, 14, 18]   # off-peak, morning rush, afternoon, evening rush
    urgency_tiers = ["CRITICAL", "URGENT", "SCHEDULED"]

    rows = []
    for method in methods:
        engine = build_engine(method)
        ff = FeasibilityFilter(travel_time_engine=engine, safety_margin_fraction=0.15)
        for hour in test_hours:
            hour_scenarios = [
                {"request_id": s.request_id, "site_lat": s.site_latitude,
                 "site_lon": s.site_longitude, "urgency": s.urgency, "hour_of_day": hour}
                for s in scenarios
            ]
            results = ff.batch_filter(donor_ids, donor_lats, donor_lons, hour_scenarios)
            for urgency in urgency_tiers:
                tier_results = [r for r in results if r.urgency == urgency]
                if not tier_results:
                    continue
                feas_rates = [r.feasibility_rate for r in tier_results]
                mean_tt = [r.mean_feasible_travel_time for r in tier_results
                           if r.mean_feasible_travel_time is not None]
                rows.append({
                    "Method": method_labels[method],
                    "Hour": hour,
                    "Urgency": urgency,
                    "Mean Feasibility Rate": float(np.mean(feas_rates)),
                    "Std Feasibility Rate": float(np.std(feas_rates)),
                    "Mean Feasible Travel Time (min)": float(np.mean(mean_tt)) if mean_tt else None,
                    "N Scenarios": len(tier_results),
                })

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # 4. Save outputs
    # -----------------------------------------------------------------
    logger.info("[4/5] Saving tables and figures ...")
    fig_dir = Path(output_dir) / "figures"
    tab_dir = Path(output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    table_path = tab_dir / "spatial_feasibility_summary.md"
    md = "# EXP-2: Spatial Feasibility Analysis (SYNTHETIC_SIMULATION)\n\n"
    md += "> **Data Stream**: SYNTHETIC_SIMULATION — results are in-silico only.\n\n"
    md += df.to_markdown(index=False)
    table_path.write_text(md, encoding="utf-8")
    logger.info("Table saved: %s", table_path)

    # Feasibility rate heatmap by method and hour for URGENT tier
    _plot_feasibility_heatmap(df, fig_dir)

    # Distance distribution comparison
    _plot_distance_distributions(dist_results, method_labels, fig_dir)

    # -----------------------------------------------------------------
    # 5. Console summary
    # -----------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 65)
    print("EXP-2 RESULTS — Spatial Feasibility (SYNTHETIC_SIMULATION)")
    print("=" * 65)
    print(f"\nDistance Method Rank Concordance vs. Haversine:")
    print(f"  Euclidean    Kendall τ = {rank_taus['euclidean']:.4f}")
    print(f"  Network Grid Kendall τ = {rank_taus['network_grid']:.4f}")
    print(f"\nFeasibility Rates (URGENT tier, all hours):")
    urgent_df = df[df["Urgency"] == "URGENT"][["Method", "Hour", "Mean Feasibility Rate"]]
    for _, row in urgent_df.iterrows():
        print(f"  {row['Method']:<16} Hour={int(row['Hour']):02d}h  "
              f"Feasibility={row['Mean Feasibility Rate']:.1%}")
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Outputs: {output_dir}/")
    print("\nNOTE: All results from SYNTHETIC_SIMULATION stream.")
    print("These are NOT real-world clinical observations.")


def _plot_feasibility_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    urgent = df[df["Urgency"] == "URGENT"].copy()
    if urgent.empty:
        return
    pivot = urgent.pivot_table(
        index="Method", columns="Hour",
        values="Mean Feasibility Rate", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h:02d}:00" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Mean Feasibility Rate")
    ax.set_title("EXP-2: Donor Feasibility Rate by Method & Hour (URGENT tier)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "feasibility_heatmap_exp2.png", dpi=150)
    plt.close(fig)


def _plot_distance_distributions(
    dist_results: dict, method_labels: dict, fig_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"euclidean": "#2E86AB", "haversine": "#3BB273", "network_grid": "#E84855"}
    for method, dists in dist_results.items():
        ax.hist(dists, bins=40, alpha=0.6, color=colors[method],
                label=method_labels[method], density=True)
    ax.set_xlabel("Distance to Reference Site (km)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("EXP-2: Distance Distribution Comparison (SYNTHETIC_SIMULATION)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "distance_distributions_exp2.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-2: Spatial Feasibility Evaluation (SYNTHETIC_SIMULATION)"
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
