"""
experiments/exp5_statistical_analysis.py
EXP-5: Statistical Evaluation of Recruitment Policies

Runs paired hypothesis testing on the outcomes of Mass Broadcast vs Dynamic Wave policies.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.evaluation.stats_testing import paired_ttest, wilcoxon_test
from experiments.exp4_dynamic_recruitment import run_experiment as run_exp4

logger = get_logger("exp5_statistical_analysis")


def parse_and_test(report_path: str = "reports/exp4_raw_results.csv"):
    """
    Parses the CSV data from EXP-4 to run paired statistical tests.
    """
    logger.info("Running paired statistical tests on EXP-4 results...")
    
    file_path = Path(report_path)
    if not file_path.exists():
        logger.error(f"Required experiment results not found: {report_path}")
        logger.error("Blocked: Run experiments/exp4_dynamic_recruitment.py first to generate actual outcomes.")
        return None
        
    df = pd.read_csv(file_path)
    
    # Ensure paired observations by matching Scenario_ID
    mass = df[df["Policy"] == "Mass Broadcast"].set_index("Scenario_ID")
    dynamic = df[df["Policy"] == "Dynamic Wave (Proposed)"].set_index("Scenario_ID")
    
    # Inner join on scenarios to strictly enforce pairing
    paired_scenarios = mass.index.intersection(dynamic.index)
    
    if len(paired_scenarios) == 0:
        logger.error("No matching scenario IDs found between policies. Cannot perform paired tests.")
        return None
        
    logger.info(f"Found {len(paired_scenarios)} paired scenario observations.")
    
    # 1. T-Test on TTF (ignoring NaNs from unmet quotas)
    ttf_mass = mass.loc[paired_scenarios, "Time_To_Fulfill"].dropna()
    ttf_dyn = dynamic.loc[paired_scenarios, "Time_To_Fulfill"].dropna()
    common_ttf = ttf_mass.index.intersection(ttf_dyn.index)
    
    if len(common_ttf) > 1:
        ttf_res = paired_ttest(ttf_dyn.loc[common_ttf].values, ttf_mass.loc[common_ttf].values)
        logger.info("Time-To-Fulfill (Dynamic vs Mass): %s", ttf_res)
    else:
        logger.warning("Not enough paired TTF observations for t-test.")
        ttf_res = None
    
    # 2. Wilcoxon on Alerts Sent
    alerts_mass = mass.loc[paired_scenarios, "Alerts_Sent"].values
    alerts_dyn = dynamic.loc[paired_scenarios, "Alerts_Sent"].values
    
    alerts_res = wilcoxon_test(alerts_dyn, alerts_mass)
    logger.info("Alerts Sent (Dynamic vs Mass): %s", alerts_res)
    
    return [ttf_res, alerts_res]

if __name__ == "__main__":
    results = parse_and_test("reports/exp4_raw_results.csv")
    if results:
        print("\nEXP-5 Statistical Tests Complete.")
    else:
        print("\nEXP-5 FAILED: Run EXP-4 first to generate data.")
