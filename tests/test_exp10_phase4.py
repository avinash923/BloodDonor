"""
tests/test_exp10_phase4.py
Regression tests for Phase 4 Comparative Experiments.
Validates reproducibility, experimental constraints, and metric calculation correctness.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.seed import set_global_seed
from experiments.exp10_phase4_comparisons import run_phase4_experiments
from src.evaluation.stats_testing import wilcoxon_test, StatTestResult
import scipy.stats as stats

def test_statistical_pairing_mechanics():
    """Test that paired McNemar and Wilcoxon calculate appropriately on controlled data."""
    # Mock data
    np.random.seed(42)
    # 10 cases: both success, 5 cases A success B fail, 3 cases A fail B success, 2 cases both fail
    # Policy A
    quota_a = [True]*15 + [False]*5
    time_a = list(np.random.normal(30, 5, 10)) + [np.nan]*5 + [np.nan]*5
    
    # Policy B
    quota_b = [True]*10 + [False]*5 + [True]*3 + [False]*2
    time_b = list(np.random.normal(35, 5, 10)) + [np.nan]*5 + list(np.random.normal(40, 5, 3)) + [np.nan]*2
    
    df = pd.DataFrame({
        "Scenario_ID": list(range(20)) * 2,
        "Policy": ["A"]*20 + ["B"]*20,
        "Quota_Met_Deadline": quota_a + quota_b,
        "Quota_Met_Eventual": quota_a + quota_b,
        "Time_To_Fulfill_Eventual": time_a + time_b
    })
    
    ref_df = df[df["Policy"] == "A"].sort_values("Scenario_ID").set_index("Scenario_ID")
    comp_df = df[df["Policy"] == "B"].sort_values("Scenario_ID").set_index("Scenario_ID")
    
    # Exact McNemar's Test for Quota Met
    b = ((comp_df["Quota_Met_Deadline"] == True) & (ref_df["Quota_Met_Deadline"] == False)).sum() # B success, A fail => 3
    c = ((comp_df["Quota_Met_Deadline"] == False) & (ref_df["Quota_Met_Deadline"] == True)).sum() # B fail, A success => 5
    n_discordant = b + c
    
    # Continuity corrected stat
    mcnemar_stat = (abs(b - c) - 1.0)**2 / n_discordant
    
    try:
        mcnemar_exact_p = stats.binomtest(min(b, c), n_discordant, 0.5).pvalue
    except AttributeError:
        mcnemar_exact_p = stats.binom_test(min(b, c), n_discordant, 0.5)
        
    assert b == 3
    assert c == 5
    assert mcnemar_stat == (1.0**2) / 8
    assert mcnemar_exact_p > 0.0
    
    # Wilcoxon conditional test
    joint_success = comp_df["Quota_Met_Eventual"] & ref_df["Quota_Met_Eventual"]
    assert joint_success.sum() == 10
    
    tt_comp = comp_df.loc[joint_success, "Time_To_Fulfill_Eventual"].values
    tt_ref = ref_df.loc[joint_success, "Time_To_Fulfill_Eventual"].values
    
    valid_mask = np.isfinite(tt_comp) & np.isfinite(tt_ref)
    assert valid_mask.sum() == 10
    
    w_res = wilcoxon_test(tt_comp, tt_ref)
    assert w_res.p_value >= 0.0 and w_res.p_value <= 1.0


def test_fairness_across_policies():
    """Ensure different policies face the exact same initial candidates, without state leakage."""
    from src.data.synthetic_generator import SyntheticUrbanGenerator
    from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
    from experiments.exp4_dynamic_recruitment import build_candidates
    
    gen = SyntheticUrbanGenerator(seed=42)
    pool = gen.generate_donor_pool(n=10)
    
    donor_lats = [d.latitude for d in pool.records]
    donor_lons = [d.longitude for d in pool.records]
    
    tt_engine = TravelTimeEngine(distance_method="network_grid", congestion_model=CongestionModel())
    tt = tt_engine.travel_time_minutes(donor_lats, donor_lons, 0.0, 0.0, 12)
    
    # Run 1
    candidates1, probs1 = build_candidates(pool.records, tt)
    
    # Modify candidates1 to simulate mutation during an episode
    candidates1[0].recent_alert_count += 1
    candidates1[0].is_eligible = False
    candidates1[1].synthetic_response_propensity = 0.0
    
    # Run 2
    candidates2, probs2 = build_candidates(pool.records, tt)
    
    # Verify state isolation: Mutations in candidates1 did not affect pool.records or candidates2
    for c1, c2, r in zip(candidates1, candidates2, pool.records):
        assert c2.donor_id == r.donor_id
        assert c2.recent_alert_count == r.recent_alert_count
        assert c2.is_eligible == r.is_eligible
        
    # Check specific fields were completely separated
    assert candidates2[0].recent_alert_count != candidates1[0].recent_alert_count
    assert candidates2[0].is_eligible != candidates1[0].is_eligible
    assert candidates2[1].synthetic_response_propensity != candidates1[1].synthetic_response_propensity
    
    # Check travel times and true probabilities are identical
    for c1, c2 in zip(candidates1, candidates2):
        assert c1.travel_time_minutes == c2.travel_time_minutes
    assert probs1 == probs2
