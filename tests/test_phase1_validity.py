"""
tests/test_phase1_validity.py
Regression tests for Phase 1 data validity and traceability improvements.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.schemas.donor_schema import DonorProfile, DonorStream
from src.ranking.udi_engine import DonorCandidate
from src.data.scenario_manager import ScenarioManager
from src.data.synthetic_generator import SyntheticDonorRecord, SyntheticEmergencyScenario
from experiments.exp5_statistical_analysis import parse_and_test


def test_uci_donation_score_is_not_probability():
    """Test that a UCI donation score cannot be silently mapped to response probability."""
    profile = DonorProfile(
        donor_id="D1",
        stream=DonorStream.UCI_PILOT,
        donation_behavior_score=0.85
    )
    
    # Assert that response_probability is distinctly None
    assert profile.response_probability is None
    assert profile.donation_behavior_score == 0.85

    # Should raise error if Candidate tries to use it as probability without explicitly defining it
    with pytest.raises(ValueError, match="Neither recruitment response probability nor synthetic propensity provided."):
        cand = DonorCandidate(
            donor_id=profile.donor_id,
            travel_time_minutes=15.0
        )
        _ = cand.effective_probability


def test_exp5_refuses_missing_data(caplog):
    """Test that EXP-5 stops when actual data is missing instead of generating mock data."""
    res = parse_and_test("reports/non_existent_results.csv")
    assert res is None
    assert "Required experiment results not found" in caplog.text


def test_scenario_manager_roundtrip(tmp_path):
    """Test that scenarios can be saved and identically loaded to enforce paired testing."""
    manager = ScenarioManager(data_dir=str(tmp_path))
    
    # Create fake data
    from src.data.synthetic_generator import SyntheticDonorPool
    
    rec = SyntheticDonorRecord(donor_id="SYN1", synthetic_response_propensity=0.5)
    pool = SyntheticDonorPool(records=[rec])
    scen = SyntheticEmergencyScenario(request_id="REQ1")
    
    manager.save_scenario_set("test_set", pool, [scen], 42, {"config_key": "val"})
    
    loaded_pool, loaded_scen, loaded_meta = manager.load_scenario_set("test_set")
    
    assert loaded_pool.records[0].donor_id == "SYN1"
    assert loaded_pool.records[0].synthetic_response_propensity == 0.5
    assert loaded_scen[0].request_id == "REQ1"
    assert loaded_meta["seed"] == 42


def test_exp5_enforces_paired_scenarios(tmp_path):
    """Test that EXP-5 matches on Scenario_ID to ensure identical scenario instances."""
    csv_path = tmp_path / "exp4_raw_results.csv"
    
    # Unmatched scenarios
    df = pd.DataFrame([
        {"Policy": "Mass Broadcast", "Scenario_ID": "S1", "Time_To_Fulfill": 10, "Alerts_Sent": 10},
        {"Policy": "Dynamic Wave (Proposed)", "Scenario_ID": "S2", "Time_To_Fulfill": 15, "Alerts_Sent": 5},
    ])
    df.to_csv(csv_path, index=False)
    
    res = parse_and_test(str(csv_path))
    assert res is None  # Should fail due to 0 paired scenarios


def test_exp5_successful_paired_scenarios(tmp_path):
    """Test that EXP-5 succeeds when there are matched scenarios."""
    csv_path = tmp_path / "exp4_raw_results.csv"
    
    # Matched scenarios
    df = pd.DataFrame([
        {"Policy": "Mass Broadcast", "Scenario_ID": "S1", "Time_To_Fulfill": 10, "Alerts_Sent": 10},
        {"Policy": "Mass Broadcast", "Scenario_ID": "S2", "Time_To_Fulfill": 12, "Alerts_Sent": 12},
        {"Policy": "Dynamic Wave (Proposed)", "Scenario_ID": "S1", "Time_To_Fulfill": 15, "Alerts_Sent": 5},
        {"Policy": "Dynamic Wave (Proposed)", "Scenario_ID": "S2", "Time_To_Fulfill": 14, "Alerts_Sent": 6},
    ])
    df.to_csv(csv_path, index=False)
    
    res = parse_and_test(str(csv_path))
    assert res is not None
    assert len(res) == 2  # Returns [ttf_res, alerts_res]
