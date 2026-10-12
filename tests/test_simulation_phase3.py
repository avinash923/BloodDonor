"""
tests/test_simulation_phase3.py
Regression tests for Phase 3: Dynamic Recruitment Simulator Validation.
Validates the state machine, UDI dynamic re-ranking, and fatigue rules.
"""

import pytest
import math
from src.simulation.simulator import SimulationEpisode, SimEvent
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.response_model import StochasticResponseModel
from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.data.schemas.request_schema import EmergencyRequest
from src.data.schemas.dispatch_schema import DonorState
from datetime import datetime

class MockResponseModel(StochasticResponseModel):
    """Deterministic mock for response sampling."""
    def __init__(self, responses):
        super().__init__(seed=42)
        self.responses = responses # dict: donor_id -> (accepted, latency)

    def sample_response(self, probability: float, recent_alerts: int = 0):
        # The true model doesn't pass donor_id, but our mock needs to know who is being sampled.
        # Since we can't change the signature easily, we will pop from a list of predefined outcomes 
        # assuming the order of dispatch is known.
        return self.responses.pop(0)


def create_base_scenario(t_critical=60.0, units=1):
    req = EmergencyRequest(
        request_id="TEST_REQ_1",
        blood_type_required="O-",
        units_required=units,
        urgency="CRITICAL",
        site_latitude=0.0,
        site_longitude=0.0,
        request_timestamp=datetime.now()
    )
    # Using the new multiplicative logic
    udi_engine = UDIEngine(formula_mode="multiplicative", safety_margin=0.0, w_urgency=0, w_timeliness=1, w_probability=1)
    return req, udi_engine


def test_immediate_acceptance_and_arrival():
    req, udi_engine = create_base_scenario()
    
    candidates = [
        DonorCandidate(donor_id="D1", travel_time_minutes=15.0, synthetic_response_propensity=0.9, is_eligible=True)
    ]
    
    # Mock: accept immediately, latency = 5.0
    class DetResponse:
        def sample_response(self, probability: float, recent_alerts: int = 0):
            from src.simulation.response_model import DonorResponseOutcome
            return DonorResponseOutcome(accepted=True, latency_minutes=5.0)

    sim = SimulationEpisode(
        request=req,
        candidates={"D1": 0.9},
        ranked_pool=udi_engine.score_and_rank(candidates, "CRITICAL"),
        t_critical_minutes=60.0,
        response_model=DetResponse(),
        dispatch_policy=DispatchCascadePolicy(w1_mult=1.0, w_sub_mult=1.0),
        udi_engine=udi_engine,
        unranked_candidates=candidates
    )
    
    res = sim.run()
    assert res.quota_met
    assert res.units_fulfilled == 1
    assert res.time_to_fulfillment_minutes == 20.0 # 5 latency + 15 ETA
    assert res.dispatch_log[0].final_state == DonorState.ARRIVED
    assert candidates[0].recent_alert_count == 1 # Fatigue updated


def test_donor_declines_triggers_next_wave():
    req, udi_engine = create_base_scenario(units=1)
    
    candidates = [
        DonorCandidate(donor_id="D1", travel_time_minutes=10.0, synthetic_response_propensity=0.9),
        DonorCandidate(donor_id="D2", travel_time_minutes=15.0, synthetic_response_propensity=0.8)
    ]
    
    class MultiResponse:
        def __init__(self):
            self.calls = 0
        def sample_response(self, probability: float, recent_alerts: int = 0):
            from src.simulation.response_model import DonorResponseOutcome
            self.calls += 1
            if self.calls == 1:
                return DonorResponseOutcome(accepted=False, latency_minutes=2.0)
            return DonorResponseOutcome(accepted=True, latency_minutes=3.0)

    sim = SimulationEpisode(
        request=req,
        candidates={"D1": 0.9, "D2": 0.8},
        ranked_pool=udi_engine.score_and_rank(candidates, "CRITICAL"),
        t_critical_minutes=60.0,
        response_model=MultiResponse(),
        dispatch_policy=DispatchCascadePolicy(w1_mult=1.0, w_sub_mult=1.0),
        wave_timeout=10.0,
        udi_engine=udi_engine,
        unranked_candidates=candidates
    )
    
    res = sim.run()
    assert res.quota_met
    assert res.waves_used == 3
    assert res.total_notifications_sent == 2
    
    # Check states
    assert sim.donor_states["D1"] == DonorState.DECLINED
    assert sim.donor_states["D2"] == DonorState.ARRIVED
    
    # D2 dispatched at wave 2 which happens at t=10.0 (wave timeout)
    # D2 response = 10.0 + 3.0 = 13.0
    # D2 arrival = 13.0 + 15.0 = 28.0
    assert res.time_to_fulfillment_minutes == 28.0


def test_late_arrival_expires():
    req, udi_engine = create_base_scenario(units=1)
    
    # 50 min travel time, T_crit = 60
    candidates = [
        DonorCandidate(donor_id="D1", travel_time_minutes=50.0, synthetic_response_propensity=0.9)
    ]
    
    class SlowResponse:
        def sample_response(self, probability: float, recent_alerts: int = 0):
            from src.simulation.response_model import DonorResponseOutcome
            return DonorResponseOutcome(accepted=True, latency_minutes=15.0)

    sim = SimulationEpisode(
        request=req,
        candidates={"D1": 0.9},
        ranked_pool=udi_engine.score_and_rank(candidates, "CRITICAL"),
        t_critical_minutes=60.0,
        t_observation_horizon=60.0,
        response_model=SlowResponse(),
        dispatch_policy=DispatchCascadePolicy(w1_mult=1.0),
        udi_engine=udi_engine,
        unranked_candidates=candidates
    )
    
    res = sim.run()
    assert not res.quota_met
    assert sim.donor_states["D1"] == DonorState.EXPIRED
    assert res.dispatch_log[0].final_state == DonorState.EXPIRED


def test_final_state_is_string_and_trace_writer_handles_it():
    """
    Regression test for EXP-9 AttributeError.

    DispatchEvent.Config has `use_enum_values = True`, so Pydantic stores
    final_state as the enum's underlying str value rather than a DonorState
    instance.  The trace writer must NOT call .value on it; it must format it
    directly with str().

    This test confirms:
      1. final_state is a plain str at runtime, not a DonorState enum object.
      2. The str representation matches the expected DonorState name.
      3. A simulated trace write (f"{ev.final_state}") succeeds without
         AttributeError and produces the correct output.
    """
    req, udi_engine = create_base_scenario()

    candidates = [
        DonorCandidate(donor_id="D1", travel_time_minutes=5.0, synthetic_response_propensity=0.9, is_eligible=True)
    ]

    class ImmediateAccept:
        def sample_response(self, probability: float, recent_alerts: int = 0):
            from src.simulation.response_model import DonorResponseOutcome
            return DonorResponseOutcome(accepted=True, latency_minutes=2.0)

    sim = SimulationEpisode(
        request=req,
        candidates={"D1": 0.9},
        ranked_pool=udi_engine.score_and_rank(candidates, "CRITICAL"),
        t_critical_minutes=60.0,
        response_model=ImmediateAccept(),
        dispatch_policy=DispatchCascadePolicy(w1_mult=1.0, w_sub_mult=1.0),
        udi_engine=udi_engine,
        unranked_candidates=candidates,
    )

    res = sim.run()

    assert len(res.dispatch_log) == 1
    ev = res.dispatch_log[0]

    # 1. The field must be a plain str (use_enum_values=True strips the Enum wrapper)
    assert isinstance(ev.final_state, str), (
        f"Expected final_state to be str, got {type(ev.final_state)}"
    )

    # 2. The string value must match the expected state name
    assert ev.final_state == "ARRIVED", (
        f"Expected 'ARRIVED', got {ev.final_state!r}"
    )

    # 3. The trace writer pattern must not raise AttributeError
    try:
        serialized = f"  -> Final State: {ev.final_state}"
    except AttributeError as exc:
        pytest.fail(f"Trace writer raised AttributeError on ev.final_state: {exc}")

    assert "ARRIVED" in serialized

    # 4. Enum comparison still works because DonorState is str, Enum
    assert ev.final_state == DonorState.ARRIVED

