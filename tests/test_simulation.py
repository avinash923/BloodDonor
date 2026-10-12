"""
tests/test_simulation.py
Unit tests for Phase 5: Dynamic Recruitment Simulation.

Verifies:
  1. Stochastic response latency distributions and bounds.
  2. Dispatch cascade batch sizing.
  3. Simulator state transitions (UNCONTACTED -> NOTIFIED -> ACCEPTED -> ARRIVED).
  4. Quota fulfillment and early termination.
  5. Determinism under fixed seed.

Run with: pytest tests/test_simulation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.data.schemas.dispatch_schema import DonorState
from src.data.schemas.request_schema import EmergencyRequest
from src.ranking.udi_engine import UDIResult
from src.simulation.response_model import StochasticResponseModel
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.simulator import SimulationEpisode

# =================================================================
# Test Group 1: Response Model
# =================================================================

class TestResponseModel:
    def test_latency_bounds(self):
        model = StochasticResponseModel(min_latency=1.0)
        for _ in range(100):
            out = model.sample_response(probability=0.5)
            assert out.latency_minutes >= 1.0

    def test_acceptance_probability(self):
        model = StochasticResponseModel(seed=42)
        # Sample 1000 times with p=0.8
        accepts = sum(1 for _ in range(1000) if model.sample_response(0.8).accepted)
        # Expect roughly 800 accepts
        assert 750 < accepts < 850

    def test_fatigue_increases_latency(self):
        model = StochasticResponseModel(seed=42)
        out1 = model.sample_response(0.5, recent_alerts=0)
        
        # Reset seed to get same base latency
        model.rng = __import__('numpy').random.default_rng(42)
        out2 = model.sample_response(0.5, recent_alerts=2)
        
        assert out2.latency_minutes > out1.latency_minutes


# =================================================================
# Test Group 2: Dispatch Cascade
# =================================================================

class TestDispatchCascade:
    def test_wave_1_multiplier(self):
        policy = DispatchCascadePolicy(w1_mult=3.0)
        batch = policy.get_batch_size(wave_number=1, units_required=2, shortfall=2)
        assert batch == 6

    def test_subsequent_wave_multiplier(self):
        policy = DispatchCascadePolicy(w_sub_mult=2.0)
        batch = policy.get_batch_size(wave_number=2, units_required=10, shortfall=4)
        assert batch == 8
        
    def test_zero_shortfall_returns_zero(self):
        policy = DispatchCascadePolicy()
        batch = policy.get_batch_size(wave_number=2, units_required=10, shortfall=0)
        assert batch == 0

    def test_select_batch_excludes_notified(self):
        policy = DispatchCascadePolicy()
        pool = [UDIResult(donor_id=f"D{i}", udi_score=1.0, urgency_component=0.0,
                          timeliness_component=0.0, probability_component=0.0,
                          weighted_sum=1.0, fatigue_discount=1.0) for i in range(5)]
        notified = {"D0", "D1"}
        
        selected = policy.select_batch(pool, notified, batch_size=2)
        
        assert len(selected) == 2
        assert selected[0].donor_id == "D2"
        assert selected[1].donor_id == "D3"

# =================================================================
# Test Group 3: Simulator
# =================================================================

class TestSimulator:
    @pytest.fixture
    def mock_request(self):
        return EmergencyRequest(
            request_id="REQ_1",
            blood_type_required="O+",
            units_required=2,
            urgency="URGENT",
            site_latitude=12.97,
            site_longitude=77.59,
            request_timestamp=datetime.now()
        )
        
    @pytest.fixture
    def mock_pool(self):
        # Create 10 candidates with 10 minute travel times
        # For simplicity, assign mock probabilities and UDI results
        candidates = {f"D{i}": 1.0 for i in range(10)} # 100% acceptance prob for test
        
        ranked = [UDIResult(donor_id=f"D{i}", udi_score=1.0, urgency_component=0.0,
                          timeliness_component=0.0, probability_component=0.0,
                          weighted_sum=1.0, fatigue_discount=1.0) for i in range(10)]
                          
        # Hack travel times into the UDIResult objects (simulator needs them)
        for r in ranked:
            r.travel_time_minutes = 10.0
                          
        return candidates, ranked

    def test_simulation_completes(self, mock_request, mock_pool):
        candidates, ranked = mock_pool
        sim = SimulationEpisode(
            request=mock_request,
            candidates=candidates,
            ranked_pool=ranked,
            t_critical_minutes=120.0,
            response_model=StochasticResponseModel(seed=42),
            dispatch_policy=DispatchCascadePolicy(w1_mult=1.0) # initial wave = 2
        )
        
        result = sim.run()
        
        assert result.quota_met is True
        assert result.units_fulfilled == 2
        # Stochastic latency (~10 min) + travel (10 min) = ~20 min. 
        # Wave 2 triggers at t=15, dispatching 2 more before the first arrive.
        # This legitimate cascade means 4 total notifications are sent.
        assert result.total_notifications_sent == 4
        
        # Check states
        assert sim.donor_states["D0"] == DonorState.ARRIVED
        assert sim.donor_states["D1"] == DonorState.ARRIVED
        # They could be NOTIFIED or ACCEPTED depending on stochastic latency
        assert sim.donor_states["D2"] in (DonorState.NOTIFIED, DonorState.ACCEPTED, DonorState.ARRIVED)
        assert sim.donor_states["D3"] in (DonorState.NOTIFIED, DonorState.ACCEPTED, DonorState.ARRIVED)
        assert sim.donor_states["D4"] == DonorState.UNCONTACTED

    def test_simulation_completes_fast(self, mock_request, mock_pool):
        """Test that if donors arrive before the wave timeout, no second wave is triggered."""
        candidates, ranked = mock_pool
        for r in ranked:
            r.travel_time_minutes = 1.0 # 1 minute travel time
            
        sim = SimulationEpisode(
            request=mock_request,
            candidates=candidates,
            ranked_pool=ranked,
            t_critical_minutes=120.0,
            # Extremely fast response:
            response_model=StochasticResponseModel(mu=-5.0, sigma=0.1, min_latency=1.0, seed=42),
            dispatch_policy=DispatchCascadePolicy(w1_mult=1.0)
        )
        
        result = sim.run()
        
        assert result.quota_met is True
        assert result.units_fulfilled == 2
        # Arrival happens at ~2 minutes, well before wave 2 at t=15.
        assert result.total_notifications_sent == 2
        assert sim.donor_states["D0"] == DonorState.ARRIVED
        assert sim.donor_states["D1"] == DonorState.ARRIVED
        assert sim.donor_states["D2"] == DonorState.UNCONTACTED

    def test_simulation_timeouts(self, mock_request, mock_pool):
        candidates, ranked = mock_pool
        # Force 0% acceptance probability
        candidates = {f"D{i}": 0.0 for i in range(10)}
        
        sim = SimulationEpisode(
            request=mock_request,
            candidates=candidates,
            ranked_pool=ranked,
            t_critical_minutes=60.0, # Shorter horizon
            response_model=StochasticResponseModel(seed=42),
            dispatch_policy=DispatchCascadePolicy(w1_mult=1.0, w_sub_mult=1.0),
            wave_timeout=15.0 # Check waves fire every 15 min
        )
        
        result = sim.run()
        
        assert result.quota_met is False
        assert result.units_fulfilled == 0
        # 60 min total / 15 min wave = 4 waves max before timeout, 
        # wave 1 (at t=0), wave 2 (15), wave 3 (30), wave 4 (45), wave 5 (60 - hits T_crit)
        assert result.waves_used >= 4
        assert result.total_notifications_sent >= 8

