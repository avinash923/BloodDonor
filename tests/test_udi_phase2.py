"""
tests/test_udi_phase2.py
Regression tests for Phase 2/3 UDI engine improvements:
- Multiplicative formula behaviour
- Probability floor in multiplicative mode
- Deterministic tie-breaking (ETA → probability → donor_id)
- Additive vs multiplicative divergence on edge cases
"""

import pytest
import math
from src.ranking.udi_engine import UDIEngine, DonorCandidate, UDIResult


# ---------------------------------------------------------------
# Multiplicative formula tests
# ---------------------------------------------------------------

class TestMultiplicativeFormula:
    """Tests for the approved multiplicative UDI formula."""

    def test_basic_scoring(self):
        """Multiplicative UDI = P × T_score × Fatigue, no urgency scalar."""
        engine = UDIEngine(formula_mode="multiplicative", safety_margin=0.15)
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=10.0,
            synthetic_response_propensity=0.8,
            recent_alert_count=0,
        )
        result = engine.score_candidate(candidate, "CRITICAL")
        # T_crit=60, T_buffer=9.0, T_budget=51.0
        # T_score = 1 - 10/51 ≈ 0.8039
        # UDI = 0.8 × 0.8039 × 1.0 ≈ 0.6431
        assert result.is_feasible
        assert result.is_eligible
        assert 0.60 < result.udi_score < 0.70
        # urgency_component should be 0 in multiplicative mode
        assert result.urgency_component == 0.0

    def test_zero_probability_zeroes_score(self):
        """A zero probability must zero out the multiplicative score (no floor by default)."""
        engine = UDIEngine(formula_mode="multiplicative", probability_floor=0.0)
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=5.0,
            synthetic_response_propensity=0.0,
        )
        result = engine.score_candidate(candidate, "CRITICAL")
        assert result.is_feasible  # they can physically arrive
        assert result.udi_score == 0.0

    def test_infeasible_if_eta_exceeds_budget(self):
        """Donor beyond T_budget is infeasible."""
        engine = UDIEngine(formula_mode="multiplicative", safety_margin=0.15)
        # CRITICAL: T_crit=60, T_buffer=9, T_budget=51
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=52.0,
            synthetic_response_propensity=0.9,
        )
        result = engine.score_candidate(candidate, "CRITICAL")
        assert not result.is_feasible
        assert result.udi_score == 0.0

    def test_missing_eta_is_infeasible(self):
        """Donor with no travel time estimate is infeasible."""
        engine = UDIEngine(formula_mode="multiplicative")
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=None,
            synthetic_response_propensity=0.5,
        )
        result = engine.score_candidate(candidate, "URGENT")
        assert not result.is_feasible
        assert result.udi_score == 0.0

    def test_ineligible_donor_zeroed(self):
        """Ineligible donors get UDI=0 regardless of other factors."""
        engine = UDIEngine(formula_mode="multiplicative")
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=5.0,
            synthetic_response_propensity=0.9,
            is_eligible=False,
        )
        result = engine.score_candidate(candidate, "CRITICAL")
        assert not result.is_eligible
        assert result.udi_score == 0.0

    def test_fatigue_discounts_score(self):
        """Alert fatigue reduces the final score."""
        engine = UDIEngine(formula_mode="multiplicative", lambda_fatigue=0.30)
        base = DonorCandidate(
            donor_id="D_base",
            travel_time_minutes=10.0,
            synthetic_response_propensity=0.8,
            recent_alert_count=0,
        )
        fatigued = DonorCandidate(
            donor_id="D_fatigued",
            travel_time_minutes=10.0,
            synthetic_response_propensity=0.8,
            recent_alert_count=5,
        )
        r_base = engine.score_candidate(base, "CRITICAL")
        r_fatigued = engine.score_candidate(fatigued, "CRITICAL")
        assert r_base.udi_score > r_fatigued.udi_score
        assert r_fatigued.fatigue_discount < r_base.fatigue_discount
        expected_fatigue = math.exp(-0.30 * 5)
        assert abs(r_fatigued.fatigue_discount - expected_fatigue) < 1e-6

    def test_urgency_changes_timeliness_curve(self):
        """CRITICAL vs SCHEDULED should produce different T_scores for same ETA."""
        engine = UDIEngine(formula_mode="multiplicative", safety_margin=0.15)
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=30.0,
            synthetic_response_propensity=0.5,
        )
        r_crit = engine.score_candidate(candidate, "CRITICAL")
        r_sched = engine.score_candidate(candidate, "SCHEDULED")
        # CRITICAL: T_budget=51, 30/51≈0.588, T_score≈0.412
        # SCHEDULED: T_budget=204, 30/204≈0.147, T_score≈0.853
        # With same P, SCHEDULED T_score >> CRITICAL T_score
        assert r_sched.timeliness_component > r_crit.timeliness_component
        assert r_sched.udi_score > r_crit.udi_score


# ---------------------------------------------------------------
# Probability floor tests
# ---------------------------------------------------------------

class TestProbabilityFloor:
    """Tests for the configurable probability floor."""

    def test_floor_prevents_zero(self):
        """With floor=0.01, a zero probability is raised to 0.01."""
        engine = UDIEngine(formula_mode="multiplicative", probability_floor=0.01)
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=5.0,
            synthetic_response_propensity=0.0,
        )
        result = engine.score_candidate(candidate, "CRITICAL")
        assert result.udi_score > 0.0
        assert result.probability_component == 0.01

    def test_floor_does_not_raise_above_floor(self):
        """Floor should not affect probabilities already above the floor."""
        engine = UDIEngine(formula_mode="multiplicative", probability_floor=0.05)
        candidate = DonorCandidate(
            donor_id="D1",
            travel_time_minutes=5.0,
            synthetic_response_propensity=0.8,
        )
        result = engine.score_candidate(candidate, "CRITICAL")
        assert result.probability_component == 0.8

    def test_floor_comparison_sweep(self):
        """Compare scores at floors 0, 0.01, 0.05 for a zero-probability donor."""
        scores = {}
        for floor in [0.0, 0.01, 0.05]:
            engine = UDIEngine(formula_mode="multiplicative", probability_floor=floor)
            candidate = DonorCandidate(
                donor_id="D1",
                travel_time_minutes=10.0,
                synthetic_response_propensity=0.0,
            )
            result = engine.score_candidate(candidate, "CRITICAL")
            scores[floor] = result.udi_score
        # Score should increase monotonically with floor
        assert scores[0.0] == 0.0
        assert scores[0.01] > scores[0.0]
        assert scores[0.05] > scores[0.01]


# ---------------------------------------------------------------
# Tie-breaking tests
# ---------------------------------------------------------------

class TestTieBreaking:
    """Tests for deterministic tie-breaking: UDI desc → ETA asc → P desc → ID asc."""

    def test_tie_broken_by_eta(self):
        """When UDI scores are tied, closer donor ranks higher."""
        engine = UDIEngine(formula_mode="multiplicative")
        c1 = DonorCandidate(donor_id="D1", travel_time_minutes=20.0, synthetic_response_propensity=0.5)
        c2 = DonorCandidate(donor_id="D2", travel_time_minutes=10.0, synthetic_response_propensity=0.5)
        # Same probability, different ETA → different UDI scores, but let's test
        # with crafted values that produce identical scores
        # Actually with multiplicative formula, different ETAs produce different scores.
        # Use additive mode where urgency contribution can mask ETA differences.
        engine_add = UDIEngine(
            formula_mode="additive", w_urgency=1.0, w_timeliness=0.0, w_probability=0.0
        )
        c1 = DonorCandidate(donor_id="D_far", travel_time_minutes=20.0, synthetic_response_propensity=0.5)
        c2 = DonorCandidate(donor_id="D_near", travel_time_minutes=10.0, synthetic_response_propensity=0.5)
        results = engine_add.score_and_rank([c1, c2], "CRITICAL")
        # Scores are identical (only urgency, which is constant)
        assert results[0].udi_score == results[1].udi_score
        # Tie broken by ETA ascending → D_near first
        assert results[0].donor_id == "D_near"
        assert results[1].donor_id == "D_far"

    def test_tie_broken_by_donor_id(self):
        """When UDI and ETA are identical, donor_id alphabetical order breaks tie."""
        engine = UDIEngine(
            formula_mode="additive", w_urgency=1.0, w_timeliness=0.0, w_probability=0.0
        )
        c1 = DonorCandidate(donor_id="D_banana", travel_time_minutes=10.0, synthetic_response_propensity=0.5)
        c2 = DonorCandidate(donor_id="D_apple", travel_time_minutes=10.0, synthetic_response_propensity=0.5)
        results = engine.score_and_rank([c1, c2], "CRITICAL")
        # Same UDI, same ETA, same P → alphabetical donor_id
        assert results[0].donor_id == "D_apple"
        assert results[1].donor_id == "D_banana"


# ---------------------------------------------------------------
# Additive vs Multiplicative comparison
# ---------------------------------------------------------------

class TestFormulaComparison:
    """Compare additive and multiplicative formulas on edge cases."""

    def test_nearby_low_probability_donor(self):
        """
        A donor who is very close but has low response probability
        should rank higher in additive (compensatory) than multiplicative.
        """
        candidate = DonorCandidate(
            donor_id="D_close_unlikely",
            travel_time_minutes=2.0,
            synthetic_response_propensity=0.05,
        )
        add_engine = UDIEngine(formula_mode="additive")
        mul_engine = UDIEngine(formula_mode="multiplicative")
        r_add = add_engine.score_candidate(candidate, "CRITICAL")
        r_mul = mul_engine.score_candidate(candidate, "CRITICAL")
        # Additive: gets credit from urgency and timeliness even with low P
        # Multiplicative: P=0.05 drags everything down
        assert r_add.udi_score > r_mul.udi_score

    def test_formula_mode_validation(self):
        """Invalid formula mode should raise ValueError."""
        with pytest.raises(ValueError, match="formula_mode must be"):
            UDIEngine(formula_mode="hybrid")
