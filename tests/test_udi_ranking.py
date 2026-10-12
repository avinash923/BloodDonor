"""
tests/test_udi_ranking.py
Unit tests for Phase 4: UDI Multi-Criteria Ranking Engine & Baselines.

Verifies:
  1. UDI weight constraint enforcement (w_u + w_t + w_p == 1)
  2. UDI score bounds strictly in [0, 1]
  3. Ineligible donor always receives UDI = 0
  4. Infeasible donor (τ > T_eff) always receives UDI = 0
  5. Zero travel time maximises timeliness component
  6. Fatigue penalty monotonically decreases with alert count
  7. No fatigue (lambda=0) leaves score unchanged from weighted sum
  8. Higher urgency U(e) produces higher UDI (all else equal)
  9. Higher response probability produces higher UDI (all else equal)
  10. Ranked list is ordered descending by UDI score
  11. top_k returns at most k feasible results
  12. Baseline rankers return same candidate count as input
  13. Proximity-First: nearest donor ranked 1st
  14. Probability-First: highest-probability donor ranked 1st
  15. Heuristic produces scores in [0, 1]
  16. Random ranking is non-deterministic across different seeds
  17. precision_at_k returns 1.0 when all top-k are true responders
  18. precision_at_k returns 0.0 when no top-k are responders
  19. DonorCandidate raises ValueError for out-of-range probability
  20. DonorCandidate raises ValueError for negative travel time

Run with: pytest tests/test_udi_ranking.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
import numpy as np

from src.ranking.udi_engine import (
    UDIEngine,
    DonorCandidate,
    UDIResult,
    precision_at_k,
)
from src.ranking.baselines import (
    RandomRanking,
    ProximityFirstRanking,
    ProbabilityFirstRanking,
    HeuristicRanking,
    get_all_baselines,
)


# =================================================================
# Fixtures
# =================================================================

@pytest.fixture
def engine() -> UDIEngine:
    """Default balanced UDI engine."""
    return UDIEngine(w_urgency=0.30, w_timeliness=0.40, w_probability=0.30,
                     lambda_fatigue=0.30, safety_margin=0.15)


def make_candidate(
    donor_id: str = "D001",
    travel_time: float = 30.0,
    prob: float = 0.50,
    alerts: int = 0,
    eligible: bool = True,
) -> DonorCandidate:
    return DonorCandidate(
        donor_id=donor_id,
        travel_time_minutes=travel_time,
        recruitment_response_probability=prob,
        recent_alert_count=alerts,
        is_eligible=eligible,
    )


@pytest.fixture
def small_pool() -> list[DonorCandidate]:
    """A small pool of 5 candidates with varied travel times and probabilities."""
    return [
        make_candidate("D1", travel_time=10.0, prob=0.80, alerts=0),
        make_candidate("D2", travel_time=25.0, prob=0.60, alerts=1),
        make_candidate("D3", travel_time=50.0, prob=0.40, alerts=2),
        make_candidate("D4", travel_time=80.0, prob=0.70, alerts=0),  # infeasible for CRITICAL
        make_candidate("D5", travel_time=5.0,  prob=0.20, alerts=4),
    ]


# =================================================================
# Test Group 1: Weight Validation
# =================================================================

class TestWeightValidation:
    def test_valid_weights_accepted(self):
        engine = UDIEngine(w_urgency=0.30, w_timeliness=0.40, w_probability=0.30, formula_mode="additive")
        assert engine.w_urgency == pytest.approx(0.30)

    def test_weights_not_summing_to_one_raises_in_additive(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            UDIEngine(w_urgency=0.30, w_timeliness=0.40, w_probability=0.40, formula_mode="additive")

    def test_negative_weight_raises_in_additive(self):
        with pytest.raises(ValueError, match="non-negative"):
            UDIEngine(w_urgency=-0.10, w_timeliness=0.60, w_probability=0.50, formula_mode="additive")

    def test_extreme_all_urgency_weight(self):
        engine = UDIEngine(w_urgency=1.0, w_timeliness=0.0, w_probability=0.0, formula_mode="additive")
        assert engine.w_urgency == pytest.approx(1.0)
        
    def test_multiplicative_mode_ignores_weight_sum(self):
        # Should not raise an error even if sum != 1
        engine = UDIEngine(w_urgency=0.0, w_timeliness=1.0, w_probability=1.0, formula_mode="multiplicative")
        assert engine.w_timeliness == 1.0
        
    def test_multiplicative_mode_accepts_negative_weights(self):
        # Multiplicative mode ignores these weights entirely
        engine = UDIEngine(w_urgency=-0.5, w_timeliness=2.0, w_probability=1.5, formula_mode="multiplicative")
        assert engine.w_urgency == -0.5

    def test_get_weights_returns_correct_dict(self, engine):
        w = engine.get_weights()
        assert abs(w["w_urgency"] + w["w_timeliness"] + w["w_probability"] - 1.0) < 1e-9


# =================================================================
# Test Group 2: UDI Score Bounds
# =================================================================

class TestUDIBounds:
    def test_score_in_unit_interval_feasible(self, engine):
        c = make_candidate(travel_time=20.0, prob=0.60, alerts=0)
        result = engine.score_candidate(c, urgency="URGENT")
        assert 0.0 <= result.udi_score <= 1.0

    def test_score_in_unit_interval_high_alerts(self, engine):
        c = make_candidate(travel_time=20.0, prob=0.60, alerts=10)
        result = engine.score_candidate(c, urgency="URGENT")
        assert 0.0 <= result.udi_score <= 1.0

    def test_maximum_score_conditions(self, engine):
        """Zero travel time, probability=1, no fatigue should approach max."""
        c = make_candidate(travel_time=0.0, prob=1.0, alerts=0)
        result = engine.score_candidate(c, urgency="CRITICAL")
        assert result.udi_score > 0.9, (
            f"Expected near-maximum UDI, got {result.udi_score:.4f}"
        )

    def test_minimum_score_zero_for_ineligible(self, engine):
        c = make_candidate(eligible=False)
        result = engine.score_candidate(c, urgency="URGENT")
        assert result.udi_score == pytest.approx(0.0)

    def test_minimum_score_zero_for_infeasible(self, engine):
        """Donor at 200 min travel time is infeasible for CRITICAL (T_eff=51 min)."""
        c = make_candidate(travel_time=200.0, prob=0.99, alerts=0)
        result = engine.score_candidate(c, urgency="CRITICAL")
        assert result.udi_score == pytest.approx(0.0)
        assert result.is_feasible is False

    def test_ineligible_flag_set_correctly(self, engine):
        c = make_candidate(eligible=False)
        result = engine.score_candidate(c, urgency="URGENT")
        assert result.is_eligible is False

    def test_all_scores_nonneg_on_pool(self, engine, small_pool):
        results = engine.score_and_rank(small_pool, urgency="URGENT")
        for r in results:
            assert r.udi_score >= 0.0


# =================================================================
# Test Group 3: Component Behaviour
# =================================================================

class TestComponentBehaviour:
    def test_zero_travel_time_maximises_timeliness(self, engine):
        c0 = make_candidate(travel_time=0.0, prob=0.50, alerts=0)
        c30 = make_candidate(travel_time=30.0, prob=0.50, alerts=0)
        r0 = engine.score_candidate(c0, "URGENT")
        r30 = engine.score_candidate(c30, "URGENT")
        assert r0.timeliness_component > r30.timeliness_component

    def test_higher_probability_raises_score(self, engine):
        c_low = make_candidate(travel_time=20.0, prob=0.20, alerts=0)
        c_high = make_candidate(travel_time=20.0, prob=0.80, alerts=0)
        r_low = engine.score_candidate(c_low, "URGENT")
        r_high = engine.score_candidate(c_high, "URGENT")
        assert r_high.udi_score > r_low.udi_score

    def test_tighter_urgency_budget_reduces_timeliness_score(self, engine):
        """Under multiplicative logic, a tighter CRITICAL budget penalizes a 10min ETA more than a SCHEDULED budget."""
        c = make_candidate(travel_time=10.0, prob=0.50, alerts=0)
        r_crit = engine.score_candidate(c, "CRITICAL")
        r_sched = engine.score_candidate(c, "SCHEDULED")
        
        if engine.formula_mode == "multiplicative":
            # 10 mins is a larger fraction of the 60min critical budget than the 240min scheduled budget
            assert r_crit.udi_score < r_sched.udi_score
        else:
            # Additive mode retains explicit urgency weight scoring
            assert r_crit.udi_score > r_sched.udi_score

    def test_fatigue_discount_monotonically_decreasing(self, engine):
        scores = []
        for alerts in [0, 1, 2, 5, 10]:
            c = make_candidate(travel_time=10.0, prob=0.50, alerts=alerts)
            r = engine.score_candidate(c, "URGENT")
            scores.append(r.udi_score)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at alerts={i} ({scores[i]:.4f}) not >= "
                f"score at alerts={i+1} ({scores[i+1]:.4f})"
            )

    def test_zero_lambda_no_fatigue_effect(self):
        """With lambda=0, fatigue discount = 1.0 regardless of alert count."""
        engine_no_fatigue = UDIEngine(
            w_urgency=0.30, w_timeliness=0.40, w_probability=0.30,
            lambda_fatigue=0.0,
        )
        c0 = make_candidate(travel_time=20.0, prob=0.50, alerts=0)
        c10 = make_candidate(travel_time=20.0, prob=0.50, alerts=10)
        r0 = engine_no_fatigue.score_candidate(c0, "URGENT")
        r10 = engine_no_fatigue.score_candidate(c10, "URGENT")
        assert r0.udi_score == pytest.approx(r10.udi_score, abs=1e-9)
        assert r0.fatigue_discount == pytest.approx(1.0)
        assert r10.fatigue_discount == pytest.approx(1.0)

    def test_fatigue_discount_formula(self, engine):
        """Verify Φ_fatigue = exp(-λ * n_alerts) matches manual calculation."""
        n_alerts = 3
        expected_discount = math.exp(-engine.lambda_fatigue * n_alerts)
        c = make_candidate(travel_time=10.0, prob=0.50, alerts=n_alerts)
        r = engine.score_candidate(c, "URGENT")
        assert r.fatigue_discount == pytest.approx(expected_discount, rel=1e-6)

    def test_unknown_urgency_raises(self, engine):
        c = make_candidate()
        with pytest.raises(ValueError, match="Unknown urgency tier"):
            engine.score_candidate(c, "LEVEL_X")


# =================================================================
# Test Group 4: Ranked List Properties
# =================================================================

class TestRankedList:
    def test_ranked_list_sorted_descending(self, engine, small_pool):
        results = engine.score_and_rank(small_pool, urgency="URGENT")
        scores = [r.udi_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_integers_start_at_one(self, engine, small_pool):
        results = engine.score_and_rank(small_pool, urgency="URGENT")
        ranks = [r.rank for r in results]
        assert ranks[0] == 1
        assert sorted(ranks) == list(range(1, len(results) + 1))

    def test_all_candidates_present_in_output(self, engine, small_pool):
        results = engine.score_and_rank(small_pool, urgency="URGENT")
        returned_ids = {r.donor_id for r in results}
        input_ids = {c.donor_id for c in small_pool}
        assert returned_ids == input_ids

    def test_top_k_returns_at_most_k(self, engine, small_pool):
        top = engine.top_k(small_pool, urgency="URGENT", k=2)
        assert len(top) <= 2

    def test_top_k_all_feasible(self, engine, small_pool):
        top = engine.top_k(small_pool, urgency="URGENT", k=5)
        for r in top:
            assert r.is_feasible is True
            assert r.is_eligible is True

    def test_empty_pool_returns_empty(self, engine):
        results = engine.score_and_rank([], urgency="URGENT")
        assert results == []

    def test_single_eligible_feasible_donor_ranked_first(self, engine):
        pool = [make_candidate("ONLY", travel_time=10.0, prob=0.5)]
        results = engine.score_and_rank(pool, urgency="URGENT")
        assert results[0].donor_id == "ONLY"
        assert results[0].rank == 1


# =================================================================
# Test Group 5: DonorCandidate Validation
# =================================================================

class TestDonorCandidateValidation:
    def test_invalid_probability_above_one(self):
        with pytest.raises(ValueError, match="recruitment_response_probability"):
            DonorCandidate("D", travel_time_minutes=10.0, recruitment_response_probability=1.5)

    def test_invalid_probability_negative(self):
        with pytest.raises(ValueError, match="recruitment_response_probability"):
            DonorCandidate("D", travel_time_minutes=10.0, recruitment_response_probability=-0.1)

    def test_invalid_negative_travel_time(self):
        with pytest.raises(ValueError, match="travel_time_minutes"):
            DonorCandidate("D", travel_time_minutes=-5.0, recruitment_response_probability=0.5)

    def test_invalid_negative_alert_count(self):
        with pytest.raises(ValueError, match="recent_alert_count"):
            DonorCandidate("D", travel_time_minutes=10.0, recruitment_response_probability=0.5,
                           recent_alert_count=-1)

    def test_boundary_probability_zero(self):
        c = DonorCandidate("D", travel_time_minutes=10.0, recruitment_response_probability=0.0)
        assert c.recruitment_response_probability == 0.0

    def test_boundary_probability_one(self):
        c = DonorCandidate("D", travel_time_minutes=10.0, recruitment_response_probability=1.0)
        assert c.recruitment_response_probability == 1.0

    def test_zero_travel_time_valid(self):
        c = DonorCandidate("D", travel_time_minutes=0.0, recruitment_response_probability=0.5)
        assert c.travel_time_minutes == 0.0


# =================================================================
# Test Group 6: Baseline Rankers
# =================================================================

class TestBaselines:
    @pytest.fixture
    def pool(self) -> list[DonorCandidate]:
        return [
            make_candidate("A", travel_time=5.0,  prob=0.30),
            make_candidate("B", travel_time=20.0, prob=0.80),
            make_candidate("C", travel_time=50.0, prob=0.50),
            make_candidate("D", travel_time=10.0, prob=0.60),
        ]

    def test_proximity_first_nearest_is_rank_1(self, pool):
        ranker = ProximityFirstRanking()
        results = ranker.rank(pool, urgency="URGENT")
        assert results[0].donor_id == "A"   # travel_time=5.0 is nearest

    def test_probability_first_highest_prob_is_rank_1(self, pool):
        ranker = ProbabilityFirstRanking()
        results = ranker.rank(pool, urgency="URGENT")
        assert results[0].donor_id == "B"   # prob=0.80 is highest

    def test_heuristic_scores_in_unit_interval(self, pool):
        ranker = HeuristicRanking()
        results = ranker.rank(pool, urgency="URGENT")
        for r in results:
            assert 0.0 <= r.udi_score <= 1.0

    def test_random_ranking_returns_all_candidates(self, pool):
        ranker = RandomRanking(seed=42)
        results = ranker.rank(pool, urgency="URGENT")
        returned_ids = {r.donor_id for r in results}
        assert returned_ids == {"A", "B", "C", "D"}

    def test_random_ranking_differs_across_seeds(self, pool):
        r1 = RandomRanking(seed=1).rank(pool, urgency="URGENT")
        r2 = RandomRanking(seed=999).rank(pool, urgency="URGENT")
        ids1 = [r.donor_id for r in r1]
        ids2 = [r.donor_id for r in r2]
        assert ids1 != ids2, "Different seeds should produce different orderings"

    def test_baselines_return_same_count_as_input(self, pool):
        for name, ranker in get_all_baselines().items():
            results = ranker.rank(pool, urgency="URGENT")
            assert len(results) == len(pool), (
                f"{name} returned {len(results)} results for {len(pool)} candidates"
            )

    def test_ineligible_donor_excluded_from_proximity(self):
        pool = [
            make_candidate("ELIGIBLE", travel_time=5.0,  prob=0.5, eligible=True),
            make_candidate("INELIGIBLE", travel_time=2.0, prob=0.9, eligible=False),
        ]
        ranker = ProximityFirstRanking()
        results = ranker.rank(pool, urgency="URGENT")
        # Ineligible is excluded completely by the filter
        assert len(results) == 1
        assert results[0].donor_id == "ELIGIBLE"


# =================================================================
# Test Group 7: Precision@k Metric
# =================================================================

class TestPrecisionAtK:
    @pytest.fixture
    def ranked(self, engine) -> list[UDIResult]:
        pool = [
            make_candidate("D1", travel_time=5.0,  prob=0.90, alerts=0),
            make_candidate("D2", travel_time=10.0, prob=0.70, alerts=0),
            make_candidate("D3", travel_time=20.0, prob=0.50, alerts=0),
            make_candidate("D4", travel_time=30.0, prob=0.30, alerts=0),
            make_candidate("D5", travel_time=40.0, prob=0.10, alerts=0),
        ]
        return engine.score_and_rank(pool, urgency="URGENT")

    def test_precision_at_k_perfect(self, ranked):
        top_3_ids = {r.donor_id for r in ranked[:3]}
        p = precision_at_k(ranked, k=3, ground_truth_responders=top_3_ids)
        assert p == pytest.approx(1.0)

    def test_precision_at_k_zero(self, ranked):
        p = precision_at_k(ranked, k=3, ground_truth_responders={"NOT_IN_LIST"})
        assert p == pytest.approx(0.0)

    def test_precision_at_k_partial(self, ranked):
        top_2_ids = {r.donor_id for r in ranked[:2]}
        p = precision_at_k(ranked, k=4, ground_truth_responders=top_2_ids)
        assert p == pytest.approx(0.5)

    def test_precision_at_k_zero_k(self, ranked):
        p = precision_at_k(ranked, k=0, ground_truth_responders={"D1"})
        assert p == pytest.approx(0.0)

    def test_precision_at_k_k_exceeds_list(self, ranked):
        all_ids = {r.donor_id for r in ranked}
        p = precision_at_k(ranked, k=100, ground_truth_responders=all_ids)
        assert p == pytest.approx(1.0)
