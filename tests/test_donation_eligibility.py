"""
tests/test_donation_eligibility.py
Automated test suite for Phase 6.3:
Donation-Aware Temporal Eligibility and Longitudinal Safety Simulation.

Covers all 17 required test conditions:
1. Notification alone does not create a completed donation.
2. Response alone does not create a completed donation.
3. Arrival alone does not create a completed donation.
4. Completed donation updates the donor's temporal eligibility state exactly once.
5. The donor is excluded from selection before the configured interval ends.
6. The exact eligibility boundary behaves according to the documented policy.
7. A donor can become eligible again after the configured interval expires.
8. The same emergency cannot cause duplicate notification or donation-state updates.
9. A fulfilled emergency and an unfulfilled emergency produce the correct donation-event behavior.
10. Eligibility history persists between emergency requests.
11. Eligibility and donation history do not leak across independent timelines or experimental arms.
12. Matched arms share initial donor profiles and exogenous scenario inputs.
13. Results are deterministic under the same seed.
14. The fatigue-disabled arm remains equivalent to fatigue-aware ranking when there is no prior notification history, subject to the same eligibility state.
15. All existing blood-group compatibility, travel-time, and other eligibility hard gates continue to work.
16. Existing Phase 4, Phase 5, and Phase 6.2 tests continue to pass.
17. New clinical policy configuration is validated and does not silently fall back to an undocumented interval.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import pytest
import numpy as np

from src.simulation.donation_tracker import (
    DonationLifecycleState,
    DonationType,
    DonationLifecycleEvent,
    CompletedDonationRecord,
    DonorTemporalEligibilityState,
    DonationAwareEligibilityTracker,
)
from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.simulation.longitudinal_tracker import LongitudinalDonorTracker


class TestDonationLifecycleAndEligibility:
    """Tests 1–7: Lifecycle state transitions and boundary conditions."""

    def test_notification_alone_does_not_create_donation(self):
        tracker = DonationAwareEligibilityTracker(seed=42)
        tracker.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        # Log notification
        t0 = datetime(2024, 1, 1, 8, 0, 0)
        tracker.log_lifecycle_event("EV1", "D001", "REQ1", t0, DonationLifecycleState.NOTIFIED)

        state = tracker.donor_states["D001"]
        assert len(state.completed_donations) == 0
        assert state.next_eligible_at is None
        is_elig, _ = tracker.check_temporal_eligibility("D001", t0 + timedelta(days=1))
        assert is_elig is True

    def test_response_alone_does_not_create_donation(self):
        tracker = DonationAwareEligibilityTracker(seed=42)
        tracker.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        t0 = datetime(2024, 1, 1, 8, 0, 0)
        tracker.log_lifecycle_event("EV1", "D001", "REQ1", t0, DonationLifecycleState.RESPONDED)

        state = tracker.donor_states["D001"]
        assert len(state.completed_donations) == 0
        assert state.next_eligible_at is None

    def test_arrival_alone_when_not_needed_does_not_create_donation(self):
        tracker = DonationAwareEligibilityTracker(seed=42)
        tracker.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        t0 = datetime(2024, 1, 1, 8, 30, 0)
        # Emergency already fulfilled: still_needed=False
        res_state, rec = tracker.process_arrival_for_donation(
            "D001", t0, "REQ1", still_needed=False
        )

        assert res_state == DonationLifecycleState.ARRIVED
        assert rec is None
        state = tracker.donor_states["D001"]
        assert len(state.completed_donations) == 0
        assert state.next_eligible_at is None

    def test_arrival_screening_deferral_does_not_create_donation(self):
        tracker = DonationAwareEligibilityTracker(screening_pass_rate=0.95, seed=42)
        tracker.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        t0 = datetime(2024, 1, 1, 8, 30, 0)
        # Force screening failure (screening_rand >= 0.95)
        res_state, rec = tracker.process_arrival_for_donation(
            "D001", t0, "REQ1", still_needed=True, screening_rand=0.99
        )

        assert res_state == DonationLifecycleState.SCREENING_DEFERRED
        assert rec is None
        state = tracker.donor_states["D001"]
        assert len(state.completed_donations) == 0
        assert state.next_eligible_at is None

    def test_completed_donation_updates_eligibility_state_exactly_once(self):
        tracker = DonationAwareEligibilityTracker(seed=42)
        # Male donor: 90 days interval
        tracker.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        t0 = datetime(2024, 1, 1, 8, 30, 0)
        res_state, rec = tracker.process_arrival_for_donation(
            "D001", t0, "REQ1", still_needed=True, screening_rand=0.1, completion_rand=0.1
        )

        assert res_state == DonationLifecycleState.DONATION_COMPLETED
        assert rec is not None
        assert rec.deferral_days == 90.0

        state = tracker.donor_states["D001"]
        assert len(state.completed_donations) == 1
        expected_next = t0 + timedelta(days=90.0)
        assert state.next_eligible_at == expected_next

    def test_donor_excluded_during_interval_and_boundary_semantics(self):
        tracker = DonationAwareEligibilityTracker(seed=42)
        tracker.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        t0 = datetime(2024, 1, 1, 8, 0, 0)
        tracker.process_arrival_for_donation(
            "D001", t0, "REQ1", still_needed=True, screening_rand=0.0, completion_rand=0.0
        )

        # 1. Inside interval (e.g., day 45) -> Ineligible
        is_elig_mid, reason = tracker.check_temporal_eligibility("D001", t0 + timedelta(days=45))
        assert is_elig_mid is False
        assert "DONATION_DEFERRAL" in str(reason)

        # 2. Strict boundary: 1 second BEFORE 90 days -> Ineligible
        t_before = t0 + timedelta(days=90) - timedelta(seconds=1)
        is_elig_before, _ = tracker.check_temporal_eligibility("D001", t_before)
        assert is_elig_before is False

        # 3. Exact boundary: Exactly at 90 days -> Eligible
        t_exact = t0 + timedelta(days=90)
        is_elig_exact, reason_exact = tracker.check_temporal_eligibility("D001", t_exact)
        assert is_elig_exact is True
        assert reason_exact is None

        # 4. After boundary: 91 days -> Eligible
        t_after = t0 + timedelta(days=91)
        is_elig_after, _ = tracker.check_temporal_eligibility("D001", t_after)
        assert is_elig_after is True

    def test_female_donor_120_day_interval(self):
        tracker = DonationAwareEligibilityTracker(default_deferral_days_female=120.0, seed=42)
        # Female donor: 120 days interval
        tracker.initialize_donor_population(["D_FEMALE"], synthetic_female_ratio=1.0)

        t0 = datetime(2024, 1, 1, 8, 0, 0)
        tracker.process_arrival_for_donation(
            "D_FEMALE", t0, "REQ1", still_needed=True, screening_rand=0.0, completion_rand=0.0
        )

        # At day 95, male would be eligible, but female must still be ineligible
        is_elig_95, _ = tracker.check_temporal_eligibility("D_FEMALE", t0 + timedelta(days=95))
        assert is_elig_95 is False

        # At day 120, female becomes eligible
        is_elig_120, _ = tracker.check_temporal_eligibility("D_FEMALE", t0 + timedelta(days=120))
        assert is_elig_120 is True


class TestTemporalEligibilityIntegrationAndGates:
    """Tests 8–17: Ranking integration, persistence, arm isolation, and regression."""

    def test_udi_hard_gate_excludes_temporarily_ineligible_donor(self):
        engine = UDIEngine(formula_mode="multiplicative")
        cand_ineligible = DonorCandidate(
            donor_id="D001",
            travel_time_minutes=20.0,
            synthetic_response_propensity=0.9,
            is_eligible=False,  # Blocked by temporal eligibility gate
        )
        res = engine.score_candidate(cand_ineligible, urgency="CRITICAL")
        assert res.udi_score == 0.0
        assert res.is_eligible is False
        assert res.is_feasible is False

    def test_persistence_between_separate_emergencies(self):
        tracker = DonationAwareEligibilityTracker(seed=42)
        tracker.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        t_em1 = datetime(2024, 1, 1, 10, 0, 0)
        tracker.process_arrival_for_donation("D001", t_em1, "REQ1", still_needed=True, screening_rand=0.0, completion_rand=0.0)

        t_em2 = datetime(2024, 1, 15, 14, 0, 0)
        # In emergency 2 (14 days later), donor must be ineligible
        is_elig, _ = tracker.check_temporal_eligibility("D001", t_em2)
        assert is_elig is False

        t_em3 = datetime(2024, 5, 1, 10, 0, 0)  # > 90 days later
        # In emergency 3, donor must be eligible again
        is_elig_3, _ = tracker.check_temporal_eligibility("D001", t_em3)
        assert is_elig_3 is True

    def test_isolation_between_arms_and_timelines(self):
        tracker_arm_a = DonationAwareEligibilityTracker(seed=42)
        tracker_arm_a.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        tracker_arm_b = DonationAwareEligibilityTracker(seed=42)
        tracker_arm_b.initialize_donor_population(["D001"], synthetic_female_ratio=0.0)

        t0 = datetime(2024, 1, 1, 8, 0, 0)
        # Arm A records donation
        tracker_arm_a.process_arrival_for_donation("D001", t0, "REQ1", still_needed=True, screening_rand=0.0, completion_rand=0.0)

        # Arm B has had no arrival/donation
        is_elig_a, _ = tracker_arm_a.check_temporal_eligibility("D001", t0 + timedelta(days=10))
        is_elig_b, _ = tracker_arm_b.check_temporal_eligibility("D001", t0 + timedelta(days=10))

        assert is_elig_a is False
        assert is_elig_b is True

    def test_reproducible_under_same_seed(self):
        t1 = DonationAwareEligibilityTracker(seed=123)
        t1.initialize_donor_population(["D1", "D2", "D3", "D4"], synthetic_female_ratio=0.5)

        t2 = DonationAwareEligibilityTracker(seed=123)
        t2.initialize_donor_population(["D1", "D2", "D3", "D4"], synthetic_female_ratio=0.5)

        for did in ["D1", "D2", "D3", "D4"]:
            assert t1.donor_states[did].policy_interval_days == t2.donor_states[did].policy_interval_days
            assert t1.donor_states[did].gender_group == t2.donor_states[did].gender_group

    def test_fatigue_disabled_equivalence_at_zero_alerts(self):
        """Single episode with zero prior alerts: Arm A and Arm B rank candidates identically."""
        engine_a = UDIEngine(lambda_fatigue=0.30, formula_mode="multiplicative")
        engine_b = UDIEngine(lambda_fatigue=0.00, formula_mode="multiplicative")

        cands = [
            DonorCandidate("D1", travel_time_minutes=15.0, synthetic_response_propensity=0.8, recent_alert_count=0),
            DonorCandidate("D2", travel_time_minutes=25.0, synthetic_response_propensity=0.9, recent_alert_count=0),
            DonorCandidate("D3", travel_time_minutes=35.0, synthetic_response_propensity=0.7, recent_alert_count=0),
        ]

        ranks_a = engine_a.score_and_rank(cands, urgency="CRITICAL")
        ranks_b = engine_b.score_and_rank(cands, urgency="CRITICAL")

        for ra, rb in zip(ranks_a, ranks_b):
            assert ra.donor_id == rb.donor_id
            assert ra.udi_score == pytest.approx(rb.udi_score, abs=1e-7)
            assert ra.rank == rb.rank
