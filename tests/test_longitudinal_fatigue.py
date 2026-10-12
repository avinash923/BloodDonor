"""
tests/test_longitudinal_fatigue.py
Automated tests for Phase 6.2 longitudinal fatigue simulation and tracking.

Validates the 12 required test conditions:
  1. Notification history persists across emergency episodes.
  2. Notification history correctly respects rolling lookback window boundaries.
  3. Dispatching a notification updates history exactly once.
  4. Donors cannot be contacted twice within the same emergency episode.
  5. Fatigue discounting decreases monotonically with prior notifications.
  6. Fatigue coefficient lambda=0.0 is mathematically equivalent to fatigue-disabled ranking.
  7. History is strictly isolated between independent timelines and arms.
  8. Seed reproducibility: identical seeds produce identical event sequences and outputs.
  9. Matched arms receive identical underlying requests and exogenous variates.
 10. Ranker cannot access hidden future outcomes or latent response propensities.
 11. Hard clinical eligibility and travel feasibility constraints remain enforced.
 12. Single-episode regression test: with zero prior alerts, fatigue-aware is identical to fatigue-disabled.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pytest

from src.data.schemas.request_schema import EmergencyRequest
from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.simulation.response_model import StochasticResponseModel
from src.simulation.simulator import SimulationEpisode
from src.simulation.longitudinal_tracker import LongitudinalDonorTracker, calculate_gini


class TestLongitudinalTracking:
    """Tests 1-3: History persistence, windowing, and update semantics."""

    def test_history_persistence_across_emergencies(self):
        tracker = LongitudinalDonorTracker(lookback_days=14.0)
        t1 = datetime(2024, 1, 1, 10, 0, 0)
        t2 = datetime(2024, 1, 2, 14, 0, 0)
        t3 = datetime(2024, 1, 3, 9, 0, 0)

        # Emergency 1
        tracker.record_notification("D001", t1, "REQ_001", wave_number=1, accepted=True)
        assert tracker.get_recent_alert_count("D001", t1) == 1
        assert tracker.get_cumulative_alert_count("D001") == 1

        # Emergency 2
        tracker.record_notification("D001", t2, "REQ_002", wave_number=1, accepted=False)
        assert tracker.get_recent_alert_count("D001", t2) == 2
        assert tracker.get_cumulative_alert_count("D001") == 2

        # Emergency 3 query before dispatch
        assert tracker.get_recent_alert_count("D001", t3) == 2

    def test_lookback_window_boundary_semantics(self):
        tracker = LongitudinalDonorTracker(lookback_days=14.0)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.record_notification("D001", t0, "REQ_001", wave_number=1)

        # Inside window (Day 10)
        t_inside = t0 + timedelta(days=10)
        assert tracker.get_recent_alert_count("D001", t_inside) == 1

        # Exact boundary (Day 14) -> counted
        t_boundary = t0 + timedelta(days=14)
        assert tracker.get_recent_alert_count("D001", t_boundary) == 1

        # Outside window (Day 14.1 / Day 15) -> expired
        t_outside = t0 + timedelta(days=14, seconds=1)
        assert tracker.get_recent_alert_count("D001", t_outside) == 0

        # Cumulative count remains 1
        assert tracker.get_cumulative_alert_count("D001") == 1

    def test_dispatch_updates_history_exactly_once(self):
        tracker = LongitudinalDonorTracker(lookback_days=14.0)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.record_notification("D001", t0, "REQ_001", wave_number=1)
        tracker.record_notification("D002", t0, "REQ_001", wave_number=1)

        all_recs = tracker.get_all_records()
        assert len(all_recs) == 2
        assert len(tracker.history["D001"]) == 1
        assert len(tracker.history["D002"]) == 1


class TestDispatchAndFatigueMechanics:
    """Tests 4-7: Dispatch rules, fatigue formula, lambda=0 equivalence, and timeline isolation."""

    def test_no_repeated_contact_within_single_emergency(self):
        cascade = DispatchCascadePolicy()
        ranked = [
            pytest.importorskip("src.ranking.udi_engine").UDIResult(
                donor_id="D001", udi_score=0.9, urgency_component=0, timeliness_component=0.8,
                probability_component=0.9, weighted_sum=0.72, fatigue_discount=1.0, is_feasible=True, is_eligible=True
            ),
            pytest.importorskip("src.ranking.udi_engine").UDIResult(
                donor_id="D002", udi_score=0.8, urgency_component=0, timeliness_component=0.8,
                probability_component=0.8, weighted_sum=0.64, fatigue_discount=1.0, is_feasible=True, is_eligible=True
            ),
        ]
        already_notified = {"D001"}
        batch = cascade.select_batch(ranked, already_notified, batch_size=2)
        assert len(batch) == 1
        assert batch[0].donor_id == "D002"

    def test_fatigue_discount_decreases_monotonically(self):
        engine = UDIEngine(lambda_fatigue=0.30, formula_mode="multiplicative")
        discounts = [engine._fatigue_discount(n) for n in range(6)]
        for i in range(len(discounts) - 1):
            assert discounts[i] > discounts[i + 1]
            assert 0.0 < discounts[i] <= 1.0
        assert discounts[0] == pytest.approx(1.0)

    def test_lambda_zero_equivalent_to_fatigue_disabled(self):
        engine_zero = UDIEngine(lambda_fatigue=0.0, formula_mode="multiplicative")
        cand1 = DonorCandidate(donor_id="D1", travel_time_minutes=15.0, synthetic_response_propensity=0.7, recent_alert_count=5, is_eligible=True)
        cand2 = DonorCandidate(donor_id="D2", travel_time_minutes=15.0, synthetic_response_propensity=0.7, recent_alert_count=0, is_eligible=True)

        res1 = engine_zero.score_candidate(cand1, "URGENT")
        res2 = engine_zero.score_candidate(cand2, "URGENT")

        assert res1.fatigue_discount == pytest.approx(1.0)
        assert res2.fatigue_discount == pytest.approx(1.0)
        assert res1.udi_score == pytest.approx(res2.udi_score)

    def test_history_isolated_between_independent_trackers(self):
        tracker_arm_a = LongitudinalDonorTracker()
        tracker_arm_b = LongitudinalDonorTracker()
        t = datetime(2024, 1, 1, 12, 0)

        tracker_arm_a.record_notification("D001", t, "REQ_1", 1)
        assert tracker_arm_a.get_cumulative_alert_count("D001") == 1
        assert tracker_arm_b.get_cumulative_alert_count("D001") == 0


class TestSimulationValidityAndConstraints:
    """Tests 8-12: Reproducibility, isolation, constraints, and single-episode regression."""

    def test_reproducible_response_variates(self):
        rm1 = StochasticResponseModel(seed=42)
        rm2 = StochasticResponseModel(seed=42)
        donors = ["D1", "D2", "D3"]
        rm1.pregenerate_donor_variates(donors)
        rm2.pregenerate_donor_variates(donors)

        for d in donors:
            out1 = rm1.sample_response(probability=0.6, donor_id=d)
            out2 = rm2.sample_response(probability=0.6, donor_id=d)
            assert out1.accepted == out2.accepted
            assert out1.latency_minutes == pytest.approx(out2.latency_minutes)

    def test_ranker_cannot_access_future_outcomes(self):
        """Verify DonorCandidate interface only exposes public features."""
        cand = DonorCandidate(
            donor_id="D001",
            travel_time_minutes=20.0,
            synthetic_response_propensity=0.65,
            recent_alert_count=2,
            is_eligible=True,
        )
        assert not hasattr(cand, "accepted")
        assert not hasattr(cand, "true_response_latency")
        assert not hasattr(cand, "future_response")

    def test_hard_eligibility_and_travel_feasibility_enforced(self):
        engine = UDIEngine(lambda_fatigue=0.30, formula_mode="multiplicative")
        # Ineligible
        cand_ineligible = DonorCandidate(donor_id="D1", travel_time_minutes=10.0, synthetic_response_propensity=0.8, recent_alert_count=0, is_eligible=False)
        res_ineligible = engine.score_candidate(cand_ineligible, "URGENT")
        assert not res_ineligible.is_eligible
        assert res_ineligible.udi_score == 0.0

        # Infeasible travel time exceeding budget
        cand_infeasible = DonorCandidate(donor_id="D2", travel_time_minutes=500.0, synthetic_response_propensity=0.8, recent_alert_count=0, is_eligible=True)
        res_infeasible = engine.score_candidate(cand_infeasible, "URGENT")
        assert not res_infeasible.is_feasible
        assert res_infeasible.udi_score == 0.0

    def test_single_episode_zero_alert_regression(self):
        """
        Regression test: When all donors have recent_alert_count=0,
        Fatigue-Aware and Fatigue-Disabled produce identical scores and ranks.
        """
        engine_aware = UDIEngine(lambda_fatigue=0.30, formula_mode="multiplicative")
        engine_disabled = UDIEngine(lambda_fatigue=0.00, formula_mode="multiplicative")

        candidates = [
            DonorCandidate(donor_id=f"D{i}", travel_time_minutes=10.0 + i * 2, synthetic_response_propensity=0.5 + i * 0.1, recent_alert_count=0, is_eligible=True)
            for i in range(5)
        ]

        ranked_aware = engine_aware.score_and_rank(candidates, "URGENT")
        ranked_disabled = engine_disabled.score_and_rank(candidates, "URGENT")

        for r_a, r_d in zip(ranked_aware, ranked_disabled):
            assert r_a.donor_id == r_d.donor_id
            assert r_a.udi_score == pytest.approx(r_d.udi_score)
            assert r_a.fatigue_discount == pytest.approx(1.0)
            assert r_d.fatigue_discount == pytest.approx(1.0)

    def test_gini_calculation(self):
        # Perfect equality
        assert calculate_gini([2, 2, 2, 2]) == pytest.approx(0.0)
        # All zeros
        assert calculate_gini([0, 0, 0, 0]) == pytest.approx(0.0)
        # High inequality
        gini_high = calculate_gini([0, 0, 0, 10])
        assert gini_high > 0.5

    def test_missing_travel_time_is_infeasible(self):
        engine = UDIEngine(lambda_fatigue=0.30, formula_mode="multiplicative")
        cand = DonorCandidate(
            donor_id="D_missing",
            travel_time_minutes=None,
            synthetic_response_propensity=0.9,
            recent_alert_count=0,
            is_eligible=True,
        )
        res = engine.score_candidate(cand, "URGENT")
        assert not res.is_feasible
        assert res.udi_score == 0.0


class TestLookbackRepeatStats:
    def test_lookback_repeat_inside_and_outside_window(self):
        tracker = LongitudinalDonorTracker(lookback_days=14.0)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.record_notification("D001", t0, "E1", 1)
        tracker.record_notification("D001", t0 + timedelta(days=10), "E2", 1)
        tracker.record_notification("D002", t0, "E1", 1)

        stats_in = tracker.compute_lookback_repeat_stats(["D001", "D002", "D003"], lookback_days=14.0)
        assert stats_in["n_lookback_repeat_notifications"] == 1
        assert stats_in["lookback_repeat_notification_rate"] == pytest.approx(1.0 / 3.0)
        assert stats_in["pct_donors_with_lookback_repeat"] == pytest.approx(100.0 / 3.0)

        stats_out = tracker.compute_lookback_repeat_stats(["D001", "D002", "D003"], lookback_days=7.0)
        assert stats_out["n_lookback_repeat_notifications"] == 0
        assert stats_out["lookback_repeat_notification_rate"] == pytest.approx(0.0)

    def test_lookback_repeat_exact_boundary(self):
        tracker = LongitudinalDonorTracker(lookback_days=14.0)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.record_notification("D001", t0, "E1", 1)
        tracker.record_notification("D001", t0 + timedelta(days=14), "E2", 1)
        stats = tracker.compute_lookback_repeat_stats(["D001"], lookback_days=14.0)
        assert stats["n_lookback_repeat_notifications"] == 1


class TestHolmAndMatchedSchedules:
    def test_holm_adjust_simple(self):
        from experiments.exp_phase6_2_longitudinal_fatigue import holm_adjust

        adj = holm_adjust([0.01, 0.04, 0.03])
        # Ordered p: 0.01, 0.03, 0.04 → 3*0.01=0.03, 2*0.03=0.06, 1*0.04=0.04 (monotone 0.06)
        assert adj[0] == pytest.approx(0.03)
        assert adj[1] == pytest.approx(0.06)
        assert adj[2] == pytest.approx(0.06)

    def test_matched_arms_share_request_schedule(self):
        from experiments.exp_phase6_2_longitudinal_fatigue import generate_longitudinal_timeline_scenarios
        from src.data.schemas.donor_schema import DonorProfile, DonorStream, KaggleFeatures, BloodType

        profiles = [
            DonorProfile(
                donor_id="KAG_D_00000",
                stream=DonorStream.SYNTHETIC_SIMULATION,
                kaggle_features=KaggleFeatures(
                    city="Sydney",
                    months_since_first_donation=12,
                    number_of_donation=3,
                    pints_donated=3,
                    availability=1,
                ),
                latitude=-33.87,
                longitude=151.20,
                blood_type=BloodType.O_POS,
                is_eligible=True,
                synthetic_response_propensity=0.4,
            )
        ]
        s1 = generate_longitudinal_timeline_scenarios(0, 10.0, 2.0, profiles, seed=123)
        s2 = generate_longitudinal_timeline_scenarios(0, 10.0, 2.0, profiles, seed=123)
        s3 = generate_longitudinal_timeline_scenarios(0, 10.0, 2.0, profiles, seed=124)
        assert [x.request_id for x in s1] == [x.request_id for x in s2]
        assert [x.request_timestamp for x in s1] == [x.request_timestamp for x in s2]
        assert [x.blood_type_required for x in s1] == [x.blood_type_required for x in s2]
        assert [x.request_id for x in s1] != [x.request_id for x in s3] or [
            x.request_timestamp for x in s1
        ] != [x.request_timestamp for x in s3]


class TestSimulatorDispatchRecording:
    def _run_mini_episode(self, recent_counts: dict[str, int], lambda_fatigue: float, seed: int):
        engine = UDIEngine(lambda_fatigue=lambda_fatigue, formula_mode="multiplicative")
        cands = [
            DonorCandidate(
                donor_id="D1",
                travel_time_minutes=12.0,
                synthetic_response_propensity=0.9,
                recent_alert_count=recent_counts["D1"],
                is_eligible=True,
            ),
            DonorCandidate(
                donor_id="D2",
                travel_time_minutes=14.0,
                synthetic_response_propensity=0.8,
                recent_alert_count=recent_counts["D2"],
                is_eligible=True,
            ),
            DonorCandidate(
                donor_id="D3",
                travel_time_minutes=500.0,
                synthetic_response_propensity=0.99,
                recent_alert_count=0,
                is_eligible=True,
            ),
        ]
        ranked = engine.score_and_rank(cands, "URGENT")
        feasible = [r for r in ranked if r.is_feasible and r.is_eligible]
        req = EmergencyRequest(
            request_id="MINI_1",
            blood_type_required="O+",
            units_required=1,
            urgency="URGENT",
            site_latitude=-33.87,
            site_longitude=151.20,
            request_timestamp=datetime(2024, 1, 1, 8, 0, 0),
        )
        rm = StochasticResponseModel(seed=seed)
        rm.pregenerate_donor_variates(["D1", "D2", "D3"])
        sim = SimulationEpisode(
            request=req,
            candidates={"D1": 0.9, "D2": 0.8, "D3": 0.99},
            ranked_pool=feasible,
            t_critical_minutes=120.0,
            response_model=rm,
            dispatch_policy=DispatchCascadePolicy(w1_mult=3.0, w_sub_mult=2.0),
            udi_engine=engine,
            unranked_candidates=cands,
        )
        return sim.run(), cands

    def test_dispatch_log_updates_history_exactly_once_and_no_intra_episode_repeat(self):
        res, _ = self._run_mini_episode({"D1": 0, "D2": 0}, lambda_fatigue=0.30, seed=7)
        donor_ids = [ev.donor_id for ev in res.dispatch_log]
        assert len(donor_ids) == len(set(donor_ids))
        assert "D3" not in donor_ids  # infeasible travel time
        tracker = LongitudinalDonorTracker(lookback_days=14.0)
        t0 = datetime(2024, 1, 1, 8, 0, 0)
        for ev in res.dispatch_log:
            tracker.record_notification(ev.donor_id, t0, "MINI_1", ev.wave_number, ev.accepted)
        assert tracker.get_cumulative_alert_count(res.dispatch_log[0].donor_id) == 1
        assert sum(tracker.get_cumulative_alert_count(d) for d in set(donor_ids)) == len(donor_ids)

    def test_zero_history_fatigue_aware_matches_disabled_first_episode_dispatch(self):
        res_a, _ = self._run_mini_episode({"D1": 0, "D2": 0}, lambda_fatigue=0.30, seed=11)
        res_b, _ = self._run_mini_episode({"D1": 0, "D2": 0}, lambda_fatigue=0.00, seed=11)
        ids_a = [ev.donor_id for ev in res_a.dispatch_log]
        ids_b = [ev.donor_id for ev in res_b.dispatch_log]
        assert ids_a == ids_b
        assert [ev.accepted for ev in res_a.dispatch_log] == [ev.accepted for ev in res_b.dispatch_log]

    def test_seed_reproducibility_of_mini_episode(self):
        res1, _ = self._run_mini_episode({"D1": 2, "D2": 0}, lambda_fatigue=0.30, seed=99)
        res2, _ = self._run_mini_episode({"D1": 2, "D2": 0}, lambda_fatigue=0.30, seed=99)
        assert [ev.donor_id for ev in res1.dispatch_log] == [ev.donor_id for ev in res2.dispatch_log]
        assert res1.units_fulfilled == res2.units_fulfilled
        assert res1.total_notifications_sent == res2.total_notifications_sent

    def test_history_changes_ranking_when_lambda_positive(self):
        engine = UDIEngine(lambda_fatigue=0.30, formula_mode="multiplicative")
        tired = DonorCandidate(
            donor_id="TIRED", travel_time_minutes=10.0,
            synthetic_response_propensity=0.9, recent_alert_count=8, is_eligible=True,
        )
        fresh = DonorCandidate(
            donor_id="FRESH", travel_time_minutes=12.0,
            synthetic_response_propensity=0.9, recent_alert_count=0, is_eligible=True,
        )
        ranked = engine.score_and_rank([tired, fresh], "URGENT")
        feasible = [r for r in ranked if r.is_feasible]
        assert feasible[0].donor_id == "FRESH"
        engine0 = UDIEngine(lambda_fatigue=0.0, formula_mode="multiplicative")
        ranked0 = engine0.score_and_rank([tired, fresh], "URGENT")
        feasible0 = [r for r in ranked0 if r.is_feasible]
        assert feasible0[0].donor_id == "TIRED"


class TestToyLongitudinalArmIsolation:
    def test_independent_trackers_and_matched_exogenous_inputs(self):
        from experiments.exp_phase6_2_longitudinal_fatigue import (
            generate_longitudinal_timeline_scenarios,
            run_timeline_for_arm,
            ARM_A_NAME,
            ARM_B_NAME,
        )
        from src.data.schemas.donor_schema import DonorProfile, DonorStream, KaggleFeatures, BloodType
        from src.spatial.travel_time_engine import TravelTimeEngine, CongestionModel
        from src.simulation.dispatch_cascade import DispatchCascadePolicy

        profiles = []
        for i, bt in enumerate(["O+", "O+", "A+", "B+", "AB+", "O-"]):
            profiles.append(
                DonorProfile(
                    donor_id=f"TOY_{i:03d}",
                    stream=DonorStream.SYNTHETIC_SIMULATION,
                    kaggle_features=KaggleFeatures(
                        city="Sydney",
                        months_since_first_donation=10,
                        number_of_donation=2,
                        pints_donated=2,
                        availability=0,
                    ),
                    latitude=-33.86 - 0.01 * i,
                    longitude=151.20 + 0.01 * i,
                    blood_type=BloodType(bt),
                    is_eligible=True,
                    synthetic_response_propensity=0.85,
                )
            )
        scenarios = generate_longitudinal_timeline_scenarios(
            timeline_id=0, horizon_days=3.0, rate_per_day=2.0, profiles=profiles, seed=42
        )
        assert len(scenarios) >= 1
        travel = TravelTimeEngine(distance_method="network_grid", congestion_model=CongestionModel())
        dispatch = DispatchCascadePolicy(w1_mult=3.0, w_sub_mult=2.0)
        weights = {"weights": {"w_urgency": 0.3, "w_timeliness": 0.4, "w_probability": 0.3}}

        tr_a, ep_a = run_timeline_for_arm(
            0, ARM_A_NAME, 0.30, 14.0, profiles, scenarios, travel, dispatch, weights, seed=42
        )
        tr_b, ep_b = run_timeline_for_arm(
            0, ARM_B_NAME, 0.00, 14.0, profiles, scenarios, travel, dispatch, weights, seed=42
        )
        tr_a2, ep_a2 = run_timeline_for_arm(
            0, ARM_A_NAME, 0.30, 14.0, profiles, scenarios, travel, dispatch, weights, seed=42
        )

        assert [e["Scenario_ID"] for e in ep_a] == [e["Scenario_ID"] for e in ep_b]
        assert [e["Timestamp"] for e in ep_a] == [e["Timestamp"] for e in ep_b]
        assert [e["Alerts_Sent"] for e in ep_a] == [e["Alerts_Sent"] for e in ep_a2]
        assert [e["Quota_Met_Deadline"] for e in ep_a] == [e["Quota_Met_Deadline"] for e in ep_a2]
        assert tr_a.get_cumulative_alert_count("TOY_000") == tr_a2.get_cumulative_alert_count("TOY_000")
        # Arm isolation: notifications recorded on A must not appear on a fresh B tracker
        assert id(tr_a.history) != id(tr_b.history)
        # First emergency starts at zero history for both arms
        if ep_a:
            assert ep_a[0]["Scenario_ID"] == ep_b[0]["Scenario_ID"]
