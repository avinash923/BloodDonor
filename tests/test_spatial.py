"""
tests/test_spatial.py
Unit tests for Phase 3: Spatial Engine & Travel-Time Feasibility.

Verifies:
  1. Distance metric correctness (known coordinates → known distances)
  2. Haversine symmetry (d(A,B) == d(B,A))
  3. Zero distance for identical coordinates
  4. Network grid distance >= Haversine distance (detour > 1)
  5. Batch pairwise matrix shape correctness
  6. CongestionModel: valid range, boundary hour values
  7. TravelTimeEngine: congestion lengthens travel times at peak vs off-peak
  8. TravelTimeEngine: timeliness score bounds [0, 1]
  9. FeasibilityFilter: donor at zero distance is always feasible
  10. FeasibilityFilter: donor beyond T_critical is never feasible
  11. FeasibilityFilter: safety margin reduces effective horizon
  12. FeasibilityFilter: batch mode returns one result per scenario
  13. SyntheticGenerator: determinism under same seed
  14. SyntheticGenerator: donor coordinates within bounding box
  15. SyntheticGenerator: eligibility rule (recency >= 2 months)

Run with: pytest tests/test_spatial.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.spatial.distance_metrics import (
    haversine_distance_km,
    euclidean_distance_km,
    network_grid_distance_km,
    pairwise_distance_matrix,
)
from src.spatial.travel_time_engine import CongestionModel, TravelTimeEngine
from src.spatial.feasibility_filter import FeasibilityFilter
from src.data.synthetic_generator import SyntheticUrbanGenerator


# =================================================================
# Known Reference Points
# =================================================================

# Bengaluru city center ↔ Whitefield (~18 km apart geodesically)
_BLR_CENTER = (12.9716, 77.5946)
_BLR_WHITEFIELD = (12.9698, 77.7499)
_EXPECTED_HAVERSINE_KM_APPROX = 16.5   # geodesic ~16–17 km


# =================================================================
# Test Group 1: Distance Metrics
# =================================================================

class TestDistanceMetrics:
    def test_haversine_known_distance(self):
        dist = haversine_distance_km(*_BLR_CENTER, *_BLR_WHITEFIELD)
        assert 15.0 < dist < 18.5, (
            f"Haversine distance {dist:.2f} km outside expected range [15, 18.5]"
        )

    def test_haversine_symmetry(self):
        d_ab = haversine_distance_km(*_BLR_CENTER, *_BLR_WHITEFIELD)
        d_ba = haversine_distance_km(*_BLR_WHITEFIELD, *_BLR_CENTER)
        np.testing.assert_allclose(d_ab, d_ba, rtol=1e-10,
                                   err_msg="Haversine is not symmetric")

    def test_haversine_zero_distance(self):
        d = haversine_distance_km(12.97, 77.59, 12.97, 77.59)
        assert abs(d) < 1e-9, f"Distance to same point should be ~0, got {d}"

    def test_euclidean_positive(self):
        d = euclidean_distance_km(*_BLR_CENTER, *_BLR_WHITEFIELD)
        assert d > 0

    def test_euclidean_zero_distance(self):
        d = euclidean_distance_km(12.0, 77.0, 12.0, 77.0)
        assert abs(d) < 1e-9

    def test_network_grid_greater_than_haversine(self):
        d_hav = haversine_distance_km(*_BLR_CENTER, *_BLR_WHITEFIELD)
        d_net = network_grid_distance_km(*_BLR_CENTER, *_BLR_WHITEFIELD, detour_factor=1.35)
        assert d_net > d_hav, "Network grid distance should exceed Haversine"
        np.testing.assert_allclose(d_net, d_hav * 1.35, rtol=1e-9)

    def test_pairwise_matrix_shape(self):
        n_donors, n_sites = 20, 5
        donor_lats = np.linspace(12.80, 13.10, n_donors)
        donor_lons = np.linspace(77.45, 77.75, n_donors)
        site_lats = np.linspace(12.85, 13.05, n_sites)
        site_lons = np.linspace(77.50, 77.70, n_sites)
        D = pairwise_distance_matrix(donor_lats, donor_lons, site_lats, site_lons)
        assert D.shape == (n_donors, n_sites)

    def test_pairwise_matrix_nonnegative(self):
        n = 10
        lats = np.random.default_rng(7).uniform(12.8, 13.1, n)
        lons = np.random.default_rng(7).uniform(77.45, 77.75, n)
        D = pairwise_distance_matrix(lats, lons, lats[:3], lons[:3])
        assert (D >= 0).all()

    def test_invalid_distance_method_raises(self):
        with pytest.raises(ValueError, match="Unknown distance method"):
            pairwise_distance_matrix(
                np.array([12.97]), np.array([77.59]),
                np.array([12.97]), np.array([77.75]),
                method="invalid_method",
            )

    def test_haversine_vectorised_matches_scalar(self):
        lats = np.array([12.80, 12.90, 13.00])
        lons = np.array([77.45, 77.55, 77.65])
        site_lat, site_lon = 12.97, 77.59
        vec = haversine_distance_km(lats, lons, site_lat, site_lon)
        scalars = np.array([
            haversine_distance_km(lats[i], lons[i], site_lat, site_lon)
            for i in range(3)
        ])
        np.testing.assert_allclose(vec, scalars, rtol=1e-10)


# =================================================================
# Test Group 2: Congestion Model
# =================================================================

class TestCongestionModel:
    def test_all_values_in_unit_interval(self):
        cm = CongestionModel()
        for h in range(24):
            k = cm.kappa(h)
            assert 0 < k <= 1.0, f"kappa({h}) = {k} outside (0, 1]"

    def test_peak_hour_kappa_below_offpeak(self):
        cm = CongestionModel()
        k_peak = cm.kappa(8)    # 08:00 — morning rush
        k_offpeak = cm.kappa(3) # 03:00 — night off-peak
        assert k_peak < k_offpeak, "Peak κ should be lower than off-peak κ"

    def test_invalid_hour_raises(self):
        cm = CongestionModel()
        with pytest.raises(ValueError, match="hour_of_day"):
            cm.kappa(24)

    def test_effective_speed_lower_at_peak(self):
        cm = CongestionModel()
        v_base = 30.0
        v_peak = cm.effective_speed_kmh(8, v_base)
        v_offpeak = cm.effective_speed_kmh(3, v_base)
        assert v_peak < v_offpeak

    def test_effective_speed_at_free_flow(self):
        profile = [1.0] * 24
        cm = CongestionModel(hourly_profile=profile)
        assert cm.effective_speed_kmh(12, 30.0) == pytest.approx(30.0)

    def test_invalid_profile_length_raises(self):
        with pytest.raises(ValueError):
            CongestionModel(hourly_profile=[1.0] * 23)

    def test_invalid_profile_value_raises(self):
        profile = [1.0] * 24
        profile[5] = 1.1   # > 1 is invalid
        with pytest.raises(ValueError):
            CongestionModel(hourly_profile=profile)


# =================================================================
# Test Group 3: Travel Time Engine
# =================================================================

class TestTravelTimeEngine:
    def test_travel_time_positive(self):
        engine = TravelTimeEngine(base_speed_kmh=30.0, distance_method="haversine")
        tt = engine.travel_time_minutes(*_BLR_CENTER, *_BLR_WHITEFIELD, hour_of_day=10)
        assert tt > 0

    def test_travel_time_increases_at_peak(self):
        engine = TravelTimeEngine(base_speed_kmh=30.0, distance_method="haversine")
        tt_peak = engine.travel_time_minutes(*_BLR_CENTER, *_BLR_WHITEFIELD, hour_of_day=8)
        tt_offpeak = engine.travel_time_minutes(*_BLR_CENTER, *_BLR_WHITEFIELD, hour_of_day=3)
        assert tt_peak > tt_offpeak, "Peak-hour travel time should be longer than off-peak"

    def test_travel_time_zero_distance(self):
        engine = TravelTimeEngine(base_speed_kmh=30.0)
        tt = engine.travel_time_minutes(12.97, 77.59, 12.97, 77.59, hour_of_day=10)
        assert abs(tt) < 1e-6

    def test_pairwise_matrix_shape_and_nonneg(self):
        engine = TravelTimeEngine()
        lats = np.linspace(12.80, 13.10, 50)
        lons = np.linspace(77.45, 77.75, 50)
        site_lats = np.array([12.97, 13.00])
        site_lons = np.array([77.59, 77.62])
        M = engine.pairwise_travel_time_matrix(lats, lons, site_lats, site_lons, hour_of_day=10)
        assert M.shape == (50, 2)
        assert (M >= 0).all()

    def test_timeliness_score_bounds(self):
        engine = TravelTimeEngine()
        # Donor at site: score = 1.0
        s1 = engine.timeliness_score(0.0, t_critical_minutes=60.0)
        assert s1 == pytest.approx(1.0)
        # Donor exactly at horizon: score = 0.0
        s2 = engine.timeliness_score(60.0, t_critical_minutes=60.0)
        assert s2 == pytest.approx(0.0)
        # Donor beyond horizon: clamped to 0.0
        s3 = engine.timeliness_score(90.0, t_critical_minutes=60.0)
        assert s3 == pytest.approx(0.0)

    def test_timeliness_score_mid_range(self):
        engine = TravelTimeEngine()
        score = engine.timeliness_score(30.0, t_critical_minutes=60.0)
        assert score == pytest.approx(0.5)

    def test_network_grid_slower_than_haversine(self):
        e_hav = TravelTimeEngine(base_speed_kmh=30.0, distance_method="haversine")
        e_net = TravelTimeEngine(base_speed_kmh=30.0, distance_method="network_grid")
        tt_hav = e_hav.travel_time_minutes(*_BLR_CENTER, *_BLR_WHITEFIELD, hour_of_day=10)
        tt_net = e_net.travel_time_minutes(*_BLR_CENTER, *_BLR_WHITEFIELD, hour_of_day=10)
        assert tt_net > tt_hav, "Network grid should produce longer travel time than Haversine"


# =================================================================
# Test Group 4: Feasibility Filter
# =================================================================

class TestFeasibilityFilter:
    @pytest.fixture
    def ff(self):
        engine = TravelTimeEngine(base_speed_kmh=30.0, distance_method="haversine")
        return FeasibilityFilter(travel_time_engine=engine, safety_margin_fraction=0.15)

    def test_donor_at_zero_distance_is_feasible(self, ff):
        site_lat, site_lon = 12.97, 77.59
        result = ff.filter_donors(
            donor_ids=["D0"], donor_lats=np.array([site_lat]),
            donor_lons=np.array([site_lon]),
            site_lat=site_lat, site_lon=site_lon,
            urgency="CRITICAL", hour_of_day=10,
        )
        assert result.n_feasible == 1
        assert result.feasible_results[0].is_feasible is True

    def test_distant_donor_is_infeasible_for_critical(self, ff):
        # Place donor 100 km away; CRITICAL = 60 min → ~200 km range at 30 km/h free flow
        # But at 100 km + congestion at peak + detour → should be infeasible
        # Use a donor that is clearly too far (200 km away)
        result = ff.filter_donors(
            donor_ids=["D_FAR"],
            donor_lats=np.array([13.80]),   # ~100+ km north
            donor_lons=np.array([77.59]),
            site_lat=12.97, site_lon=77.59,
            urgency="CRITICAL", hour_of_day=8,  # peak hour
        )
        assert result.n_feasible == 0
        assert result.infeasible_results[0].is_feasible is False

    def test_safety_margin_reduces_effective_horizon(self, ff):
        t_crit = ff.horizons["URGENT"]    # 120 min
        t_eff = ff.effective_horizon("URGENT")
        assert t_eff < t_crit
        assert t_eff == pytest.approx(t_crit * 0.85)

    def test_unknown_urgency_raises(self, ff):
        with pytest.raises(ValueError, match="Unknown urgency tier"):
            ff.effective_horizon("EMERGENCY")

    def test_feasibility_rate_in_unit_interval(self, ff):
        n = 20
        rng = np.random.default_rng(42)
        lats = rng.uniform(12.80, 13.10, n)
        lons = rng.uniform(77.45, 77.75, n)
        result = ff.filter_donors(
            donor_ids=[f"D{i}" for i in range(n)],
            donor_lats=lats, donor_lons=lons,
            site_lat=12.97, site_lon=77.59,
            urgency="URGENT", hour_of_day=10,
        )
        assert 0.0 <= result.feasibility_rate <= 1.0

    def test_total_equals_feasible_plus_infeasible(self, ff):
        n = 15
        rng = np.random.default_rng(7)
        lats = rng.uniform(12.80, 13.10, n)
        lons = rng.uniform(77.45, 77.75, n)
        result = ff.filter_donors(
            donor_ids=[f"D{i}" for i in range(n)],
            donor_lats=lats, donor_lons=lons,
            site_lat=12.97, site_lon=77.59,
            urgency="SCHEDULED", hour_of_day=14,
        )
        assert result.n_feasible + len(result.infeasible_results) == n

    def test_batch_mode_returns_one_result_per_scenario(self, ff):
        n = 10
        rng = np.random.default_rng(99)
        lats = rng.uniform(12.80, 13.10, n)
        lons = rng.uniform(77.45, 77.75, n)
        scenarios = [
            {"request_id": f"R{i}", "site_lat": 12.97, "site_lon": 77.59,
             "urgency": "URGENT", "hour_of_day": 10}
            for i in range(5)
        ]
        results = ff.batch_filter(
            [f"D{i}" for i in range(n)], lats, lons, scenarios
        )
        assert len(results) == 5

    def test_feasible_donor_timeliness_in_unit_interval(self, ff):
        site_lat, site_lon = 12.97, 77.59
        # Place a donor very close to site
        result = ff.filter_donors(
            donor_ids=["D_CLOSE"],
            donor_lats=np.array([12.971]),
            donor_lons=np.array([77.591]),
            site_lat=site_lat, site_lon=site_lon,
            urgency="URGENT", hour_of_day=10,
        )
        if result.feasible_results:
            score = result.feasible_results[0].timeliness_score
            assert 0.0 <= score <= 1.0


# =================================================================
# Test Group 5: Synthetic Generator
# =================================================================

class TestSyntheticGenerator:
    @pytest.fixture
    def gen(self):
        return SyntheticUrbanGenerator(seed=42)

    def test_donor_pool_size(self, gen):
        pool = gen.generate_donor_pool(n=200)
        assert pool.n_donors == 200

    def test_coordinates_in_bounding_box(self, gen):
        pool = gen.generate_donor_pool(n=100)
        for d in pool.records:
            assert gen.lat_min <= d.latitude <= gen.lat_max, (
                f"Donor latitude {d.latitude} outside bounding box"
            )
            assert gen.lon_min <= d.longitude <= gen.lon_max, (
                f"Donor longitude {d.longitude} outside bounding box"
            )

    def test_eligibility_respects_deferral_rule(self, gen):
        pool = gen.generate_donor_pool(n=500)
        for d in pool.records:
            if d.recency_months < 2.0:
                assert not d.is_eligible, (
                    f"Donor with recency={d.recency_months:.1f} should be ineligible"
                )

    def test_determinism_same_seed(self):
        gen_a = SyntheticUrbanGenerator(seed=42)
        gen_b = SyntheticUrbanGenerator(seed=42)
        pool_a = gen_a.generate_donor_pool(n=50)
        pool_b = gen_b.generate_donor_pool(n=50)
        for a, b in zip(pool_a.records, pool_b.records):
            assert a.latitude == pytest.approx(b.latitude)
            assert a.longitude == pytest.approx(b.longitude)
            assert a.blood_type == b.blood_type

    def test_different_seeds_produce_different_pools(self):
        gen_a = SyntheticUrbanGenerator(seed=1)
        gen_b = SyntheticUrbanGenerator(seed=2)
        pool_a = gen_a.generate_donor_pool(n=50)
        pool_b = gen_b.generate_donor_pool(n=50)
        lats_a = [d.latitude for d in pool_a.records]
        lats_b = [d.latitude for d in pool_b.records]
        assert lats_a != lats_b

    def test_scenario_generation_count(self, gen):
        site_lats, site_lons, site_ids = gen.generate_sites(n_sites=3)
        scenarios = gen.generate_emergency_scenarios(
            n=30, site_lats=site_lats, site_lons=site_lons, site_ids=site_ids
        )
        assert len(scenarios) == 30

    def test_scenario_urgency_values(self, gen):
        site_lats, site_lons, site_ids = gen.generate_sites()
        scenarios = gen.generate_emergency_scenarios(
            n=100, site_lats=site_lats, site_lons=site_lons, site_ids=site_ids
        )
        valid_urgencies = {"CRITICAL", "URGENT", "SCHEDULED"}
        for s in scenarios:
            assert s.urgency in valid_urgencies

    def test_data_tag_immutable(self, gen):
        pool = gen.generate_donor_pool(n=5)
        for d in pool.records:
            assert d.data_tag == "SYNTHETIC_SIMULATION"
