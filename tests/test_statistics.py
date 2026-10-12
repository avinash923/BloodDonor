"""
tests/test_statistics.py
Unit tests for Phase 6: Statistical Evaluation tools.

Verifies:
  1. Paired t-test correctly identifies significant differences.
  2. Wilcoxon signed-rank test handles identical samples properly.
  3. Bootstrap CI generates reasonable bounds.
  4. Advanced metrics (AFI, Shortfall Ratio) compute correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.evaluation.stats_testing import paired_ttest, wilcoxon_test, bootstrap_ci
from src.evaluation.metrics import calculate_alert_fatigue_index, calculate_shortfall_ratio
from src.data.schemas.dispatch_schema import EpisodeResult


class TestStatsTesting:
    def test_paired_ttest_significant(self):
        # Sample 1 consistently higher than Sample 2
        s1 = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        s2 = np.array([8.0, 9.5, 10.0, 8.0, 9.0])
        
        res = paired_ttest(s1, s2)
        assert res.significant is True
        assert res.p_value < 0.05
        assert res.effect_size > 1.0 # Large effect

    def test_wilcoxon_identical(self):
        # Identical samples should not be significant
        s1 = np.array([1.0, 2.0, 3.0, 4.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0])
        
        res = wilcoxon_test(s1, s2)
        assert res.significant is False
        assert res.p_value == 1.0

    def test_bootstrap_ci(self):
        # Data centered around 10
        data = np.random.normal(loc=10.0, scale=1.0, size=100)
        lower, upper = bootstrap_ci(data, seed=42)
        
        assert lower < 10.5
        assert upper > 9.5
        assert lower < upper

class TestAdvancedMetrics:
    def test_alert_fatigue_index(self):
        ep = EpisodeResult(
            episode_id="E1",
            request_id="R1",
            units_required=5,
            units_fulfilled=5,
            total_notifications_sent=10,
            waves_used=2,
            dispatch_log=[]
        )
        assert calculate_alert_fatigue_index(ep) == 2.0

    def test_shortfall_ratio(self):
        ep = EpisodeResult(
            episode_id="E1",
            request_id="R1",
            units_required=10,
            units_fulfilled=7,
            total_notifications_sent=15,
            waves_used=2,
            dispatch_log=[]
        )
        ep.compute_derived_metrics()
        assert calculate_shortfall_ratio(ep) == 0.3
