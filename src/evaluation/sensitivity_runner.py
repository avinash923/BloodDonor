"""
src/evaluation/sensitivity_runner.py
Sensitivity analysis framework.

Evaluates system stability across ranges of parameters (e.g., fatigue lambda,
urgency weight) to test robustness.
"""

from __future__ import annotations

import numpy as np
from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SensitivityRunner:
    def __init__(self, base_engine: UDIEngine):
        self.base_engine = base_engine

    def sweep_fatigue_lambda(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        lambda_values: list[float],
        true_responders: set[str],
        k: int = 10
    ) -> dict[float, float]:
        """
        Sweep across fatigue lambda values, returning precision@k for each.
        """
        results = {}
        for l in lambda_values:
            # Need a custom config for this lambda
            # Reusing weights from base_engine but updating lambda
            alt_engine = UDIEngine(
                w_urgency=self.base_engine.w_urgency,
                w_timeliness=self.base_engine.w_timeliness,
                w_probability=self.base_engine.w_probability,
                lambda_fatigue=l,
                urgency_scores=self.base_engine.urgency_scores,
                urgency_horizons=self.base_engine.urgency_horizons,
                safety_margin=self.base_engine.safety_margin
            )
            
            ranked = alt_engine.score_and_rank(candidates, urgency)
            from src.ranking.udi_engine import precision_at_k
            results[l] = precision_at_k(ranked, k, true_responders)
            
        return results
