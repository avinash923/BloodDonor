"""
src/evaluation/ablation_runner.py
Component ablation testing framework.

Runs the ranking and simulation engines using different weight configurations
(e.g., dropping Probability, dropping Timeliness) to evaluate their individual impact.
"""

from __future__ import annotations

import pandas as pd
from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AblationRunner:
    """
    Executes an ablation study across configured weight drops.
    """
    def __init__(self, base_engine: UDIEngine, ablation_configs: dict[str, dict[str, float]]):
        self.base_engine = base_engine
        self.ablation_configs = ablation_configs
        
    def evaluate_configs(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        true_responders: set[str],
        k: int = 10
    ) -> dict[str, float]:
        """
        Evaluate full model and all ablations on a single scenario, returning precision@k.
        """
        results = {}
        
        # 1. Full Model
        full_ranked = self.base_engine.score_and_rank(candidates, urgency)
        from src.ranking.udi_engine import precision_at_k
        results["Full_UDI"] = precision_at_k(full_ranked, k, true_responders)
        
        # 2. Ablations
        for config_name, weights in self.ablation_configs.items():
            ablated_ranked = self.base_engine.score_with_config(
                candidates, urgency, weights
            )
            results[config_name] = precision_at_k(ablated_ranked, k, true_responders)
            
        return results
