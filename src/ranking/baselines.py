"""
src/ranking/baselines.py
Operational baseline ranking strategies for EXP-3 comparison.

Implements four baselines against which the UDI is evaluated:
  1. RandomRanking        — random permutation (lower bound)
  2. ProximityFirstRanking — sort ascending by travel time (nearest first)
  3. ProbabilityFirstRanking — sort descending by ML response probability
  4. HeuristicRanking     — proximity × probability product (naive combined)

All baselines share the same DonorCandidate input schema as the UDI engine
and return a list of UDIResult-compatible dicts for uniform evaluation.

DATA STREAM: SYNTHETIC_SIMULATION (Phase 4 ranking evaluation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ranking.udi_engine import DonorCandidate, UDIResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =================================================================
# Shared Result adapter (keeps evaluation interface uniform)
# =================================================================

def _make_result(
    candidate: DonorCandidate,
    score: float,
    rank: int,
    t_eff: float,
) -> UDIResult:
    """Wrap a baseline score into a UDIResult for uniform comparison."""
    is_feasible = candidate.is_eligible and candidate.travel_time_minutes <= t_eff
    return UDIResult(
        donor_id=candidate.donor_id,
        udi_score=float(np.clip(score, 0.0, 1.0)),
        urgency_component=0.0,
        timeliness_component=0.0,
        probability_component=0.0,
        weighted_sum=float(score),
        fatigue_discount=1.0,
        travel_time_minutes=candidate.travel_time_minutes,  # Required by simulator for arrival scheduling
        rank=rank,
        is_feasible=is_feasible,
        is_eligible=candidate.is_eligible,
    )


_URGENCY_HORIZONS: dict[str, int] = {
    "CRITICAL":  60,
    "URGENT":    120,
    "SCHEDULED": 240,
}


# =================================================================
# 1. Random Ranking
# =================================================================

class RandomRanking:
    """
    Assign random priority scores to all eligible donors.
    Serves as a stochastic lower-bound reference.

    Args:
        seed: Random seed for reproducibility.
    """

    name: str = "Random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def rank(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        safety_margin: float = 0.15,
    ) -> list[UDIResult]:
        """Return candidates in a random order."""
        t_eff = _URGENCY_HORIZONS[urgency] * (1.0 - safety_margin)
        eligible = [c for c in candidates if c.is_eligible]
        scores = self.rng.random(size=len(eligible))
        paired = sorted(zip(scores, eligible), reverse=True)
        results = []
        for rank_idx, (score, c) in enumerate(paired):
            results.append(_make_result(c, score, rank_idx + 1, t_eff))
        logger.debug("Random ranking: %d candidates.", len(results))
        return results


# =================================================================
# 2. Proximity-First Ranking (nearest donor first)
# =================================================================

class ProximityFirstRanking:
    """
    Rank donors by ascending travel time (nearest to site = highest priority).

    This is the conventional distance-only dispatch policy used as a
    primary operational baseline in emergency services literature.
    """

    name: str = "Proximity-First"

    def rank(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        safety_margin: float = 0.15,
    ) -> list[UDIResult]:
        """Return candidates sorted nearest-first."""
        t_crit = float(_URGENCY_HORIZONS[urgency])
        t_eff = t_crit * (1.0 - safety_margin)
        eligible = [c for c in candidates if c.is_eligible]

        # Proximity score: 1 - (τ / T_crit), clamped to [0, 1]
        def prox_score(c: DonorCandidate) -> float:
            return float(np.clip(1.0 - c.travel_time_minutes / t_crit, 0.0, 1.0))

        paired = sorted(eligible, key=prox_score, reverse=True)
        results = []
        for rank_idx, c in enumerate(paired):
            score = prox_score(c)
            results.append(_make_result(c, score, rank_idx + 1, t_eff))

        logger.debug("Proximity-First ranking: %d candidates.", len(results))
        return results


# =================================================================
# 3. Probability-First Ranking (highest ML propensity first)
# =================================================================

class ProbabilityFirstRanking:
    """
    Rank donors by descending predicted response probability.

    Useful for assessing the standalone value of the ML model
    without any spatial filtering or urgency weighting.
    """

    name: str = "Probability-First"

    def rank(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        safety_margin: float = 0.15,
    ) -> list[UDIResult]:
        """Return candidates sorted by ML probability descending."""
        t_eff = _URGENCY_HORIZONS[urgency] * (1.0 - safety_margin)
        eligible = [c for c in candidates if c.is_eligible]
        paired = sorted(eligible, key=lambda c: c.effective_probability, reverse=True)
        results = []
        for rank_idx, c in enumerate(paired):
            results.append(
                _make_result(c, c.effective_probability, rank_idx + 1, t_eff)
            )
        logger.debug("Probability-First ranking: %d candidates.", len(results))
        return results


# =================================================================
# 4. Heuristic Combined Ranking (proximity × probability)
# =================================================================

class HeuristicRanking:
    """
    Naive combined baseline: score = timeliness × response_probability.

    A simple product that merges proximity and propensity without
    multi-criteria weight tuning or urgency awareness. Represents the
    "reasonable but untuned" baseline that practitioners might implement
    without formal UDI formulation.
    """

    name: str = "Heuristic (Proximity × Probability)"

    def rank(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        safety_margin: float = 0.15,
    ) -> list[UDIResult]:
        """Return candidates sorted by timeliness × probability descending."""
        t_crit = float(_URGENCY_HORIZONS[urgency])
        t_eff = t_crit * (1.0 - safety_margin)
        eligible = [c for c in candidates if c.is_eligible]

        def heuristic_score(c: DonorCandidate) -> float:
            timeliness = float(np.clip(1.0 - c.travel_time_minutes / t_crit, 0.0, 1.0))
            return timeliness * c.effective_probability

        paired = sorted(eligible, key=heuristic_score, reverse=True)
        results = []
        for rank_idx, c in enumerate(paired):
            score = heuristic_score(c)
            results.append(_make_result(c, score, rank_idx + 1, t_eff))

        logger.debug("Heuristic ranking: %d candidates.", len(results))
        return results


# =================================================================
# Registry: all baselines by name
# =================================================================

def get_all_baselines(seed: int = 42) -> dict[str, object]:
    """Return a dict of all instantiated baseline rankers keyed by name."""
    return {
        "Random": RandomRanking(seed=seed),
        "Proximity-First": ProximityFirstRanking(),
        "Probability-First": ProbabilityFirstRanking(),
        "Heuristic": HeuristicRanking(),
    }
