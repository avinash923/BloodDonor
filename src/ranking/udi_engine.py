"""
src/ranking/udi_engine.py
Unified Donor Index (UDI) multi-criteria scoring and ranking engine.

Formula:
    UDI(i, j, e) = [ w_u · U(e)
                    + w_t · (1 - τ_ij / T_crit(e))
                    + w_p · P̂_response(i)
                   ] × Φ_fatigue(i)

Components:
    U(e)               — Normalised urgency score ∈ [0, 1]
    1 - τ/T_crit       — Timeliness score ∈ [0, 1] (0 if infeasible)
    P̂_response(i)      — Calibrated ML response probability ∈ [0, 1]
    Φ_fatigue(i)       — Alert-fatigue discount ∈ (0, 1]
                         = exp(-λ_fatigue · N_recent_alerts(i))

    w_u + w_t + w_p = 1 (enforced at construction)

Output: UDI ∈ [0, 1] for feasible donors; 0.0 for infeasible donors.

DATA STREAM: SYNTHETIC_SIMULATION (Phase 4 ranking evaluation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------
# Urgency score map — must mirror request_schema.py and udi_weights.yaml
# -----------------------------------------------------------------
_URGENCY_SCORES: dict[str, float] = {
    "CRITICAL":  1.0,
    "URGENT":    0.7,
    "SCHEDULED": 0.4,
}

_URGENCY_HORIZONS: dict[str, int] = {
    "CRITICAL":  60,
    "URGENT":    120,
    "SCHEDULED": 240,
}


# =================================================================
# Input record for a single donor candidate
# =================================================================

@dataclass
class DonorCandidate:
    """
    Unified input record for UDI computation.

    All floating-point scores must be in [0, 1]; violations raise ValueError.
    travel_time_minutes may exceed T_critical (donor will receive UDI = 0).
    """
    donor_id: str
    travel_time_minutes: Optional[float] = None # τ_ij from TravelTimeEngine, None if missing
    recruitment_response_probability: Optional[float] = None  # P̂(i) from recruitment model (if any)
    synthetic_response_propensity: Optional[float] = None     # Simulated propensity (for in-silico)
    recent_alert_count: int = 0         # N_recent_alerts(i) for fatigue penalty
    is_eligible: bool = True            # Hard eligibility gate (8-week deferral etc.)
    
    @property
    def effective_probability(self) -> float:
        """Use true recruitment probability if available, else synthetic propensity. Never UCI behavior score."""
        if self.recruitment_response_probability is not None:
            return self.recruitment_response_probability
        if self.synthetic_response_propensity is not None:
            return self.synthetic_response_propensity
        raise ValueError("Neither recruitment response probability nor synthetic propensity provided.")

    def __post_init__(self) -> None:
        if self.recruitment_response_probability is not None and not (0.0 <= self.recruitment_response_probability <= 1.0):
            raise ValueError(f"recruitment_response_probability must be in [0, 1]")
        if self.synthetic_response_propensity is not None and not (0.0 <= self.synthetic_response_propensity <= 1.0):
            raise ValueError(f"synthetic_response_propensity must be in [0, 1]")
        if self.travel_time_minutes is not None and self.travel_time_minutes < 0:
            raise ValueError(
                f"travel_time_minutes must be ≥ 0, "
                f"got {self.travel_time_minutes} for donor {self.donor_id}."
            )
        if self.recent_alert_count < 0:
            raise ValueError(
                f"recent_alert_count must be ≥ 0, "
                f"got {self.recent_alert_count}."
            )


@dataclass
class UDIResult:
    """
    Output record for a single donor's UDI computation.
    """
    donor_id: str
    udi_score: float                    # Final UDI ∈ [0, 1]
    urgency_component: float            # w_u · U(e)
    timeliness_component: float         # w_t · (1 - τ/T_crit)
    probability_component: float        # P̂(i) or w_p · P̂(i)
    weighted_sum: float                 # Pre-fatigue score
    fatigue_discount: float             # Φ_fatigue(i) ∈ (0, 1]
    travel_time_minutes: Optional[float] = None
    rank: int = 0                       # Assigned rank (1 = highest priority)
    is_feasible: bool = True
    is_eligible: bool = True


# =================================================================
# UDI Engine
# =================================================================

class UDIEngine:
    """
    Computes Unified Donor Index scores for a pool of donor candidates
    given a specific emergency request context.

    Args:
        w_urgency:       Weight for urgency component (w_u).
        w_timeliness:    Weight for timeliness component (w_t).
        w_probability:   Weight for ML probability component (w_p).
        lambda_fatigue:  Exponential decay rate for alert-fatigue penalty.
        urgency_scores:  Override map for urgency tier → U(e) score.
        urgency_horizons: Override map for urgency tier → T_crit (minutes).
        safety_margin:   Fraction of T_crit reserved for processing (mirrors
                         FeasibilityFilter); timeliness computed against
                         raw T_crit, but donors beyond T_eff get score = 0.
    """

    def __init__(
        self,
        w_urgency: float = 0.30,
        w_timeliness: float = 0.40,
        w_probability: float = 0.30,
        lambda_fatigue: float = 0.30,
        urgency_scores: dict[str, float] | None = None,
        urgency_horizons: dict[str, int] | None = None,
        safety_margin: float = 0.15,
        formula_mode: str = "multiplicative",
        probability_floor: float = 0.0,
    ) -> None:
        if formula_mode not in ("additive", "multiplicative"):
            raise ValueError("formula_mode must be 'additive' or 'multiplicative'")
        self.formula_mode = formula_mode
        
        if self.formula_mode == "additive":
            self._validate_weights(w_urgency, w_timeliness, w_probability)
            
        self.w_urgency = w_urgency
        self.w_timeliness = w_timeliness
        self.w_probability = w_probability
        self.lambda_fatigue = lambda_fatigue
        self.urgency_scores = urgency_scores or dict(_URGENCY_SCORES)
        self.urgency_horizons = urgency_horizons or dict(_URGENCY_HORIZONS)
        self.safety_margin = safety_margin
        self.probability_floor = probability_floor

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def score_candidate(
        self,
        candidate: DonorCandidate,
        urgency: str,
        t_remaining: Optional[float] = None,
    ) -> UDIResult:
        """
        Compute UDI score for a single donor candidate.

        Ineligible donors and donors whose travel time exceeds the effective
        horizon are assigned UDI = 0.0 and marked infeasible/ineligible.

        Args:
            candidate: DonorCandidate with travel time and probability.
            urgency:   Emergency tier string.
            t_remaining: Optional remaining time budget (minutes).

        Returns:
            UDIResult with score, components, and feasibility flags.
        """
        u_score = self._urgency_score(urgency)
        t_crit = float(self.urgency_horizons[urgency])
        t_eff = t_crit * (1.0 - self.safety_margin)
        t_rem = t_remaining if t_remaining is not None else t_crit

        # --- Eligibility hard gate ---
        if not candidate.is_eligible:
            return UDIResult(
                donor_id=candidate.donor_id,
                udi_score=0.0,
                urgency_component=0.0, timeliness_component=0.0,
                probability_component=0.0, weighted_sum=0.0,
                fatigue_discount=0.0, is_feasible=False, is_eligible=False,
            )

        # --- Feasibility hard gate ---
        eta = candidate.travel_time_minutes
        
        if self.formula_mode == "additive":
            if eta is None or eta > t_eff:
                return UDIResult(
                    donor_id=candidate.donor_id, udi_score=0.0,
                    urgency_component=0.0, timeliness_component=0.0, probability_component=0.0,
                    weighted_sum=0.0, fatigue_discount=0.0, travel_time_minutes=eta,
                    is_feasible=False, is_eligible=True,
                )
            
            urgency_comp = self.w_urgency * u_score
            timeliness_raw = float(np.clip(1.0 - (eta / t_crit), 0.0, 1.0))
            timeliness_comp = self.w_timeliness * timeliness_raw
            probability_comp = self.w_probability * candidate.effective_probability
            weighted_sum = urgency_comp + timeliness_comp + probability_comp
            fatigue = self._fatigue_discount(candidate.recent_alert_count)
            udi = float(np.clip(weighted_sum * fatigue, 0.0, 1.0))
            
            return UDIResult(
                donor_id=candidate.donor_id, udi_score=udi,
                urgency_component=urgency_comp, timeliness_component=timeliness_comp, probability_component=probability_comp,
                weighted_sum=weighted_sum, fatigue_discount=fatigue, travel_time_minutes=eta,
                is_feasible=True, is_eligible=True,
            )
            
        elif self.formula_mode == "multiplicative":
            t_buffer = t_crit * self.safety_margin
            t_budget = max(0.0, min(t_rem, t_crit) - t_buffer)
            
            if eta is None or t_budget <= 0 or eta > t_budget:
                return UDIResult(
                    donor_id=candidate.donor_id, udi_score=0.0,
                    urgency_component=0.0, timeliness_component=0.0, probability_component=0.0,
                    weighted_sum=0.0, fatigue_discount=0.0, travel_time_minutes=eta,
                    is_feasible=False, is_eligible=True,
                )
                
            t_score = float(np.clip(1.0 - (eta / t_budget), 0.0, 1.0))
            probability_comp = max(candidate.effective_probability, self.probability_floor)
            fatigue = self._fatigue_discount(candidate.recent_alert_count)
            
            weighted_sum = probability_comp * t_score
            udi = float(np.clip(weighted_sum * fatigue, 0.0, 1.0))
            
            return UDIResult(
                donor_id=candidate.donor_id, udi_score=udi,
                urgency_component=0.0, timeliness_component=t_score, probability_component=probability_comp,
                weighted_sum=weighted_sum, fatigue_discount=fatigue, travel_time_minutes=eta,
                is_feasible=True, is_eligible=True,
            )

    def score_and_rank(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        t_remaining: Optional[float] = None,
    ) -> list[UDIResult]:
        """
        Score all candidates and return a ranked list (highest UDI first).

        Infeasible and ineligible donors are placed at the tail of the
        ranked list with rank assigned after feasible donors.

        Args:
            candidates: List of DonorCandidate records.
            urgency:    Emergency urgency tier.
            t_remaining: Optional remaining time budget (minutes).

        Returns:
            List of UDIResult sorted by udi_score descending.
        """
        results = [self.score_candidate(c, urgency, t_remaining) for c in candidates]
        
        # Tie-breaking: 1. UDI desc, 2. ETA asc, 3. ResponseScore desc, 4. DonorID asc
        def sort_key(r: UDIResult):
            eta = r.travel_time_minutes if r.travel_time_minutes is not None else float('inf')
            return (-r.udi_score, eta, -r.probability_component, r.donor_id)
            
        results.sort(key=sort_key)
        for rank_idx, r in enumerate(results):
            r.rank = rank_idx + 1
        logger.debug(
            "UDI scoring: urgency=%s | %d candidates | %d feasible | "
            "top UDI=%.4f",
            urgency,
            len(results),
            sum(1 for r in results if r.is_feasible),
            results[0].udi_score if results else 0.0,
        )
        return results

    def top_k(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        k: int,
    ) -> list[UDIResult]:
        """
        Return the top-k ranked feasible donors by UDI score.

        Args:
            candidates: Full donor pool.
            urgency:    Emergency tier.
            k:          Number of top candidates to return.

        Returns:
            Up to k UDIResult records (may be fewer if feasible pool < k).
        """
        ranked = self.score_and_rank(candidates, urgency)
        feasible = [r for r in ranked if r.is_feasible and r.is_eligible]
        return feasible[:k]

    # -----------------------------------------------------------------
    # Batch scoring across multiple weight configurations
    # -----------------------------------------------------------------

    def score_with_config(
        self,
        candidates: list[DonorCandidate],
        urgency: str,
        config: dict,
    ) -> list[UDIResult]:
        """
        Score candidates using an alternative weight configuration dict
        (e.g., from udi_weights.yaml experimental_configs).

        Args:
            candidates: Donor pool.
            urgency:    Emergency tier.
            config:     Dict with keys 'w_urgency', 'w_timeliness', 'w_probability'.

        Returns:
            Ranked list of UDIResult under the alternative config.
        """
        alt_engine = UDIEngine(
            w_urgency=config["w_urgency"],
            w_timeliness=config["w_timeliness"],
            w_probability=config["w_probability"],
            lambda_fatigue=self.lambda_fatigue,
            urgency_scores=self.urgency_scores,
            urgency_horizons=self.urgency_horizons,
            safety_margin=self.safety_margin,
            formula_mode=self.formula_mode,
            probability_floor=config.get("probability_floor", self.probability_floor),
        )
        return alt_engine.score_and_rank(candidates, urgency)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _urgency_score(self, urgency: str) -> float:
        score = self.urgency_scores.get(urgency)
        if score is None:
            raise ValueError(
                f"Unknown urgency tier '{urgency}'. "
                f"Valid: {list(self.urgency_scores.keys())}"
            )
        return score

    def _fatigue_discount(self, n_recent_alerts: int) -> float:
        """
        Φ_fatigue(i) = exp(-λ · N_recent_alerts)
        Bounded to (0, 1]. Never reaches exactly 0.
        """
        return math.exp(-self.lambda_fatigue * max(0, n_recent_alerts))

    @staticmethod
    def _validate_weights(w_u: float, w_t: float, w_p: float) -> None:
        total = w_u + w_t + w_p
        if not all(w >= 0 for w in (w_u, w_t, w_p)):
            raise ValueError("All UDI weights must be non-negative.")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"UDI weights must sum to 1.0, got {total:.6f} "
                f"(w_u={w_u}, w_t={w_t}, w_p={w_p})."
            )

    def get_weights(self) -> dict[str, float]:
        """Return current weight configuration."""
        return {
            "w_urgency": self.w_urgency,
            "w_timeliness": self.w_timeliness,
            "w_probability": self.w_probability,
            "lambda_fatigue": self.lambda_fatigue,
        }


# =================================================================
# Precision@k Metric
# =================================================================

def precision_at_k(
    ranked_results: list[UDIResult],
    k: int,
    ground_truth_responders: set[str],
) -> float:
    """
    Compute Precision@k: fraction of top-k ranked donors who are
    in the ground-truth responder set.

    Used in EXP-3 to compare UDI vs. baseline ranking strategies.

    Args:
        ranked_results:          Ordered list of UDIResult (rank 1 = first).
        k:                       Cut-off rank.
        ground_truth_responders: Set of donor_ids who actually respond.

    Returns:
        Precision@k ∈ [0, 1].
    """
    if k <= 0 or not ranked_results:
        return 0.0
    
    top_k_results = ranked_results[:k]
    top_k_ids = {r.donor_id for r in top_k_results}
    hits = top_k_ids & ground_truth_responders
    
    denominator = min(k, len(ranked_results))
    return len(hits) / denominator
