"""
src/simulation/response_model.py
Stochastic donor response latency and acceptance sampling.

Models the donor's reaction to a notification:
1.  **Will they accept?** Bernoulli trial using their calibrated response probability.
2.  **How long will they take to respond?** LogNormal distribution for latency,
    modified by alert fatigue.

DATA STREAM: SYNTHETIC_SIMULATION
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DonorResponseOutcome:
    """Outcome of a simulated donor notification."""
    accepted: bool
    latency_minutes: float


class StochasticResponseModel:
    """
    Models donor acceptance and response latency.

    Args:
        mu:           LogNormal mean parameter for latency.
        sigma:        LogNormal std parameter for latency.
        fatigue_mult: Multiplier for latency per recent alert.
        min_latency:  Minimum reaction time (minutes).
        seed:         Random seed for stochastic trials.
    """

    def __init__(
        self,
        mu: float = 2.0,
        sigma: float = 0.8,
        fatigue_mult: float = 1.2,
        min_latency: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.fatigue_mult = fatigue_mult
        self.min_latency = min_latency
        self.rng = np.random.default_rng(seed)

    def sample_response(
        self,
        probability: float,
        recent_alerts: int = 0,
        donor_id: Optional[str] = None,
    ) -> DonorResponseOutcome:
        """
        Sample the donor's response outcome and latency.

        Args:
            probability:   The donor's true response probability (e.g., from ML).
            recent_alerts: Number of recent alerts (fatigue).
            donor_id:      Optional donor ID for shared random variates across policies.

        Returns:
            DonorResponseOutcome (accepted, latency_minutes).
        """
        # If pregenerated variates dict exists and donor_id is provided, use shared variate
        if donor_id is not None and hasattr(self, "_donor_variates") and donor_id in self._donor_variates:
            u, base_latency = self._donor_variates[donor_id]
        else:
            u = self.rng.random()
            base_latency = self.rng.lognormal(mean=self.mu, sigma=self.sigma)

        # 1. Acceptance trial
        accepted = bool(u < probability)

        # 2. Fatigue slows down response
        fatigue_factor = self.fatigue_mult ** recent_alerts
        latency = base_latency * fatigue_factor
        latency = max(self.min_latency, latency)

        return DonorResponseOutcome(accepted=accepted, latency_minutes=latency)

    def pregenerate_donor_variates(self, donor_ids: list[str]) -> None:
        """
        Pre-generate uniform and lognormal random variates for a set of donors.
        Ensures a donor's simulated outcome is invariant to contact ordering across policies.
        """
        self._donor_variates = {}
        for did in donor_ids:
            u = float(self.rng.random())
            base_lat = float(self.rng.lognormal(mean=self.mu, sigma=self.sigma))
            self._donor_variates[did] = (u, base_lat)
