"""
src/simulation/dispatch_cascade.py
Multi-wave batch dispatch and dynamic recruitment logic.

Calculates how many donors to notify in a given wave based on
the required units and the current shortfall.
"""

from __future__ import annotations

import math
from typing import Sequence

from src.ranking.udi_engine import UDIResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DispatchCascadePolicy:
    """
    Determines the batch size for donor notifications.
    
    Args:
        w1_mult: Multiplier for the initial wave (e.g., 3.0 * Q).
        w_sub_mult: Multiplier for subsequent waves (e.g., 2.0 * Shortfall).
    """

    def __init__(
        self,
        w1_mult: float = 3.0,
        w_sub_mult: float = 2.0,
    ) -> None:
        self.w1_mult = w1_mult
        self.w_sub_mult = w_sub_mult

    def get_batch_size(
        self,
        wave_number: int,
        units_required: int,
        shortfall: int,
    ) -> int:
        """
        Calculate how many donors to notify in the current wave.

        Args:
            wave_number:    Current wave index (1-indexed).
            units_required: Total units required for the episode.
            shortfall:      Remaining units needed.

        Returns:
            Number of donors to select for this wave.
        """
        if shortfall <= 0:
            return 0

        if wave_number == 1:
            size = math.ceil(units_required * self.w1_mult)
        else:
            size = math.ceil(shortfall * self.w_sub_mult)
            
        # Ensure we always send at least one notification if we have a shortfall
        return max(1, size)

    def select_batch(
        self,
        ranked_candidates: list[UDIResult],
        already_notified: set[str],
        batch_size: int,
    ) -> list[UDIResult]:
        """
        Select the next batch of donors to notify from the ranked list.

        Args:
            ranked_candidates: Ranked list of feasible donors.
            already_notified:  Set of donor IDs already contacted in this episode.
            batch_size:        Number of donors to select.

        Returns:
            List of UDIResult for the selected donors.
        """
        selected = []
        for c in ranked_candidates:
            if len(selected) >= batch_size:
                break
            if c.donor_id not in already_notified:
                selected.append(c)
        return selected
