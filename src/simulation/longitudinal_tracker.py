"""
src/simulation/longitudinal_tracker.py
Longitudinal notification history and rolling window fatigue state tracker.

Tracks discrete dispatch notifications dispatched to donors across independent
chronological emergencies over extended horizons (e.g., 30 days).

DATA STREAM: SYNTHETIC_SIMULATION
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class NotificationRecord:
    """Immutable record of an individual donor notification event."""
    donor_id: str
    timestamp: datetime
    emergency_id: str
    wave_number: int
    accepted: Optional[bool] = None


class LongitudinalDonorTracker:
    """
    Manages persistent donor state and notification history across sequential emergencies.
    
    Args:
        lookback_days: Length of the rolling time window in days for counting recent alerts.
    """

    def __init__(self, lookback_days: float = 14.0) -> None:
        self.lookback_days = float(lookback_days)
        # donor_id -> list of NotificationRecord in chronological order
        self.history: dict[str, list[NotificationRecord]] = {}

    def record_notification(
        self,
        donor_id: str,
        timestamp: datetime,
        emergency_id: str,
        wave_number: int,
        accepted: Optional[bool] = None,
    ) -> NotificationRecord:
        """
        Record a newly dispatched notification event.
        Must be called exactly once per dispatched alert.
        """
        rec = NotificationRecord(
            donor_id=donor_id,
            timestamp=timestamp,
            emergency_id=emergency_id,
            wave_number=wave_number,
            accepted=accepted,
        )
        if donor_id not in self.history:
            self.history[donor_id] = []
        self.history[donor_id].append(rec)
        return rec

    def get_recent_alert_count(
        self,
        donor_id: str,
        current_time: datetime,
        lookback_days: Optional[float] = None,
    ) -> int:
        """
        Count notifications dispatched to this donor within (current_time - lookback, current_time].
        
        Boundary semantics:
          A notification at timestamp t is counted if:
            current_time - lookback <= t <= current_time
        """
        records = self.history.get(donor_id)
        if not records:
            return 0

        w_days = float(lookback_days) if lookback_days is not None else self.lookback_days
        cutoff = current_time - timedelta(days=w_days)

        count = 0
        for rec in reversed(records):
            if rec.timestamp < cutoff:
                break # Since records are chronological, earlier records are also < cutoff
            if rec.timestamp <= current_time:
                count += 1
        return count

    def get_cumulative_alert_count(self, donor_id: str) -> int:
        """Total lifetime notifications dispatched to this donor in this timeline."""
        return len(self.history.get(donor_id, []))

    def get_all_records(self) -> list[NotificationRecord]:
        """Return all notification records across all donors in chronological order."""
        all_recs = []
        for rec_list in self.history.values():
            all_recs.extend(rec_list)
        all_recs.sort(key=lambda r: (r.timestamp, r.emergency_id, r.wave_number, r.donor_id))
        return all_recs

    def compute_donor_summary(self, pool_donor_ids: list[str]) -> list[dict]:
        """
        Compute aggregate notification statistics per donor across the population.
        """
        summaries = []
        for did in pool_donor_ids:
            recs = self.history.get(did, [])
            n_tot = len(recs)
            n_acc = sum(1 for r in recs if r.accepted is True)
            summaries.append({
                "donor_id": did,
                "total_alerts": n_tot,
                "total_acceptances": n_acc,
                "notified_ever": n_tot > 0,
                "multiple_alerts": n_tot >= 2,
                "heavy_alerts": n_tot >= 5,
            })
        return summaries

    def compute_lookback_repeat_stats(
        self,
        pool_donor_ids: list[str],
        lookback_days: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Repeated-notification statistics using the rolling lookback window.

        A dispatched notification is a lookback-repeat if the same donor already
        had at least one earlier notification in
        [t - lookback_days, t), where t is the current notification timestamp.

        Boundary semantics match get_recent_alert_count except that the current
        notification itself is excluded (left-closed, right-open prior interval).

        Returns:
            n_lookback_repeat_notifications: count of repeat dispatches
            lookback_repeat_notification_rate: repeats / total notifications
                (0 if no notifications were dispatched)
            pct_donors_with_lookback_repeat: 100 * donors with ≥1 lookback-repeat
                / |pool| (denominator is the full donor pool, including never-contacted)
        """
        w_days = float(lookback_days) if lookback_days is not None else self.lookback_days
        n_repeat = 0
        n_total = 0
        n_donors_repeat = 0

        for did in pool_donor_ids:
            recs = self.history.get(did, [])
            n_total += len(recs)
            donor_had_repeat = False
            for i, rec in enumerate(recs):
                cutoff = rec.timestamp - timedelta(days=w_days)
                prior_in_window = 0
                for prev in recs[:i]:
                    if cutoff <= prev.timestamp < rec.timestamp:
                        prior_in_window += 1
                if prior_in_window >= 1:
                    n_repeat += 1
                    donor_had_repeat = True
            if donor_had_repeat:
                n_donors_repeat += 1

        n_pool = len(pool_donor_ids)
        return {
            "n_lookback_repeat_notifications": float(n_repeat),
            "lookback_repeat_notification_rate": (
                float(n_repeat) / float(n_total) if n_total > 0 else 0.0
            ),
            "pct_donors_with_lookback_repeat": (
                100.0 * n_donors_repeat / n_pool if n_pool > 0 else 0.0
            ),
        }


def calculate_gini(values: np.ndarray | list[float]) -> float:
    """
    Compute Gini coefficient of inequality for notification distribution.
    0.0 = perfect equality (all donors receive equal alerts).
    1.0 = maximal inequality (all alerts concentrated on one donor).
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))
