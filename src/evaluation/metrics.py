"""
src/evaluation/metrics.py
Defines advanced evaluation metrics for simulation episodes.

Includes:
  - Alert Fatigue Index (AFI)
  - Mean Time to Fulfillment
  - Shortfall Ratio
"""

from __future__ import annotations

import numpy as np
from src.data.schemas.dispatch_schema import EpisodeResult


def calculate_alert_fatigue_index(episode: EpisodeResult) -> float:
    """
    Alert Fatigue Index (AFI) = Total Notifications / Required Units
    
    A lower value is better (closer to 1.0 means exactly enough donors were notified).
    If required units is 0, returns 0.
    """
    if episode.units_required == 0:
        return 0.0
    return episode.total_notifications_sent / episode.units_required


def calculate_shortfall_ratio(episode: EpisodeResult) -> float:
    """
    Shortfall Ratio = Shortfall / Required Units
    
    0.0 means perfectly fulfilled. 1.0 means complete failure.
    """
    if episode.units_required == 0:
        return 0.0
    return episode.shortfall / episode.units_required

def aggregate_episode_metrics(episodes: list[EpisodeResult]) -> dict[str, float]:
    """Aggregate metrics across multiple episodes."""
    if not episodes:
        return {}
        
    afis = [calculate_alert_fatigue_index(ep) for ep in episodes]
    shortfalls = [calculate_shortfall_ratio(ep) for ep in episodes]
    times = [ep.time_to_fulfillment_minutes for ep in episodes if ep.quota_met and ep.time_to_fulfillment_minutes is not None]
    
    return {
        "mean_afi": float(np.mean(afis)),
        "mean_shortfall_ratio": float(np.mean(shortfalls)),
        "quota_met_rate": float(np.mean([ep.quota_met for ep in episodes])),
        "mean_time_to_fulfillment": float(np.mean(times)) if times else float('nan'),
        "mean_waves_used": float(np.mean([ep.waves_used for ep in episodes]))
    }
