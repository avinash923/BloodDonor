"""
src/models/donor_propensity.py
Donor return-propensity inference (XGBoost trained on UCI RFMT data).

The trained model is shipped as a *portable* JSON artefact: the boosted trees
exported with ``Booster.get_dump(dump_format="json")`` plus a Platt-calibration
pair (a, b) and provenance metadata. This module evaluates that artefact in
pure Python, so the standard-library API server needs neither xgboost nor
scikit-learn at runtime and never unpickles anything.

What the probability means
--------------------------
The UCI Blood Transfusion label is "donated blood in March 2007" for donors
of the Hsin-Chu City (Taiwan) Blood Transfusion Service Center, given their
Recency / Frequency / Time history as of that month. The model therefore
estimates the probability that a donor *with a prior donation history*
voluntarily donates in the following month. It is NOT an emergency
invitation-response probability and must not be substituted for one without
validation on real invitation-response data (see udi_engine.DonorCandidate).

DATA STREAM: UCI_PILOT (real-world behavioural data)
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Average Gregorian month, used to convert timestamps to UCI-style months.
DAYS_PER_MONTH = 30.4375

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "donor_propensity_xgb.json"

RAW_FEATURES: tuple[str, ...] = ("recency_months", "frequency", "tenure_months")
ENGINEERED_FEATURES: tuple[str, ...] = RAW_FEATURES + (
    "donation_rate",     # F / T            — donations per month of tenure
    "recency_ratio",     # R / T            — share of tenure spent lapsed
    "mean_gap_months",   # (T - R) / max(F - 1, 1) — average inter-donation gap
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {"raw": RAW_FEATURES, "engineered": ENGINEERED_FEATURES}


def engineer_row(recency: float, frequency: float, tenure: float, feature_set: str = "engineered") -> list[float]:
    """Build one model input row from Recency / Frequency / Time (months, count, months)."""
    r, f, t = float(recency), float(frequency), float(tenure)
    if feature_set == "raw":
        return [r, f, t]
    if feature_set != "engineered":
        raise ValueError(f"Unknown feature set: {feature_set}")
    t_safe = max(t, 1.0)
    return [r, f, t, f / t_safe, r / t_safe, max(t - r, 0.0) / max(f - 1.0, 1.0)]


def _f32(value: float) -> float:
    """Round to IEEE float32, the precision XGBoost uses for features and split thresholds."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# =================================================================
# Portable tree ensemble
# =================================================================

class PortableTreeEnsemble:
    """Evaluate an XGBoost JSON tree dump without importing xgboost.

    XGBoost routes a sample to the ``yes`` child when ``x < split_condition``
    (both in float32) and to the ``missing`` child when the feature is NaN.
    The raw margin is ``base_margin + sum(leaf values)``.
    """

    def __init__(self, trees: Sequence[dict], feature_names: Sequence[str], base_margin: float) -> None:
        self.feature_index = {name: i for i, name in enumerate(feature_names)}
        self.base_margin = float(base_margin)
        self._trees = [self._compile(tree) for tree in trees]

    def _compile(self, tree: dict) -> dict:
        nodes: dict[int, tuple] = {}
        stack = [tree]
        while stack:
            node = stack.pop()
            if "leaf" in node:
                nodes[node["nodeid"]] = ("leaf", float(node["leaf"]))
                continue
            split = node["split"]
            index = self.feature_index[split] if split in self.feature_index else int(split.lstrip("f"))
            nodes[node["nodeid"]] = ("split", index, _f32(float(node["split_condition"])),
                                     node["yes"], node["no"], node.get("missing", node["yes"]))
            stack.extend(node["children"])
        return nodes

    @property
    def n_trees(self) -> int:
        return len(self._trees)

    def margin(self, row: Sequence[float]) -> float:
        x = [_f32(v) if v == v else v for v in row]  # keep NaN as NaN
        total = 0.0
        for nodes in self._trees:
            node = nodes[0]
            while node[0] == "split":
                _, index, threshold, yes, no, missing = node
                value = x[index]
                node = nodes[missing if value != value else (yes if value < threshold else no)]
            total += node[1]
        return self.base_margin + total


# =================================================================
# Donor history → RFMT features
# =================================================================

@dataclass
class RFMTFeatures:
    """UCI-style donor history summary at a reference time."""
    recency_months: float
    frequency: int
    tenure_months: float
    out_of_training_range: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"recency_months": round(self.recency_months, 2), "frequency": self.frequency,
                "tenure_months": round(self.tenure_months, 2),
                "out_of_training_range": list(self.out_of_training_range)}


def rfmt_from_history(
    donation_times: Iterable[datetime],
    now: datetime,
    prior_donations: int = 0,
    prior_first_at: Optional[datetime] = None,
    prior_last_at: Optional[datetime] = None,
) -> Optional[RFMTFeatures]:
    """Summarise a donor's history as Recency / Frequency / Time in months.

    ``donation_times`` are donations recorded by this system; ``prior_*``
    describe history reported before registration. Returns None when the
    donor has no donation at all — the UCI population contains only donors
    with at least one previous donation, so the model does not apply.
    """
    recorded = list(donation_times)
    times = sorted(recorded)
    if prior_donations > 0:
        if prior_last_at is None:
            raise ValueError("prior_last_at is required when prior_donations > 0")
        times = sorted(times + [prior_first_at or prior_last_at, prior_last_at])
    frequency = int(prior_donations) + len(recorded)
    if frequency <= 0 or not times:
        return None
    months = lambda delta: max(delta.total_seconds() / 86400.0 / DAYS_PER_MONTH, 0.0)
    return RFMTFeatures(recency_months=months(now - times[-1]), frequency=frequency,
                        tenure_months=months(now - times[0]))


# =================================================================
# Loaded model
# =================================================================

class DonorPropensityModel:
    """Calibrated XGBoost donor return-propensity model loaded from JSON."""

    def __init__(self, artefact: dict) -> None:
        if artefact.get("format") != "portable-xgboost-trees/v1":
            raise ValueError("Unsupported donor propensity model format")
        self.artefact = artefact
        self.feature_set: str = artefact["feature_set"]
        self.feature_names: list[str] = list(artefact["feature_names"])
        if tuple(self.feature_names) != FEATURE_SETS[self.feature_set]:
            raise ValueError("Model feature names do not match the declared feature set")
        self.ensemble = PortableTreeEnsemble(artefact["trees"], self.feature_names, artefact["base_margin"])
        calibration = artefact.get("calibration") or {"a": 1.0, "b": 0.0}
        self.calib_a = float(calibration["a"])
        self.calib_b = float(calibration["b"])
        self.training_ranges: dict[str, list[float]] = artefact["training_ranges"]
        self.base_rate = float(artefact["training_data"]["positive_rate"])

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "DonorPropensityModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def version(self) -> str:
        return self.artefact.get("model_version", "unknown")

    def clip_to_training_range(self, features: RFMTFeatures) -> RFMTFeatures:
        """Clamp R/F/T to the observed UCI range and record which were clamped."""
        values = {"recency_months": features.recency_months, "frequency": features.frequency,
                  "tenure_months": features.tenure_months}
        flagged = []
        for name, value in values.items():
            low, high = self.training_ranges[name]
            if not low <= value <= high:
                flagged.append(name)
                values[name] = min(max(value, low), high)
        # Tenure can never be shorter than recency.
        values["tenure_months"] = max(values["tenure_months"], values["recency_months"])
        return RFMTFeatures(values["recency_months"], int(values["frequency"]), values["tenure_months"], flagged)

    def predict_proba(self, recency_months: float, frequency: float, tenure_months: float) -> float:
        row = engineer_row(recency_months, frequency, tenure_months, self.feature_set)
        return _sigmoid(self.calib_a * self.ensemble.margin(row) + self.calib_b)

    def predict_many(self, rows: Iterable[Sequence[float]]) -> list[float]:
        return [self.predict_proba(*row) for row in rows]

    def estimate(self, features: RFMTFeatures) -> dict:
        """Probability plus the inputs and caveats an operator needs to read it."""
        clipped = self.clip_to_training_range(features)
        probability = self.predict_proba(clipped.recency_months, clipped.frequency, clipped.tenure_months)
        return {
            "probability": round(probability, 4),
            "relative_to_base_rate": round(probability / self.base_rate, 2) if self.base_rate else None,
            "base_rate": round(self.base_rate, 4),
            "features": {**features.as_dict(), "out_of_training_range": clipped.out_of_training_range},
            "model_version": self.version,
            "meaning": "Estimated probability of a voluntary donation in the next month, "
                       "from donation history (UCI Blood Transfusion RFMT model). "
                       "Not an emergency-response probability.",
        }
