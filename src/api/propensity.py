"""Advisory donor return-propensity for the local application.

Wraps the portable XGBoost model (src/models/donor_propensity.py). The value is
shown to staff and donors next to rankings; it is deliberately NOT fed into the
UDI score, because it predicts voluntary return donation (UCI label), not the
response to an emergency invitation.
"""
import os
from datetime import datetime

from src.models.donor_propensity import DEFAULT_MODEL_PATH, DonorPropensityModel, rfmt_from_history
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PropensityService:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, path=None):
        path = path or os.getenv("DONOR_PROPENSITY_MODEL") or DEFAULT_MODEL_PATH
        try:
            return cls(DonorPropensityModel.load(path))
        except FileNotFoundError:
            logger.warning("Donor propensity model not found at %s; predictions disabled.", path)
        except (ValueError, KeyError) as error:
            logger.warning("Donor propensity model at %s is invalid (%s); predictions disabled.", path, error)
        return cls(None)

    def estimate(self, donor, donation_state, now):
        """Return the model estimate for one donor, or why none is available."""
        if self.model is None:
            return {"available": False, "reason": "MODEL_NOT_LOADED"}
        history = donor.get("donation_history") or {}
        recorded = [record.completed_at for record in donation_state.completed_donations] if donation_state else []
        parse = lambda key: datetime.fromisoformat(history[key]) if history.get(key) else None
        features = rfmt_from_history(recorded, now, history.get("prior_donations", 0),
                                     parse("first_donation_at"), parse("last_donation_at"))
        if features is None:
            return {"available": False, "reason": "NO_DONATION_HISTORY"}
        sources = ([history.get("source", "self_reported")] if history else []) + (["recorded_in_app"] if recorded else [])
        return {"available": True, **self.model.estimate(features), "history_sources": sources}

    def probability(self, donor, donation_state, now):
        result = self.estimate(donor, donation_state, now)
        return result["probability"] if result["available"] else None
