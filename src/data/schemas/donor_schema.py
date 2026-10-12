"""
data/schemas/donor_schema.py
Pydantic v2 schema definitions for Donor profiles.
Used across all data streams (UCI pilot, synthetic, empirical).
Each stream populates only the fields relevant to it.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class BloodType(str, Enum):
    A_POS = "A+"
    A_NEG = "A-"
    B_POS = "B+"
    B_NEG = "B-"
    AB_POS = "AB+"
    AB_NEG = "AB-"
    O_POS = "O+"
    O_NEG = "O-"
    UNKNOWN = "UNKNOWN"


class DonorStream(str, Enum):
    """
    Identifies which data stream a donor record belongs to.
    Prevents cross-stream contamination in analysis.
    """
    UCI_PILOT = "UCI_PILOT"                  # Real UCI RFMT benchmark data
    KAGGLE_PILOT = "KAGGLE_PILOT"            # Kaggle availability dataset
    SYNTHETIC_SIMULATION = "SYNTHETIC_SIMULATION"  # In-silico synthetic data
    EMPIRICAL = "EMPIRICAL"                  # Real-world field recruitment data


class UCIFeatures(BaseModel):
    """
    Features from the UCI Blood Transfusion Service Center dataset.
    All four RFMT attributes.
    """
    recency: int = Field(..., ge=0, description="Months since last donation (R)")
    frequency: int = Field(..., ge=1, description="Total number of donations (F)")
    monetary: int = Field(..., ge=250, description="Total blood donated in c.c. (M = 250*F)")
    time: int = Field(..., ge=2, description="Months since first donation (T)")
    donated_march_2007: int = Field(..., ge=0, le=1, description="Target: donated in March 2007 (1=Yes, 0=No)")

    @field_validator("monetary")
    @classmethod
    def monetary_consistent_with_frequency(cls, v: int, info) -> int:
        """M must equal 250 * F in the UCI dataset."""
        if "frequency" in info.data and v != info.data["frequency"] * 250:
            raise ValueError(
                f"Monetary value {v} inconsistent with frequency {info.data['frequency']} "
                f"(expected {info.data['frequency'] * 250})."
            )
        return v

class KaggleFeatures(BaseModel):
    """
    Features from the Kaggle dataset.
    """
    city: str = Field(..., description="Donor's city")
    months_since_first_donation: int = Field(..., ge=0)
    number_of_donation: int = Field(..., ge=0)
    pints_donated: int = Field(..., ge=0)
    availability: int = Field(..., ge=0, le=1, description="Dataset historical availability (1=Yes, 0=No). NOT emergency response.")


class DonorProfile(BaseModel):
    """
    Unified donor profile schema used across spatial, ranking, and simulation modules.
    Fields are optional to accommodate both UCI pilot (no GPS) and synthetic scenarios.
    """
    donor_id: str = Field(..., description="Unique donor identifier")
    stream: DonorStream = Field(..., description="Data origin classification")

    # --- Behavioral (UCI features) ---
    uci_features: Optional[UCIFeatures] = Field(None, description="RFMT features (UCI stream only)")
    
    # --- Kaggle (Availability) ---
    kaggle_features: Optional[KaggleFeatures] = Field(None, description="Features (Kaggle stream only)")

    # --- Spatial (Synthetic / Empirical) ---
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="GPS latitude")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="GPS longitude")

    # --- Blood Type (Synthetic / Empirical) ---
    blood_type: BloodType = Field(BloodType.UNKNOWN)

    # --- Recruitment State Tracking (Simulation) ---
    is_eligible: bool = Field(True, description="Eligibility flag (8-week deferral, health status)")
    recent_alert_count: int = Field(0, ge=0, description="Number of alerts received in recent cycle")
    donation_behavior_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="UCI model donation-behavior proxy score (historical)"
    )
    response_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Probability for recruitment response (requires valid recruitment data)"
    )
    synthetic_response_propensity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Explicitly simulated propensity for synthetic experiments"
    )
    simulated_current_availability: Optional[int] = Field(
        None, description="Simulated current operational availability for specific scenarios"
    )

    class Config:
        use_enum_values = True
