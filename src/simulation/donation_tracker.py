"""
src/simulation/donation_tracker.py
Longitudinal Donation-Aware Temporal Eligibility State & Lifecycle Tracker.

Phase 6.3: Implements explicit donor event transitions:
    NOTIFIED -> RESPONDED / DECLINED -> ARRIVED -> SCREENED -> DONATION_COMPLETED

And enforces temporal whole-blood donation deferral gates:
    - Male: 90 days (Drugs and Cosmetics Rules, 1945 / NBTC guidelines)
    - Female: 120 days (NBTC guidelines)
    - Default / Unspecified: 90 days (configurable)

DATA STREAM: SYNTHETIC_SIMULATION (In-silico research only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Sequence
import numpy as np


class DonationLifecycleState(str, Enum):
    """
    Explicit finite state machine states for donor event tracking in recruitment episodes.
    """
    NOTIFIED = "NOTIFIED"
    RESPONDED = "RESPONDED"
    DECLINED = "DECLINED"
    NO_RESPONSE = "NO_RESPONSE"
    ARRIVED = "ARRIVED"
    DID_NOT_ARRIVE = "DID_NOT_ARRIVE"
    SCREENING_PASSED = "SCREENING_PASSED"
    SCREENING_DEFERRED = "SCREENING_DEFERRED"
    DONATION_COMPLETED = "DONATION_COMPLETED"


class DonationType(str, Enum):
    """Whole-blood or apheresis donation type."""
    WHOLE_BLOOD = "WHOLE_BLOOD"
    APHERESIS_PLATELETS = "APHERESIS_PLATELETS"
    APHERESIS_PLASMA = "APHERESIS_PLASMA"


@dataclass(frozen=True)
class CompletedDonationRecord:
    """Immutable record of an actual completed donation event."""
    donor_id: str
    completed_at: datetime
    emergency_id: str
    donation_type: DonationType = DonationType.WHOLE_BLOOD
    deferral_days: float = 90.0
    next_eligible_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class DonationLifecycleEvent:
    """Immutable log of any state transition in the donation lifecycle."""
    event_id: str
    donor_id: str
    emergency_id: str
    timestamp: datetime
    state: DonationLifecycleState
    details: Optional[str] = None


@dataclass
class DonorTemporalEligibilityState:
    """Persistent temporal eligibility state for a single synthetic donor."""
    donor_id: str
    policy_interval_days: float  # e.g. 90.0 for male, 120.0 for female
    gender_group: str = "UNSPECIFIED"  # "MALE", "FEMALE", "UNSPECIFIED" (synthetic assignment only)
    completed_donations: list[CompletedDonationRecord] = field(default_factory=list)
    next_eligible_at: Optional[datetime] = None
    last_donation_at: Optional[datetime] = None

    def is_eligible_at(self, query_time: datetime) -> tuple[bool, Optional[str]]:
        """
        Check if the donor is eligible for whole blood donation at query_time.
        
        Boundary semantics:
            If query_time < next_eligible_at: INELIGIBLE (within deferral interval)
            If query_time >= next_eligible_at: ELIGIBLE (interval has expired)
        """
        if self.next_eligible_at is None:
            return True, None
        if query_time < self.next_eligible_at:
            days_left = (self.next_eligible_at - query_time).total_seconds() / 86400.0
            return False, f"DONATION_DEFERRAL: {days_left:.1f} days remaining until {self.next_eligible_at.isoformat()}"
        return True, None


class DonationAwareEligibilityTracker:
    """
    Manages longitudinal donation records, screening outcomes, and temporal eligibility
    hard gates across repeated emergencies within a simulation timeline.
    """

    def __init__(
        self,
        default_deferral_days_male: float = 90.0,
        default_deferral_days_female: float = 120.0,
        default_deferral_days_unspecified: float = 90.0,
        screening_pass_rate: float = 0.95,
        donation_completion_rate: float = 0.98,
        seed: int = 42,
    ) -> None:
        self.deferral_days_male = float(default_deferral_days_male)
        self.deferral_days_female = float(default_deferral_days_female)
        self.deferral_days_unspecified = float(default_deferral_days_unspecified)
        self.screening_pass_rate = float(screening_pass_rate)
        self.donation_completion_rate = float(donation_completion_rate)
        self.rng = np.random.default_rng(seed)

        # donor_id -> DonorTemporalEligibilityState
        self.donor_states: dict[str, DonorTemporalEligibilityState] = {}
        # Complete audit trail of lifecycle events
        self.events_log: list[DonationLifecycleEvent] = []

        # Audit counters
        self.attempted_selections_prevented: int = 0
        self.donors_deferred_count: int = 0

    def initialize_donor_population(
        self,
        donor_ids: Sequence[str],
        synthetic_female_ratio: float = 0.30,
        custom_intervals: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Initialize synthetic policy intervals for the donor population.
        
        Note: Kaggle dataset does NOT contain donor sex or clinical history.
        The female_ratio here generates synthetic policy groups (90d vs 120d)
        strictly for in-silico simulation of heterogeneous policy intervals.
        """
        self.donor_states.clear()
        self.events_log.clear()
        self.attempted_selections_prevented = 0

        for did in donor_ids:
            if custom_intervals and did in custom_intervals:
                interval = custom_intervals[did]
                group = "CUSTOM"
            else:
                is_female = bool(self.rng.random() < synthetic_female_ratio)
                if is_female:
                    interval = self.deferral_days_female
                    group = "FEMALE"
                else:
                    interval = self.deferral_days_male
                    group = "MALE"

            self.donor_states[did] = DonorTemporalEligibilityState(
                donor_id=did,
                policy_interval_days=interval,
                gender_group=group,
                completed_donations=[],
                next_eligible_at=None,
                last_donation_at=None,
            )

    def check_temporal_eligibility(
        self, donor_id: str, query_time: datetime
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a donor is currently eligible under the donation deferral policy.
        If donor is unknown, defaults to True (unconstrained).
        """
        state = self.donor_states.get(donor_id)
        if state is None:
            return True, None
        return state.is_eligible_at(query_time)

    def record_prevented_selection(self) -> None:
        """Increment count of candidate exclusions caused by the temporal eligibility gate."""
        self.attempted_selections_prevented += 1

    def log_lifecycle_event(
        self,
        event_id: str,
        donor_id: str,
        emergency_id: str,
        timestamp: datetime,
        state: DonationLifecycleState,
        details: Optional[str] = None,
    ) -> DonationLifecycleEvent:
        """Append an event to the lifecycle audit trail."""
        ev = DonationLifecycleEvent(
            event_id=event_id,
            donor_id=donor_id,
            emergency_id=emergency_id,
            timestamp=timestamp,
            state=state,
            details=details,
        )
        self.events_log.append(ev)
        return ev

    def process_arrival_for_donation(
        self,
        donor_id: str,
        arrival_time: datetime,
        emergency_id: str,
        still_needed: bool,
        screening_rand: Optional[float] = None,
        completion_rand: Optional[float] = None,
    ) -> tuple[DonationLifecycleState, Optional[CompletedDonationRecord]]:
        """
        Process a donor's physical arrival at the blood center / hospital.
        
        Lifecycle logic:
            1. If still_needed is False (quota already filled before this donor arrived),
               no donation occurs (donor arrived, but not needed).
            2. Simulated clinical screening: donor passes with probability screening_pass_rate.
               If fails -> SCREENING_DEFERRED (no donation).
            3. Simulated donation procedure: donor completes with probability donation_completion_rate.
               If passes -> DONATION_COMPLETED.
               
        Only DONATION_COMPLETED updates the donor's next_eligible_at and triggers deferral!
        """
        # Log arrival
        self.log_lifecycle_event(
            event_id=f"ARR_{emergency_id}_{donor_id}",
            donor_id=donor_id,
            emergency_id=emergency_id,
            timestamp=arrival_time,
            state=DonationLifecycleState.ARRIVED,
            details="Donor physically arrived at site",
        )

        if not still_needed:
            self.log_lifecycle_event(
                event_id=f"NOT_NEEDED_{emergency_id}_{donor_id}",
                donor_id=donor_id,
                emergency_id=emergency_id,
                timestamp=arrival_time,
                state=DonationLifecycleState.ARRIVED,
                details="Emergency quota already met; donor unit not needed",
            )
            return DonationLifecycleState.ARRIVED, None

        # Simulated Medical Screening
        s_rand = screening_rand if screening_rand is not None else float(self.rng.random())
        if s_rand >= self.screening_pass_rate:
            self.log_lifecycle_event(
                event_id=f"SCR_DEF_{emergency_id}_{donor_id}",
                donor_id=donor_id,
                emergency_id=emergency_id,
                timestamp=arrival_time,
                state=DonationLifecycleState.SCREENING_DEFERRED,
                details=f"Donor deferred at clinical screening (draw {s_rand:.3f} >= {self.screening_pass_rate})",
            )
            return DonationLifecycleState.SCREENING_DEFERRED, None

        self.log_lifecycle_event(
            event_id=f"SCR_PASS_{emergency_id}_{donor_id}",
            donor_id=donor_id,
            emergency_id=emergency_id,
            timestamp=arrival_time,
            state=DonationLifecycleState.SCREENING_PASSED,
            details="Donor passed medical screening",
        )

        # Simulated Donation Completion
        c_rand = completion_rand if completion_rand is not None else float(self.rng.random())
        if c_rand >= self.donation_completion_rate:
            self.log_lifecycle_event(
                event_id=f"DON_INCOMP_{emergency_id}_{donor_id}",
                donor_id=donor_id,
                emergency_id=emergency_id,
                timestamp=arrival_time,
                state=DonationLifecycleState.SCREENING_PASSED,
                details="Phlebotomy aborted / incomplete (no full unit collected)",
            )
            return DonationLifecycleState.SCREENING_PASSED, None

        # Donation Completed!
        state = self.donor_states.get(donor_id)
        if state is None:
            # Fallback for untracked donor
            interval_days = self.deferral_days_unspecified
            state = DonorTemporalEligibilityState(
                donor_id=donor_id,
                policy_interval_days=interval_days,
                gender_group="UNSPECIFIED",
            )
            self.donor_states[donor_id] = state
        else:
            interval_days = state.policy_interval_days

        next_eligible = arrival_time + timedelta(days=interval_days)
        rec = CompletedDonationRecord(
            donor_id=donor_id,
            completed_at=arrival_time,
            emergency_id=emergency_id,
            donation_type=DonationType.WHOLE_BLOOD,
            deferral_days=interval_days,
            next_eligible_at=next_eligible,
        )

        state.completed_donations.append(rec)
        state.last_donation_at = arrival_time
        state.next_eligible_at = next_eligible

        self.log_lifecycle_event(
            event_id=f"DON_COMP_{emergency_id}_{donor_id}",
            donor_id=donor_id,
            emergency_id=emergency_id,
            timestamp=arrival_time,
            state=DonationLifecycleState.DONATION_COMPLETED,
            details=f"Whole-blood unit collected. Next eligible: {next_eligible.isoformat()} ({interval_days}d deferral)",
        )

        return DonationLifecycleState.DONATION_COMPLETED, rec

    def get_summary_metrics(self, pool_donor_ids: Sequence[str]) -> dict:
        """Compute aggregate summary metrics across the donor pool for a timeline."""
        total_donations = 0
        donors_with_donations = 0
        donors_currently_deferred_at_end = 0
        donations_per_donor = []

        all_records = []
        for did in pool_donor_ids:
            state = self.donor_states.get(did)
            n_don = len(state.completed_donations) if state else 0
            donations_per_donor.append(n_don)
            total_donations += n_don
            if n_don > 0:
                donors_with_donations += 1
            if state and state.completed_donations:
                all_records.extend(state.completed_donations)

        # Count screening events
        screening_passes = sum(1 for e in self.events_log if e.state == DonationLifecycleState.SCREENING_PASSED)
        screening_def = sum(1 for e in self.events_log if e.state == DonationLifecycleState.SCREENING_DEFERRED)
        donations_comp = sum(1 for e in self.events_log if e.state == DonationLifecycleState.DONATION_COMPLETED)

        return {
            "total_completed_donations": int(total_donations),
            "donors_with_at_least_one_donation": int(donors_with_donations),
            "attempted_selections_prevented": int(self.attempted_selections_prevented),
            "screening_passes_count": int(screening_passes),
            "screening_deferrals_count": int(screening_def),
            "donations_completed_count": int(donations_comp),
            "all_donation_records": all_records,
        }
