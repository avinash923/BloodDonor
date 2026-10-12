"""
src/simulation/simulator.py
Discrete-event simulation runner & state tracker.

Coordinates the emergency scenario, ranks candidates, dispatches waves,
and steps through time to process responses and arrivals.

DATA STREAM: SYNTHETIC_SIMULATION
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field
import uuid
import heapq

from src.data.schemas.dispatch_schema import DispatchEvent, EpisodeResult, DonorState
from src.data.schemas.request_schema import EmergencyRequest
from src.ranking.udi_engine import UDIResult, DonorCandidate
from src.simulation.response_model import StochasticResponseModel
from src.simulation.dispatch_cascade import DispatchCascadePolicy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(order=True)
class SimEvent:
    """Discrete event in the simulation."""
    time: float
    event_type: str = field(compare=False)
    donor_id: Optional[str] = field(default=None, compare=False)


class SimulationEpisode:
    """
    Executes a single emergency recruitment episode.
    """

    def __init__(
        self,
        request: EmergencyRequest,
        candidates: dict[str, float], # donor_id -> true probability (for sim only)
        ranked_pool: list[UDIResult],
        t_critical_minutes: float,
        response_model: StochasticResponseModel,
        dispatch_policy: DispatchCascadePolicy,
        wave_timeout: float = 15.0,
        max_waves: int = 10,
        t_observation_horizon: float = 1440.0,
        udi_engine: Optional[UDIEngine] = None,
        unranked_candidates: Optional[list[DonorCandidate]] = None,
    ) -> None:
        self.request = request
        self.candidates_prob = candidates
        self.ranked_pool = ranked_pool
        self.t_critical = t_critical_minutes
        
        self.response_model = response_model
        self.dispatch_policy = dispatch_policy
        self.wave_timeout = wave_timeout
        self.max_waves = max_waves
        
        self.udi_engine = udi_engine
        self.unranked_candidates = unranked_candidates

        self.time: float = 0.0
        self.wave_number: int = 1
        self.units_fulfilled: int = 0
        self.units_fulfilled_on_time: int = 0
        self.late_arrivals: int = 0
        self.t_observation_horizon = t_observation_horizon
        
        self.events: list[SimEvent] = []  # Priority queue
        self.dispatch_log: list[DispatchEvent] = []
        self.notified_donors: set[str] = set()
        
        # State tracking
        self.donor_states: dict[str, DonorState] = {}
        for r in ranked_pool:
            self.donor_states[r.donor_id] = DonorState.UNCONTACTED

    def push_event(self, event: SimEvent) -> None:
        heapq.heappush(self.events, event)

    def pop_event(self) -> Optional[SimEvent]:
        if self.events:
            return heapq.heappop(self.events)
        return None

    def run(self) -> EpisodeResult:
        """Run the simulation until quota met or time expires."""
        
        # Initial wave
        self.push_event(SimEvent(time=0.0, event_type="WAVE_DISPATCH"))

        while self.events and self.time <= self.t_observation_horizon:
            event = self.pop_event()
            
            # Fast forward time
            if event.time > self.t_observation_horizon:
                break
            self.time = event.time

            if event.event_type == "WAVE_DISPATCH":
                self._handle_wave_dispatch()
            elif event.event_type == "DONOR_RESPONSE":
                self._handle_donor_response(event.donor_id)
            elif event.event_type == "DONOR_ARRIVAL":
                self._handle_donor_arrival(event.donor_id)

            # Check termination
            if self.units_fulfilled >= self.request.units_required:
                break

        # Finalize episode
        result = EpisodeResult(
            episode_id=f"EP_{uuid.uuid4().hex[:8]}",
            request_id=self.request.request_id,
            units_required=self.request.units_required,
            units_fulfilled=self.units_fulfilled,
            units_fulfilled_on_time=self.units_fulfilled_on_time,
            late_arrivals=self.late_arrivals,
            total_notifications_sent=len(self.notified_donors),
            waves_used=min(self.wave_number, self.max_waves),
            dispatch_log=self.dispatch_log,
        )
        if self.units_fulfilled >= self.request.units_required:
            result.time_to_fulfillment_minutes = self.time
        if self.units_fulfilled_on_time >= self.request.units_required:
            # We don't have the exact exact on-time minute tracked easily if it was already reached,
            # but if this event was the one that fulfilled it on time, self.time is it.
            # A more robust way is scanning the dispatch_log, but we'll approximate as self.time
            # since the loop breaks exactly when units_fulfilled >= units_required.
            # Wait, units_fulfilled includes late arrivals. The loop breaks when units_fulfilled == req.
            # If units_fulfilled_on_time == req, it means they all arrived <= t_critical.
            result.time_to_on_time_fulfillment = self.time
            
        result.compute_derived_metrics()
        return result

    def _handle_wave_dispatch(self) -> None:
        if self.wave_number > self.max_waves:
            return

        shortfall = self.request.units_required - self.units_fulfilled
        batch_size = self.dispatch_policy.get_batch_size(
            self.wave_number, self.request.units_required, shortfall
        )

        if self.udi_engine and self.unranked_candidates:
            t_rem = self.t_critical - self.time
            # Re-rank based on remaining time
            new_ranked = self.udi_engine.score_and_rank(
                self.unranked_candidates,
                urgency=self.request.urgency,
                t_remaining=t_rem
            )
            self.ranked_pool = [r for r in new_ranked if r.is_feasible and r.is_eligible]
            logger.debug("Time %5.1f | Re-ranked candidates with t_remaining=%.1f", self.time, t_rem)

        selected = self.dispatch_policy.select_batch(
            self.ranked_pool, self.notified_donors, batch_size
        )

        if not selected:
            return # No more feasible donors

        logger.debug(
            "Time %5.1f | Wave %d | Shortfall %d | Dispatching %d alerts",
            self.time, self.wave_number, shortfall, len(selected)
        )

        for r in selected:
            self.notified_donors.add(r.donor_id)
            self.donor_states[r.donor_id] = DonorState.NOTIFIED
            
            # Increment alert fatigue for this donor
            if self.unranked_candidates:
                for cand in self.unranked_candidates:
                    if cand.donor_id == r.donor_id:
                        cand.recent_alert_count += 1
                        break

            # Record dispatch event
            disp_event = DispatchEvent(
                event_id=f"EV_{uuid.uuid4().hex[:8]}",
                episode_id="PENDING", # Set later
                wave_number=self.wave_number,
                donor_id=r.donor_id,
                dispatched_at_minutes=self.time,
                final_state=DonorState.NOTIFIED
            )
            self.dispatch_log.append(disp_event)

            # Sample response
            true_prob = self.candidates_prob[r.donor_id]
            try:
                outcome = self.response_model.sample_response(probability=true_prob, donor_id=r.donor_id)
            except TypeError:
                outcome = self.response_model.sample_response(probability=true_prob)
            
            # Schedule response event
            resp_time = self.time + outcome.latency_minutes
            if resp_time <= self.t_observation_horizon:
                self.push_event(SimEvent(time=resp_time, event_type="DONOR_RESPONSE", donor_id=r.donor_id))
            
            # Store outcome temporarily in the dispatch event to process later
            disp_event.accepted = outcome.accepted
            disp_event.response_latency_minutes = outcome.latency_minutes

        # Schedule next wave if needed
        next_wave_time = self.time + self.wave_timeout
        if next_wave_time <= self.t_critical:
            self.push_event(SimEvent(time=next_wave_time, event_type="WAVE_DISPATCH"))
            
        self.wave_number += 1

    def _handle_donor_response(self, donor_id: str) -> None:
        # Find the dispatch event
        disp_event = next(e for e in self.dispatch_log if e.donor_id == donor_id)
        
        if disp_event.accepted:
            self.donor_states[donor_id] = DonorState.ACCEPTED
            disp_event.final_state = DonorState.ACCEPTED
            
            # Find the travel time — first from ranked_pool (may have been re-ranked),
            # then from unranked_candidates (always has the raw spatial travel time).
            tt: Optional[float] = None
            ranked_match = next((r for r in self.ranked_pool if r.donor_id == donor_id), None)
            if ranked_match is not None and ranked_match.travel_time_minutes is not None:
                tt = ranked_match.travel_time_minutes
            elif self.unranked_candidates:
                cand_match = next((c for c in self.unranked_candidates if c.donor_id == donor_id), None)
                if cand_match is not None:
                    tt = cand_match.travel_time_minutes
            
            if tt is None:
                # Cannot schedule arrival without travel time; mark as expired
                logger.warning(
                    "Donor %s accepted but travel time is unavailable; marking EXPIRED.", donor_id
                )
                self.donor_states[donor_id] = DonorState.EXPIRED
                disp_event.final_state = DonorState.EXPIRED
                return
            
            # Schedule arrival
            arrival_time = self.time + tt
            if arrival_time <= self.t_observation_horizon:
                disp_event.arrival_time_minutes = arrival_time
                self.push_event(SimEvent(time=arrival_time, event_type="DONOR_ARRIVAL", donor_id=donor_id))
            else:
                self.donor_states[donor_id] = DonorState.EXPIRED
                disp_event.final_state = DonorState.EXPIRED
        else:
            self.donor_states[donor_id] = DonorState.DECLINED
            disp_event.final_state = DonorState.DECLINED
            
    def _handle_donor_arrival(self, donor_id: str) -> None:
        self.donor_states[donor_id] = DonorState.ARRIVED
        disp_event = next(e for e in self.dispatch_log if e.donor_id == donor_id)
        disp_event.final_state = DonorState.ARRIVED
        
        self.units_fulfilled += 1
        if self.time <= self.t_critical:
            self.units_fulfilled_on_time += 1
        else:
            self.late_arrivals += 1
            
        logger.debug("Time %5.1f | Donor %s arrived. Fulfilled: %d (On Time: %d) / %d", 
                     self.time, donor_id, self.units_fulfilled, self.units_fulfilled_on_time, self.request.units_required)
