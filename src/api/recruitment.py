"""Broadcast donor recruitment for the local application.

When a doctor raises a request, every donor who
  * has a compatible blood group (exact match, or all RBC-compatible groups
    when the doctor allows it),
  * is available, temporally eligible and not committed to another request,
  * has a consented location inside the search radius, and
  * can reach the hospital by road within the doctor's time limit
receives an in-app invitation at once. Donors who register, share a location
or become eligible later are invited by a periodic sweep while the request is
still open.

Invited donors are ranked for the doctor by a match score:
    match = P(donate) x (1 - road ETA / time remaining)
where P(donate) is the XGBoost donor-propensity estimate from the donor's
donation history (recency, frequency, tenure). Donors without any history get
the model's training base rate and are flagged as such.

Invitations are in-app records (plus the optional webhook outbox). All
transitions run under SystemState.lock, including the reservation of units.
"""
from datetime import datetime, timedelta, timezone
import math
from uuid import uuid4

from src.ranking.udi_engine import DonorCandidate, UDIEngine
from src.api.donor_accounts import donor_blood_types, last_donation, location_fresh, location_ready

# Upper bound on average road speed used to skip donors who cannot possibly
# arrive in time before any routing call is made.
MAX_PLAUSIBLE_SPEED_KMH = 100.0
DEFAULT_PRIOR_PROBABILITY = 0.25
CONTACT_STATUSES = ("ACCEPTED", "COMPLETED")
STATUS_ORDER = {"ACCEPTED": 0, "COMPLETED": 1, "PENDING": 2, "DECLINED": 3, "WITHDRAWN": 4,
                "CANCELLED": 5, "EXPIRED": 6}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    hav = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(hav)))


def deadline(request):
    return datetime.fromisoformat(request["created_at"]) + timedelta(minutes=request["deadline_minutes"])


class RecruitmentManager:
    sweep_interval_seconds = 30

    def __init__(self, state, sweep_interval_seconds=None):
        self.state = state
        if sweep_interval_seconds is not None:
            if sweep_interval_seconds <= 0:
                raise ValueError("Sweep interval must be positive")
            self.sweep_interval_seconds = sweep_interval_seconds

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def invitations(self, request_id):
        return [i for i in self.state.invitations if i["request_id"] == request_id]

    def reserved_elsewhere(self, donor_id, request_id):
        return any(i["donor_id"] == donor_id and i["request_id"] != request_id
                   and i["status"] == "ACCEPTED" for i in self.state.invitations)

    def deadline(self, request):
        return deadline(request)

    def _prior(self):
        model = getattr(getattr(self.state, "propensity", None), "model", None)
        return float(model.base_rate) if model is not None else DEFAULT_PRIOR_PROBABILITY

    def evaluate(self, request, now, donor_ids=None):
        """Score matching donors for a request. Returns rows sorted by match score.

        Rows include donors who cannot arrive in time (``can_arrive_in_time``
        False) so accepted donors running late stay visible; only feasible
        rows are ever invited.
        """
        remaining = (self.deadline(request) - now).total_seconds() / 60.0
        if remaining <= 0:
            return []
        groups = donor_blood_types(request)
        radius = request.get("search_radius_km")
        bound = radius if radius else remaining / 60.0 * MAX_PLAUSIBLE_SPEED_KMH
        tracker = self.state.donation_tracker
        selected, straight = [], {}
        for did, donor in self.state.users.items():
            if donor_ids is not None and did not in donor_ids:
                continue
            if (donor.get("role") != "donor" or not donor.get("availability")
                    or donor.get("blood_type") not in groups or not location_ready(donor, now)
                    or self.reserved_elsewhere(did, request["request_id"])):
                continue
            if not tracker.check_temporal_eligibility(did, now)[0]:
                continue
            distance = haversine_km(donor["latitude"], donor["longitude"], request["latitude"], request["longitude"])
            if distance > bound:
                continue
            straight[did] = distance
            selected.append((did, donor["latitude"], donor["longitude"]))
        if not selected:
            return []

        routes = self.state.routing_service.compute_donor_etas(
            selected, hospital_lat=request["latitude"], hospital_lon=request["longitude"],
            hospital_id=request["hospital_id"], departure_time=now)
        engine = UDIEngine(formula_mode="multiplicative", lambda_fatigue=0.0, safety_margin=0.0,
                           urgency_scores={"TIME_LIMIT": 1.0},
                           urgency_horizons={"TIME_LIMIT": float(request["deadline_minutes"])})
        propensity = getattr(self.state, "propensity", None)
        prior = self._prior()
        rows = []
        for did, _, _ in selected:
            donor = self.state.users[did]
            route = routes.get(did)
            eta = route.duration_minutes if route is not None and route.is_success else None
            donation_state = tracker.donor_states.get(did)
            probability = propensity.probability(donor, donation_state, now) if propensity else None
            result = engine.score_candidate(
                DonorCandidate(donor_id=did, travel_time_minutes=eta if eta is None or eta >= 0 else None,
                               recruitment_response_probability=probability if probability is not None else prior),
                "TIME_LIMIT", t_remaining=remaining)
            history = last_donation(donor, donation_state)
            captured = donor.get("location_captured_at")
            rows.append({
                "donor_id": did, "donor_name": donor["name"], "blood_type": donor["blood_type"],
                "distance_km": round(straight[did], 2),
                "road_distance_km": round(route.distance_meters / 1000.0, 2)
                    if route is not None and route.distance_meters is not None else None,
                "travel_time_minutes": round(eta, 1) if eta is not None else None,
                "route_provider": route.provider if route is not None else None,
                "can_arrive_in_time": result.is_feasible,
                "donation_likelihood": round(probability, 4) if probability is not None else None,
                "likelihood_basis": "donation_history" if probability is not None else "no_history_prior",
                "match_score": round(result.udi_score, 4),
                "total_donations": history["total_donations"],
                "last_donation_at": history["last_donation_at"],
                "location_source": donor.get("location_source", "demo"),
                "location_updated_at": captured,
                "location_live": location_fresh(donor, now),
            })
        rows.sort(key=lambda r: (not r["can_arrive_in_time"], -r["match_score"],
                                 r["travel_time_minutes"] if r["travel_time_minutes"] is not None else math.inf,
                                 r["donor_id"]))
        for index, row in enumerate(rows, 1):
            row["rank"] = index
        return rows

    # Backwards-compatible name used by older callers and tests.
    def rank(self, request, now, donor_ids=None):
        return [row for row in self.evaluate(request, now, donor_ids) if row["can_arrive_in_time"]]

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def _needed(self, request, invitations=None):
        invitations = invitations if invitations is not None else self.invitations(request["request_id"])
        accepted = sum(i["status"] == "ACCEPTED" for i in invitations)
        return max(0, request["units_required"] - request["units_fulfilled"] - accepted)

    def _invite(self, request, row, now):
        invitation = {
            "invitation_id": "INV_" + uuid4().hex, "request_id": request["request_id"],
            "donor_id": row["donor_id"], "blood_type_required": request["blood_type_required"],
            "donor_blood_type": row["blood_type"], "hospital_name": request["hospital_name"],
            "hospital_address": request.get("hospital_address", ""),
            "hospital_lat": request["latitude"], "hospital_lon": request["longitude"],
            "status": "PENDING", "sent_at": now.isoformat(), "expires_at": self.deadline(request).isoformat(),
            "travel_time_minutes": row["travel_time_minutes"], "distance_km": row["distance_km"],
            "road_distance_km": row["road_distance_km"], "route_provider": row["route_provider"],
            "match_score": row["match_score"], "donation_likelihood": row["donation_likelihood"],
        }
        self.state.invitations.append(invitation)
        self.state.users[row["donor_id"]]["recent_alerts"] = self.state.users[row["donor_id"]].get("recent_alerts", 0) + 1
        self.state.notifications.queue(invitation)
        return invitation

    def start(self, request, now=None):
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            request["recruitment"] = {"mode": "BROADCAST", "status": "BROADCASTING",
                                      "started_at": now.isoformat(), "last_sweep_at": None,
                                      "invited_count": 0}
            self._sweep_request(request, now)
            self._settle(request, now)
            self.state.save()

    def _sweep_request(self, request, now, donor_ids=None):
        recruitment = request.setdefault("recruitment", {"mode": "BROADCAST", "invited_count": 0})
        recruitment["last_sweep_at"] = now.isoformat()
        if request["status"] != "ACTIVE" or self.deadline(request) <= now or self._needed(request) == 0:
            return 0
        contacted = {i["donor_id"] for i in self.invitations(request["request_id"])}
        candidates = {did for did, user in self.state.users.items() if user.get("role") == "donor"} - contacted
        if donor_ids is not None:
            candidates &= set(donor_ids)
        if not candidates:
            return 0
        invited = 0
        for row in self.evaluate(request, now, candidates):
            if row["can_arrive_in_time"]:
                self._invite(request, row, now)
                invited += 1
        recruitment["invited_count"] = recruitment.get("invited_count", 0) + invited
        return invited

    def sweep(self, now=None, donor_ids=None):
        """Invite newly matching donors (new registrations, fresh locations) to open requests."""
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            invited = 0
            for request in self.state.emergencies.values():
                if request["status"] == "ACTIVE" and request.get("recruitment"):
                    invited += self._sweep_request(request, now, donor_ids)
                    self._settle(request, now)
            if invited:
                self.state.save()
            return invited

    def release_donor(self, donor_id, reason, now=None):
        """Close a donor's unanswered invitations (they switched availability off or revoked location)."""
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            for invitation in self.state.invitations:
                if invitation["donor_id"] == donor_id and invitation["status"] == "PENDING":
                    invitation.update(status="CANCELLED", closed_reason=reason, responded_at=now.isoformat())
            for request in self.state.emergencies.values():
                if request["status"] == "ACTIVE" and request.get("recruitment"):
                    self._settle(request, now)

    def _settle(self, request, now):
        """Apply deadline, fill and re-open rules after any change."""
        recruitment = request.setdefault("recruitment", {"mode": "BROADCAST", "invited_count": 0})
        invitations = self.invitations(request["request_id"])
        if request["status"] == "ACTIVE" and self.deadline(request) <= now:
            request["status"] = "EXPIRED"
        if request["status"] != "ACTIVE":
            for invitation in invitations:
                if invitation["status"] in ("PENDING", "ACCEPTED"):
                    invitation["status"] = "EXPIRED" if request["status"] == "EXPIRED" else "CANCELLED"
                    invitation["closed_reason"] = "REQUEST_" + request["status"]
            recruitment.update(status="CLOSED")
            return
        needed = self._needed(request, invitations)
        if needed == 0:
            for invitation in invitations:
                if invitation["status"] == "PENDING":
                    invitation.update(status="CANCELLED", closed_reason="TARGET_REACHED")
            recruitment.update(status="TARGET_REACHED")
            return
        # A withdrawal re-opens invitations that were closed only because the target was reached.
        for invitation in invitations:
            if invitation["status"] == "CANCELLED" and invitation.get("closed_reason") == "TARGET_REACHED":
                invitation["status"] = "PENDING"
                invitation.pop("closed_reason", None)
        pending = any(i["status"] == "PENDING" for i in invitations)
        recruitment.update(status="BROADCASTING" if pending else "WAITING_FOR_DONORS")

    def tick(self, now=None):
        """Called by the server loop even when no browser is polling."""
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            changed = False
            for request in self.state.emergencies.values():
                if request["status"] != "ACTIVE" or not request.get("recruitment"):
                    continue
                if self.deadline(request) <= now:
                    self._settle(request, now)
                    changed = True
                    continue
                last = request["recruitment"].get("last_sweep_at")
                if not last or (now - datetime.fromisoformat(last)).total_seconds() >= self.sweep_interval_seconds:
                    if self._sweep_request(request, now):
                        changed = True
                    self._settle(request, now)
            if changed:
                self.state.save()

    def respond(self, invitation, action, now=None):
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            self.tick(now)
            request = self.state.emergencies[invitation["request_id"]]
            if action == "WITHDRAW":
                if invitation["status"] != "ACCEPTED":
                    raise ValueError("Only an accepted invitation can be withdrawn")
                invitation.update(status="WITHDRAWN", responded_at=now.isoformat())
                self._settle(request, now)
                self.state.save()
                return
            if invitation["status"] != "PENDING":
                raise ValueError("Invitation already answered, expired or closed because enough donors accepted")
            if request["status"] != "ACTIVE" or self.deadline(request) <= now:
                raise ValueError("Request closed")
            if action == "ACCEPT":
                if self._needed(request) == 0:
                    raise ValueError("Enough donors have already accepted")
                if self.reserved_elsewhere(invitation["donor_id"], request["request_id"]):
                    raise ValueError("You have already accepted another request")
                rows = self.evaluate(request, now, {invitation["donor_id"]})
                if not rows:
                    raise ValueError("Donor unavailable, ineligible or without a shared location")
                row = rows[0]
                if not row["can_arrive_in_time"]:
                    raise ValueError("From your current location you cannot reach the hospital within the time limit")
                invitation.update(travel_time_minutes=row["travel_time_minutes"], distance_km=row["distance_km"],
                                  road_distance_km=row["road_distance_km"], route_provider=row["route_provider"],
                                  route_checked_at=now.isoformat(), accepted_at=now.isoformat(),
                                  expected_arrival_at=(now + timedelta(minutes=row["travel_time_minutes"])).isoformat())
            invitation["status"] = "ACCEPTED" if action == "ACCEPT" else "DECLINED"
            invitation["responded_at"] = now.isoformat()
            self._settle(request, now)
            self.state.save()

    # -----------------------------------------------------------------
    # Doctor view
    # -----------------------------------------------------------------

    def snapshot(self, request, now=None):
        """Invited donors (ranked), accepted donors with contact details, and a summary."""
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            self.tick(now)
            invitations = self.invitations(request["request_id"])
            latest = {}
            for invitation in invitations:
                latest[invitation["donor_id"]] = invitation
            committed = [i for i in latest.values() if i["status"] in CONTACT_STATUSES]
            # Live road ETAs for donors who are still coming or may still answer.
            refreshed = {row["donor_id"]: row for row in self.evaluate(
                request, now, {i["donor_id"] for i in latest.values() if i["status"] in ("ACCEPTED", "PENDING")})} \
                if request["status"] == "ACTIVE" else {}
            tracker = self.state.donation_tracker

            donors = []
            for invitation in latest.values():
                donor = self.state.users.get(invitation["donor_id"], {})
                live = refreshed.get(invitation["donor_id"]) or {}
                value = lambda key: live[key] if key in live else invitation.get(key)
                donors.append({
                    "invitation_id": invitation["invitation_id"], "donor_id": invitation["donor_id"],
                    "donor_name": donor.get("name", invitation["donor_id"]),
                    "blood_type": donor.get("blood_type", invitation.get("donor_blood_type")),
                    "invitation_status": invitation["status"], "closed_reason": invitation.get("closed_reason"),
                    "travel_time_minutes": value("travel_time_minutes"),
                    "distance_km": value("distance_km"),
                    "road_distance_km": value("road_distance_km"),
                    "route_provider": value("route_provider"),
                    "can_arrive_in_time": live.get("can_arrive_in_time"),
                    "match_score": value("match_score"),
                    "donation_likelihood": value("donation_likelihood"),
                    "total_donations": live.get("total_donations"),
                    "last_donation_at": live.get("last_donation_at"),
                    "sent_at": invitation.get("sent_at"), "responded_at": invitation.get("responded_at"),
                })
            donors.sort(key=lambda d: (STATUS_ORDER.get(d["invitation_status"], 9), -(d["match_score"] or 0), d["donor_id"]))
            for index, row in enumerate(donors, 1):
                row["rank"] = index

            accepted = []
            for invitation in committed:
                did = invitation["donor_id"]
                donor = self.state.users.get(did, {})
                fresh = refreshed.get(did)
                history = last_donation(donor, tracker.donor_states.get(did))
                eta = fresh["travel_time_minutes"] if fresh else invitation.get("travel_time_minutes")
                accepted.append({
                    "invitation_id": invitation["invitation_id"], "donor_id": did,
                    "donor_name": donor.get("name", did), "blood_type": donor.get("blood_type"),
                    "phone": donor.get("phone") or None, "email": donor.get("email") or None,
                    "invitation_status": invitation["status"],
                    "accepted_at": invitation.get("accepted_at") or invitation.get("responded_at"),
                    "completed_at": invitation.get("completed_at"),
                    "travel_time_minutes": eta,
                    "is_feasible": bool(fresh and fresh["can_arrive_in_time"]) if invitation["status"] == "ACCEPTED" else True,
                    "road_distance_km": fresh["road_distance_km"] if fresh else invitation.get("road_distance_km"),
                    "distance_km": fresh["distance_km"] if fresh else invitation.get("distance_km"),
                    "route_provider": fresh["route_provider"] if fresh else invitation.get("route_provider"),
                    "expected_arrival_at": (now + timedelta(minutes=eta)).isoformat()
                        if fresh and eta is not None else invitation.get("expected_arrival_at"),
                    "last_donation_at": history["last_donation_at"],
                    "total_donations": history["total_donations"],
                    "donation_likelihood": invitation.get("donation_likelihood"),
                    "latitude": donor.get("latitude"), "longitude": donor.get("longitude"),
                    "location_updated_at": donor.get("location_captured_at"),
                    "location_source": donor.get("location_source", "demo"),
                    "location_live": location_fresh(donor, now),
                    "udi_score": invitation.get("match_score") or 0,
                })
            accepted.sort(key=lambda a: (a["invitation_status"] != "ACCEPTED",
                                         a["travel_time_minutes"] if a["travel_time_minutes"] is not None else math.inf,
                                         a["donor_id"]))
            for index, row in enumerate(accepted, 1):
                row["rank"] = index

            counts = {status: sum(1 for i in latest.values() if i["status"] == status) for status in STATUS_ORDER}
            recruitment = request.get("recruitment", {})
            end = self.deadline(request)
            summary = {
                "mode": recruitment.get("mode", "BROADCAST"), "status": recruitment.get("status"),
                "last_sweep_at": recruitment.get("last_sweep_at"),
                "invited_count": len(latest), "contacted_count": len(latest),
                "pending_count": counts["PENDING"], "accepted_count": counts["ACCEPTED"],
                "declined_count": counts["DECLINED"], "completed_count": counts["COMPLETED"],
                "withdrawn_count": counts["WITHDRAWN"],
                "remaining_needed": self._needed(request, invitations),
                "units_fulfilled": request["units_fulfilled"], "units_required": request["units_required"],
                "request_status": request["status"], "deadline_at": end.isoformat(),
                "seconds_remaining": max(0, int((end - now).total_seconds())),
                "time_limit_minutes": request["deadline_minutes"],
                "matching_blood_types": list(donor_blood_types(request)),
                "search_radius_km": request.get("search_radius_km"),
                "delivery_channel": "in_app",
                "routing_provider": self.state.routing_service.provider_type.value,
            }
            return donors, accepted, summary
