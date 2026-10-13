"""Road-time ranked, persisted invitation waves for the local application.

Invitations are in-app records. This module does not send external push/SMS.
All transitions run under SystemState.lock, including donor reservations.
"""
from datetime import datetime, timedelta, timezone
import math
from uuid import uuid4

from src.ranking.udi_engine import DonorCandidate
from src.api.donor_accounts import location_ready


class RecruitmentManager:
    batch_size = 10

    def __init__(self, state, response_window_seconds=60):
        if response_window_seconds <= 0:
            raise ValueError("Response window must be positive")
        self.state = state
        self.response_window_seconds = response_window_seconds
        self._snapshots = {}

    def invitations(self, request_id):
        return [i for i in self.state.invitations if i["request_id"] == request_id]

    def reserved_elsewhere(self, donor_id, request_id):
        return any(i["donor_id"] == donor_id and i["request_id"] != request_id
                   and i["status"] == "ACCEPTED" for i in self.state.invitations)

    def deadline(self, request):
        return datetime.fromisoformat(request["created_at"]) + timedelta(minutes=request["deadline_minutes"])

    def rank(self, request, now, donor_ids=None):
        candidates, coords = [], {}
        for did, donor in self.state.users.items():
            if donor_ids is not None and did not in donor_ids:
                continue
            if (donor.get("role") != "donor" or not donor.get("availability")
                    or donor.get("blood_type") != request["blood_type_required"]
                    or self.reserved_elsewhere(did, request["request_id"])):
                continue
            if not location_ready(donor, now):
                continue
            # Cheap geographic prefilter before any paid road-routing call (50 km).
            lat1, lat2 = math.radians(donor["latitude"]), math.radians(request["latitude"])
            dlat = lat2 - lat1
            dlon = math.radians(request["longitude"] - donor["longitude"])
            hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            if 6371 * 2 * math.asin(min(1, math.sqrt(hav))) > 50:
                continue
            eligible, _ = self.state.donation_tracker.check_temporal_eligibility(did, now)
            if not eligible:
                continue
            coords[did] = (donor["latitude"], donor["longitude"])
            candidates.append(DonorCandidate(donor_id=did, synthetic_response_propensity=0.85,
                                             recent_alert_count=donor.get("recent_alerts", 0)))
        populated = self.state.routing_service.populate_candidates_with_etas(
            candidates, coords, hospital_lat=request["latitude"], hospital_lon=request["longitude"],
            hospital_id=request["hospital_id"], departure_time=now)
        ranked = self.state.udi_engine.score_and_rank(populated, urgency=request["urgency"])
        budget = max(0, (self.deadline(request) - now).total_seconds() / 60) * 0.85
        results = []
        for result in ranked:
            eta = result.travel_time_minutes
            if not result.is_feasible or not result.is_eligible or eta is None or not math.isfinite(eta) or eta < 0 or eta > budget:
                continue
            donor = self.state.users[result.donor_id]
            propensity = getattr(self.state, "propensity", None)
            results.append({"donor_id": result.donor_id, "donor_name": donor["name"],
                            # Advisory only: voluntary-return model output, not part of UDI or ordering.
                            "donation_propensity": propensity.probability(
                                donor, self.state.donation_tracker.donor_states.get(result.donor_id), now)
                                if propensity else None,
                            "blood_type": donor["blood_type"], "udi_score": round(result.udi_score, 4),
                            "travel_time_minutes": round(eta, 1), "is_feasible": True,
                            "is_eligible": True, "recent_alerts": donor.get("recent_alerts", 0),
                            "route_checked_at": now.isoformat()})
        results.sort(key=lambda item: (item["travel_time_minutes"], -item["udi_score"], item["donor_id"]))
        for index, item in enumerate(results, 1):
            item["rank"] = index
        return results

    def start(self, request, now=None):
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            request["recruitment"] = {"batch_number": 0, "batch_size": self.batch_size,
                "response_window_seconds": self.response_window_seconds,
                "next_batch_at": None, "status": "READY", "pool": self.rank(request, now)}
            self._advance(request, now)
            self.state.save()

    def _advance(self, request, now):
        self._snapshots.pop(request["request_id"], None)
        recruitment = request["recruitment"]
        invitations = self.invitations(request["request_id"])
        for invitation in invitations:
            if invitation["status"] == "PENDING" and invitation.get("expires_at") and datetime.fromisoformat(invitation["expires_at"]) <= now:
                invitation["status"] = "EXPIRED"
        accepted = sum(i["status"] == "ACCEPTED" for i in invitations)
        needed = max(0, request["units_required"] - request["units_fulfilled"] - accepted)
        if request["status"] != "ACTIVE" or self.deadline(request) <= now:
            if request["status"] == "ACTIVE":
                request["status"] = "EXPIRED"
            for invitation in invitations:
                if invitation["status"] in ("PENDING", "ACCEPTED"):
                    invitation["status"] = "CANCELLED"
            recruitment.update(status="CLOSED", next_batch_at=None)
            return
        if needed == 0:
            for invitation in invitations:
                if invitation["status"] == "PENDING":
                    invitation["status"] = "CANCELLED"
            recruitment.update(status="TARGET_RESERVED", next_batch_at=None)
            return
        if any(i["status"] == "PENDING" for i in invitations):
            recruitment["status"] = "AWAITING_RESPONSES"
            return
        contacted = {i["donor_id"] for i in invitations}
        remaining_ids = {did for did, donor in self.state.users.items() if donor.get("role") == "donor"} - contacted
        remaining = ([r for r in recruitment["pool"] if r["donor_id"] in remaining_ids]
                     if recruitment["batch_number"] == 0 and recruitment["pool"] else
                     self.rank(request, now, remaining_ids)) if remaining_ids else []
        if not remaining:
            recruitment.update(status="EXHAUSTED", next_batch_at=None)
            return
        recruitment["batch_number"] += 1
        expiry = min(now + timedelta(seconds=recruitment["response_window_seconds"]), self.deadline(request))
        recruitment.update(status="AWAITING_RESPONSES", next_batch_at=expiry.isoformat())
        for candidate in remaining[:self.batch_size]:
            did = candidate["donor_id"]
            self.state.invitations.append({"invitation_id": "INV_" + uuid4().hex,
                "request_id": request["request_id"], "donor_id": did,
                "blood_type_required": request["blood_type_required"], "urgency": request["urgency"],
                "hospital_name": request["hospital_name"], "hospital_lat": request["latitude"],
                "hospital_lon": request["longitude"], "status": "PENDING", "sent_at": now.isoformat(),
                "expires_at": expiry.isoformat(), "batch_number": recruitment["batch_number"],
                "travel_time_minutes": candidate["travel_time_minutes"]})
            self.state.users[did]["recent_alerts"] += 1
            self.state.notifications.queue(self.state.invitations[-1])

    def tick(self, now=None):
        """Called by the server even when no browser or donor is polling."""
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            changed = False
            for request in self.state.emergencies.values():
                recruitment = request.get("recruitment")
                if not recruitment or request["status"] != "ACTIVE":
                    continue
                due = recruitment.get("next_batch_at")
                if self.deadline(request) <= now or (due and datetime.fromisoformat(due) <= now):
                    self._advance(request, now)
                    changed = True
            if changed:
                self.state.save()

    def respond(self, invitation, action, now=None):
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            self.tick(now)
            if invitation["status"] != "PENDING":
                raise ValueError("Invitation already answered, expired or cancelled")
            request = self.state.emergencies[invitation["request_id"]]
            if request["status"] != "ACTIVE" or self.deadline(request) <= now:
                raise ValueError("Request closed")
            if action == "ACCEPT":
                accepted = sum(i["status"] == "ACCEPTED" for i in self.invitations(request["request_id"]))
                if request["units_fulfilled"] + accepted >= request["units_required"]:
                    raise ValueError("Enough donors have already accepted")
                routes = self.rank(request, now, {invitation["donor_id"]})
                if not routes:
                    raise ValueError("Donor unavailable, reserved elsewhere, ineligible or unable to arrive in time")
                invitation["travel_time_minutes"] = routes[0]["travel_time_minutes"]
                invitation["route_checked_at"] = now.isoformat()
            invitation["status"] = "ACCEPTED" if action == "ACCEPT" else "DECLINED"
            invitation["responded_at"] = now.isoformat()
            if request.get("recruitment"):
                self._advance(request, now)
            self._snapshots.clear()
            self.state.save()

    def snapshot(self, request, now=None):
        now = now or datetime.now(timezone.utc)
        with self.state.lock:
            self.tick(now)
            invitations = self.invitations(request["request_id"])
            by_donor = {i["donor_id"]: i for i in invitations}
            # Refresh actual road times for the list shown to staff.
            cached = self._snapshots.get(request["request_id"])
            signature = tuple((did, donor.get("availability"), donor.get("latitude"), donor.get("longitude"), location_ready(donor, now),
                               donor.get("recent_alerts"), self.state.donation_tracker.donor_states.get(did).next_eligible_at
                               if did in self.state.donation_tracker.donor_states else None)
                              for did, donor in self.state.users.items())
            if request["status"] != "ACTIVE":
                results = []
            elif cached and cached[2] == signature and 0 <= (now - cached[0]).total_seconds() < 15:
                results = [dict(item) for item in cached[1]]
            else:
                results = self.rank(request, now)
                self._snapshots[request["request_id"]] = (now, [dict(item) for item in results], signature)
            for candidate in results:
                invitation = by_donor.get(candidate["donor_id"], {})
                candidate["invitation_status"] = invitation.get("status", "QUEUED")
                candidate["batch_number"] = invitation.get("batch_number", 0)
            accepted = [r for r in results if r["invitation_status"] == "ACCEPTED"]
            # A newly failed route must not silently hide a donor's reservation.
            missing = [i for i in invitations if i["status"] == "ACCEPTED" and i["donor_id"] not in {r["donor_id"] for r in accepted}]
            for invitation in missing:
                donor = self.state.users[invitation["donor_id"]]
                accepted.append({"donor_id": donor["user_id"], "donor_name": donor["name"],
                    "blood_type": donor["blood_type"], "travel_time_minutes": None,
                    "is_feasible": False, "is_eligible": True, "udi_score": 0,
                    "recent_alerts": donor["recent_alerts"], "invitation_status": "ACCEPTED",
                    "batch_number": invitation.get("batch_number", 0), "rank": len(accepted) + 1})
            for index, candidate in enumerate(accepted, 1):
                candidate = dict(candidate)
                candidate["rank"] = index
                accepted[index - 1] = candidate
            recruitment = request.get("recruitment", {})
            summary = {key: value for key, value in recruitment.items() if key != "pool"}
            summary.update(accepted_count=len(accepted), pending_count=sum(i["status"] == "PENDING" for i in invitations),
                contacted_count=len(invitations), remaining_needed=max(0, request["units_required"] - request["units_fulfilled"] - len(accepted)),
                units_fulfilled=request["units_fulfilled"], units_required=request["units_required"], request_status=request["status"])
            summary.update(delivery_channel="in_app", routing_provider=self.state.routing_service.provider_type.value)
            return results, accepted, summary
