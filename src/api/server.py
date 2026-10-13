"""
src/api/server.py
Lightweight Standard-Library HTTP REST Server for Emergency Blood Recommendation.

Serves the browser client in web/ and the JSON API below without extra
dependencies (standard library `http.server`).

Endpoints:
  POST /api/v1/auth/login                  -> User login (donor or hospital_staff)
  POST /api/v1/auth/register               -> Donor registration (contact, history, optional location)
  GET  /api/v1/donor/profile               -> Donor profile, eligibility, last donation
  POST /api/v1/donor/profile               -> Update contact details / availability
  POST /api/v1/donor/location              -> Share (GPS or map-picked) or revoke location
  POST /api/v1/donor/history               -> Update self-reported donation history
  GET  /api/v1/donor/invitations           -> Emergency invitations with hospital and doctor contact
  POST /api/v1/donor/invitations/respond   -> ACCEPT, DECLINE or WITHDRAW
  POST /api/v1/emergency/requests          -> Doctor raises a request (broadcast to all matching donors)
  GET  /api/v1/emergency/requests          -> List requests with live counts
  POST /api/v1/emergency/requests/cancel   -> Doctor cancels an open request
  GET  /api/v1/emergency/recommendations   -> Invited donors (ranked), accepted donors with contact
  GET  /api/v1/emergency/routes            -> Road route (OSRM) with polyline for the map
  POST /api/v1/emergency/donation/complete -> Record a completed donation
  GET  /api/v1/facilities                  -> Blood-bank directory (search, state filter, pages)

DATA STREAM: REST_API (Backend coordinator)
"""

from __future__ import annotations

import json
import os
import math
import secrets
import mimetypes
from pathlib import Path
from uuid import uuid4
from dataclasses import asdict
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import threading

from src.ranking.udi_engine import UDIEngine
from src.spatial.routing_service import RoutingService
from src.spatial.routing_provider import LocationCoordinate
from src.simulation.donation_tracker import DonationAwareEligibilityTracker, DonationLifecycleState, CompletedDonationRecord, DonationType
from src.utils.logger import get_logger
from src.api.recruitment import RecruitmentManager, haversine_km
from src.api.donor_accounts import (new_donor, password_matches, location_ready, location_fresh, update_location,
                                    revoke_location, donation_history, history_from_dates, last_donation,
                                    clean_phone, clean_email, donor_blood_types, BLOOD_TYPES)
from src.api.propensity import PropensityService
from src.api.notifications import NotificationDispatcher
from src.simulation.donation_tracker import DonorTemporalEligibilityState

logger = get_logger("blood_donor_api_server")

REGISTERED_DEFERRAL_DAYS = 120.0
MIN_TIME_LIMIT_MINUTES, MAX_TIME_LIMIT_MINUTES = 10, 2880
LEGACY_URGENCY_MINUTES = {"CRITICAL": 60, "URGENT": 120, "SCHEDULED": 240}
REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_text(value, limit):
    return value.strip()[:limit] if isinstance(value, str) else ""


def apply_reported_deferral(state: DonorTemporalEligibilityState, donor: dict) -> None:
    """A self-reported recent donation defers the donor exactly like an in-app donation."""
    history = donor.get("donation_history") or {}
    if not history.get("last_donation_at"):
        return
    reported = datetime.fromisoformat(history["last_donation_at"])
    if state.last_donation_at is None or reported > state.last_donation_at:
        state.last_donation_at = reported
        state.next_eligible_at = reported + timedelta(days=state.policy_interval_days)


class SystemState:
    """Thread-safe in-memory state shared across the REST API."""

    def __init__(self, state_path: Optional[str] = None, demo_donors: int = 0,
                 seed_demo: bool = True, sweep_interval_seconds: Optional[float] = None):
        self.lock = threading.RLock()
        self.state_path = Path(state_path) if state_path else None
        self.sessions: dict[str, str] = {}
        self.notification_outbox = {}
        self.seed_demo = seed_demo
        directory_path = REPO_ROOT / "data/processed/blood_bank_directory.json"
        self.facilities = json.loads(directory_path.read_text(encoding="utf-8"))["facilities"] if directory_path.exists() else []
        self.facility_states = sorted({f["state"] for f in self.facilities if f.get("state")})
        self.udi_engine = UDIEngine(formula_mode="multiplicative")
        self.routing_service = RoutingService()
        self.donation_tracker = DonationAwareEligibilityTracker(seed=42)
        self.propensity = PropensityService.load()
        startup = datetime.now(timezone.utc)
        demo_history = lambda count, last, first: donation_history(
            {"prior_donation_count": count, "months_since_last_donation": last,
             "months_since_first_donation": first}, startup, source="synthetic_demo")

        # The doctor account exists in every mode; demo donors only when seeding demo data.
        self.users = {
            "staff_001": {
                "user_id": "staff_001",
                "role": "hospital_staff",
                "name": "Dr. Ramesh Patel",
                "hospital_id": "HOSP_CITY_01",
                "hospital_name": "City Central Trauma & Blood Bank",
                "phone": "",
                "latitude": 12.9915,
                "longitude": 77.5852,
            },
        }
        if seed_demo:
            self.users.update({
                "donor_001": {
                    "user_id": "donor_001", "role": "donor", "name": "Sarah Chen", "blood_type": "O+",
                    "phone": "Demo account", "latitude": 12.9352, "longitude": 77.6245,
                    "availability": True, "recent_alerts": 1, "donation_history": demo_history(6, 3, 30),
                },
                "donor_002": {
                    "user_id": "donor_002", "role": "donor", "name": "Rahul Sharma", "blood_type": "A+",
                    "phone": "Demo account", "latitude": 12.9716, "longitude": 77.5946,
                    "availability": True, "recent_alerts": 0, "donation_history": demo_history(1, 14, 14),
                },
            })
            # Additional clearly labelled synthetic accounts.
            for index in range(3, demo_donors + 3):
                did = f"donor_{index:03d}"
                self.users[did] = {"user_id": did, "role": "donor", "name": f"Demo Donor {index:03d}",
                    "blood_type": "O+", "phone": "Demo account", "latitude": 12.9915 + index * 0.0007,
                    "longitude": 77.5852 + index * 0.0005, "availability": True, "recent_alerts": 0,
                    # Deterministic, explicitly synthetic histories give the likelihood column some spread.
                    "donation_history": demo_history(1 + index * 5 % 12, index * 7 % 25,
                                                     index * 7 % 25 + 4 * (index * 5 % 12))}

        self.donation_tracker.initialize_donor_population(
            [did for did, user in self.users.items() if user["role"] == "donor"],
            synthetic_female_ratio=0.5,
        )

        self.emergencies: dict[str, dict] = {}
        self.invitations: list[dict] = []
        if seed_demo:
            self.emergencies["REQ_TRAUMA_101"] = {
                "request_id": "REQ_TRAUMA_101", "hospital_id": "HOSP_CITY_01",
                "hospital_name": "City Central Trauma & Blood Bank", "hospital_address": "",
                "facility_source": "demo", "latitude": 12.9915, "longitude": 77.5852,
                "blood_type_required": "O+", "include_compatible": False, "search_radius_km": None,
                "units_required": 3, "units_fulfilled": 1, "deadline_minutes": 60,
                "created_at": startup.isoformat(), "status": "ACTIVE",
                "created_by": "staff_001", "doctor_name": "Dr. Ramesh Patel", "contact_phone": "",
                "notes": "Seeded demo request",
                "recruitment": {"mode": "BROADCAST", "status": "BROADCASTING",
                                "last_sweep_at": startup.isoformat(), "invited_count": 1},
            }
            self.invitations.append({
                "invitation_id": "INV_1001", "request_id": "REQ_TRAUMA_101", "donor_id": "donor_001",
                "blood_type_required": "O+", "donor_blood_type": "O+",
                "hospital_name": "City Central Trauma & Blood Bank", "hospital_address": "",
                "hospital_lat": 12.9915, "hospital_lon": 77.5852, "status": "PENDING",
                "sent_at": startup.isoformat(), "expires_at": (startup + timedelta(minutes=60)).isoformat(),
            })
        if self.state_path and self.state_path.exists():
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.users = saved["users"]
            self.emergencies = saved["emergencies"]
            self.invitations = saved["invitations"]
            self.notification_outbox = saved.get("notification_outbox", {})
            for did, donor in saved["donation_states"].items():
                state = self.donation_tracker.donor_states.setdefault(did, DonorTemporalEligibilityState(did, donor["policy_interval_days"]))
                state.policy_interval_days = donor["policy_interval_days"]
                state.next_eligible_at = datetime.fromisoformat(donor["next_eligible_at"]) if donor["next_eligible_at"] else None
                state.last_donation_at = datetime.fromisoformat(donor["last_donation_at"]) if donor["last_donation_at"] else None
                state.completed_donations = [CompletedDonationRecord(
                    **{**rec, "completed_at": datetime.fromisoformat(rec["completed_at"]),
                       "next_eligible_at": datetime.fromisoformat(rec["next_eligible_at"]),
                       "donation_type": DonationType(rec["donation_type"])}
                ) for rec in donor["completed_donations"]]
            # Older state files used invitation waves; they continue as broadcasts.
            for emergency in self.emergencies.values():
                if emergency.get("recruitment") is not None:
                    emergency["recruitment"].setdefault("mode", "BROADCAST")
                    emergency["recruitment"].pop("pool", None)

        self.recruitment = RecruitmentManager(self, sweep_interval_seconds)
        self.notifications = NotificationDispatcher(self)

    def save(self):
        """Atomically persist activity; sessions deliberately expire on restart."""
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        payload = {"users": self.users, "emergencies": self.emergencies,
                   "notification_outbox": self.notification_outbox,
                   "invitations": self.invitations,
                   "donation_states": {did: asdict(state) for did, state in self.donation_tracker.donor_states.items()}}
        temporary.write_text(json.dumps(payload, default=lambda value: value.isoformat()), encoding="utf-8")
        temporary.replace(self.state_path)


def matching_donor(user: dict, emergency: dict) -> bool:
    """Blood-group match as used for invitations (exact, or RBC-compatible when allowed)."""
    return (user.get("role") == "donor" and user.get("availability", False)
            and user.get("blood_type") in donor_blood_types(emergency))


def request_view(emergency: dict) -> dict:
    """Request record plus live counts for the doctor's list."""
    invitations = STATE.recruitment.invitations(emergency["request_id"])
    deadline_at = STATE.recruitment.deadline(emergency)
    view = {key: value for key, value in emergency.items() if key != "recruitment"}
    view.update(
        time_limit_minutes=emergency["deadline_minutes"], deadline_at=deadline_at.isoformat(),
        seconds_remaining=max(0, int((deadline_at - datetime.now(timezone.utc)).total_seconds())),
        invited_count=len({i["donor_id"] for i in invitations}),
        pending_count=sum(i["status"] == "PENDING" for i in invitations),
        accepted_count=sum(i["status"] == "ACCEPTED" for i in invitations),
        declined_count=sum(i["status"] == "DECLINED" for i in invitations),
        recruitment_status=(emergency.get("recruitment") or {}).get("status"),
    )
    return view


STATE = SystemState()


class BloodDonorApiHandler(BaseHTTPRequestHandler):
    """Handles REST API requests from the web client."""

    def log_message(self, format, *args):  # keep polling noise out of the console
        logger.debug("%s - %s", self.address_string(), format % args)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def _send_json(self, status_code: int, data: Any):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Optional[dict]:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length < 0 or content_length > 65536:
                return None
            if content_length == 0:
                return {}
            raw = self.rfile.read(content_length).decode("utf-8")
            body = json.loads(raw)
            return body if isinstance(body, dict) else None
        except Exception as e:
            logger.error("Failed to parse request JSON: %s", str(e))
            return None

    def _authenticate(self) -> Optional[dict]:
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            user = STATE.users.get(STATE.sessions.get(token, ""))
            if user:
                return user
        return None

    def _require_role(self, role: str) -> Optional[dict]:
        user = self._authenticate()
        if not user:
            self._send_json(401, {"error": "Login required"})
            return None
        if user["role"] != role:
            self._send_json(403, {"error": "This resource requires hospital staff" if role == "hospital_staff"
                                  else "This resource requires a donor account"})
            return None
        return user

    def _serve_asset(self, path: str):
        root = Path(self.server.web_root).resolve()
        target = (root / urllib.parse.unquote(path).lstrip("/")).resolve()
        if not target.is_relative_to(root):
            self._send_json(403, {"error": "Invalid asset path"})
            return
        if path == "/":
            target = root / "index.html"
        if not target.is_file():
            self._send_json(404, {"error": "App asset not found"})
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if getattr(self.server, "web_root", None) and not path.startswith("/api/"):
            self._serve_asset(path)
            return
        with STATE.lock:
            self._handle_get(path, query)

    def _handle_get(self, path, query):
        if path == "/api/v1/health":
            provider = STATE.routing_service.provider
            self._send_json(200, {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "routing_provider": STATE.routing_service.provider_type.value,
                "routing_available": getattr(provider, "available", True),
                "routing_base_url": getattr(provider, "base_url", None),
                "external_notifications_configured": STATE.notifications.configured,
                "facility_records": len(STATE.facilities),
                "facilities_with_coordinates": sum(1 for f in STATE.facilities if f.get("latitude") is not None),
                "registered_donors": sum(1 for u in STATE.users.values() if u.get("role") == "donor"),
                "demo_accounts_enabled": STATE.seed_demo and os.getenv("ALLOW_DEMO_LOGIN", "true").lower() == "true",
                "staff_demo_login": os.getenv("ALLOW_DEMO_LOGIN", "true").lower() == "true"
                                    and not STATE.users.get("staff_001", {}).get("password_record"),
            })
            return

        if path == "/api/v1/facilities":
            if not self._require_role("hospital_staff"):
                return
            term = query.get("search", [""])[0].strip().lower()
            state_filter = query.get("state", [""])[0]
            with_coordinates = query.get("has_coordinates", ["false"])[0].lower() == "true"
            try:
                page = int(query.get("page", ["1"])[0])
                page_size = int(query.get("page_size", ["100"])[0])
                if page < 1 or not 1 <= page_size <= 100:
                    raise ValueError()
            except ValueError:
                self._send_json(400, {"error": "Page must be positive and page_size between 1 and 100"})
                return
            facilities = [f for f in STATE.facilities
                          if (not term or term in " ".join(str(f.get(k, "")) for k in
                                                           ("name", "state", "city", "district", "address", "pincode")).lower())
                          and (not state_filter or f.get("state") == state_filter)
                          and (not with_coordinates or f.get("latitude") is not None)]
            near = query.get("near", [""])[0]
            if near:
                try:
                    lat, lon = (float(v) for v in near.split(","))
                except ValueError:
                    self._send_json(400, {"error": "near must be 'latitude,longitude'"})
                    return
                facilities = [dict(f, distance_km=round(haversine_km(lat, lon, f["latitude"], f["longitude"]), 2))
                              for f in facilities if f.get("latitude") is not None]
                facilities.sort(key=lambda f: f["distance_km"])
            start = (page - 1) * page_size
            self._send_json(200, {"facilities": facilities[start:start + page_size], "total": len(facilities),
                                  "page": page, "page_size": page_size, "has_next": start + page_size < len(facilities),
                                  "states": STATE.facility_states,
                                  "with_coordinates": sum(1 for f in facilities if f.get("latitude") is not None),
                                  "source_date": "2015-09",
                                  "notice": "Historical September 2015 directory; verify facility details and coordinates before use"})
            return

        if path == "/api/v1/donor/profile":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Unauthorized or not a donor"})
                return
            did = user["user_id"]
            now_dt = datetime.now(timezone.utc)
            is_elig, reason = STATE.donation_tracker.check_temporal_eligibility(did, now_dt)
            don_state = STATE.donation_tracker.donor_states.get(did)
            history = last_donation(user, don_state)
            self._send_json(200, {
                "user_id": did, "name": user["name"], "blood_type": user["blood_type"],
                "phone": user.get("phone", ""), "email": user.get("email", ""),
                "latitude": user["latitude"], "longitude": user["longitude"],
                "availability": user["availability"], "recent_alerts": user.get("recent_alerts", 0),
                "is_temporally_eligible": is_elig, "eligibility_reason": reason,
                "next_eligible_at": don_state.next_eligible_at.isoformat() if don_state and don_state.next_eligible_at else None,
                "completed_donations_count": len(don_state.completed_donations) if don_state else 0,
                "total_donations": history["total_donations"], "last_donation_at": history["last_donation_at"],
                "account_source": user.get("account_source", "synthetic_demo"),
                "location_ready": location_ready(user), "location_live": location_fresh(user, now_dt),
                "location_consent": user.get("location_consent", False),
                "location_source": user.get("location_source"),
                "location_captured_at": user.get("location_captured_at"),
                "location_accuracy_meters": user.get("location_accuracy_meters"),
                "donation_propensity": STATE.propensity.estimate(user, don_state, now_dt),
            })
            return

        if path == "/api/v1/donor/invitations":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Unauthorized"})
                return
            STATE.recruitment.tick()
            now_dt = datetime.now(timezone.utc)
            result = []
            for inv in STATE.invitations:
                if inv["donor_id"] != user["user_id"]:
                    continue
                emergency = STATE.emergencies.get(inv["request_id"], {})
                deadline_at = STATE.recruitment.deadline(emergency) if emergency else None
                doctor = STATE.users.get(emergency.get("created_by", ""), {})
                result.append({
                    **inv,
                    "units_required": emergency.get("units_required"),
                    "units_fulfilled": emergency.get("units_fulfilled"),
                    "request_status": emergency.get("status"),
                    "time_limit_minutes": emergency.get("deadline_minutes"),
                    "deadline_at": deadline_at.isoformat() if deadline_at else None,
                    "seconds_remaining": max(0, int((deadline_at - now_dt).total_seconds())) if deadline_at else 0,
                    "doctor_name": emergency.get("doctor_name") or doctor.get("name"),
                    "contact_phone": emergency.get("contact_phone") or doctor.get("phone") or None,
                    "notes": emergency.get("notes", ""),
                    "remaining_needed": STATE.recruitment._needed(emergency) if emergency else 0,
                })
            result.sort(key=lambda i: i["sent_at"], reverse=True)
            result.sort(key=lambda i: {"PENDING": 0, "ACCEPTED": 1}.get(i["status"], 2))
            self._send_json(200, {"invitations": result})
            return

        if path == "/api/v1/emergency/requests":
            if not self._require_role("hospital_staff"):
                return
            STATE.recruitment.tick()
            req_list = sorted((request_view(e) for e in STATE.emergencies.values()),
                              key=lambda r: r["created_at"], reverse=True)
            req_list.sort(key=lambda r: r["status"] != "ACTIVE")
            self._send_json(200, {"requests": req_list})
            return

        if path == "/api/v1/emergency/recommendations":
            if not self._require_role("hospital_staff"):
                return
            req_id = query.get("request_id", [None])[0]
            if not req_id or req_id not in STATE.emergencies:
                self._send_json(404, {"error": f"Emergency request {req_id} not found"})
                return
            try:
                page = int(query.get("page", ["1"])[0])
                page_size = int(query.get("page_size", ["10"])[0])
                if page < 1 or not 1 <= page_size <= 500:
                    raise ValueError()
            except ValueError:
                self._send_json(400, {"error": "Page must be a positive integer"})
                return
            view = query.get("view", ["all"])[0]
            if view not in ("all", "accepted"):
                self._send_json(400, {"error": "View must be all or accepted"})
                return
            emergency = STATE.emergencies[req_id]
            donors, accepted, summary = STATE.recruitment.snapshot(emergency)
            selected = accepted if view == "accepted" else donors
            start = (page - 1) * page_size
            self._send_json(200, {"request_id": req_id, "emergency_request": request_view(emergency),
                "recommendations": selected[start:start + page_size], "accepted_donors": accepted,
                "recruitment": summary, "page": page, "page_size": page_size,
                "total": len(selected), "has_next": start + page_size < len(selected)})
            return

        if path == "/api/v1/emergency/routes":
            user = self._authenticate()
            if not user:
                self._send_json(401, {"error": "Login required"})
                return
            donor_id = query.get("donor_id", [None])[0]
            req_id = query.get("request_id", [None])[0]
            if user["role"] == "donor" and (donor_id != user["user_id"] or not any(
                inv["donor_id"] == donor_id and inv["request_id"] == req_id
                for inv in STATE.invitations
            )):
                self._send_json(403, {"error": "Route access denied"})
                return
            if not donor_id or donor_id not in STATE.users or STATE.users[donor_id]["role"] != "donor":
                self._send_json(404, {"error": "Donor not found"})
                return
            if not req_id or req_id not in STATE.emergencies:
                self._send_json(404, {"error": "Emergency request not found"})
                return

            d_user = STATE.users[donor_id]
            em_req = STATE.emergencies[req_id]
            if not location_ready(d_user):
                self._send_json(409, {"error": "The donor has not shared a location yet. Share your location to see the route."})
                return

            orig = LocationCoordinate(d_user["latitude"], d_user["longitude"])
            dest = LocationCoordinate(em_req["latitude"], em_req["longitude"])
            route_est = STATE.routing_service.provider.compute_detailed_route(
                origin=orig, destination=dest, origin_id=donor_id,
                destination_id=em_req["hospital_id"], departure_time=datetime.now(timezone.utc))
            self._send_json(200, {
                "donor_id": donor_id, "donor_name": d_user["name"], "request_id": req_id,
                "hospital_name": em_req["hospital_name"], "hospital_address": em_req.get("hospital_address", ""),
                "origin": {"latitude": orig.latitude, "longitude": orig.longitude},
                "destination": {"latitude": dest.latitude, "longitude": dest.longitude},
                "straight_line_km": round(haversine_km(orig.latitude, orig.longitude, dest.latitude, dest.longitude), 2),
                "is_success": route_est.is_success,
                "duration_minutes": round(route_est.duration_minutes, 1) if route_est.duration_minutes is not None else None,
                "distance_km": round(route_est.distance_meters / 1000.0, 2) if route_est.distance_meters is not None else None,
                "polyline_encoded": route_est.polyline_encoded,
                "provider": route_est.provider,
                "error_code": route_est.error_code,
                "location_updated_at": d_user.get("location_captured_at"),
            })
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        with STATE.lock:
            self._handle_post()

    def _handle_post(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()
        if body is None:
            self._send_json(400, {"error": "Malformed JSON body"})
            return
        if any(key in body and body[key] is not None and not isinstance(body[key], str)
               for key in ("donor_id", "request_id", "invitation_id")):
            self._send_json(400, {"error": "Identifiers must be text"})
            return

        if path == "/api/v1/auth/register":
            try:
                donor = new_donor(body)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            did = donor["user_id"]
            if did in STATE.users:
                self._send_json(409, {"error": "Username already exists"})
                return
            STATE.users[did] = donor
            tracker_state = DonorTemporalEligibilityState(did, REGISTERED_DEFERRAL_DAYS)
            apply_reported_deferral(tracker_state, donor)
            STATE.donation_tracker.donor_states[did] = tracker_state
            STATE.save()
            if location_ready(donor):
                STATE.recruitment.sweep(donor_ids={did})
            token = secrets.token_urlsafe(32)
            STATE.sessions[token] = did
            self._send_json(201, {"token": token, "user_id": did, "name": donor["name"], "role": "donor"})
            return

        if path == "/api/v1/auth/login":
            username = body.get("username", "")
            if not isinstance(username, str):
                self._send_json(400, {"error": "Username must be text"})
                return
            username = username.strip()
            user = STATE.users.get(username)
            if user and user.get("password_record") and not password_matches(body.get("password"), user["password_record"]):
                user = None
            if user and not user.get("password_record") and os.getenv("ALLOW_DEMO_LOGIN", "true").lower() != "true":
                user = None
            if user:
                token = secrets.token_urlsafe(32)
                STATE.sessions[token] = username
                self._send_json(200, {"token": token, "user_id": user["user_id"],
                                      "name": user["name"], "role": user["role"]})
            else:
                self._send_json(401, {"error": "Invalid username or password"})
            return

        if path == "/api/v1/donor/profile":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Donor login required"})
                return
            try:
                updates = {}
                if "phone" in body:
                    updates["phone"] = clean_phone(body["phone"])
                if "email" in body:
                    updates["email"] = clean_email(body["email"])
                if "available" in body:
                    if not isinstance(body["available"], bool):
                        raise ValueError("available must be true or false")
                    if body["available"] and not location_ready(user):
                        raise ValueError("Share your location before marking yourself available")
                    updates["availability"] = body["available"]
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            user.update(updates)
            if updates.get("availability") is False:
                STATE.recruitment.release_donor(user["user_id"], "DONOR_UNAVAILABLE")
            STATE.save()
            if user.get("availability"):
                STATE.recruitment.sweep(donor_ids={user["user_id"]})
            self._send_json(200, {"phone": user.get("phone", ""), "email": user.get("email", ""),
                                  "availability": user["availability"]})
            return

        if path == "/api/v1/donor/history":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Donor login required"})
                return
            try:
                history = donation_history(history_from_dates(body))
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            user["donation_history"] = history
            tracker_state = STATE.donation_tracker.donor_states.setdefault(
                user["user_id"], DonorTemporalEligibilityState(user["user_id"], REGISTERED_DEFERRAL_DAYS))
            apply_reported_deferral(tracker_state, user)
            STATE.save()
            estimate = STATE.propensity.estimate(user, tracker_state, datetime.now(timezone.utc))
            self._send_json(200, {"donation_history": history, "donation_propensity": estimate,
                                  **last_donation(user, tracker_state)})
            return

        if path == "/api/v1/donor/location":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Donor login required"})
                return
            if user.get("account_source") != "registered":
                self._send_json(409, {"error": "Register a donor account to share real device locations"})
                return
            try:
                if body.get("consent") is False:
                    revoke_location(user)
                    STATE.recruitment.release_donor(user["user_id"], "LOCATION_REVOKED")
                else:
                    update_location(user, body)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            STATE.save()
            if location_ready(user):
                STATE.recruitment.sweep(donor_ids={user["user_id"]})
            self._send_json(200, {"location_ready": location_ready(user), "availability": user["availability"],
                                  "location_live": location_fresh(user)})
            return

        if path == "/api/v1/donor/invitations/respond":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Unauthorized"})
                return
            inv_id = body.get("invitation_id")
            action = body.get("action")
            if action not in ("ACCEPT", "DECLINE", "WITHDRAW"):
                self._send_json(400, {"error": "Action must be 'ACCEPT', 'DECLINE' or 'WITHDRAW'"})
                return
            target_inv = next((inv for inv in STATE.invitations
                               if inv["invitation_id"] == inv_id and inv["donor_id"] == user["user_id"]), None)
            if not target_inv:
                self._send_json(404, {"error": "Invitation not found"})
                return
            # Acceptance can carry a fresh device fix so the ETA is checked from where the donor is now.
            if action == "ACCEPT" and isinstance(body.get("location"), dict) and user.get("account_source") == "registered":
                try:
                    update_location(user, {**body["location"], "consent": True})
                except ValueError as error:
                    logger.info("Ignoring location sent with acceptance: %s", error)
            try:
                STATE.recruitment.respond(target_inv, action)
            except ValueError as error:
                self._send_json(409, {"error": str(error)})
                return
            past = {"ACCEPT": "accepted", "DECLINE": "declined", "WITHDRAW": "withdrawn"}[action]
            self._send_json(200, {"message": f"Invitation {past}", "invitation": target_inv})
            return

        if path == "/api/v1/emergency/requests":
            user = self._authenticate()
            if not user or user.get("role") != "hospital_staff":
                self._send_json(401, {"error": "Unauthorized: hospital staff only"})
                return
            blood_type = body.get("blood_type_required", "O+")
            units = body.get("units_required", 1)
            time_limit = body.get("time_limit_minutes")
            if time_limit is None and "urgency" in body:
                time_limit = LEGACY_URGENCY_MINUTES.get(body["urgency"])
                if time_limit is None:
                    self._send_json(400, {"error": "Invalid urgency"})
                    return
            time_limit = 60 if time_limit is None else time_limit
            if blood_type not in BLOOD_TYPES:
                self._send_json(400, {"error": "Invalid blood type"})
                return
            if type(time_limit) is not int or not MIN_TIME_LIMIT_MINUTES <= time_limit <= MAX_TIME_LIMIT_MINUTES:
                self._send_json(400, {"error": f"Time limit must be a whole number of minutes from "
                                               f"{MIN_TIME_LIMIT_MINUTES} to {MAX_TIME_LIMIT_MINUTES}"})
                return
            if type(units) is not int or not 1 <= units <= 100:
                self._send_json(400, {"error": "Units required must be an integer between 1 and 100"})
                return
            radius = body.get("search_radius_km")
            if radius is not None and (type(radius) not in (int, float) or not math.isfinite(radius) or not 1 <= radius <= 1000):
                self._send_json(400, {"error": "Search radius must be between 1 and 1000 km, or empty for any distance"})
                return
            include_compatible = body.get("include_compatible", False)
            if not isinstance(include_compatible, bool):
                self._send_json(400, {"error": "include_compatible must be true or false"})
                return
            try:
                contact_phone = clean_phone(body.get("contact_phone"), required=False)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            try:
                hosp_lat = float(body.get("latitude", user["latitude"]))
                hosp_lon = float(body.get("longitude", user["longitude"]))
                if not math.isfinite(hosp_lat) or not math.isfinite(hosp_lon) or not -90 <= hosp_lat <= 90 or not -180 <= hosp_lon <= 180:
                    raise ValueError("Invalid coordinates")
            except (ValueError, TypeError):
                self._send_json(400, {"error": "Invalid hospital coordinates"})
                return

            facility = None
            facility_id = body.get("facility_id")
            if facility_id is not None:
                facility = next((f for f in STATE.facilities if f["facility_id"] == facility_id), None)
                if not facility or facility["latitude"] is None or facility["longitude"] is None:
                    self._send_json(400, {"error": "Select a facility with usable coordinates"})
                    return
                if body.get("facility_confirmed") is not True:
                    self._send_json(400, {"error": "Confirm the historical facility name and map coordinates before use"})
                    return
                hosp_lat, hosp_lon = facility["latitude"], facility["longitude"]

            hospital_name = _clean_text(body.get("hospital_name"), 150)
            address = ", ".join(p for p in (facility.get("address"), facility.get("district"), facility.get("state"))
                                if p and p != "NA") if facility else _clean_text(body.get("hospital_address"), 250)
            req_id = f"REQ_{uuid4().hex[:12].upper()}"
            new_req = {
                "request_id": req_id,
                "hospital_id": facility["facility_id"] if facility else user.get("hospital_id", "HOSP_01"),
                "hospital_name": facility["name"] if facility else (hospital_name or user.get("hospital_name", "General Hospital")),
                "hospital_address": address,
                "facility_source": "historical_directory_2015" if facility else ("map_or_manual" if "latitude" in body else "staff_profile"),
                "latitude": hosp_lat,
                "longitude": hosp_lon,
                "blood_type_required": blood_type,
                "include_compatible": include_compatible,
                "search_radius_km": radius,
                "units_required": units,
                "units_fulfilled": 0,
                "deadline_minutes": time_limit,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "ACTIVE",
                "created_by": user["user_id"],
                "doctor_name": user["name"],
                "contact_phone": contact_phone or user.get("phone", ""),
                "notes": _clean_text(body.get("notes"), 500),
            }
            STATE.emergencies[req_id] = new_req
            STATE.recruitment.start(new_req)
            self._send_json(201, {**request_view(new_req), "recruitment": new_req["recruitment"]})
            return

        if path == "/api/v1/emergency/requests/cancel":
            user = self._authenticate()
            if not user or user.get("role") != "hospital_staff":
                self._send_json(401, {"error": "Unauthorized: hospital staff only"})
                return
            emergency = STATE.emergencies.get(body.get("request_id"))
            if not emergency:
                self._send_json(404, {"error": "Emergency request not found"})
                return
            if emergency["status"] != "ACTIVE":
                self._send_json(409, {"error": "Only an active request can be cancelled"})
                return
            emergency["status"] = "CANCELLED"
            emergency["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            STATE.recruitment._settle(emergency, datetime.now(timezone.utc))
            STATE.save()
            self._send_json(200, request_view(emergency))
            return

        if path == "/api/v1/emergency/donation/complete":
            user = self._authenticate()
            if not user or user.get("role") != "hospital_staff":
                self._send_json(401, {"error": "Unauthorized: hospital staff only"})
                return
            donor_id = body.get("donor_id")
            req_id = body.get("request_id")
            if not donor_id or donor_id not in STATE.users or STATE.users[donor_id]["role"] != "donor":
                self._send_json(404, {"error": "Donor not found"})
                return
            if not req_id or req_id not in STATE.emergencies:
                self._send_json(404, {"error": "Emergency request not found"})
                return
            emergency = STATE.emergencies[req_id]
            now_dt = datetime.now(timezone.utc)
            eligible, _ = STATE.donation_tracker.check_temporal_eligibility(donor_id, now_dt)
            accepted = next((inv for inv in STATE.invitations if inv["donor_id"] == donor_id
                             and inv["request_id"] == req_id and inv["status"] == "ACCEPTED"), None)
            if emergency["status"] != "ACTIVE" or not eligible or not accepted \
                    or STATE.users[donor_id].get("blood_type") not in donor_blood_types(emergency):
                self._send_json(409, {"error": "Requires an active request, an eligible matching donor and an accepted invitation"})
                return

            outcome, rec = STATE.donation_tracker.process_arrival_for_donation(
                donor_id=donor_id, arrival_time=now_dt, emergency_id=req_id,
                still_needed=True, screening_rand=0.0, completion_rand=0.0)
            if outcome != DonationLifecycleState.DONATION_COMPLETED or not rec:
                self._send_json(409, {"error": "Donation was not completed"})
                return
            emergency["units_fulfilled"] += 1
            accepted["status"] = "COMPLETED"
            accepted["completed_at"] = now_dt.isoformat()
            if emergency["units_fulfilled"] >= emergency["units_required"]:
                emergency["status"] = "FULFILLED"
                emergency["fulfilled_at"] = now_dt.isoformat()
            # The donor is now deferred: close their other open invitations.
            for inv in STATE.invitations:
                if inv["donor_id"] == donor_id and inv["status"] in ("PENDING", "ACCEPTED"):
                    inv.update(status="CANCELLED", closed_reason="DONOR_DONATED")
            STATE.recruitment._settle(emergency, now_dt)
            STATE.save()
            self._send_json(200, {
                "message": "Donation recorded and deferral started.",
                "donor_id": donor_id, "outcome_state": outcome.value,
                "deferral_days": rec.deferral_days, "next_eligible_at": rec.next_eligible_at.isoformat(),
                "request_status": emergency["status"],
            })
            return

        self._send_json(404, {"error": "Endpoint not found"})


def run_api_server(host: str = "127.0.0.1", port: int = 8000, web_root: Optional[str] = None) -> HTTPServer:
    """Start the standard library HTTP server."""
    class DemoHTTPServer(HTTPServer):
        # Windows otherwise permits overlapping listeners on an occupied port.
        allow_reuse_address = False

        def service_actions(self):
            try:
                STATE.recruitment.tick()
                STATE.notifications.tick()
            except Exception:
                logger.exception("Recruitment scheduler tick failed")

    server = DemoHTTPServer((host, port), BloodDonorApiHandler)
    server.web_root = web_root
    logger.info("Blood Donor Recommendation API running at http://%s:%d", host, port)
    return server


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blood donor matching API and web client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-file", default="data/runtime/demo_state.json")
    parser.add_argument("--web-root", default=str(REPO_ROOT / "web"))
    parser.add_argument("--live", action="store_true",
                        help="Real data only: no demo donors and no seeded request (doctor account staff_001 is kept)")
    parser.add_argument("--demo-donors", type=int, default=30,
                        help="Additional synthetic O+ donor accounts (ignored with --live)")
    parser.add_argument("--sweep-interval-seconds", type=float, default=30,
                        help="How often open requests look for newly available donors")
    args = parser.parse_args()
    # Free OpenStreetMap road routing by default when the server is started directly.
    os.environ.setdefault("ROUTING_PROVIDER", "osrm")
    STATE = SystemState(args.state_file, 0 if args.live else args.demo_donors,
                        seed_demo=not args.live, sweep_interval_seconds=args.sweep_interval_seconds)
    server = run_api_server(args.host, args.port, args.web_root)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down API server.")
        server.server_close()
