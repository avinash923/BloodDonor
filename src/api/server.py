"""
src/api/server.py
Lightweight Standard-Library HTTP REST Server for Emergency Blood Recommendation.

Provides the required REST API endpoints for the Phase 9 Flutter mobile application
without introducing unnecessary or uninstalled dependencies.
Built using standard library `http.server` with JSON request/response handling.

Endpoints:
  POST /api/v1/auth/login                  -> User login (donor or hospital_staff)
  GET  /api/v1/donor/profile               -> Donor profile, eligibility & recent alerts
  GET  /api/v1/donor/invitations           -> Active emergency invitations for donor
  POST /api/v1/donor/invitations/respond   -> Accept or decline emergency invitation
  POST /api/v1/emergency/requests          -> Create emergency request (hospital staff)
  GET  /api/v1/emergency/requests          -> List emergency requests
  GET  /api/v1/emergency/recommendations   -> Get ranked candidates via UDI engine & routing
  GET  /api/v1/emergency/routes            -> Get detailed route polyline and travel estimate
  POST /api/v1/emergency/donation/complete -> Record confirmed donation and update eligibility

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
from datetime import datetime, timezone
from typing import Optional, Any
import threading

from src.ranking.udi_engine import UDIEngine, DonorCandidate
from src.spatial.routing_service import RoutingService
from src.spatial.routing_provider import LocationCoordinate, TravelMode, RoutingPreference
from src.simulation.donation_tracker import DonationAwareEligibilityTracker, DonationLifecycleState, CompletedDonationRecord, DonationType
from src.utils.logger import get_logger
from src.api.recruitment import RecruitmentManager
from src.api.donor_accounts import new_donor, password_matches, location_ready, update_location, revoke_location
from src.api.notifications import NotificationDispatcher
from src.simulation.donation_tracker import DonorTemporalEligibilityState

logger = get_logger("blood_donor_api_server")


class SystemState:
    """Thread-safe in-memory state shared across the REST API."""

    def __init__(self, state_path: Optional[str] = None, response_window_seconds: int = 60, demo_donors: int = 0):
        self.lock = threading.RLock()
        self.state_path = Path(state_path) if state_path else None
        self.sessions: dict[str, str] = {}
        self.notification_outbox = {}
        directory_path = Path(__file__).resolve().parents[2] / "data/processed/blood_bank_directory.json"
        self.facilities = json.loads(directory_path.read_text(encoding="utf-8"))["facilities"] if directory_path.exists() else []
        self.udi_engine = UDIEngine(formula_mode="multiplicative")
        self.routing_service = RoutingService()
        self.donation_tracker = DonationAwareEligibilityTracker(seed=42)

        # Preloaded demo users
        self.users = {
            "donor_001": {
                "user_id": "donor_001",
                "role": "donor",
                "name": "Sarah Chen",
                "blood_type": "O+",
                "phone": "+91 98765 43210",
                "latitude": 12.9352,
                "longitude": 77.6245,
                "availability": True,
                "recent_alerts": 1,
            },
            "donor_002": {
                "user_id": "donor_002",
                "role": "donor",
                "name": "Rahul Sharma",
                "blood_type": "A+",
                "phone": "+91 98765 12345",
                "latitude": 12.9716,
                "longitude": 77.5946,
                "availability": True,
                "recent_alerts": 0,
            },
            "staff_001": {
                "user_id": "staff_001",
                "role": "hospital_staff",
                "name": "Dr. Ramesh Patel",
                "hospital_id": "HOSP_CITY_01",
                "hospital_name": "City Central Trauma & Blood Bank",
                "latitude": 12.9915,
                "longitude": 77.5852,
            },
        }

        # Additional clearly labelled synthetic accounts make multi-wave demos possible.
        for index in range(3, demo_donors + 3):
            did = f"donor_{index:03d}"
            self.users[did] = {"user_id": did, "role": "donor", "name": f"Demo Donor {index:03d}",
                "blood_type": "O+", "phone": "Demo account", "latitude": 12.9915 + index * 0.0007,
                "longitude": 77.5852 + index * 0.0005, "availability": True, "recent_alerts": 0}

        # Initialize donation tracker for demo donors
        self.donation_tracker.initialize_donor_population(
            [did for did, user in self.users.items() if user["role"] == "donor"],
            synthetic_female_ratio=0.5,
        )

        # In-memory emergency requests
        self.emergencies: dict[str, dict] = {
            "REQ_TRAUMA_101": {
                "request_id": "REQ_TRAUMA_101",
                "hospital_id": "HOSP_CITY_01",
                "hospital_name": "City Central Trauma & Blood Bank",
                "latitude": 12.9915,
                "longitude": 77.5852,
                "blood_type_required": "O+",
                "units_required": 3,
                "units_fulfilled": 1,
                "urgency": "CRITICAL",
                "deadline_minutes": 60,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "ACTIVE",
            }
        }

        # In-memory invitations
        self.invitations: list[dict] = [
            {
                "invitation_id": "INV_1001",
                "request_id": "REQ_TRAUMA_101",
                "donor_id": "donor_001",
                "blood_type_required": "O+",
                "urgency": "CRITICAL",
                "hospital_name": "City Central Trauma & Blood Bank",
                "hospital_lat": 12.9915,
                "hospital_lon": 77.5852,
                "status": "PENDING", # PENDING, ACCEPTED, DECLINED
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
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

        self.recruitment = RecruitmentManager(self, response_window_seconds)
        self.notifications = NotificationDispatcher(self)

    def save(self):
        """Atomically persist demo activity; sessions deliberately expire on restart."""
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
    """Demo recruitment uses exact blood-group matching, as invitation dispatch does."""
    return (user.get("role") == "donor" and user.get("availability", False)
            and user.get("blood_type") == emergency["blood_type_required"])


STATE = SystemState()


class BloodDonorApiHandler(BaseHTTPRequestHandler):
    """Handles REST API requests from the Flutter client."""

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
        """Simple header token or query parameter auth check."""
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
            self._send_json(403, {"error": "This resource requires hospital staff"})
            return None
        return user

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        web_root = getattr(self.server, "web_root", None)
        if web_root and not path.startswith("/api/"):
            root = Path(web_root).resolve()
            target = (root / urllib.parse.unquote(path).lstrip("/")).resolve()
            if not target.is_relative_to(root):
                self._send_json(403, {"error": "Invalid asset path"})
                return
            if path == "/":
                target = root / "index.html"
            if not target.is_file():
                self._send_json(404, {"error": "App asset not found. Build the Flutter web demo first."})
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 1. Health check
        if path == "/api/v1/health":
            self._send_json(200, {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "routing_provider": STATE.routing_service.provider_type.value,
                "external_notifications_configured": STATE.notifications.configured,
                "facility_records": len(STATE.facilities),
            })
            return

        if path == "/api/v1/facilities":
            if not self._require_role("hospital_staff"):
                return
            term = query.get("search", [""])[0].lower()
            facilities = [f for f in STATE.facilities if term in " ".join(str(f.get(k, "")) for k in ("name", "state", "city", "district")).lower()]
            self._send_json(200, {"facilities": facilities[:100], "total": len(facilities),
                                  "source_date": "2015-09", "notice": "Historical directory; verify facility and coordinates before use"})
            return

        # 2. Donor Profile
        if path == "/api/v1/donor/profile":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Unauthorized or not a donor"})
                return

            did = user["user_id"]
            now_dt = datetime.now(timezone.utc)
            is_elig, reason = STATE.donation_tracker.check_temporal_eligibility(did, now_dt)
            don_state = STATE.donation_tracker.donor_states.get(did)

            resp = {
                "user_id": did,
                "name": user["name"],
                "blood_type": user["blood_type"],
                "phone": user["phone"],
                "latitude": user["latitude"],
                "longitude": user["longitude"],
                "availability": user["availability"],
                "recent_alerts": user["recent_alerts"],
                "is_temporally_eligible": is_elig,
                "eligibility_reason": reason,
                "next_eligible_at": don_state.next_eligible_at.isoformat() if don_state and don_state.next_eligible_at else None,
                "completed_donations_count": len(don_state.completed_donations) if don_state else 0,
                "account_source": user.get("account_source", "synthetic_demo"),
                "location_ready": location_ready(user),
                "location_consent": user.get("location_consent", False),
                "location_captured_at": user.get("location_captured_at"),
                "location_accuracy_meters": user.get("location_accuracy_meters"),
            }
            self._send_json(200, resp)
            return

        # 3. Donor Invitations
        if path == "/api/v1/donor/invitations":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Unauthorized"})
                return

            did = user["user_id"]
            with STATE.lock:
                donor_invites = [inv for inv in STATE.invitations if inv["donor_id"] == did]
            self._send_json(200, {"invitations": donor_invites})
            return

        # 4. Emergency Requests (Hospital Staff)
        if path == "/api/v1/emergency/requests":
            if not self._require_role("hospital_staff"):
                return
            with STATE.lock:
                req_list = list(STATE.emergencies.values())
            self._send_json(200, {"requests": req_list})
            return

        # 5. Recommendations for a Request (Hospital Staff)
        if path == "/api/v1/emergency/recommendations":
            if not self._require_role("hospital_staff"):
                return
            req_id = query.get("request_id", [None])[0]
            if not req_id or req_id not in STATE.emergencies:
                self._send_json(404, {"error": f"Emergency request {req_id} not found"})
                return

            try:
                page = int(query.get("page", ["1"])[0])
                if page < 1:
                    raise ValueError()
            except ValueError:
                self._send_json(400, {"error": "Page must be a positive integer"})
                return
            results, accepted, summary = STATE.recruitment.snapshot(STATE.emergencies[req_id])
            view = query.get("view", ["all"])[0]
            if view not in ("all", "accepted"):
                self._send_json(400, {"error": "View must be all or accepted"})
                return
            selected = accepted if view == "accepted" else results
            start = (page - 1) * 10
            self._send_json(200, {"request_id": req_id, "urgency": STATE.emergencies[req_id]["urgency"],
                "recommendations": selected[start:start + 10], "accepted_donors": accepted,
                "recruitment": summary, "page": page, "page_size": 10,
                "total": len(selected), "has_next": start + 10 < len(selected)})
            return

        # 6. Detailed Route with Polyline
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
                self._send_json(409, {"error": "Donor must share a fresh, accurate location first"})
                return

            orig = LocationCoordinate(d_user["latitude"], d_user["longitude"])
            dest = LocationCoordinate(em_req["latitude"], em_req["longitude"])

            route_est = STATE.routing_service.provider.compute_detailed_route(
                origin=orig,
                destination=dest,
                origin_id=donor_id,
                destination_id=em_req["hospital_id"],
            )

            self._send_json(200, {
                "donor_id": donor_id,
                "request_id": req_id,
                "hospital_name": em_req["hospital_name"],
                "origin": {"latitude": orig.latitude, "longitude": orig.longitude},
                "destination": {"latitude": dest.latitude, "longitude": dest.longitude},
                "is_success": route_est.is_success,
                "duration_minutes": round(route_est.duration_minutes, 1) if route_est.duration_minutes is not None else None,
                "distance_km": round(route_est.distance_meters / 1000.0, 2) if route_est.distance_meters is not None else None,
                "polyline_encoded": route_est.polyline_encoded,
                "provider": route_est.provider,
                "error_code": route_est.error_code,
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

        # 1. Login
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
            STATE.donation_tracker.donor_states[did] = DonorTemporalEligibilityState(did, 120.0)
            STATE.save()
            token = secrets.token_urlsafe(32)
            STATE.sessions[token] = did
            self._send_json(201, {"token": token, "user_id": did, "name": donor["name"], "role": "donor"})
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
                else:
                    update_location(user, body)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            STATE.recruitment._snapshots.clear()
            if location_ready(user):
                for emergency in STATE.emergencies.values():
                    if emergency["status"] == "ACTIVE" and emergency.get("recruitment", {}).get("status") == "EXHAUSTED":
                        STATE.recruitment._advance(emergency, datetime.now(timezone.utc))
            STATE.save()
            self._send_json(200, {"location_ready": location_ready(user), "availability": user["availability"]})
            return

        if path == "/api/v1/auth/login":
            username = body.get("username", "")
            if not isinstance(username, str):
                self._send_json(400, {"error": "Username must be text"})
                return
            username = username.strip()
            # Demo authentication: matches donor_001, donor_002, or staff_001
            user = STATE.users.get(username)
            if user and user.get("password_record") and not password_matches(body.get("password"), user["password_record"]):
                user = None
            if user and not user.get("password_record") and os.getenv("ALLOW_DEMO_LOGIN", "true").lower() != "true":
                user = None
            if user:
                token = secrets.token_urlsafe(32)
                STATE.sessions[token] = username
                self._send_json(200, {
                    "token": token,
                    "user_id": user["user_id"],
                    "name": user["name"],
                    "role": user["role"],
                })
            else:
                self._send_json(401, {"error": "Invalid username. Use 'donor_001' or 'staff_001'."})
            return

        # 2. Donor Respond to Invitation
        if path == "/api/v1/donor/invitations/respond":
            user = self._authenticate()
            if not user or user.get("role") != "donor":
                self._send_json(401, {"error": "Unauthorized"})
                return

            inv_id = body.get("invitation_id")
            action = body.get("action") # "ACCEPT" or "DECLINE"

            if action not in ("ACCEPT", "DECLINE"):
                self._send_json(400, {"error": "Action must be 'ACCEPT' or 'DECLINE'"})
                return

            with STATE.lock:
                target_inv = next((inv for inv in STATE.invitations if inv["invitation_id"] == inv_id and inv["donor_id"] == user["user_id"]), None)
                if not target_inv:
                    self._send_json(404, {"error": "Invitation not found"})
                    return

                try:
                    STATE.recruitment.respond(target_inv, action)
                except ValueError as error:
                    self._send_json(409, {"error": str(error)})
                    return

            self._send_json(200, {
                "message": f"Invitation {action.lower()}ed successfully",
                "invitation": target_inv,
            })
            return

        # 3. Create Emergency Request (Staff)
        if path == "/api/v1/emergency/requests":
            user = self._authenticate()
            if not user or user.get("role") != "hospital_staff":
                self._send_json(401, {"error": "Unauthorized: hospital staff only"})
                return

            blood_type = body.get("blood_type_required", "O+")
            units = body.get("units_required", 1)
            urgency = body.get("urgency", "URGENT")
            if blood_type not in ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-") or urgency not in ("CRITICAL", "URGENT", "SCHEDULED"):
                self._send_json(400, {"error": "Invalid blood type or urgency"})
                return
            if type(units) is not int or not 1 <= units <= 100:
                self._send_json(400, {"error": "Units required must be an integer between 1 and 100"})
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

            req_id = f"REQ_{uuid4().hex}"
            new_req = {
                "request_id": req_id,
                "hospital_id": facility["facility_id"] if facility else user.get("hospital_id", "HOSP_01"),
                "hospital_name": facility["name"] if facility else user.get("hospital_name", "General Hospital"),
                "facility_source": "historical_directory_2015" if facility else "demo",
                "latitude": hosp_lat,
                "longitude": hosp_lon,
                "blood_type_required": blood_type,
                "units_required": units,
                "units_fulfilled": 0,
                "urgency": urgency,
                "deadline_minutes": 60 if urgency == "CRITICAL" else (120 if urgency == "URGENT" else 240),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "ACTIVE",
            }

            with STATE.lock:
                STATE.emergencies[req_id] = new_req
                STATE.recruitment.start(new_req)

            self._send_json(201, new_req)
            return

        # 4. Complete Donation and Trigger Temporal Deferral Gate
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
            eligible, _ = STATE.donation_tracker.check_temporal_eligibility(donor_id, datetime.now(timezone.utc))
            accepted = next((inv for inv in STATE.invitations if inv["donor_id"] == donor_id and inv["request_id"] == req_id and inv["status"] == "ACCEPTED"), None)
            if emergency["status"] != "ACTIVE" or not eligible or not matching_donor(STATE.users[donor_id], emergency) or not accepted:
                self._send_json(409, {"error": "Requires an active request, an eligible matching donor and an accepted invitation"})
                return

            now_dt = datetime.now(timezone.utc)
            # Process arrival & complete donation in donation tracker
            outcome, rec = STATE.donation_tracker.process_arrival_for_donation(
                donor_id=donor_id,
                arrival_time=now_dt,
                emergency_id=req_id or "ADHOC",
                still_needed=True,
                screening_rand=0.0, # guaranteed pass
                completion_rand=0.0, # guaranteed completion
            )

            with STATE.lock:
                if outcome == DonationLifecycleState.DONATION_COMPLETED and rec:
                    STATE.emergencies[req_id]["units_fulfilled"] += 1
                    accepted["status"] = "COMPLETED"
                    if emergency["units_fulfilled"] >= emergency["units_required"]:
                        emergency["status"] = "FULFILLED"
                        for inv in STATE.invitations:
                            if inv["request_id"] == req_id and inv["status"] in ("PENDING", "ACCEPTED"):
                                inv["status"] = "CANCELLED"
                    for inv in STATE.invitations:
                        if inv["donor_id"] == donor_id and inv["status"] in ("PENDING", "ACCEPTED"):
                            inv["status"] = "CANCELLED"
                    if emergency.get("recruitment"):
                        STATE.recruitment._advance(emergency, now_dt)
                    STATE.save()
                else:
                    self._send_json(409, {"error": "Donation was not completed"})
                    return

            self._send_json(200, {
                "message": "Donation successfully recorded and temporal deferral initiated.",
                "donor_id": donor_id,
                "outcome_state": outcome.value,
                "deferral_days": rec.deferral_days if rec else 90.0,
                "next_eligible_at": rec.next_eligible_at.isoformat() if rec else None,
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
    parser = argparse.ArgumentParser(description="Local Blood Donor demonstration API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-file", default="data/runtime/demo_state.json")
    parser.add_argument("--web-root", default="mobile/build/web")
    parser.add_argument("--response-window-seconds", type=int, default=60)
    parser.add_argument("--demo-donors", type=int, default=30,
                        help="Additional synthetic O+ donor accounts for wave demonstrations")
    args = parser.parse_args()
    STATE = SystemState(args.state_file, args.response_window_seconds, args.demo_donors)
    server = run_api_server(args.host, args.port, args.web_root)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down API server.")
        server.server_close()
