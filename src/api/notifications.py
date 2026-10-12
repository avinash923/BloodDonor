"""Durable invitation webhook outbox for an operator-configured push/SMS gateway.

No external messages are sent unless both endpoint and bearer token are configured.
The gateway must deduplicate Idempotency-Key and map donor IDs to opted-in devices.
"""
from datetime import datetime, timedelta, timezone
import json
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class NotificationDispatcher:
    def __init__(self, state):
        self.state = state
        self.endpoint = os.getenv("NOTIFICATION_WEBHOOK_URL", "")
        self.token = os.getenv("NOTIFICATION_WEBHOOK_TOKEN", "")
        if self.endpoint and (urlparse(self.endpoint).scheme != "https" or not urlparse(self.endpoint).hostname):
            raise ValueError("Notification webhook must be an HTTPS endpoint")

    @property
    def configured(self):
        return bool(self.endpoint and self.token)

    def queue(self, invitation):
        self.state.notification_outbox.setdefault(invitation["invitation_id"], {
            "invitation_id": invitation["invitation_id"], "status": "QUEUED",
            "attempts": 0, "next_attempt_at": invitation["sent_at"]})

    def tick(self, now=None):
        now = now or datetime.now(timezone.utc)
        if not self.configured:
            return
        for item in self.state.notification_outbox.values():
            if item["status"] not in ("QUEUED", "RETRY") or datetime.fromisoformat(item["next_attempt_at"]) > now:
                continue
            invitation = next((i for i in self.state.invitations if i["invitation_id"] == item["invitation_id"]), None)
            if not invitation or invitation["status"] != "PENDING" or datetime.fromisoformat(invitation["expires_at"]) <= now:
                item["status"] = "CANCELLED"
                self.state.save()
                continue
            # Do not send donor GPS, password data or phone numbers to the gateway.
            payload = {"event": "donor.invited", **{k: invitation[k] for k in (
                "invitation_id", "request_id", "donor_id", "hospital_name", "blood_type_required", "urgency", "expires_at")}}
            req = Request(self.endpoint, data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}",
                                   "Idempotency-Key": invitation["invitation_id"]})
            item["attempts"] += 1
            try:
                with urlopen(req, timeout=5) as response:
                    if not 200 <= response.status < 300:
                        raise OSError("Gateway rejected request")
                item.update(status="GATEWAY_ACCEPTED", accepted_at=now.isoformat())
            except Exception:
                # Do not persist exception messages that could contain URL secrets.
                item.update(status="FAILED" if item["attempts"] >= 3 else "RETRY",
                            next_attempt_at=(now + timedelta(seconds=30 * item["attempts"])).isoformat())
            self.state.save()
            break  # One bounded delivery attempt per server loop; never claim device receipt.
