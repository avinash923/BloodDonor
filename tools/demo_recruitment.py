"""Exercise a running local demo and save the actual API results for review.

Uses synthetic demo accounts only; sends no external notifications.
Start the server with demo donors first (e.g. .\\start-demo.ps1 -Port 8012), then run:
    python tools/demo_recruitment.py --base-url http://127.0.0.1:8012
"""
import argparse
import json
from pathlib import Path
import urllib.request


def run(base_url, output):
    output.mkdir(parents=True, exist_ok=True)

    def call(path, token=None, body=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(base_url + "/api/v1/" + path, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def login(user_id):
        return call("auth/login", body={"username": user_id})["token"]

    def accept(donor_id, request_id):
        token = login(donor_id)
        invitation = next(i for i in call("donor/invitations", token)["invitations"]
                          if i["request_id"] == request_id and i["status"] == "PENDING")
        return call("donor/invitations/respond", token,
                    {"invitation_id": invitation["invitation_id"], "action": "ACCEPT"})

    staff = login("staff_001")
    request = call("emergency/requests", staff, {
        "blood_type_required": "O+", "units_required": 3, "time_limit_minutes": 60,
        "search_radius_km": 25, "hospital_name": "Demo Hospital", "contact_phone": "+91 80 0000 0000"})
    request_id = request["request_id"]
    path = "emergency/recommendations?request_id=" + request_id + "&page_size=500"
    first = call(path, staff)
    summary = first["recruitment"]
    assert summary["pending_count"] == summary["invited_count"] > 0
    output.joinpath("broadcast.json").write_text(json.dumps(first, indent=2), encoding="utf-8")
    print(f"Broadcast: {summary['invited_count']} matching donors alerted at once", flush=True)

    ranked = [r["donor_id"] for r in first["recommendations"]]
    for donor_id in ranked[:3]:
        accept(donor_id, request_id)
    final = call(path, staff)
    summary = final["recruitment"]
    assert summary["accepted_count"] == 3 and summary["pending_count"] == 0
    assert summary["status"] == "TARGET_REACHED"
    accepted = final["accepted_donors"]
    assert all(a["phone"] and a["travel_time_minutes"] is not None for a in accepted)
    output.joinpath("accepted_donors.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    result = {"base_url": base_url, "request_id": request_id, "routing_provider": summary["routing_provider"],
              "alerted": summary["invited_count"], "accepted": len(accepted), "status": summary["status"],
              "accepted_donors": [{k: a[k] for k in ("donor_name", "phone", "last_donation_at",
                                                     "total_donations", "travel_time_minutes")} for a in accepted],
              "external_notifications_sent": False}
    output.joinpath("result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--output", default="reports/demo_recruitment")
    args = parser.parse_args()
    run(args.base_url.rstrip("/"), Path(args.output))
