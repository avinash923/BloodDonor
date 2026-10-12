"""Exercise a running local demo and save the actual API results for review.

Uses synthetic demo accounts only; sends no external notifications.
Run: python tools/demo_recruitment.py --base-url http://127.0.0.1:8012
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def run(base_url, output):
    output.mkdir(parents=True, exist_ok=True)

    def call(path, token=None, body=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(base_url + "/api/v1/" + path, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    def login(donor_id):
        return call("auth/login", body={"username": donor_id})["token"]

    def accept(donor_id, request_id):
        token = login(donor_id)
        invitation = next(i for i in call("donor/invitations", token)["invitations"]
                          if i["request_id"] == request_id and i["status"] == "PENDING")
        return call("donor/invitations/respond", token,
                    {"invitation_id": invitation["invitation_id"], "action": "ACCEPT"})

    staff = login("staff_001")
    request = call("emergency/requests", staff,
        {"blood_type_required": "O+", "units_required": 3, "urgency": "CRITICAL"})
    request_id = request["request_id"]
    path = "emergency/recommendations?request_id=" + request_id
    first = call(path, staff)
    assert first["recruitment"]["pending_count"] == 10
    first_ids = [r["donor_id"] for r in first["recommendations"] if r["invitation_status"] == "PENDING"]
    assert len(first_ids) == 10
    output.joinpath("first_batch.json").write_text(json.dumps(first, indent=2), encoding="utf-8")
    print("Batch 1: 10 donors invited, sorted by road ETA", flush=True)
    page2 = call(path + "&page=2", staff)
    assert page2["recruitment"]["contacted_count"] == 10
    assert all(r["invitation_status"] == "QUEUED" for r in page2["recommendations"])
    for donor_id in first_ids[:2]:
        accept(donor_id, request_id)
    print("Two donors accepted; 1 additional donor needed. Waiting for server timer...", flush=True)
    # No HTTP requests during the timeout: the server timer advances the wave.
    time.sleep(first["recruitment"]["response_window_seconds"] + 0.8)
    second = call(path, staff)
    summary = second["recruitment"]
    assert summary["batch_number"] == 2 and summary["pending_count"] == 10
    assert summary["accepted_count"] == 2 and summary["contacted_count"] == 20
    output.joinpath("second_batch.json").write_text(json.dumps(second, indent=2), encoding="utf-8")
    pending = [r for r in call(path + "&page=2", staff)["recommendations"] if r["invitation_status"] == "PENDING"]
    assert pending and pending[0]["donor_id"] not in first_ids
    accept(pending[0]["donor_id"], request_id)
    final = call(path + "&view=accepted", staff)
    assert final["recruitment"]["accepted_count"] == 3
    assert final["recruitment"]["pending_count"] == 0
    assert final["recruitment"]["status"] == "TARGET_RESERVED"
    etas = [r["travel_time_minutes"] for r in final["recommendations"]]
    assert etas == sorted(etas)
    output.joinpath("accepted_donors.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    output.joinpath("request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    result = {"base_url": base_url, "request_id": request_id, "routing_provider": final["recruitment"]["routing_provider"],
        "batch_1_invited": 10, "batch_2_invited": 10, "accepted": 3,
        "awaiting_response": 0, "remaining_needed": 0, "status": "TARGET_RESERVED",
        "accepted_donors": final["recommendations"], "external_notifications_sent": False}
    output.joinpath("result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--output", type=Path, default=Path("reports/demo_recruitment"))
    arguments = parser.parse_args()
    run(arguments.base_url, arguments.output)
