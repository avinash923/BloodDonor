"""Local donor accounts and consented foreground location snapshots."""
import hashlib
import hmac
import math
import re
import secrets
from datetime import datetime, timedelta, timezone

BLOOD_TYPES = ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-")
MAX_LOCATION_AGE_SECONDS = 900
MAX_ACCURACY_METERS = 250
MAX_PRIOR_DONATIONS = 200
MAX_HISTORY_MONTHS = 720
DAYS_PER_MONTH = 30.4375
HISTORY_FIELDS = ("prior_donation_count", "months_since_last_donation", "months_since_first_donation")


def password_record(password):
    if not isinstance(password, str) or not 10 <= len(password) <= 128:
        raise ValueError("Password must contain 10–128 characters")
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600000).hex()
    return {"salt": salt, "digest": digest}


def password_matches(password, record):
    if not isinstance(password, str) or len(password) > 128:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(record["salt"]), 600000).hex()
    return hmac.compare_digest(digest, record["digest"])


def donation_history(body, now=None, source="self_reported"):
    """Validate optional self-reported history (counts and months before now).

    Returns None when no history fields are supplied or the donor reports no
    prior donations. Months are converted to timestamps so they age correctly.
    """
    if all(body.get(key) is None for key in HISTORY_FIELDS):
        return None
    count, last, first = (body.get(key) for key in HISTORY_FIELDS)
    if type(count) is not int or not 0 <= count <= MAX_PRIOR_DONATIONS:
        raise ValueError(f"Prior donation count must be a whole number from 0 to {MAX_PRIOR_DONATIONS}")
    if count == 0:
        if last is not None or first is not None:
            raise ValueError("Donation dates cannot be given without prior donations")
        return None
    if first is None and count == 1:
        first = last
    for label, value in (("last", last), ("first", first)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= MAX_HISTORY_MONTHS:
            raise ValueError(f"Months since {label} donation must be a number from 0 to {MAX_HISTORY_MONTHS}")
    if first < last:
        raise ValueError("The first donation cannot be more recent than the last donation")
    if count == 1 and first != last:
        raise ValueError("With one prior donation, the first and last donation are the same")
    now = now or datetime.now(timezone.utc)
    at = lambda months: (now - timedelta(days=months * DAYS_PER_MONTH)).isoformat()
    return {"prior_donations": count, "first_donation_at": at(first), "last_donation_at": at(last),
            "source": source, "reported_at": now.isoformat()}


def new_donor(body):
    username = body.get("username")
    name = body.get("name")
    if not isinstance(username, str) or not re.fullmatch(r"[a-z][a-z0-9_]{3,39}", username):
        raise ValueError("Username: 4–40 lowercase letters, numbers or underscores, starting with a letter")
    if username.startswith(("donor_", "staff_")):
        raise ValueError("This username prefix is reserved for demo accounts")
    if not isinstance(name, str) or not 2 <= len(name.strip()) <= 100:
        raise ValueError("Enter a name with 2–100 characters")
    if body.get("blood_type") not in BLOOD_TYPES:
        raise ValueError("Select a valid blood group")
    return {"user_id": username, "role": "donor", "name": name.strip(),
            "blood_type": body["blood_type"], "phone": "", "latitude": None,
            "longitude": None, "availability": False, "recent_alerts": 0,
            "account_source": "registered", "location_consent": False,
            "donation_history": donation_history(body),
            "password_record": password_record(body.get("password"))}


def location_ready(donor, now=None):
    # Legacy accounts are explicitly synthetic; never require real device GPS for them.
    if donor.get("account_source") != "registered":
        return donor.get("latitude") is not None and donor.get("longitude") is not None
    if not donor.get("location_consent"):
        return False
    try:
        captured = datetime.fromisoformat(donor["location_captured_at"])
        age = ((now or datetime.now(timezone.utc)) - captured).total_seconds()
        accuracy = donor["location_accuracy_meters"]
        return 0 <= age <= MAX_LOCATION_AGE_SECONDS and 0 <= accuracy <= MAX_ACCURACY_METERS
    except (KeyError, ValueError, TypeError):
        return False


def update_location(donor, body, now=None):
    now = now or datetime.now(timezone.utc)
    if body.get("consent") is not True:
        raise ValueError("Explicit location-sharing consent is required")
    values = [body.get(k) for k in ("latitude", "longitude", "accuracy_meters")]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Coordinates and accuracy must be finite numbers")
    lat, lon, accuracy = values
    if not -90 <= lat <= 90 or not -180 <= lon <= 180 or not 0 <= accuracy <= MAX_ACCURACY_METERS:
        raise ValueError("Invalid coordinates or location accuracy exceeds 250 metres")
    try:
        captured = datetime.fromisoformat(body["captured_at"].replace("Z", "+00:00"))
        age = (now - captured).total_seconds()
        if not 0 <= age <= 120:
            raise ValueError()
    except (KeyError, AttributeError, ValueError, TypeError):
        raise ValueError("Location must include a timezone and have been captured within the last two minutes") from None
    previous = donor.get("location_captured_at")
    if previous and captured < datetime.fromisoformat(previous):
        raise ValueError("An older location cannot replace a newer location")
    donor.update(latitude=lat, longitude=lon, location_accuracy_meters=accuracy,
                 location_captured_at=captured.isoformat(), location_received_at=now.isoformat(),
                 location_consent=True, location_source="device", availability=True)


def revoke_location(donor):
    donor.update(latitude=None, longitude=None, location_consent=False, availability=False)
    for field in ("location_accuracy_meters", "location_captured_at", "location_received_at", "location_source"):
        donor.pop(field, None)
