"""Local donor accounts and consented donor location snapshots."""
import hashlib
import hmac
import math
import re
import secrets
from datetime import date, datetime, timedelta, timezone

BLOOD_TYPES = ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-")
# Red-cell (whole blood / packed RBC) donor groups acceptable for each recipient group.
RBC_COMPATIBLE_DONORS = {
    "O-": ("O-",),
    "O+": ("O-", "O+"),
    "A-": ("O-", "A-"),
    "A+": ("O-", "O+", "A-", "A+"),
    "B-": ("O-", "B-"),
    "B+": ("O-", "O+", "B-", "B+"),
    "AB-": ("O-", "A-", "B-", "AB-"),
    "AB+": BLOOD_TYPES,
}
# A location older than this is still used for matching, but staff see it as stale.
MAX_LOCATION_AGE_SECONDS = 900
LOCATION_SOURCES = ("device", "manual")
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


def donor_blood_types(request):
    """Donor groups invited for a request: exact match, or every RBC-compatible group."""
    needed = request["blood_type_required"]
    return RBC_COMPATIBLE_DONORS[needed] if request.get("include_compatible") else (needed,)


def _parse_date(value, label, today):
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a date (YYYY-MM-DD)") from None
    if parsed > today:
        raise ValueError(f"{label} cannot be in the future")
    if (today - parsed).days > MAX_HISTORY_MONTHS * DAYS_PER_MONTH:
        raise ValueError(f"{label} is too far in the past")
    return parsed


def history_from_dates(body, now=None):
    """Turn calendar dates (last/first donation) into the month fields used by donation_history."""
    if not body.get("last_donation_date"):
        return body
    now = now or datetime.now(timezone.utc)
    today = now.date()
    last = _parse_date(body["last_donation_date"], "Last donation date", today)
    count = body.get("prior_donation_count")
    if count in (None, 0):
        count = 1
    first = _parse_date(body["first_donation_date"], "First donation date", today) if body.get("first_donation_date") else None
    if first is None:
        if count > 1:
            raise ValueError("Enter the approximate date of your first donation")
        first = last
    if first > last:
        raise ValueError("The first donation cannot be more recent than the last donation")
    months = lambda day: (today - day).days / DAYS_PER_MONTH
    return {**body, "prior_donation_count": count,
            "months_since_last_donation": months(last), "months_since_first_donation": months(first)}


def clean_phone(value, required=True):
    if value in (None, "") and not required:
        return ""
    if not isinstance(value, str) or not re.fullmatch(r"\+?[0-9][0-9 \-]{6,18}[0-9]", value.strip()):
        raise ValueError("Enter a valid contact phone number (digits, spaces, dashes, optional +)")
    return value.strip()


def clean_email(value):
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or len(value) > 120 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()):
        raise ValueError("Enter a valid email address or leave it empty")
    return value.strip()


def last_donation(donor, donation_state):
    """Most recent known donation (self-reported or recorded in this system) and total count."""
    history = donor.get("donation_history") or {}
    times = [datetime.fromisoformat(history["last_donation_at"])] if history.get("last_donation_at") else []
    recorded = list(donation_state.completed_donations) if donation_state else []
    times += [record.completed_at for record in recorded]
    return {"last_donation_at": max(times).isoformat() if times else None,
            "total_donations": int(history.get("prior_donations", 0)) + len(recorded),
            "donations_recorded_in_app": len(recorded)}


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
    phone = clean_phone(body.get("phone"))
    email = clean_email(body.get("email"))
    history = donation_history(history_from_dates(body))
    donor = {"user_id": username, "role": "donor", "name": name.strip(),
             "blood_type": body["blood_type"], "phone": phone, "email": email, "latitude": None,
             "longitude": None, "availability": False, "recent_alerts": 0,
             "account_source": "registered", "location_consent": False,
             "donation_history": history,
             "registered_at": datetime.now(timezone.utc).isoformat(),
             "password_record": password_record(body.get("password"))}
    if isinstance(body.get("location"), dict):
        update_location(donor, {**body["location"], "consent": True})
    return donor


def location_ready(donor, now=None):
    """True when the donor has a consented location that matching can use.

    Registered donors keep their last shared location (live GPS or a point they
    picked on the map) until they revoke it; its age is reported separately so
    staff can judge it. Acceptance refreshes the location when the device allows.
    """
    if donor.get("latitude") is None or donor.get("longitude") is None:
        return False
    # Legacy demo accounts are explicitly synthetic; they never share real device GPS.
    if donor.get("account_source") != "registered":
        return True
    return bool(donor.get("location_consent"))


def location_fresh(donor, now=None):
    """Live device fix captured within MAX_LOCATION_AGE_SECONDS."""
    if not location_ready(donor) or donor.get("location_source", "device") != "device":
        return False
    try:
        captured = datetime.fromisoformat(donor["location_captured_at"])
        age = ((now or datetime.now(timezone.utc)) - captured).total_seconds()
        return 0 <= age <= MAX_LOCATION_AGE_SECONDS
    except (KeyError, ValueError, TypeError):
        return False


def update_location(donor, body, now=None):
    now = now or datetime.now(timezone.utc)
    if body.get("consent") is not True:
        raise ValueError("Explicit location-sharing consent is required")
    source = body.get("source", "device")
    if source not in LOCATION_SOURCES:
        raise ValueError("Location source must be device or manual")
    if source == "manual" and body.get("accuracy_meters") is None:
        body = {**body, "accuracy_meters": None}
        values = [body.get(k) for k in ("latitude", "longitude")]
    else:
        values = [body.get(k) for k in ("latitude", "longitude", "accuracy_meters")]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Coordinates and accuracy must be finite numbers")
    lat, lon, accuracy = body.get("latitude"), body.get("longitude"), body.get("accuracy_meters")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180 or (
            accuracy is not None and not 0 <= accuracy <= MAX_ACCURACY_METERS):
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
                 location_consent=True, location_source=source, availability=True)


def revoke_location(donor):
    donor.update(latitude=None, longitude=None, location_consent=False, availability=False)
    for field in ("location_accuracy_meters", "location_captured_at", "location_received_at", "location_source"):
        donor.pop(field, None)
