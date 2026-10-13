# LifePulse — Emergency Blood Donor Matching

A Python backend and browser app for emergency blood requests. A doctor raises a request and chooses the
time limit; every donor with a matching blood group who can reach the hospital by road within that limit is
alerted at once. Donors are ranked by road travel time and by a predicted likelihood to donate. When a donor
accepts, the doctor sees their name, phone number, email, last donation, total donations, live ETA and route.

Maps and routing use free OpenStreetMap services — no API key is needed:

* **Map tiles:** OpenStreetMap via Leaflet.
* **Road routing and ETAs:** OSRM (`router.project-osrm.org` by default). These are free-flow estimates
  without live traffic. The public server has rate limits and no uptime guarantee, so set `OSRM_BASE_URL`
  to a self-hosted OSRM for production. If OSRM can't be reached, the server falls back to an offline
  estimate. The UI labels these estimates "Approximate" and shows a dashed straight line instead of a road route.
* **Address search:** Nominatim. It is called from the browser only when you press Search, as its usage
  policy requires.

## Start the live system (real data only, Windows)

```powershell
.\start-full.ps1
```

Open **http://127.0.0.1:8000**. Live mode (`--live`) loads no demo donors and no seeded request. Activity is
stored in `data/runtime/live_state.json`. The doctor account is `staff_001`. It uses username-only demo login,
which `ALLOW_DEMO_LOGIN=false` disables. Donors register themselves in the app.

On other platforms, or by hand:

```bash
python -m src.api.server --live --port 8000 --state-file data/runtime/live_state.json
```

When the server starts this way, it uses OSRM unless `ROUTING_PROVIDER` is set (`synthetic` or `google`).

## Start the demo (synthetic donors)

```powershell
.\start-demo.ps1 -Port 8012 -DemoDonors 30 -StateFile data/runtime/demo_state.json
python tools/demo_recruitment.py --base-url http://127.0.0.1:8012
```

The demo adds labelled synthetic O+ donors (`donor_003` onwards), plus `donor_001` (O+) and `donor_002` (A+).
It also adds one seeded request. The script raises a 3-unit request, checks that all matching donors were
alerted together, and accepts three of them. It then saves the API responses to `reports/demo_recruitment/`.

## Workflow

### Doctor

1. Click **Raise Blood Request** and enter:
   * blood group;
   * units needed;
   * **arrival time limit** (30 min to 24 h, or a custom 10–2880 min);
   * alert radius (5–100 km, or any distance reachable in time);
   * the contact number donors should call;
   * optionally, whether to also alert RBC-compatible groups (for example O−/O+ for an O+ request).
2. Set the hospital location in one of four ways:
   * search the blood-bank directory (you must confirm the entry, since the data is from 2015);
   * search an address (Nominatim);
   * use your GPS;
   * click the map.
3. The request is **broadcast to every matching donor at once**. A donor is alerted when they:
   * have a matching blood group;
   * are available;
   * are outside their deferral interval;
   * are not committed to another request;
   * have a shared location inside the radius;
   * have a road ETA within the time limit.
   Donors who register, share a location or switch alerts back on while the request is open are alerted
   automatically (checked every 30 s, and immediately when they share a location).
4. **View donors** opens the live request view:
   * countdown, units, alerted/waiting/declined/accepted counts, and a map.
   * **Accepted donors:** name, phone (tap to call), email, last donation date, total donations, accept time,
     live road ETA and expected arrival, distance, and how fresh their location is. Each card has a route
     button and a button to confirm the donation.
   * **All alerted donors:** ranked list with distance, ETA, likelihood to donate, match score and status.
     Contact details appear only after a donor accepts.
5. Confirming a donation records it, starts the donor's deferral and closes the request when all units are
   donated. Once enough donors have accepted, alerts that are still waiting close. If an accepted donor
   withdraws, they reopen. A request expires at its time limit, and the doctor can cancel it at any time.

### Donor

* Registration asks for:
  * name, mobile number (required) and email (optional);
  * blood group;
  * previous donations (count, last date, and first date if more than one);
  * optional location, by GPS or by picking a point on the map or searching an address.

  A self-reported donation inside the deferral interval (120 days for registered donors) keeps the donor
  from being alerted until it has passed.
* The portal shows each alert with:
  * hospital and address;
  * a countdown to the time limit;
  * the donor's own travel time;
  * the doctor's name and phone;
  * the number of units still needed;
  * the doctor's note.

  The donor can choose **I can donate**, **Can't come**, **Route**, or after accepting
  **I can no longer come**. New alerts can play a sound and show a browser notification after
  **Enable alert sound & notifications**.
* Accepting sends a fresh GPS fix when the browser allows. The ETA is then rechecked from where the donor is
  now, and the acceptance is refused if they can't arrive within the time limit.
* Location stays usable until the donor stops sharing it. The doctor sees whether it is live GPS (under 15 min
  old), an older GPS fix, or a point picked on the map. Turning on **Keep my GPS location updated** sends a
  fix at most once a minute while the page is open. There is no background tracking.
* Donors don't see the blood-bank directory or the system pages.

## Ranking: predicted likelihood to donate

The match score is `P(donate) × (1 − road ETA ÷ time remaining)`. Donors who can't arrive in time are not
alerted. `P(donate)` comes from the XGBoost donor-propensity model (`models/donor_propensity_xgb.json`,
scored in pure Python). It uses the donor's recency, frequency and time since first donation, taken from
the self-reported history plus donations recorded in the app. Donors with no history get the model's
training base rate and are shown as "no history".

Caveat: the model was trained on the UCI Blood Transfusion Service Center data, whose label is "donated
again in a given month". It is a proxy for "will respond to an emergency alert", not a measurement of it.
Nested cross-validation gives ROC-AUC 0.749 ± 0.036 on that data, the same as logistic regression. Retrain
with `python experiments/exp11_xgboost_donor_propensity.py`. All alerted donors are invited regardless of
score; the score only orders the doctor's list.

## Blood-bank directory

The doctor's **Blood Banks** tab covers the September 2015 directory: 2,947 facilities, of which 898 have
coordinate pairs passing basic numeric checks. You can:

* search by name, city, district, address or pincode;
* filter by state, or to entries that have a map location;
* page through 25 rows at a time on a map;
* use any entry with a location as the hospital for a request.

The directory contains no donor records or stock levels, and its positions are unverified. To repeat the
import, run `python tools/import_blood_banks.py PATH_TO_XLS` (requires `xlrd`).

## Notifications

Alerts appear in the donor portal, polled every 4 s. A browser notification and sound also play after the
donor opts in. External push/SMS uses the optional webhook outbox: set both `NOTIFICATION_WEBHOOK_URL` (HTTPS)
and `NOTIFICATION_WEBHOOK_TOKEN`. The gateway must map donor IDs to opted-in devices and deduplicate
`Idempotency-Key`. No gateway is configured by default.

## API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/auth/register`, `/api/v1/auth/login` | Donor registration; login |
| GET/POST | `/api/v1/donor/profile` | Profile with last donation; update phone, email, availability |
| POST | `/api/v1/donor/location`, `/api/v1/donor/history` | Share/revoke location; update donation history |
| GET | `/api/v1/donor/invitations` | Alerts with hospital, doctor contact, countdown |
| POST | `/api/v1/donor/invitations/respond` | `ACCEPT` (optional `location`), `DECLINE`, `WITHDRAW` |
| POST/GET | `/api/v1/emergency/requests` | Raise a request (`time_limit_minutes`, `search_radius_km`, `include_compatible`, …); list |
| POST | `/api/v1/emergency/requests/cancel` | Cancel an open request |
| GET | `/api/v1/emergency/recommendations?request_id=` | Ranked alerted donors, accepted donors with contact, summary |
| GET | `/api/v1/emergency/routes?donor_id=&request_id=` | Road route with encoded polyline |
| POST | `/api/v1/emergency/donation/complete` | Record a donation |
| GET | `/api/v1/facilities` | Directory: `search`, `state`, `has_coordinates`, `near=lat,lon`, `page`, `page_size` |

The legacy `urgency` field (`CRITICAL`/`URGENT`/`SCHEDULED`) is still accepted and maps to a 60/120/240-minute
time limit.

## Verification

```bash
python -m pytest tests -q
```

The suite covers:

* broadcast recruitment, time limits, overbooking and withdrawal;
* routing (including OSRM against a local fake server, the fallback and the circuit breaker);
* registration, contact and location validation, and access control;
* persistence;
* the research modules.

Historical experiment outputs in `reports/` and notes in `docs/` are retained; see `configs/` and
`experiments/` for the research pipeline. This is not a clinical system. It has no identity verification,
clinical screening or production account recovery.
