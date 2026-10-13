# Emergency Blood Donor Recommendation Demo

A Python recommendation and research engine with a Flutter donor portal and hospital dispatch desk. The demo uses synthetic travel estimates and exact blood-group matching. Response propensity and deferral settings are simulated research inputs.

## Start the Full Live Version (Windows)

To run the complete production-grade system without synthetic wave donors, with real donor registration, live UDI mathematical engine, UCI XGBoost return-propensity model, and the integrated web dashboard:

```powershell
.\start-full.ps1
```

Open **http://127.0.0.1:8000**. The launcher starts the full live API server with persistent state (`data/runtime/full_system_state.json`), serves the web dashboard, and automatically opens your browser.

## Start the browser demo (Windows)

From PowerShell:

```powershell
.\start-demo.ps1
```

Open **http://127.0.0.1:8000**. The demo script populates synthetic donor wave accounts for test simulations. Stop with Ctrl+C.

If another server uses port 8000, use `.\start-demo.ps1 -Port 8010` and open **http://127.0.0.1:8010**. For native clients on this port, supply `--dart-define=API_BASE_URL=http://10.0.2.2:8010` on Android emulators.

For a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd mobile
flutter pub get
cd ..
.\start-demo.ps1 -Rebuild
```

## Demonstrate the complete workflow

### Registered donors and location sharing

The initial **Sign In** screen serves donors and hospital staff. Choose **Register as a donor** to create a donor account with a username, password and self-reported blood group. Registered accounts require a password; the labelled demo accounts retain username-only login. Set `ALLOW_DEMO_LOGIN=false` to disable demo account login. This is still a local demonstration, without production account recovery, identity verification or clinical screening.

In the **Donor Portal**, consent to location sharing and choose **Share / refresh current location**. The app requests device permission and saves the latest latitude, longitude, accuracy and capture time in the configured runtime state file. Registration does not modify the historical facility spreadsheet or automatically add records to an ML training dataset. Denied permissions leave the donor outside nearby matching. Registered donors require a location captured within 15 minutes and accuracy within 250 metres. A 50 km straight-line prefilter precedes routing. Acceptance captures a new foreground fix and rechecks travel feasibility. **Stop sharing and remove saved location** clears the saved coordinates and makes the donor unavailable. No background tracking is enabled. Browser location capture requires localhost or HTTPS; Android and iOS require device permission. Real device capture still needs verification on the target device.

Hospital staff can search the imported **September 2015 blood-bank directory** while creating a request. It contains 2,947 facilities, of which 898 have coordinate pairs passing basic numeric checks. Entries with missing coordinates cannot be selected for routing. Facility names, operations and map positions must be checked before use. This directory contains neither individual donor records nor live blood-stock quantities. Its original publication/license has not been independently verified. To repeat the import, install `xlrd` and run `python tools/import_blood_banks.py PATH_TO_XLS`.

External delivery has an optional persistent webhook outbox. Configure both `NOTIFICATION_WEBHOOK_URL` (HTTPS) and `NOTIFICATION_WEBHOOK_TOKEN` to connect an operator-controlled push/SMS gateway. The gateway must map donor IDs to opted-in devices or addresses, deduplicate the invitation's `Idempotency-Key`, and deliver the notification. Attempts retry up to three times, subject to invitation expiry. `GATEWAY_ACCEPTED` means HTTP acceptance, not confirmed device delivery. No gateway is configured or externally verified in this demo. Without it, invitations remain in-app only.

Recruitment now uses road-time-ranked invitation waves of **10 donors**. Each wave has a response window (60 seconds by default). The server expires unanswered invitations and contacts the next eligible donors automatically, even when the app is closed. An accepted invitation reserves one outstanding unit; pending invitations are cancelled once enough donors accept. Completed donations are recorded separately.

The **Donor Recruitment** screen shows batch number, accepted/pending counts, outstanding demand, response deadline, and separate **All candidates** / **Accepted donors** views. Both views support pages of 10; browsing pages never sends invitations. Donor and recruitment screens refresh every five seconds. Accepted donors are ordered by refreshed road ETA, with UDI as a tie-breaker. ETA snapshots are reused in memory for up to 15 seconds during polling; acceptance and new waves recheck routes.

For a new multi-batch demo with 30 additional synthetic O+ donors and a ten-second response window:

```powershell
.\start-demo.ps1 -Rebuild -Port 8012 -ResponseWindowSeconds 10 -DemoDonors 30 -StateFile data/runtime/new_waves_demo.json
```

Existing state files retain their donor population. A new state filename gives you the additional demo accounts (`donor_003` through `donor_032`). They have explicitly synthetic locations and names. Sign in as any donor ID to view and respond to their in-app invitations.

To reproduce the automated two-batch scenario against that running server:

```powershell
.\.venv\Scripts\python.exe tools/demo_recruitment.py --base-url http://127.0.0.1:8012
```

It creates a three-unit request, accepts two donors from the first batch, waits for the server to invite the next ten, and accepts a third donor from batch two. It saves the actual API responses and results in `reports/demo_recruitment/`. Use a fresh state file to repeat independently of previous reservations.

Invitations currently appear **inside the donor portal**. No push notification or SMS delivery is implemented. Location checks use the saved donor coordinates; this change does not add live GPS collection. Without Google credentials, travel times remain synthetic. The batch logic uses the configured routing provider and excludes failed or late routes before inviting donors.

1. Sign in as `staff_001`. Create an **O+**, **1 unit**, **CRITICAL** request. Open its recruitment screen: the nearest reachable matching donors appear. Confirmation is disabled until acceptance.
2. Log out, then sign in as one of the invited donor IDs. Accept the invitation for the newly created request. Open **View Route** to see the key-free route preview.
3. Log out, sign in as `staff_001`, and reopen that request. Confirm the completed donation. The request shows **FULFILLED** and **1 / 1** units. Further confirmations are rejected.
4. Sign in as that donor again. The profile shows the demo deferral interval and updated donation count.

`donor_002` is an A+ donor for A+ requests. Login is deliberately a username-only demo entry point. These accounts do not represent real people.

Backend requests, invitations, alert counts, and donation records persist in `data/runtime/demo_state.json`; sessions expire when the server restarts. To start a separate clean demo without overwriting existing activity:

```powershell
.\.venv\Scripts\python.exe -m src.api.server --state-file data/runtime/fresh_demo.json
```

The login screen's **Offline Mock Mode** runs without the backend. It retains actions across refreshes and role changes within the app session; browser reload resets offline data. Offline scores and ETAs are fixed illustrative values. Live backend mode runs the UDI engine and synthetic routing service.

## Donor return-propensity model (XGBoost)

The API attaches an XGBoost estimate of each donor's likelihood to donate again to donor profiles
(`donation_propensity`) and to recruitment candidate rows. It is trained on the real **UCI Blood
Transfusion Service Center** data: 748 donors from Hsin-Chu City, Taiwan, with the label "donated in
March 2007". The inputs are recency, frequency and tenure in months. The earlier Kaggle "availability"
data was dropped because every model scored chance-level ROC-AUC (0.47–0.51) on it.

* Retrain and regenerate the report: `python experiments/exp11_xgboost_donor_propensity.py`
  (about 80 seconds on 4 cores). Results go to `reports/xgboost_propensity/`, `visuals/xgboost_propensity/`
  and `notes/comparison-xgboost-propensity.md`.
* Nested 5×5 cross-validation gives ROC-AUC **0.749 ± 0.036** and a Brier score of 0.153, against 0.181 for
  the base rate. That is the same as logistic regression (p = 0.77). The repo's previous XGBoost
  defaults overfit (ROC-AUC 0.680).
* The artefact `models/donor_propensity_xgb.json` contains the trees, the Platt calibration and the
  provenance. The API scores it in pure Python, so the server needs neither xgboost nor a pickle.
  Override the path with `DONOR_PROPENSITY_MODEL`.
* Inputs come from in-app completed donations and from optional self-reported history.
  Registration accepts `prior_donation_count`, `months_since_last_donation` and
  `months_since_first_donation`; `POST /api/v1/donor/history` updates them. Donors with no donation
  receive no estimate, because the UCI population has at least one donation each. Demo accounts carry
  labelled synthetic histories.
* The value is **advisory**. It predicts voluntary return donation, not emergency-invitation response,
  so it does not change UDI scores or invitation order. The Flutter screens do not display it yet.

## Mobile and routing configuration

```powershell
cd mobile
flutter run --dart-define=MOCK_MODE=true
# Android emulator connecting to the API (automatically uses 10.0.2.2:8000):
flutter run
# Custom backend:
flutter run --dart-define=API_BASE_URL=http://YOUR_HOST:8000
```

The browser uses its page origin by default. Native apps use the Android emulator URL or loopback on other platforms. For another device, bind the backend with `--host 0.0.0.0` and supply the computer's LAN address. Use a trusted local demonstration network.

No Maps key is required for the default coordinate/ETA preview. To enable Android interactive Maps, set `ANDROID_MAPS_API_KEY` in the build environment or `MAPS_API_KEY` in `mobile/android/local.properties`, then run with `--dart-define=MAPS_ENABLED=true`. For iOS, configure the native Google Maps SDK before enabling maps. Browser Maps require a Maps JavaScript SDK key/script in `mobile/web/index.html`; the default web demo leaves Maps disabled.

The backend honors `ROUTING_PROVIDER` and `GOOGLE_MAPS_API_KEY` from environment variables. `.env.example` documents them; `.env` files are not loaded automatically. Synthetic routing supplies estimated duration and distance with no fabricated road polyline. Server Routes keys are separate from client Maps SDK keys.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.test-tmp
cd mobile
flutter analyze
flutter test
flutter build web --no-web-resources-cdn
```

The Python suite covers research modules, routing, validation, access control, invitation transitions, duplicate prevention, closure and persistence. Mobile tests cover serialization, screens, polylines and the complete offline workflow.

Historical experiment outputs in `reports/` and notes in `docs/` are retained. See `configs/` and `experiments/` for the research pipeline. This local demo does not provide push notifications, production identity management, clinical screening, or deployment infrastructure.
