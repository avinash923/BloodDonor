# Phase 9 — Flutter Mobile Application and Interactive Google Maps Integration

## 1. Executive Summary

Phase 9 integrates a mobile application front-end and lightweight REST API layer with the validated blood donor recommendation engine and Phase 8 routing service. 
The implementation connects donors and authorized hospital personnel to the backend system without altering the core UDI mathematical formulation, simulation runners, or historical experiment results.

### Core Achievements
- **Flutter Mobile Application**: Built a complete cross-platform Flutter application in `mobile/` with typed models, responsive UI, accessible feedback, and role-aware navigation.
- **Backend REST Gateway**: Implemented a lightweight, standard-library HTTP server in `src/api/server.py` exposing authorized donor, triage coordinator, UDI recommendation, and routing endpoints.
- **Interactive Google Maps Interface**: Integrated Google Maps (`google_maps_flutter`) rendering origin donor coordinates, emergency hospital sites, and decoded polyline routes received from the Phase 8 routing service.
- **Zero-ETA Safety Policy**: Enforced that unavailable or failed routes never report `0.0` minutes or false feasibility in the UI or API layers.
- **Offline / Mock Development Mode**: Built-in mock mode enables end-to-end development, testing, and UI demonstration without requiring live Google Maps API keys or generating external billable requests.
- **Automated Test Suites**: 
  - **Flutter Tests**: 16 unit and widget tests covering models, polyline decoders, bounding box calculations, screen states, and mock workflows.
  - **Backend Tests**: 273 Python tests passing with zero regressions across the research suite.

---

## 2. System Architecture & Workflow

```
[Emergency Alert] 
       │
       ▼
[Hospital Desk / Coordinator] (StaffDashboardScreen)
  - Submits blood type, unit demand & urgency tier
       │
       ▼
[Backend Engine] (src/api/server.py)
  - Biological Compatibility & Temporal Eligibility Filters
  - Multiplicative UDI Candidate Ranking (src/ranking/udi_engine.py)
  - Route Estimation via Routing Service (Synthetic or Google Routes)
       │
       ▼
[Recruitment & Notification Dispatch]
  - Hospital staff views ranked candidate cards with UDI scores
  - Donor receives push/in-app alert on DonorDashboardScreen
       │
       ▼
[Donor Response & Route Navigation]
  - Donor reviews triage urgency and accepts
  - Donor navigates via Interactive Google Maps Screen (MapRouteScreen)
       │
       ▼
[Donation Confirmation & Deferral Update]
  - Staff confirms clinical donation event
  - Temporal eligibility rule applies mandatory inter-donation deferral
```

---

## 3. Directory Structure

```
BloodDonor/
├── src/
│   └── api/
│       ├── __init__.py
│       └── server.py             # Python HTTP REST API Server
├── tests/
│   └── test_api_server.py        # API endpoint regression & safety tests
├── mobile/
│   ├── lib/
│   │   ├── config/
│   │   │   └── app_config.dart   # Base URL, timeouts, mock mode toggle
│   │   ├── models/
│   │   │   └── models.dart       # UserProfile, EmergencyRequestModel, RouteDetailModel
│   │   ├── services/
│   │   │   └── api_service.dart  # REST client with auth headers and timeout management
│   │   ├── utils/
│   │   │   └── polyline_decoder.dart # Encoded polyline decoder and bounds calculator
│   │   ├── screens/
│   │   │   ├── login_screen.dart             # Role selection & authentication
│   │   │   ├── donor_dashboard_screen.dart   # Profile, eligibility, alerts, accept/decline
│   │   │   ├── staff_dashboard_screen.dart   # Emergency triage desk & request creation
│   │   │   ├── recommendations_screen.dart   # Ranked UDI candidate list & donation confirmation
│   │   │   └── map_route_screen.dart         # Google Maps polyline & marker visualization
│   │   └── main.dart             # App bootstrap
│   ├── test/
│   │   ├── models_test.dart            # JSON deserialization & zero-ETA validation
│   │   ├── polyline_decoder_test.dart  # Polyline string decoding & LatLngBounds tests
│   │   ├── widget_screens_test.dart    # Role, donor, staff, and recommendation screen tests
│   │   └── widget_test.dart            # App root smoke test
│   └── android/
│       └── app/src/main/AndroidManifest.xml # Permissions & ${MAPS_API_KEY} meta-data
└── docs/
    └── phase9_flutter_mobile_integration.md
```

---

## 4. API Endpoints Contract

The backend REST API (`src/api/server.py`) operates over port 8000 by default and requires authorization tokens for protected resources:

| Endpoint | Method | Role | Description |
|---|---|---|---|
| `/api/v1/health` | GET | Public | Healthcheck and service status |
| `/api/v1/auth/login` | POST | Public | Authenticates donor or hospital coordinator |
| `/api/v1/donor/profile` | GET | Donor | Returns profile, availability, and temporal eligibility |
| `/api/v1/donor/invitations` | GET | Donor | Returns pending emergency alerts for authenticated donor |
| `/api/v1/donor/invitations/respond` | POST | Donor | Accepts or declines recruitment invitation |
| `/api/v1/emergency/requests` | GET, POST | Staff | Retrieves active emergencies or broadcasts a new request |
| `/api/v1/emergency/recommendations` | GET | Staff | Returns UDI-ranked candidates and travel times |
| `/api/v1/emergency/routes` | GET | Donor/Staff | Returns origin/destination coordinates and encoded polyline |
| `/api/v1/emergency/donation/complete` | POST | Staff | Confirms completed donation and enforces deferral |

### Security & Key Isolation
- The server-side Google Routes API key (`GOOGLE_MAPS_API_KEY`) is stored strictly on the server and is **never** sent to the mobile client.
- The mobile client Google Maps SDK key is restricted by Android package name (`com.example.blood_donor_app`) and SHA-1 fingerprint.

---

## 5. Mobile Application Screens

### 5.1 Welcome & Authentication (`LoginScreen`)
- Provides role-aware entry for **Registered Donors** and **Hospital Personnel**.
- Features quick demo accounts (`donor_001`, `staff_001`) for rapid demonstration.
- Includes a toggle for **Offline Mock Mode**, allowing standalone operation without backend dependencies.

### 5.2 Donor Dashboard (`DonorDashboardScreen`)
- Displays donor blood group, donor ID, and clinical eligibility badge.
- Shows temporal eligibility status (e.g. *Clinically Eligible to Donate* vs *In Temporary Deferral Interval*).
- Lists incoming emergency recruitment invitations with urgency tiers (`CRITICAL`, `URGENT`).
- Allows single-tap **Accept Alert** or **Decline**, updating invitation state immediately.

### 5.3 Emergency Request Desk (`StaffDashboardScreen`)
- Displays ongoing hospital triage demands and fulfilled units.
- Floating action button to broadcast a new emergency with target blood group, urgency level, and unit count.
- Direct navigation into the candidate recommendation engine.

### 5.4 UDI Recommendation & Recruitment (`RecommendationsScreen`)
- Lists donor candidates ranked by multiplicative UDI priority score.
- Displays donor rank, blood group compatibility, alert count, and road travel time.
- **View Map** button initiates interactive route visualization.
- **Confirm Donation** button logs clinical completion and triggers automated donor deferral.

### 5.5 Interactive Google Maps (`MapRouteScreen`)
- Places azure pin for donor location and red pin for emergency hospital site.
- Renders the decoded polyline route using `PolylineDecoder`.
- Handles infeasible or failed routes safely with "Route Infeasible" messaging (never default 0 min).

---

## 6. Google Maps SDK Setup Instructions

To enable live Google Maps rendering on Android:

1. **Google Cloud Console**:
   - Enable **Maps SDK for Android** in your Google Cloud Project.
   - Under **Credentials**, create an API key restricted to Android apps.
   - Specify package name `com.example.blood_donor_app` and your debug keystore SHA-1 certificate fingerprint.
2. **Environment / Build Setup**:
   - In `mobile/android/local.properties` or build arguments, provide:
     ```properties
     MAPS_API_KEY=AIzaSy...YourAndroidMapsKey...
     ```
   - In `mobile/android/app/src/main/AndroidManifest.xml`:
     ```xml
     <meta-data
         android:name="com.google.android.geo.API_KEY"
         android:value="${MAPS_API_KEY}"/>
     ```
3. **Running the App**:
   - For Android Emulator:
     ```bash
     cd mobile
     flutter run
     ```
   - Backend URL defaults to `http://10.0.2.2:8000` when running inside Android emulators.

---

## 7. Verification & Automated Test Results

### Flutter Test Suite
```bash
cd mobile
flutter test
```
- **Results**: 16 tests executed, **16 passed (100%)**.
- Tests cover `models_test.dart` (5 tests), `polyline_decoder_test.dart` (6 tests), `widget_screens_test.dart` (4 tests), and `widget_test.dart` (1 test).

### Backend Pytest Suite
```bash
pytest tests/
```
- **Results**: 273 tests executed, **273 passed (100%)** in 8.87 seconds.
- Validates that research modules (UDI engine, baselines, ML estimators, longitudinal fatigue, donation safety simulation) remain strictly intact.
