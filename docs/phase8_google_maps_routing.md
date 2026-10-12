# Phase 8 — Google Maps Platform Routes API Integration

**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System  
**Repository:** `C:\BloodDonor`  
**Phase:** 8 (Routes API Integration & Unified Spatial Provider Architecture)

---

## 1. Overview & Architecture

Phase 8 connects the existing emergency donor recommendation and dynamic recruitment engine to road travel-time estimates via the **Google Maps Platform Routes API** while maintaining 100% backward-compatible, offline, deterministic synthetic simulation capabilities.

The spatial layer decouples routing providers through an abstract interface:
- **`SyntheticRoutingProvider` (Default):** Uses the project's local urban grid metrics and 24-hour congestion multipliers (`κ_traffic(t)`). Requires no API key, runs completely offline, incurs zero billing costs, and preserves seeded research reproducibility.
- **`GoogleMapsRoutesProvider`:** Production-ready client for Google Maps Platform Routes API v2:
  - Multi-candidate matrix evaluation via `computeRouteMatrix` (`https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix`).
  - Detailed single-route and polyline calculation via `computeRoutes` (`https://routes.googleapis.com/directions/v2:computeRoutes`).
- **`RoutingService`:** High-level coordinator that handles configuration switching (`ROUTING_PROVIDER`), coordinate validation, batching, and populating `DonorCandidate.travel_time_minutes` for the UDI ranking engine.

```
+-------------------------------------------------------------------------------+
|                             Emergency Blood Request                           |
|                        (Hospital Lat/Lon, Urgency Horizon)                    |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                                RoutingService                                 |
|         - Validates Donor Lat/Lon Coordinates in [-90, 90], [-180, 180]       |
|         - Chunks Candidates into Batches (<= 100 origins / call)              |
+-------------------------------------------------------------------------------+
                 |                                              |
     [ROUTING_PROVIDER=synthetic]                   [ROUTING_PROVIDER=google]
                 |                                              |
                 v                                              v
+-----------------------------------+       +-----------------------------------+
|      SyntheticRoutingProvider     |       |      GoogleMapsRoutesProvider     |
| - Deterministic Local Grid Engine |       | - Routes API (computeRouteMatrix) |
| - 24-Hour Traffic Congestion κ(t) |       | - Live Traffic-Aware Duration     |
| - Offline & Seed Reproducible     |       | - Safe Failure Handling (None)    |
+-----------------------------------+       +-----------------------------------+
                 \                                              /
                  \                                            /
                   v                                          v
+-------------------------------------------------------------------------------+
|                       RouteEstimate (Typed Domain Model)                      |
|      - duration_minutes (float, minutes) or None if unreachable               |
|      - NEVER converts failure or missing route to 0.0 minutes                 |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                        UDIEngine.score_and_rank()                             |
|      - Consumes travel_time_minutes in [0, T_eff]                             |
|      - None ETA -> UDI = 0.0, is_feasible = False (Hard Gate)                 |
+-------------------------------------------------------------------------------+
```

---

## 2. Official Google Maps Routes API Methods Used

### A. Compute Route Matrix (`distanceMatrix/v2:computeRouteMatrix`)
- **Endpoint:** `POST https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix`
- **Purpose:** Computes travel times and distances for multiple candidate donors travelling to a single emergency hospital site.
- **Request Headers:**
  - `Content-Type: application/json`
  - `X-Goog-Api-Key: <GOOGLE_MAPS_API_KEY>`
  - `X-Goog-FieldMask: originIndex,destinationIndex,status,condition,distanceMeters,duration,staticDuration`
- **Request Payload Structure:**
  ```json
  {
    "origins": [
      { "waypoint": { "location": { "latLng": { "latitude": 12.9716, "longitude": 77.5946 } } } }
    ],
    "destinations": [
      { "waypoint": { "location": { "latLng": { "latitude": 13.0012, "longitude": 77.6105 } } } }
    ],
    "travelMode": "DRIVE",
    "routingPreference": "TRAFFIC_AWARE",
    "departureTime": "2026-10-11T14:30:00Z"
  }
  ```
- **Response Parsing:**
  - `distanceMeters`: Parsed as road distance in meters (converted to km when required).
  - `duration`: Protobuf duration string (e.g., `"1845s"`) parsed into float seconds (`1845.0s`), then converted to minutes (`30.75 min`) for direct consumption by `UDIEngine`.
  - `staticDuration`: Preserved for baseline non-traffic comparison.
  - `status.code`: Must equal `0` (OK); non-zero codes (or `ROUTE_NOT_FOUND`) are marked `is_success=False` with `duration_minutes=None`.

### B. Compute Routes (`directions/v2:computeRoutes`)
- **Endpoint:** `POST https://routes.googleapis.com/directions/v2:computeRoutes`
- **Purpose:** Used when individual detailed turn-by-turn routes or visual polylines are requested for front-end rendering.
- **Field Mask:** `routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline`
- **Returns:** Encoded polyline string suitable for rendering in the Phase 9 Flutter map interface.

---

## 3. Strict Failure & Safety Policy (Zero-ETA Prevention)

In an emergency blood recruitment system, **an unknown travel time must NEVER be treated as zero travel time**. 

1. **Explicit Infeasibility:** If a route lookup fails, times out, encounters a network error, or has no valid road connection (`ROUTE_NOT_FOUND`), the adapter sets `is_success=False` and `duration_minutes=None`.
2. **Hard Isochrone Gate:** In `UDIEngine.score_candidate()`, when `travel_time_minutes is None`, the donor candidate receives:
   - `udi_score = 0.0`
   - `is_feasible = False`
   - `rank = 0` (ineligible for dispatch)
3. **No Artificially Favorable Scores:** A failed route lookup can never result in an artificially high timeliness score or accidental recruitment dispatch.

---

## 4. Security, Credentials & Quota Safeguards

1. **Environment-Variable Only:** The server-side API key is read exclusively from `GOOGLE_MAPS_API_KEY`.
2. **Never Committed to Git:** `.env.example` contains only placeholder values. No real keys exist in code, logs, or reports.
3. **Log Masking:** `GoogleMapsRoutesProvider` and `RoutingService` log operational metrics and donor IDs, but **never** log API keys, request headers, or full coordinate payloads.
4. **Batch Limits:** Candidates are chunked into batches of at most 100 origins per request, well within Google's 625 element limit per call.
5. **Bounded Exponential Backoff:** Transient 5xx server errors and network dropouts retry at most twice with exponential backoff (0.5s, 1.0s). 4xx client errors (e.g. 400 Invalid Argument, 403 Forbidden) fail immediately without retrying.

---

## 5. Google Cloud Console Setup Guide

To enable live Google Maps routing in production or field trials:

1. **Create or Select a Google Cloud Project:**
   - Navigate to the [Google Cloud Console](https://console.cloud.google.com/).
   - Link an active billing account.
2. **Enable the Routes API:**
   - Go to **APIs & Services > Library**.
   - Search for **Routes API** (Ensure you select "Routes API", not the legacy "Directions API" or "Distance Matrix API").
   - Click **Enable**.
3. **Create and Restrict Credentials:**
   - Go to **APIs & Services > Credentials > Create Credentials > API Key**.
   - Under **API restrictions**, select **Restrict key** and choose only **Routes API**.
   - Under **Application restrictions**, set server IP restrictions matching your deployment servers.
   - **CRITICAL:** Do NOT reuse this secret server key in client-side Flutter applications.
4. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env and set:
   ROUTING_PROVIDER=google
   GOOGLE_MAPS_API_KEY=your_actual_restricted_api_key_here
   ```
5. **Set Quotas and Budget Alerts:**
   - Set daily request quotas under **APIs & Services > Routes API > Quotas** to prevent unexpected billing.

---

## 6. Offline Testing & Research Reproducibility

To ensure existing research experiments remain 100% reproducible and offline-safe:
- By default, `ROUTING_PROVIDER=synthetic`.
- Running `pytest` runs the full test suite completely offline using mocked responses for Google Routes calls and local computations for synthetic calls.
- Historical experimental datasets (Phase 4: 3,500 rows, Phase 5: 3,000 rows, Phase 6.2: 3,108 rows, Phase 6.3: 16,340 rows) remain completely intact and untouched.

---

## 7. Automated Test Suite Results

- **Routing Provider Tests (`tests/test_google_maps_routing.py`):** **12 passed, 0 failed** in 0.28s.
- **Full Regression Test Suite:** **266 passed, 0 failed** across all 17 test modules in 6.46s.
