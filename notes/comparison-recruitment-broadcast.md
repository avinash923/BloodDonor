# Recruitment, routing and real-time design choices (October 2026)

Context: the requested changes were:

* send a doctor's request to all donors;
* let the doctor choose the arrival time limit instead of an urgency tier;
* rank nearby donors by an XGBoost prediction;
* show accepted donors' contact details and last donation;
* use free map APIs;
* work with real (not seeded) data.

## 1. Who receives the request

| Option | Behaviour | Pros | Cons |
|---|---|---|---|
| A. Waves of 10 (previous) | Nearest 10, next 10 after a 60 s window | Limits alert fatigue | Slow when most donors ignore alerts; contradicts "send to all" |
| **B. Broadcast (chosen)** | Every feasible donor at once; sweep for late joiners | Fastest fill; matches the request | More alerts per donor; needs fill-and-reopen logic |
| C. Hybrid | Broadcast to the top-K by score, then everyone after N seconds | Balances speed and fatigue | Extra tuning; harder to explain to staff |

"Feasible" means:

* compatible blood group;
* available;
* outside the deferral interval;
* not committed elsewhere;
* inside the radius;
* road ETA within the time remaining.

Donors who cannot arrive in time are never alerted, which keeps the broadcast from becoming national spam.
Once enough donors accept, pending alerts close; a withdrawal re-opens them.

Next experiment: replay `reports/phase4_kaggle_comparison` scenarios through B vs C. Measure time-to-fill
against alerts per fulfilled unit.

## 2. Time limit instead of urgency tier

The doctor picks the minutes (10–2880). They drive three things:

* invitation expiry;
* the straight-line prefilter (`time × 100 km/h`);
* the UDI timeliness term, using a per-request horizon with no safety margin.

Legacy `urgency` values map to 60/120/240 minutes, so old clients still work.

## 3. Ranking

`match = P(donate) × (1 − ETA / time remaining)`. This is the existing multiplicative UDI with
`P = XGBoost estimate`, fatigue switched off (everyone is alerted anyway) and no safety margin.

Alternatives considered:

* **Additive UDI:** less intuitive; a near donor with zero likelihood still scores well.
* **Lexicographic (ETA first, likelihood as tie-break):** simple, but ignores "who has given more blood".

Caveat (also in the README): the UCI label is voluntary return donation, not emergency response.
Donors with no history get the training base rate rather than 0, so new registrations are not buried.

## 4. Free routing and maps

| Option | Notes |
|---|---|
| **OSRM public server (chosen default)** | No key. `/table` gives batch ETAs and `/route` gives the polyline. Free-flow only; demo server is rate-limited, so self-host for production |
| GraphHopper / OpenRouteService | Free tiers need an API key |
| Valhalla self-hosted | Good alternative if OSRM self-hosting is chosen anyway |
| Browser-side OSRM calls | Would leak donor coordinates to the client before acceptance; rejected |

Safeguards:

* a 60 s result cache;
* a 60 s circuit breaker, because the server is single-threaded and a timeout on every poll would freeze it;
* a synthetic fallback labelled `synthetic_fallback`, which the UI shows as "Approximate".

Address search uses Nominatim on button press only, per its usage policy. Tiles are OpenStreetMap via Leaflet.

## 5. Real-time delivery

Polling (4 s donor, 3–5 s doctor) was kept over Server-Sent Events. SSE would need `ThreadingHTTPServer`,
and several GET paths read shared dicts without the lock. All GET handling now runs under the state lock,
which makes a later move to SSE safer.

Next experiment: an SSE spike behind `ThreadingHTTPServer`, measuring the latency gain over 4 s polling.

## 6. Location freshness

The previous rule (registered donors needed a GPS fix under 15 minutes old) meant almost no donor was ever
matchable. Now the last consented location (GPS or map pin) is usable until revoked. Its age and source are
shown to the doctor. Acceptance sends a fresh fix and re-checks the ETA.
