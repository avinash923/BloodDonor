# Phase 3 — Integration Report: Kaggle Dataset to Synthetic Emergency Recruitment Simulator

**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System  
**Phase:** 3 — Dataset Integration & Simulation Adapter  
**Date:** October 11, 2026  
**Status:** Complete  

---

## Executive Summary

Phase 3 establishes the bridge between the cleaned Kaggle blood donor dataset (`data/raw/blood_donor_dataset_modeling.csv`) and the synthetic emergency recruitment simulator. The implementation adheres strictly to empirical model-selection boundaries, privacy principles (zero PII), explicit data provenance, and leakage-free ranking mechanics.

---

## 1. Task 1: Dataset Verification & Model-Selection Boundary

### 1.1 Dataset Verification
- **Path:** `data/raw/blood_donor_dataset_modeling.csv`
- **SHA-256 Checksum:** `d3d1a7e301de160b547816e41fcf61880817e5125bbc4b4364e49adcc3eb02a3` (matches Phase 1 manifest)
- **Total Records:** 10,000 rows
- **Approved Columns:**
  1. `city` (categorical, 8 Australian cities)
  2. `blood_group` (categorical, 8 ABO/Rh types: A+, A-, B+, B-, AB+, AB-, O+, O-)
  3. `months_since_first_donation` (integer)
  4. `number_of_donation` (integer)
  5. `pints_donated` (integer)
  6. `availability` (binary: 0 or 1)

### 1.2 Phase 2 Model-Selection Audit
- **Winning Model:** XGBoost
- **Training 5-fold CV ROC-AUC:** 0.4959
- **Held-out Test ROC-AUC:** 0.4731
- **Empirical Implication:** The Phase 2 classification benchmark demonstrates that the feature set available in the Kaggle dataset (`city`, `blood_group`, `months_since_first_donation`, `number_of_donation`, `pints_donated`) provides little to no predictive discrimination for the recorded `availability` target.
- **Architectural Decision:** 
  - The Phase 2 availability classifier output is **NOT** used as a recruitment-response probability.
  - We do **not** claim availability is proven to be theoretical random noise, but rather acknowledge that the tabular features do not support predictive modeling of operational responsiveness.
  - All Phase 2 artifacts and benchmarks remain strictly preserved.

---

## 2. Task 2: Dataset-to-Simulation Adapter

### 2.1 Implementation
- **Module:** `src/data/kaggle_simulation_adapter.py` (`KaggleSimulationAdapter`)
- **Loader:** Integrates seamlessly with `src/data/kaggle_loader.py` (`load_modeling_dataset`).

### 2.2 Privacy & PII Exclusion
- Zero direct identifiers are used or loaded: `name`, `email`, `password`, `contact_number`, and original `donor_id` are strictly excluded.
- Every simulation profile is assigned a newly generated synthetic identifier (e.g., `DONOR_KAG_000001`).

### 2.3 Field Provenance Mapping

| Field | Source Type | Description |
|---|---|---|
| `city` | Kaggle Dataset | Real demographic municipality (8 cities) |
| `blood_group` | Kaggle Dataset | ABO/Rh blood type (used in clinical compatibility filter) |
| `months_since_first_donation` | Kaggle Dataset | Donor history duration |
| `number_of_donation` | Kaggle Dataset | Historical donation frequency |
| `pints_donated` | Kaggle Dataset | Total volume donated |
| `availability` | Kaggle Dataset | Historical self-reported willingness label |
| `donor_id` | Synthetic | Anonymized stable identifier (`DONOR_KAG_xxxxxx`) |
| `latitude`, `longitude` | Synthetic | Bounded within realistic city bounding boxes |
| `is_eligible` | Synthetic | 56-day inter-donation deferral window flag |
| `simulated_current_availability`| Synthetic | Dynamic operational readiness for current emergency |
| `synthetic_response_propensity` | Synthetic (Hidden) | Ground-truth propensity $\text{Beta}(2,5)$ for event simulation |
| `recent_alert_count` | Synthetic | Tracked recruitment fatigue count |

---

## 3. Task 3 & 5: Availability Semantics & Leakage Prevention

### 3.1 Three Distinct Availability / Response Concepts
1. **Historical Dataset Availability (`donor.availability`):** Recorded willingness in static CSV. Preserved solely as an audit and feature attribute; not assumed to represent real-time emergency readiness.
2. **Operational Simulated Availability (`simulated_current_availability`):** Simulated Bernoulli trial ($p=0.70$) representing whether the donor is reachable and able to answer the phone right now.
3. **Synthetic Response Propensity (`synthetic_response_propensity`):** Continuous parameter $\sim \text{Beta}(2,5)$ (mean $\approx 0.286$) controlling acceptance probability within `StochasticResponseModel`.

### 3.2 Leakage Prevention
- `synthetic_response_propensity` is **strictly hidden simulation truth**. It is never exposed to the UDI ranking formula, baseline policies, or eligibility filters.
- The UDI engine ranks donors based solely on observed features: travel time feasibility, fatigue discount, eligibility, compatibility, and donor history.

---

## 4. Task 4 & 6: Synthetic Spatial Engine & Recruitment Architecture

### 4.1 Geographic Bounding Boxes
Because the Kaggle dataset lacks geographic coordinates, synthetic coordinates are sampled uniformly within defined bounding boxes around the donor's recorded city:
- **Sydney:** Lat $[-33.95, -33.80]$, Lon $[151.10, 151.30]$
- **Melbourne:** Lat $[-37.85, -37.70]$, Lon $[144.90, 145.05]$
- **Brisbane:** Lat $[-27.52, -27.40]$, Lon $[153.00, 153.15]$
- **Perth:** Lat $[-32.00, -31.85]$, Lon $[115.80, 115.95]$
- **Adelaide:** Lat $[-35.00, -34.85]$, Lon $[138.55, 138.70]$
- **Hobart:** Lat $[-42.92, -42.83]$, Lon $[147.28, 147.38]$
- **Darwin:** Lat $[-12.48, -12.40]$, Lon $[130.83, 130.91]$
- **Canberra:** Lat $[-35.33, -35.22]$, Lon $[149.06, 149.18]$

### 4.2 Recruitment Architecture Preservation
The core engine modules were reused without mathematical alteration:
- Multiplicative UDI engine: $\text{UDI} = S_{\text{urgency}} \times S_{\text{timeliness}} \times S_{\text{compatibility}} \times S_{\text{reliability}} \times (1 - \text{fatigue\_penalty})$
- Dynamic cascade dispatching (`DispatchCascadePolicy`)
- Stochastic outcome resolution (`StochasticResponseModel`)
- Discrete-event simulation loop (`SimulationEpisode`)

---

## 5. Task 7 & 8: Pilot Execution and Automated Test Results

### 5.1 Test Suite (`tests/test_phase3_integration.py`)
A comprehensive test suite of 34 automated unit and integration tests was implemented and executed:
- **Test execution result:** **34 passed in 2.16s**
- **Categories covered:**
  1. CSV loading & column validation (3 tests)
  2. Profile mapping & schema compliance (4 tests)
  3. PII exclusion & ID synthesis (3 tests)
  4. Separation of availability concepts (3 tests)
  5. Synthetic coordinate provenance & city bounds (4 tests)
  6. Fixed-seed reproducibility (3 tests)
  7. Leakage prevention: response propensity hidden from ranking (3 tests)
  8. Full UDI engine & baseline compatibility (4 tests)
  9. Donor pool uniqueness & sampling (3 tests)
  10. Scenario serialization & manifest generation (4 tests)

### 5.2 Pilot Simulation Execution
- **Script:** `experiments/exp_phase3_pilot.py`
- **Donor Pool:** 200 unique Kaggle donor profiles sampled with seed 42
- **Scenarios Evaluated:** 20 emergency requests (CRITICAL: 3, URGENT: 10, SCHEDULED: 7)
- **Quota Met Rate:** 90.0% (18 / 20)
- **On-Time Quota Met Rate:** 90.0% (18 / 20)
- **Late Arrivals:** 0
- **Mean Waves Required:** 3.6
- **Mean Notifications:** 13.5

---

## 6. Artifact Manifest

| Artifact Path | Description |
|---|---|
| `src/data/kaggle_simulation_adapter.py` | Adapter converting Kaggle records to simulation `DonorProfile` |
| `src/data/kaggle_scenario_generator.py` | Integrated scenario set generator and manifest writer |
| `tests/test_phase3_integration.py` | 34 automated integration and validation tests |
| `experiments/exp_phase3_pilot.py` | Pilot experiment running end-to-end simulation |
| `reports/phase3_dataset_integration/dataset_profile_audit.json` | Profile distribution and provenance audit |
| `reports/phase3_dataset_integration/scenario_manifest.json` | Run metadata, seeds, bounding boxes, parameters |
| `reports/phase3_dataset_integration/sample_donor_profiles.csv` | Export of 200 simulation donor profiles |
| `reports/phase3_dataset_integration/sample_emergency_scenarios.csv` | Export of 20 generated emergency scenarios |
| `reports/phase3_dataset_integration/pilot_episode_results.csv` | Detailed episode results from pilot run |
| `reports/phase3_dataset_integration/pilot_summary.json` | Metric aggregates from pilot run |
| `reports/phase3_dataset_integration/integration_report.md` | This Phase 3 completion report |

---

## 7. Limitations and Assumptions

1. **Synthetic Spatial Assumption:** The Kaggle dataset only identifies municipality (`city`). All coordinates ($lat, lon$) are generated within realistic city bounding boxes. They do not represent real donor residential addresses.
2. **Synthetic Behavioral Dynamics:** The Kaggle `availability` feature does not reflect real-time responsiveness. Response propensities are synthetically generated under a calibrated $\text{Beta}(2,5)$ distribution.
3. **Simulation Boundary:** This simulation models recruitment dynamics and dispatch policies under simulated emergency conditions; it does not replace clinical donor eligibility screening or real-world dispatch decisions.
