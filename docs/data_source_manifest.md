# Data Source Manifest

## 1. UCI Blood Transfusion Service Center Data
*   **Source:** UCI Machine Learning Repository
*   **Target:** donated (binary indicator for donation in March 2007)
*   **Status:** Pilot modeling phase complete.

## 2. Kaggle Availability Dataset (Phase 1 Restart)
*   **Source:** Kaggle (Obtained via user)
*   **Path:** data/raw/blood_donor_dataset_modeling.csv
*   **Target:** vailability (General availability classification, mapped to binary 1=Yes, 0=No)
*   **Selected Features:** city, lood_group, months_since_first_donation, 
umber_of_donation, pints_donated.
*   **Excluded Features:** donor_id (identifier), created_at (unverified time-series relevance).
*   **Forbidden Features (PII):** 
ame, email, password, contact_number (Removed upstream, validated via loader).
*   **Checksum (SHA256):** d3d1a7e301de160b547816e41fcf61880817e5125bbc4b4364e49adcc3eb02a3
*   **Target Uncertainty:** The vailability flag indicates general willingness but does not equate to time-pressured emergency recruitment probability. It must be synthesized or scaled during Phase 4 dynamic simulations.
*   **Seed:** 42 (used for CV splitting and preprocessing).
