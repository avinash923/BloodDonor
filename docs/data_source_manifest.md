# Data Source Manifest

## 1. UCI Blood Transfusion Service Center Data
*   **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center
    (Yeh, Yang & Ting, Expert Systems with Applications, 2008, doi:10.1016/j.eswa.2008.07.018).
*   **Population:** 748 donors from the Blood Transfusion Service Center, Hsin-Chu City, Taiwan.
*   **Target:** donated (binary indicator for donation in March 2007); 178 positives (23.8 %).
*   **Path:** data/raw/uci/transfusion.data (committed; 748 rows, header row, LF line endings).
*   **Retrieval:** the UCI host is blocked from the build environment, so the copy was taken from
    PyCaret's public mirror (https://raw.githubusercontent.com/pycaret/datasets/main/data/common/blood.csv).
    `src/data/uci_loader.py` accepts any source only if it has 748 rows, 178 positives and content
    SHA-256 0c5bfc6d287ced28e723cf1485d9d73f3c38c7c26e524b756a5b6ac2b29274be (parsed int64 values,
    independent of header text and line endings). File SHA-256:
    85663c7d8af3bedf4a7b9ae788c41c4e6509a7af01e259b1cd9dc36e6065231c.
*   **Status:** Used by EXP-11 to train the deployed XGBoost donor return-propensity model
    (models/donor_propensity_xgb.json). See reports/xgboost_propensity/experiment_report.md.

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
*   **Outcome:** held-out ROC-AUC 0.47–0.51 for all four models (reports/tables/availability_model_metrics.csv);
    the label is not predictable from these features. models/best_availability_model.pkl is retained
    only as a historical artefact and is not used by the application.
