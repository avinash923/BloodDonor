# Phase 0: Kaggle Dataset Restart Audit

## 1. Execution Blocker: Resolved
**Status:** The expected dataset file `data/raw/blood_donor_dataset_modeling.csv` is now present in the workspace.
The dataset has been successfully audited. The raw file contains 10,000 rows and 6 columns. There are 0 missing values, and 37 duplicate rows. PII columns have been successfully excluded.

## 2. Dataset Schema & Privacy Audit

Based on the provided schema: donor_id, name, email, password, contact_number, city, blood_group, availability, months_since_first_donation, number_of_donation, pints_donated, created_at

### A. Privacy & Security Requirements
*   **Excluded Features:** 
ame, email, password, contact_number MUST be completely excluded from all modelling, logging, and generated outputs. These contain severe Personally Identifiable Information (PII) and security credentials.
*   **Dataset Handling:** The raw dataset must never be committed to Git. A .gitignore entry must be verified.
*   **Identifier Exclusion:** donor_id is an index/identifier and must be excluded from predictive features to prevent data leakage.

### B. Feature Selection & Semantics
*   **Target Variable:** vailability. This represents a general classification target. *Crucial Limitation:* This is not equivalent to an emergency recruitment response probability. It must not be treated as such during emergency simulations.
*   **Predictors:** months_since_first_donation, 
umber_of_donation, pints_donated (historical frequency/monetary), lood_group, city (categorical features).
*   **Optional Features:** created_at will be excluded unless its specific Kaggle documentation provides a valid time-series rationale.

## 3. Existing Codebase Integration & Reusability

### Retained Modules (Do not delete/rebuild)
*   **ML Wrappers (configs/ml_models.yaml, etc.):** Can be reused, but the target variable will switch from UCI's donated to the Kaggle vailability target.
*   **Spatial Feasibility Engine (TravelTimeEngine):** Fully reusable. However, because the Kaggle dataset lacks precise coordinates (only providing city), city-level centroid mapping or synthetic coordinate generation within bounds will be required for the simulator.
*   **UDI Ranking & Baselines (aselines.py, udi_engine.py):** Fully reusable. The probability output from the new Kaggle model will feed into these existing structures.
*   **Simulator & Evaluation (simulator.py, exp10...):** Fully reusable for Phase 4-style experiments.

## 4. Scientific Limitations & Experimental Segregation
*   **Label Mismatch:** The Kaggle target vailability does not equate to the time-pressured emergency response label required by the dynamic simulator.
*   **Phased Strategy:**
    *   *Sub-Experiment A:* Train the classification model on the Kaggle dataset to predict vailability.
    *   *Sub-Experiment B:* For the dynamic recruitment simulation, use the Kaggle donor pool as the base, but synthesize emergency response probabilities (using the calibrated model output as a propensity base) and synthetic emergency scenarios.

## 5. Next Steps
1.  **Resolve Blocker:** Place lood_donor_dataset.csv in data/raw/.
2.  **Verify Data:** Once the file is present, execute a statistical audit (class balance, missing values, duplicates).
3.  **Create Loader:** Create src/data/kaggle_loader.py to safely load, anonymize, and preprocess the dataset.
4.  **Train Model:** Update ML wrappers to train on the new dataset.
