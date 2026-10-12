# System Audit Report
**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System
**Date:** 2026-10-10

## 1. Implementation Status (What is Built)
The codebase contains a comprehensive structure spanning the 7 requested phases:
- **Phase 1:** Pydantic schemas (`data/schemas/`).
- **Phase 2:** ML Wrappers for UCI data (`src/models/`).
- **Phase 3:** Spatial Feasibility & Travel Time Engines (`src/spatial/`).
- **Phase 4:** UDI Ranking Engine & Baselines (`src/ranking/`).
- **Phase 5:** Discrete Event Simulator (`src/simulation/`).
- **Phase 6:** Statistical Evaluation tools (`src/evaluation/`).
- **Phase 7:** Automated Report Generator (`src/evaluation/report_generator.py`).

## 2. Incomplete, Broken, or Untested Code
- **Mocked Data in Analysis:** `experiments/exp5_statistical_analysis.py` (Lines 34-42) does not load results from previous simulations. Instead, it generates random noise (`np.random.normal`) to perform statistical tests. This renders EXP-5 scientifically invalid.
- **Disconnected Data Pipeline:** The experiment scripts (`exp2` through `exp7`) currently generate fresh synthetic data pools on the fly rather than writing intermediate results (e.g., simulated donor behaviors) to a shared file/database. This breaks longitudinal traceability across experiments.
- **Test Execution Blocked:** Automated tests (`pytest`) could not be run because terminal command execution is currently denied by the environment permissions. Static analysis indicates tests are written, but runtime correctness is unverified.

## 3. Experiment Runner Execution
- **Command Run Attempted:** `pytest tests/` 
- **Result:** **FAILED (User Denied Permission)**. 
- Due to the inability to run terminal commands, I cannot empirically verify if all 7 experiment scripts execute from end-to-end without runtime exceptions.

## 4. Scientific Soundness & Critical Flaws
There are two major scientific discrepancies in the current implementation:

### A. UDI Formula Mismatch
- **Proposed Formula:** Multiplicative (e.g., $UDI = U(e) \times (1 - \tau/T_{crit}) \times P(i) \times \Phi(i)$)
- **Implemented Formula:** Additive Weighted (see [`src/ranking/udi_engine.py`](file:///c:/BloodDonor/src/ranking/udi_engine.py#L192-L204))
  ```python
  urgency_comp = self.w_urgency * u_score
  timeliness_comp = self.w_timeliness * timeliness_raw
  probability_comp = self.w_probability * candidate.response_probability
  weighted_sum = urgency_comp + timeliness_comp + probability_comp
  udi = float(np.clip(weighted_sum * fatigue, 0.0, 1.0))
  ```
- **Why this matters:** An additive formula allows poor performance in one metric (e.g., donor is very far away) to be offset by a high score in another (e.g., highly urgent request). A multiplicative formula strictly penalizes candidates who fail heavily on *any* single criterion.

### B. UCI Dataset Mislabeling
- The UCI Blood Transfusion dataset models *historical, macroscopic donation behavior* (whether a person donated in a given month based on RFM). 
- The simulation treats this ML output as the probability of a donor responding *immediately* to an emergency dispatch alert. This is a severe mislabeling of the prediction target.

## 5. Essential Missing Tasks Before Reporting
Before any results can be considered publishable, the following blockers must be resolved:
1. **Refactor UDI Engine:** Update [`src/ranking/udi_engine.py`](file:///c:/BloodDonor/src/ranking/udi_engine.py) to implement the true multiplicative formula and adjust configuration weights accordingly.
2. **Connect the Data Pipeline:** Modify `exp4_dynamic_recruitment.py` to write its simulation episode outcomes to a CSV file.
3. **Fix EXP-5:** Rewrite `exp5_statistical_analysis.py` to parse the actual CSV output from EXP-4 instead of generating random noise.
4. **Address UCI Framing:** Explicitly document in the report that the ML model provides a "baseline propensity" proxy, not a real-time behavioral guarantee, to protect the scientific integrity of the paper.

## 6. Data Provenance (Measured vs. Synthetic)
- **Measured (Real-World):** Phase 1 & 2 ML evaluations (AUC, F1) use the actual UCI dataset.
- **Synthetic (In-Silico):** Phase 3 (Spatial), Phase 4 (Ranking), Phase 5 (Simulation), Phase 6 (Stats/Ablation), and Phase 7 (Reporting) all use 100% synthetic spatial coordinates and stochastically generated donor pools. **No simulated results represent real-world clinical evidence.**
