# Phase 6.1 — Verified Experimental Results and Comprehensive Research Audit Report

**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System  
**Audit & Consolidation Date:** 2026-10-11  
**Authoritative Reference:** This document supersedes all preliminary narrative claims and consolidates the audited empirical findings across Phases 1 through 5.  
**Dataset Reference:** Cleaned Kaggle Modeling Dataset (`data/raw/blood_donor_dataset_modeling.csv`)  
- **SHA-256 Checksum:** `d3d1a7e301de160b547816e41fcf61880817e5125bbc4b4364e49adcc3eb02a3`  
- **Sample Size:** 2,000 sampled donor profiles across 8 blood groups and geo-spatial coordinates.  

---

## 1. Executive Summary & Audit Background

Following an independent audit of the experimental artifacts across Phases 1 through 5, all simulation code, data pipelines, statistical calculations, and narrative reports were subjected to verification against raw episode outputs.

### Summary of Audit Findings
1. **Empirical Integrity of Core Comparisons:** The Phase 4 benchmark evaluating 7 recruitment policies across 500 paired emergency scenarios (3,500 simulation episodes) and the Phase 5 ablation study across 6 arms (3,000 simulation episodes) are computationally sound, leakage-free, and reproducible.
2. **Structural Invariance of Alert Fatigue in Single-Episode Simulation:** The fatigue penalty factor $\Phi_{\text{fatigue}}(i) = \exp(-\lambda \cdot N_{\text{recent\_alerts}}(i))$ is structurally inactive in single-episode simulations because all candidate donors are initialized with zero prior alerts ($N_{\text{recent\_alerts}} = 0$, giving $\Phi = 1.0$), and the cascade policy strictly skips already-contacted donors (`select_batch()` excludes `already_notified`). Consequently, donors are contacted at most once per episode. Ablation C (No Fatigue) produces results identical to the Full System ($\Delta = 0.000$, $p = 1.0$).
3. **Correction of Erroneous Narrative Claims:** Prior preliminary drafts of the Phase 5 report erroneously asserted an increase in Alert Fatigue Index from 5.14 to 5.23 ($p = 1.03 \times 10^{-8}$) for Ablation C. These numbers were ungrounded in experimental data and have been removed. Furthermore, the sensitivity sweep across $\lambda$ showed a flat-line invariance, which represents structural invariance under single-episode constraints rather than active parameter tuning.
4. **Clarification of Arm Equivalence:** Ablation B (No Timeliness Score) and Ablation E (Response Score Only) are mathematically equivalent under the single-episode simulation design because fatigue discounting is structurally 1.0.
5. **Nuance in Additive vs. Multiplicative Formulations:** In Phase 4, UDI-Additive (Static) achieved a faster mean fulfillment time (26.66 min vs. 27.78 min, Wilcoxon $p = 1.70 \times 10^{-9}$), whereas Multiplicative Dynamic UDI enforces strict conjunctive criteria (infeasible/unresponsive donors cannot compensate with proximity), achieving marginally higher deadline fulfillment (99.8% vs. 99.6%).

---

## 2. Phase 2: Kaggle Availability-Classification Benchmark

### 2.1 Benchmark Design and Model Selection
The Phase 2 experiment evaluated machine learning classifiers on predicting the Kaggle dataset's binary `availability` label:
- **Dataset Partitioning:** 80% stratified training set ($N=1,600$), 20% held-out test set ($N=400$).
- **Cross-Validation:** 5-fold stratified cross-validation on the training set to prevent data leakage.
- **Model Selection Criterion:** Training cross-validation ROC-AUC.
- **Selected Model:** XGBoost (CV ROC-AUC: 0.4959, near random chance).

### 2.2 Critical Boundary Limitation
The Kaggle `availability` feature reflects static self-reported or historical donor status. It does **not** represent a validated emergency recruitment-response probability. In emergency scenarios, willingness to respond depends on real-time factors (time of day, urgency, travel friction, and alert frequency). To ensure methodological rigor, the synthetic simulation episodes utilize controlled stochastic response models (`StochasticResponseModel`) parameterized by donor responsiveness profiles, rather than uncalibrated classification probabilities.

---

## 3. Phase 4: Seven-Policy Comparative Experiment (500 Paired Scenarios)

### 3.1 Experimental Configuration
- **Scenarios Evaluated:** 500 paired synthetic emergency requests under diverse urgency horizons (CRITICAL, URGENT, SCHEDULED) and multi-unit demands.
- **Donor Pool:** 2,000 Kaggle donor profiles with realistic urban coordinates and BPR congestion travel times.
- **Policies Compared:**
  1. **Random:** Unguided candidate selection from the compatible donor pool.
  2. **Proximity-First:** Ranked strictly ascending by travel time (shortest ETA first).
  3. **Probability-First:** Ranked strictly descending by response propensity.
  4. **Heuristic:** Product of timeliness score and response propensity ($T_{\text{score}} \times P_{\text{response}}$).
  5. **UDI-Additive (Static):** Weighted linear combination: $w_u U(e) + w_t (1 - \tau / T_{\text{crit}}) + w_p P(i)$ computed once at initial dispatch.
  6. **UDI-Multiplicative (Static):** Multiplicative formulation ($P(i) \times T_{\text{score}} \times \Phi$) computed once at initial dispatch.
  7. **UDI-Multiplicative (Dynamic) [Reference]:** Multiplicative UDI dynamically recomputed at each dispatch wave as remaining deadline budget $T_{\text{budget}}$ shrinks.

### 3.2 Aggregate Policy Performance Metrics

All metrics independently recalculated and verified from `reports/phase4_kaggle_comparison/episode_results.csv` (3,500 rows):

| Policy Name | Deadline Fulfillment (%) | Eventual Fulfillment (%) | Mean Time to Deadline Met (min) | Median Time to Deadline Met (min) | Alert Fatigue Index (AFI) | Late Arrival Rate | Shortfall Ratio | Failed Deadlines (out of 500) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random** | 97.0% | 99.8% | 43.59 | 42.88 | 6.7320 | 1.0504 | 0.0007 | 15 |
| **Proximity-First** | 98.6% | 99.6% | 30.51 | 26.07 | 5.0154 | 0.3504 | 0.0014 | 7 |
| **Probability-First** | 99.8% | 99.8% | 32.11 | 29.84 | 5.5836 | 0.0000 | 0.0007 | 1 |
| **Heuristic** | 99.8% | 99.8% | 28.77 | 26.11 | 5.2281 | 0.0000 | 0.0007 | 1 |
| **UDI-Additive (Static)** | 99.6% | 99.8% | **26.66** | **23.60** | **4.9524** | 0.0700 | 0.0007 | 2 |
| **UDI-Multiplicative (Static)** | 99.8% | 99.8% | 28.05 | 25.18 | 5.1721 | 0.0000 | 0.0007 | 1 |
| **UDI-Multiplicative (Dynamic)** | **99.8%** | **99.8%** | 27.78 | 25.18 | 5.1449 | 0.0000 | 0.0007 | 1 |

*Note on AFI:* AFI is defined as total notification alerts sent divided by total units requested ($\sum \text{Alerts} / \sum \text{Units Req}$).

### 3.3 Paired Statistical Hypothesis Testing vs. Reference Policy (Dynamic UDI)

Pairwise difference is defined as:
$$\Delta = \text{Reference Policy (Dynamic UDI)} - \text{Comparison Policy}$$
- **Time Difference ($\Delta \text{Time}$):** Negative value indicates Dynamic UDI is faster (fewer minutes).
- **Alerts Difference ($\Delta \text{Alerts}$):** Negative value indicates Dynamic UDI sends fewer alerts.
- **Multiple Testing Correction:** Family-Wise Error Rate (FWER) controlled using the step-down Holm-Bonferroni procedure across the 6 baseline comparisons.

| Baseline Policy | Discordant Pairs ($b / c$) | Unadjusted McNemar $p$ | Holm-Adjusted McNemar $p$ | Mean Time Diff (min) [95% CI] | Wilcoxon Signed-Rank $p$ | Mean Alerts Diff | Alerts $t$-test $p$ | Statistically Significant Difference? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Random** | 0 / 14 | $0.00012$ | **$0.00073$** | $-17.11$ [$-18.77, -15.43$] | $9.85 \times 10^{-63}$ | $-4.54$ | $4.57 \times 10^{-70}$ | **Yes** (Dynamic UDI significantly superior) |
| **Proximity-First** | 0 / 6 | $0.03125$ | **$0.15625$** | $-3.30$ [$-4.81, -1.67$] | $7.83 \times 10^{-3}$ | $+0.37$ | $0.0466$ | **No** (Deadline diff not significant after Holm FWER) |
| **Probability-First** | 0 / 0 | $1.00000$ | $1.00000$ | $-4.33$ [$-5.00, -3.68$] | $2.06 \times 10^{-38}$ | $-1.25$ | $3.88 \times 10^{-41}$ | **Yes** (Dynamic UDI faster & fewer alerts) |
| **Heuristic** | 0 / 0 | $1.00000$ | $1.00000$ | $-0.99$ [$-1.35, -0.65$] | $9.96 \times 10^{-11}$ | $-0.24$ | $1.03 \times 10^{-8}$ | **Yes** (Dynamic UDI faster & fewer alerts) |
| **UDI-Additive (Static)** | 0 / 1 | $1.00000$ | $1.00000$ | **$+1.05$** [$+0.61, +1.51$] | $1.70 \times 10^{-9}$ | $+0.55$ | $1.62 \times 10^{-15}$ | **Yes** (Additive Static is faster & sends fewer alerts) |
| **UDI-Multiplicative (Static)** | 0 / 0 | $1.00000$ | $1.00000$ | $-0.27$ [$-0.46, -0.10$] | $8.98 \times 10^{-5}$ | $-0.08$ | $0.00179$ | **Yes** (Dynamic re-ranking is faster & saves alerts) |

### 3.4 Key Statistical Interpretations
1. **Random Dispatch Inferiority:** Dynamic UDI reduces recruitment latency by an average of 17.11 minutes and reduces broadcast overhead by 4.54 alerts per request compared to random dispatch ($p_{\text{Holm}} = 0.00073$).
2. **Proximity-First vs. Dynamic UDI:** While Proximity-First achieved nominal unadjusted significance ($p = 0.03125$, with 6 discordant scenarios where Dynamic UDI met the deadline but Proximity-First failed), this effect does **not** survive Holm-Bonferroni correction ($p_{\text{Holm}} = 0.15625$). However, Proximity-First exhibited a substantially higher late-arrival rate (0.3504 vs. 0.0000) due to dispatching unresponsive proximate donors.
3. **Additive vs. Multiplicative Formulation:** UDI-Additive (Static) fulfilled emergency requests 1.05 minutes faster than Dynamic UDI ($p = 1.70 \times 10^{-9}$) with lower alert overhead (AFI 4.95 vs. 5.14). The advantage of the Multiplicative formulation lies in conjunctivity: it guarantees that donors with near-zero willingness or marginal travel times are not selected based on high scores in a single dimension.

---

## 4. Phase 5: Controlled Ablation Study (500 Paired Scenarios)

### 4.1 Component Isolation Methodology
The ablation study evaluated 6 arms across the identical 500 scenarios to isolate individual algorithmic components:
- **Full System:** Dynamic Multiplicative UDI with response score, timeliness score, fatigue discount, and dynamic wave re-ranking.
- **Ablation A (No Response Score):** Sets $P_{\text{response}}(i) = 1.0$; ranking driven by travel time and timeliness.
- **Ablation B (No Timeliness Score):** Sets $T_{\text{score}}(i, e, t_k) = 1.0$ in ranking formula, while retaining hard feasibility filtering ($t_{\text{travel}} + t_{\text{prep}} \le T_{\text{budget}}$).
- **Ablation C (No Fatigue Discount):** Sets $\Phi_{\text{fatigue}}(i) = 1.0$.
- **Ablation D (No Dynamic Re-ranking):** Retains full multiplicative UDI but ranks donors once at $t=0$, holding rankings static across waves.
- **Ablation E (Response Score Only):** Ranks solely by $P_{\text{response}}(i)$, omitting timeliness and fatigue scoring.

### 4.2 Verified Ablation Study Results

Recalculated directly from `reports/phase5_ablation_sensitivity/ablation_episode_results.csv`:

| Ablation Arm | Deadline Fulfillment (%) | Mean Fulfillment Time (min) | Alert Fatigue Index (AFI) | Late Arrival Rate | Paired Time Diff vs. Full System ($\Delta$) | Wilcoxon $p_{\text{Time}}$ | Paired Alerts Diff vs. Full System | Alerts $t$-test $p$ | Operational Finding |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Full System** | 99.8% | 27.78 | 5.1449 | 0.0000 | Reference | Reference | Reference | Reference | Full Dynamic UDI baseline |
| **Ablation A** | 98.6% | 30.51 | 4.9706 | 0.0703 | $-2.94$ min slower | $0.0178$ | $+0.50$ alerts | $0.00557$ | Response score is critical for on-time fulfillment |
| **Ablation B** | 99.8% | 32.08 | 5.5745 | 0.0000 | $-4.30$ min slower | $2.19 \times 10^{-38}$ | $-1.23$ alerts | $9.57 \times 10^{-43}$ | Timeliness score accelerates fulfillment by 4.30 min |
| **Ablation C** | 99.8% | 27.78 | 5.1449 | 0.0000 | $\pm 0.00$ min | NaN ($p=1.0$) | $\pm 0.00$ alerts | NaN ($p=1.0$) | **Identical to Full System** (fatigue structurally inactive) |
| **Ablation D** | 99.8% | 28.05 | 5.1721 | 0.0000 | $-0.27$ min slower | $8.98 \times 10^{-5}$ | $-0.08$ alerts | $0.00179$ | Dynamic re-ranking provides verified speed & alert gains |
| **Ablation E** | 99.8% | 32.08 | 5.5745 | 0.0000 | $-4.30$ min slower | $2.19 \times 10^{-38}$ | $-1.23$ alerts | $9.57 \times 10^{-43}$ | **Identical to Ablation B** (structural equivalence) |

### 4.3 Detailed Component Explanations
1. **Response Score ($P_{\text{response}}$):** Omitting donor response propensity reduces deadline fulfillment from 99.8% to 98.6% (7 failures vs. 1 failure) and adds 2.73 minutes of fulfillment latency. Without response propensity, dispatch cascades prioritize nearby donors regardless of willingness, leading to wasted initial waves.
2. **Timeliness Score ($T_{\text{score}}$):** Omitting timeliness scoring while retaining hard feasibility filtering delays mean fulfillment by 4.30 minutes ($p = 2.19 \times 10^{-38}$) and forces 1.23 additional alerts per emergency.
3. **Dynamic Wave Re-ranking:** Updating remaining deadline budgets $T_{\text{budget}}$ across waves achieves statistically significant reductions in fulfillment latency ($-0.27$ min, $p = 8.98 \times 10^{-5}$) and alert overhead ($p = 0.00179$) relative to static ranking. Ablation D is mathematically identical to Phase 4's static multiplicative policy, confirming consistency across experiments.
4. **Fatigue Discount Inactivity (Ablation C & E):** In single-episode simulation, each donor is contacted at most once because `select_batch()` skips already-notified candidates. Because donors start with zero prior alerts, $\Phi_{\text{fatigue}} = \exp(0) = 1.0$ at candidate selection. Ablation C is therefore identical to Full System, and Ablation E is identical to Ablation B.

---

## 5. Phase 5: Multi-Dimensional Sensitivity Analysis (30 Configurations)

Verified against `reports/phase5_ablation_sensitivity/sensitivity_results.csv`:

| Parameter Dimension | Values Evaluated | Deadline Fulfillment (%) | Mean Time to Fulfillment (min) | Alert Fatigue Index (AFI) | Empirical Behavior Under Tested Assumptions |
|---|:---:|:---:|:---:|:---:|---|
| **Fatigue Decay Parameter ($\lambda$)** | $0.0, 0.15, 0.30, 0.45, 0.60$ | 100.0% $\rightarrow$ 100.0% | 27.22 $\rightarrow$ 27.22 | 5.09 $\rightarrow$ 5.09 | Invariant flat line under single-episode conditions (does not demonstrate active tuning). |
| **Urgency Horizon Tightness** | $0.75\times, 1.00\times, 1.25\times$ | 100.0% $\rightarrow$ 100.0% | 25.79 $\rightarrow$ 27.28 | 4.89 $\rightarrow$ 5.09 | Dynamic UDI prioritizes rapid responders under tighter deadlines, maintaining 100% success. |
| **Candidate Pool Size ($N$)** | $500, 1000, 2000$ | 84.0% $\rightarrow$ 100.0% | 34.30 $\rightarrow$ 27.22 | 4.64 $\rightarrow$ 5.09 | Pool size $\le 1000$ significantly degrades fulfillment (84% at $N=500$, shortfall ratio 0.091). |
| **Response Propensity Shift** | $0.70\times, 1.00\times, 1.30\times$ | 97.0% $\rightarrow$ 100.0% | 32.07 $\rightarrow$ 23.43 | 5.60 $\rightarrow$ 4.74 | Low donor responsiveness requires additional cascade waves, increasing AFI by +0.86. |
| **Congestion Multiplier** | $1.00\times, 1.25\times, 1.50\times$ | 100.0% $\rightarrow$ 100.0% | 27.22 $\rightarrow$ 32.94 | 5.09 $\rightarrow$ 5.70 | Severe congestion adds +5.72 min to transit; 100% deadline rate preserved via feasibility safety margins. |
| **Operational Reachability** | $0.50, 0.70, 0.90$ | 97.0% $\rightarrow$ 100.0% | 31.86 $\rightarrow$ 23.68 | 5.57 $\rightarrow$ 4.78 | Low phone contact rates (50%) are mitigated by cascade dispatch at the cost of +0.79 extra alerts/req. |
| **Blood Units Multiplier** | $1\times, 2\times, 3\times$ | 100.0% $\rightarrow$ 93.0% | 27.22 $\rightarrow$ 34.86 | 5.09 $\rightarrow$ 4.83 | Multi-unit emergencies encounter local donor shortages (7% deadline failure, 11.9% late arrival at 3 units). |
| **Dispatch Wave 1 Multiplier** | $2.0\times, 3.0\times, 4.0\times$ | 100.0% $\rightarrow$ 100.0% | 30.56 $\rightarrow$ 24.68 | 4.42 $\rightarrow$ 5.81 | Speed-overhead trade-off: larger initial wave accelerates fulfillment by 5.88 min at +1.39 alerts. |
| **Response Decision Latency ($\mu$)** | $1.5, 2.0, 2.5$ | 100.0% $\rightarrow$ 100.0% | 24.20 $\rightarrow$ 31.40 | 4.71 $\rightarrow$ 5.56 | Slower donor phone decisions translate directly into delivery delays (+7.20 min from $\mu=1.5$ to $2.5$). |

---

## 6. Scientific Status of Claims & Research Recommendations

### 6.1 Claims Fully Supported by Experimental Evidence
1. **Dynamic UDI vs. Random Dispatch:** Multiplicative Dynamic UDI is statistically superior to unguided random dispatch in deadline fulfillment ($p_{\text{Holm}} = 0.00073$), fulfillment speed ($-17.11$ min, $p = 9.85 \times 10^{-63}$), and alert overhead ($-4.54$ alerts, $p = 4.57 \times 10^{-70}$).
2. **Response Propensity Necessity:** Incorporating response propensity (Ablation A) is necessary to avoid recruitment failure; omitting it reduces deadline fulfillment to 98.6% and increases fulfillment time by 2.73 minutes.
3. **Timeliness Necessity:** Timeliness scoring (Ablation B) accelerates fulfillment by 4.30 minutes ($p = 2.19 \times 10^{-38}$) and reduces alert overhead by 1.23 alerts per emergency.
4. **Dynamic Wave Re-ranking Value:** Dynamically updating remaining deadline budgets across waves (Ablation D) significantly accelerates fulfillment ($-0.27$ min, $p = 8.98 \times 10^{-5}$) and reduces alert counts ($p = 0.00179$).
5. **Urban Friction Robustness:** Dynamic UDI maintains 100% deadline fulfillment under severe traffic congestion (1.50x travel time) and tight urgency deadlines (0.75x horizon).
6. **Candidate Pool Thresholds:** A minimum candidate pool size of $\ge 1,000$ donors is required for operational reliability in metropolitan emergency dispatch.

### 6.2 Claims NOT Supported by Current Data (Rectified)
1. **Alert Fatigue Tuning in Single-Episode Runs:** The claim that fatigue discounting reduced AFI from 5.23 to 5.14 ($p = 1.03 \times 10^{-8}$) was unsupported and has been removed. Within single-episode scenarios, the fatigue factor is structurally invariant.
2. **Sensitivity to Decay Parameter $\lambda$:** Varying $\lambda$ between 0.0 and 0.60 produced identical flat-line results. This is evidence of structural inactivity under single-contact assumptions, not evidence of optimal tuning.
3. **Independence of Ablation E:** Ablation E is mathematically and empirically identical to Ablation B under single-episode constraints.
4. **Universal Superiority over Additive Formulation:** UDI-Additive (Static) achieved faster mean fulfillment times (26.66 min vs. 27.78 min) than Multiplicative Dynamic UDI. The advantage of Multiplicative Dynamic UDI is conjunctive safety against edge-case failures, not raw speed.

### 6.3 Open Research Questions Requiring Future Experimentation
1. **Longitudinal Multi-Episode Recruitment:** To evaluate the fatigue discounting mechanism ($\Phi = \exp(-\lambda N)$), a longitudinal simulation spanning hundreds of sequential emergencies across weeks or months is required, allowing donor alert counts to accumulate over time.
2. **Empirical Response Probability Calibration:** Integrating authorized donor invitation-response data to replace synthetic response propensities with real-world calibrated probabilities.
