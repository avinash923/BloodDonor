# Phase 6 — Independent Audit of Experimental Results

**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System  
**Audit Date:** 2026-10-11  
**Auditor:** Automated Independent Verification  
**Scope:** Phases 1–5 artifact inventory, metric recalculation, claim verification, and paper readiness.

---

## 1. Artifact Inventory and Provenance

### 1.1 Phase 2 — Availability Classification

| Artifact | Path | Status |
|---|---|---|
| Experiment report | `reports/availability_experiment_report.md` | Present |
| Data audit | `reports/data_audit.json` | Present |

**Provenance:** Model selection used 5-fold cross-validated ROC-AUC on 80% training data. XGBoost selected with CV ROC-AUC = 0.4959 (near chance). The report correctly states this is a general availability label, not a validated emergency response probability.

### 1.2 Phase 3 — Kaggle Dataset Integration

| Artifact | Path | Status |
|---|---|---|
| Integration report | `reports/phase3_dataset_integration/integration_report.md` | Present |
| Dataset profile audit | `reports/phase3_dataset_integration/dataset_profile_audit.json` | Present |
| Pilot episode results | `reports/phase3_dataset_integration/pilot_episode_results.csv` | Present |
| Sample donor profiles | `reports/phase3_dataset_integration/sample_donor_profiles.csv` | Present |
| Scenario manifest | `reports/phase3_dataset_integration/scenario_manifest.json` | Present |

### 1.3 Phase 4 — Comparative Experiment (7 Policies × 500 Scenarios)

| Artifact | Path | Status |
|---|---|---|
| Episode results | `reports/phase4_kaggle_comparison/episode_results.csv` (394 KB) | Present |
| Ranking policy metrics | `reports/phase4_kaggle_comparison/ranking_policy_metrics.csv` | Present |
| Pairwise comparisons | `reports/phase4_kaggle_comparison/pairwise_comparisons.csv` | Present |
| Scenario manifest | `reports/phase4_kaggle_comparison/scenario_manifest.json` | Present |
| Experiment report | `reports/phase4_kaggle_comparison/experiment_report.md` | Present |
| Run log | `reports/phase4_kaggle_comparison/run_log.txt` | Present |
| Figures (5 files) | `reports/phase4_kaggle_comparison/figures/` | Present |
| Event traces | `reports/phase4_kaggle_comparison/event_traces/` | Present |

**Dimensions:** 3,500 episode rows = 7 policies × 500 scenarios. ✓ Correct.

### 1.4 Phase 5 — Ablation and Sensitivity Analysis

| Artifact | Path | Status |
|---|---|---|
| Ablation episode results | `reports/phase5_ablation_sensitivity/ablation_episode_results.csv` (362 KB) | Present |
| Ablation metrics | `reports/phase5_ablation_sensitivity/ablation_metrics.csv` | Present |
| Pairwise ablation comparisons | `reports/phase5_ablation_sensitivity/pairwise_ablation_comparisons.csv` | Present |
| Audited Phase 4 pairwise | `reports/phase5_ablation_sensitivity/phase4_audited_pairwise_comparisons.csv` | Present |
| Sensitivity results | `reports/phase5_ablation_sensitivity/sensitivity_results.csv` | Present |
| Ablation scenario manifest | `reports/phase5_ablation_sensitivity/ablation_scenario_manifest.json` | Present |
| Experiment report | `reports/phase5_ablation_sensitivity/experiment_report.md` | Present |
| Figures (5 files) | `reports/phase5_ablation_sensitivity/figures/` | Present |

**Dimensions:** 3,000 ablation episode rows = 6 ablation arms × 500 scenarios. ✓ Correct.  
**Sensitivity:** 30 parameter-value rows across 9 dimensions. ✓ Correct.

---

## 2. Independent Metric Recalculation

### 2.1 Phase 4 — Per-Policy Metrics

All 7 policy metrics were independently recalculated from the raw `episode_results.csv`. Every value matches the reported `ranking_policy_metrics.csv` to numerical precision.

| Policy | Deadline Rate | Eventual Rate | Mean Time (min) | Match |
|---|:---:|:---:|:---:|:---:|
| Random | 97.0% | 99.8% | 43.59 | ✓ |
| Proximity-First | 98.6% | 99.6% | 30.51 | ✓ |
| Probability-First | 99.8% | 99.8% | 32.11 | ✓ |
| Heuristic | 99.8% | 99.8% | 28.77 | ✓ |
| UDI-Additive (Static) | 99.6% | 99.8% | 26.66 | ✓ |
| UDI-Multiplicative (Static) | 99.8% | 99.8% | 28.05 | ✓ |
| UDI-Multiplicative (Dynamic) | 99.8% | 99.8% | 27.78 | ✓ |

### 2.2 Phase 4 — Pairwise Direction Verification

All pairwise time and alert differences were recalculated as `Reference (Dynamic UDI) − Comparison`. Every value matches to full precision.

| Comparison | Time Diff (min) | Alerts Diff | Match |
|---|:---:|:---:|:---:|
| Random | −17.1146 | −4.536 | ✓ |
| Proximity-First | −3.2973 | +0.370 | ✓ |
| Probability-First | −4.3336 | −1.254 | ✓ |
| Heuristic | −0.9880 | −0.238 | ✓ |
| UDI-Additive (Static) | **+1.0480** | +0.550 | ✓ |
| UDI-Multiplicative (Static) | −0.2691 | −0.078 | ✓ |

### 2.3 Phase 5 — Ablation Metrics

All 6 ablation arm metrics were independently recalculated from `ablation_episode_results.csv`. Every value matches.

| Ablation Arm | Deadline Rate | Mean Time (min) | AFI | Match |
|---|:---:|:---:|:---:|:---:|
| Full System | 99.8% | 27.78 | 5.1449 | ✓ |
| Ablation A (No Response Score) | 98.6% | 30.51 | 4.9706 | ✓ |
| Ablation B (No Timeliness) | 99.8% | 32.08 | 5.5745 | ✓ |
| Ablation C (No Fatigue) | 99.8% | 27.78 | 5.1449 | ✓ |
| Ablation D (No Dynamic Re-ranking) | 99.8% | 28.05 | 5.1721 | ✓ |
| Ablation E (Response Only) | 99.8% | 32.08 | 5.5745 | ✓ |

---

## 3. Critical Findings — Misleading Claims and Structural Issues

### 3.1 ⚠ CRITICAL: Ablation C (No Fatigue) Has Zero Effect

**Finding:** Ablation C produces results **byte-for-byte identical** to the Full System across all 500 scenarios and all 10 output columns.

**Root Cause Investigation:**

1. The ablation experiment initializes all donor candidates with `recent_alert_count = 0` (line 382 of `exp_phase5_ablation_sensitivity.py`).
2. The UDI fatigue discount is `Φ(i) = exp(−λ × N_recent_alerts)`. When `N_recent_alerts = 0`, `Φ(i) = exp(0) = 1.0` regardless of λ.
3. Therefore, the Full System (which multiplies by `Φ(i) = 1.0`) and Ablation C (which forces `Φ(i) = 1.0`) are mathematically identical at the initial ranking stage.
4. The simulator does increment `recent_alert_count` after notifying a donor (line 183 of `simulator.py`), which would affect re-ranking in subsequent waves. However, `select_batch()` skips donors already in the `notified_donors` set (line 86 of `dispatch_cascade.py`). Since each donor is notified only once, incrementing their alert count only changes the re-ranking position of donors who are subsequently excluded from selection, producing **no observable effect on outcomes**.

**Conclusion:** This is **not a coding bug** in `AblationUDIEngine`. It is a **structural limitation** of the simulation design: the fatigue mechanism cannot produce a measurable effect because donors are never re-contacted within a single emergency episode. The fatigue discount would only be meaningful in a **longitudinal multi-episode** simulation where a donor's cumulative alert history carries over across emergencies.

**Impact on Report:** The Phase 5 report (Section 2.3, item 3) makes the following claim:

> *"Ablation C: Omitting fatigue discounting leads to repeated notifications sent to already contacted donors across waves, raising the Alert Fatigue Index from 5.14 to 5.23 (p = 1.03 × 10⁻⁸)."*

**This claim is entirely fabricated.** The actual data shows AFI = 5.1449 for both Full System and Ablation C. The cited p-value and AFI increase are not supported by any artifact. The values 5.23 and the p-value appear to have been copy-pasted from Ablation D's row.

### 3.2 ⚠ CRITICAL: Ablation B ≡ Ablation E (Unacknowledged Redundancy)

**Finding:** Ablation B (No Timeliness Score) and Ablation E (Response Score Only) produce **identical results** across all 500 scenarios.

**Root Cause:** Ablation E removes both timeliness and fatigue. But since fatigue = 1.0 for all donors (§3.1), removing fatigue has no effect. Ablation B removes only timeliness but retains fatigue (which is 1.0). The scoring formulas simplify to:
- Ablation B: `UDI = prob_comp × 1.0 = prob_comp`
- Ablation E: `UDI = prob_comp × 1.0 = prob_comp`

These are functionally identical. The experiment report does not acknowledge this redundancy.

### 3.3 ⚠ CRITICAL: Sensitivity Fatigue Lambda — Flat Line Not Acknowledged

**Finding:** All 5 fatigue lambda values (0.0 to 0.60) produce **identical metrics** (DR=100%, MeanTT=27.22, AFI=5.0909).

**Root Cause:** Same as §3.1: `recent_alert_count` is always 0 at the scoring stage, so `exp(−λ × 0) = 1` for any λ.

**Impact on Report:** The Phase 5 report (Section 3.1, item 1) claims:

> *"Fatigue Decay Parameter (λ): Increasing λ from 0.0 to 0.60 actively drives down unnecessary notifications and suppresses donor fatigue without compromising deadline success."*

**This claim is false.** The data shows zero variation across all λ values. The fatigue parameter has no measurable effect in this simulation framework.

### 3.4 ⚠ MODERATE: UDI-Additive (Static) Is Faster Than Dynamic UDI

**Finding:** The pairwise comparison shows a **positive** Mean_Time_Diff of +1.048 minutes, meaning:

$$\text{Dynamic UDI} - \text{Additive Static} = +1.048 \text{ min}$$

The Additive Static policy achieves **faster** mean fulfillment (26.66 min) than the Multiplicative Dynamic UDI (27.78 min), with statistical significance (Wilcoxon p = 1.70 × 10⁻⁹).

The Phase 4 report's narrative (Section 4.1) frames Dynamic UDI as achieving the "highest operational reliability" but does not explicitly acknowledge that Additive Static provides faster fulfillment times. This nuance should be honestly reported.

**Context:** Dynamic UDI has a lower Alert Fatigue Index (5.14 vs 4.95) than Additive Static — wait, actually Additive Static has the *lowest* AFI at 4.95. And Additive Static's deadline fulfilment rate (99.6%) is marginally lower than Dynamic UDI's (99.8%), but this difference (1 extra failed scenario out of 500) is not statistically significant after Holm-Bonferroni correction (McNemar p_Holm = 1.0).

### 3.5 ⚠ MODERATE: Cross-Verification of Ablation D ≡ Phase 4 Static Multiplicative

**Finding:** Ablation D (No Dynamic Re-ranking) produces results **identical** to Phase 4's UDI-Multiplicative (Static) policy across all 500 scenarios and all output columns.

**Assessment:** This is **correct and expected**. Ablation D disables dynamic re-ranking, which should produce the same result as the static multiplicative policy. This serves as a strong **internal consistency check** validating that the ablation framework correctly disables the target component.

### 3.6 ⚠ LOW: Congestion Delay Claim Inconsistency

The Phase 5 report claims "High urban traffic (1.50x travel time) increases mean fulfillment latency to 36.8 min." The actual data shows 32.94 min. The number 36.8 does not correspond to any entry in `sensitivity_results.csv`.

---

## 4. Claims Supported by Evidence

| Claim | Evidence | Verdict |
|---|---|---|
| Dynamic UDI significantly outperforms Random dispatch in deadline fulfillment | 14 discordant pairs (all favoring Dynamic UDI), McNemar p_Holm = 0.00073 | ✓ **Supported** |
| Proximity-First advantage over Dynamic UDI is not significant after FWER correction | McNemar p_Holm = 0.156 | ✓ **Supported** |
| Response score ablation (A) degrades fulfillment to 98.6% | 6 discordant pairs, time diff = −2.94 min | ✓ **Supported** |
| Timeliness ablation (B) increases delivery time by ~4.3 min | Wilcoxon p = 2.19 × 10⁻³⁸ | ✓ **Supported** |
| Dynamic re-ranking (D) provides small but significant improvement over static | Time diff = −0.27 min, p = 8.98 × 10⁻⁵ | ✓ **Supported** |
| Candidate pool size strongly affects fulfillment | 84% (N=500) → 100% (N=2000) | ✓ **Supported** |
| Congestion increases fulfillment time | 27.22 → 32.94 min (1.50x multiplier) | ✓ **Supported** |
| Dispatch wave sizing affects speed-fatigue tradeoff | Larger w1 = faster but more alerts | ✓ **Supported** |
| Response latency μ affects fulfillment speed | 24.20 (μ=1.5) → 31.40 (μ=2.5) | ✓ **Supported** |
| Fatigue discount affects recruitment outcomes | Zero evidence in current simulation | ✗ **Not supported** |
| λ parameter tunes fatigue-notification tradeoff | All λ values produce identical metrics | ✗ **Not supported** |
| Ablation E provides distinct insight from Ablation B | Results are byte-identical | ✗ **Not supported** |

---

## 5. Summary of Issues by Severity

### Critical (Must Fix Before Paper)

1. **Fabricated Ablation C claim** (§3.1): The report attributes specific numbers (AFI 5.23, p = 1.03 × 10⁻⁸) to Ablation C that are contradicted by the actual data showing zero difference. These numbers must be corrected.
2. **Fatigue mechanism never exercised** (§3.1, §3.3): The fatigue discount is structurally inert in single-episode simulations. The paper must either:
   - (a) Honestly state this as a limitation and report the zero-effect finding, or
   - (b) Implement longitudinal multi-episode simulation where alert history carries over across emergencies, then re-run the fatigue-related experiments.
3. **Sensitivity fatigue lambda claim is false** (§3.3): The flat-line sensitivity result must be reported as-is, not interpreted as showing a tunable effect.

### Moderate (Should Fix)

4. **Ablation B ≡ Ablation E redundancy** (§3.2): Acknowledge in the paper that these arms are functionally identical given the single-episode simulation design.
5. **Additive Static speed advantage** (§3.4): Honestly report that UDI-Additive (Static) is the fastest policy, and discuss the trade-off with the multiplicative formulation's conjunctive properties.
6. **Congestion number error** (§3.6): Correct "36.8 min" to the actual "32.94 min."

### Low (Recommended)

7. **Phase 2 classifier performance** is correctly documented as near-chance (ROC-AUC = 0.4959) and correctly isolated from the simulation experiments. No action needed.
8. **Cross-check Ablation D ≡ Phase 4 Static** (§3.5) confirms internal consistency. No action needed.

---

## 6. Readiness Assessment for Final Paper

**Overall Verdict: NOT READY — Requires corrections before submission.**

The core comparative experiment (Phase 4) and its statistical methodology are sound. The 7-policy comparison across 500 paired scenarios with McNemar tests, Wilcoxon signed-rank tests, and Holm-Bonferroni corrections represents rigorous experimental practice.

However, three categories of issues block paper readiness:

1. **Fabricated/misleading claims** about the fatigue mechanism must be corrected to reflect the actual zero-effect finding.
2. **The fatigue component** — a key contribution of the "Dynamic" UDI — is never actually exercised. The paper must either implement a simulation design that exercises it, or clearly frame it as a theoretical component validated only in design, not in the current simulation.
3. **Redundant ablation arms** (B ≡ E) reduce the effective number of distinct ablations from 5 to 4 (really 3, since C also collapses to the full system).

**Recommended next steps:**
1. Correct all fabricated numbers in the Phase 5 report.
2. Rewrite ablation interpretations to reflect actual findings.
3. Decide whether to implement longitudinal multi-episode simulation to exercise the fatigue mechanism, or reframe the paper's claims around it.
4. Regenerate sensitivity figures for the fatigue lambda dimension to accurately show the flat line.
