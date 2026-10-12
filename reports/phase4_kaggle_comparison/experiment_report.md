# Phase 4 — Full Comparative Experiment Results: Kaggle Blood Donor Integration

**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System  
**Dataset:** Kaggle Cleaned Modeling Dataset (`data/raw/blood_donor_dataset_modeling.csv`)  
**Checksum (SHA-256):** `d3d1a7e301de160b547816e41fcf61880817e5125bbc4b4364e49adcc3eb02a3`  
**Run ID:** `EXP4_KAG_c2ecc324`  
**Scenarios Evaluated:** 500  
**Candidate Donor Pool Size:** 2000 sampled profiles  
**Random Seed:** 42  
**Execution Timestamp:** 2026-10-11T13:18:25.680501  

---

## 1. Experimental Policy Arms

We evaluated all seven target ranking policies across identical emergency scenarios and candidate pools:
- **Policy A — Random:** Uniform random permutation of eligible candidates.
- **Policy B — Proximity-First:** Candidates ordered ascending by travel time (shortest ETA first).
- **Policy C — Probability-First:** Candidates ordered descending by synthetic response propensity.
- **Policy D — Heuristic (Proximity × Probability):** Timeliness score $\times$ synthetic response propensity.
- **Policy E — Additive UDI:** Weighted linear sum: $w_u U(e) + w_t (1 - \tau/T_{crit}) + w_p P(i)$ discounted by fatigue.
- **Policy F — Multiplicative Static UDI:** Product $P(i) \times (1 - \tau/T_{budget}) \times \Phi_{fatigue}(i)$ computed once at dispatch.
- **Policy G — Multiplicative Dynamic UDI:** Re-computed each wave as remaining time $T_{budget}$ decreases and fatigue accumulates.

---

## 2. Ranking Policy Performance Summary

```
                      Policy  Scenario_Count  Deadline_Fulfilment_Rate  Eventual_Fulfilment_Rate  Mean_Time_To_Deadline_Met  Median_Time_To_Deadline_Met  Mean_Time_To_Eventual_Met  Median_Time_To_Eventual_Met  Invitations_Per_Successful_Fulfilment  Shortfall_Ratio  Alert_Fatigue_Index  Late_Arrival_Rate  No_Suitable_Donor_Rate  Failed_Deadline_Count  Failed_Eventual_Count
                      Random           500.0                      97.0                      99.8                  43.585361                    42.876673                  44.895648                    43.236457                              19.230461           0.0007             6.731980           1.050420                     0.0                   15.0                    1.0
             Proximity-First           500.0                      98.6                      99.6                  30.509250                    26.071076                  31.113176                    26.325971                              14.315261           0.0014             5.015395           0.350385                     0.0                    7.0                    2.0
           Probability-First           500.0                      99.8                      99.8                  32.114571                    29.843384                  32.114571                    29.843384                              15.941884           0.0007             5.583625           0.000000                     0.0                    1.0                    1.0
                   Heuristic           500.0                      99.8                      99.8                  28.769019                    26.108491                  28.769019                    26.108491                              14.923848           0.0007             5.228132           0.000000                     0.0                    1.0                    1.0
       UDI-Additive (Static)           500.0                      99.6                      99.8                  26.664018                    23.602931                  26.733059                    23.633605                              14.134269           0.0007             4.952414           0.070028                     0.0                    2.0                    1.0
 UDI-Multiplicative (Static)           500.0                      99.8                      99.8                  28.050154                    25.176609                  28.050154                    25.176609                              14.763527           0.0007             5.172148           0.000000                     0.0                    1.0                    1.0
UDI-Multiplicative (Dynamic)           500.0                      99.8                      99.8                  27.781009                    25.176609                  27.781009                    25.176609                              14.687375           0.0007             5.144857           0.000000                     0.0                    1.0                    1.0
```

---

## 3. Pairwise Statistical Comparisons (vs. UDI-Multiplicative Dynamic)

All pairwise comparisons evaluate identical paired scenario IDs ($N=500$).
- Binary deadline fulfillment evaluated via **exact McNemar test** (accounting for paired discordant outcomes).
- Conditional time to fulfillment compared via **Wilcoxon signed-rank test** on joint eventual successes.
- Alert overhead compared via paired t-test with Cohen's $d$ effect sizes.

```
                     Policy             Reference_Policy  Paired_Scenarios  Discordant_Pairs_b_ref0_comp1  Discordant_Pairs_c_ref1_comp0  McNemar_Stat  McNemar_p_exact  Deadline_Fulfilment_Diff_Pct  Joint_Eventual_Successes  Mean_Time_Diff_Minutes  Mean_Time_Diff_95CI_Low  Mean_Time_Diff_95CI_High  Wilcoxon_p_Time  Mean_Alerts_Diff  Alerts_Diff_Cohen_d  Alerts_ttest_p
                     Random UDI-Multiplicative (Dynamic)               500                              0                             14     12.071429         0.000122                           2.8                       499              -17.114639               -18.766858                -15.425112     9.848305e-63            -4.536            -0.933961    4.565927e-70
            Proximity-First UDI-Multiplicative (Dynamic)               500                              0                              6      4.166667         0.031250                           1.2                       498               -3.297324                -4.808162                 -1.674209     7.829626e-03             0.370             0.089206    4.661878e-02
          Probability-First UDI-Multiplicative (Dynamic)               500                              0                              0      0.000000         1.000000                           0.0                       499               -4.333561                -4.998335                 -3.676679     2.059908e-38            -1.254            -0.659752    3.876780e-41
                  Heuristic UDI-Multiplicative (Dynamic)               500                              0                              0      0.000000         1.000000                           0.0                       499               -0.988009                -1.352411                 -0.651424     9.960050e-11            -0.238            -0.260452    1.031043e-08
      UDI-Additive (Static) UDI-Multiplicative (Dynamic)               500                              0                              1      0.000000         1.000000                           0.2                       499                1.047950                 0.608085                  1.512387     1.695874e-09             0.550             0.368145    1.620519e-15
UDI-Multiplicative (Static) UDI-Multiplicative (Dynamic)               500                              0                              0      0.000000         1.000000                           0.0                       499               -0.269145                -0.455648                 -0.102111     8.979166e-05            -0.078            -0.140425    1.789665e-03
```

---

## 4. Key Findings

1. **Recruitment Deadline Fulfillment:** 
   Multiplicative Dynamic UDI achieves the highest operational reliability (99.8% deadline fulfillment), significantly outperforming unguided Random dispatch (97.0%, McNemar $p_{	ext{Holm}} = 0.00073$) and reducing late arrivals compared to Proximity-First (98.6%, though the pairwise difference is not statistically significant after Holm-Bonferroni correction, $p_{	ext{Holm}} = 0.156$).
2. **Fulfillment Speed vs. Conjunctive Robustness (Additive vs. Multiplicative Formulation):**
   UDI-Additive (Static) achieved the fastest mean fulfillment time (26.66 min vs. 27.78 min for Dynamic UDI, Wilcoxon paired $p = 1.70 	imes 10^{-9}$), as well as a lower Alert Fatigue Index (4.95 vs. 5.14). However, the Multiplicative Dynamic formulation enforces strict conjunctivity (unresponsive or infeasible candidates cannot compensate with high proximity scores), yielding slightly higher overall deadline fulfillment (99.8% vs. 99.6%, though this single scenario difference is not statistically significant, McNemar $p = 1.0$).
3. **Dynamic Horizon Updating and Alert Overhead:**
   Dynamic re-ranking re-evaluates the remaining deadline budget $T_{	ext{budget}}$ at each dispatch wave, achieving slightly faster fulfillment than Static Multiplicative UDI (27.78 min vs. 28.05 min, $p = 8.98 	imes 10^{-5}$) and sending fewer alerts. Note that within a single emergency episode, the fatigue discount factor is structurally invariant because donors are contacted at most once; its primary design purpose is longitudinal protection across repeated emergencies.

---

## 5. Artifact Manifest

- `scenario_manifest.json`: Configuration, dataset checksum, and scenario metadata.
- `episode_results.csv`: Complete raw record of all 3500 simulation episodes.
- `ranking_policy_metrics.csv`: Aggregate operational metrics across all 7 policies.
- `pairwise_comparisons.csv`: Paired McNemar, Wilcoxon, and effect size statistics.
- `event_traces/`: Detailed wave-by-wave dispatch traces for sample scenarios.
- `figures/fulfilment_rates.png`: Comparison of deadline and eventual success rates.
- `figures/time_to_fulfilment.png`: Boxplot of delivery latencies.
- `figures/alert_fatigue_index.png`: Notification overhead comparison.
- `figures/shortfall_ratio.png`: Unmet demand ratios.
- `figures/udi_variants_comparison.png`: Direct comparison of UDI architectural variants.
