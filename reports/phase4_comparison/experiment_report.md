# Phase 4: Comparative Experiment Results (SUPERSEDED)

> **SUPERSEDED NOTICE:** This document contains preliminary comparative results from an earlier synthetic evaluation. The official, leakage-free benchmark utilizing the full cleaned Kaggle modeling dataset (500 paired scenarios across 7 ranking policies) is documented in 
eports/phase4_kaggle_comparison/experiment_report.md and consolidated in 
eports/phase6_corrected_results.md.

---

## Aggregate Metrics

```
                      Policy  Deadline_Fulfilment_Rate  Eventual_Fulfilment_Rate  Mean_Time_To_Fulfill_Eventual  Median_Time_To_Fulfill_Eventual  Invitations_Per_Successful_Request  Mean_Shortfall_Ratio  Alert_Fatigue_Index  Late_Arrival_Rate  No_Suitable_Donor_Rate  Failed_Requests_Count
                   Heuristic                      99.8                     100.0                      29.453485                        29.003785                           14.718000              0.000000             5.267717           0.071582                     0.0                    0.0
           Probability-First                     100.0                     100.0                      43.909119                        41.559540                           19.598000              0.000000             7.014316           0.000000                     0.0                    0.0
             Proximity-First                      97.4                      99.4                      33.650914                        28.373013                           14.177062              0.002147             5.088762           0.860832                     0.0                    3.0
                      Random                      95.2                      99.6                      74.124010                        73.918589                           29.152610              0.001432            10.417323           1.648746                     0.0                    2.0
       UDI-Additive (Static)                      99.4                     100.0                      30.220784                        27.892411                           14.834000              0.000000             5.309234           0.214746                     0.0                    0.0
UDI-Multiplicative (Dynamic)                     100.0                     100.0                      29.933559                        30.136684                           15.098000              0.000000             5.403722           0.000000                     0.0                    0.0
 UDI-Multiplicative (Static)                      99.6                     100.0                      30.327897                        30.420502                           15.206000              0.000000             5.442377           0.143164                     0.0                    0.0
```

## Statistical Comparisons

```
                     Policy                     Baseline  McNemar_Stat  McNemar_p_approx  McNemar_p_exact  Wilcoxon_p_conditional  Joint_Successes
                     Random UDI-Multiplicative (Dynamic)     22.041667          0.000003     1.192093e-07            1.283831e-82              498
            Proximity-First UDI-Multiplicative (Dynamic)     11.076923          0.000874     2.441406e-04            3.417385e-01              497
          Probability-First UDI-Multiplicative (Dynamic)      0.000000          1.000000     1.000000e+00            1.006449e-64              500
                  Heuristic UDI-Multiplicative (Dynamic)      0.000000          1.000000     1.000000e+00            1.761879e-01              500
      UDI-Additive (Static) UDI-Multiplicative (Dynamic)      1.333333          0.248213     2.500000e-01            4.005884e-01              500
UDI-Multiplicative (Static) UDI-Multiplicative (Dynamic)      0.500000          0.479500     5.000000e-01            7.778287e-01              500
```
