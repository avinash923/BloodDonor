# Phase 6.3 — Donation-Aware Temporal Eligibility and Longitudinal Safety Simulation

**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System
**Study Type:** In-silico research experiment. Results are **not** clinical validation and do **not** prove real-world clinical safety.
**Random seed:** `42` | **Execution elapsed:** `376.30 s`
**Official Clinical Guidance:** Drugs and Cosmetics Rules, 1945 (Schedule F, Part XII-B, as amended) & National Blood Transfusion Council (NBTC) / Directorate General of Health Services (DGHS) Guidelines, Ministry of Health and Family Welfare, Government of India: Whole-blood donation interval of minimum 3 months (90 days) for males and 4 months (120 days) for females.

## 1. Research Question

How does donation-aware temporal eligibility change emergency fulfilment and notification-burden results, and do the observed effects of fatigue-aware ranking remain when donor eligibility changes over time?

## 2. Experimental Design (Matched 2x2 Factorial)

Across matched independent 180-day timelines, four arms were evaluated:
- **Arm A (Fatigue-Aware + Donation Cooldown)**: Multiplicative dynamic UDI with fatigue discount ($\lambda=0.30$) and donation temporal eligibility hard gate.
- **Arm B (Fatigue-Disabled + Donation Cooldown)**: Multiplicative dynamic UDI with fatigue disabled ($\Phi=1.0$) and donation temporal eligibility hard gate.
- **Arm C (Fatigue-Aware + No Cooldown)**: Fatigue-aware UDI with no donation cooldown (comparison baseline).
- **Arm D (Fatigue-Disabled + No Cooldown)**: Fatigue-disabled UDI with no donation cooldown (comparison baseline).

## 3. Statutory & Clinical Guidance Reference

Under the **Drugs and Cosmetics Rules, 1945** (Schedule F, Part XII-B, as amended) and technical manuals issued by the **National Blood Transfusion Council (NBTC)** and **Directorate General of Health Services (DGHS)**, Ministry of Health and Family Welfare (MoHFW), Government of India:
- **Male whole-blood donation interval:** Minimum 3 months (90 days).
- **Female whole-blood donation interval:** Minimum 4 months (120 days).
These mandatory intervals allow adequate replenishment of donor red blood cells and iron stores.

## 4. Key Paired Statistical Findings

### Principal Fatigue Comparison (Arm A vs Arm B, within Donation Cooldown)

| Metric | Arm A Mean | Arm B Mean | Mean Difference (A − B) [95% CI] | Wilcoxon $p$ | Holm $p$ |
|---|---:|---:|---|---:|---:|
| Lookback repeat-notification rate | 0.3177 | 0.4026 | -0.0848 [-0.0906, -0.0797] | 6.1035e-05 | 4.8828e-04 |
| Percentage of donors with >=2 alerts (%) | 43.1033 | 40.8733 | 2.2300 [1.8167, 2.6568] | 6.5167e-04 | 4.5617e-03 |
| Percentage of donors with >=5 alerts (%) | 17.8633 | 17.6567 | 0.2067 [-0.2335, 0.6767] | 6.7007e-01 | 1.0000e+00 |
| Maximum notifications received by a single donor | 17.2667 | 19.7333 | -2.4667 [-4.2000, -0.7333] | 2.3308e-02 | 1.3985e-01 |
| Mean notifications per donor (full pool) | 2.0866 | 2.0711 | 0.0155 [-0.0061, 0.0361] | 2.0776e-01 | 8.3105e-01 |
| Alert Fatigue Index (alerts / units required) | 5.4636 | 5.4240 | 0.0396 [-0.0175, 0.0928] | 2.0776e-01 | 8.3105e-01 |
| Deadline fulfilment rate (% of emergencies) | 95.8997 | 95.7020 | 0.1976 [-0.6532, 0.9869] | 6.3777e-01 | 1.0000e+00 |
| Mean time to fulfilment among deadline-met (min) | 32.3070 | 31.7103 | 0.5967 [0.0123, 1.1761] | 7.2998e-02 | 3.6499e-01 |
| Gini concentration of alerts over pool | 0.6585 | 0.6854 | -0.0270 [-0.0309, -0.0233] | 6.1035e-05 | nan |
| Mean alerts per contacted donor | 3.8743 | 4.0678 | -0.1935 [-0.2276, -0.1591] | 6.1035e-05 | nan |
| Eventual quota fulfilment rate (%) | 95.9980 | 95.8762 | 0.1218 [-0.6552, 0.8322] | 8.4692e-01 | nan |
| Attempted selections prevented by gate | 46724.2000 | 47796.3333 | -1072.1333 [-1577.9650, -638.3250] | 1.5233e-03 | nan |
| Total completed synthetic donations | 695.6000 | 694.2000 | 1.4000 [-5.3333, 8.4683] | 7.5332e-01 | nan |

### Principal Eligibility Comparison (Arm A vs Arm C, Cooldown within Fatigue-Aware)

| Metric | Arm A Mean | Arm C Mean | Mean Difference (A − C) [95% CI] | Wilcoxon $p$ |
|---|---:|---:|---|---:|
| Lookback repeat-notification rate | 0.3177 | 0.3264 | -0.0086 [-0.0122, -0.0046] | 3.3569e-03 |
| Percentage of donors with >=2 alerts (%) | 43.1033 | 30.1233 | 12.9800 [12.4167, 13.5733] | 6.1035e-05 |
| Percentage of donors with >=5 alerts (%) | 17.8633 | 16.5000 | 1.3633 [0.8733, 1.8833] | 9.8154e-04 |
| Maximum notifications received by a single donor | 17.2667 | 24.8667 | -7.6000 [-9.5350, -5.8667] | 6.3376e-04 |
| Mean notifications per donor (full pool) | 2.0866 | 1.9563 | 0.1303 [0.1092, 0.1516] | 6.1035e-05 |
| Alert Fatigue Index (alerts / units required) | 5.4636 | 5.1203 | 0.3433 [0.2859, 0.3996] | 6.1035e-05 |
| Deadline fulfilment rate (% of emergencies) | 95.8997 | 99.5336 | -3.6339 [-4.2514, -3.0628] | 6.1035e-05 |
| Mean time to fulfilment among deadline-met (min) | 32.3070 | 27.9358 | 4.3711 [3.7316, 5.0203] | 6.1035e-05 |
| Gini concentration of alerts over pool | 0.6585 | 0.7835 | -0.1250 [-0.1300, -0.1204] | 6.1035e-05 |
| Mean alerts per contacted donor | 3.8743 | 4.9476 | -1.0733 [-1.1429, -1.0049] | 6.1035e-05 |
| Eventual quota fulfilment rate (%) | 95.9980 | 99.5811 | -3.5831 [-4.1873, -2.9801] | 6.1035e-05 |
| Attempted selections prevented by gate | 46724.2000 | 0.0000 | 46724.2000 [44284.3350, 49271.2183] | 6.1035e-05 |
| Total completed synthetic donations | 695.6000 | 712.4667 | -16.8667 [-22.2683, -11.2000] | 9.7588e-04 |

## 5. Temporal Safety Gate & Ineligible Contact Prevention

- **Arm A (Fatigue-Aware):** Prevented an average of **46724.2** attempted candidate rankings of deferred donors per 180-day timeline.
- **Arm B (Fatigue-Disabled):** Prevented an average of **47796.3** attempted candidate rankings of deferred donors per 180-day timeline.
In the absence of this temporal gate (Arms C & D), recent donors would be repeatedly recruited despite clinical deferral rules.

## 6. Limitations & Boundaries

- Synthetic demographics: Gender groups and policy intervals were synthetically assigned since Kaggle dataset lacks demographic attributes.
- Fixed screening probabilities: Simulated 95% clinical pass rate and 98% phlebotomy completion rate are synthetic modeling assumptions.
- Operational boundary: The UDI system is an algorithmic decision-support tool and must never replace authorized blood bank medical screening.