# Phase 6.2 — Longitudinal Multi-Episode Fatigue Simulation

**Project:** AI-Based Time-Aware Emergency Blood Donor Recommendation and Dynamic Recruitment System
**Study type:** In-silico research experiment. Results are **not** clinical validation and do **not** prove real-world donor behaviour.
**Random seed:** `42`
**Execution timestamp:** 2026-10-11T18:17:25.611449
**Elapsed seconds:** 220.47
**Dataset SHA-256:** `d3d1a7e301de160b547816e41fcf61880817e5125bbc4b4364e49adcc3eb02a3`

## 1. Research question

Across repeated synthetic emergency incidents, does fatigue-aware dynamic UDI ranking reduce notification burden per donor, and what trade-off, if any, does it introduce in deadline fulfilment?

## 2. Design

Matched comparison of the same **Multiplicative Dynamic UDI** architecture:

- **Arm A (fatigue-aware):** $\lambda = 0.30$, lookback $= 14$ days.
- **Arm B (fatigue-disabled):** $\lambda = 0$, so $\Phi_{\mathrm{fatigue}}(i)=1$ for every donor. All other ranking, eligibility, feasibility, dispatch, and stopping rules are identical.

Both arms receive the same emergency-arrival schedule, donor profiles, request requirements, synthetic travel conditions, and latent response variates within each timeline. Policies may diverge after the first invitation because earlier contacts change later ranking and fulfilment; those downstream differences are recorded, not forced to match.

Independent cluster for uncertainty: **the timeline**. Multiple emergencies within a timeline are dependent. Paired differences are Arm A minus Arm B, with a scenario-level paired bootstrap 95% CI (`n_resamples=2000`) and Wilcoxon signed-rank tests on the timeline means. No formal non-inferiority margin was pre-specified; fulfilment results are descriptive paired differences only.

Primary endpoints (predeclared): lookback-repeat notification rate, horizon repeat rate (≥2 alerts), heavy-contact rate (≥5 alerts), max alerts per donor, mean alerts per donor (full pool), Alert Fatigue Index, deadline fulfilment rate, and mean time to fulfilment among deadline-met emergencies. Holm–Bonferroni adjustment is applied only to that primary family. Remaining metrics and the λ/window grid are exploratory.

## 3. Configuration

| Parameter | Value |
|---|---|
| Timelines | 20 |
| Horizon | 30.0 days |
| Emergency rate | 2.5 per day (exponential inter-arrival) |
| Emergencies per arm (realized) | 1554 |
| Matched episode runs (A+B) | 3108 |
| Donor pool size | 2000 sampled Kaggle profiles (same pool on every timeline) |
| Dispatch | wave 1 size $\lceil 3Q\rceil$, subsequent $\lceil 2\times$shortfall$\rceil$ |
| Wave timeout / max waves | 15 min / 10 |
| Observation horizon | 1440 min |
| Travel time | `TravelTimeEngine` network-grid distance with congestion by hour of day; missing/non-finite ETA is infeasible (not treated as 0) |
| Fatigue formula | $\Phi=\exp(-\lambda N_{\mathrm{recent\ alerts}})$ as in `UDIEngine._fatigue_discount` |
| Lookback | closed interval $[T-W, T]$ at ranking time $T=$ request timestamp |
| Schedule seed | `seed + 1000 + timeline_id * 37` |
| Response-variate seed | `seed + timeline_id * 10000 + scenario_index * 10` |

Urgency mix and unit ranges follow the existing synthetic generator (`CRITICAL`/`URGENT`/`SCHEDULED` with horizons 60/120/240 minutes).

## 4. Data provenance

| Class | Contents |
|---|---|
| Kaggle-derived | city, blood group, months since first donation, number of donations, pints donated, historical `availability` label (not used as emergency response) |
| Synthetic | donor IDs, coordinates, eligibility (static 90% draw), `synthetic_response_propensity` ~ Beta(2,5) independent of availability, emergency times/sites/demand, travel times, response accept/latency, congestion |
| Directly measured in this run | dispatched notifications, fulfilment flags, times, shortfall, late arrivals, rolling alert counts |
| Untested with real emergency recruitment data | whether donors actually respond less after repeated alerts; operational travel times; true eligibility after donation |

This experiment uses the same ranking input as Phase 4/5: `synthetic_response_propensity` is both the ranker's $P(i)$ and the simulator's Bernoulli parameter. That is optimistic information relative to a field system. The ranker does not observe future accept/decline draws, latencies, or arrivals. Kaggle `availability` is not reinterpreted as recruitment response.

Response **probability** is not reduced by fatigue in this study. `StochasticResponseModel` can inflate latency by `fatigue_mult ** recent_alerts`, but the simulator does not pass `recent_alerts` (default 0). The study therefore measures **notification allocation and burden**, not a proven psychological response to repeated alerts.

## 5. Metric definitions

- **Total notifications / timeline:** $\sum_e N^{\mathrm{disp}}(e)$, counted from `select_batch` dispatches (not from the ranked list).
- **Mean/median notifications per donor:** over the full pool of $N$ donors, including zeros.
- **% donors with ≥2 (or ≥5) notifications:** $100\times|\{i: N_i\ge k\}|/N$ over the 30-day horizon.
- **Lookback-repeat notification rate:** number of dispatches for which the same donor already had ≥1 earlier dispatch in $[t-W, t)$, divided by all dispatches. Denominator is notifications, not donors.
- **% donors with a lookback repeat:** $100\times$ donors with ≥1 lookback-repeat dispatch $/N$.
- **Notifications per deadline-fulfilled emergency:** (timeline total alerts) / (count of deadline-met emergencies). **NaN** if a timeline has zero deadline fulfilments; those timelines are dropped from the paired mean of this metric.
- **Alert Fatigue Index:** $\mathrm{AFI}=(\sum N^{\mathrm{disp}})/(\sum Q)$ using the existing definition (alerts / units required), pooled within a timeline.
- **Gini:** Gini coefficient of per-donor notification counts over the full pool (0 = equal alerts, including zeros).
- **Deadline fulfilment rate:** $100\times$ mean of `quota_met_on_time` over emergencies in the timeline.
- **Eventual fulfilment rate:** $100\times$ mean of `quota_met` (arrivals within the 1440 min observation window).
- **Unfulfilled counts/proportions:** emergencies with the corresponding quota flag false.
- **Time to fulfilment (deadline-met):** mean of `Time_To_Fulfill_Deadline` **only among emergencies that met the deadline**. This is a conditional mean; it is not a mean over all emergencies.
- **Time to fulfilment (eventual):** analogous conditional mean among eventually fulfilled emergencies.
- **Late-arrival rate / shortfall ratio:** $\sum$ late arrivals / $\sum Q$ and $\sum$ shortfall / $\sum Q$.

## 6. Limitations (omitted longitudinal clinical constraints)

- No post-donation deferral: a donor who arrives may be contacted again in a later emergency if still synthetically eligible.
- Eligibility is a static synthetic flag, not an 8-week clinical rule updated over the horizon.
- No haemoglobin, infection-marker, or age screens.
- `simulated_current_availability` is not used as a hard filter (same as Phase 4/5).
- Adapter Poisson `recent_alert_count` on profiles is ignored; every timeline starts at zero historical alerts.
- Within an emergency, `select_batch()` still forbids contacting the same donor twice.

## 7. Primary paired results (timeline means)

Realized episode counts: Arm A `1554`, Arm B `1554` (must match).

| Metric | Arm A mean | Arm B mean | A − B (95% CI) | Wilcoxon $p$ | Holm $p$ (primary family) |
|---|---:|---:|---|---:|---:|
| Lookback-window repeat-notification rate (repeats / all alerts) | 0.4088 | 0.5686 | -0.1598 (95% CI -0.1667 to -0.1531) | 1.9073e-06 | 1.5259e-05 |
| Percentage of donors with ≥2 alerts over the horizon (%) | 13.9000 | 11.4500 | 2.4500 (95% CI 2.1450 to 2.8101) | 8.8324e-05 | 4.8279e-04 |
| Percentage of donors with ≥5 alerts over the horizon (%) | 1.5750 | 3.2675 | -1.6925 (95% CI -1.8700 to -1.5250) | 1.9073e-06 | 1.5259e-05 |
| Maximum notifications received by a single donor | 7.3000 | 12.8000 | -5.5000 (95% CI -6.2000 to -4.7500) | 8.0465e-05 | 4.8279e-04 |
| Mean notifications per donor (full pool, including zeros) | 0.5522 | 0.5480 | 0.0042 (95% CI -0.0002 to 0.0085) | 6.3988e-02 | 1.9196e-01 |
| Alert Fatigue Index (total alerts / total units required) | 5.1311 | 5.0934 | 0.0377 (95% CI -0.0025 to 0.0770) | 7.0157e-02 | 1.9196e-01 |
| Deadline fulfilment rate (% of emergencies) | 99.5787 | 99.7742 | -0.1955 (95% CI -0.4679 to 0.0617) | 1.3801e-01 | 1.9196e-01 |
| Mean time to fulfilment among deadline-met emergencies (min) | 27.9978 | 27.5056 | 0.4921 (95% CI 0.1655 to 0.8182) | 1.0689e-02 | 4.2755e-02 |

### Secondary paired results

| Metric | Arm A mean | Arm B mean | A − B (95% CI) | Wilcoxon $p$ |
|---|---:|---:|---|---:|
| Percentage of donors with a lookback-window repeat (%) | 11.7975 | 10.4750 | 1.3225 (95% CI 1.0049 to 1.6500) | 1.0306e-04 |
| Mean notifications per contacted donor | 1.9740 | 2.5871 | -0.6131 (95% CI -0.6502 to -0.5735) | 1.9073e-06 |
| Gini coefficient of notification counts over the full pool | 0.8100 | 0.8758 | -0.0658 (95% CI -0.0711 to -0.0609) | 1.9073e-06 |
| Eventual quota fulfilment rate (% of emergencies) | 99.6482 | 99.7742 | -0.1260 (95% CI -0.3771 to 0.1219) | 2.7332e-01 |
| Mean time to fulfilment among eventually-met emergencies (min) | 28.0262 | 27.5056 | 0.5206 (95% CI 0.1821 to 0.8456) | 9.4357e-03 |
| Proportion of emergencies missing the deadline | 0.0042 | 0.0023 | 0.0020 (95% CI -0.0006 to 0.0047) | 1.3801e-01 |
| Mean notifications dispatched per emergency | 14.2034 | 14.0916 | 0.1119 (95% CI 0.0013 to 0.2176) | 5.8573e-02 |
| Notifications per deadline-fulfilled emergency | 14.2640 | 14.1239 | 0.1402 (95% CI 0.0123 to 0.2637) | 4.0136e-02 |
| Late arrivals / units required | 0.0002 | 0.0000 | 0.0002 (95% CI 0.0000 to 0.0007) | 3.1731e-01 |
| Unfulfilled units / units required | 0.0020 | 0.0015 | 0.0005 (95% CI -0.0005 to 0.0014) | 2.7332e-01 |

## 8. Interpretation (honest, in-silico only)

Fatigue-aware ranking **reduced concentration of alerts** on the most-contacted donors: mean max alerts per timeline 7.30 vs 12.80 (difference -5.5000 (95% CI -6.2000 to -4.7500)), and heavy-contact prevalence 1.575% vs 3.267% (difference -1.6925 (95% CI -1.8700 to -1.5250)).

The share of the pool receiving **two or more** alerts over the full horizon moved in the **opposite** direction: 13.900% (Arm A) vs 11.450% (Arm B), difference 2.4500 (95% CI 2.1450 to 2.8101). That pattern is consistent with load-spreading: down-ranking recently notified donors sends later invitations to previously unused donors, so more people receive a small number of alerts while fewer people receive a large number.

Lookback-window repeat-notification rate: Arm A 0.4088 vs Arm B 0.5686, difference -0.1598 (95% CI -0.1667 to -0.1531).

Deadline fulfilment: Arm A 99.5787% vs Arm B 99.7742%, difference -0.1955 (95% CI -0.4679 to 0.0617). No non-inferiority margin was declared; this interval should be read as a paired difference, not as proof of clinical non-inferiority.

Conditional mean time to fulfilment among **deadline-met** emergencies: Arm A 27.998 min vs Arm B 27.506 min, difference 0.4921 (95% CI 0.1655 to 0.8182). AFI (alerts / units required): Arm A 5.1311 vs Arm B 5.0934, difference 0.0377 (95% CI -0.0025 to 0.0770).

The 95% CI for the deadline-fulfilment difference includes 0.

Phase 4 and Phase 5 single-episode measurements were not recomputed and are not replaced by these results.

## 9. Sensitivity (exploratory)

Grid: $\lambda\in[0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6]$, lookback $\in[7.0, 14.0, 30.0]$ days, first 5 timelines, **same response-variate seeds as the primary arms**. $\lambda=0$ is ranking-equivalent to Arm B regardless of lookback. No multiplicity-adjusted inference is claimed for this grid.

```
 Lambda_Fatigue  Lookback_Days  Mean_Deadline_Fulfilment_Rate  Mean_Time_To_Deadline_Met  Mean_Alert_Fatigue_Index  Mean_Pct_Repeat_Notified  Mean_Pct_Heavy_Notified  Mean_Max_Alerts_Single_Donor  Mean_Gini_Notification_Inequality  Mean_Lookback_Repeat_Notification_Rate  Mean_Unfulfilled_Deadline_Proportion  N_Timelines
           0.00            7.0                      99.587629                  27.911800                  5.129998                     12.45                     3.95                          13.8                           0.869977                                0.476994                              0.004124            5
           0.00           14.0                      99.587629                  27.911800                  5.129998                     12.45                     3.95                          13.8                           0.869977                                0.599734                              0.004124            5
           0.00           30.0                      99.587629                  27.911800                  5.129998                     12.45                     3.95                          13.8                           0.869977                                0.629489                              0.004124            5
           0.05            7.0                      99.587629                  27.912640                  5.117761                     12.79                     3.88                          13.0                           0.864140                                0.456201                              0.004124            5
           0.05           14.0                      99.587629                  27.794608                  5.107377                     13.06                     3.79                          12.4                           0.858479                                0.572357                              0.004124            5
           0.05           30.0                      99.587629                  27.920959                  5.111570                     13.37                     3.68                          12.0                           0.853275                                0.600570                              0.004124            5
           0.10            7.0                      99.587629                  27.666798                  5.084070                     13.23                     3.71                          11.8                           0.857087                                0.428743                              0.004124            5
           0.10           14.0                      99.587629                  27.760220                  5.099352                     13.65                     3.51                          10.6                           0.844980                                0.542605                              0.004124            5
           0.10           30.0                      99.587629                  27.936260                  5.113729                     14.35                     3.11                           9.4                           0.833179                                0.568510                              0.004124            5
           0.20            7.0                      99.309851                  27.850710                  5.108226                     13.88                     3.30                          10.8                           0.840265                                0.374210                              0.006901            5
           0.20           14.0                      99.309851                  28.125382                  5.148115                     14.88                     2.80                           8.2                           0.818784                                0.488071                              0.006901            5
           0.20           30.0                      99.309851                  28.321978                  5.149603                     15.75                     1.89                           7.0                           0.795527                                0.509245                              0.006901            5
           0.30            7.0                      99.309851                  28.165996                  5.163075                     14.75                     2.90                           9.4                           0.824527                                0.329377                              0.006901            5
           0.30           14.0                      99.309851                  28.538721                  5.205141                     15.62                     2.01                           7.2                           0.795024                                0.436728                              0.006901            5
           0.30           30.0                      99.309851                  29.183551                  5.230981                     16.78                     1.01                           6.6                           0.765172                                0.467201                              0.006901            5
           0.45            7.0                      99.309851                  29.039895                  5.244089                     15.48                     2.35                           8.2                           0.803834                                0.272143                              0.006901            5
           0.45           14.0                      99.587629                  29.561470                  5.303316                     16.52                     1.23                           6.2                           0.765056                                0.379682                              0.004124            5
           0.45           30.0                      99.587629                  30.819455                  5.397236                     17.75                     0.47                           5.4                           0.727259                                0.421700                              0.004124            5
           0.60            7.0                      99.587629                  29.997238                  5.321059                     16.19                     1.87                           7.4                           0.787212                                0.226670                              0.004124            5
           0.60           14.0                      99.587629                  30.404342                  5.391782                     16.93                     0.73                           5.6                           0.742035                                0.335238                              0.004124            5
           0.60           30.0                      99.587629                  31.387701                  5.462956                     17.74                     0.32                           5.4                           0.701237                                0.388186                              0.004124            5
```

A larger $\lambda$ is not automatically better: inspect both max/repeat burden and deadline fulfilment in the table above before any tuning recommendation.

## 10. Reproducibility and tests

Re-run:

```
python experiments/exp_phase6_2_longitudinal_fatigue.py --timelines 20 --days 30.0 --rate 2.5 --donors 2000 --seed 42
```

Independent recomputation of timeline aggregates from `episode_results.csv` and paired differences from `timeline_metrics.csv`: **passed** (`metric_verification.json`).

Focused tests live in `tests/test_longitudinal_fatigue.py`. Existing Phase 4/5 tests remain in the project test suite and must not be weakened.

## 11. Artifacts

- `episode_results.csv` — one row per emergency × arm
- `timeline_metrics.csv` — one row per timeline × arm
- `paired_comparisons.csv`
- `sensitivity_results.csv`
- `donor_timeline_summary.csv` — per-donor alert totals for every timeline × arm
- `donor_notification_history.csv` — event-level dispatches for timeline 0 only (audit sample)
- `request_schedule.csv` — shared exogenous emergency schedule
- `scenario_manifest.json`, `run_log.txt`, `metric_verification.json`
- `figures/` — burden, fulfilment, scatter, and sensitivity plots
- `event_traces/` — short traces for the first two timelines

## 12. Recommended next phase

If a later phase introduces behavioural response decay, it must be specified separately from ranking-side $\Phi_{\mathrm{fatigue}}$ and must not be fitted on these same simulation outcomes. A documented donation cooldown would change both burden and fulfilment and should be added only with an explicit clinical rule. This report does not proceed to paper writing or deployment.
