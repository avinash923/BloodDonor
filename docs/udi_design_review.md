# UDI Mathematical Design & Review Report

**Date:** 2026-10-10
**Module:** `src/ranking/udi_engine.py`

## 1. Current Implementation Inspection

**File Path:** `src/ranking/udi_engine.py`
**Target Class:** `UDIEngine`
**Target Function:** `score_candidate`

The current code implements an **additive** weighting scheme combined with a **multiplicative** fatigue discount.

**Exact Mathematical Implementation:**
1. **Urgency Component**: `urgency_comp = w_urgency * U(e)` where `U(e)` is a scalar (e.g., 1.0 for CRITICAL).
2. **Timeliness Component**: `timeliness_comp = w_timeliness * max(0, 1 - (travel_time / T_crit))`
3. **Probability Component**: `probability_comp = w_probability * P(response)`
4. **Weighted Sum**: `weighted_sum = urgency_comp + timeliness_comp + probability_comp`
5. **Fatigue Discount**: `fatigue = exp(-lambda * N_recent_alerts)`
6. **Final UDI**: `udi = clip(weighted_sum * fatigue, 0.0, 1.0)`

**Constraints & Hard Gates:**
- **Eligibility**: Ineligible donors immediately receive `UDI = 0.0`.
- **Feasibility**: Donors whose travel time exceeds the effective horizon (`T_crit * (1 - safety_margin)`) immediately receive `UDI = 0.0`.
- **Tie-Breaking**: The python `list.sort(key=lambda r: r.udi_score, reverse=True)` uses a stable sort. If scores are tied, the original array order (arbitrary) is preserved.

---

## 2. Mathematical Comparison: Additive vs. Multiplicative

### A. Existing Additive Formula
$UDI_{add}(d) = (W_u U(d) + W_t T(d) + W_p P(d)) \times \Phi_{fatigue}(d)$

- **Effect on Ranking:** This is a *compensatory* model. A very low score in probability can be offset by being very close to the hospital (high $T(d)$) and a high urgency weight. 
- **Zero Components:** A zero in one component (e.g., $P(d)=0$) does not zero out the entire score; the candidate relies on the other weights.

### B. Proposed Multiplicative Formula
$UDI_{mult}(d) = P(d) \times T(d) \times C_{comp}(d) \times E_{elig}(d) \times W_{urgency} \times \Phi_{fatigue}(d)$

- **Effect on Ranking:** This is a *strict intersection* model. A donor must score reasonably well on all axes.
- **Zero Components:** If any single factor drops to zero, the entire score becomes zero.
- **Role of Compatibility ($C$) & Eligibility ($E$):** These are binary variables $\{0, 1\}$. If a candidate passes both checks (both equal 1), they act as identity multipliers ($x \cdot 1 \cdot 1 = x$) and thus mathematically do not differentiate valid candidates.
- **Role of Urgency Weight ($W_{urgency}$):** **Critical flaw in intra-request ranking.** For any single emergency request, the urgency is constant. Multiplying every donor's score by the exact same positive constant (e.g., $W_{urgency} = 0.7$) simply scales all scores linearly. It **does not change the relative ordering** of the donors.

---

## 3. Normalization and Parameter Recommendations

- **Normalization:** All core continuous metrics ($P, T, \Phi$) are properly bounded to $[0, 1]$.
- **Handling Missing Travel Time:** If a network travel-time estimate fails or is missing, the system should fall back to a Haversine geometric estimate with an added uncertainty penalty (e.g., multiply by 1.35), rather than assuming $T=0$ (which zeroes the multiplicative score).
- **Urgency Representation:** Because a scalar multiplier does not affect donor ordering within a single request, urgency should strictly dictate the **shape of the timeliness curve**. CRITICAL requests have a steep drop-off (e.g., $T_{crit} = 60$ min), while SCHEDULED requests have a gentle drop-off ($T_{crit} = 240$ min). This successfully reorders donors based on urgency without redundant scalar multiplication.

---

## 4. Potential Weaknesses and Edge Cases

- **Additive Weakness (Current):** Risks inviting highly improbable donors simply because they live next door to the hospital.
- **Multiplicative Weakness (Proposed):** Highly sensitive to zeroes. If the ML model outputs $P=0.0$ for a donor, they are entirely excluded even if they are the only person in the city with the correct blood type. (Mitigation: floor probability at $0.01$).
- **Tie-Breaking:** Neither formula explicitly defines a tie-breaker. (Mitigation: explicitly secondary-sort by travel time ascending).

---

## 5. Controlled Experimental Design (EXP-8)

To determine the empirical superiority of either model, we will design an A/B test using the `ScenarioManager` framework established in Phase 1.

**Methodology:**
1. **Scenarios:** Load 500 identical synthetic emergency scenarios (varying CRITICAL, URGENT, SCHEDULED) and identical donor pools with deterministic random seeds.
2. **Control Arm:** Rank donors using $UDI_{add}$ and execute the dynamic wave simulation.
3. **Treatment Arm:** Rank donors using $UDI_{mult}$ and execute the dynamic wave simulation.
4. **Metrics Evaluated:**
   - **Fulfillment %**: Requests that met their unit quota before $T_{crit}$.
   - **Time-to-Fulfill**: Mean time until required units arrive.
   - **Alert Fatigue Index**: Total notifications sent / successful arrivals.
   - **Shortfall Ratio**: Unmet units / required units.
5. **Statistical Test:** Paired t-tests (for continuous metrics like time-to-fulfill) and Wilcoxon signed-rank tests (for distributions with heavy tails, like alerts sent) matched exactly on `Scenario_ID`.

---

## 6. Recommended Primary Formulation

I recommend adopting the **Multiplicative Formulation**, but with a mathematical correction regarding Urgency. 

**Recommended Final UDI Formula:**
$$ UDI_{final}(d) = P_{resp}(d) \times \left(1 - \frac{\tau_d}{T_{crit}(Urgency)}\right) \times \Phi_{fatigue}(d) $$

*Reasons:*
1. It strictly enforces that a donor must be probable, timely, and unfatigued.
2. It removes the mathematical redundancy of the $W_{urgency}$ multiplier.
3. Compatibility and Eligibility are implemented as hard gates prior to scoring, rather than terms in the equation.

---

## 7. Decisions Requiring Researcher Approval

Before I write any code to implement Phase 2, please approve or modify the following decisions:

1. **Urgency Multiplier Removal:** Do you approve removing the static scalar multiplier for Urgency, instead relying on Urgency to define the strictness of the Timeliness curve ($T_{crit}$)?
2. **Probability Floor:** Do you approve flooring $P_{resp}$ at `0.01` to prevent absolute zeroing of the multiplicative equation for nearby donors?
3. **Tie-Breaking:** Do you approve explicitly tie-breaking identical UDI scores using travel time (nearest first)?
