## Income modeling from last 12 months (net_salary + net_benefits) — brainstorm

## Problem statement
We receive the last **12 running months** of:

- `net_salary` (regular salary + possible bonuses / variable pay)
- `net_benefits` (e.g., pension / parental leave support)

Current approach (“standard haircut mean”) is too blunt because:

- **Annual raise / decrease**: we want to estimate *current* salary from months *since the change*, rather than average old+new levels.
- **Unstable salary / bonuses**: we want to detect “variable income patterns” and route them to a different process (instead of assuming stability).
- **Special cases**: e.g. **parental leave** for 1–3 months (salary near zero, benefits present) should not incorrectly depress the long-run salary estimate.

---

## Proposed outputs (per applicant)
- **income_segment**: one of `stable_salary`, `recent_level_shift`, `variable_salary_bonus`, `leave_or_benefit_substitution`, `mixed_or_unclear`
- **estimated_monthly_salary_current**
- **estimated_monthly_benefits_current**
- **estimated_monthly_total_current** (salary + benefits, if your DI uses total net income)
- **stability_score** / **volatility_score**
- **flags / reason_codes**: explain why a segment was chosen (auditability)
- **confidence**: low/medium/high (can start as heuristic, later learned)

---

## General principles (to keep it compliance-friendly)
- Use **transparent, monotonic statistics** (means, medians, quantiles, simple thresholds).
- Avoid “behavioral inference” logic that is hard to defend; keep segmentation tied to observed income patterns (level shifts, volatility, benefits substitution).
- Always store **reason codes** and underlying stats.

---

## Step 0: Pre-processing / hygiene
- **Chronology**: ensure month order is correct (oldest → newest).
- **Missing months**: decide how to treat gaps (drop, impute 0, or mark “low confidence”).
- **Negative or extreme values**: validate; winsorize if needed (document rules).
- **Totals**: compute `net_total = net_salary + net_benefits` per month.

---

## Step 1: Compute simple features (fast and high-signal)
Let `s[i]` be monthly `net_salary` and `b[i]` be `net_benefits`, for i=1..12 (12 = most recent).

### Salary stability / volatility
- **median_salary** = median(s)
- **iqr_salary** = IQR(s) = q75(s) - q25(s)
- **cv_salary** = std(s) / max(mean(s), eps)
- **spike_count** = number of months where s[i] > median_salary + k * iqr_salary (k ~ 1.5–3)
- **low_count** = months where s[i] < median_salary - k * iqr_salary
- **bimodality hint**: are there two clusters? (simple proxy: spike_count >= 2 and low_count >= 2)

### “Recent vs history” indicators (for raises/decreases)
- **mean_last3** vs **mean_prev9**
- **median_last3** vs **median_prev9**
- **ratio_last3_to_prev9**

### Benefits substitution / leave indicators
- **benefit_months**: count of months with b[i] > benefit_min
- **salary_near_zero_months**: count of months with s[i] < salary_min
- **substitution_months**: months where s[i] < salary_min AND b[i] > benefit_min
- **substitution_run_length**: longest consecutive run of substitution_months (1–3 is common for leave)

---

## Step 2: Segment the applicant (rule-based starting point)
This is a pragmatic, explainable first version. (Later, learn the segment as a classifier.)

### Segment A: `leave_or_benefit_substitution`
Trigger if there is a short recent run of substitution months:

- substitution_run_length in {1,2,3} and occurs in the last N months (e.g., last 4–6 months)

Reason codes:
- `salary_near_zero_with_benefits_present`
- `short_consecutive_substitution_run`

### Segment B: `recent_level_shift` (raise or decrease)
Detect a step change in salary level.

Simple approach (transparent):
- Scan candidate split points `t` (e.g., t=4..10), compute:
  - pre = s[1..t], post = s[t+1..12]
  - score = |median(post) - median(pre)| / max(iqr(s), eps)
- Choose split with max score; call it a level shift if:
  - max_score > threshold (e.g., 1.5–3) AND
  - post segment has at least M months (e.g., 3) AND
  - post segment is not ultra-volatile (cv(post) below a cap)

Reason codes:
- `salary_level_shift_detected`
- `post_shift_months_count_ok`

### Segment C: `variable_salary_bonus`
Trigger when salary is volatile / spiky:
- cv_salary > cv_threshold OR spike_count >= spike_threshold OR bimodality proxy triggers

Reason codes:
- `salary_high_volatility`
- `salary_spike_pattern`

### Segment D: `stable_salary`
Trigger when none of the above and salary is stable:
- cv_salary <= stable_cv AND spike_count small

Reason codes:
- `salary_stable`

### Segment E: `mixed_or_unclear`
Fallback if multiple segments trigger or data quality is poor:
- missing months, contradictory signals, or both high benefits + high salary volatility

Reason codes:
- `mixed_signals_or_low_data_quality`

---

## Step 3: Estimate “current income” per segment

### A) Stable salary (`stable_salary`)
Replace “haircut mean” with a robust “current level” estimate:
- **Option 1**: `estimated_salary = median(last_3_or_6_months)`
- **Option 2**: exponentially weighted mean over last 6–12 months (more weight on recent)
- **Option 3**: trimmed mean (drop top/bottom 1 month)

Benefits:
- `estimated_benefits = median(nonzero b in last 6–12 months)` (or 0 if usually none)

### B) Level shift (`recent_level_shift`)
Estimate salary from **post-shift months**:
- `estimated_salary = median(post_shift_months)`
- Guardrail: if the post-shift window is small (e.g., only 2–3 months), blend:
  - `estimated_salary = w * median(post) + (1-w) * median(pre)` where w increases with post length

If it’s a **decrease**, this naturally reduces the estimate (safer).

Benefits:
- same as stable, but consider excluding substitution months if present.

### C) Variable salary / bonuses (`variable_salary_bonus`)
Key idea: split into **base salary** + **variable pay** and treat variable pay conservatively.

Heuristic decomposition:
- Identify “typical months” as those within [q25, q75] (or within median ± k*IQR).
- **base_salary = median(typical months)**
- **bonus_component = mean(max(0, s[i] - base_salary))** over spike months
- **included_variable_pay = haircut * bonus_component** (haircut could be strong, e.g., 0–50%)

Then:
- `estimated_salary = base_salary + included_variable_pay`

#### Proposed “different process” for these cases (instead of standard haircut mean)
Pick one (or combine):

- **Variable-income flow (auto, conservative)**:
  - Use `base_salary` only (or base + small haircut variable) for DI
  - Set a **separate product limit cap** for variable-income applicants
  - Show reason code: `variable_income_conservative_assessment`

- **Enhanced verification flow**:
  - Request extra documents (payslips / employment contract / employer statement)
  - If verified, allow higher inclusion of variable pay
  - Decision outcome could become “maybe” / “needs info” depending on your decision taxonomy

- **Manual review / referral flow** (if volumes allow):
  - Route to underwriting queue when volatility_score is above a high threshold
  - Underwriter can confirm whether bonuses are contractual/recurring

### D) Leave or benefits substitution (`leave_or_benefit_substitution`)
Goal: don’t let 1–3 leave months destroy the salary estimate.

Two separate estimates can be useful:

- **Current total income** (what they get now):
  - `current_total = median(net_total of last 1–3 months)`
- **Expected normal salary** (what they usually earn when not on leave):
  - Exclude substitution months, estimate salary from non-substitution months:
    - `estimated_salary_normal = median(s where not substitution)`

Product/policy options:
- If underwriting focuses on ability to pay **immediately**, use `current_total`.
- If loan is long-term and you can assume return-to-work, use:
  - `min(estimated_salary_normal, cap)` or blend with current_total based on policy.

Reason codes:
- `benefits_substitution_detected`
- `salary_estimated_excluding_substitution_months`

---

## Scoring: “stability / volatility score” (for routing)
Simple first version:

- volatility_score = weighted sum of:
  - min(1, cv_salary / cv_cap)
  - min(1, spike_count / spike_cap)
  - min(1, |median_last3 - median_prev9| / shift_cap)

Routing:
- low score → stable/level-shift estimator
- medium score → variable-income flow (conservative)
- high score → enhanced verification / manual review

---

## Calibration and evaluation ideas
- Backtest on historical data:
  - compare estimated salary vs “ground truth” (if available) or subsequent observed salary
  - measure downstream DI accuracy and credit outcomes
- Track:
  - conversion impact (limits shown)
  - default/loss impact
  - referral/verification rates (capacity planning)
- Segment-specific thresholds (often necessary): income level buckets, employment types, bureau thin/thick.

---

## Open questions to finalize the approach
- Do you need to estimate **salary only** or **total income (salary+benefits)** for DI?
- Do you have a way to label **bonus months** (salary descriptors), or only net amounts?
- How many “MAYBE/manual” cases can you operationally handle?
- For leave cases, is it acceptable to assume return-to-work (policy/regulatory)?

