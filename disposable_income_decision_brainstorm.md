## Disposable income (declared vs modeled) — decisioning brainstorm

## Problem recap
You have two disposable income (DI) values per applicant:

- **Declared DI**: computed from applicant-declared income/expenses/debts.
- **Modeled DI**: independently modeled from bureau data.

They can differ:

- **Minor differences** (often rounding to 100s/1000s) → should still be **YES** (auto-approve, no referral).
- **Material differences** (undeclared debts, overstated income) → should be **MAYBE** (refer).
- **Model imperfections**: modeled DI can be biased/noisy; you’d like to avoid losing customers if declared DI is slightly higher but still plausible. Ideally show the **higher limit** when safe.

This document lists implementable strategies + a Python skeleton.

---

## Desired outputs
For each case produce:

- **decision_outcome**: `"yes"` or `"maybe"` (you can add `"no"` later if needed).
- **di_for_limit**: the DI value used for max loan amount calculation (could be declared, modeled, or a fused/capped value).
- **reason_codes**: list of strings explaining why (for auditability, monitoring, and tuning).
- **metrics**: diff features (abs/relative/log-ratio), consistency signals, etc.

---

## Key design ideas

### 1) Treat the two DIs as noisy measurements of the same latent “true DI”
Both sources have error:

- **Declared DI error** is often dominated by *rounding* and self-report noise.
- **Modeled DI error** depends on model quality and bureau coverage and may be *systematically biased* for certain segments.

So the “right” decision is usually not “pick declared or modeled”, but:

- detect **inconsistency risk** (refer if too inconsistent), and
- select a **safe-but-not-too-low** DI for limit when consistent.

---

## Features to compute (high-signal, easy to implement)
Given `declared_di` and `modeled_di`:

- **abs_diff**: `declared_di - modeled_di`
- **abs_diff_mag**: `abs(abs_diff)`
- **rel_diff**: `abs_diff / max(abs(modeled_di), eps)`
- **log_ratio**: `log(max(declared_di, eps) / max(modeled_di, eps))` (more stable across scales)
- **sign**: is declared higher than modeled?

Model uncertainty (if available):

- **modeled_sigma**: estimated model std or prediction interval width
- **z_score**: `(declared_di - modeled_mean) / modeled_sigma`

Optional (very useful if you have them):

- **component-level diffs** (income vs expenses vs debts) rather than only DI
- **bureau debt flags** (new debt lines, utilization, etc.)
- **applicant friction metrics** (drop-off sensitivity to a lower limit)

---

## Strategy A — Simple, robust rule-based tolerance bands (recommended starting point)
Define a “minor difference” tolerance that scales with magnitude (tolerating typical self-report noise without relying on source-specific heuristics).

### A1) Tolerance-band consistency check: `minor_tol = max(abs_floor, rel_floor * scale)`
Where:

- `abs_floor` protects low-income cases (e.g., 50–200 currency units)
- `rel_floor` protects high-income cases (e.g., 2%–10%)
- `scale = max(abs(declared_di), abs(modeled_di))`

### A2) Decision rule
- If `abs_diff_mag <= minor_tol`: **YES**
- Else:
  - if declared is **much higher** than modeled → **MAYBE**
  - if declared is much **lower** than modeled → often still **YES** (conservative), but track for potential data quality

### A3) Which DI to use for the front-end limit (when decision is YES)
Goal: don’t lose customers due to model being slightly low, but don’t overshoot.

Common “safe uplift” patterns:

- **Use declared DI, but cap uplift vs modeled**:
  - `di_for_limit = min(declared_di, modeled_di + uplift_cap)`
  - `uplift_cap` can be `minor_tol` or a separate tuned value
- **Use max(declared, modeled) only when within tolerance**:
  - if consistent → `di_for_limit = max(declared_di, modeled_di)`
  - else (even if still yes) → `di_for_limit = min(declared_di, modeled_di)`

Recommended conservative default:

- if `declared_di >= modeled_di` and within tolerance → use `min(declared_di, modeled_di + minor_tol)`
- otherwise use `declared_di` (declared lower is already conservative)

---

## Strategy B — Uncertainty-aware rules (if your bureau model can output confidence)
If the modeled DI comes with an uncertainty estimate (`modeled_sigma` or prediction interval):

- compute `z = (declared_di - modeled_mean) / max(modeled_sigma, eps)`
- treat as **consistent** if `|z| <= z_yes` (e.g., 1.5–2.5)
- refer if `z > z_referral` and declared is higher (e.g., > 3)

For front-end limit:

- allow uplift up to a confidence bound:
  - `di_for_limit = min(declared_di, modeled_mean + k_offer * modeled_sigma)`

This aligns “uplift allowed” with model uncertainty: if the model is unsure, you allow more room.

---

## Strategy C — Bayesian fusion (best long-term; still simple to implement)
Model:

- `declared ~ Normal(true_di, sigma_declared)`
- `modeled ~ Normal(true_di, sigma_modeled)`

Then the posterior mean is a weighted average (higher weight to smaller sigma).

Use cases:

- **di_for_limit** = posterior mean (or a conservative quantile like 20th percentile)
- **MAYBE referral** if posterior probability of “declared is overstating by > X” is high

This gives a principled knob to trade off “customer conversion” vs “risk”.

---

## Strategy D — Learn thresholds / objective-optimal decisioning from data
Once you have historical outcomes (defaults, manual review outcomes, repayment behavior, conversion/drop-off):

- learn a model for **P(misreporting | features)** or **expected loss**
- choose decision `"maybe"` when expected value of referral exceeds auto-approval
- choose `di_for_limit` that maximizes profit subject to constraints (risk, fairness, policy)

This is how you make it “optimal” in a measurable sense:

- objective: maximize expected profit / approval / conversion
- constraints: loss rate, regulatory limits, fairness constraints, referral capacity

---

## Practical monitoring & tuning loop (important)
Whatever strategy you start with, bake in:

- **reason codes** (diff too large, declared>modeled, low modeled, etc.)
- **metrics logs** (abs/rel diff, sigma, z-score)
- **A/B testing** for “uplift cap” policy (customer conversion vs loss)

---

## Python skeleton (rule-based + optional uncertainty)
Below is a compact structure you can implement immediately, then calibrate thresholds with real data.

```python
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class DecisionConfig:
    # Minor-difference tolerance
    abs_floor: float = 100.0          # currency units
    rel_floor: float = 0.05           # 5%

    # Referral thresholds (extra guardrails)
    abs_referral: float = 800.0       # currency units
    rel_referral: float = 0.25        # 25%

    # Uncertainty-aware knobs (if modeled_sigma provided)
    z_yes: float = 2.0
    z_referral: float = 3.0
    k_offer: float = 2.0              # uplift bound in sigmas


@dataclass(frozen=True)
class DecisionResult:
    decision_outcome: str             # "yes" | "maybe"
    di_for_limit: float
    reason_codes: tuple[str, ...]
    abs_diff: float
    rel_diff: float
    minor_tol: float
    z_score: Optional[float]


def _is_valid_number(x: Optional[float]) -> bool:
    return x is not None and isfinite(x)


def compute_minor_tolerance(declared_di: float, modeled_di: float, cfg: DecisionConfig) -> float:
    scale = max(abs(declared_di), abs(modeled_di))
    tol = max(cfg.abs_floor, cfg.rel_floor * scale)
    return tol


def decide_disposable_income(
    declared_di: float,
    modeled_di_mean: float,
    modeled_di_sigma: Optional[float] = None,
    cfg: DecisionConfig = DecisionConfig(),
) -> DecisionResult:
    # Basic validation
    if not _is_valid_number(declared_di) or not _is_valid_number(modeled_di_mean):
        raise ValueError("declared_di and modeled_di_mean must be finite numbers")

    eps = 1e-9
    abs_diff = declared_di - modeled_di_mean
    rel_diff = abs_diff / max(abs(modeled_di_mean), eps)

    minor_tol = compute_minor_tolerance(declared_di, modeled_di_mean, cfg)

    z_score: Optional[float] = None
    if _is_valid_number(modeled_di_sigma) and modeled_di_sigma > 0:
        z_score = abs_diff / modeled_di_sigma

    reason_codes: list[str] = []

    # Primary consistency check
    minor = abs(abs_diff) <= minor_tol

    # Additional referral guards (focus on declared >> modeled)
    declared_much_higher = (abs_diff > cfg.abs_referral) and (abs(rel_diff) > cfg.rel_referral)
    z_ref = (z_score is not None) and (z_score > cfg.z_referral) and (abs_diff > 0)

    if minor:
        decision = "yes"
        reason_codes.append("diff_within_minor_tolerance")
    elif declared_much_higher or z_ref:
        decision = "maybe"
        reason_codes.append("declared_significantly_higher_than_modeled")
    else:
        # Declared lower than modeled (or moderately different): usually safe to auto-approve,
        # but keep a trace for data quality / tuning.
        decision = "yes"
        reason_codes.append("declared_not_significantly_higher_than_modeled")

    # Choose DI for limit (business policy)
    if decision == "yes":
        if declared_di >= modeled_di_mean:
            # Allow some uplift vs modeled, but cap it.
            if z_score is not None:
                uplift_cap = cfg.k_offer * modeled_di_sigma  # type: ignore[arg-type]
                reason_codes.append("uplift_cap_from_model_sigma")
            else:
                uplift_cap = minor_tol
                reason_codes.append("uplift_cap_from_minor_tolerance")

            di_for_limit = min(declared_di, modeled_di_mean + uplift_cap)
        else:
            # Declared is already conservative; stick to it.
            di_for_limit = declared_di
    else:
        # If referring, be conservative in what you show (or show nothing until review).
        di_for_limit = min(declared_di, modeled_di_mean)
        reason_codes.append("referred_conservative_limit")

    return DecisionResult(
        decision_outcome=decision,
        di_for_limit=float(di_for_limit),
        reason_codes=tuple(reason_codes),
        abs_diff=float(abs_diff),
        rel_diff=float(rel_diff),
        minor_tol=float(minor_tol),
        z_score=float(z_score) if z_score is not None else None,
    )
```

---

## Calibration checklist (to make it “optimal” in practice)
- Collect labeled data: manual review outcomes, detected undeclared debts, post-loan loss/default, conversion/drop-off.
- Tune:
  - `abs_floor`, `rel_floor` to hit your desired **referral rate** and **false referral rate**
  - `abs_referral`, `rel_referral` to catch true misreporting cases
  - `uplift_cap` policy to maximize conversion without increasing loss
- Segment tuning (often needed):
  - income level buckets
  - employment type
  - bureau thin vs thick file
  - existing customer vs new-to-bank

---

## Open questions (useful to answer before finalizing)
- Is DI computed from *multiple* declared fields (income, housing, other debts), and do you have those components?
- Do you have modeled DI uncertainty (sigma/quantiles), or only a point estimate?
- How noisy are declared figures in your market (typical absolute/relative error)?
- Can the front-end show a range/“up to X” vs a single limit?

