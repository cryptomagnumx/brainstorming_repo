"""
Income modeling from the last 12 months of net_salary and net_benefits.

Goal
----
Provide an explainable, compliance-friendly approach that:
- Detects recent salary level shifts (raise/decrease) and estimates the "current" level.
- Detects variable/bonus-like salary patterns and routes them to an alternative process.
- Detects short benefit-substitution periods (e.g., parental leave) and avoids depressing
  the normal salary estimate due to near-zero salary months with benefits.

This module is deliberately heuristic and transparent: medians, quantiles, CV, simple thresholds.
Tune thresholds with historical outcomes and operational capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from statistics import fmean, pstdev
from typing import Optional, Sequence


SEG_STABLE = "stable_salary"
SEG_LEVEL_SHIFT = "recent_level_shift"
SEG_VARIABLE = "variable_salary_bonus"
SEG_LEAVE_SUBSTITUTION = "leave_or_benefit_substitution"
SEG_MIXED = "mixed_or_unclear"


ROUTE_STANDARD = "standard"
ROUTE_VARIABLE_INCOME_CONSERVATIVE = "variable_income_conservative"
ROUTE_ENHANCED_VERIFICATION = "enhanced_verification"
ROUTE_MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class MonthIncome:
    """One month of net income inputs."""

    net_salary: float
    net_benefits: float


@dataclass(frozen=True)
class IncomeModelConfig:
    # --- Leave / benefit substitution detection ---
    salary_min_for_substitution: float = 50.0
    benefit_min_for_substitution: float = 50.0
    substitution_recent_window_months: int = 6  # look for substitution runs in the last N months
    substitution_run_max_months: int = 3  # typical leave run length to treat specially

    # --- Stability / variability (salary) ---
    stable_cv_max: float = 0.15
    stable_spike_months_max: int = 1
    stable_low_months_max: int = 1

    variable_cv_min: float = 0.25
    spike_k_iqr: float = 2.0  # spike if salary > median + k*IQR (when IQR > 0)
    low_k_iqr: float = 2.0  # low if salary < median - k*IQR (when IQR > 0)
    spike_rel_when_iqr_zero: float = 0.20  # fallback: spike if salary > median*(1+rel)
    low_rel_when_iqr_zero: float = 0.20  # fallback: low if salary < median*(1-rel)
    spike_months_min: int = 2
    bimodal_spike_min: int = 2
    bimodal_low_min: int = 2

    # --- Level shift detection ---
    shift_min_pre_months: int = 4
    shift_min_post_months: int = 3
    shift_max_pre_months: int = 10
    shift_score_min: float = 2.0  # |post_median - pre_median| / max(IQR_all, eps)
    shift_abs_min: float = 200.0  # require material absolute change
    shift_post_cv_max: float = 0.20  # post-shift months should be reasonably stable

    # --- Estimation windows ---
    stable_salary_recent_months: int = 6
    benefits_recent_months: int = 6
    leave_benefits_recent_months: int = 3
    leave_current_total_recent_months: int = 3
    shift_full_weight_post_months: int = 6  # post_len >= this => full weight on post segment

    # --- Variable income estimation ---
    variable_pay_inclusion: float = 0.25  # include this fraction of variable pay (0..1)

    # --- Routing based on volatility score ---
    route_variable_income_max_score: float = 0.45
    route_enhanced_verification_max_score: float = 0.75

    # --- Volatility score construction ---
    volatility_cv_cap: float = 0.40
    volatility_spike_cap: float = 4.0
    volatility_shift_abs_cap: float = 1500.0
    volatility_w_cv: float = 0.5
    volatility_w_spike: float = 0.3
    volatility_w_shift: float = 0.2


@dataclass(frozen=True)
class IncomeModelStats:
    median_salary: float
    iqr_salary: float
    cv_salary: float
    spike_count: int
    low_count: int

    median_last3: float
    median_prev9: float
    shift_abs_last3_prev9: float

    benefit_months: int
    substitution_months: int
    substitution_recent_max_run: int

    best_shift_score: float
    best_shift_split: Optional[int]  # split index: pre = [0:split), post = [split:12]
    best_shift_pre_median: Optional[float]
    best_shift_post_median: Optional[float]


@dataclass(frozen=True)
class IncomeModelResult:
    income_segment: str
    processing_route: str
    confidence: str  # "high" | "medium" | "low"
    reason_codes: tuple[str, ...]

    estimated_monthly_salary: float
    estimated_monthly_benefits: float
    estimated_monthly_total: float

    # Optional transparency values useful for leave/variable cases
    salary_normal_excluding_substitution: Optional[float]
    total_current_recent: Optional[float]
    base_salary_estimate: Optional[float]
    included_variable_pay: Optional[float]

    volatility_score: float
    stats: IncomeModelStats


def _require_12_months(months: Sequence[MonthIncome]) -> None:
    if len(months) != 12:
        raise ValueError(f"Expected 12 months of data, got {len(months)}")


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile with p in [0, 1]."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    p = min(1.0, max(0.0, p))
    k = (len(xs) - 1) * p
    f = floor(k)
    c = ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _median(values: Sequence[float]) -> float:
    return _percentile(values, 0.5)


def _iqr(values: Sequence[float]) -> float:
    return _percentile(values, 0.75) - _percentile(values, 0.25)


def _cv(values: Sequence[float], eps: float = 1e-9) -> float:
    if not values:
        return 0.0
    mu = fmean(values)
    sigma = pstdev(values)
    return float(sigma / max(abs(mu), eps))


def _longest_true_run(flags: Sequence[bool]) -> int:
    best = 0
    run = 0
    for v in flags:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _compute_volatility_score(
    *,
    cv_salary: float,
    spike_count: int,
    shift_abs_last3_prev9: float,
    cfg: IncomeModelConfig,
) -> float:
    cv_comp = min(1.0, cv_salary / max(cfg.volatility_cv_cap, 1e-9))
    spike_comp = min(1.0, spike_count / max(cfg.volatility_spike_cap, 1e-9))
    shift_comp = min(1.0, shift_abs_last3_prev9 / max(cfg.volatility_shift_abs_cap, 1e-9))

    w_sum = cfg.volatility_w_cv + cfg.volatility_w_spike + cfg.volatility_w_shift
    if w_sum <= 0:
        return 0.0
    score = (
        cfg.volatility_w_cv * cv_comp
        + cfg.volatility_w_spike * spike_comp
        + cfg.volatility_w_shift * shift_comp
    ) / w_sum
    return float(min(1.0, max(0.0, score)))


def _detect_level_shift(s: Sequence[float], cfg: IncomeModelConfig) -> tuple[bool, float, Optional[int], Optional[float], Optional[float]]:
    """
    Returns:
      (shift_detected, best_score, best_split, pre_median, post_median)
    """
    iqr_all = _iqr(s)
    denom = max(iqr_all, 1e-9)

    best_score = 0.0
    best_split: Optional[int] = None
    best_pre_m: Optional[float] = None
    best_post_m: Optional[float] = None

    # pre_len is the split index (pre = s[:pre_len], post = s[pre_len:])
    max_pre_len = min(cfg.shift_max_pre_months, len(s) - cfg.shift_min_post_months)
    for pre_len in range(cfg.shift_min_pre_months, max_pre_len + 1):
        post_len = len(s) - pre_len
        if post_len < cfg.shift_min_post_months:
            continue

        pre = list(s[:pre_len])
        post = list(s[pre_len:])
        pre_m = _median(pre)
        post_m = _median(post)
        shift_abs = abs(post_m - pre_m)
        score = shift_abs / denom

        # Require post segment to be reasonably stable; otherwise this is likely variable income.
        if _cv(post) > cfg.shift_post_cv_max:
            continue

        if score > best_score:
            best_score = float(score)
            best_split = pre_len
            best_pre_m = float(pre_m)
            best_post_m = float(post_m)

    shift_detected = False
    if best_split is not None and best_pre_m is not None and best_post_m is not None:
        shift_abs_best = abs(best_post_m - best_pre_m)
        shift_detected = (best_score >= cfg.shift_score_min) and (shift_abs_best >= cfg.shift_abs_min)

    return shift_detected, float(best_score), best_split, best_pre_m, best_post_m


def _estimate_benefits(b: Sequence[float], window: int) -> float:
    if window <= 0:
        window = len(b)
    recent = list(b[-window:]) if len(b) > window else list(b)
    return float(_median(recent))


def model_income_last_12_months(
    months: Sequence[MonthIncome],
    *,
    cfg: IncomeModelConfig = IncomeModelConfig(),
) -> IncomeModelResult:
    """
    Main entrypoint.

    Input ordering: months[0] = oldest, months[11] = most recent.
    """
    _require_12_months(months)

    s = [float(m.net_salary) for m in months]
    b = [float(m.net_benefits) for m in months]

    for i, (sv, bv) in enumerate(zip(s, b, strict=True)):
        _require_finite(f"net_salary[{i}]", sv)
        _require_finite(f"net_benefits[{i}]", bv)

    total = [sv + bv for sv, bv in zip(s, b, strict=True)]
    eps = 1e-9

    # --- Leave / substitution flags ---
    substitution_flags = [
        (sv < cfg.salary_min_for_substitution) and (bv > cfg.benefit_min_for_substitution)
        for sv, bv in zip(s, b, strict=True)
    ]
    substitution_months = int(sum(1 for x in substitution_flags if x))
    recent_window = max(1, min(cfg.substitution_recent_window_months, 12))
    substitution_recent_max_run = _longest_true_run(substitution_flags[-recent_window:])

    # --- Salary stats ---
    median_salary = _median(s)
    iqr_salary = _iqr(s)
    cv_salary = _cv(s)

    if iqr_salary > 0:
        spike_threshold = median_salary + cfg.spike_k_iqr * iqr_salary
        low_threshold = median_salary - cfg.low_k_iqr * iqr_salary
    else:
        spike_threshold = median_salary * (1.0 + cfg.spike_rel_when_iqr_zero)
        low_threshold = median_salary * (1.0 - cfg.low_rel_when_iqr_zero)

    spike_count = int(sum(1 for sv in s if sv > spike_threshold))
    low_count = int(sum(1 for sv in s if sv < low_threshold))

    # --- Recent vs history shift proxy (last3 vs prev9) ---
    median_last3 = _median(s[-3:])
    median_prev9 = _median(s[:-3])
    shift_abs_last3_prev9 = float(abs(median_last3 - median_prev9))

    # --- Benefits stats ---
    benefit_months = int(sum(1 for bv in b if bv > cfg.benefit_min_for_substitution))

    # --- Level shift detection (raise/decrease) ---
    shift_detected, best_shift_score, best_shift_split, best_pre_m, best_post_m = _detect_level_shift(s, cfg)

    stats = IncomeModelStats(
        median_salary=float(median_salary),
        iqr_salary=float(iqr_salary),
        cv_salary=float(cv_salary),
        spike_count=int(spike_count),
        low_count=int(low_count),
        median_last3=float(median_last3),
        median_prev9=float(median_prev9),
        shift_abs_last3_prev9=float(shift_abs_last3_prev9),
        benefit_months=int(benefit_months),
        substitution_months=int(substitution_months),
        substitution_recent_max_run=int(substitution_recent_max_run),
        best_shift_score=float(best_shift_score),
        best_shift_split=best_shift_split,
        best_shift_pre_median=best_pre_m,
        best_shift_post_median=best_post_m,
    )

    volatility_score = _compute_volatility_score(
        cv_salary=cv_salary,
        spike_count=spike_count,
        shift_abs_last3_prev9=shift_abs_last3_prev9,
        cfg=cfg,
    )

    # --- Segment detection (transparent, prioritised) ---
    leave_trigger = 1 <= substitution_recent_max_run <= cfg.substitution_run_max_months

    variable_trigger = (
        (cv_salary >= cfg.variable_cv_min)
        or (spike_count >= cfg.spike_months_min)
        or ((spike_count >= cfg.bimodal_spike_min) and (low_count >= cfg.bimodal_low_min))
    )

    stable_trigger = (
        (cv_salary <= cfg.stable_cv_max)
        and (spike_count <= cfg.stable_spike_months_max)
        and (low_count <= cfg.stable_low_months_max)
    )

    # If multiple strong triggers, route to mixed/unclear.
    strong_triggers = int(leave_trigger) + int(shift_detected) + int(variable_trigger)

    reason_codes: list[str] = []
    confidence = "medium"

    if strong_triggers >= 2:
        segment = SEG_MIXED
        processing_route = ROUTE_MANUAL_REVIEW
        confidence = "low"
        reason_codes.append("multiple_strong_income_patterns_detected")
    elif leave_trigger:
        segment = SEG_LEAVE_SUBSTITUTION
        processing_route = ROUTE_STANDARD
        confidence = "medium"
        reason_codes.extend(
            [
                "salary_near_zero_with_benefits_present_recently",
                "short_consecutive_substitution_run",
            ]
        )
    elif shift_detected:
        segment = SEG_LEVEL_SHIFT
        processing_route = ROUTE_STANDARD
        confidence = "high" if (best_shift_split is not None and (12 - best_shift_split) >= 4) else "medium"
        reason_codes.append("salary_level_shift_detected")
        if best_pre_m is not None and best_post_m is not None:
            if best_post_m > best_pre_m:
                reason_codes.append("post_shift_salary_higher_than_pre_shift")
            elif best_post_m < best_pre_m:
                reason_codes.append("post_shift_salary_lower_than_pre_shift")
    elif variable_trigger:
        segment = SEG_VARIABLE
        if volatility_score <= cfg.route_variable_income_max_score:
            processing_route = ROUTE_VARIABLE_INCOME_CONSERVATIVE
        elif volatility_score <= cfg.route_enhanced_verification_max_score:
            processing_route = ROUTE_ENHANCED_VERIFICATION
        else:
            processing_route = ROUTE_MANUAL_REVIEW
        confidence = "medium"
        reason_codes.append("salary_high_variability_detected")
        if spike_count >= cfg.spike_months_min:
            reason_codes.append("salary_spike_pattern")
    elif stable_trigger:
        segment = SEG_STABLE
        processing_route = ROUTE_STANDARD
        confidence = "high"
        reason_codes.append("salary_stable")
    else:
        segment = SEG_MIXED
        processing_route = ROUTE_MANUAL_REVIEW
        confidence = "low"
        reason_codes.append("income_pattern_unclear")

    # --- Estimation per segment ---
    estimated_salary: float
    estimated_benefits: float
    salary_normal_excl_subst: Optional[float] = None
    total_current_recent: Optional[float] = None
    base_salary_estimate: Optional[float] = None
    included_variable_pay: Optional[float] = None

    if segment == SEG_STABLE:
        w = max(1, min(cfg.stable_salary_recent_months, 12))
        estimated_salary = float(_median(s[-w:]))
        estimated_benefits = _estimate_benefits(b, cfg.benefits_recent_months)
        reason_codes.append("salary_estimated_as_recent_median")

    elif segment == SEG_LEVEL_SHIFT and best_shift_split is not None:
        pre = list(s[:best_shift_split])
        post = list(s[best_shift_split:])
        pre_m = _median(pre)
        post_m = _median(post)

        w = min(1.0, len(post) / max(cfg.shift_full_weight_post_months, 1))
        estimated_salary = float(w * post_m + (1.0 - w) * pre_m)
        if w < 1.0:
            reason_codes.append("salary_estimate_blended_due_to_short_post_shift_window")
        else:
            reason_codes.append("salary_estimated_from_post_shift_months")

        estimated_benefits = _estimate_benefits(b, cfg.benefits_recent_months)

    elif segment == SEG_VARIABLE:
        # Base salary from typical range (q25..q75)
        q25 = _percentile(s, 0.25)
        q75 = _percentile(s, 0.75)
        typical = [sv for sv in s if q25 <= sv <= q75]
        base = _median(typical) if typical else _median(s)
        base_salary_estimate = float(base)

        # Variable pay component: average spike excess above base.
        if iqr_salary > 0:
            spike_value = median_salary + cfg.spike_k_iqr * iqr_salary
        else:
            spike_value = median_salary * (1.0 + cfg.spike_rel_when_iqr_zero)

        spike_excesses = [max(0.0, sv - base) for sv in s if sv > spike_value]
        avg_excess = fmean(spike_excesses) if spike_excesses else 0.0

        included_variable_pay = float(cfg.variable_pay_inclusion * avg_excess)
        estimated_salary = float(base + included_variable_pay)
        estimated_benefits = _estimate_benefits(b, cfg.benefits_recent_months)

        reason_codes.append("salary_decomposed_into_base_plus_variable_component")
        reason_codes.append("variable_component_included_with_haircut")

        # If the route is conservative, you may want to ignore variable pay entirely.
        if processing_route == ROUTE_VARIABLE_INCOME_CONSERVATIVE:
            estimated_salary = float(base)
            included_variable_pay = 0.0
            reason_codes.append("variable_component_excluded_for_conservative_route")

    elif segment == SEG_LEAVE_SUBSTITUTION:
        # Estimate "normal salary" by excluding substitution months.
        non_subst_salaries = [
            sv
            for sv, sub in zip(s, substitution_flags, strict=True)
            if not sub and sv >= cfg.salary_min_for_substitution
        ]
        salary_normal = _median(non_subst_salaries) if non_subst_salaries else _median(s)
        salary_normal_excl_subst = float(salary_normal)

        # Current observed total income (salary+benefits) from recent months (useful for policy decisions).
        tw = max(1, min(cfg.leave_current_total_recent_months, 12))
        total_current_recent = float(_median(total[-tw:]))

        # Policy-neutral default:
        # - salary estimate = normal salary (excl substitution), for "long-run capacity"
        # - benefits estimate = recent benefits median, for "current situation"
        # This allows downstream policy to decide whether to use normal salary, current total, or a blend.
        estimated_salary = float(salary_normal)
        estimated_benefits = _estimate_benefits(b, cfg.leave_benefits_recent_months)

        reason_codes.append("salary_estimated_excluding_substitution_months")
        reason_codes.append("benefits_estimated_from_recent_months")

    else:
        # Mixed/unclear fallback: conservative and route to review.
        w = max(1, min(cfg.stable_salary_recent_months, 12))
        estimated_salary = float(_median(s[-w:]))
        estimated_benefits = _estimate_benefits(b, cfg.benefits_recent_months)
        reason_codes.append("fallback_recent_median_estimate")

    estimated_total = float(estimated_salary + estimated_benefits)

    return IncomeModelResult(
        income_segment=segment,
        processing_route=processing_route,
        confidence=confidence,
        reason_codes=tuple(reason_codes),
        estimated_monthly_salary=float(estimated_salary),
        estimated_monthly_benefits=float(estimated_benefits),
        estimated_monthly_total=float(estimated_total),
        salary_normal_excluding_substitution=salary_normal_excl_subst,
        total_current_recent=total_current_recent,
        base_salary_estimate=base_salary_estimate,
        included_variable_pay=included_variable_pay,
        volatility_score=float(volatility_score),
        stats=stats,
    )


def model_income_last_12_months_from_arrays(
    net_salary_12: Sequence[float],
    net_benefits_12: Sequence[float],
    *,
    cfg: IncomeModelConfig = IncomeModelConfig(),
) -> IncomeModelResult:
    """Convenience wrapper if you already have two arrays."""
    if len(net_salary_12) != 12 or len(net_benefits_12) != 12:
        raise ValueError("Expected net_salary_12 and net_benefits_12 to be length 12")
    months = [MonthIncome(float(s), float(b)) for s, b in zip(net_salary_12, net_benefits_12, strict=True)]
    return model_income_last_12_months(months, cfg=cfg)


if __name__ == "__main__":
    # Minimal demo (tune values/thresholds for your market).
    demo_salary = [3000, 3000, 3050, 3000, 3000, 3000, 3200, 3200, 3200, 3500, 3200, 3200]
    demo_benefits = [0] * 12
    res = model_income_last_12_months_from_arrays(demo_salary, demo_benefits)
    print(res.income_segment, res.processing_route, res.confidence)
    print(res.estimated_monthly_salary, res.estimated_monthly_benefits, res.estimated_monthly_total)
    print(res.reason_codes)

