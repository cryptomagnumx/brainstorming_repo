"""
Income modeling from the last 12 months of net_salary and net_benefits.

This module intentionally implements ONLY these two logics:
1) Recent salary level shift detection (raise/decrease) -> estimate from post-shift months
2) Variable salary / bonus-like pattern detection -> conservative base + haircutted variable pay

Everything else (e.g., parental leave / benefit substitution special handling) is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from statistics import fmean, pstdev
from typing import Optional, Sequence


SEG_BASELINE_MEAN_12M = "baseline_mean_12m"
SEG_LEVEL_SHIFT = "recent_level_shift"
SEG_VARIABLE = "variable_salary_bonus"

ROUTE_STANDARD = "standard"
ROUTE_VARIABLE_INCOME = "variable_income_process"


@dataclass(frozen=True)
class MonthIncome:
    """One month of net income inputs (oldest -> newest)."""

    net_salary: float
    net_benefits: float


@dataclass(frozen=True)
class IncomeModelConfig:
    # --- Variable salary detection ---
    variable_cv_min: float = 0.25
    spike_k_iqr: float = 2.0  # spike if salary > median + k*IQR (when IQR > 0)
    low_k_iqr: float = 2.0  # low if salary < median - k*IQR (when IQR > 0)
    spike_rel_when_iqr_zero: float = 0.20  # fallback spike if salary > median*(1+rel)
    low_rel_when_iqr_zero: float = 0.20  # fallback low if salary < median*(1-rel)
    spike_months_min: int = 2
    bimodal_spike_min: int = 2
    bimodal_low_min: int = 2

    # Variable salary estimation (bonus inclusion)
    variable_pay_inclusion: float = 0.25  # include this fraction of variable pay (0..1)

    # --- Level shift detection ---
    # Stakeholder-friendly approach:
    # Compare "recent typical salary" (median of last N months) vs "earlier typical salary"
    # (median of the other 12-N months). Choose the N with the largest change from the
    # allowed options below.
    shift_post_window_months_options: tuple[int, ...] = (3, 6)
    shift_abs_min: float = 200.0  # require material absolute change
    shift_rel_min: float = 0.10  # and at least 10% change vs earlier median


@dataclass(frozen=True)
class IncomeModelStats:
    # Simple 12-month baselines (used when no special pattern triggers)
    mean_salary_12m: float
    mean_benefits_12m: float

    # Salary distribution / variability
    median_salary: float
    iqr_salary: float
    cv_salary: float
    spike_count: int
    low_count: int

    # Simple recent-vs-history proxy
    median_last3: float
    median_prev9: float
    shift_abs_last3_prev9: float

    # Level-shift detection (chosen window from config options)
    shift_abs_change: float
    shift_rel_change: float
    shift_split: Optional[int]  # split index: pre = s[:split], post = s[split:]
    shift_post_window_months: Optional[int]
    shift_pre_median: Optional[float]
    shift_post_median: Optional[float]


@dataclass(frozen=True)
class IncomeModelResult:
    income_segment: str  # one of SEG_*
    processing_route: str  # ROUTE_STANDARD or ROUTE_VARIABLE_INCOME
    confidence: str  # "high" | "medium"
    reason_codes: tuple[str, ...]

    estimated_monthly_salary: float
    estimated_monthly_benefits: float
    estimated_monthly_total: float

    # Extra transparency for VARIABLE / LEVEL_SHIFT segments
    base_salary_estimate: Optional[float]
    included_variable_pay: Optional[float]

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


def _detect_level_shift(
    s: Sequence[float],
    cfg: IncomeModelConfig,
) -> tuple[bool, float, float, Optional[int], Optional[float], Optional[float], Optional[int]]:
    """
    Returns:
      (shift_detected, abs_change, rel_change, split, pre_median, post_median, post_window_months)

    Simplified logic:
    - Try a small set of "recent window" sizes (e.g., last 3 months, last 6 months)
    - For each, compare median(recent) vs median(earlier)
    - Pick the window with the largest absolute change
    - Detect a shift if the change is large both in absolute and relative terms
    """
    eps = 1e-9

    best_abs = 0.0
    best_rel = 0.0
    best_split: Optional[int] = None
    best_pre_m: Optional[float] = None
    best_post_m: Optional[float] = None
    best_post_window: Optional[int] = None

    for post_window in cfg.shift_post_window_months_options:
        if post_window <= 0 or post_window >= len(s):
            continue
        split = len(s) - post_window
        pre = list(s[:split])
        post = list(s[split:])

        pre_m = _median(pre)
        post_m = _median(post)
        abs_change = float(abs(post_m - pre_m))
        rel_change = float(abs_change / max(abs(pre_m), eps))

        if abs_change > best_abs:
            best_abs = abs_change
            best_rel = rel_change
            best_split = split
            best_pre_m = float(pre_m)
            best_post_m = float(post_m)
            best_post_window = int(post_window)

    shift_detected = (best_split is not None) and (best_abs >= cfg.shift_abs_min) and (best_rel >= cfg.shift_rel_min)
    return shift_detected, float(best_abs), float(best_rel), best_split, best_pre_m, best_post_m, best_post_window


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

    # Simple baselines (used only if no special pattern triggers)
    mean_salary_12m = float(fmean(s))
    mean_benefits_12m = float(fmean(b))

    # --- Salary stats (for variable + shift logic) ---
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

    median_last3 = _median(s[-3:])
    median_prev9 = _median(s[:-3])
    shift_abs_last3_prev9 = float(abs(median_last3 - median_prev9))

    # --- Logic 2: level shift detection ---
    (
        shift_detected,
        shift_abs_change,
        shift_rel_change,
        shift_split,
        shift_pre_median,
        shift_post_median,
        shift_post_window_months,
    ) = _detect_level_shift(s, cfg)

    # --- Logic 3: variable salary detection ---
    variable_trigger = (
        (cv_salary >= cfg.variable_cv_min)
        or (spike_count >= cfg.spike_months_min)
        or ((spike_count >= cfg.bimodal_spike_min) and (low_count >= cfg.bimodal_low_min))
    )

    stats = IncomeModelStats(
        mean_salary_12m=float(mean_salary_12m),
        mean_benefits_12m=float(mean_benefits_12m),
        median_salary=float(median_salary),
        iqr_salary=float(iqr_salary),
        cv_salary=float(cv_salary),
        spike_count=int(spike_count),
        low_count=int(low_count),
        median_last3=float(median_last3),
        median_prev9=float(median_prev9),
        shift_abs_last3_prev9=float(shift_abs_last3_prev9),
        shift_abs_change=float(shift_abs_change),
        shift_rel_change=float(shift_rel_change),
        shift_split=shift_split,
        shift_post_window_months=shift_post_window_months,
        shift_pre_median=shift_pre_median,
        shift_post_median=shift_post_median,
    )

    # --- Choose segment + estimate salary ---
    reason_codes: list[str] = []
    base_salary_estimate: Optional[float] = None
    included_variable_pay: Optional[float] = None

    if variable_trigger:
        segment = SEG_VARIABLE
        route = ROUTE_VARIABLE_INCOME
        confidence = "medium"
        reason_codes.append("salary_high_variability_detected")
        if spike_count >= cfg.spike_months_min:
            reason_codes.append("salary_spike_pattern")

        # Base salary from typical range (q25..q75)
        q25 = _percentile(s, 0.25)
        q75 = _percentile(s, 0.75)
        typical = [sv for sv in s if q25 <= sv <= q75]
        base = _median(typical) if typical else _median(s)
        base_salary_estimate = float(base)

        # Variable pay: average spike excess above base, then apply haircut.
        spike_value = spike_threshold
        spike_excesses = [max(0.0, sv - base) for sv in s if sv > spike_value]
        avg_excess = fmean(spike_excesses) if spike_excesses else 0.0

        included_variable_pay = float(cfg.variable_pay_inclusion * avg_excess)
        estimated_salary = float(base + included_variable_pay)
        reason_codes.append("salary_estimated_as_base_plus_haircutted_variable_pay")

    elif shift_detected and shift_split is not None and shift_post_median is not None:
        segment = SEG_LEVEL_SHIFT
        route = ROUTE_STANDARD
        post_len = 12 - shift_split
        confidence = "high" if post_len >= 4 else "medium"
        reason_codes.append("salary_level_shift_detected")
        if shift_post_window_months is not None:
            reason_codes.append(f"shift_window_months_{shift_post_window_months}")
        estimated_salary = float(shift_post_median)
        reason_codes.append("salary_estimated_as_recent_window_median")

    else:
        segment = SEG_BASELINE_MEAN_12M
        route = ROUTE_STANDARD
        confidence = "medium"
        estimated_salary = float(mean_salary_12m)
        reason_codes.append("salary_estimated_by_mean_12m")

    # Benefits: no special handling in this simplified version.
    estimated_benefits = float(mean_benefits_12m)
    reason_codes.append("benefits_estimated_by_mean_12m")
    estimated_total = float(estimated_salary + estimated_benefits)

    return IncomeModelResult(
        income_segment=segment,
        processing_route=route,
        confidence=confidence,
        reason_codes=tuple(reason_codes),
        estimated_monthly_salary=float(estimated_salary),
        estimated_monthly_benefits=float(estimated_benefits),
        estimated_monthly_total=float(estimated_total),
        base_salary_estimate=base_salary_estimate,
        included_variable_pay=included_variable_pay,
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
    # Demo 1: raise (level shift)
    salary_raise = [3000, 3000, 3050, 3000, 3000, 3000, 3300, 3300, 3300, 3300, 3300, 3300]
    benefits_zero = [0] * 12
    print(model_income_last_12_months_from_arrays(salary_raise, benefits_zero))

    # Demo 2: variable pay / bonuses
    salary_bonus = [2800, 2800, 2800, 4500, 2800, 2800, 5200, 2800, 2800, 4800, 2800, 2800]
    print(model_income_last_12_months_from_arrays(salary_bonus, benefits_zero))

