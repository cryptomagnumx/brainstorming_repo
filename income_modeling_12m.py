"""
Special salary estimators from last 12 months of `net_salary`.

This module is designed for use as Python UDFs on a Polars DataFrame.

It provides two explainable "special-case" estimators:
- Variable / bonus-like salary estimator
- Recent level-shift (raise/decrease) estimator

Each estimator returns:
  (estimated_salary, triggered)

Where:
- triggered == True  -> estimated_salary is a float
- triggered == False -> estimated_salary is None (use your baseline model outside this repo)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from statistics import fmean, pstdev
from typing import Optional, Sequence


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

    # --- Level shift detection (stakeholder-friendly) ---
    # Compare median salary in the most recent N months vs median salary in the earlier 12-N months.
    # Try these candidate windows and pick the one with the largest absolute change.
    shift_post_window_months_options: tuple[int, ...] = (3, 6)
    shift_abs_min: float = 200.0  # require material absolute change
    shift_rel_min: float = 0.10  # and at least 10% change vs earlier median


def _to_float_list_12(values: Sequence[float] | None) -> Optional[list[float]]:
    """
    Convert a 12-element sequence into a float list.

    Returns None if:
    - values is None
    - length != 12
    - any element is missing / not a finite number
    """
    if values is None:
        return None
    if len(values) != 12:
        return None

    out: list[float] = []
    for x in values:
        if x is None:  # type: ignore[redundant-expr]
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if not isfinite(v):
            return None
        out.append(v)
    return out


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


def estimate_salary_variable_bonus_12m(
    net_salary_12: Sequence[float] | None,
    *,
    cfg: IncomeModelConfig = IncomeModelConfig(),
) -> tuple[Optional[float], bool]:
    """
    Variable/bonus-like salary estimator.

    Trigger condition (simple, explainable):
    - CV high OR spike months present OR bimodal-ish (spikes + lows)

    Estimate (explainable):
    - base salary = median of typical months (q25..q75)
    - variable component = average spike excess above base
    - include haircut fraction of variable component
    """
    s = _to_float_list_12(net_salary_12)
    if s is None:
        return None, False

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

    triggered = bool(
        (cv_salary >= cfg.variable_cv_min)
        or (spike_count >= cfg.spike_months_min)
        or ((spike_count >= cfg.bimodal_spike_min) and (low_count >= cfg.bimodal_low_min))
    )
    if not triggered:
        return None, False

    q25 = _percentile(s, 0.25)
    q75 = _percentile(s, 0.75)
    typical = [sv for sv in s if q25 <= sv <= q75]
    base = float(_median(typical) if typical else _median(s))

    spike_excesses = [max(0.0, sv - base) for sv in s if sv > spike_threshold]
    avg_excess = float(fmean(spike_excesses) if spike_excesses else 0.0)

    inclusion = float(min(1.0, max(0.0, cfg.variable_pay_inclusion)))
    estimated_salary = float(base + inclusion * avg_excess)
    return estimated_salary, True


def estimate_salary_level_shift_12m(
    net_salary_12: Sequence[float] | None,
    *,
    cfg: IncomeModelConfig = IncomeModelConfig(),
) -> tuple[Optional[float], bool]:
    """
    Level-shift estimator (raise/decrease).

    Simplified stakeholder-friendly logic:
    - For each post window N in cfg.shift_post_window_months_options:
      compare median(last N months) vs median(first 12-N months)
    - Pick the N with the largest absolute change
    - Trigger if change is large in both absolute and relative terms
    - If triggered, return the median of the recent window as the estimate
    """
    s = _to_float_list_12(net_salary_12)
    if s is None:
        return None, False

    eps = 1e-9
    best_abs = 0.0
    best_rel = 0.0
    best_post_median: Optional[float] = None

    for post_window in cfg.shift_post_window_months_options:
        if post_window <= 0 or post_window >= 12:
            continue
        split = 12 - post_window
        pre = s[:split]
        post = s[split:]

        pre_m = _median(pre)
        post_m = _median(post)
        abs_change = float(abs(post_m - pre_m))
        rel_change = float(abs_change / max(abs(pre_m), eps))

        if abs_change > best_abs:
            best_abs = abs_change
            best_rel = rel_change
            best_post_median = float(post_m)

    triggered = bool((best_post_median is not None) and (best_abs >= cfg.shift_abs_min) and (best_rel >= cfg.shift_rel_min))
    if not triggered:
        return None, False

    return best_post_median, True


def estimate_salary_special_cases_12m(
    net_salary_12: Sequence[float] | None,
    *,
    cfg: IncomeModelConfig = IncomeModelConfig(),
) -> tuple[Optional[float], bool]:
    """
    Convenience function:
    - Apply variable-salary estimator first
    - If not triggered, try level-shift estimator
    - If neither triggers, return (None, False)
    """
    est, trig = estimate_salary_variable_bonus_12m(net_salary_12, cfg=cfg)
    if trig:
        return est, True
    est, trig = estimate_salary_level_shift_12m(net_salary_12, cfg=cfg)
    if trig:
        return est, True
    return None, False


if __name__ == "__main__":
    cfg = IncomeModelConfig()

    # Level shift example (raise)
    salary_raise = [3000, 3000, 3050, 3000, 3000, 3000, 3300, 3300, 3300, 3300, 3300, 3300]
    print("shift:", estimate_salary_level_shift_12m(salary_raise, cfg=cfg))

    # Variable pay example (bonuses)
    salary_bonus = [2800, 2800, 2800, 4500, 2800, 2800, 5200, 2800, 2800, 4800, 2800, 2800]
    print("variable:", estimate_salary_variable_bonus_12m(salary_bonus, cfg=cfg))
    print("special:", estimate_salary_special_cases_12m(salary_bonus, cfg=cfg))

