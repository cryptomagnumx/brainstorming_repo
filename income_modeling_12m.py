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


def estimate_salary_special_cases_list_12m(
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


def estimate_salary_special_cases_12m(  # noqa: N802 (Polars UDF name kept as requested)
    df,
    keys=None,
    *,
    cfg: IncomeModelConfig = IncomeModelConfig(),
    salary_col: str = "net_salary",
    order_col: Optional[str] = None,
):
    """
    Group UDF for Polars `map_groups`.

    Expected usage:
      df.group_by(...).map_groups(estimate_salary_special_cases_12m, ...)

    Input:
    - df: a `polars.DataFrame` for a single group (typically 12 rows)
    - keys: group keys (optional; if provided, they are copied into the output row)

    Output: a one-row `polars.DataFrame` with:
    - variable_estimated_salary, variable_triggered
    - shift_estimated_salary, shift_triggered

    Notes:
    - If the group does not have 12 usable salary values, both sub-models return (None, False).
    - If `order_col` is provided (and exists), the group is sorted by it before extracting salaries.
      If not provided, we auto-sort by a common time column name if present.
    """
    import importlib

    try:
        pl = importlib.import_module("polars")
    except Exception as e:  # pragma: no cover
        raise ImportError("Polars is required for the group UDF version of estimate_salary_special_cases_12m") from e

    # Some APIs may pass (keys, df) instead of (df, keys); support both.
    if not isinstance(df, pl.DataFrame) and isinstance(keys, pl.DataFrame):
        df, keys = keys, df
    if not isinstance(df, pl.DataFrame):
        raise TypeError("estimate_salary_special_cases_12m expects a polars.DataFrame as the first (or second) argument")

    # Determine ordering for "last N months" logic.
    if order_col is not None and order_col in df.columns:
        df_sorted = df.sort(order_col)
    else:
        # Best-effort auto-detect.
        auto_order_candidates = (
            "month_index",
            "month_idx",
            "month",
            "period",
            "date",
            "as_of_date",
            "as_of",
            "timestamp",
        )
        found = next((c for c in auto_order_candidates if c in df.columns), None)
        df_sorted = df.sort(found) if found is not None else df

    # Extract a 12-month salary list (oldest -> newest).
    salary_list: Optional[Sequence[float]]
    if salary_col not in df_sorted.columns:
        salary_list = None
    elif df_sorted.height == 1:
        v = df_sorted[salary_col][0]
        salary_list = v if isinstance(v, (list, tuple)) else None
    else:
        vals = df_sorted[salary_col].to_list()
        salary_list = vals[-12:] if len(vals) >= 12 else None

    var_est, var_trig = estimate_salary_variable_bonus_12m(salary_list, cfg=cfg)
    shift_est, shift_trig = estimate_salary_level_shift_12m(salary_list, cfg=cfg)

    out: dict[str, object] = {
        "variable_estimated_salary": var_est,
        "variable_triggered": bool(var_trig),
        "shift_estimated_salary": shift_est,
        "shift_triggered": bool(shift_trig),
    }

    # Copy keys into the output row (optional).
    if keys is not None:
        if isinstance(keys, dict):
            out.update(keys)
        elif isinstance(keys, (list, tuple)):
            # Support either [(name, value), ...] or plain tuples of values.
            if all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in keys):
                out.update({str(k): v for k, v in keys})  # type: ignore[misc]
            else:
                out.update({f"key_{i}": v for i, v in enumerate(keys)})
        else:
            out["key"] = keys

    return pl.DataFrame(
        [out],
        schema={
            "variable_estimated_salary": pl.Float64,
            "variable_triggered": pl.Boolean,
            "shift_estimated_salary": pl.Float64,
            "shift_triggered": pl.Boolean,
            # key columns (if any) are inferred
        },
    )


if __name__ == "__main__":
    cfg = IncomeModelConfig()

    # Level shift example (raise)
    salary_raise = [3000, 3000, 3050, 3000, 3000, 3000, 3300, 3300, 3300, 3300, 3300, 3300]
    print("shift:", estimate_salary_level_shift_12m(salary_raise, cfg=cfg))

    # Variable pay example (bonuses)
    salary_bonus = [2800, 2800, 2800, 4500, 2800, 2800, 5200, 2800, 2800, 4800, 2800, 2800]
    print("variable:", estimate_salary_variable_bonus_12m(salary_bonus, cfg=cfg))
    print("special(list):", estimate_salary_special_cases_list_12m(salary_bonus, cfg=cfg))

