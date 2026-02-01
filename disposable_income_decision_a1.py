"""
Strategy A1: adaptive tolerance bands for declared vs modeled disposable income.

This module implements a practical "YES vs MAYBE" rule:
- Small differences (often rounding) -> YES
- Declared DI significantly higher than modeled DI -> MAYBE (refer)

It also returns a DI value to use for the customer-facing limit, with a capped uplift
when declared DI is slightly higher but still within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class A1Config:
    # Minor-difference tolerance (A1)
    abs_floor: float = 100.0  # currency units
    rel_floor: float = 0.05  # 5% of scale

    # Rounding inference (for rounding_allowance)
    rounding_steps: tuple[float, ...] = (1000.0, 500.0, 100.0, 50.0, 10.0)
    rounding_close_frac: float = 0.05  # within 5% of step from a multiple
    rounding_allowance_frac: float = 0.5  # allow half a rounding step

    # Extra guardrails for referral (focus on declared >> modeled)
    abs_referral: float = 800.0  # currency units
    rel_referral: float = 0.25  # 25%


@dataclass(frozen=True)
class A1Decision:
    decision_outcome: str  # "yes" | "maybe"
    di_for_limit: float
    reason_codes: tuple[str, ...]

    # Helpful metrics for monitoring/tuning
    declared_di: float
    modeled_di: float
    abs_diff: float
    rel_diff: float
    minor_tol: float
    rounding_step: Optional[float]


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def infer_rounding_step(
    value: float,
    *,
    steps: tuple[float, ...],
    close_frac: float,
) -> Optional[float]:
    """
    Return the largest step that `value` appears rounded to (approximately).

    Example (with steps including 100 and 1000):
      5000 -> 1000
      3200 -> 100
      1234 -> None
    """
    v = abs(value)
    for step in steps:
        if step <= 0:
            continue
        nearest = round(v / step) * step
        if abs(v - nearest) <= (close_frac * step):
            return step
    return None


def compute_minor_tolerance(declared_di: float, modeled_di: float, cfg: A1Config) -> tuple[float, Optional[float]]:
    """
    A1 tolerance:

      minor_tol = max(abs_floor, rel_floor * scale, rounding_allowance)

    where scale = max(|declared|, |modeled|) and rounding_allowance is derived from
    inferred rounding of declared DI (often 100/1000).
    """
    scale = max(abs(declared_di), abs(modeled_di))
    tol = max(cfg.abs_floor, cfg.rel_floor * scale)

    rounding_step = infer_rounding_step(
        declared_di,
        steps=cfg.rounding_steps,
        close_frac=cfg.rounding_close_frac,
    )
    if rounding_step is not None:
        tol = max(tol, cfg.rounding_allowance_frac * rounding_step)

    return tol, rounding_step


def decide_disposable_income_a1(
    declared_di: float,
    modeled_di: float,
    *,
    cfg: A1Config = A1Config(),
) -> A1Decision:
    """
    Decide "yes" vs "maybe" using Strategy A (A1 tolerance bands).

    Policy for `di_for_limit`:
    - If YES and declared >= modeled: allow uplift but cap it to (modeled + minor_tol)
    - If YES and declared < modeled: use declared (already conservative)
    - If MAYBE: use min(declared, modeled) (conservative while referring)
    """
    _require_finite("declared_di", declared_di)
    _require_finite("modeled_di", modeled_di)

    eps = 1e-9
    abs_diff = declared_di - modeled_di
    rel_diff = abs_diff / max(abs(modeled_di), eps)

    minor_tol, rounding_step = compute_minor_tolerance(declared_di, modeled_di, cfg)
    minor = abs(abs_diff) <= minor_tol

    reason_codes: list[str] = []

    declared_significantly_higher = (
        abs_diff > 0
        and abs_diff > cfg.abs_referral
        and abs(rel_diff) > cfg.rel_referral
    )

    if minor:
        decision = "yes"
        reason_codes.append("diff_within_minor_tolerance")
    elif declared_significantly_higher:
        decision = "maybe"
        reason_codes.append("declared_significantly_higher_than_modeled")
    else:
        # Not "significantly higher": default to YES; track for tuning.
        decision = "yes"
        reason_codes.append("declared_not_significantly_higher_than_modeled")

    if decision == "yes":
        if declared_di >= modeled_di:
            di_for_limit = min(declared_di, modeled_di + minor_tol)
            reason_codes.append("uplift_capped_by_minor_tolerance")
        else:
            di_for_limit = declared_di
            reason_codes.append("declared_lower_is_conservative")
    else:
        di_for_limit = min(declared_di, modeled_di)
        reason_codes.append("referred_conservative_limit")

    return A1Decision(
        decision_outcome=decision,
        di_for_limit=float(di_for_limit),
        reason_codes=tuple(reason_codes),
        declared_di=float(declared_di),
        modeled_di=float(modeled_di),
        abs_diff=float(abs_diff),
        rel_diff=float(rel_diff),
        minor_tol=float(minor_tol),
        rounding_step=rounding_step,
    )

