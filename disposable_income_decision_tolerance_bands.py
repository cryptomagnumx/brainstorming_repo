"""
Tolerance-band decisioning for declared vs modeled disposable income.
Outputs:
- decision_outcome: "yes" (auto) or "maybe" (refer)
- di_for_limit: DI used for customer-facing max loan limit (with a capped uplift)


## Strategy A — Simple, robust rule-based tolerance bands (recommended starting point)
Define a “minor difference” tolerance that scales with magnitude (tolerating typical self-report noise without relying on source-specific heuristics).

### A1) Tolerance-band consistency check: `minor_tol = max(abs_floor, rel_floor * scale)`
Where:

- `abs_floor` protects low-income cases (e.g., 50–200 currency units)
- `rel_floor` protects high-income cases (e.g., 2%–10%)
- `scale = max(abs(declared_di), abs(modeled_di))`

"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ToleranceBandConfig:
    # "Minor difference" tolerance
    minor_abs_floor: float = 100.0  # currency units
    minor_rel_floor: float = 0.05  # 5% of scale

    # "Clear difference" tolerance for referral (only when declared > modeled)
    referral_abs_floor: float = 800.0  # currency units
    referral_rel_floor: float = 0.25  # 25% of scale


@dataclass(frozen=True)
class DisposableIncomeDecision:
    decision_outcome: str  # "yes" | "maybe"
    di_for_limit: float
    reason_codes: tuple[str, ...]

    # Metrics for monitoring/tuning
    declared_di: float
    modeled_di: float
    abs_diff: float
    rel_diff: float
    minor_tol: float
    referral_tol: float


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _compute_tolerances(declared_di: float, modeled_di: float, cfg: ToleranceBandConfig) -> tuple[float, float]:
    """
    Computes:

      scale = max(|declared|, |modeled|)
      minor_tol = max(minor_abs_floor, minor_rel_floor * scale)
      referral_tol = max(referral_abs_floor, referral_rel_floor * scale)
    """
    scale = max(abs(declared_di), abs(modeled_di))
    minor_tol = max(cfg.minor_abs_floor, cfg.minor_rel_floor * scale)
    referral_tol = max(cfg.referral_abs_floor, cfg.referral_rel_floor * scale)
    return minor_tol, referral_tol


def decide_disposable_income_tolerance_bands(
    declared_di: float,
    modeled_di: float,
    *,
    cfg: ToleranceBandConfig = ToleranceBandConfig(),
) -> DisposableIncomeDecision:
    """
    Tolerance-band decisioning:

    - YES if declared and modeled DI are within the minor tolerance band.
    - MAYBE (refer) if declared DI is above modeled DI by more than the referral tolerance band.
    - Otherwise YES (track via reason codes for tuning).

    Policy for `di_for_limit`:
    - If YES and declared >= modeled: allow uplift but cap to (modeled + minor_tol)
    - If YES and declared < modeled: use declared (already conservative)
    - If MAYBE: use min(declared, modeled) (conservative while referring)
    """
    _require_finite("declared_di", declared_di)
    _require_finite("modeled_di", modeled_di)

    eps = 1e-9
    abs_diff = declared_di - modeled_di
    rel_diff = abs_diff / max(abs(modeled_di), eps)

    minor_tol, referral_tol = _compute_tolerances(declared_di, modeled_di, cfg)

    minor = abs(abs_diff) <= minor_tol
    declared_significantly_higher = abs_diff > referral_tol

    reason_codes: list[str] = []

    if minor:
        decision = "yes"
        reason_codes.append("diff_within_minor_tolerance")

    elif declared_significantly_higher:
        decision = "maybe"
        reason_codes.append("declared_significantly_higher_than_modeled")

    else:
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

    return DisposableIncomeDecision(
        decision_outcome=decision,
        di_for_limit=float(di_for_limit),
        reason_codes=tuple(reason_codes),
        declared_di=float(declared_di),
        modeled_di=float(modeled_di),
        abs_diff=float(abs_diff),
        rel_diff=float(rel_diff),
        minor_tol=float(minor_tol),
        referral_tol=float(referral_tol),
    )

