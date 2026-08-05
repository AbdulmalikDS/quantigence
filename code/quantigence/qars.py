"""Quantum-Adjusted Risk Score (QARS).

Operationalizes the multi-factor quantum-risk model of Grigaliunas & Bruzgiene
(*Towards a Unified Quantum Risk Assessment*, Electronics 2025, 14(17):3338),
which extends Mosca's inequality (X + Y > Z) into a score over timeline,
sensitivity, and exposure. Our additions are the continuous sigmoid urgency
mapping and the agent-driven parameter calibration used by the rest of the
system; the naming and three-factor decomposition follow prior work.

Mosca's inequality: an asset is at risk when X + Y > Z, where
    X = migration time (years to re-secure the asset),
    Y = shelf-life (years the data must stay confidential),
    Z = collapse time (years until a cryptographically-relevant QC exists).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    """An asset to be scored. Times in years; sensitivity/exploitability in [0, 1]."""
    name: str
    migration_time: float      # X
    shelf_life: float          # Y
    collapse_time: float       # Z
    sensitivity: float         # S in [0, 1]
    exploitability: float      # E in [0, 1]

    def __post_init__(self) -> None:
        if self.collapse_time <= 0:
            raise ValueError(f"collapse_time (Z) must be > 0, got {self.collapse_time}")
        for field in ("migration_time", "shelf_life"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be >= 0")
        for field in ("sensitivity", "exploitability"):
            v = getattr(self, field)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {v}")


@dataclass(frozen=True)
class Weights:
    """Composite weights. Must be non-negative and sum to 1."""
    temporal: float = 0.5      # w_T
    sensitivity: float = 0.3   # w_S
    exploitability: float = 0.2  # w_E

    def __post_init__(self) -> None:
        total = self.temporal + self.sensitivity + self.exploitability
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"weights must sum to 1, got {total}")
        if min(self.temporal, self.sensitivity, self.exploitability) < 0:
            raise ValueError("weights must be non-negative")


def urgency_ratio(asset: Asset) -> float:
    """r(a) = (X + Y) / Z. r > 1 means a Mosca violation (asset at risk)."""
    return (asset.migration_time + asset.shelf_life) / asset.collapse_time


def temporal_urgency(asset: Asset, alpha: float = 10.0) -> float:
    """T(a) = sigmoid(alpha * (r - 1)), saturating to 1 as the ratio crosses the
    Mosca cliff at r = 1. alpha controls steepness; T = 0.5 exactly at r = 1."""
    r = urgency_ratio(asset)
    # Guard against overflow for extreme ratios; the sigmoid saturates anyway.
    z = alpha * (r - 1.0)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def qars(asset: Asset, weights: Weights | None = None, alpha: float = 10.0) -> float:
    """Composite score R = w_T*T + w_S*S + w_E*E, in [0, 1]."""
    w = weights or Weights()
    return (
        w.temporal * temporal_urgency(asset, alpha)
        + w.sensitivity * asset.sensitivity
        + w.exploitability * asset.exploitability
    )


def mosca_violation(asset: Asset) -> bool:
    """True iff X + Y > Z."""
    return urgency_ratio(asset) > 1.0


if __name__ == "__main__":
    # Self-check: exercises the cliff, bounds, and Mosca agreement.
    safe = Asset("VPN-safe", migration_time=1, shelf_life=2, collapse_time=15,
                 sensitivity=0.3, exploitability=0.2)
    at_risk = Asset("Archive-TS", migration_time=3, shelf_life=25, collapse_time=10,
                    sensitivity=1.0, exploitability=0.5)

    assert math.isclose(urgency_ratio(safe), 3 / 15)
    assert not mosca_violation(safe)
    assert mosca_violation(at_risk)

    # T = 0.5 exactly at the cliff (r = 1).
    cliff = Asset("cliff", migration_time=5, shelf_life=5, collapse_time=10,
                  sensitivity=0, exploitability=0)
    assert math.isclose(temporal_urgency(cliff), 0.5, abs_tol=1e-9)

    # Composite stays in [0, 1] and the at-risk asset scores higher than the safe one.
    assert 0.0 <= qars(safe) <= 1.0
    assert 0.0 <= qars(at_risk) <= 1.0
    assert qars(at_risk) > qars(safe)

    # Higher urgency ratio => higher temporal factor (monotonic).
    assert temporal_urgency(at_risk) > temporal_urgency(safe)

    print(f"QARS(safe)    = {qars(safe):.3f}  (r={urgency_ratio(safe):.2f})")
    print(f"QARS(at_risk) = {qars(at_risk):.3f}  (r={urgency_ratio(at_risk):.2f})")
    print("self-check passed")
