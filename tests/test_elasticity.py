"""
tests/test_elasticity.py

Tests for price-elasticity fitting and piecewise demand curve construction.
Uses synthetic data with a known, engineered elasticity so we can check
the fitted coefficient recovers the true value, not just that it runs.
"""

import numpy as np
import pandas as pd
import pytest

from src.elasticity import (
    _fit_log_log,
    fit_dept_pooled_elasticities,
    fit_elasticities,
    predicted_demand,
    build_piecewise_curve,
    MIN_OBS_FOR_ITEM_STORE_FIT,
)


def _make_synthetic_group(true_elasticity: float, true_intercept: float, n: int = 50, seed: int = 0) -> pd.DataFrame:
    """
    Generate synthetic (price, units_sold) data from a known log-log relationship,
    with a bit of noise, so fitted coefficients can be checked against ground truth.
    """
    rng = np.random.default_rng(seed)
    prices = rng.uniform(2.0, 8.0, size=n)
    noise = rng.normal(0, 0.05, size=n)
    log_units = true_intercept + true_elasticity * np.log(prices) + noise
    units_sold = np.round(np.exp(log_units) - 1).clip(min=0)

    return pd.DataFrame({
        "item_id": ["ITEM_1"] * n,
        "store_id": ["STORE_1"] * n,
        "dept_id": ["DEPT_1"] * n,
        "sell_price": prices,
        "units_sold": units_sold,
        "has_event": [False] * n,
    })


def test_fit_log_log_recovers_known_elasticity():
    """Fitting synthetic data generated from a known elasticity should recover it closely."""
    true_elasticity = -1.5
    true_intercept = 3.0
    df = _make_synthetic_group(true_elasticity, true_intercept, n=200)

    fit = _fit_log_log(df)

    assert fit is not None
    assert fit["elasticity"] == pytest.approx(true_elasticity, abs=0.2)
    assert fit["r2"] > 0.8


def test_fit_log_log_returns_none_with_no_price_variation():
    """A constant price gives no variation to estimate elasticity from — should return None."""
    df = pd.DataFrame({
        "sell_price": [4.0] * 20,
        "units_sold": [10] * 20,
        "has_event": [False] * 20,
    })
    fit = _fit_log_log(df)
    assert fit is None


def test_fit_elasticities_uses_item_store_fit_when_enough_data():
    """Item-store pairs with enough observations should get their own fit, not the dept fallback."""
    df = _make_synthetic_group(-1.2, 2.5, n=MIN_OBS_FOR_ITEM_STORE_FIT + 5)
    result = fit_elasticities(df, min_obs=MIN_OBS_FOR_ITEM_STORE_FIT)

    assert len(result) == 1
    assert result.iloc[0]["fit_level"] == "item_store"


def test_fit_elasticities_falls_back_to_dept_pooled_with_sparse_data():
    """
    An item-store pair with too few observations should fall back to the
    department-pooled estimate rather than being dropped or fit unreliably.
    """
    # sparse item-store (below min_obs)
    sparse_item = _make_synthetic_group(-1.0, 2.0, n=3, seed=1)
    sparse_item["item_id"] = "SPARSE_ITEM"
    sparse_item["store_id"] = "STORE_2"

    # plenty of dept-level data to pool from (different item, same dept)
    dept_data = _make_synthetic_group(-1.0, 2.0, n=100, seed=2)
    dept_data["item_id"] = "OTHER_ITEM"
    dept_data["store_id"] = "STORE_3"

    combined = pd.concat([sparse_item, dept_data], ignore_index=True)

    result = fit_elasticities(combined, min_obs=MIN_OBS_FOR_ITEM_STORE_FIT)

    sparse_row = result[result["item_id"] == "SPARSE_ITEM"].iloc[0]
    assert sparse_row["fit_level"] == "dept_pooled"


def test_predicted_demand_decreases_as_price_increases():
    """With a negative elasticity, predicted demand should fall as price rises."""
    demand_low_price = predicted_demand(intercept=3.0, elasticity=-1.5, price=2.0)
    demand_high_price = predicted_demand(intercept=3.0, elasticity=-1.5, price=8.0)
    assert demand_low_price > demand_high_price


def test_build_piecewise_curve_sorted_by_price_ascending():
    """Breakpoints must be sorted ascending by price for valid SOS2 constraints."""
    curve = build_piecewise_curve(base_price=10.0, intercept=3.0, elasticity=-1.2)
    prices = [pt[0] for pt in curve]
    assert prices == sorted(prices)


def test_build_piecewise_curve_demand_non_increasing_in_price():
    """For a negative-elasticity SKU, demand at each breakpoint should not increase with price."""
    curve = build_piecewise_curve(base_price=10.0, intercept=3.0, elasticity=-1.2)
    demands = [pt[1] for pt in curve]
    assert all(earlier >= later for earlier, later in zip(demands, demands[1:]))


def test_build_piecewise_curve_covers_all_discount_tiers():
    """Number of breakpoints should match the number of discount tiers provided."""
    tiers = [0.0, 0.25, 0.50]
    curve = build_piecewise_curve(base_price=10.0, intercept=3.0, elasticity=-1.0, discount_tiers=tiers)
    assert len(curve) == len(tiers)