"""
tests/test_model.py

Feasibility and correctness checks for the markdown MIP (src/model.py).
"""

import pandas as pd
import pytest

from src.model import build_model, solve_model


@pytest.fixture
def tiny_sku_week_panel():
    """A minimal 1-SKU, 4-week panel to build a small solvable MIP instance against."""
    return pd.DataFrame({
        "item_id": ["FOODS_1_001"] * 4,
        "store_id": ["CA_1"] * 4,
        "week": [1, 2, 3, 4],
        "starting_inventory": [40.0, None, None, None],
        "unit_cost": [2.00] * 4,
        "base_price": [5.00] * 4,
        "intercept": [4.0] * 4,
        "elasticity": [-1.0] * 4,
    })


def test_model_builds_without_error(tiny_sku_week_panel):
    """The model should construct successfully from a minimal valid panel."""
    m = build_model(tiny_sku_week_panel)
    assert m is not None


def test_model_is_feasible(tiny_sku_week_panel):
    """A small, well-formed instance should solve to a feasible (not infeasible) result."""
    m = build_model(tiny_sku_week_panel)
    result = solve_model(m)
    assert result.feasible is True


def test_inventory_conservation_holds(tiny_sku_week_panel):
    """Total units sold across the horizon should never exceed starting inventory."""
    m = build_model(tiny_sku_week_panel)
    result = solve_model(m)
    total_sold = result.schedule["units_sold"].sum()
    assert total_sold <= tiny_sku_week_panel["starting_inventory"].iloc[0]


def test_prices_are_monotonically_non_increasing(tiny_sku_week_panel):
    """Discounts should never decrease week over week for a given SKU."""
    m = build_model(tiny_sku_week_panel)
    result = solve_model(m)
    prices = result.schedule.sort_values("week")["price"].tolist()
    assert all(earlier >= later for earlier, later in zip(prices, prices[1:]))


def test_margin_floor_respected(tiny_sku_week_panel):
    """Price should never drop below cost, except in a designated final clearance week."""
    m = build_model(tiny_sku_week_panel)
    result = solve_model(m)
    unit_cost = tiny_sku_week_panel["unit_cost"].iloc[0]
    non_final_weeks = result.schedule.sort_values("week").iloc[:-1]
    assert (non_final_weeks["price"] >= unit_cost).all()


def test_min_tier_by_sku_floor_prevents_shallower_tier():
    """
    Regression test for the cross-window monotonicity fix: passing a
    min_tier_by_sku floor should fix out any tier shallower than that floor
    on the SKU's first week in this panel.
    """
    panel = pd.DataFrame({
        "item_id": ["ITEM_1"] * 4,
        "store_id": ["STORE_1"] * 4,
        "week": [1, 2, 3, 4],
        "starting_inventory": [40.0, None, None, None],
        "unit_cost": [2.00] * 4,
        "base_price": [10.00] * 4,
        "intercept": [4.0] * 4,
        "elasticity": [-1.0] * 4,
    })
    m = build_model(panel, min_tier_by_sku={("ITEM_1", "STORE_1"): 2})
    result = solve_model(m)

    assert result.feasible is True
    first_week_tier = result.schedule.sort_values("week").iloc[0]["tier"]
    assert first_week_tier >= 2