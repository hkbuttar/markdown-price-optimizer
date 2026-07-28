"""
tests/test_inventory_assumptions.py

Tests for the starting_inventory / unit_cost assumption-generation step.
M5 has no real inventory or cost data, so these tests validate that the
derived assumptions are internally consistent (scaled to each SKU's own
demand, correct margin math, correctly placed only on the first week per
SKU) rather than checking against any "true" value, since none exists.
"""

import pandas as pd
import pytest

from src.inventory_assumptions import (
    add_inventory_and_cost_assumptions,
    DEFAULT_INVENTORY_WEEKS_OF_SUPPLY,
    DEFAULT_ASSUMED_MARGIN_FRACTION,
    MIN_STARTING_INVENTORY,
)


def _make_panel(n_weeks: int = 5, units_per_week: list[float] = None, base_price: float = 10.0) -> pd.DataFrame:
    """A single-SKU panel with a known, fixed weekly sales pattern so the
    resulting average (and therefore the inventory assumption) is predictable."""
    units_per_week = units_per_week or [10.0] * n_weeks
    assert len(units_per_week) == n_weeks

    return pd.DataFrame({
        "item_id": ["ITEM_1"] * n_weeks,
        "store_id": ["STORE_1"] * n_weeks,
        "week": list(range(1, n_weeks + 1)),
        "units_sold": units_per_week,
        "base_price": [base_price] * n_weeks,
    })


def test_starting_inventory_scales_with_average_weekly_sales():
    """A SKU selling 10 units/week on average with 8 weeks-of-supply should get
    starting_inventory = 80, matching the documented sizing heuristic exactly."""
    panel = _make_panel(units_per_week=[10.0] * 5)
    result = add_inventory_and_cost_assumptions(panel, inventory_weeks_of_supply=8)

    first_week = result.sort_values("week").iloc[0]
    assert first_week["starting_inventory"] == pytest.approx(80.0)


def test_starting_inventory_uses_average_not_first_week_sales():
    """Inventory should be based on the SKU's average sales across the whole
    history, not just an arbitrary single week -- catches a common off-by-scope bug."""
    panel = _make_panel(units_per_week=[0.0, 20.0, 0.0, 20.0, 0.0])  # mean = 8.0
    result = add_inventory_and_cost_assumptions(panel, inventory_weeks_of_supply=8)

    first_week = result.sort_values("week").iloc[0]
    assert first_week["starting_inventory"] == pytest.approx(8.0 * 8)


def test_starting_inventory_only_populated_on_first_week():
    """model.py expects starting_inventory as non-null ONLY on a SKU's first
    week -- weeks 2+ must remain null/None, not repeated or zero."""
    panel = _make_panel(n_weeks=5)
    result = add_inventory_and_cost_assumptions(panel)
    result = result.sort_values("week").reset_index(drop=True)

    assert pd.notna(result.loc[0, "starting_inventory"])
    assert result.loc[1:, "starting_inventory"].isna().all()


def test_starting_inventory_respects_minimum_floor():
    """A SKU with near-zero average sales shouldn't get an unrealistically tiny
    or zero starting inventory -- MIN_STARTING_INVENTORY should act as a floor."""
    panel = _make_panel(units_per_week=[0.0, 0.0, 0.1, 0.0, 0.0])  # mean ~ 0.02
    result = add_inventory_and_cost_assumptions(panel, inventory_weeks_of_supply=8)

    first_week = result.sort_values("week").iloc[0]
    assert first_week["starting_inventory"] == pytest.approx(MIN_STARTING_INVENTORY)


def test_unit_cost_matches_margin_assumption():
    """unit_cost should equal base_price * (1 - margin), applied consistently
    across every week, not just the first."""
    panel = _make_panel(base_price=20.0)
    result = add_inventory_and_cost_assumptions(panel, assumed_margin_fraction=0.40)

    expected_cost = 20.0 * (1 - 0.40)
    assert result["unit_cost"].apply(lambda x: x == pytest.approx(expected_cost)).all()


def test_unit_cost_respects_custom_margin_argument():
    """Passing a different margin should change unit_cost proportionally,
    confirming the parameter is actually wired through and not hardcoded."""
    panel = _make_panel(base_price=10.0)
    result_default = add_inventory_and_cost_assumptions(panel, assumed_margin_fraction=0.40)
    result_higher_margin = add_inventory_and_cost_assumptions(panel, assumed_margin_fraction=0.60)

    assert result_default["unit_cost"].iloc[0] == pytest.approx(6.0)
    assert result_higher_margin["unit_cost"].iloc[0] == pytest.approx(4.0)
    assert result_higher_margin["unit_cost"].iloc[0] < result_default["unit_cost"].iloc[0]


def test_multiple_skus_get_independent_assumptions():
    """Each SKU's inventory and cost should be derived from its own sales/price,
    not leaked or averaged across SKUs in the same panel."""
    sku_a = _make_panel(units_per_week=[10.0] * 5, base_price=10.0)
    sku_a["item_id"] = "ITEM_A"

    sku_b = _make_panel(units_per_week=[40.0] * 5, base_price=20.0)
    sku_b["item_id"] = "ITEM_B"

    combined = pd.concat([sku_a, sku_b], ignore_index=True)
    result = add_inventory_and_cost_assumptions(combined, inventory_weeks_of_supply=8)

    inv_a = result[(result["item_id"] == "ITEM_A") & result["starting_inventory"].notna()]["starting_inventory"].iloc[0]
    inv_b = result[(result["item_id"] == "ITEM_B") & result["starting_inventory"].notna()]["starting_inventory"].iloc[0]

    assert inv_a == pytest.approx(80.0)   # 10 * 8
    assert inv_b == pytest.approx(320.0)  # 40 * 8
    assert inv_a != inv_b


def test_falls_back_to_sell_price_if_base_price_missing():
    """Some panels may only have sell_price rather than base_price (e.g. the
    raw processed panel before merging with elasticities) -- should still work."""
    panel = _make_panel(base_price=10.0)
    panel = panel.rename(columns={"base_price": "sell_price"})

    result = add_inventory_and_cost_assumptions(panel, assumed_margin_fraction=0.40)
    assert result["unit_cost"].iloc[0] == pytest.approx(10.0 * 0.60)