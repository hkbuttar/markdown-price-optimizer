"""
tests/test_solve.py

Tests for solve.py: panel filtering, orchestration, and baseline revenue
simulations. Uses small synthetic panels so these run fast in CI.
"""

import pandas as pd
import pytest

from src.solve import (
    filter_panel,
    run_solve,
    simulate_flat_discount_baseline,
    simulate_calendar_baseline,
    FLAT_DISCOUNT,
)


def _make_multi_category_panel(n_weeks: int = 6) -> pd.DataFrame:
    """Two categories, three departments total, so filtering has something real to narrow down."""
    rows = []
    specs = [
        ("FOODS", "FOODS_1", "ITEM_F1", "STORE_1"),
        ("FOODS", "FOODS_2", "ITEM_F2", "STORE_1"),
        ("HOBBIES", "HOBBIES_1", "ITEM_H1", "STORE_1"),
    ]
    for cat_id, dept_id, item_id, store_id in specs:
        for i, w in enumerate(range(1, n_weeks + 1)):
            rows.append({
                "item_id": item_id,
                "store_id": store_id,
                "cat_id": cat_id,
                "dept_id": dept_id,
                "week": w,
                "starting_inventory": 40.0 if i == 0 else None,  # was 80.0 -- too high for this curve's demand ceiling
                "unit_cost": 6.0,
                "base_price": 10.0,
                "intercept": 4.0,
                "elasticity": -1.0,
            })
    return pd.DataFrame(rows)


# --- filter_panel ---

def test_filter_panel_by_category():
    panel = _make_multi_category_panel()
    filtered = filter_panel(panel, category="FOODS")

    assert set(filtered["cat_id"].unique()) == {"FOODS"}
    assert set(filtered["dept_id"].unique()) == {"FOODS_1", "FOODS_2"}


def test_filter_panel_by_department():
    panel = _make_multi_category_panel()
    filtered = filter_panel(panel, department="FOODS_1")

    assert set(filtered["dept_id"].unique()) == {"FOODS_1"}
    assert set(filtered["item_id"].unique()) == {"ITEM_F1"}


def test_filter_panel_by_category_and_department_combined():
    panel = _make_multi_category_panel()
    filtered = filter_panel(panel, category="FOODS", department="FOODS_2")
    assert set(filtered["item_id"].unique()) == {"ITEM_F2"}


def test_filter_panel_by_weeks_truncates_correctly():
    panel = _make_multi_category_panel(n_weeks=10)
    filtered = filter_panel(panel, category="FOODS", department="FOODS_1", weeks=4)
    assert sorted(filtered["week"].unique()) == [1, 2, 3, 4]


def test_filter_panel_preserves_starting_inventory_after_week_truncation():
    panel = _make_multi_category_panel(n_weeks=10)
    filtered = filter_panel(panel, department="FOODS_1", weeks=4)

    assert filtered["starting_inventory"].notna().sum() == 1
    assert filtered.sort_values("week").iloc[0]["starting_inventory"] == 40.0


def test_filter_panel_nonexistent_category_returns_empty():
    panel = _make_multi_category_panel()
    filtered = filter_panel(panel, category="ELECTRONICS")
    assert filtered.empty


def test_filter_panel_raises_on_missing_cat_id_column():
    panel = _make_multi_category_panel().drop(columns=["cat_id"])
    with pytest.raises(ValueError):
        filter_panel(panel, category="FOODS")


# --- run_solve ---

def test_run_solve_raises_on_empty_panel():
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        run_solve(empty)


def test_run_solve_produces_feasible_result_on_valid_subset():
    panel = _make_multi_category_panel(n_weeks=8)
    filtered = filter_panel(panel, department="FOODS_1")

    result = run_solve(filtered, window_size=6, step_size=3)

    assert result.all_feasible is True
    assert result.total_objective_value > 0


def test_run_solve_respects_department_isolation():
    panel = _make_multi_category_panel(n_weeks=6)
    result = run_solve(panel, window_size=4, step_size=2)

    dept_ids = {c.dept_id for c in result.cluster_results}
    assert dept_ids == {"FOODS_1", "FOODS_2", "HOBBIES_1"}


# --- baseline simulations ---

def test_flat_discount_baseline_produces_positive_revenue():
    panel = _make_multi_category_panel(n_weeks=8)
    revenue = simulate_flat_discount_baseline(panel)
    assert revenue > 0


def test_flat_discount_baseline_uses_discounted_price_not_base_price():
    """Sanity check: revenue should reflect the discounted price, not full base_price,
    so a naive bug reusing base_price directly would be caught here."""
    panel = _make_multi_category_panel(n_weeks=8)

    revenue_default_discount = simulate_flat_discount_baseline(panel, discount=FLAT_DISCOUNT)
    revenue_deeper_discount = simulate_flat_discount_baseline(panel, discount=0.50)

    # deeper discount -> lower price per unit; whether total revenue rises or
    # falls depends on elasticity, but the two runs must not be identical,
    # confirming the discount argument actually changes the computed price
    assert revenue_default_discount != revenue_deeper_discount


def test_calendar_baseline_produces_positive_revenue():
    panel = _make_multi_category_panel(n_weeks=8)
    revenue = simulate_calendar_baseline(panel)
    assert revenue > 0


def test_calendar_baseline_respects_inventory_ceiling():
    """Total revenue under the calendar baseline should never imply selling more
    units than a SKU's starting_inventory, across the whole horizon."""
    panel = _make_multi_category_panel(n_weeks=20)  # long horizon, should hit inventory limit
    revenue = simulate_calendar_baseline(panel)

    # upper bound: if every unit of every SKU's inventory sold at full price,
    # that's the absolute ceiling revenue could ever reach
    max_possible_revenue = panel.drop_duplicates(["item_id", "store_id"])["starting_inventory"].sum() * 10.0
    assert revenue <= max_possible_revenue


def test_baseline_functions_skip_skus_without_starting_inventory():
    """A SKU row lacking a starting_inventory value (e.g. filtered out by --weeks
    excluding its true first week) should be skipped, not crash the simulation."""
    panel = _make_multi_category_panel(n_weeks=6)
    # simulate a truncation that dropped the row carrying starting_inventory
    panel_missing_inventory = panel[panel["week"] > 1].copy()

    # should not raise, and should simply contribute 0 revenue for that SKU
    revenue = simulate_flat_discount_baseline(panel_missing_inventory)
    assert revenue == 0.0