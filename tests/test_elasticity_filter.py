"""
tests/test_elasticity_filter.py

Tests for the elasticity-quality / structural-feasibility filtering step.
"""

import pandas as pd
import pytest

from src.elasticity_filter import max_sellable_units, filter_to_feasible_skus


def _make_sku_row(intercept: float, elasticity: float, base_price: float, r2: float, starting_inventory: float) -> dict:
    return {
        "intercept": intercept,
        "elasticity": elasticity,
        "base_price": base_price,
        "r2": r2,
        "starting_inventory": starting_inventory,
    }


def _make_panel_from_specs(specs: list[dict], n_weeks: int = 8) -> pd.DataFrame:
    rows = []
    for i, spec in enumerate(specs):
        item_id = f"ITEM_{i}"
        for i_week, w in enumerate(range(1, n_weeks + 1)):
            rows.append({
                "item_id": item_id,
                "store_id": "STORE_1",
                "week": w,
                "starting_inventory": spec["starting_inventory"] if i_week == 0 else None,
                "base_price": spec["base_price"],
                "intercept": spec["intercept"],
                "elasticity": spec["elasticity"],
                "r2": spec["r2"],
            })
    return pd.DataFrame(rows)


def test_max_sellable_units_scales_with_weeks():
    """Doubling the horizon should double max sellable units."""
    row = pd.Series(_make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.5, starting_inventory=40))
    sellable_4wk = max_sellable_units(row, n_weeks=4)
    sellable_8wk = max_sellable_units(row, n_weeks=8)
    assert sellable_8wk == pytest.approx(sellable_4wk * 2)


def test_filter_excludes_low_r2_skus():
    """A SKU with r2 below the threshold should be excluded regardless of feasibility."""
    specs = [
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.05, starting_inventory=20),  # low r2
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.50, starting_inventory=20),  # good r2
    ]
    panel = _make_panel_from_specs(specs)
    filtered, stats = filter_to_feasible_skus(panel, min_r2=0.10)

    assert "ITEM_0" not in filtered["item_id"].unique()
    assert "ITEM_1" in filtered["item_id"].unique()
    assert stats["excluded_low_r2"] == 1


def test_filter_excludes_structurally_infeasible_skus():
    """A SKU whose max sellable demand can't clear its inventory should be
    excluded even with a good r2 fit."""
    specs = [
        _make_sku_row(intercept=1.0, elasticity=-0.5, base_price=10.0, r2=0.50, starting_inventory=10000),  # tiny demand, huge inventory
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.50, starting_inventory=20),      # feasible
    ]
    panel = _make_panel_from_specs(specs)
    filtered, stats = filter_to_feasible_skus(panel, min_r2=0.10)

    assert "ITEM_0" not in filtered["item_id"].unique()
    assert "ITEM_1" in filtered["item_id"].unique()
    assert stats["excluded_structurally_infeasible"] == 1


def test_filter_excludes_positive_elasticity_as_economically_nonsensical():
    """A positive elasticity implies demand rises with price -- backwards for a
    normal good, and a sign of an overfit/degenerate regression, not a usable curve."""
    specs = [
        _make_sku_row(intercept=1.2, elasticity=20.7, base_price=1.0, r2=0.15, starting_inventory=10),  # positive elasticity
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.50, starting_inventory=20),  # normal
    ]
    panel = _make_panel_from_specs(specs)
    filtered, stats = filter_to_feasible_skus(panel, min_r2=0.10)

    assert "ITEM_0" not in filtered["item_id"].unique()
    assert "ITEM_1" in filtered["item_id"].unique()
    assert stats["excluded_bad_elasticity_sign"] == 1


def test_filter_excludes_implausibly_extreme_negative_elasticity():
    """An elasticity far beyond min_elasticity is also likely a degenerate fit
    on sparse data, not a genuine extremely price-sensitive product."""
    specs = [
        _make_sku_row(intercept=4.0, elasticity=-50.0, base_price=10.0, r2=0.50, starting_inventory=20),  # implausibly extreme
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.50, starting_inventory=20),   # normal
    ]
    panel = _make_panel_from_specs(specs)
    filtered, stats = filter_to_feasible_skus(panel, min_r2=0.10, min_elasticity=-10.0)

    assert "ITEM_0" not in filtered["item_id"].unique()
    assert "ITEM_1" in filtered["item_id"].unique()


def test_filter_keeps_skus_passing_both_checks():
    """A SKU with a good fit, sensible elasticity, and achievable clearance should be retained."""
    specs = [
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.50, starting_inventory=20),
    ]
    panel = _make_panel_from_specs(specs)
    filtered, stats = filter_to_feasible_skus(panel, min_r2=0.10)

    assert stats["kept"] == 1
    assert stats["kept_fraction"] == 1.0


def test_filter_stats_total_matches_unique_skus():
    """Stats total_skus should match the number of distinct SKUs in the input."""
    specs = [
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.50, starting_inventory=20),
        _make_sku_row(intercept=4.0, elasticity=-1.0, base_price=10.0, r2=0.02, starting_inventory=20),
        _make_sku_row(intercept=1.0, elasticity=-0.3, base_price=10.0, r2=0.50, starting_inventory=99999),
    ]
    panel = _make_panel_from_specs(specs)
    _, stats = filter_to_feasible_skus(panel, min_r2=0.10)

    assert stats["total_skus"] == 3