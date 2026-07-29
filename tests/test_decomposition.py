"""
tests/test_decomposition.py

Tests for department clustering, SKU batching, and rolling-horizon decomposition.
Uses small synthetic panels so these run fast in CI while still exercising
the actual window-splitting, batching, commit, and inventory-rollforward logic.
"""

import pandas as pd
import pytest

from src.decomposition import (
    cluster_by_department,
    _make_windows,
    _batch_skus,
    rolling_horizon_solve,
    solve_by_department,
)


def _make_sku_panel(n_weeks: int, starting_inventory: float = 40.0, base_price: float = 10.0, unit_cost: float = 3.0) -> pd.DataFrame:
    """A single-SKU panel spanning n_weeks, with intercept/elasticity set for a
    mild, well-behaved demand curve so the MIP solves quickly and predictably.
    intercept=4.0 and starting_inventory=40.0 keep the 95% clearance target
    achievable within the test horizon given this demand level."""
    weeks = list(range(1, n_weeks + 1))
    rows = []
    for i, w in enumerate(weeks):
        rows.append({
            "item_id": "ITEM_1",
            "store_id": "STORE_1",
            "dept_id": "DEPT_A",
            "week": w,
            "starting_inventory": starting_inventory if i == 0 else None,
            "unit_cost": unit_cost,
            "base_price": base_price,
            "intercept": 4.0,
            "elasticity": -1.0,
        })
    return pd.DataFrame(rows)


def _make_two_dept_panel(n_weeks: int = 4) -> pd.DataFrame:
    dept_a = _make_sku_panel(n_weeks)
    dept_b = _make_sku_panel(n_weeks)
    dept_b["item_id"] = "ITEM_2"
    dept_b["store_id"] = "STORE_2"
    dept_b["dept_id"] = "DEPT_B"
    return pd.concat([dept_a, dept_b], ignore_index=True)


def test_cluster_by_department_splits_correctly():
    """Each department should get its own sub-panel, with no cross-contamination."""
    panel = _make_two_dept_panel()
    clusters = cluster_by_department(panel)

    assert set(clusters.keys()) == {"DEPT_A", "DEPT_B"}
    assert (clusters["DEPT_A"]["item_id"] == "ITEM_1").all()
    assert (clusters["DEPT_B"]["item_id"] == "ITEM_2").all()


def test_cluster_by_department_requires_dept_column():
    """Should raise a clear error if dept_id is missing, rather than failing silently."""
    panel = _make_sku_panel(4).drop(columns=["dept_id"])
    with pytest.raises(ValueError):
        cluster_by_department(panel)


def test_make_windows_covers_all_weeks_without_gaps():
    """Every week in the horizon should appear in at least one window."""
    weeks = list(range(1, 13))  # 12 weeks
    windows = _make_windows(weeks, window_size=8, step_size=4)

    covered = set()
    for w in windows:
        covered.update(w)
    assert covered == set(weeks)


def test_make_windows_final_window_reaches_horizon_end():
    """The last window must always include the final week of the horizon."""
    weeks = list(range(1, 15))  # 14 weeks, doesn't divide evenly by step_size=4
    windows = _make_windows(weeks, window_size=8, step_size=4)
    assert windows[-1][-1] == weeks[-1]


def test_make_windows_rejects_step_larger_than_window():
    """step_size > window_size would skip weeks entirely -- should raise, not silently skip."""
    with pytest.raises(ValueError):
        _make_windows(list(range(1, 10)), window_size=4, step_size=6)


def test_batch_skus_splits_into_correct_sizes():
    """Batching should split a SKU list into chunks of at most batch_size, with no SKU dropped."""
    sku_keys = [(f"ITEM_{i}", "STORE_1") for i in range(23)]
    batches = _batch_skus(sku_keys, batch_size=10)

    assert len(batches) == 3
    assert [len(b) for b in batches] == [10, 10, 3]
    all_keys = [key for batch in batches for key in batch]
    assert all_keys == sku_keys


def test_batch_skus_single_batch_when_under_limit():
    """A SKU list smaller than batch_size should produce exactly one batch."""
    sku_keys = [(f"ITEM_{i}", "STORE_1") for i in range(5)]
    batches = _batch_skus(sku_keys, batch_size=100)
    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_rolling_horizon_solve_is_feasible_and_commits_all_weeks():
    """A small, well-behaved single-SKU panel should solve feasibly across all windows,
    with every week ultimately appearing exactly once in the committed schedule."""
    panel = _make_sku_panel(n_weeks=10)
    schedule, objective_value, feasible, n_windows = rolling_horizon_solve(
        panel, window_size=6, step_size=3
    )

    assert feasible is True
    assert n_windows >= 2
    assert sorted(schedule["week"].unique()) == list(range(1, 11))
    assert schedule.groupby("week").size().eq(1).all()


def test_rolling_horizon_solve_respects_clearance_only_at_true_horizon_end():
    """Total units sold across the full horizon should meet the clearance fraction,
    even though interior windows are solved without a clearance requirement."""
    panel = _make_sku_panel(n_weeks=10, starting_inventory=40.0)
    schedule, _, feasible, _ = rolling_horizon_solve(
        panel, window_size=6, step_size=3, clearance_fraction=0.95
    )

    assert feasible is True
    total_sold = schedule["units_sold"].sum()
    assert total_sold >= 0.95 * 40.0


def test_solve_by_department_returns_one_cluster_result_per_department():
    """solve_by_department should produce a separate ClusterResult for each department."""
    panel = _make_two_dept_panel(n_weeks=6)
    result = solve_by_department(panel, window_size=4, step_size=2)

    dept_ids = {c.dept_id for c in result.cluster_results}
    assert dept_ids == {"DEPT_A", "DEPT_B"}


def test_solve_by_department_combined_schedule_includes_both_departments():
    """The combined schedule should include committed rows from every department cluster."""
    panel = _make_two_dept_panel(n_weeks=6)
    result = solve_by_department(panel, window_size=4, step_size=2)

    assert set(result.schedule["dept_id"].unique()) == {"DEPT_A", "DEPT_B"}


def test_solve_by_department_can_filter_to_subset_of_departments():
    """Passing `departments=` should restrict solving to only those clusters."""
    panel = _make_two_dept_panel(n_weeks=6)
    result = solve_by_department(panel, window_size=4, step_size=2, departments=["DEPT_A"])

    assert len(result.cluster_results) == 1
    assert result.cluster_results[0].dept_id == "DEPT_A"


def test_solve_by_department_batches_large_departments_by_sku_count():
    """A department with more SKUs than sku_batch_size should still solve
    correctly end-to-end, just internally split into multiple batches."""
    panels = []
    for i in range(5):
        p = _make_sku_panel(n_weeks=6)
        p["item_id"] = f"ITEM_{i}"
        panels.append(p)
    dept_panel = pd.concat(panels, ignore_index=True)

    result = solve_by_department(dept_panel, window_size=4, step_size=2, sku_batch_size=2)

    assert len(result.cluster_results) == 1
    assert result.cluster_results[0].feasible is True
    assert set(result.schedule["item_id"].unique()) == {f"ITEM_{i}" for i in range(5)}