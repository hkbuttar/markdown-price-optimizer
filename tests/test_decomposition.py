"""
tests/test_decomposition.py

Tests for department clustering and rolling-horizon decomposition.
Uses small synthetic panels so these run fast in CI while still exercising
the actual window-splitting, commit, and inventory-rollforward logic.
"""

import pandas as pd
import pytest

from src.decomposition import (
    cluster_by_department,
    _make_windows,
    rolling_horizon_solve,
    solve_by_department,
)


def _make_sku_panel(n_weeks: int, starting_inventory: float = 40.0, base_price: float = 10.0, unit_cost: float = 3.0) -> pd.DataFrame:
    """A single-SKU panel spanning n_weeks, with intercept/elasticity set for a
    mild, well-behaved demand curve so the MIP solves quickly and predictably.
    intercept=4.0 (rather than 2.0) ensures meaningful demand even at full price,
    and starting_inventory=40.0 keeps the 95% clearance target achievable within
    the test horizon given that demand level."""
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


def test_rolling_horizon_solve_is_feasible_and_commits_all_weeks():
    """A small, well-behaved single-SKU panel should solve feasibly across all windows,
    with every week ultimately appearing exactly once in the committed schedule."""
    panel = _make_sku_panel(n_weeks=10)
    schedule, objective_value, feasible, n_windows = rolling_horizon_solve(
        panel, window_size=6, step_size=3
    )

    assert feasible is True
    assert n_windows >= 2  # confirms multiple windows actually ran, not just one
    assert sorted(schedule["week"].unique()) == list(range(1, 11))
    # each week committed exactly once -- no double-counting across overlapping windows
    assert schedule.groupby("week").size().eq(1).all()


def test_rolling_horizon_solve_respects_clearance_only_at_true_horizon_end():
    """Total units sold across the full horizon should meet the clearance fraction,
    even though interior windows are solved without a clearance requirement."""
    panel = _make_sku_panel(n_weeks=10, starting_inventory=100.0)
    schedule, _, feasible, _ = rolling_horizon_solve(
        panel, window_size=6, step_size=3, clearance_fraction=0.95
    )

    assert feasible is True
    total_sold = schedule["units_sold"].sum()
    assert total_sold >= 0.95 * 100.0


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