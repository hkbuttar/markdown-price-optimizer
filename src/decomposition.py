"""
decomposition.py

Hybrid decomposition for the markdown MIP: solving every SKU jointly over the
full clearance horizon is computationally impractical at full catalog scale,
so this module splits the problem three ways:

1. Category/department clustering -- SKUs are grouped by dept_id (a natural,
   pre-existing grouping) and solved as independent sub-MIPs, since
   cross-department price interactions are assumed weak.
2. SKU batching within department -- large departments are further split into
   batches of at most sku_batch_size SKUs, since solving that many SKUs
   jointly within a single rolling-horizon window is itself a combinatorially
   hard MIP. Batches are solved independently and their schedules/objectives
   are combined.
3. Rolling-horizon decomposition -- within a batch, instead of solving all
   weeks jointly, a sliding window of `window_size` weeks is solved exactly,
   the first `step_size` weeks are committed, inventory is rolled forward
   using the *actual* committed sales, and the window advances. The
   end-of-horizon clearance constraint is only enforced on the window that
   reaches the true end of the horizon.

Cross-window monotonicity: each window's MIP enforces monotonic (non-increasing)
pricing WITHIN that window, but nothing inherently stops the NEXT window from
re-solving back toward tier 0 (full price) once a fresh window begins. To
prevent price from illegally "resetting" upward across a window boundary, the
last COMMITTED tier for each SKU is tracked and passed into the next window's
build_model() call as a floor (min_tier_by_sku), fixing any shallower tier to
zero on that SKU's first week in the new window.

Usage:
    from src.decomposition import solve_by_department
    result = solve_by_department(panel_df, window_size=8, step_size=4, sku_batch_size=100)
"""

from dataclasses import dataclass, field

import pandas as pd

from src.model import build_model, solve_model, DEFAULT_CLEARANCE_FRACTION, DEFAULT_DISCOUNT_TIERS

DEFAULT_SKU_BATCH_SIZE = 100


@dataclass
class ClusterResult:
    dept_id: str
    feasible: bool
    schedule: pd.DataFrame
    objective_value: float | None
    n_windows_solved: int


@dataclass
class DecompositionResult:
    schedule: pd.DataFrame
    cluster_results: list[ClusterResult] = field(default_factory=list)

    @property
    def total_objective_value(self) -> float:
        return sum(c.objective_value for c in self.cluster_results if c.objective_value is not None)

    @property
    def all_feasible(self) -> bool:
        return all(c.feasible for c in self.cluster_results)


def cluster_by_department(panel_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the panel into one sub-panel per dept_id."""
    if "dept_id" not in panel_df.columns:
        raise ValueError("panel_df must include a 'dept_id' column to cluster by department")
    return {dept: group.reset_index(drop=True) for dept, group in panel_df.groupby("dept_id")}


def _weeks_sorted(panel_df: pd.DataFrame) -> list:
    return sorted(panel_df["week"].unique())


def _make_windows(weeks: list, window_size: int, step_size: int) -> list[list]:
    """
    Build overlapping (or adjacent, if step_size == window_size) rolling windows
    over the full week list. The last window is truncated/extended to always
    reach the final week, so the true horizon end is always covered exactly once.
    """
    if step_size > window_size:
        raise ValueError("step_size cannot exceed window_size (would skip weeks)")

    windows = []
    start_idx = 0
    n = len(weeks)

    while start_idx < n:
        end_idx = min(start_idx + window_size, n)
        windows.append(weeks[start_idx:end_idx])
        if end_idx == n:
            break
        start_idx += step_size

    return windows


def _batch_skus(sku_keys: list[tuple], batch_size: int) -> list[list[tuple]]:
    """Split a department's SKU list into batches of at most batch_size SKUs,
    so no single rolling-horizon window ever solves more than batch_size SKUs jointly."""
    return [sku_keys[i:i + batch_size] for i in range(0, len(sku_keys), batch_size)]


def rolling_horizon_solve(
    sku_panel_df: pd.DataFrame,
    window_size: int = 8,
    step_size: int = 4,
    discount_tiers: list[float] = None,
    clearance_fraction: float = DEFAULT_CLEARANCE_FRACTION,
    solver_name: str = "cbc",
) -> tuple[pd.DataFrame, float, bool, int]:
    """
    Solve one batch's panel (already filtered to a single department/batch, or
    any SKU set small enough for model.build_model to handle directly) using a
    rolling horizon. Returns (committed_schedule, total_objective, all_feasible, n_windows).
    """
    discount_tiers = discount_tiers or DEFAULT_DISCOUNT_TIERS
    weeks = _weeks_sorted(sku_panel_df)
    final_week = weeks[-1]
    windows = _make_windows(weeks, window_size, step_size)

    # Precompute each SKU's true starting inventory ONCE from the full panel.
    # This removes any dependence on that value happening to fall neatly on
    # a window's first week -- it's restored explicitly wherever needed below.
    sku_true_inventory: dict[tuple, float] = {}
    for (item_id, store_id), group in sku_panel_df.groupby(["item_id", "store_id"]):
        inv_series = group["starting_inventory"].dropna()
        if not inv_series.empty:
            sku_true_inventory[(item_id, store_id)] = float(inv_series.iloc[0])

    committed_rows = []
    total_objective = 0.0
    all_feasible = True
    current_inventory: dict[tuple, float] = {}
    # tracks the last COMMITTED tier per SKU, carried forward as a floor so
    # price cannot illegally rise back toward full price at a window boundary
    min_tier_floor: dict[tuple, int] = {}

    for window_idx, window_weeks in enumerate(windows):
        is_final_window = final_week in window_weeks
        window_df = sku_panel_df[sku_panel_df["week"].isin(window_weeks)].copy()
        first_week_of_window = window_weeks[0]

        for (item_id, store_id), group in window_df.groupby(["item_id", "store_id"]):
            sku = (item_id, store_id)
            group_sorted = group.sort_values("week")
            earliest_week_in_window = group_sorted["week"].iloc[0]

            if sku in current_inventory:
                # rolled forward from a prior window: overwrite this window's
                # first week with the actual remaining inventory
                mask = (
                    (window_df["item_id"] == item_id)
                    & (window_df["store_id"] == store_id)
                    & (window_df["week"] == first_week_of_window)
                )
                window_df.loc[mask, "starting_inventory"] = current_inventory[sku]
            elif not group["starting_inventory"].notna().any():
                # first appearance of this SKU in any window, but its true
                # starting_inventory row didn't land inside this window's
                # slice -- restore it explicitly on the earliest row present
                true_inv = sku_true_inventory.get(sku)
                if true_inv is not None:
                    mask = (
                        (window_df["item_id"] == item_id)
                        & (window_df["store_id"] == store_id)
                        & (window_df["week"] == earliest_week_in_window)
                    )
                    window_df.loc[mask, "starting_inventory"] = true_inv
            # else: this SKU's true first row is already intact in this window

        window_clearance = clearance_fraction if is_final_window else 0.0

        m = build_model(
            window_df,
            discount_tiers=discount_tiers,
            clearance_fraction=window_clearance,
            min_tier_by_sku=min_tier_floor,
        )
        result = solve_model(m, solver_name=solver_name)

        if not result.feasible:
            all_feasible = False
            continue

        commit_weeks = window_weeks if is_final_window else window_weeks[:step_size]
        committed = result.schedule[result.schedule["week"].isin(commit_weeks)].copy()
        committed_rows.append(committed)

        committed_revenue = (committed["price"] * committed["units_sold"]).sum()
        total_objective += committed_revenue

        for (item_id, store_id), sku_committed in committed.groupby(["item_id", "store_id"]):
            sku = (item_id, store_id)
            sku_committed_sorted = sku_committed.sort_values("week")

            starting_inv = window_df[
                (window_df["item_id"] == item_id) & (window_df["store_id"] == store_id)
            ]["starting_inventory"].dropna().iloc[0]
            sold_in_committed_weeks = sku_committed_sorted["units_sold"].sum()
            current_inventory[sku] = max(starting_inv - sold_in_committed_weeks, 0.0)

            # carry forward the last committed tier as next window's floor,
            # so price cannot illegally rise back toward full price
            last_committed_tier = sku_committed_sorted["tier"].iloc[-1]
            min_tier_floor[sku] = int(last_committed_tier)

    if committed_rows:
        full_schedule = pd.concat(committed_rows, ignore_index=True).sort_values(
            ["item_id", "store_id", "week"]
        ).reset_index(drop=True)
    else:
        full_schedule = pd.DataFrame()

    return full_schedule, total_objective, all_feasible, len(windows)


def solve_by_department(
    panel_df: pd.DataFrame,
    window_size: int = 8,
    step_size: int = 4,
    discount_tiers: list[float] = None,
    clearance_fraction: float = DEFAULT_CLEARANCE_FRACTION,
    solver_name: str = "cbc",
    departments: list[str] = None,
    sku_batch_size: int = DEFAULT_SKU_BATCH_SIZE,
) -> DecompositionResult:
    """
    Full hybrid solve: cluster by department, sub-batch large departments by
    SKU count to keep each joint MIP tractable, then solve each batch with a
    rolling horizon. This is the top-level entry point that replaces
    attempting one monolithic MIP over the entire catalog.
    """
    clusters = cluster_by_department(panel_df)
    if departments is not None:
        clusters = {dept: df for dept, df in clusters.items() if dept in departments}

    cluster_results = []
    all_schedules = []

    for dept_id, dept_df in clusters.items():
        sku_keys = list(dept_df[["item_id", "store_id"]].drop_duplicates().itertuples(index=False, name=None))
        sku_batches = _batch_skus(sku_keys, sku_batch_size)

        dept_schedules = []
        dept_objective = 0.0
        dept_feasible = True
        dept_windows = 0

        for batch_idx, batch_keys in enumerate(sku_batches):
            batch_keys_df = pd.DataFrame(batch_keys, columns=["item_id", "store_id"])
            batch_df = dept_df.merge(batch_keys_df, on=["item_id", "store_id"])

            print(f"    {dept_id} batch {batch_idx + 1}/{len(sku_batches)} ({len(batch_keys)} SKUs)...")

            schedule, objective_value, feasible, n_windows = rolling_horizon_solve(
                batch_df,
                window_size=window_size,
                step_size=step_size,
                discount_tiers=discount_tiers,
                clearance_fraction=clearance_fraction,
                solver_name=solver_name,
            )

            if not schedule.empty:
                dept_schedules.append(schedule)
            dept_objective += objective_value
            dept_feasible = dept_feasible and feasible
            dept_windows += n_windows

        dept_schedule = (
            pd.concat(dept_schedules, ignore_index=True) if dept_schedules else pd.DataFrame()
        )
        if not dept_schedule.empty:
            dept_schedule = dept_schedule.copy()
            dept_schedule["dept_id"] = dept_id
            all_schedules.append(dept_schedule)

        cluster_results.append(ClusterResult(
            dept_id=dept_id,
            feasible=dept_feasible,
            schedule=dept_schedule,
            objective_value=dept_objective,
            n_windows_solved=dept_windows,
        ))

    combined_schedule = (
        pd.concat(all_schedules, ignore_index=True) if all_schedules else pd.DataFrame()
    )

    return DecompositionResult(schedule=combined_schedule, cluster_results=cluster_results)