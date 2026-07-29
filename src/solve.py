"""
solve.py

Solves the markdown MIP via hybrid decomposition (department clustering +
SKU batching + rolling horizon), with optional filtering to a single
category/department/week window for fast iteration, and naive baseline
comparisons (flat discount, calendar-based markdown) for the Results section.

Usage:
    python -m src.solve                                   # full solve, all departments
    python -m src.solve --category FOODS                  # one category
    python -m src.solve --department HOBBIES_2 --weeks 8  # fast debug subset
    python -m src.solve --sku-batch-size 50               # smaller batches for harder departments
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.decomposition import solve_by_department, DEFAULT_SKU_BATCH_SIZE
from src.elasticity import predicted_demand
from src.model import DEFAULT_DISCOUNT_TIERS

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")

FLAT_DISCOUNT = 0.20  # naive baseline: flat 20% off every week
CALENDAR_SCHEDULE = [0.0, 0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.50]  # cycles if horizon is longer


def load_model_ready_panel(path: Path = PROCESSED_DIR / "model_ready_panel.csv") -> pd.DataFrame:
    """Load the panel already merged with elasticities and inventory/cost assumptions."""
    return pd.read_csv(path)


def filter_panel(
    df: pd.DataFrame,
    category: str | None = None,
    department: str | None = None,
    weeks: int | None = None,
) -> pd.DataFrame:
    """
    Filter the panel to a category and/or specific department, and/or truncate
    to the first N weeks per SKU (by sorted week value) for faster iteration.
    """
    filtered = df.copy()

    if category is not None:
        if "cat_id" not in filtered.columns:
            raise ValueError("panel has no 'cat_id' column to filter by category")
        filtered = filtered[filtered["cat_id"] == category]

    if department is not None:
        if "dept_id" not in filtered.columns:
            raise ValueError("panel has no 'dept_id' column to filter by department")
        filtered = filtered[filtered["dept_id"] == department]

    if filtered.empty:
        return filtered

    if weeks is not None:
        # keep only the first `weeks` distinct week values per SKU, so
        # starting_inventory (set on each SKU's true first week) is preserved
        sorted_weeks = sorted(filtered["week"].unique())[:weeks]
        filtered = filtered[filtered["week"].isin(sorted_weeks)]

    return filtered.reset_index(drop=True)


def run_solve(
    panel_df: pd.DataFrame,
    window_size: int = 8,
    step_size: int = 4,
    discount_tiers: list[float] = None,
    solver_name: str = "cbc",
    sku_batch_size: int = DEFAULT_SKU_BATCH_SIZE,
):
    """Run the hybrid decomposition solve on an already-filtered panel."""
    if panel_df.empty:
        raise ValueError("panel_df is empty after filtering -- check --category/--department/--weeks")

    return solve_by_department(
        panel_df,
        window_size=window_size,
        step_size=step_size,
        discount_tiers=discount_tiers or DEFAULT_DISCOUNT_TIERS,
        solver_name=solver_name,
        sku_batch_size=sku_batch_size,
    )


def _tier_lookup(base_price: float, intercept: float, elasticity: float, discount: float) -> tuple[float, float]:
    """Predicted (price, demand) for one specific discount level."""
    price = round(base_price * (1 - discount), 2)
    demand = predicted_demand(intercept, elasticity, price)
    return price, demand


def simulate_flat_discount_baseline(panel_df: pd.DataFrame, discount: float = FLAT_DISCOUNT) -> float:
    """
    Revenue under a naive flat-discount policy: every SKU discounted the same
    fixed amount every week, selling min(predicted demand, remaining inventory).
    """
    total_revenue = 0.0
    for (item_id, store_id), group in panel_df.groupby(["item_id", "store_id"]):
        group = group.sort_values("week")
        inv_series = group["starting_inventory"].dropna()
        if inv_series.empty:
            continue
        remaining_inv = inv_series.iloc[0]
        base_price = group["base_price"].iloc[0]
        intercept = group["intercept"].iloc[0]
        elasticity = group["elasticity"].iloc[0]

        price, weekly_demand = _tier_lookup(base_price, intercept, elasticity, discount)

        for _ in group["week"]:
            if remaining_inv <= 0:
                break
            sold = min(weekly_demand, remaining_inv)
            total_revenue += sold * price
            remaining_inv -= sold

    return total_revenue


def simulate_calendar_baseline(panel_df: pd.DataFrame, schedule: list[float] = None) -> float:
    """
    Revenue under a calendar-based markdown policy: discount deepens on a fixed
    weekly schedule (common retail heuristic), regardless of SKU-specific elasticity.
    """
    schedule = schedule or CALENDAR_SCHEDULE
    total_revenue = 0.0

    for (item_id, store_id), group in panel_df.groupby(["item_id", "store_id"]):
        group = group.sort_values("week").reset_index(drop=True)
        inv_series = group["starting_inventory"].dropna()
        if inv_series.empty:
            continue
        remaining_inv = inv_series.iloc[0]
        base_price = group["base_price"].iloc[0]
        intercept = group["intercept"].iloc[0]
        elasticity = group["elasticity"].iloc[0]

        for i in range(len(group)):
            if remaining_inv <= 0:
                break
            discount = schedule[i % len(schedule)]
            price, weekly_demand = _tier_lookup(base_price, intercept, elasticity, discount)
            sold = min(weekly_demand, remaining_inv)
            total_revenue += sold * price
            remaining_inv -= sold

    return total_revenue


def main():
    parser = argparse.ArgumentParser(description="Solve the markdown MIP, with optional filtering and baseline comparisons.")
    parser.add_argument("--panel", default=str(PROCESSED_DIR / "model_ready_panel.csv"))
    parser.add_argument("--category", default=None, help="Filter to one category (e.g. FOODS)")
    parser.add_argument("--department", default=None, help="Filter to one department (e.g. HOBBIES_2)")
    parser.add_argument("--weeks", type=int, default=None, help="Limit to the first N weeks per SKU")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--step-size", type=int, default=4)
    parser.add_argument("--sku-batch-size", type=int, default=DEFAULT_SKU_BATCH_SIZE, help="Max SKUs solved jointly per MIP window (higher = fewer, larger solves)")
    parser.add_argument("--solver", default="cbc")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip flat/calendar baseline comparisons (faster for quick debug runs)")
    parser.add_argument("--out-schedule", default=str(OUTPUTS_DIR / "sample_markdown_schedule.csv"))
    parser.add_argument("--out-summary", default=str(OUTPUTS_DIR / "results_summary.json"))
    args = parser.parse_args()

    panel = load_model_ready_panel(Path(args.panel))
    print(f"Loaded panel: {len(panel):,} rows")

    filtered = filter_panel(panel, category=args.category, department=args.department, weeks=args.weeks)
    n_skus = filtered.groupby(["item_id", "store_id"]).ngroups if not filtered.empty else 0
    print(f"Filtered panel: {len(filtered):,} rows, {n_skus:,} SKU-store pairs")

    print("\n--- Running MIP decomposition (hybrid: dept clustering + SKU batching + rolling horizon) ---")
    t0 = time.time()
    result = run_solve(
        filtered,
        window_size=args.window_size,
        step_size=args.step_size,
        solver_name=args.solver,
        sku_batch_size=args.sku_batch_size,
    )
    solve_time = time.time() - t0

    print(f"\nSolved in {solve_time:.1f}s")
    for c in result.cluster_results:
        flag = "OK" if c.feasible else "INFEASIBLE"
        print(f"  [{flag}] {c.dept_id}: objective={c.objective_value}, windows={c.n_windows_solved}")

    mip_revenue = result.total_objective_value

    summary = {
        "solve_time_seconds": solve_time,
        "mip_total_revenue": mip_revenue,
        "all_departments_feasible": result.all_feasible,
        "per_department_status": {
            c.dept_id: {"feasible": c.feasible, "objective_value": c.objective_value, "n_windows_solved": c.n_windows_solved}
            for c in result.cluster_results
        },
    }

    if not args.skip_baselines:
        print("\n--- Computing naive baselines ---")
        flat_revenue = simulate_flat_discount_baseline(filtered)
        calendar_revenue = simulate_calendar_baseline(filtered)

        print(f"MIP-optimized (hybrid decomposition): ${mip_revenue:,.2f}")
        print(f"Flat {int(FLAT_DISCOUNT*100)}% discount baseline:      ${flat_revenue:,.2f}")
        print(f"Calendar-based markdown baseline:      ${calendar_revenue:,.2f}")

        summary["flat_discount_baseline_revenue"] = flat_revenue
        summary["calendar_baseline_revenue"] = calendar_revenue

        if flat_revenue > 0:
            pct_vs_flat = (mip_revenue - flat_revenue) / flat_revenue * 100
            print(f"MIP recovers {pct_vs_flat:+.1f}% vs. flat discount")
            summary["pct_improvement_vs_flat"] = pct_vs_flat

        if calendar_revenue > 0:
            pct_vs_calendar = (mip_revenue - calendar_revenue) / calendar_revenue * 100
            print(f"MIP recovers {pct_vs_calendar:+.1f}% vs. calendar-based markdown")
            summary["pct_improvement_vs_calendar"] = pct_vs_calendar

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result.schedule.to_csv(args.out_schedule, index=False)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved schedule to {args.out_schedule}")
    print(f"Saved summary to {args.out_summary}")


if __name__ == "__main__":
    main()