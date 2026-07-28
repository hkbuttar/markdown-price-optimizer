"""
inventory_assumptions.py

M5 does not include real inventory levels or unit costs. This module derives
both as documented modeling assumptions, calibrated off each SKU's own
observed sales history, so the assumptions are at least internally consistent
with real demand patterns rather than arbitrary constants.

Assumptions (state explicitly in the report):
- starting_inventory: `inventory_weeks_of_supply` times the SKU's average
  weekly units sold (a common retail heuristic -- inventory sized to cover
  N weeks of typical demand).
- unit_cost: `assumed_margin_fraction` below the SKU's own base_price
  (e.g. a 40% margin means cost = 60% of price).

Usage:
    python -m src.inventory_assumptions
"""

import argparse
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

DEFAULT_INVENTORY_WEEKS_OF_SUPPLY = 8
DEFAULT_ASSUMED_MARGIN_FRACTION = 0.40
MIN_STARTING_INVENTORY = 5.0


def add_inventory_and_cost_assumptions(
    panel_df: pd.DataFrame,
    inventory_weeks_of_supply: float = DEFAULT_INVENTORY_WEEKS_OF_SUPPLY,
    assumed_margin_fraction: float = DEFAULT_ASSUMED_MARGIN_FRACTION,
) -> pd.DataFrame:
    """
    Adds starting_inventory (only populated on each SKU's first week, matching
    what model.py expects) and unit_cost columns to a SKU-week panel.
    """
    df = panel_df.copy().sort_values(["item_id", "store_id", "week"]).reset_index(drop=True)

    avg_weekly_sales = (
        df.groupby(["item_id", "store_id"])["units_sold"]
        .transform("mean")
    )
    inventory_estimate = (avg_weekly_sales * inventory_weeks_of_supply).clip(lower=MIN_STARTING_INVENTORY)

    df["unit_cost"] = (df["base_price"] if "base_price" in df.columns else df["sell_price"]) * (1 - assumed_margin_fraction)

    first_week_mask = ~df.duplicated(subset=["item_id", "store_id"], keep="first")
    df["starting_inventory"] = None
    df.loc[first_week_mask, "starting_inventory"] = inventory_estimate[first_week_mask]

    return df


def main():
    parser = argparse.ArgumentParser(description="Add starting_inventory/unit_cost assumptions to the SKU-week panel.")
    parser.add_argument("--panel", default=str(PROCESSED_DIR / "sku_week_panel.csv"))
    parser.add_argument("--elasticities", default=str(PROCESSED_DIR / "elasticities.csv"))
    parser.add_argument("--out", default=str(PROCESSED_DIR / "model_ready_panel.csv"))
    parser.add_argument("--inventory-weeks", type=float, default=DEFAULT_INVENTORY_WEEKS_OF_SUPPLY)
    parser.add_argument("--margin", type=float, default=DEFAULT_ASSUMED_MARGIN_FRACTION)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    elasticities = pd.read_csv(args.elasticities)

    merged = panel.merge(
        elasticities[["item_id", "store_id", "base_price", "intercept", "elasticity"]],
        on=["item_id", "store_id"],
        how="inner",  # only keep SKUs that got a fitted elasticity curve
    )

    # normalize schema: data_prep.py's week identifier is 'wm_yr_wk' (M5's native
    # year-week code), but model.py/decomposition.py expect a column named 'week'.
    # Renaming here, once, at the model-ready seam, keeps every downstream file
    # working with a single consistent schema.
    merged = merged.rename(columns={"wm_yr_wk": "week"})

    model_ready = add_inventory_and_cost_assumptions(
        merged, inventory_weeks_of_supply=args.inventory_weeks, assumed_margin_fraction=args.margin
    )

    model_ready.to_csv(args.out, index=False)
    print(f"Wrote {len(model_ready):,} rows to {args.out}")
    print(f"SKUs with starting_inventory assigned: {model_ready['starting_inventory'].notna().sum():,}")


if __name__ == "__main__":
    main()