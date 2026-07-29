"""
elasticity_filter.py

Filters the model-ready panel down to SKUs with a sufficiently reliable
elasticity fit, an economically sensible (negative) elasticity, AND
sufficient predicted demand to plausibly clear their assumed starting
inventory within the horizon. This is a documented modeling/scoping
decision, not a bug workaround: given the price-elasticity fit quality
observed on this dataset (median R^2 ~0.05), a meaningful fraction of SKUs
lack the price variation needed to support a reliable estimate -- and some
sparse fits produce economically backwards (positive) elasticities that
technically clear an R^2 threshold but are not usable demand curves.
Including these would make department-level MIPs infeasible end-to-end
rather than surfacing a useful markdown recommendation.

Usage:
    python -m src.elasticity_filter --min-r2 0.1
"""

import argparse
from pathlib import Path

import pandas as pd

from src.elasticity import predicted_demand
from src.model import DEFAULT_DISCOUNT_TIERS

PROCESSED_DIR = Path("data/processed")

DEFAULT_MIN_R2 = 0.10
DEFAULT_CLEARANCE_FRACTION_CHECK = 0.95
DEFAULT_MAX_ELASTICITY = 0.0    # elasticity must be negative (demand falls as price rises)
DEFAULT_MIN_ELASTICITY = -10.0  # guard against implausibly extreme negative fits too


def max_sellable_units(row: pd.Series, n_weeks: int, discount_tiers: list[float] = None) -> float:
    """Best-case total units sellable over n_weeks at the deepest-demand tier."""
    discount_tiers = discount_tiers or DEFAULT_DISCOUNT_TIERS
    best_demand = max(
        predicted_demand(row["intercept"], row["elasticity"], round(row["base_price"] * (1 - d), 2))
        for d in discount_tiers
    )
    return best_demand * n_weeks


def filter_to_feasible_skus(
    panel_df: pd.DataFrame,
    min_r2: float = DEFAULT_MIN_R2,
    clearance_fraction: float = DEFAULT_CLEARANCE_FRACTION_CHECK,
    max_elasticity: float = DEFAULT_MAX_ELASTICITY,
    min_elasticity: float = DEFAULT_MIN_ELASTICITY,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (filtered_panel, stats) where stats reports how many SKUs were
    excluded and why -- worth reporting directly in the results/limitations
    section rather than hiding.
    """
    sku_level = panel_df.drop_duplicates(["item_id", "store_id"]).dropna(subset=["starting_inventory"])

    n_weeks_by_sku = panel_df.groupby(["item_id", "store_id"])["week"].nunique()
    sku_level = sku_level.merge(
        n_weeks_by_sku.rename("n_weeks"), on=["item_id", "store_id"], how="left"
    )

    sku_level["max_sellable"] = sku_level.apply(
        lambda row: max_sellable_units(row, n_weeks=row["n_weeks"]), axis=1
    )
    sku_level["structurally_feasible"] = sku_level["max_sellable"] >= clearance_fraction * sku_level["starting_inventory"]
    sku_level["fit_reliable"] = sku_level["r2"] >= min_r2
    # economically sensible: elasticity must be negative (demand falls as price
    # rises) and not implausibly extreme -- catches overfit garbage that
    # technically clears the R2 bar but has the wrong sign or magnitude
    sku_level["elasticity_sensible"] = (
        (sku_level["elasticity"] < max_elasticity) & (sku_level["elasticity"] >= min_elasticity)
    )

    keep_mask = sku_level["structurally_feasible"] & sku_level["fit_reliable"] & sku_level["elasticity_sensible"]
    keep_keys = set(zip(sku_level.loc[keep_mask, "item_id"], sku_level.loc[keep_mask, "store_id"]))

    filtered = panel_df[
        panel_df.apply(lambda r: (r["item_id"], r["store_id"]) in keep_keys, axis=1)
    ].reset_index(drop=True)

    stats = {
        "total_skus": len(sku_level),
        "excluded_low_r2": int((~sku_level["fit_reliable"]).sum()),
        "excluded_bad_elasticity_sign": int((~sku_level["elasticity_sensible"] & sku_level["fit_reliable"]).sum()),
        "excluded_structurally_infeasible": int(
            (~sku_level["structurally_feasible"] & sku_level["fit_reliable"] & sku_level["elasticity_sensible"]).sum()
        ),
        "kept": int(keep_mask.sum()),
        "kept_fraction": float(keep_mask.mean()),
    }

    return filtered, stats


def main():
    parser = argparse.ArgumentParser(description="Filter model-ready panel to SKUs with reliable, economically sensible, and feasible elasticity fits.")
    parser.add_argument("--panel", default=str(PROCESSED_DIR / "model_ready_panel.csv"))
    parser.add_argument("--out", default=str(PROCESSED_DIR / "model_ready_panel_filtered.csv"))
    parser.add_argument("--min-r2", type=float, default=DEFAULT_MIN_R2)
    parser.add_argument("--clearance-fraction", type=float, default=DEFAULT_CLEARANCE_FRACTION_CHECK)
    parser.add_argument("--max-elasticity", type=float, default=DEFAULT_MAX_ELASTICITY)
    parser.add_argument("--min-elasticity", type=float, default=DEFAULT_MIN_ELASTICITY)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    filtered, stats = filter_to_feasible_skus(
        panel,
        min_r2=args.min_r2,
        clearance_fraction=args.clearance_fraction,
        max_elasticity=args.max_elasticity,
        min_elasticity=args.min_elasticity,
    )

    filtered.to_csv(args.out, index=False)

    print(f"Total SKUs: {stats['total_skus']:,}")
    print(f"Excluded (R\u00b2 < {args.min_r2}): {stats['excluded_low_r2']:,}")
    print(f"Excluded (elasticity sign/magnitude implausible): {stats['excluded_bad_elasticity_sign']:,}")
    print(f"Excluded (structurally infeasible at {args.clearance_fraction:.0%} clearance): {stats['excluded_structurally_infeasible']:,}")
    print(f"Kept: {stats['kept']:,} ({stats['kept_fraction']:.1%})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()