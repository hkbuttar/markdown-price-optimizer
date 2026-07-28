"""
data_prep.py

Loads the raw M5 files (sell_prices, sales_train_evaluation, calendar),
melts the wide sales table to long format, joins in prices and dates,
and builds the full SKU-week panel across ALL categories, stores, and items.

Processes one category at a time to bound peak memory usage, then
concatenates into a single processed panel.

Usage:
    python -m src.data_prep
    python -m src.data_prep --category FOODS   # single category, for quick iteration
"""

import argparse
import gc
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ALL_CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_calendar(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load calendar.csv and map day columns (d_1, d_2, ...) to real dates.

    This version of calendar.csv has no 'd' column, so it's reconstructed
    from row order: row 0 -> d_1, row 1 -> d_2, etc. Calendar rows are
    chronological and align 1:1 with the d_1...d_n columns in the sales file.
    """
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    calendar = calendar.sort_values("date").reset_index(drop=True)
    calendar["d"] = "d_" + (calendar.index + 1).astype(str)

    cols = ["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    return calendar[cols]


def load_sell_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load sell_prices.csv: item/store price by week (wm_yr_wk). Downcast for memory."""
    prices = pd.read_csv(
        raw_dir / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    return prices


def melt_sales_chunk(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Melt one category's wide sales block into long format, with memory-efficient dtypes."""
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    long_df = sales_wide.melt(
        id_vars=ID_COLS,
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold",
    )
    long_df["units_sold"] = long_df["units_sold"].astype("int16")
    for col in ID_COLS:
        long_df[col] = long_df[col].astype("category")
    return long_df


def build_panel_for_category(
    category: str,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Build the SKU-week panel for a single category (keeps memory bounded)."""
    # read only this category's rows from the wide sales file
    sales_wide = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    sales_wide = sales_wide[sales_wide["cat_id"] == category].reset_index(drop=True)

    sales_long = melt_sales_chunk(sales_wide)
    del sales_wide
    gc.collect()

    panel = sales_long.merge(calendar, on="d", how="left")
    del sales_long
    gc.collect()

    panel = panel.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")

    weekly = (
        panel.groupby(["item_id", "store_id", "cat_id", "dept_id", "wm_yr_wk"], observed=True)
        .agg(
            units_sold=("units_sold", "sum"),
            sell_price=("sell_price", "mean"),
            week_start=("date", "min"),
            has_event=("event_name_1", lambda x: x.notna().any()),
        )
        .reset_index()
        .sort_values(["cat_id", "item_id", "store_id", "wm_yr_wk"])
    )

    del panel
    gc.collect()
    return weekly


def build_panel(
    categories: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """
    Build the full SKU-week panel across all categories, all stores, all items.
    Processes category-by-category to bound peak memory.
    """
    categories = categories or ALL_CATEGORIES

    calendar = load_calendar(raw_dir)
    prices = load_sell_prices(raw_dir)

    category_panels = []
    for cat in categories:
        print(f"Processing category: {cat} ...")
        cat_panel = build_panel_for_category(cat, calendar, prices, raw_dir)
        print(f"  -> {len(cat_panel):,} SKU-store-week rows")
        category_panels.append(cat_panel)

    full_panel = pd.concat(category_panels, ignore_index=True)
    return full_panel


def main():
    parser = argparse.ArgumentParser(description="Prepare full-scale M5 data into a SKU-week panel.")
    parser.add_argument(
        "--category",
        default=None,
        help="Single category to process (e.g. FOODS). Omit to process ALL categories.",
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Path to raw data directory")
    parser.add_argument("--out", default=str(PROCESSED_DIR / "sku_week_panel.csv"), help="Output CSV path")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    categories = [args.category] if args.category else ALL_CATEGORIES

    panel = build_panel(categories=categories, raw_dir=Path(args.raw_dir))

    panel.to_csv(args.out, index=False)
    print(f"\nWrote {len(panel):,} total SKU-store-week rows to {args.out}")
    print(panel.head())
    print(f"\nCategories included: {panel['cat_id'].unique().tolist()}")
    print(f"Stores included: {panel['store_id'].nunique()}")
    print(f"Items included: {panel['item_id'].nunique()}")


if __name__ == "__main__":
    main()