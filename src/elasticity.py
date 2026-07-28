"""
elasticity.py

Fits price-response (elasticity) curves per item-store, with a department-level
pooled fallback when an item-store pair has too few observations to fit reliably.
Also builds piecewise-linear demand breakpoints for each SKU's discount ladder,
for use in model.py's SOS2 constraints.

Model: log(units_sold + 1) = intercept + elasticity * log(sell_price) + event_coef * has_event

Usage:
    python -m src.elasticity
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
MIN_OBS_FOR_ITEM_STORE_FIT = 10

DEFAULT_DISCOUNT_TIERS = [0.0, 0.10, 0.20, 0.30, 0.50]  # 0%, 10%, 20%, 30%, 50% off


def load_panel(path: Path = PROCESSED_DIR / "sku_week_panel.csv") -> pd.DataFrame:
    """Load the processed SKU-week panel and drop rows with no recorded price."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["sell_price"]).copy()
    df = df[df["sell_price"] > 0]
    return df


def _fit_log_log(df: pd.DataFrame) -> dict | None:
    """
    Fit log(units_sold + 1) = intercept + elasticity * log(price) + event_coef * has_event
    via ordinary least squares (closed-form, no external regression dependency).
    Returns None if there's no price variation to estimate elasticity from.
    """
    log_price = np.log(df["sell_price"].to_numpy())
    log_units = np.log(df["units_sold"].to_numpy() + 1)
    event = df["has_event"].astype(int).to_numpy()

    if np.ptp(log_price) < 1e-6:
        return None  # no price variation to estimate elasticity from

    X = np.column_stack([np.ones(len(df)), log_price, event])
    y = log_units

    try:
        coefs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    # lstsq handles rank-deficient X gracefully (e.g. no event weeks in the sample
    # -> all-zero event column) via a least-norm solution, so no strict rank check here.

    intercept, elasticity, event_coef = coefs
    y_pred = X @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "intercept": float(intercept),
        "elasticity": float(elasticity),
        "event_coef": float(event_coef),
        "r2": float(r2),
        "n_obs": len(df),
    }


def fit_dept_pooled_elasticities(df: pd.DataFrame) -> dict[str, dict]:
    """Fit one pooled elasticity curve per department, used as a fallback."""
    pooled = {}
    for dept_id, group in df.groupby("dept_id", observed=True):
        fit = _fit_log_log(group)
        if fit is not None:
            pooled[dept_id] = fit
    return pooled


def fit_elasticities(
    df: pd.DataFrame,
    min_obs: int = MIN_OBS_FOR_ITEM_STORE_FIT,
) -> pd.DataFrame:
    """
    Fit elasticity per item-store pair. Falls back to the department-pooled
    estimate when an item-store pair has fewer than `min_obs` observations
    or a degenerate fit (e.g. constant price, so no variation to learn from).

    Returns a DataFrame: item_id, store_id, dept_id, base_price, intercept,
    elasticity, event_coef, r2, n_obs, fit_level ('item_store' or 'dept_pooled').
    """
    dept_fallbacks = fit_dept_pooled_elasticities(df)

    records = []
    for (item_id, store_id), group in df.groupby(["item_id", "store_id"], observed=True):
        dept_id = group["dept_id"].iloc[0]
        base_price = group["sell_price"].median()

        fit = None
        fit_level = None
        if len(group) >= min_obs:
            fit = _fit_log_log(group)
            fit_level = "item_store"

        if fit is None:
            fit = dept_fallbacks.get(dept_id)
            fit_level = "dept_pooled"

        if fit is None:
            # no usable fit at any level (e.g. entire dept has constant price) — skip
            continue

        records.append({
            "item_id": item_id,
            "store_id": store_id,
            "dept_id": dept_id,
            "base_price": base_price,
            "intercept": fit["intercept"],
            "elasticity": fit["elasticity"],
            "event_coef": fit["event_coef"],
            "r2": fit["r2"],
            "n_obs": fit["n_obs"],
            "fit_level": fit_level,
        })

    return pd.DataFrame.from_records(records)


def predicted_demand(intercept: float, elasticity: float, price: float, has_event: bool = False, event_coef: float = 0.0) -> float:
    """Predict expected units sold at a given price from a fitted log-log curve."""
    log_units = intercept + elasticity * np.log(price) + event_coef * int(has_event)
    return max(float(np.exp(log_units) - 1), 0.0)


def build_piecewise_curve(
    base_price: float,
    intercept: float,
    elasticity: float,
    discount_tiers: list[float] = None,
) -> list[tuple[float, float]]:
    """
    Build (price, predicted_demand) breakpoints across the discount ladder,
    for use as SOS2 breakpoints in model.py's piecewise-linear demand constraint.

    Returns breakpoints sorted by price ascending.
    """
    discount_tiers = discount_tiers or DEFAULT_DISCOUNT_TIERS
    breakpoints = []
    for discount in discount_tiers:
        price = round(base_price * (1 - discount), 2)
        if price <= 0:
            continue
        demand = predicted_demand(intercept, elasticity, price)
        breakpoints.append((price, demand))

    breakpoints.sort(key=lambda pt: pt[0])
    return breakpoints


def build_all_curves(elasticity_df: pd.DataFrame, discount_tiers: list[float] = None) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Build piecewise curves for every item-store row in a fitted elasticity table."""
    curves = {}
    for _, row in elasticity_df.iterrows():
        key = (row["item_id"], row["store_id"])
        curves[key] = build_piecewise_curve(
            base_price=row["base_price"],
            intercept=row["intercept"],
            elasticity=row["elasticity"],
            discount_tiers=discount_tiers,
        )
    return curves


def main():
    parser = argparse.ArgumentParser(description="Fit price-elasticity curves from the SKU-week panel.")
    parser.add_argument("--panel", default=str(PROCESSED_DIR / "sku_week_panel.csv"))
    parser.add_argument("--out", default=str(PROCESSED_DIR / "elasticities.csv"))
    parser.add_argument("--min-obs", type=int, default=MIN_OBS_FOR_ITEM_STORE_FIT)
    args = parser.parse_args()

    df = load_panel(Path(args.panel))
    elasticity_df = fit_elasticities(df, min_obs=args.min_obs)

    elasticity_df.to_csv(args.out, index=False)

    print(f"Fitted {len(elasticity_df):,} item-store elasticity curves")
    print(f"  item_store-level fits: {(elasticity_df['fit_level'] == 'item_store').sum():,}")
    print(f"  dept_pooled fallbacks: {(elasticity_df['fit_level'] == 'dept_pooled').sum():,}")
    print(f"  median elasticity: {elasticity_df['elasticity'].median():.3f}")
    print(f"  median R²: {elasticity_df['r2'].median():.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()