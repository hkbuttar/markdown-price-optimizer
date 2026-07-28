"""
model.py

Formulates and solves the markdown/clearance pricing MIP.

Decision structure: for each SKU (item_id, store_id) and week, exactly one
discount tier is selected from a discrete price ladder. Because the ladder is
already discrete, each tier's demand is a known constant looked up from the
SKU's fitted price-response curve (elasticity.py) -- this is a tier-disaggregated
linear MIP: revenue stays linear because "units sold at tier t" is its own
continuous variable, gated to zero unless tier t's binary is selected. No
bilinear price * units_sold term ever appears in the model.

Usage (programmatic):
    from src.model import build_model, solve_model
    m = build_model(panel_df)
    result = solve_model(m)
    result.schedule  # DataFrame: item_id, store_id, week, tier, price, units_sold
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.elasticity import build_piecewise_curve, DEFAULT_DISCOUNT_TIERS

DEFAULT_CLEARANCE_FRACTION = 0.95
DEFAULT_MIN_MARGIN = 0.0
DEFAULT_SALVAGE_PENALTY = 1.0
DEFAULT_ELASTICITY = -1.0
DEFAULT_BASE_DEMAND = 20.0


@dataclass
class SolveResult:
    feasible: bool
    objective_value: float | None
    schedule: pd.DataFrame


def _default_elasticity_params(base_price: float) -> tuple[float, float]:
    """
    Fallback elasticity parameters when a panel row has no fitted intercept/elasticity
    (e.g. quick tests, or a SKU with insufficient data even for the dept_pooled fallback).
    Calibrated so predicted demand at base_price equals DEFAULT_BASE_DEMAND.
    """
    elasticity = DEFAULT_ELASTICITY
    intercept = np.log(DEFAULT_BASE_DEMAND + 1) - elasticity * np.log(base_price)
    return intercept, elasticity


def _tier_data_for_row(row: pd.Series, discount_tiers: list[float]) -> list[tuple[int, float, float]]:
    """
    Build (tier_index, price, demand) triples for one SKU-week, ordered so that
    tier_index 0 = highest price (no discount) and tier_index increases as
    discount depth increases -- this ordering is what the monotonic-pricing
    constraint relies on.
    """
    base_price = float(row["base_price"])

    intercept = row.get("intercept")
    elasticity = row.get("elasticity")
    if pd.isna(intercept) or pd.isna(elasticity):
        intercept, elasticity = _default_elasticity_params(base_price)
    else:
        intercept, elasticity = float(intercept), float(elasticity)

    curve = build_piecewise_curve(base_price, intercept, elasticity, discount_tiers)
    curve_desc = list(reversed(curve))  # ascending price -> descending price
    return [(idx, price, demand) for idx, (price, demand) in enumerate(curve_desc)]


def build_model(
    panel_df: pd.DataFrame,
    discount_tiers: list[float] = None,
    clearance_fraction: float = DEFAULT_CLEARANCE_FRACTION,
    min_margin: float = DEFAULT_MIN_MARGIN,
    salvage_penalty_per_unit: float = DEFAULT_SALVAGE_PENALTY,
    allow_emergency_final_week: bool = True,
) -> pyo.ConcreteModel:
    """
    Build the markdown MIP for a set of SKUs over their shared clearance horizon.

    Expected panel_df columns:
        item_id, store_id, week, starting_inventory (non-null on the first week
        per SKU), unit_cost, base_price, and optionally intercept, elasticity
        (fitted curves from elasticity.py -- falls back to a default demand
        curve if absent, e.g. for quick tests).
    """
    discount_tiers = discount_tiers or DEFAULT_DISCOUNT_TIERS
    n_tiers = len(discount_tiers)

    df = panel_df.copy().sort_values(["item_id", "store_id", "week"]).reset_index(drop=True)
    sku_keys = list(df[["item_id", "store_id"]].drop_duplicates().itertuples(index=False, name=None))

    starting_inventory: dict[tuple, float] = {}
    unit_cost: dict[tuple, float] = {}
    weeks_by_sku: dict[tuple, list] = {}
    tier_data: dict[tuple, list[tuple[int, float, float]]] = {}  # (item_id, store_id, week) -> triples

    for item_id, store_id in sku_keys:
        sku = (item_id, store_id)
        sku_rows = df[(df["item_id"] == item_id) & (df["store_id"] == store_id)].sort_values("week")

        inv_series = sku_rows["starting_inventory"].dropna()
        if inv_series.empty:
            raise ValueError(f"No starting_inventory provided for SKU {sku}")
        starting_inventory[sku] = float(inv_series.iloc[0])
        unit_cost[sku] = float(sku_rows["unit_cost"].iloc[0])
        weeks_by_sku[sku] = list(sku_rows["week"])

        for _, row in sku_rows.iterrows():
            tier_data[(item_id, store_id, row["week"])] = _tier_data_for_row(row, discount_tiers)

    sku_week_keys = [(i, s, w) for (i, s) in sku_keys for w in weeks_by_sku[(i, s)]]
    sku_week_tier_keys = [(*sw, t) for sw in sku_week_keys for t in range(n_tiers)]

    model = pyo.ConcreteModel()

    model.SKU_WEEKS = pyo.Set(initialize=sku_week_keys, dimen=3)
    model.SKU_WEEK_TIERS = pyo.Set(initialize=sku_week_tier_keys, dimen=4)

    model.y = pyo.Var(model.SKU_WEEK_TIERS, domain=pyo.Binary)
    model.units_sold_tier = pyo.Var(model.SKU_WEEK_TIERS, domain=pyo.NonNegativeReals)
    model.units_sold = pyo.Var(model.SKU_WEEKS, domain=pyo.NonNegativeReals)
    model.inv = pyo.Var(model.SKU_WEEKS, domain=pyo.NonNegativeReals)

    model.constraints = pyo.ConstraintList()

    # margin floor: zero out tiers priced below cost + min_margin, except the
    # final week if emergency clearance pricing is allowed there
    for item_id, store_id in sku_keys:
        sku = (item_id, store_id)
        weeks = weeks_by_sku[sku]
        final_week = weeks[-1]
        floor_price = unit_cost[sku] * (1 + min_margin)

        for w in weeks:
            is_final = (w == final_week) and allow_emergency_final_week
            for t, price, demand in tier_data[(item_id, store_id, w)]:
                if not is_final and price < floor_price - 1e-9:
                    model.y[item_id, store_id, w, t].fix(0)

    for item_id, store_id in sku_keys:
        sku = (item_id, store_id)
        weeks = weeks_by_sku[sku]

        for w in weeks:
            # exactly one tier chosen per SKU-week
            model.constraints.add(
                sum(model.y[item_id, store_id, w, t] for t in range(n_tiers)) == 1
            )

            # units sold "at" each tier gated to zero unless that tier is chosen,
            # capped at that tier's predicted demand
            for t, price, demand in tier_data[(item_id, store_id, w)]:
                model.constraints.add(
                    model.units_sold_tier[item_id, store_id, w, t]
                    <= demand * model.y[item_id, store_id, w, t]
                )

            # total units sold this week = sum across tiers (only the chosen tier is nonzero)
            model.constraints.add(
                model.units_sold[item_id, store_id, w]
                == sum(model.units_sold_tier[item_id, store_id, w, t] for t in range(n_tiers))
            )

            # can't sell more than what's in stock this week
            model.constraints.add(
                model.units_sold[item_id, store_id, w] <= model.inv[item_id, store_id, w]
            )

        # inventory rollforward
        for i, w in enumerate(weeks):
            if i == 0:
                model.constraints.add(model.inv[item_id, store_id, w] == starting_inventory[sku])
            else:
                prev_w = weeks[i - 1]
                model.constraints.add(
                    model.inv[item_id, store_id, w]
                    == model.inv[item_id, store_id, prev_w] - model.units_sold[item_id, store_id, prev_w]
                )

        # monotonic pricing: discount depth (tier index) cannot decrease week over week
        for i in range(1, len(weeks)):
            w_prev, w_curr = weeks[i - 1], weeks[i]
            tier_index_prev = sum(t * model.y[item_id, store_id, w_prev, t] for t in range(n_tiers))
            tier_index_curr = sum(t * model.y[item_id, store_id, w_curr, t] for t in range(n_tiers))
            model.constraints.add(tier_index_curr >= tier_index_prev)

        # end-of-horizon clearance requirement (with a redundant total-inventory safety cap)
        total_sold = sum(model.units_sold[item_id, store_id, w] for w in weeks)
        model.constraints.add(total_sold >= clearance_fraction * starting_inventory[sku])
        model.constraints.add(total_sold <= starting_inventory[sku])

    # objective: maximize revenue, minus a salvage penalty on any unsold leftover inventory
    revenue_terms = []
    leftover_terms = []
    for item_id, store_id in sku_keys:
        sku = (item_id, store_id)
        weeks = weeks_by_sku[sku]
        for w in weeks:
            for t, price, demand in tier_data[(item_id, store_id, w)]:
                revenue_terms.append(price * model.units_sold_tier[item_id, store_id, w, t])
        total_sold = sum(model.units_sold[item_id, store_id, w] for w in weeks)
        leftover_terms.append(starting_inventory[sku] - total_sold)

    model.objective = pyo.Objective(
        expr=sum(revenue_terms) - salvage_penalty_per_unit * sum(leftover_terms),
        sense=pyo.maximize,
    )

    # stash bookkeeping needed by solve_model() to reconstruct the schedule
    model._sku_keys = sku_keys
    model._weeks_by_sku = weeks_by_sku
    model._tier_data = tier_data

    return model


def solve_model(model: pyo.ConcreteModel, solver_name: str = "cbc", tee: bool = False) -> SolveResult:
    """Solve the MIP and extract a flat schedule DataFrame (item_id, store_id, week, tier, price, units_sold)."""
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(
            f"Solver '{solver_name}' not found. Install it (e.g. `brew install cbc` on macOS, "
            f"`sudo apt-get install coinor-cbc` on Ubuntu) or pass a different solver_name."
        )

    results = solver.solve(model, tee=tee)
    term_condition = results.solver.termination_condition

    feasible = term_condition in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )

    if not feasible:
        return SolveResult(feasible=False, objective_value=None, schedule=pd.DataFrame())

    rows = []
    for item_id, store_id in model._sku_keys:
        for w in model._weeks_by_sku[(item_id, store_id)]:
            for t, price, demand in model._tier_data[(item_id, store_id, w)]:
                y_val = pyo.value(model.y[item_id, store_id, w, t])
                if y_val is not None and y_val > 0.5:
                    rows.append({
                        "item_id": item_id,
                        "store_id": store_id,
                        "week": w,
                        "tier": t,
                        "price": price,
                        "units_sold": pyo.value(model.units_sold[item_id, store_id, w]),
                    })

    schedule = pd.DataFrame(rows).sort_values(["item_id", "store_id", "week"]).reset_index(drop=True)
    objective_value = pyo.value(model.objective)

    return SolveResult(feasible=True, objective_value=objective_value, schedule=schedule)