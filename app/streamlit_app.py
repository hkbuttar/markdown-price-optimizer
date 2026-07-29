"""
streamlit_app.py

Interactive demo for the markdown/clearance pricing MIP. Loads pre-computed
results (outputs/results_summary.json, outputs/sample_markdown_schedule.csv)
rather than re-running the ~19-minute full solve live, so the app is fast and
demoable from a persistent deployed URL.

Run locally:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve paths relative to the repo root regardless of where streamlit is
# launched from, mirroring the same fix used in the notebook.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

st.set_page_config(page_title="Markdown Price Optimizer", page_icon="🏷️", layout="wide")


@st.cache_data
def load_results_summary() -> dict:
    with open(OUTPUTS_DIR / "results_summary.json") as f:
        return json.load(f)


@st.cache_data
def load_schedule() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "sample_markdown_schedule.csv")


def _missing_outputs_message():
    st.error(
        "No saved results found. Run the full solve first:\n\n"
        "`python3 -m src.solve --panel data/processed/model_ready_panel_filtered.csv`\n\n"
        "This generates `outputs/results_summary.json` and `outputs/sample_markdown_schedule.csv`, "
        "which this app reads from."
    )


def main():
    st.title("🏷️ Markdown & Clearance Pricing Optimizer")
    st.caption(
        "A Mixed-Integer Programming model that recommends weekly discount tiers per SKU to maximize "
        "recovered revenue before a clearance deadline, built on Walmart's M5 retail dataset."
    )

    if not (OUTPUTS_DIR / "results_summary.json").exists() or not (OUTPUTS_DIR / "sample_markdown_schedule.csv").exists():
        _missing_outputs_message()
        return

    results = load_results_summary()
    schedule = load_schedule()

    # ---------------------------------------------------------------
    # Headline metrics
    # ---------------------------------------------------------------
    st.header("Results Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "MIP-Optimized Revenue",
        f"${results['mip_total_revenue']:,.0f}",
    )
    col2.metric(
        "vs. Flat 20% Discount",
        f"${results['flat_discount_baseline_revenue']:,.0f}",
        delta=f"{results.get('pct_improvement_vs_flat', 0):+.1f}%",
    )
    col3.metric(
        "vs. Calendar-Based Markdown",
        f"${results['calendar_baseline_revenue']:,.0f}",
        delta=f"{results.get('pct_improvement_vs_calendar', 0):+.1f}%",
    )

    st.caption(
        f"Solved in {results['solve_time_seconds']/60:.1f} minutes across all departments "
        f"(all feasible: {results['all_departments_feasible']}). CPU only, no GPU required."
    )

    # ---------------------------------------------------------------
    # Revenue comparison chart
    # ---------------------------------------------------------------
    st.subheader("Recovered Revenue: MIP vs. Naive Baselines")

    revenue_comparison = pd.DataFrame({
        "Policy": ["Flat 20% Discount", "Calendar-Based Markdown", "MIP-Optimized (this project)"],
        "Recovered Revenue": [
            results["flat_discount_baseline_revenue"],
            results["calendar_baseline_revenue"],
            results["mip_total_revenue"],
        ],
    }).set_index("Policy")

    st.bar_chart(revenue_comparison)

    # ---------------------------------------------------------------
    # Per-department breakdown
    # ---------------------------------------------------------------
    st.subheader("Recovered Revenue by Department")

    dept_status = results["per_department_status"]
    dept_df = pd.DataFrame([
        {
            "Department": dept,
            "Recovered Revenue": v["objective_value"],
            "Rolling-Horizon Windows Solved": v["n_windows_solved"],
            "Feasible": "✅" if v["feasible"] else "❌",
        }
        for dept, v in dept_status.items()
    ]).sort_values("Recovered Revenue", ascending=False)

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(dept_df.set_index("Department")["Recovered Revenue"])
    with col_table:
        st.dataframe(dept_df.set_index("Department"), use_container_width=True)

    # ---------------------------------------------------------------
    # Interactive SKU schedule explorer
    # ---------------------------------------------------------------
    st.header("Explore a SKU's Optimized Markdown Schedule")

    sku_summary = (
        schedule.groupby(["item_id", "store_id"])
        .agg(n_tiers=("tier", "nunique"), max_tier=("tier", "max"), n_weeks=("week", "count"))
        .reset_index()
    )
    sku_summary["label"] = sku_summary["item_id"] + " @ " + sku_summary["store_id"]

    col_filters, col_dept = st.columns(2)
    with col_filters:
        min_tiers = st.slider(
            "Minimum distinct discount tiers used",
            min_value=1,
            max_value=int(sku_summary["n_tiers"].max()),
            value=1,
            help="Filter to SKUs whose optimized schedule uses at least this many different discount levels.",
        )
    with col_dept:
        dept_filter = st.selectbox(
            "Filter by department",
            options=["All"] + sorted(schedule["dept_id"].unique().tolist()) if "dept_id" in schedule.columns else ["All"],
        )

    filtered_summary = sku_summary[sku_summary["n_tiers"] >= min_tiers]
    if dept_filter != "All" and "dept_id" in schedule.columns:
        valid_keys = schedule[schedule["dept_id"] == dept_filter][["item_id", "store_id"]].drop_duplicates()
        filtered_summary = filtered_summary.merge(valid_keys, on=["item_id", "store_id"])

    if filtered_summary.empty:
        st.warning("No SKUs match the current filters. Try lowering the minimum tier count.")
        return

    selected_label = st.selectbox(
        f"Choose a SKU ({len(filtered_summary):,} match the current filters)",
        options=filtered_summary.sort_values("n_tiers", ascending=False)["label"].tolist(),
    )

    selected_item, selected_store = selected_label.split(" @ ")
    sku_schedule = schedule[
        (schedule["item_id"] == selected_item) & (schedule["store_id"] == selected_store)
    ].sort_values("week").reset_index(drop=True)
    sku_schedule["week_number"] = range(1, len(sku_schedule) + 1)

    st.line_chart(sku_schedule.set_index("week_number")[["price"]])
    st.bar_chart(sku_schedule.set_index("week_number")[["units_sold"]])

    with st.expander("View raw schedule data"):
        st.dataframe(sku_schedule[["week_number", "tier", "price", "units_sold"]], use_container_width=True)

    # ---------------------------------------------------------------
    # Methodology footer
    # ---------------------------------------------------------------
    st.divider()
    st.caption(
        "Methodology: Mixed-Integer Program with piecewise-linear (tier-disaggregated) demand response, "
        "solved via hybrid decomposition (department clustering + SKU batching + rolling horizon) using "
        "Pyomo and the open-source CBC solver. Data: Walmart M5 Forecasting dataset. "
        "Full details in `notebooks/final_project.ipynb` and the project README."
    )


if __name__ == "__main__":
    main()