"""
tests/test_data_prep.py

Sanity checks for the data preparation pipeline (src/data_prep.py).
These run against small in-memory fixtures, not the full raw M5 files,
so they're fast enough to run on every push in CI.
"""

import pandas as pd
import pytest

from src.data_prep import melt_sales_chunk, ID_COLS


@pytest.fixture
def tiny_wide_sales():
    """A minimal 2-item, 3-day wide sales frame matching the M5 schema."""
    return pd.DataFrame({
        "item_id": ["FOODS_1_001", "FOODS_1_002"],
        "dept_id": ["FOODS_1", "FOODS_1"],
        "cat_id": ["FOODS", "FOODS"],
        "store_id": ["CA_1", "CA_1"],
        "state_id": ["CA", "CA"],
        "d_1": [3, 0],
        "d_2": [5, 2],
        "d_3": [1, 4],
    })


def test_melt_sales_chunk_row_count(tiny_wide_sales):
    """Melting should produce (n_items x n_days) rows."""
    long_df = melt_sales_chunk(tiny_wide_sales)
    assert len(long_df) == 2 * 3  # 2 items x 3 day columns


def test_melt_sales_chunk_preserves_ids(tiny_wide_sales):
    """All id columns should survive the melt untouched."""
    long_df = melt_sales_chunk(tiny_wide_sales)
    for col in ID_COLS:
        assert col in long_df.columns


def test_melt_sales_chunk_values_correct(tiny_wide_sales):
    """Spot-check that units_sold values map to the right item/day after melting."""
    long_df = melt_sales_chunk(tiny_wide_sales)
    row = long_df[(long_df["item_id"] == "FOODS_1_001") & (long_df["d"] == "d_2")]
    assert row["units_sold"].iloc[0] == 5


def test_melt_sales_chunk_units_sold_dtype(tiny_wide_sales):
    """units_sold should be downcast to int16 for memory efficiency at full scale."""
    long_df = melt_sales_chunk(tiny_wide_sales)
    assert long_df["units_sold"].dtype == "int16"