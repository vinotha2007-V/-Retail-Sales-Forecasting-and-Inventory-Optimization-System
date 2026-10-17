"""
tests/test_data.py
------------------
Pytest tests for src/data/generate_data.py and src/data/preprocess.py.

Run with:
    pytest tests/test_data.py -v

These tests do NOT write to the real data/ folder — they use tmp_path
fixtures so the test suite is self-contained and repeatable.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

# ── Make 'src' importable regardless of where pytest is invoked from ─────────
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.generate_data import (
    generate_dataset,
    START_DATE,
    END_DATE,
)
from src.data.preprocess import (
    validate_columns,
    quality_report,
    clean_data,
    load_data,
    REQUIRED_COLUMNS,
    KEY_COLUMNS,
    NON_NEGATIVE_COLUMNS,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def generated_df(tmp_path_factory):
    """Generate the dataset once for the whole test module."""
    out_dir = tmp_path_factory.mktemp("raw")
    out_path = str(out_dir / "retail_sales.csv")
    df = generate_dataset(output_path=out_path)
    return df, out_path


@pytest.fixture(scope="module")
def clean_df(generated_df):
    """Return the cleaned version of the generated dataset."""
    df_raw, _ = generated_df
    return clean_data(df_raw)


# ═══════════════════════════════════════════════════════════════════
# Tests: generate_data
# ═══════════════════════════════════════════════════════════════════

class TestGenerateData:

    def test_reproducibility(self, tmp_path):
        """Two calls with the same seed must produce identical DataFrames."""
        path1 = str(tmp_path / "run1.csv")
        path2 = str(tmp_path / "run2.csv")
        df1 = generate_dataset(output_path=path1)
        df2 = generate_dataset(output_path=path2)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))

    def test_required_columns_present(self, generated_df):
        """All required columns must exist in the generated dataset."""
        df, _ = generated_df
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self, generated_df):
        """
        Expected rows = days * stores * products.
        2022-01-01 to 2023-12-31 = 730 days, 5 stores, 20 products.
        """
        df, _ = generated_df
        expected_days = len(pd.date_range(start=START_DATE, end=END_DATE, freq="D"))
        expected_rows = expected_days * 5 * 20
        assert len(df) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(df)}"
        )

    def test_date_range(self, generated_df):
        """Dataset must cover exactly START_DATE to END_DATE."""
        df, _ = generated_df
        df["date"] = pd.to_datetime(df["date"])
        assert str(df["date"].min().date()) == START_DATE
        assert str(df["date"].max().date()) == END_DATE

    def test_dates_parse_as_datetime(self, generated_df):
        """After to_datetime, the date column must be datetime64."""
        df, _ = generated_df
        parsed = pd.to_datetime(df["date"])
        assert pd.api.types.is_datetime64_any_dtype(parsed)

    def test_units_sold_non_negative(self, generated_df):
        """units_sold must be >= 0 everywhere."""
        df, _ = generated_df
        assert (df["units_sold"] >= 0).all(), \
            "Negative values found in units_sold"

    def test_stock_on_hand_non_negative(self, generated_df):
        """stock_on_hand must be >= 0 everywhere."""
        df, _ = generated_df
        assert (df["stock_on_hand"] >= 0).all(), \
            "Negative values found in stock_on_hand"

    def test_unit_price_positive(self, generated_df):
        """unit_price must be strictly positive."""
        df, _ = generated_df
        assert (df["unit_price"] > 0).all(), \
            "Non-positive unit_price found"

    def test_unit_cost_positive(self, generated_df):
        """unit_cost must be strictly positive."""
        df, _ = generated_df
        assert (df["unit_cost"] > 0).all(), \
            "Non-positive unit_cost found"

    def test_promotion_flag_binary(self, generated_df):
        """promotion_flag must only contain 0 or 1."""
        df, _ = generated_df
        assert set(df["promotion_flag"].unique()).issubset({0, 1}), \
            "promotion_flag contains values other than 0/1"

    def test_stockout_flag_binary(self, generated_df):
        """stockout_flag must only contain 0 or 1."""
        df, _ = generated_df
        assert set(df["stockout_flag"].unique()).issubset({0, 1}), \
            "stockout_flag contains values other than 0/1"

    def test_stockout_zero_sales(self, generated_df):
        """On stockout days the units_sold must be 0."""
        df, _ = generated_df
        stockout_rows = df[df["stockout_flag"] == 1]
        assert (stockout_rows["units_sold"] == 0).all(), \
            "Non-zero sales found on stockout days"

    def test_csv_file_created(self, generated_df):
        """The output CSV file must exist after generation."""
        _, out_path = generated_df
        assert os.path.isfile(out_path), f"CSV not found at {out_path}"

    def test_csv_roundtrip(self, generated_df):
        """CSV written to disk must reload to the same shape."""
        df, out_path = generated_df
        df_reloaded = pd.read_csv(out_path)
        assert df_reloaded.shape == df.shape


# ═══════════════════════════════════════════════════════════════════
# Tests: preprocess — validation
# ═══════════════════════════════════════════════════════════════════

class TestValidateColumns:

    def test_valid_dataframe_passes(self, generated_df):
        """validate_columns must not raise for a correctly structured df."""
        df, _ = generated_df
        validate_columns(df)   # should not raise

    def test_missing_column_raises(self, generated_df):
        """validate_columns must raise ValueError when a column is missing."""
        df, _ = generated_df
        df_bad = df.drop(columns=["units_sold"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(df_bad)

    def test_multiple_missing_columns_raises(self, generated_df):
        """validate_columns must report all missing columns at once."""
        df, _ = generated_df
        df_bad = df.drop(columns=["units_sold", "store_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(df_bad)


# ═══════════════════════════════════════════════════════════════════
# Tests: preprocess — clean_data
# ═══════════════════════════════════════════════════════════════════

class TestCleanData:

    def test_unique_store_product_date_after_cleaning(self, clean_df):
        """After cleaning, store-product-date combinations must be unique."""
        dupes = clean_df.duplicated(subset=KEY_COLUMNS, keep=False).sum()
        assert dupes == 0, f"Found {dupes} duplicate key rows after cleaning"

    def test_no_negative_units_sold_after_cleaning(self, clean_df):
        """units_sold must be >= 0 after cleaning."""
        assert (clean_df["units_sold"] >= 0).all()

    def test_no_negative_stock_after_cleaning(self, clean_df):
        """stock_on_hand must be >= 0 after cleaning."""
        assert (clean_df["stock_on_hand"] >= 0).all()

    def test_date_column_is_datetime_after_cleaning(self, clean_df):
        """The date column must be datetime64 after cleaning."""
        assert pd.api.types.is_datetime64_any_dtype(clean_df["date"])

    def test_stockout_rows_preserved(self, clean_df):
        """Stockout rows must NOT be removed during cleaning."""
        assert "stockout_flag" in clean_df.columns
        n_stockouts = clean_df["stockout_flag"].sum()
        assert n_stockouts > 0, \
            "Expected some stockout rows to be preserved, but found none"

    def test_duplicate_removal(self, generated_df):
        """Duplicated rows must be reduced to one after cleaning."""
        df_raw, _ = generated_df
        # Manually inject duplicates
        first_row = df_raw.iloc[[0]]
        df_with_dups = pd.concat([df_raw, first_row, first_row], ignore_index=True)
        df_cleaned = clean_data(df_with_dups)
        dupes = df_cleaned.duplicated(subset=KEY_COLUMNS, keep=False).sum()
        assert dupes == 0

    def test_negative_values_clipped(self, generated_df):
        """Negative units_sold values must be clipped to 0, not dropped."""
        df_raw, _ = generated_df
        df_bad = df_raw.copy()
        # Inject negatives
        df_bad.loc[0, "units_sold"] = -5
        df_bad.loc[1, "stock_on_hand"] = -10
        df_cleaned = clean_data(df_bad)
        assert df_cleaned["units_sold"].iloc[0] == 0
        assert df_cleaned["stock_on_hand"].iloc[1] == 0

    def test_missing_structural_rows_dropped(self, generated_df):
        """Rows with missing date/store_id/product_id must be dropped."""
        df_raw, _ = generated_df
        df_bad = df_raw.copy()
        df_bad.loc[0, "date"] = None
        df_bad.loc[1, "store_id"] = None
        df_cleaned = clean_data(df_bad)
        # Those two rows should be gone
        assert len(df_cleaned) == len(df_raw) - 2

    def test_row_count_preserved_for_clean_input(self, generated_df):
        """
        If the raw data has no issues, clean_data must return the same
        number of rows (no silent data loss).
        """
        df_raw, _ = generated_df
        df_cleaned = clean_data(df_raw)
        assert len(df_cleaned) == len(df_raw)


# ═══════════════════════════════════════════════════════════════════
# Tests: quality_report
# ═══════════════════════════════════════════════════════════════════

class TestQualityReport:

    def test_clean_data_has_no_issues(self, clean_df):
        """quality_report on clean data must show zero problems."""
        report = quality_report(clean_df)
        assert report["missing_values"]  == {}
        assert report["duplicate_keys"]  == 0
        assert report["negative_counts"] == {}

    def test_detects_missing_values(self, generated_df):
        df_raw, _ = generated_df
        df_bad = df_raw.copy()
        df_bad.loc[0, "units_sold"] = None
        report = quality_report(df_bad)
        assert "units_sold" in report["missing_values"]

    def test_detects_duplicates(self, generated_df):
        df_raw, _ = generated_df
        df_bad = pd.concat([df_raw, df_raw.iloc[[0]]], ignore_index=True)
        report = quality_report(df_bad)
        assert report["duplicate_keys"] >= 2   # original + duplicate both flagged

    def test_detects_negative_values(self, generated_df):
        df_raw, _ = generated_df
        df_bad = df_raw.copy()
        df_bad.loc[0, "units_sold"] = -1
        report = quality_report(df_bad)
        assert "units_sold" in report["negative_counts"]


# ═══════════════════════════════════════════════════════════════════
# Tests: load_data
# ═══════════════════════════════════════════════════════════════════

class TestLoadData:

    def test_load_raises_for_missing_file(self, tmp_path):
        """load_data must raise FileNotFoundError for a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "does_not_exist.csv"))

    def test_load_parses_dates(self, generated_df):
        """CSV reloaded via load_data must have datetime date column."""
        _, out_path = generated_df
        df = load_data(out_path)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
