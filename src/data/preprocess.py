# -*- coding: utf-8 -*-
"""
src/data/preprocess.py
----------------------
Data loading, validation, cleaning, and saving for the retail sales dataset.

Design decisions documented here so the logic is easy to follow:

1.  DATE PARSING
        The 'date' column is stored as "YYYY-MM-DD" strings in the CSV.
        We parse it with pandas and keep it as datetime64 so time-series
        operations work correctly downstream.

2.  MISSING VALUES
        Structural columns (date, store_id, product_id, category) — rows
        with missing values in these are dropped; they cannot be imputed.
        Numeric columns — missing values are flagged and reported. We do NOT
        fill units_sold / stock_on_hand with forward-fill or median because
        silently inventing demand would distort forecasting.

3.  DUPLICATE RECORDS
        A store-product-date combination should be unique. If duplicates exist
        we keep the first occurrence and log the count. This is conservative:
        we never sum them (which would double-count sales) or average them
        (which would hide data entry errors).

4.  STOCKOUT ROWS
        Rows where stockout_flag == 1 are KEPT in the cleaned dataset.
        Removing them would hide lost-sales events. The stockout_flag column
        lets downstream models distinguish "zero demand" from
        "could not sell — was out of stock". Observed sales during stockouts
        likely UNDERESTIMATE true demand.

5.  NEGATIVE VALUES
        Negative units_sold or stock_on_hand are data errors. We report them
        and clip to 0, consistent with how the generator enforces non-negativity.
"""

import os
import sys
import pandas as pd

# ── Column specification ───────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "category",
    "units_sold",
    "unit_price",
    "promotion_flag",
    "discount_pct",
    "stock_on_hand",
    "stockout_flag",
    "supplier_lead_time_days",
    "unit_cost",
]

NUMERIC_COLUMNS = [
    "units_sold",
    "unit_price",
    "promotion_flag",
    "discount_pct",
    "stock_on_hand",
    "stockout_flag",
    "supplier_lead_time_days",
    "unit_cost",
]

NON_NEGATIVE_COLUMNS = ["units_sold", "stock_on_hand"]

KEY_COLUMNS = ["store_id", "product_id", "date"]

# Default paths (relative to project root — resolved at runtime)
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RAW_PATH       = os.path.join(_ROOT, "data", "raw",       "retail_sales.csv")
DEFAULT_PROCESSED_PATH = os.path.join(_ROOT, "data", "processed", "retail_sales_clean.csv")


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(filepath: str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw retail sales CSV and parse dates.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            "Run 'python main.py' first to generate the dataset."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Check that all required columns are present.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def quality_report(df: pd.DataFrame) -> dict:
    """
    Analyse the DataFrame and return a quality summary dictionary.

    Checks performed
    ----------------
    - Total rows
    - Missing values per column
    - Duplicate store-product-date records
    - Invalid negative values in non-negative columns

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict  with keys:
        total_rows, missing_values, duplicate_keys, negative_counts
    """
    # Missing values
    missing = df[REQUIRED_COLUMNS].isnull().sum()
    missing = missing[missing > 0].to_dict()

    # Duplicate keys
    dup_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)
    n_dup    = dup_mask.sum()

    # Negative values in columns that must be >= 0
    neg_counts = {}
    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                neg_counts[col] = int(n_neg)

    return {
        "total_rows":     len(df),
        "missing_values": missing,
        "duplicate_keys": int(n_dup),
        "negative_counts": neg_counts,
    }


def print_quality_report(report: dict, label: str = "RAW") -> None:
    """Pretty-print a quality report dict."""
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  DATA QUALITY REPORT  [{label}]")
    print(sep)
    print(f"  Total rows          : {report['total_rows']:,}")

    if report["missing_values"]:
        print(f"  Missing values      :")
        for col, n in report["missing_values"].items():
            print(f"      {col:<30} {n:>6}")
    else:
        print(f"  Missing values      : None")

    print(f"  Duplicate keys      : {report['duplicate_keys']:,}")

    if report["negative_counts"]:
        print(f"  Negative values     :")
        for col, n in report["negative_counts"].items():
            print(f"      {col:<30} {n:>6}")
    else:
        print(f"  Negative values     : None")

    print(f"{sep}\n")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return the cleaned DataFrame.

    Steps (in order)
    ----------------
    1. Validate columns.
    2. Ensure 'date' is datetime.
    3. Drop rows missing structural key columns.
    4. Report and clip negative values to 0.
    5. Remove duplicate store-product-date rows (keep first).
    6. Reset index.

    Stockout rows are intentionally preserved — see module docstring.

    Parameters
    ----------
    df : pd.DataFrame   Raw dataset.

    Returns
    -------
    pd.DataFrame        Cleaned dataset.
    """
    validate_columns(df)

    df = df.copy()

    # Step 2 — ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Step 3 — drop rows with missing structural keys
    before = len(df)
    df.dropna(subset=["date", "store_id", "product_id", "category"], inplace=True)
    dropped_struct = before - len(df)
    if dropped_struct:
        print(f"[preprocess] Dropped {dropped_struct} rows with missing key columns.")

    # Step 4 — clip negatives
    for col in NON_NEGATIVE_COLUMNS:
        n_neg = (df[col] < 0).sum()
        if n_neg:
            print(f"[preprocess] Clipped {n_neg} negative value(s) in '{col}' to 0.")
            df[col] = df[col].clip(lower=0)

    # Step 5 — remove duplicate keys, keep first
    n_before = len(df)
    df.drop_duplicates(subset=KEY_COLUMNS, keep="first", inplace=True)
    n_dup_removed = n_before - len(df)
    if n_dup_removed:
        print(
            f"[preprocess] Removed {n_dup_removed} duplicate store-product-date "
            "record(s) (kept first occurrence)."
        )

    # Step 6 — reset index
    df.reset_index(drop=True, inplace=True)

    return df


def save_data(df: pd.DataFrame, filepath: str = DEFAULT_PROCESSED_PATH) -> None:
    """
    Save the cleaned DataFrame to a CSV file.

    Parameters
    ----------
    df       : pd.DataFrame
    filepath : str   Destination path.
    """
    out = os.path.normpath(filepath)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[preprocess] Cleaned data saved to: {out}")
    print(f"[preprocess] Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")


def run_pipeline(
    raw_path:       str = DEFAULT_RAW_PATH,
    processed_path: str = DEFAULT_PROCESSED_PATH,
) -> pd.DataFrame:
    """
    Run the full load → report → clean → report → save pipeline.

    Parameters
    ----------
    raw_path       : str   Path to raw CSV.
    processed_path : str   Destination for cleaned CSV.

    Returns
    -------
    pd.DataFrame   The cleaned dataset.
    """
    print("\n[preprocess] Loading raw data ...")
    df_raw = load_data(raw_path)

    # Report on raw data
    raw_report = quality_report(df_raw)
    print_quality_report(raw_report, label="RAW")

    # Clean
    print("[preprocess] Cleaning data ...")
    df_clean = clean_data(df_raw)

    # Report on cleaned data
    clean_report = quality_report(df_clean)
    print_quality_report(clean_report, label="CLEANED")

    # Stockout notice
    n_stockouts = df_clean["stockout_flag"].sum()
    print(
        f"[preprocess] NOTE: {n_stockouts:,} stockout day(s) are retained in the\n"
        "             cleaned dataset. Sales recorded on stockout days are likely\n"
        "             ZERO because inventory was unavailable, not because demand\n"
        "             was zero. The 'stockout_flag' column identifies these rows.\n"
    )

    # Save
    save_data(df_clean, processed_path)

    return df_clean


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
