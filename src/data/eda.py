# -*- coding: utf-8 -*-
"""Exploratory Data Analysis for retail sales forecasting."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "retail_sales.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "retail_sales_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

sns.set_theme(style="whitegrid")


def load_data():
    """Load the processed dataset, falling back to the raw dataset."""
    if PROCESSED_DATA_PATH.exists():
        path = PROCESSED_DATA_PATH
    elif RAW_DATA_PATH.exists():
        path = RAW_DATA_PATH
    else:
        raise FileNotFoundError(
            "Retail dataset not found. Run python main.py first."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "store_id", "product_id", "units_sold"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return df


def save_plot(filename):
    """Save the current plot and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: outputs/figures/{filename}")


def plot_daily_sales(df):
    daily = df.groupby("date", as_index=False)["units_sold"].sum()

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=daily, x="date", y="units_sold")
    plt.title("Total Daily Retail Sales")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=30)
    save_plot("daily_sales_trend.png")


def plot_monthly_sales(df):
    monthly = (
        df.set_index("date")
        .resample("MS")["units_sold"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x="date", y="units_sold", marker="o")
    plt.title("Monthly Retail Sales")
    plt.xlabel("Month")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=30)
    save_plot("monthly_sales_trend.png")


def plot_store_sales(df):
    store_sales = (
        df.groupby("store_id", as_index=False)["units_sold"]
        .sum()
        .sort_values("units_sold", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=store_sales, x="store_id", y="units_sold")
    plt.title("Total Sales by Store")
    plt.xlabel("Store")
    plt.ylabel("Units Sold")
    save_plot("store_sales_comparison.png")


def plot_top_products(df):
    product_sales = (
        df.groupby("product_id", as_index=False)["units_sold"]
        .sum()
        .nlargest(10, "units_sold")
        .sort_values("units_sold", ascending=True)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=product_sales, y="product_id", x="units_sold")
    plt.title("Top 10 Products by Sales")
    plt.xlabel("Total Units Sold")
    plt.ylabel("Product")
    save_plot("top_10_products.png")


def plot_stockout_analysis(df):
    if "stockout_flag" not in df.columns:
        print("Skipping stockout chart: stockout_flag column not found.")
        return

    stockouts = (
        df.groupby("store_id", as_index=False)["stockout_flag"]
        .sum()
        .sort_values("stockout_flag", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=stockouts, x="store_id", y="stockout_flag")
    plt.title("Stockout Records by Store")
    plt.xlabel("Store")
    plt.ylabel("Stockout Records")
    save_plot("stockouts_by_store.png")


def print_summary(df):
    print("\n--- RETAIL DATA SUMMARY ---")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Stores: {df['store_id'].nunique()}")
    print(f"Products: {df['product_id'].nunique()}")
    print(f"Total units sold: {df['units_sold'].sum():,.0f}")
    print(f"Average units sold per record: {df['units_sold'].mean():.2f}")

    if "stockout_flag" in df.columns:
        count = int(df["stockout_flag"].sum())
        rate = df["stockout_flag"].mean() * 100
        print(f"Stockout records: {count:,} ({rate:.2f}%)")

    print("\nMissing values by column:")
    print(df.isnull().sum().to_string())


def run_eda():
    df = load_data()
    print_summary(df)

    plot_daily_sales(df)
    plot_monthly_sales(df)
    plot_store_sales(df)
    plot_top_products(df)
    plot_stockout_analysis(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    run_eda()