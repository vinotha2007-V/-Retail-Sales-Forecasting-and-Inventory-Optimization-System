# -*- coding: utf-8 -*-
"""
src/data/generate_data.py
-------------------------
Generates a reproducible synthetic daily retail sales dataset.

IMPORTANT — SIMULATION NOTICE:
    All data produced by this module is entirely simulated.
    No real stores, products, prices, or sales figures are represented.
    The dataset is intended solely for algorithm development, testing,
    and educational portfolio demonstration.

Dataset coverage:
    - 2 years of daily records (2022-01-01 to 2023-12-31)
    - 5 stores  x  20 products  =  100 store-product combinations
    - 100 combinations x 730 days = 73,000 rows (before clipping)

Demand realism features:
    - Weekday effect   : weekends sell more for some categories
    - Seasonal pattern : sinusoidal annual cycle
    - Gradual trend    : slight upward drift over 2 years
    - Promotion boost  : ~15 % of days have a promotion (+30–60 % lift)
    - Random noise     : Gaussian noise per record
    - Non-negativity   : all demand and inventory values clipped at 0
"""

import os
import math
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

START_DATE = "2022-01-01"
END_DATE   = "2023-12-31"

NUM_STORES   = 5
NUM_PRODUCTS = 20

STORE_IDS   = [f"S{i:02d}" for i in range(1, NUM_STORES + 1)]
PRODUCT_IDS = [f"P{i:03d}" for i in range(1, NUM_PRODUCTS + 1)]

# Each product belongs to one of five categories (4 products each)
CATEGORIES = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden", "Sports"]
PRODUCT_CATEGORIES = {
    pid: CATEGORIES[(i) // 4]
    for i, pid in enumerate(PRODUCT_IDS)
}

# Base daily unit price per product (varies by category)
BASE_PRICES = {
    "Electronics":    [199.99, 249.99, 89.99, 349.99],
    "Clothing":       [29.99,  49.99,  19.99, 39.99],
    "Food & Beverage":[4.99,   7.99,   2.49,  12.99],
    "Home & Garden":  [34.99,  59.99,  24.99, 79.99],
    "Sports":         [44.99,  89.99,  14.99, 129.99],
}

# Cost is 40–60 % of unit price (simulated supplier cost)
COST_RATIO_RANGE = (0.40, 0.60)

# Base average daily demand (units) per category
BASE_DEMAND = {
    "Electronics":    8,
    "Clothing":       25,
    "Food & Beverage":80,
    "Home & Garden":  15,
    "Sports":         20,
}

# Weekend multiplier per category
WEEKEND_MULTIPLIER = {
    "Electronics":    1.3,
    "Clothing":       1.5,
    "Food & Beverage":1.2,
    "Home & Garden":  1.4,
    "Sports":         1.6,
}

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),   # src/data/
    "..", "..",                   # project root
    "data", "raw", "retail_sales.csv"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seasonal_factor(day_of_year: int) -> float:
    """
    Returns a sinusoidal seasonal multiplier.
    Peak ~ mid-December (day 350), trough ~ mid-June (day 170).
    Range: [0.85, 1.15]
    """
    angle = 2 * math.pi * (day_of_year - 350) / 365
    return 1.0 + 0.15 * math.cos(angle)


def _trend_factor(date: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Linear upward trend: starts at 1.0, ends at 1.10 over the full period.
    """
    total_days = (end - start).days
    elapsed    = (date - start).days
    return 1.0 + 0.10 * (elapsed / total_days)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_dataset(output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Generate and save the synthetic retail sales dataset.

    Parameters
    ----------
    output_path : str
        File path where the CSV will be saved.

    Returns
    -------
    pd.DataFrame
        The full generated dataset.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    dates        = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    start_ts     = dates[0]
    end_ts       = dates[-1]

    records = []

    # Pre-compute per-product unit prices and costs (fixed across time)
    product_prices: dict[str, float] = {}
    product_costs:  dict[str, float] = {}

    for i, pid in enumerate(PRODUCT_IDS):
        category = PRODUCT_CATEGORIES[pid]
        prices_in_cat = BASE_PRICES[category]
        price = prices_in_cat[i % len(prices_in_cat)]
        cost_ratio = rng.uniform(*COST_RATIO_RANGE)
        product_prices[pid] = round(price, 2)
        product_costs[pid]  = round(price * cost_ratio, 2)

    # Supplier lead times: 1–14 days, fixed per store-product pair
    lead_times: dict[tuple, int] = {}
    for sid in STORE_IDS:
        for pid in PRODUCT_IDS:
            lead_times[(sid, pid)] = int(rng.integers(1, 15))

    for date in dates:
        day_of_year = date.day_of_year
        weekday     = date.dayofweek   # 0=Monday … 6=Sunday
        is_weekend  = weekday >= 5

        seasonal = _seasonal_factor(day_of_year)
        trend    = _trend_factor(date, start_ts, end_ts)

        for sid in STORE_IDS:
            # Small store-level multiplier for variety
            store_multiplier = rng.uniform(0.85, 1.15)

            for pid in PRODUCT_IDS:
                category = PRODUCT_CATEGORIES[pid]

                # ── Demand calculation ──────────────────────────────────────
                base         = BASE_DEMAND[category]
                weekend_mult = WEEKEND_MULTIPLIER[category] if is_weekend else 1.0
                promo_flag   = int(rng.random() < 0.15)
                promo_boost  = rng.uniform(1.30, 1.60) if promo_flag else 1.0
                discount_pct = round(rng.uniform(10, 40), 1) if promo_flag else 0.0

                mean_demand = (
                    base
                    * seasonal
                    * trend
                    * weekend_mult
                    * store_multiplier
                    * promo_boost
                )
                noise        = rng.normal(0, max(1.0, mean_demand * 0.15))
                units_sold   = max(0, int(round(mean_demand + noise)))

                # ── Inventory calculation ───────────────────────────────────
                # Starting stock is a multiple of base demand (2–4 weeks supply)
                max_stock = int(base * rng.uniform(14, 28))
                # Daily restock with some variance; stock can deplete.
                # The lower bound 0.0 allows genuine stockout events (~3-5%).
                stock_noise   = rng.normal(0, max_stock * 0.05)
                stock_on_hand = max(0, int(round(max_stock * rng.uniform(0.0, 1.00) + stock_noise)))

                stockout_flag = int(stock_on_hand == 0)

                # If stocked out, observed sales are forced to 0
                # (true demand may be higher — preserved via stockout_flag)
                if stockout_flag:
                    units_sold = 0

                records.append({
                    "date":              date.strftime("%Y-%m-%d"),
                    "store_id":          sid,
                    "product_id":        pid,
                    "category":          category,
                    "units_sold":        units_sold,
                    "unit_price":        product_prices[pid],
                    "promotion_flag":    promo_flag,
                    "discount_pct":      discount_pct,
                    "stock_on_hand":     stock_on_hand,
                    "stockout_flag":     stockout_flag,
                    "supplier_lead_time_days": lead_times[(sid, pid)],
                    "unit_cost":         product_costs[pid],
                })

    df = pd.DataFrame(records)

    # ── Safety clipping (belt-and-suspenders) ──────────────────────────────
    for col in ["units_sold", "stock_on_hand"]:
        df[col] = df[col].clip(lower=0)

    # ── Save ───────────────────────────────────────────────────────────────
    out = os.path.normpath(output_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print(f"[generate_data] Dataset saved to: {out}")
    print(f"[generate_data] Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[generate_data] Dates : {df['date'].min()}  to  {df['date'].max()}")
    print(f"[generate_data] NOTE  : All data is SIMULATED for educational use only.")

    return df


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset()
