
"""Calculate inventory recommendations using sales forecasts."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "retail_sales_clean.csv"
FORECAST_PATH = ROOT / "outputs" / "forecasts" / "test_predictions.csv"
OUTPUT_DIR = ROOT / "outputs"
INVENTORY_DIR = OUTPUT_DIR / "inventory"

SERVICE_LEVEL = 0.95
ANNUAL_HOLDING_RATE = 0.25
ORDERING_COST = 100.0


def calculate_inventory(
    demand_history: pd.DataFrame,
    forecasts: pd.DataFrame,
    service_level: float = SERVICE_LEVEL,
) -> pd.DataFrame:
    """Return one inventory recommendation per store-product pair."""
    z_score = float(norm.ppf(service_level))

    history = demand_history.copy()
    history["date"] = pd.to_datetime(history["date"])

    forecasts = forecasts.copy()
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    recommendations = []

    for (store_id, product_id), hist_group in history.groupby(
        ["store_id", "product_id"]
    ):
        hist_group = hist_group.sort_values("date")
        forecast_group = forecasts[
            (forecasts["store_id"] == store_id)
            & (forecasts["product_id"] == product_id)
        ].sort_values("date")

        if hist_group.empty or forecast_group.empty:
            continue

        # Estimate typical daily demand from the latest 56 historical days.
        recent = hist_group.tail(56)
        daily_demand = float(recent["units_sold"].mean())
        demand_std = float(recent["units_sold"].std(ddof=1) or 0.0)
        if not np.isfinite(demand_std):
            demand_std = 0.0

        lead_time = max(
            1,
            int(round(hist_group["supplier_lead_time_days"].median())),
        )

        # Use the mean forecast for this pair as its estimated daily demand.
        forecast_demand = float(
            forecast_group["predicted_units_sold"].clip(lower=0).mean()
        )
        expected_daily_demand = max(0.0, forecast_demand)

        # Approximate protection period demand and safety stock.
        lead_time_demand = expected_daily_demand * lead_time
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        reorder_point = lead_time_demand + safety_stock

        # EOQ: annual demand and annual holding cost per unit.
        annual_demand = expected_daily_demand * 365
        unit_cost = float(hist_group["unit_cost"].median())
        holding_cost = unit_cost * ANNUAL_HOLDING_RATE

        if annual_demand > 0 and holding_cost > 0:
            eoq = np.sqrt(
                (2 * annual_demand * ORDERING_COST) / holding_cost
            )
        else:
            eoq = 0.0

        current_stock = float(hist_group.iloc[-1]["stock_on_hand"])
        order_quantity = max(0.0, eoq) if current_stock <= reorder_point else 0.0

        recommendations.append({
            "store_id": store_id,
            "product_id": product_id,
            "recent_average_daily_sales": round(daily_demand, 2),
            "forecast_daily_demand": round(expected_daily_demand, 2),
            "supplier_lead_time_days": lead_time,
            "current_stock": round(current_stock, 2),
            "safety_stock": round(float(safety_stock), 2),
            "reorder_point": round(float(reorder_point), 2),
            "eoq_units": round(float(eoq), 2),
            "recommended_order_quantity": round(float(order_quantity), 2),
            "service_level": service_level,
            "unit_cost": round(unit_cost, 2),
        })

    return pd.DataFrame(recommendations)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {DATA_PATH}\n"
            "Run python -m src.data.preprocess first."
        )

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}\n"
            "Run python -m src.forecasting.train_model first."
        )

    history = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    for frame in (history, forecasts):
        for col in ["store_id", "product_id"]:
            frame[col] = (
                frame[col]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(int)
            )


    result = calculate_inventory(history, forecasts)

    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = INVENTORY_DIR / "inventory_recommendations.csv"
    result.to_csv(output_csv, index=False)

    summary = {
        "store_product_combinations": int(len(result)),
        "items_requiring_reorder": int(
            (result["recommended_order_quantity"] > 0).sum()
        ),
        "service_level": SERVICE_LEVEL,
        "ordering_cost_assumption": ORDERING_COST,
        "annual_holding_rate_assumption": ANNUAL_HOLDING_RATE,
        "note": (
            "Prototype estimates based on historical stock and test-period "
            "forecasts; validate with current inventory data before real use."
        ),
    }
    summary_path = INVENTORY_DIR / "inventory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- INVENTORY OPTIMIZATION RESULTS ---")
    print(f"Store-product pairs: {len(result):,}")
    print(
        "Pairs requiring reorder: "
        f"{summary['items_requiring_reorder']:,}"
    )
    print(f"Recommendations saved: {output_csv.relative_to(ROOT)}")
    print(f"Summary saved: {summary_path.relative_to(ROOT)}")

    if not result.empty:
        print("\nSample recommendations:")
        print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
