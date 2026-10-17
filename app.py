
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "retail_sales_clean.csv"
FORECAST_PATH = ROOT / "outputs" / "forecasts" / "test_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "forecast_metrics.json"
INVENTORY_PATH = ROOT / "outputs" / "inventory" / "inventory_recommendations.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"

st.set_page_config(
    page_title="Retail Sales Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Retail Sales Forecasting & Inventory Optimization")
st.caption("Sales analytics • Machine learning forecasts • Inventory recommendations")

if not DATA_PATH.exists():
    st.error("Cleaned dataset not found. Run: python -m src.data.preprocess")
    st.stop()

sales = pd.read_csv(DATA_PATH, parse_dates=["date"])

forecasts = (
    pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    if FORECAST_PATH.exists()
    else pd.DataFrame()
)

inventory = (
    pd.read_csv(INVENTORY_PATH)
    if INVENTORY_PATH.exists()
    else pd.DataFrame()
)

metrics = {}
if METRICS_PATH.exists():
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

st.sidebar.header("Dashboard Filters")

stores = sorted(sales["store_id"].astype(str).unique())
selected_stores = st.sidebar.multiselect(
    "Select stores",
    stores,
    default=stores,
)

products = sorted(sales["product_id"].astype(str).unique())
selected_products = st.sidebar.multiselect(
    "Select products",
    products,
    default=products,
)

filtered = sales[
    sales["store_id"].astype(str).isin(selected_stores)
    & sales["product_id"].astype(str).isin(selected_products)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

total_units = int(filtered["units_sold"].sum())
total_revenue = float(filtered["sales_amount"].sum()) if "sales_amount" in filtered else 0.0
stockout_count = int(filtered["stockout"].sum()) if "stockout" in filtered else 0
store_count = filtered["store_id"].nunique()
product_count = filtered["product_id"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Units Sold", f"{total_units:,}")
c2.metric("Sales Amount", f"{total_revenue:,.2f}")
c3.metric("Stockout Records", f"{stockout_count:,}")
c4.metric("Stores / Products", f"{store_count} / {product_count}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Daily Sales Trend")
    daily = (
        filtered.groupby("date", as_index=False)["units_sold"]
        .sum()
        .sort_values("date")
    )
    st.line_chart(daily.set_index("date")["units_sold"])

with right:
    st.subheader("Sales by Store")
    store_sales = (
        filtered.groupby("store_id")["units_sold"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(store_sales)

st.subheader("Top 10 Products by Units Sold")
top_products = (
    filtered.groupby("product_id")["units_sold"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_products)

st.divider()
st.header("🤖 Forecast Model Performance")

if metrics:
    metric_columns = st.columns(len(metrics))
    for column, (name, value) in zip(metric_columns, metrics.items()):
        if isinstance(value, (int, float)):
            column.metric(name.replace("_", " ").upper(), f"{value:.3f}")
        else:
            column.metric(name.replace("_", " ").upper(), str(value))
else:
    st.info("Forecast metrics file not found. Run: python -m src.forecasting.train_model")

if not forecasts.empty:
    st.subheader("Actual vs Predicted Sales")
    required = {"date", "units_sold", "predicted_units_sold"}
    if required.issubset(forecasts.columns):
        forecast_view = forecasts.copy()
        if "store_id" in forecast_view:
            forecast_view["store_id"] = forecast_view["store_id"].astype(str)
        if "product_id" in forecast_view:
            forecast_view["product_id"] = forecast_view["product_id"].astype(str)

        if "store_id" in forecast_view:
            forecast_view = forecast_view[
                forecast_view["store_id"].isin(selected_stores)
            ]
        if "product_id" in forecast_view:
            forecast_view = forecast_view[
                forecast_view["product_id"].isin(selected_products)
            ]

        if not forecast_view.empty:
            trend = (
                forecast_view.groupby("date")[["units_sold", "predicted_units_sold"]]
                .mean()
                .sort_index()
            )
            st.line_chart(trend)
        else:
            st.info("No forecast records match the current filters.")
    else:
        st.info("Forecast file does not contain the expected columns.")

st.divider()
st.header("📦 Inventory Optimization")

if inventory.empty:
    st.info("Inventory recommendations not found. Run: python -m src.inventory.optimize_inventory")
else:
    if "store_id" in inventory:
        inventory["store_id"] = inventory["store_id"].astype(str)
        inventory = inventory[inventory["store_id"].isin(selected_stores)]
    if "product_id" in inventory:
        inventory["product_id"] = inventory["product_id"].astype(str)
        inventory = inventory[inventory["product_id"].isin(selected_products)]

    if "recommended_order_quantity" in inventory:
        reorder = inventory[inventory["recommended_order_quantity"] > 0]
        a, b = st.columns(2)
        a.metric("Store-product pairs", len(inventory))
        b.metric("Pairs Requiring Reorder", len(reorder))

        st.subheader("Recommended Reorders")
        st.dataframe(
            reorder.sort_values(
                "recommended_order_quantity", ascending=False
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download Inventory Recommendations",
            data=inventory.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations.csv",
            mime="text/csv",
        )

st.divider()
st.header("📈 Generated Analysis Charts")

chart_files = [
    "daily_sales_trend.png",
    "monthly_sales_trend.png",
    "store_sales_comparison.png",
    "top_10_products.png",
    "stockouts_by_store.png",
    "actual_vs_predicted_sales.png",
]

available_charts = [FIGURES_DIR / name for name in chart_files if (FIGURES_DIR / name).exists()]

if available_charts:
    for chart_path in available_charts:
        st.subheader(chart_path.stem.replace("_", " ").title())
        st.image(str(chart_path), use_container_width=True)
else:
    st.info("No saved analysis charts found yet.")

st.caption("Prototype dashboard using the project's generated retail dataset. Validate forecasts and stock assumptions before real business decisions.")
