
"""Train and evaluate a retail sales forecasting model."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "retail_sales_clean.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"

TARGET = "units_sold"
GROUP_COLS = ["store_id", "product_id"]
FEATURES = [
    "store_id",
    "product_id",
    "unit_price",
    "promotion_flag",
    "discount_pct",
    "day_of_week",
    "month",
    "day_of_month",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical sales features without using future sales."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    for col in GROUP_COLS:
        df[col] = (
            df[col]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    grouped = df.groupby(GROUP_COLS, sort=False)[TARGET]

    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Shift before rolling so today's target cannot leak into its features.
    shifted = df.groupby(GROUP_COLS, sort=False)[TARGET].shift(1)
    for window in (7, 14):
        df[f"rolling_mean_{window}"] = (
            shifted.groupby([df[c] for c in GROUP_COLS], sort=False)
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=GROUP_COLS, drop=True)
        )

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Run: python -m src.data.preprocess"
        )

    print("Loading cleaned retail dataset...")
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Loaded {len(raw):,} rows.")

    data = build_features(raw)
    if len(data) < 100:
        raise ValueError("Not enough rows after creating forecasting features.")

    # Global chronological split: never randomly mix future and past.
    data = data.sort_values("date").reset_index(drop=True)
    split_index = int(len(data) * 0.8)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(f"Training rows: {len(train):,}")
    print(f"Testing rows:  {len(test):,}")
    print("Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 0)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    # MAPE is omitted because zero sales can make it misleading.
    metrics = {
        "model": "RandomForestRegressor",
        "target": TARGET,
        "features": FEATURES,
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "test_start_date": str(test["date"].min().date()),
        "test_end_date": str(test["date"].max().date()),
        "mae": mae,
        "rmse": rmse,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "retail_sales_model.joblib"
    joblib.dump({"model": model, "features": FEATURES}, model_path)

    results = test[
        ["date", "store_id", "product_id", TARGET, "stockout_flag"]
    ].copy()
    results["predicted_units_sold"] = predictions
    results.to_csv(FORECAST_DIR / "test_predictions.csv", index=False)

    metrics_path = OUTPUT_DIR / "forecast_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print("\n--- FORECAST MODEL RESULTS ---")
    print(f"MAE:  {mae:.3f} units")
    print(f"RMSE: {rmse:.3f} units")
    print(f"Model saved: {model_path.relative_to(ROOT)}")
    print("Predictions saved: outputs/forecasts/test_predictions.csv")
    print("Metrics saved: outputs/forecast_metrics.json")
    print("Feature importance saved: outputs/feature_importance.csv")


if __name__ == "__main__":
    main()
