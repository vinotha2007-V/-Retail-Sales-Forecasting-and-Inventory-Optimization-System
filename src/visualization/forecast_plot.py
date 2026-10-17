
"""Visualize actual and predicted retail sales."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = ROOT / "outputs" / "forecasts" / "test_predictions.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Prediction file not found. Run "
            "'python -m src.forecasting.train_model' first."
        )

    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Overall daily sales comparison across all stores and products.
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            actual_sales=("units_sold", "sum"),
            predicted_sales=("predicted_units_sold", "sum"),
        )
        .sort_values("date")
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        daily["date"], daily["actual_sales"],
        label="Actual sales", linewidth=1.8
    )
    ax.plot(
        daily["date"], daily["predicted_sales"],
        label="Predicted sales", linewidth=1.8
    )
    ax.set_title("Retail Sales Forecast: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = FIGURES_DIR / "actual_vs_predicted_sales.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Prediction records: {len(df):,}")
    print(f"Graph saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
