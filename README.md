# Retail Sales Forecasting & Inventory Optimization System

A student portfolio project that demonstrates end-to-end data science skills:
time-series sales forecasting combined with data-driven inventory management.

---

## Project Goal

Retail businesses often struggle with two connected problems:
1. **Forecasting** — predicting future sales accurately enough to plan ahead.
2. **Inventory** — deciding how much stock to hold, when to reorder, and how to avoid both stockouts and overstock.

This system addresses both by building a forecasting pipeline and an inventory
optimization layer on top of it, then presenting the results through an
interactive Streamlit dashboard.

---

## Planned Modules

| Module | Location | Responsibility |
|---|---|---|
| Data ingestion & preprocessing | `src/data/` | Load raw CSVs, clean, engineer features |
| Sales forecasting | `src/forecasting/` | Train and evaluate time-series models |
| Inventory optimization | `src/inventory/` | Safety stock, reorder point, EOQ calculations |
| Visualization & reporting | `src/visualization/` | Charts, metric tables, CSV exports |
| Streamlit dashboard | `app/` | Interactive web UI for all outputs |

---

## Project Structure

```
├── app/                  # Streamlit dashboard
├── data/
│   ├── raw/              # Original, unmodified data files
│   └── processed/        # Cleaned and feature-engineered data
├── docs/                 # Project documentation
├── images/               # Screenshots and diagrams
├── models/               # Saved trained models
├── notebooks/            # Exploratory Jupyter notebooks
├── outputs/
│   ├── figures/          # Saved charts and plots
│   └── reports/          # Generated CSV / HTML reports
├── src/
│   ├── data/             # Data loading and preprocessing code
│   ├── forecasting/      # Forecasting model code
│   ├── inventory/        # Inventory optimization code
│   └── visualization/    # Plotting and export utilities
├── tests/                # Automated pytest test suite
├── main.py               # Project entry point
└── requirements.txt      # Python dependencies
```

---

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Retail-Sales-Forecasting-Inventory-Optimization
   ```

2. **Create and activate a virtual environment** (Windows)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup**
   ```bash
   python main.py
   ```

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project scaffold & configuration | ✅ Complete |
| 2 | Dataset generation & preprocessing | ✅ Complete |
| 3 | Sales forecasting models | 🔲 Planned |
| 4 | Inventory optimization | 🔲 Planned |
| 5 | Streamlit dashboard | 🔲 Planned |
| 6 | Tests & documentation | 🔲 Planned |

---

## Phase 2 — Synthetic Dataset & Preprocessing

### Synthetic Dataset

> **All data is simulated.** No real stores, products, or sales figures
> are represented. The dataset exists solely for algorithm development
> and educational portfolio demonstration.

The generator (`src/data/generate_data.py`) creates a reproducible CSV with:

| Property | Value |
|---|---|
| Date range | 2022-01-01 — 2023-12-31 (730 days) |
| Stores | 5  (`S01` – `S05`) |
| Products | 20  (`P001` – `P020`) |
| Categories | Electronics, Clothing, Food & Beverage, Home & Garden, Sports |
| Total rows | 73,000 |
| Random seed | 42 (fully reproducible) |

Demand realism features: weekday effects, annual seasonality, gradual upward
trend, random promotion boosts (~15 % of days), and Gaussian noise.

**Key columns:**

| Column | Description |
|---|---|
| `date` | Calendar date |
| `store_id` | Store identifier |
| `product_id` | Product identifier |
| `category` | Product category |
| `units_sold` | Units sold that day (0 if stocked out) |
| `unit_price` | Selling price per unit |
| `promotion_flag` | 1 if a promotion was active |
| `discount_pct` | Discount percentage (0 if no promo) |
| `stock_on_hand` | Units available at start of day |
| `stockout_flag` | 1 if store had zero inventory |
| `supplier_lead_time_days` | Days from order to delivery |
| `unit_cost` | Supplier cost per unit |

> **Stockout note:** Rows where `stockout_flag == 1` are **kept** in the
> cleaned dataset. Sales on those days are observed as zero because
> inventory was unavailable — they likely **underestimate true demand**.
> The flag lets forecasting models handle this correctly.

### Commands

```bash
# Generate dataset + run preprocessing pipeline
python main.py

# Run data tests only
pytest tests/test_data.py -v
```

### Output files

| File | Description |
|---|---|
| `data/raw/retail_sales.csv` | Original generated dataset |
| `data/processed/retail_sales_clean.csv` | Validated and cleaned dataset |

---

## Technologies

- **Python 3.10+**
- pandas, NumPy — data manipulation
- scikit-learn — machine learning models
- Matplotlib, Seaborn — visualization
- Streamlit — interactive dashboard
- pytest — automated testing
- joblib — model serialization

---

## Author

Built as a student portfolio project to demonstrate practical data science
and software engineering skills.
