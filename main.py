"""
main.py
-------
Entry point for the Retail Sales Forecasting & Inventory Optimization System.

Running this file will:
    1. Print the project banner.
    2. Generate the synthetic retail dataset (data/raw/retail_sales.csv).
    3. Run the preprocessing pipeline (data/processed/retail_sales_clean.csv).

Usage:
    python main.py
"""

import sys
import os

PROJECT_NAME = "Retail Sales Forecasting & Inventory Optimization System"


def print_banner() -> None:
    print("=" * 60)
    print(f"  {PROJECT_NAME}")
    print("=" * 60)


def run_data_pipeline() -> None:
    """
    Generate the synthetic dataset and run the preprocessing pipeline.
    Imports are kept inside this function so that merely importing main.py
    never triggers file I/O or computation.
    """
    # Add project root to sys.path so 'src' is importable from any cwd
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data.generate_data import generate_dataset
    from src.data.preprocess    import run_pipeline

    print("\n[Phase 2 - Step 1]  Generating synthetic dataset ...")
    generate_dataset()

    print("\n[Phase 2 - Step 2]  Running preprocessing pipeline ...")
    run_pipeline()

    print("\n" + "=" * 60)
    print("  Phase 2 complete.")
    print("  Raw data   ->  data/raw/retail_sales.csv")
    print("  Clean data ->  data/processed/retail_sales_clean.csv")
    print("=" * 60)


def main() -> None:
    print_banner()
    run_data_pipeline()


if __name__ == "__main__":
    main()
