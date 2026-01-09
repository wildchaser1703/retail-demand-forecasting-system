"""
Generate realistic retail sales data with seasonality, trends, and promotions.
"""

import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config  # noqa: E402


def generate_store_metadata(num_stores: int, random_state: int = 42) -> pd.DataFrame:
    """Generate store metadata with characteristics."""
    np.random.seed(random_state)

    store_types = ["Supermarket", "Hypermarket", "Convenience"]
    location_types = ["Urban", "Suburban", "Rural"]

    stores = []
    for store_id in range(1, num_stores + 1):
        stores.append(
            {
                "store_id": store_id,
                "store_type": np.random.choice(store_types),
                "location_type": np.random.choice(location_types),
                "store_size": np.random.randint(1000, 10000),  # sq ft
                "opening_year": np.random.randint(2010, 2020),
            }
        )

    return pd.DataFrame(stores)


def generate_product_metadata(
    num_products: int, num_categories: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Generate product metadata with categories."""
    np.random.seed(random_state + 1)

    categories = [
        "Groceries",
        "Beverages",
        "Personal Care",
        "Household",
        "Snacks",
    ][:num_categories]

    products = []
    for product_id in range(1, num_products + 1):
        category = categories[(product_id - 1) % len(categories)]
        products.append(
            {
                "product_id": product_id,
                "category": category,
                "price_tier": np.random.choice(["Budget", "Mid-range", "Premium"]),
                "base_price": np.random.uniform(1.0, 50.0),
                "seasonality_strength": np.random.uniform(0.1, 0.5),
            }
        )

    return pd.DataFrame(products)


def generate_sales_data(
    stores_df: pd.DataFrame,
    products_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic sales data with multiple patterns."""
    np.random.seed(random_state + 2)

    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    num_days = len(dates)

    # Create all combinations
    all_data = []

    for _, store in stores_df.iterrows():
        store_id = store["store_id"]
        store_multiplier = store["store_size"] / 5000  # Normalize around 1.0

        for _, product in products_df.iterrows():
            product_id = product["product_id"]
            base_sales = np.random.uniform(10, 100) * store_multiplier

            # Generate time series components
            sales_series = generate_time_series(
                num_days=num_days,
                base_value=base_sales,
                trend_strength=config.data.trend_strength,
                seasonality_strength=product["seasonality_strength"],
                noise_level=config.data.noise_level,
                random_state=random_state + store_id + product_id,
            )

            # Add promotions
            promo_mask = np.random.random(num_days) < config.data.promo_probability
            promo_impact = np.random.uniform(
                config.data.promo_impact_range[0],
                config.data.promo_impact_range[1],
                num_days,
            )
            sales_series = np.where(promo_mask, sales_series * promo_impact, sales_series)

            # Add holiday effects
            sales_series = add_holiday_effects(sales_series, dates)

            # Create dataframe for this store-product combination
            for i, date in enumerate(dates):
                all_data.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "product_id": product_id,
                        "sales": max(0, int(sales_series[i])),  # Ensure non-negative
                        "is_promo": promo_mask[i],
                    }
                )

    return pd.DataFrame(all_data)


def generate_time_series(
    num_days: int,
    base_value: float,
    trend_strength: float = 0.1,
    seasonality_strength: float = 0.3,
    noise_level: float = 0.15,
    random_state: int = 42,
) -> np.ndarray:
    """Generate time series with trend, seasonality, and noise."""
    np.random.seed(random_state)

    # Trend component
    trend = np.linspace(0, trend_strength * base_value, num_days)

    # Weekly seasonality (7-day cycle)
    weekly_pattern = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.2, 1.3])  # Mon-Sun
    weekly_seasonality = np.tile(weekly_pattern, num_days // 7 + 1)[:num_days]
    weekly_seasonality = (weekly_seasonality - 1) * seasonality_strength * base_value

    # Monthly seasonality (approximate 30-day cycle)
    monthly_seasonality = (
        np.sin(2 * np.pi * np.arange(num_days) / 30) * seasonality_strength * base_value * 0.5
    )

    # Yearly seasonality (365-day cycle)
    yearly_seasonality = (
        np.sin(2 * np.pi * np.arange(num_days) / 365) * seasonality_strength * base_value * 0.3
    )

    # Random noise
    noise = np.random.normal(0, noise_level * base_value, num_days)

    # Combine all components
    series = (
        base_value + trend + weekly_seasonality + monthly_seasonality + yearly_seasonality + noise
    )

    return series


def add_holiday_effects(sales_series: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Add sales boost for major holidays."""
    sales_with_holidays = sales_series.copy()

    # Define major retail holidays (US-centric)
    holidays = {
        "New Year": (1, 1),
        "Valentine": (2, 14),
        "Easter": (4, 15),  # Approximate
        "Memorial Day": (5, 27),  # Approximate
        "Independence Day": (7, 4),
        "Labor Day": (9, 2),  # Approximate
        "Halloween": (10, 31),
        "Thanksgiving": (11, 28),  # Approximate
        "Black Friday": (11, 29),  # Approximate
        "Christmas": (12, 25),
    }

    for i, date in enumerate(dates):
        for holiday_name, (month, day) in holidays.items():
            if date.month == month and date.day == day:
                # Boost sales on holiday
                if holiday_name in ["Black Friday", "Christmas"]:
                    sales_with_holidays[i] *= 1.8
                else:
                    sales_with_holidays[i] *= 1.3

                # Boost sales 1-2 days before
                if i > 0:
                    sales_with_holidays[i - 1] *= 1.2
                if i > 1:
                    sales_with_holidays[i - 2] *= 1.1

    return sales_with_holidays


def main():
    """Generate and save retail sales data."""
    parser = argparse.ArgumentParser(description="Generate realistic retail sales data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.paths.raw_data_dir),
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num-stores", type=int, default=config.data.num_stores, help="Number of stores"
    )
    parser.add_argument(
        "--num-products", type=int, default=config.data.num_products, help="Number of products"
    )
    parser.add_argument(
        "--start-date", type=str, default=config.data.start_date, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default=config.data.end_date, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument("--seed", type=int, default=config.data.random_seed, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Retail Sales Data")
    print("=" * 60)
    print(f"Stores: {args.num_stores}")
    print(f"Products: {args.num_products}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Random seed: {args.seed}")
    print()

    # Generate metadata
    print("Generating store metadata...")
    stores_df = generate_store_metadata(args.num_stores, args.seed)

    print("Generating product metadata...")
    products_df = generate_product_metadata(
        args.num_products, config.data.num_categories, args.seed
    )

    # Generate sales data
    print("Generating sales data (this may take a moment)...")
    sales_df = generate_sales_data(
        stores_df, products_df, args.start_date, args.end_date, args.seed
    )

    # Save data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving data to {output_dir}...")
    stores_df.to_csv(output_dir / "stores.csv", index=False)
    products_df.to_csv(output_dir / "products.csv", index=False)
    sales_df.to_csv(output_dir / "sales_data.csv", index=False)

    # Also save as parquet for better performance
    sales_df.to_parquet(output_dir / "sales_data.parquet", index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print("=" * 60)
    print(f"Total records: {len(sales_df):,}")
    print(f"Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"Stores: {sales_df['store_id'].nunique()}")
    print(f"Products: {sales_df['product_id'].nunique()}")
    print(f"Total sales: {sales_df['sales'].sum():,.0f}")
    print(f"Average daily sales per store-product: {sales_df['sales'].mean():.2f}")
    print(f"Promotion rate: {sales_df['is_promo'].mean():.1%}")
    print("\nFiles saved:")
    print(f"  - {output_dir / 'stores.csv'}")
    print(f"  - {output_dir / 'products.csv'}")
    print(f"  - {output_dir / 'sales_data.csv'}")
    print(f"  - {output_dir / 'sales_data.parquet'}")


if __name__ == "__main__":
    main()
