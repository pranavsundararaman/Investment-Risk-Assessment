from src.data_collection import (
    load_stock_list,
    add_yfinance_tickers,
    update_stock_prices,
)
from src.config import NIFTY50_CSV
from src.prediction_service import generate_latest_predictions


def main():
    print("=" * 60)
    print("Starting daily market update")
    print("=" * 60)

    # Load the NIFTY 50 stock universe.
    stocks = load_stock_list(NIFTY50_CSV)

    # Add Yahoo Finance ticker symbols, e.g. TCS -> TCS.NS.
    stocks = add_yfinance_tickers(stocks)

    print(f"Loaded {len(stocks)} stocks.")

    # Update only data missing from PostgreSQL.
    rows_upserted = update_stock_prices(stocks)

    print(f"Rows upserted: {rows_upserted}")

    print("=" * 60)
    print("Generating latest predictions")
    print("=" * 60)

    predictions = generate_latest_predictions()

    print(
        predictions[
            [
                "ticker",
                "predicted_volatility",
                "risk",
                "risk_rank",
            ]
        ].to_string(index=False)
    )

    print("=" * 60)
    print("Daily update completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()