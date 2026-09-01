import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import NIFTY50_CSV
from src.database import (
    create_database_engine,
    fetch_latest_dates_by_ticker,
    insert_stock_data,
)

def load_stock_list(csv_path: Path) -> pd.DataFrame:
    """
    Load the stock list from a CSV file.

    Args:
        csv_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw stock list.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return pd.read_csv(csv_path)

def validate_stock_list(stocks: pd.DataFrame) -> None:
    """
    Validate the stock list DataFrame.

    Args:
        stocks (pd.DataFrame): Stock list.

    Raises:
        ValueError: If required columns are missing.
    """

    required_columns = {
        "Company Name",
        "Industry",
        "Symbol"
    }

    missing = required_columns - set(stocks.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

def add_yfinance_tickers(stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Add Yahoo Finance ticker symbols.

    Args:
        stocks (pd.DataFrame): Validated stock list.

    Returns:
        pd.DataFrame: Updated DataFrame with a Ticker column.
    """
    stocks = stocks.copy()
    stocks["Ticker"] = (
    stocks["Symbol"]
    .astype(str)
    .str.strip()
    + ".NS"
    )
    return stocks

def download_stock_data(
    stock: pd.Series,
    start_date: str,
    end_date: str,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """
    Download historical stock data for a single stock.

    Args:
        stock (pd.Series): A row from the stock list DataFrame.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: Historical stock data with metadata.
    """

    stock_data = yf.download(
    stock["Ticker"],
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False,
    multi_level_index=False,
    )

    if stock_data.empty and allow_empty:
        return stock_data

    if stock_data.empty:
        raise ValueError(
            f"No data found for {stock['Ticker']}"
        )

    stock_data = stock_data.reset_index()

    stock_data["Ticker"] = stock["Ticker"]
    stock_data["Company Name"] = stock["Company Name"]
    stock_data["Industry"] = stock["Industry"]

    return stock_data

def download_all_stocks(
    stocks: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download historical data for all stocks in the stock list.

    Args:
        stocks (pd.DataFrame): Stock list with Yahoo Finance tickers.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: Combined historical stock data.
    """

    all_stock_data = []

    for _, stock in stocks.iterrows():
        print(f"Downloading {stock['Ticker']}...")

        stock_data = download_stock_data(
            stock,
            start_date,
            end_date,
        )

        all_stock_data.append(stock_data)

    master_data = pd.concat(
        all_stock_data,
        ignore_index=True,
    )
    master_data = master_data.sort_values(
    by=["Ticker", "Date"]
    ).reset_index(drop=True)

    return master_data


def update_stock_prices(
    stocks: pd.DataFrame,
    start_date: str = "2015-01-01",
    as_of_date: str | None = None,
) -> int:
    """Download and upsert only dates not yet stored for each ticker."""

    engine = create_database_engine()
    latest_dates = fetch_latest_dates_by_ticker(engine)
    inclusive_end_date = pd.Timestamp(as_of_date or pd.Timestamp.today()).normalize()
    exclusive_end_date = inclusive_end_date + pd.Timedelta(days=1)
    downloaded_data = []

    for _, stock in stocks.iterrows():
        ticker = stock["Ticker"]
        latest_date = latest_dates.get(ticker)
        ticker_start_date = pd.Timestamp(start_date)

        if pd.notna(latest_date):
            ticker_start_date = pd.Timestamp(latest_date) + pd.Timedelta(days=1)

        if ticker_start_date >= exclusive_end_date:
            continue

        print(
            f"Updating {ticker} from {ticker_start_date.date()} "
            f"through {inclusive_end_date.date()}..."
        )
        stock_data = download_stock_data(
            stock,
            start_date=ticker_start_date.strftime("%Y-%m-%d"),
            end_date=exclusive_end_date.strftime("%Y-%m-%d"),
            allow_empty=True,
        )

        if not stock_data.empty:
            downloaded_data.append(stock_data)

    if not downloaded_data:
        print("stock_prices is already current through the latest available trading date.")
        return 0

    updated_data = pd.concat(downloaded_data, ignore_index=True)
    return insert_stock_data(updated_data, engine)


def parse_args() -> argparse.Namespace:
    """Parse arguments for the repeatable PostgreSQL ingestion command."""

    parser = argparse.ArgumentParser(
        description="Incrementally refresh NIFTY 50 prices in PostgreSQL."
    )
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Fallback start date for a ticker not yet in stock_prices (default: 2015-01-01).",
    )
    parser.add_argument(
        "--as-of-date",
        help="Last calendar date to request, in YYYY-MM-DD format (default: today).",
    )
    return parser.parse_args()


def main() -> None:
    """Refresh NIFTY 50 prices through the latest available trading date."""

    args = parse_args()
    stocks = load_stock_list(NIFTY50_CSV)
    validate_stock_list(stocks)
    stocks = add_yfinance_tickers(stocks)
    updated_rows = update_stock_prices(
        stocks,
        start_date=args.start_date,
        as_of_date=args.as_of_date,
    )
    print(f"Completed stock-price refresh: {updated_rows} rows upserted.")


if __name__ == "__main__":
    main()
