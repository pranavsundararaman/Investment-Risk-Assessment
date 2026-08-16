from pathlib import Path
import pandas as pd
import yfinance as yf

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