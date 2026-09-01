import pandas as pd
import yfinance as yf


def download_india_vix(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download India VIX (market-wide implied volatility) from Yahoo Finance."""

    vix_data = yf.download(
        "^INDIAVIX",
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if vix_data.empty:
        raise ValueError("No India VIX data returned.")

    vix_data = vix_data.reset_index()

    vix_data = vix_data.rename(
        columns={
            "Date": "date",
            "Close": "india_vix",
        }
    )

    return vix_data[["date", "india_vix"]]


def download_nifty_index(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download NIFTY 50 index prices, as a fallback/companion to India VIX."""

    index_data = yf.download(
        "^NSEI",
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if index_data.empty:
        raise ValueError("No NIFTY index data returned.")

    index_data = index_data.reset_index()

    index_data = index_data.rename(
        columns={
            "Date": "date",
            "Close": "nifty_close",
        }
    )

    return index_data[["date", "nifty_close"]]


def calculate_market_volatility(
    index_data: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling realized volatility of the NIFTY index itself."""

    index_data = index_data.copy()

    index_data["nifty_return"] = index_data["nifty_close"].pct_change()

    index_data[f"nifty_volatility_{window}"] = (
        index_data["nifty_return"].rolling(window=window).std()
    )

    return index_data[["date", f"nifty_volatility_{window}"]]


def add_market_features(
    stock_data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Join market-wide volatility features onto every row by date."""

    stock_data = stock_data.copy()

    vix_data = download_india_vix(start_date, end_date)

    nifty_data = download_nifty_index(start_date, end_date)
    nifty_vol = calculate_market_volatility(nifty_data)

    stock_data = stock_data.merge(vix_data, on="date", how="left")
    stock_data = stock_data.merge(nifty_vol, on="date", how="left")

    return stock_data