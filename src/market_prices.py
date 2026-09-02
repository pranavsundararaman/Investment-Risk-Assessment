from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_current_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch the latest available market price and daily change
    for multiple tickers using a single yfinance request.

    Returns:
        DataFrame with:
        - ticker
        - current_price
        - previous_close
        - daily_change
        - daily_change_pct
        - price_date
    """

    if not tickers:
        return pd.DataFrame(
            columns=[
                "ticker",
                "current_price",
                "previous_close",
                "daily_change",
                "daily_change_pct",
                "price_date",
            ]
        )

    try:
        data = yf.download(
            tickers=tickers,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

    except Exception as exc:
        print(f"Unable to fetch current market prices: {exc}")

        return pd.DataFrame(
            columns=[
                "ticker",
                "current_price",
                "previous_close",
                "daily_change",
                "daily_change_pct",
                "price_date",
            ]
        )

    if data.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "current_price",
                "previous_close",
                "daily_change",
                "daily_change_pct",
                "price_date",
            ]
        )

    results = []

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                ticker_data = data[ticker]
            else:
                ticker_data = data

            ticker_data = ticker_data.dropna(subset=["Close"])

            if len(ticker_data) == 0:
                continue

            latest = ticker_data.iloc[-1]

            current_price = float(latest["Close"])

            previous_close = None
            daily_change = None
            daily_change_pct = None

            if len(ticker_data) >= 2:
                previous_close = float(ticker_data.iloc[-2]["Close"])

                if previous_close != 0:
                    daily_change = current_price - previous_close
                    daily_change_pct = (
                        daily_change / previous_close
                    ) * 100

            price_date = ticker_data.index[-1]

            results.append(
                {
                    "ticker": ticker,
                    "current_price": current_price,
                    "previous_close": previous_close,
                    "daily_change": daily_change,
                    "daily_change_pct": daily_change_pct,
                    "price_date": pd.Timestamp(price_date),
                }
            )

        except Exception as exc:
            print(f"Unable to process price for {ticker}: {exc}")

    return pd.DataFrame(results)