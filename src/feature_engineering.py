import pandas as pd
import numpy as np

def calculate_daily_returns(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily percentage returns for each stock.

    Args:
        stock_data (pd.DataFrame): Stock dataset.

    Returns:
        pd.DataFrame: Dataset with daily returns.
    """
    stock_data=stock_data.copy()
    stock_data['daily_return']= stock_data.groupby('ticker')['close'].pct_change()
    return stock_data

def calculate_log_returns(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily logarithmic returns for each stock.

    Args:
        stock_data (pd.DataFrame): Stock dataset.

    Returns:
        pd.DataFrame: Dataset with log returns.
    """
    stock_data=stock_data.copy()
    stock_data["log_return"] = np.log(stock_data.groupby("ticker")["close"].transform(lambda x: x / x.shift(1)))
    return stock_data

def calculate_sma(
    stock_data: pd.DataFrame,sma: int=20
) -> pd.DataFrame:
    """
    Calculate the {sma}-day Simple Moving Average (SMA).

    Args:
        stock_data (pd.DataFrame): Stock dataset.
        sma (int): SMA period.

    Returns:
        pd.DataFrame: Dataset with SMA.
    """

    stock_data = stock_data.copy()

    stock_data[f"sma_{sma}"] = (
        stock_data.groupby("ticker")["close"]
        .transform(
            lambda x: x.rolling(window=sma).mean()
        )
    )

    return stock_data

def calculate_ema(
    stock_data: pd.DataFrame,
    span: int = 20,
) -> pd.DataFrame:
    """
    Calculate the Exponential Moving Average (EMA).

    Args:
        stock_data (pd.DataFrame): Stock dataset.
        span (int): EMA period.

    Returns:
        pd.DataFrame: Dataset with EMA.
    """

    stock_data = stock_data.copy()

    stock_data[f"ema_{span}"] = (
        stock_data.groupby("ticker")["close"]
        .transform(
            lambda x: x.ewm(
                span=span,
                adjust=False,
            ).mean()
        )
    )

    return stock_data

def calculate_volatility(
    stock_data: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Calculate rolling volatility using daily returns.

    Args:
        stock_data (pd.DataFrame): Stock dataset.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: Dataset with rolling volatility.
    """

    stock_data = stock_data.copy()

    stock_data[f"volatility_{window}"] = (
        stock_data.groupby("ticker")["daily_return"]
        .transform(
            lambda x: x.rolling(window=window).std()
        )
    )

    return stock_data

import numpy as np

def calculate_rsi(
    stock_data: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI).

    Args:
        stock_data (pd.DataFrame): Stock dataset.
        window (int): RSI window.

    Returns:
        pd.DataFrame: Dataset with RSI.
    """

    stock_data = stock_data.copy()

    def compute_rsi(close: pd.Series) -> pd.Series:

        delta = close.diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()

        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        rsi=rsi.fillna(50)

        return rsi

    stock_data[f"rsi_{window}"] = (
        stock_data.groupby("ticker")["close"]
        .transform(compute_rsi)
    )

    return stock_data

def calculate_macd(
    stock_data: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD).

    Args:
        stock_data (pd.DataFrame): Stock dataset.
        short_window (int): Short EMA period.
        long_window (int): Long EMA period.
        signal_window (int): Signal line EMA period.

    Returns:
        pd.DataFrame: Dataset with MACD features.
    """

    stock_data = stock_data.copy()

    def compute_macd(close: pd.Series) -> pd.DataFrame:

        ema_short = close.ewm(
            span=short_window,
            adjust=False,
        ).mean()

        ema_long = close.ewm(
            span=long_window,
            adjust=False,
        ).mean()

        macd = ema_short - ema_long

        signal = macd.ewm(
            span=signal_window,
            adjust=False,
        ).mean()

        histogram = macd - signal

        return pd.DataFrame(
            {
                "macd": macd,
                "macd_signal": signal,
                "macd_histogram": histogram,
            }
        )

    macd_features = (
        stock_data.groupby("ticker")["close"]
        .apply(compute_macd)
        .reset_index(level=0, drop=True)
    )

    stock_data = pd.concat(
        [stock_data, macd_features],
        axis=1,
    )

    return stock_data

def calculate_bollinger_bands(
    stock_data: pd.DataFrame,
    window: int = 20,
    num_std: int = 2,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        stock_data (pd.DataFrame): Stock dataset.
        window (int): Rolling window.
        num_std (int): Number of standard deviations.

    Returns:
        pd.DataFrame: Dataset with Bollinger Bands.
    """

    stock_data = stock_data.copy()

    def compute_bands(close: pd.Series) -> pd.DataFrame:

        sma = close.rolling(window).mean()

        std = close.rolling(window).std()

        upper = sma + (num_std * std)

        lower = sma - (num_std * std)

        return pd.DataFrame(
            {
                "bollinger_upper": upper,
                "bollinger_middle": sma,
                "bollinger_lower": lower,
            }
        )

    bands = (
        stock_data.groupby("ticker")["close"]
        .apply(compute_bands)
        .reset_index(level=0, drop=True)
    )

    stock_data = pd.concat(
        [stock_data, bands],
        axis=1,
    )

    return stock_data

def add_multi_horizon_volatility(
    stock_data: pd.DataFrame,
    windows: list[int] = [5, 10, 20, 60],
) -> pd.DataFrame:
    """Add rolling volatility features at multiple horizons."""

    stock_data = stock_data.copy()

    for window in windows:
        stock_data = calculate_volatility(
            stock_data,
            window=window,
        )

    return stock_data

def engineer_features(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Args:
        stock_data (pd.DataFrame): Preprocessed stock dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """

    stock_data = calculate_daily_returns(stock_data)

    stock_data = calculate_log_returns(stock_data)

    stock_data = calculate_sma(stock_data)

    stock_data = calculate_ema(stock_data)

    stock_data = add_multi_horizon_volatility(stock_data)

    stock_data = calculate_rsi(stock_data)

    stock_data = calculate_macd(stock_data)

    stock_data = calculate_bollinger_bands(stock_data)

    stock_data["close_to_sma_20"] = (
    stock_data["close"] / stock_data["sma_20"] - 1
    )

    stock_data["close_to_ema_20"] = (
        stock_data["close"] / stock_data["ema_20"] - 1
    )

    stock_data["macd_ratio"] = (
        stock_data["macd"] / stock_data["close"]
    )

    stock_data["bollinger_position"] = (
        (stock_data["close"] - stock_data["bollinger_lower"])
        / (
            stock_data["bollinger_upper"]
            - stock_data["bollinger_lower"]
        )
    )
    stock_data["macd_histogram_norm"] = (
    stock_data["macd_histogram"] / stock_data["close"]
    )

    stock_data["high_low_range"] = (
        (stock_data["high"] - stock_data["low"])
        / stock_data["close"]
    )

    stock_data["volume_ratio"] = (
        stock_data["volume"]
        / stock_data.groupby("ticker")["volume"]
        .transform(lambda x: x.rolling(20).mean())
    )
    return stock_data