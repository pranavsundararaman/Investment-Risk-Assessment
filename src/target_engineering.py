import pandas as pd
import numpy as np

def create_future_volatility_target(
    stock_data: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    """Create the future volatility target."""

    stock_data["future_volatility"] = (
        stock_data
        .groupby("ticker")["daily_return"]
        .transform(
            lambda x:
            x.rolling(horizon)
             .std()
             .shift(-horizon)
        )
    )

    return stock_data

def remove_missing_targets(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows with missing target values."""

    stock_data = stock_data.dropna(
        subset=["future_volatility"]
    )

    stock_data = stock_data.reset_index(
        drop=True
    )

    return stock_data

def create_log_volatility_target(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a log-transformed future volatility target."""

    stock_data["log_future_volatility"] = np.log(
        stock_data["future_volatility"]
    )

    return stock_data