import pandas as pd


def prepare_training_dataset(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the dataset for model training."""

    stock_data = (
        stock_data
        .dropna()
        .reset_index(drop=True)
    )

    return stock_data

def split_features_target(
    stock_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target."""

    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "log_return",
        "sma_20",
        "ema_20",
        "volatility_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
    ]

    X = stock_data[feature_columns]

    y = stock_data["future_volatility"]

    return X, y

def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    split_date: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split the dataset chronologically using a date cutoff."""

    train_mask = dates < split_date
    test_mask = dates >= split_date

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test