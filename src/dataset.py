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
    "daily_return",
    "log_return",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "rsi_14",
    "macd_ratio",
    "macd_histogram_norm",
    "close_to_sma_20",
    "close_to_ema_20",
    "bollinger_position",
    "high_low_range",
    "volume_ratio",
    "india_vix",
    "nifty_volatility_20",
    ]

    X = stock_data[feature_columns]

    y = stock_data["log_future_volatility"]

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

def time_series_train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    validation_start: str,
    test_start: str,
):
    """Split data chronologically into train, validation, and test sets."""

    train_mask = dates < validation_start

    validation_mask = (
        (dates >= validation_start)
        & (dates < test_start)
    )

    test_mask = dates >= test_start

    X_train = X.loc[train_mask]
    X_validation = X.loc[validation_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_validation = y.loc[validation_mask]
    y_test = y.loc[test_mask]

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )