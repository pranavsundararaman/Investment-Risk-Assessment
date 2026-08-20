from src.database import (
    create_database_engine,
    fetch_stock_data,
)
from src.dataset import prepare_training_dataset, split_features_target
from src.preprocessing import (
    preprocess_stock_data,
)

from src.feature_engineering import (
    engineer_features,
)
from src.target_engineering import create_future_volatility_target, remove_missing_targets
from src.dataset import (
    prepare_training_dataset,
    split_features_target,
    time_series_train_test_split,
)

def main() -> None:
    """
    Test the complete data pipeline.
    """

    engine = create_database_engine()

    stock_data = fetch_stock_data(engine)

    stock_data = preprocess_stock_data(stock_data)

    stock_data = engineer_features(stock_data)

    print("\n===== FEATURE ENGINEERING COMPLETED SUCCESSFULLY =====\n")

    print("Shape:")
    print(stock_data.shape)

    print("\nColumns:")
    print(stock_data.columns.tolist())

    print("\nData Types:")
    print(stock_data.dtypes)

    print("\nMissing Values:")
    print(stock_data.isna().sum())

    print("\nFirst 30 Rows:")
    print(
        stock_data[
            [
                "ticker",
                "date",
                "close",
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
        ].head(30)
    )
    from src.target_engineering import (
        create_future_volatility_target,
    )

    stock_data = create_future_volatility_target(
        stock_data
    )

    print(
        stock_data[
            [
                "ticker",
                "date",
                "daily_return",
                "future_volatility",
            ]
        ].head(30)
    )
    stock_data = create_future_volatility_target(
        stock_data
    )

    stock_data = remove_missing_targets(
        stock_data
    )

    stock_data = prepare_training_dataset(
        stock_data
    )

    print(stock_data.isna().sum().sum())
    print(stock_data.shape)

    X, y = split_features_target(stock_data)

    X_train, X_test, y_train, y_test = (
        time_series_train_test_split(
            X,
            y,
            stock_data["date"],
            "2025-01-01",
        )
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    
    print(
    "Train:",
    stock_data.loc[X_train.index, "date"].min(),
    "to",
    stock_data.loc[X_train.index, "date"].max(),
)

    print(
        "Test:",
        stock_data.loc[X_test.index, "date"].min(),
        "to",
        stock_data.loc[X_test.index, "date"].max(),
    )
if __name__ == "__main__":
    main()