from src.database import (
    create_database_engine,
    fetch_stock_data,
)

from src.preprocessing import (
    preprocess_stock_data,
)

from src.feature_engineering import (
    engineer_features,
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

if __name__ == "__main__":
    main()