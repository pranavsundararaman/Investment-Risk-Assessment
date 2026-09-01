from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.database import (
    create_database_engine,
    fetch_stock_data,
)

from src.dataset import (
    split_features_target,
)

from src.feature_engineering import (
    engineer_features,
)

from src.market_features import (
    add_market_features,
)

from src.model_training import (
    DEFAULT_XGBOOST_MODEL_PATH,
    load_xgboost_model,
)

from src.prediction import (
    calculate_risk_thresholds,
    classify_risk,
    predict_volatility,
)

from src.preprocessing import (
    preprocess_stock_data,
)


def load_stock_data_from_database() -> pd.DataFrame:
    """Fetch historical stock prices from the configured PostgreSQL database."""

    engine = create_database_engine()

    return fetch_stock_data(engine)


def build_feature_dataset(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run the same preprocessing, stock features, and market features as training."""

    processed_data = preprocess_stock_data(
        stock_data
    )

    featured_data = engineer_features(
        processed_data
    )

    start_date = featured_data["date"].min()

    end_date = (
        featured_data["date"].max()
        + pd.Timedelta(days=1)
    )

    return add_market_features(
        featured_data,
        start_date=start_date,
        end_date=end_date,
    )


def get_model_feature_columns(
    featured_data: pd.DataFrame,
) -> list[str]:
    """Return the exact feature columns expected by the trained model."""

    if "log_future_volatility" in featured_data.columns:
        feature_source = featured_data
    else:
        feature_source = featured_data.assign(
            log_future_volatility=0.0
        )

    features, _ = split_features_target(
        feature_source
    )

    return list(features.columns)


def generate_latest_predictions(
    stock_data: pd.DataFrame | None = None,
    model_path: str | Path = DEFAULT_XGBOOST_MODEL_PATH,
) -> pd.DataFrame:
    """Generate latest volatility and risk predictions for every ticker."""

    if stock_data is None:
        stock_data = load_stock_data_from_database()

    featured_data = build_feature_dataset(
        stock_data
    )

    feature_columns = get_model_feature_columns(
        featured_data
    )

    latest_rows = (
        featured_data
        .dropna(subset=feature_columns)
        .sort_values(
            ["ticker", "date"]
        )
        .groupby(
            "ticker",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    if latest_rows.empty:
        raise ValueError(
            "No complete latest feature rows are available for prediction."
        )

    model = load_xgboost_model(
        model_path
    )

    predictions = predict_volatility(
        model,
        latest_rows[feature_columns],
    )

    low_threshold, high_threshold = (
        calculate_risk_thresholds(
            predictions
        )
    )

    risk_labels = classify_risk(
        predictions,
        low_threshold,
        high_threshold,
    )

    result = latest_rows[
        [
            "ticker",
            "date",
        ]
    ].copy()

    result["predicted_volatility"] = (
        predictions.to_numpy()
    )

    result["risk"] = (
        risk_labels
        .astype(str)
        .to_numpy()
    )

    return (
        result
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def main() -> None:
    predictions = generate_latest_predictions()

    print(
        "\nLatest volatility risk predictions:"
    )

    print(
        predictions.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()