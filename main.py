import numpy as np

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

from src.target_engineering import (
    create_future_volatility_target,
    remove_missing_targets,
    create_log_volatility_target,
)

from src.dataset import (
    prepare_training_dataset,
    split_features_target,
    time_series_train_validation_test_split,
)

from src.model_training import (
    train_xgboost,
    predict_xgboost,
)

from src.evaluation import (
    evaluate_model,
)

from src.hyperparameter_search import tune_xgboost

from src.market_features import add_market_features


def main() -> None:
    """
    Run the complete machine learning pipeline.
    """

    # =========================
    # 1. Database
    # =========================

    engine = create_database_engine()

    stock_data = fetch_stock_data(engine)

    # =========================
    # 2. Preprocessing
    # =========================

    stock_data = preprocess_stock_data(
        stock_data
    )

    # =========================
    # 3. Feature Engineering
    # =========================

    stock_data = engineer_features(
        stock_data
    )

    print(
        "\n===== FEATURE ENGINEERING COMPLETED SUCCESSFULLY =====\n"
    )

    print("Shape:")
    print(stock_data.shape)

    print("\nColumns:")
    print(stock_data.columns.tolist())

    print("\nMissing Values:")
    print(stock_data.isna().sum())

    stock_data = add_market_features(
    stock_data,
    start_date=str(stock_data["date"].min().date()),
    end_date=str(stock_data["date"].max().date()),
    )

    # =========================
    # 4. Target Engineering
    # =========================

    stock_data = create_future_volatility_target(
        stock_data
    )

    stock_data = remove_missing_targets(
        stock_data
    )

    stock_data = create_log_volatility_target(
        stock_data
    )

    # =========================
    # 5. Prepare Dataset
    # =========================

    stock_data = prepare_training_dataset(
        stock_data
    )

    print(
        "\nTotal missing values:",
        stock_data.isna().sum().sum(),
    )

    print(
        "Training dataset shape:",
        stock_data.shape,
    )

    # =========================
    # 6. Split Features / Target
    # =========================

    X, y = split_features_target(
        stock_data
    )

    # =========================
    # 7. Time-Series Split
    # =========================

    (X_train,X_validation,X_test,
    y_train,y_validation,y_test) = time_series_train_validation_test_split(
        X,
        y,
        stock_data["date"],
        "2024-01-01",
        "2025-01-01",
    )
    

    print("\nX_train:", X_train.shape)
    print("X_validation:", X_validation.shape)
    print("X_test:", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_validation:", y_validation.shape)
    print("y_test:", y_test.shape)

    print(
    "\nTrain:",
    stock_data.loc[X_train.index, "date"].min(),
    "to",
    stock_data.loc[X_train.index, "date"].max(),
)

    print(
        "Validation:",
        stock_data.loc[X_validation.index, "date"].min(),
        "to",
        stock_data.loc[X_validation.index, "date"].max(),
    )

    print(
        "Test:",
        stock_data.loc[X_test.index, "date"].min(),
        "to",
        stock_data.loc[X_test.index, "date"].max(),
    )

    print("\nFeatures:")
    print(X_train.columns.tolist())

    print("\nLog Target:")
    print(y_train.head())

    # =========================
    # 8. Hyperparameter Search
    # =========================

    best_params = tune_xgboost(
    X_train,
    y_train,
    stock_data.loc[X_train.index, "date"],
    )
    xgb_model = train_xgboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
    best_params,
    )
    # =========================
    # 9. Predict
    # =========================

    xgb_predictions_log = predict_xgboost(
        xgb_model,
        X_test,
    )

    # Convert log volatility
    # back to normal volatility

    xgb_predictions = np.exp(
        xgb_predictions_log
    )

    xgb_predictions.name = "predicted_volatility"

    print("\nXGBoost Predictions:")
    print(
        xgb_predictions.head(10)
    )

    # =========================
    # 10. Evaluate
    # =========================

    # Original target for evaluation
    y_test_original = stock_data.loc[
        X_test.index,
        "future_volatility",
    ]

    xgb_metrics = evaluate_model(
        y_test_original,
        xgb_predictions,
    )

    print(
        "\nXGBoost Performance:"
    )

    print(xgb_metrics)
    print("X_test shape at eval time:", X_test.shape)
    print("Model object id:", id(xgb_model))

    from src.garch_baseline import run_garch_baseline

    # =========================
    # 11. GARCH(1,1) Baseline
    # =========================

    test_rows = stock_data.loc[
        X_test.index, ["date", "ticker", "future_volatility"]
    ]

    test_dates_by_ticker = (
        test_rows.groupby("ticker")["date"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )

    garch_predictions = run_garch_baseline(
        stock_data,
        test_dates_by_ticker,
        horizon=20,
        refit_every=5,
    )

    comparison = test_rows.merge(
        garch_predictions,
        on=["date", "ticker"],
        how="inner",
    )

    garch_metrics = evaluate_model(
        comparison["future_volatility"],
        comparison["garch_predicted_volatility"],
    )

    print("\nGARCH(1,1) Baseline Performance:")
    print(garch_metrics)

    print("\nXGBoost R²:", xgb_metrics["r2"], "vs GARCH R²:", garch_metrics["r2"])

if __name__ == "__main__":
    main()