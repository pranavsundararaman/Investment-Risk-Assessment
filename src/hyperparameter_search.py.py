import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    n_iter: int = 30,
) -> xgb.XGBRegressor:
    """Hyperparameter search using time-series-aware, leakage-safe CV."""

    param_distributions = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.005, 0.01, 0.02, 0.05],
        "n_estimators": [300, 500, 800, 1200],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.5, 1, 2, 5],
    }

    # gap=20 skips 20 rows between each train/val fold boundary,
    # since your target looks 20 days ahead — without this, rows
    # right at the split still leak target information across folds.
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=20)

    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    print("Best CV RMSE:", -search.best_score_)

    return search.best_estimator_