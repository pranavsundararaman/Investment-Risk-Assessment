import pandas as pd
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV


class DateTimeSeriesSplit:
    """Time-series cross-validation using calendar dates."""

    def __init__(
        self,
        dates: pd.Series,
        n_splits: int = 5,
        gap_days: int = 20,
    ):
        self.dates = pd.to_datetime(dates)
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,
    ):
        dates = self.dates.loc[X.index]

        unique_dates = sorted(
            dates.dt.normalize().unique()
        )

        n_dates = len(unique_dates)

        fold_size = n_dates // (
            self.n_splits + 1
        )

        for i in range(self.n_splits):

            train_end = (
                fold_size * (i + 1)
            )

            validation_start = (
                train_end + self.gap_days
            )

            validation_end = (
                fold_size * (i + 2)
            )

            if validation_end > n_dates:
                validation_end = n_dates

            train_dates = unique_dates[
                :train_end
            ]

            validation_dates = unique_dates[
                validation_start:validation_end
            ]

            if len(validation_dates) == 0:
                continue

            train_mask = dates.isin(
                train_dates
            )

            validation_mask = dates.isin(
                validation_dates
            )
            train_indices = [
                i
                for i, value in enumerate(train_mask)
                if value
                ]

            validation_indices = [
                    i
                    for i, value in enumerate(validation_mask)
                    if value
                ]

            yield (
                train_indices,
                validation_indices,
            )

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ):
        return self.n_splits


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates: pd.Series,
    n_splits: int = 5,
    n_iter: int = 30,
) -> dict:
    """Tune XGBoost using date-based time-series CV."""

    param_distributions = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [
            0.005,
            0.01,
            0.02,
            0.05,
        ],
        "n_estimators": [
            300,
            500,
            800,
            1200,
        ],
        "min_child_weight": [
            1,
            3,
            5,
            10,
        ],
        "subsample": [
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "colsample_bytree": [
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "reg_alpha": [
            0,
            0.01,
            0.1,
            1,
        ],
        "reg_lambda": [
            0.5,
            1,
            2,
            5,
        ],
    }

    cv = DateTimeSeriesSplit(
        dates=dates,
        n_splits=n_splits,
        gap_days=20,
    )

    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    print(
        "\nBest params:",
        search.best_params_,
    )

    print(
        "Best CV RMSE:",
        -search.best_score_,
    )

    return search.best_params_