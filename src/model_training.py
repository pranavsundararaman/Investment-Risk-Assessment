from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


DEFAULT_XGBOOST_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "xgboost_model.json"
)

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model

def predict_random_forest(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Generate predictions using the trained Random Forest."""

    predictions = model.predict(X_test)

    return pd.Series(
        predictions,
        index=X_test.index,
        name="predicted_volatility",
    )

def predict_xgboost(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Generate log-volatility predictions using XGBoost."""

    predictions = model.predict(X_test)

    return pd.Series(
        predictions,
        index=X_test.index,
        name="predicted_log_volatility",
    )

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    best_params: dict,
) -> xgb.XGBRegressor:

    model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    return model


def save_xgboost_model(
    model: xgb.XGBRegressor,
    path: str | Path = DEFAULT_XGBOOST_MODEL_PATH,
) -> None:
    """Save a trained XGBoost model for production predictions."""

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)


def load_xgboost_model(
    path: str | Path = DEFAULT_XGBOOST_MODEL_PATH,
) -> xgb.XGBRegressor:
    """Load a trained XGBoost model from disk."""

    model = xgb.XGBRegressor()
    model.load_model(Path(path))

    return model
