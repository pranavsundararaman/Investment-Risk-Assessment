import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

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

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBRegressor:
    """Train an XGBoost regression model."""

    model = xgb.XGBRegressor(
    n_estimators=175,
    learning_rate=0.08,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model

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
) -> xgb.XGBRegressor:
    """Train XGBoost with early stopping."""

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
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