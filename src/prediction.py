import numpy as np
import pandas as pd
import xgboost as xgb


def predict_volatility(
    model: xgb.XGBRegressor,
    features: pd.DataFrame,
) -> pd.Series:
    """Generate 20-day forward volatility predictions."""

    log_predictions = model.predict(features)

    predictions = np.exp(log_predictions)

    return pd.Series(
        predictions,
        index=features.index,
        name="predicted_volatility",
    )


def calculate_risk_thresholds(
    training_volatility: pd.Series,
) -> tuple[float, float]:
    """Calculate risk thresholds from training volatility."""

    low_threshold = training_volatility.quantile(0.33)
    high_threshold = training_volatility.quantile(0.66)

    return low_threshold, high_threshold


def classify_risk(
    predicted_volatility: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    """Convert predicted volatility into risk categories."""

    return pd.cut(
        predicted_volatility,
        bins=[
            -np.inf,
            low_threshold,
            high_threshold,
            np.inf,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
        ],
    )