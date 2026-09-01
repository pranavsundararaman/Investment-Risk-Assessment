import numpy as np
import pandas as pd

from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

def evaluate_model(
    y_test: pd.Series,
    predictions: pd.Series,
) -> dict[str, float]:
    """Evaluate model predictions."""

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

def mean_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.Series:
    """Predict the training-set mean for every test observation."""

    mean_volatility = y_train.mean()

    predictions = pd.Series(
        mean_volatility,
        index=y_test.index,
        name="mean_prediction",
    )

    return predictions

