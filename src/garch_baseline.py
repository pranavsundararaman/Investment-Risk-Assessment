import numpy as np
import pandas as pd
from arch import arch_model


def fit_and_forecast_garch(
    returns: pd.Series,
    forecast_dates: list,
    horizon: int = 20,
    refit_every: int = 5,
) -> pd.Series:
    """
    Walk-forward GARCH(1,1) volatility forecasts for one ticker's returns.

    For each forecast date, uses only returns strictly before that date
    (no lookahead), forecasts `horizon` days ahead, and aggregates the
    forecasted variance path into a single expected volatility number —
    comparable to your rolling-std `future_volatility` target.

    Refits every `refit_every` dates instead of daily, since refitting
    GARCH on every single day is unnecessary and slow.
    """

    returns = returns.dropna().sort_index()
    predictions = {}
    model_fit = None
    since_refit = refit_every  # force a fit on the first date

    for date in forecast_dates:
        history = returns.loc[returns.index < date]

        if len(history) < 100:
            continue  # not enough history for a stable fit

        if since_refit >= refit_every:
            garch = arch_model(
                history * 100,  # percent scale — GARCH optimizers are more stable this way
                vol="Garch",
                p=1,
                q=1,
                dist="normal",
            )
            model_fit = garch.fit(disp="off")
            since_refit = 0

        since_refit += 1

        forecast = model_fit.forecast(horizon=horizon, reindex=False)
        variance_path = forecast.variance.values[-1]

        expected_volatility = np.sqrt(variance_path.mean()) / 100  # undo percent scaling

        predictions[date] = expected_volatility

    return pd.Series(predictions, name="garch_predicted_volatility")


def run_garch_baseline(
    stock_data: pd.DataFrame,
    test_dates_by_ticker: dict,
    horizon: int = 20,
    refit_every: int = 5,
) -> pd.DataFrame:
    """Run the GARCH(1,1) baseline across every ticker and combine results."""

    all_predictions = []

    for ticker, dates in test_dates_by_ticker.items():
        print(f"Fitting GARCH for {ticker}...")

        ticker_returns = (
            stock_data
            .loc[stock_data["ticker"] == ticker]
            .set_index("date")["daily_return"]
        )

        predictions = fit_and_forecast_garch(
            ticker_returns,
            dates,
            horizon=horizon,
            refit_every=refit_every,
        )

        predictions = predictions.to_frame()
        predictions["ticker"] = ticker

        all_predictions.append(predictions)

    combined = pd.concat(all_predictions)
    combined.index.name = "date"

    return combined.reset_index()