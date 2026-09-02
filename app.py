from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import engineer_features
from src.prediction_service import (
    generate_latest_predictions,
    load_stock_data_from_database,
)
from src.preprocessing import preprocess_stock_data
from src.market_prices import fetch_current_prices


RISK_ORDER = ["Low", "Moderate", "High"]

RISK_COLORS = {
    "Low": "#16a34a",
    "Moderate": "#f59e0b",
    "High": "#dc2626",
}


st.set_page_config(
    page_title="Voltra — Volatility Intelligence for Indian Markets",
    layout="wide",
)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Load latest model predictions from the existing production service."""
    predictions = generate_latest_predictions()
    predictions["date"] = pd.to_datetime(predictions["date"])
    return predictions


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_historical_features() -> pd.DataFrame:
    """Load stock history and compute existing volatility indicators."""
    stock_data = load_stock_data_from_database()

    processed_data = preprocess_stock_data(stock_data)

    featured_data = engineer_features(processed_data)

    featured_data["date"] = pd.to_datetime(featured_data["date"])

    return featured_data


@st.cache_data(ttl=5 * 60, show_spinner=False)
def load_current_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Load the latest available market price and daily change
    for all NIFTY 50 stocks.

    Prices are cached for 5 minutes to avoid repeatedly
    querying yfinance.
    """
    return fetch_current_prices(list(tickers))


def enrich_predictions(
    predictions: pd.DataFrame,
    historical_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach available company metadata to the prediction output."""

    company_lookup = (
        historical_features[["ticker", "company_name", "industry"]]
        .dropna(subset=["ticker"])
        .drop_duplicates(subset=["ticker"], keep="last")
    )

    enriched = predictions.merge(
        company_lookup,
        on="ticker",
        how="left",
    )

    return enriched.sort_values(
        "predicted_volatility",
        ascending=False,
    ).reset_index(drop=True)


def risk_count_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        predictions["risk"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("risk")
        .reset_index(name="count")
    )

    return counts


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def render_kpis(predictions: pd.DataFrame) -> None:
    latest_date = predictions["date"].max()

    counts = predictions["risk"].value_counts()

    metric_cols = st.columns(5)

    metric_cols[0].metric(
        "Latest market date",
        latest_date.strftime("%d %b %Y"),
    )

    metric_cols[1].metric(
        "Stocks analysed",
        f"{len(predictions):,}",
    )

    metric_cols[2].metric(
        "High risk",
        int(counts.get("High", 0)),
    )

    metric_cols[3].metric(
        "Moderate risk",
        int(counts.get("Moderate", 0)),
    )

    metric_cols[4].metric(
        "Low risk",
        int(counts.get("Low", 0)),
    )


def render_risk_distribution(predictions: pd.DataFrame) -> None:
    counts = risk_count_frame(predictions)

    fig = px.bar(
        counts,
        x="risk",
        y="count",
        color="risk",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk": RISK_ORDER},
        text="count",
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Risk category",
        yaxis_title="Number of stocks",
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def render_ranking_table(predictions: pd.DataFrame) -> None:
    table = predictions[
        [
            "ticker",
            "company_name",
            "industry",
            "current_price",
            "daily_change_pct",
            "date",
            "predicted_volatility",
            "risk",
        ]
    ].copy()

    table = table.rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company",
            "industry": "Industry",
            "current_price": "Price",
            "daily_change_pct": "Day Change",
            "date": "Date",
            "predicted_volatility": "Predicted Volatility",
            "risk": "Risk",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(
                format="₹%.2f",
                help="Latest available market price.",
            ),
            "Day Change": st.column_config.NumberColumn(
                format="%.2f%%",
                help="Percentage change from the previous trading session.",
            ),
            "Date": st.column_config.DateColumn(
                format="DD MMM YYYY",
            ),
            "Predicted Volatility": st.column_config.NumberColumn(
                format="%.2%",
                help="Model-estimated forward volatility.",
            ),
        },
    )


def render_stock_analysis(
    predictions: pd.DataFrame,
    historical_features: pd.DataFrame,
) -> None:
    ticker_options = (
        predictions["ticker"]
        .sort_values()
        .tolist()
    )

    selected_ticker = st.selectbox(
        "Select stock",
        ticker_options,
    )

    selected_prediction = predictions.loc[
        predictions["ticker"] == selected_ticker
    ].iloc[0]

    selected_history = (
        historical_features.loc[
            historical_features["ticker"] == selected_ticker
        ]
        .sort_values("date")
        .copy()
    )

    title = selected_prediction.get("company_name")

    if pd.isna(title) or not title:
        title = selected_ticker

    st.subheader(
        f"{selected_ticker} - {title}"
    )

    # ---------------------------------------------------------
    # Current market information
    # ---------------------------------------------------------

    current_price = selected_prediction.get(
        "current_price"
    )

    daily_change_pct = selected_prediction.get(
        "daily_change_pct"
    )

    detail_cols = st.columns(4)

    if pd.notna(current_price):
        detail_cols[0].metric(
            "Latest market price",
            f"₹{float(current_price):,.2f}",
        )
    else:
        detail_cols[0].metric(
            "Latest market price",
            "Not available",
        )

    detail_cols[1].metric(
        "Predicted volatility",
        format_percent(
            float(
                selected_prediction["predicted_volatility"]
            )
        ),
    )

    detail_cols[2].metric(
        "Risk level",
        selected_prediction["risk"],
    )

    latest_realized_volatility = (
        selected_history["volatility_20"]
        .dropna()
    )

    if latest_realized_volatility.empty:
        detail_cols[3].metric(
            "Latest 20-day volatility",
            "Not available",
        )
    else:
        detail_cols[3].metric(
            "Latest 20-day volatility",
            format_percent(
                float(
                    latest_realized_volatility.iloc[-1]
                )
            ),
        )

    # ---------------------------------------------------------
    # Daily price change
    # ---------------------------------------------------------

    if pd.notna(daily_change_pct):
        daily_change_pct = float(daily_change_pct)

        if daily_change_pct >= 0:
            st.caption(
                f"Day change: +{daily_change_pct:.2f}%"
            )
        else:
            st.caption(
                f"Day change: {daily_change_pct:.2f}%"
            )
    else:
        st.caption(
            "Day change: Not available"
        )

    # ---------------------------------------------------------
    # Historical volatility chart
    # ---------------------------------------------------------

    chart_data = selected_history[
        [
            "date",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volatility_60",
        ]
    ].dropna(
        how="all",
        subset=[
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volatility_60",
        ],
    )

    if chart_data.empty:
        st.info(
            "Historical volatility indicators are not available "
            "for this stock yet."
        )

        return

    chart_data = chart_data.tail(252)

    fig = go.Figure()

    for column, label in [
        ("volatility_5", "5-day volatility"),
        ("volatility_10", "10-day volatility"),
        ("volatility_20", "20-day volatility"),
        ("volatility_60", "60-day volatility"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=chart_data["date"],
                y=chart_data[column],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title="Realized volatility",
        legend_title_text="Indicator",
    )

    fig.update_yaxes(
        tickformat=".2%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def main() -> None:
    st.title(
        "Voltra — Volatility Intelligence for Indian Markets"
    )

    st.caption(
        "AI-powered volatility and relative risk assessment "
        "for Indian NIFTY 50 stocks."
    )

    try:
        # -----------------------------------------------------
        # Load predictions
        # -----------------------------------------------------

        with st.spinner(
            "Loading latest predictions..."
        ):
            predictions = load_predictions()

        # -----------------------------------------------------
        # Load historical features
        # -----------------------------------------------------

        with st.spinner(
            "Loading historical stock features..."
        ):
            historical_features = load_historical_features()

        # -----------------------------------------------------
        # Add company metadata
        # -----------------------------------------------------

        predictions = enrich_predictions(
            predictions,
            historical_features,
        )

        # -----------------------------------------------------
        # Load latest market prices
        # -----------------------------------------------------

        with st.spinner(
            "Fetching latest market prices..."
        ):
            tickers = tuple(
                predictions["ticker"].tolist()
            )

            current_prices = load_current_prices(
                tickers
            )

        # -----------------------------------------------------
        # Merge market prices into predictions
        # -----------------------------------------------------

        if not current_prices.empty:
            predictions = predictions.merge(
                current_prices,
                on="ticker",
                how="left",
            )
        else:
            predictions["current_price"] = pd.NA
            predictions["previous_close"] = pd.NA
            predictions["daily_change"] = pd.NA
            predictions["daily_change_pct"] = pd.NA
            predictions["price_date"] = pd.NaT

    except FileNotFoundError as exc:
        st.error(
            "The saved XGBoost model was not found. "
            "Run the training pipeline once to create "
            "models/xgboost_model.json, then reopen the dashboard."
        )

        st.exception(exc)

        return

    except Exception as exc:
        st.error(
            "The dashboard could not load predictions or stock data. "
            "Check your database connection, environment variables, "
            "saved model, and market data access."
        )

        st.exception(exc)

        return

    # ---------------------------------------------------------
    # Market overview
    # ---------------------------------------------------------

    st.markdown("### Market Overview")

    render_kpis(predictions)

    left_col, right_col = st.columns([1, 1.4])

    with left_col:
        st.markdown("### Risk Distribution")

        render_risk_distribution(
            predictions
        )

    with right_col:
        st.markdown("### Individual Stock Analysis")

        render_stock_analysis(
            predictions,
            historical_features,
        )

    # ---------------------------------------------------------
    # Risk ranking
    # ---------------------------------------------------------

    st.markdown("### Risk Ranking")

    st.caption(
        "Sorted by predicted volatility, highest first."
    )

    render_ranking_table(
        predictions
    )


if __name__ == "__main__":
    main()