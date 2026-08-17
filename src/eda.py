import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_stock_price(
    stock_data: pd.DataFrame,
    ticker: str,
) -> None:
    """
    Plot the historical closing price of a stock.

    Args:
        stock_data (pd.DataFrame): Complete stock dataset.
        ticker (str): Stock ticker to plot.
    """

    stock = stock_data[
        stock_data["ticker"] == ticker
    ]

    if stock.empty:
        raise ValueError(
            f"No data found for ticker: {ticker}"
        )

    stock = stock.sort_values(
        by="date"
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        stock["date"],
        stock["close"],
        linewidth=2,
        label="Close Price",
    )

    plt.title(
        f"{ticker} Closing Price"
    )

    plt.xlabel("Date")
    plt.ylabel("Closing Price (₹)")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

def plot_daily_return_distribution(
    stock_data: pd.DataFrame,
    ticker: str,
) -> None:
    """Plot the distribution of daily returns for a stock."""

    stock = stock_data[
        stock_data["ticker"] == ticker
    ]

    if stock.empty:
        raise ValueError(
            f"No data found for ticker: {ticker}"
        )

    plt.figure(figsize=(10, 6))

    plt.hist(
        stock["daily_return"].dropna(),
        bins=30,
        edgecolor="black",
    )

    plt.title(
        f"Daily Return Distribution - {ticker}"
    )

    plt.xlabel("Daily Return")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

def plot_correlation_heatmap(
    stock_data: pd.DataFrame,
) -> None:
    """Plot the correlation heatmap for numerical features."""

    numerical_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "log_return",
        "sma_20",
        "ema_20",
        "volatility_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
    ]

    correlation_matrix = (
        stock_data[numerical_columns]
        .corr()
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    annot=True,
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(numerical_columns)),
        numerical_columns,
        rotation=90,
    )

    plt.yticks(
        range(len(numerical_columns)),
        numerical_columns,
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.show()

def plot_boxplot_returns(
    stock_data: pd.DataFrame,
    ticker: str,
) -> None:
    """Plot a boxplot of daily returns."""

    stock = stock_data[
        stock_data["ticker"] == ticker
    ]

    if stock.empty:
        raise ValueError(
            f"No data found for ticker: {ticker}"
        )

    plt.figure(figsize=(6, 8))

    plt.boxplot(
        stock["daily_return"].dropna()
    )

    plt.title(
        f"Daily Return Box Plot - {ticker}"
    )

    plt.ylabel("Daily Return")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

def generate_summary_statistics(
    stock_data: pd.DataFrame,
) -> None:
    """Display summary statistics of the stock dataset."""

    print("\n" + "=" * 50)
    print("        STOCK DATASET SUMMARY")
    print("=" * 50)

    print(f"Total Stocks          : {stock_data['ticker'].nunique()}")

    print(f"Total Records         : {len(stock_data)}")

    print(
        f"Date Range            : "
        f"{stock_data['date'].min().date()} -> "
        f"{stock_data['date'].max().date()}"
    )

    print(
        f"Average Closing Price : "
        f"{stock_data['close'].mean():.2f}"
    )

    print(
        f"Average Daily Return  : "
        f"{stock_data['daily_return'].mean():.6f}"
    )

    print(
        f"Highest Closing Price : "
        f"{stock_data['close'].max():.2f}"
    )

    print(
        f"Lowest Closing Price  : "
        f"{stock_data['close'].min():.2f}"
    )

    print(
        f"Average Daily Volume  : "
        f"{stock_data['volume'].mean():.0f}"
    )

    print(
        f"Missing Values        : "
        f"{stock_data.isna().sum().sum()}"
    )

    print(
        f"Duplicate Rows        : "
        f"{stock_data.duplicated().sum()}"
    )

    print("=" * 50)