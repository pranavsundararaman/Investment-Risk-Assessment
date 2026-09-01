import pandas as pd

from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from src.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)

def create_database_engine() -> Engine:
    """
    Create a SQLAlchemy engine for PostgreSQL.

    Returns:
        Engine: SQLAlchemy database engine.
    """

    connection_string = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}"
        f"/{DB_NAME}"
    )

    engine = create_engine(connection_string)

    return engine

def test_database_connection(engine: Engine) -> None:
    """
    Test the PostgreSQL database connection.

    Args:
        engine (Engine): SQLAlchemy engine.
    """

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))

        print(result.scalar())

def insert_stock_data(
    stock_data: pd.DataFrame,
    engine: Engine,
) -> int:
    """
    Insert stock data into the PostgreSQL database.

    Args:
        stock_data (pd.DataFrame): Stock data to insert.
        engine (Engine): SQLAlchemy engine.
    """
    stock_data = stock_data.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Ticker": "ticker",
            "Company Name": "company_name",
            "Industry": "industry",
        }
    )

    columns = [
        "date", "open", "high", "low", "close", "volume", "ticker",
        "company_name", "industry",
    ]
    missing_columns = set(columns) - set(stock_data.columns)
    if missing_columns:
        raise ValueError(
            f"Stock data is missing columns: {', '.join(sorted(missing_columns))}"
        )

    stock_data = stock_data[columns].copy()
    stock_data["date"] = pd.to_datetime(stock_data["date"]).dt.date
    records = stock_data.where(pd.notna(stock_data), None).to_dict("records")

    if not records:
        return 0

    metadata = MetaData()
    stock_prices = Table("stock_prices", metadata, autoload_with=engine)
    statement = insert(stock_prices).values(records)
    statement = statement.on_conflict_do_update(
        index_elements=["ticker", "date"],
        set_={
            column: getattr(statement.excluded, column)
            for column in columns
            if column not in {"ticker", "date"}
        },
    )

    with engine.begin() as connection:
        connection.execute(statement)

    print(f"Upserted {len(records)} rows into stock_prices.")
    return len(records)


def fetch_latest_dates_by_ticker(engine: Engine) -> pd.Series:
    """Return the latest stored trading date for each ticker."""

    query = """
    SELECT ticker, MAX(date) AS date
    FROM stock_prices
    GROUP BY ticker;
    """
    latest_dates = pd.read_sql(query, con=engine)
    latest_dates["date"] = pd.to_datetime(latest_dates["date"])

    return latest_dates.set_index("ticker")["date"]

def fetch_stock_data(
    engine: Engine,
) -> pd.DataFrame:
    """
    Fetch stock data from PostgreSQL.

    Args:
        engine (Engine): SQLAlchemy engine.

    Returns:
        pd.DataFrame: Stock price data.
    """

    query = """
    SELECT
        date,
        open,
        high,
        low,
        close,
        volume,
        ticker,
        company_name,
        industry
    FROM stock_prices
    ORDER BY ticker, date;
    """

    stock_data = pd.read_sql(
        query,
        con=engine,
    )

    stock_data["date"] = pd.to_datetime(
        stock_data["date"]
    )

    return stock_data
