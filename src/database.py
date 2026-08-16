import pandas as pd

from sqlalchemy import create_engine, text
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
) -> None:
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

    stock_data.to_sql(
        name="stock_prices",
        con=engine,
        if_exists="append",
        index=False,
    )

    print(f"Inserted {len(stock_data)} rows into stock_prices.")

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