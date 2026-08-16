import pandas as pd

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)


from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)


def validate_data_types(
    stock_data: pd.DataFrame,
) -> None:
    """
    Validate data types of the stock dataset.

    Args:
        stock_data (pd.DataFrame): Stock dataset.
    """

    if not is_datetime64_any_dtype(stock_data["date"]):
        raise TypeError("date must be a datetime type")

    if not is_float_dtype(stock_data["open"]):
        raise TypeError("open must be float")

    if not is_float_dtype(stock_data["high"]):
        raise TypeError("high must be float")

    if not is_float_dtype(stock_data["low"]):
        raise TypeError("low must be float")

    if not is_float_dtype(stock_data["close"]):
        raise TypeError("close must be float")

    if not is_integer_dtype(stock_data["volume"]):
        raise TypeError("volume must be integer")

    if not is_string_dtype(stock_data["ticker"]):
        raise TypeError("ticker must be string")

    if not is_string_dtype(stock_data["company_name"]):
        raise TypeError("company_name must be string")

    if not is_string_dtype(stock_data["industry"]):
        raise TypeError("industry must be string")

def check_missing_values(
    stock_data: pd.DataFrame,
) -> None:
    """
    Check for missing values in the dataset.

    Args:
        stock_data (pd.DataFrame): Stock dataset.

    Raises:
        ValueError: If missing values are found.
    """

    missing = stock_data.isnull().sum()

    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            f"Missing values found:\n{missing}"
        )

def check_duplicate_rows(
    stock_data: pd.DataFrame,
) -> None:
    """
    Check for duplicate rows.

    Args:
        stock_data (pd.DataFrame): Stock dataset.

    Raises:
        ValueError: If duplicate rows are found.
    """

    duplicates = stock_data.duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Found {duplicates} duplicate rows."
        )

def sort_stock_data(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort stock data by ticker and date.

    Args:
        stock_data (pd.DataFrame): Stock dataset.

    Returns:
        pd.DataFrame: Sorted dataset.
    """

    stock_data = stock_data.sort_values(
        by=["ticker", "date"]
    )

    stock_data = stock_data.reset_index(
        drop=True
    )

    return stock_data

def validate_stock_history(
    stock_data: pd.DataFrame,
) -> None:
    """
    Validate that every stock has historical data.

    Args:
        stock_data (pd.DataFrame): Stock dataset.

    Raises:
        ValueError: If any stock has no records.
    """

    counts = stock_data.groupby(
        "ticker"
    ).size()

    empty = counts[counts == 0]

    if not empty.empty:
        raise ValueError(
            "Some stocks have no historical data."
        )

def handle_missing_values(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        stock_data (pd.DataFrame): Stock dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    return stock_data

def preprocess_stock_data(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Args:
        stock_data (pd.DataFrame): Raw stock dataset.

    Returns:
        pd.DataFrame: Preprocessed stock dataset.
    """

    validate_data_types(stock_data)

    check_missing_values(stock_data)

    check_duplicate_rows(stock_data)

    stock_data = sort_stock_data(stock_data)

    validate_stock_history(stock_data)

    stock_data = handle_missing_values(stock_data)

    return stock_data