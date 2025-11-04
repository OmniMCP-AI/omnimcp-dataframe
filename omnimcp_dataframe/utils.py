"""Utility functions for DataFrame operations."""

from typing import Optional, Union
import polars as pl


def parse_human_readable_number(value: str) -> Optional[float]:
    """Parse human-readable numbers like '1.1K', '1.3M', '9.6K' to actual numbers.

    Args:
        value: String value that may contain K/M/B/T suffixes (case-insensitive)

    Returns:
        Parsed float value, or None if parsing fails

    Examples:
        '1.1K' -> 1100.0
        '1.3M' -> 1300000.0
        '9.6k' -> 9600.0
        '183' -> 183.0
        '' -> None
    """
    if not value or not isinstance(value, str):
        return None

    # Strip whitespace and convert to lowercase for easier parsing
    value = value.strip().lower()

    if not value:
        return None

    # Define multipliers for common suffixes
    multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "t": 1_000_000_000_000,
    }

    try:
        # Check if the last character is a suffix
        if value[-1] in multipliers:
            suffix = value[-1]
            number_part = value[:-1]

            # Parse the numeric part
            number = float(number_part)

            # Apply the multiplier
            return number * multipliers[suffix]
        else:
            # No suffix, just parse as float
            return float(value)
    except (ValueError, IndexError):
        return None


def safe_column_operation(
    df: pl.DataFrame,
    column: str,
    operation: str,
    value: Optional[Union[str, int, float]] = None
) -> pl.Expr:
    """Safely create a column operation with automatic type conversion.

    Args:
        df: Polars DataFrame
        column: Column name to operate on
        operation: Operation type (eq, ne, gt, lt, gte, lte, contains, etc.)
        value: Value to compare against

    Returns:
        Polars expression for the operation
    """
    col_expr = pl.col(column)

    # Validate column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    # Helper function to cast string columns to numeric for numeric comparisons
    def cast_to_numeric_if_needed(expr: pl.Expr) -> pl.Expr:
        """Cast string columns to numeric types for numeric comparison operators"""
        numeric_ops = ["gt", "lt", "gte", "lte", "between"]

        if operation not in numeric_ops:
            return expr

        try:
            # Get the column's dtype
            col_dtype = df[column].dtype

            # If it's already numeric, no need to cast
            if col_dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            ]:
                return expr

            # If it's a string type, try to cast to float
            if col_dtype in [pl.Utf8, pl.String]:
                # First, try to parse human-readable numbers (1.1K, 1.3M, etc.)
                return expr.map_elements(
                    parse_human_readable_number,
                    return_dtype=pl.Float64,
                )

        except Exception:
            # If conversion fails, use original expression
            pass

        return expr

    # Build filter expression based on operator
    if operation == "is_null":
        return col_expr.is_null()
    elif operation == "not_null":
        return col_expr.is_not_null()
    elif operation == "eq":
        return col_expr == value
    elif operation == "ne":
        return col_expr != value
    elif operation == "gt":
        col_expr = cast_to_numeric_if_needed(col_expr)
        return col_expr > value
    elif operation == "lt":
        col_expr = cast_to_numeric_if_needed(col_expr)
        return col_expr < value
    elif operation == "gte":
        col_expr = cast_to_numeric_if_needed(col_expr)
        return col_expr >= value
    elif operation == "lte":
        col_expr = cast_to_numeric_if_needed(col_expr)
        return col_expr <= value
    elif operation == "contains":
        return col_expr.str.contains(str(value))
    elif operation == "in":
        if not isinstance(value, list):
            raise ValueError(f"Operator 'in' requires a list value, got {type(value)}")
        return col_expr.is_in(value)
    elif operation == "between":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Operator 'between' requires a two-element list, got {value}")
        col_expr = cast_to_numeric_if_needed(col_expr)
        return col_expr.is_between(value[0], value[1])
    elif operation == "regex":
        return col_expr.str.contains(str(value), literal=False)
    else:
        raise ValueError(f"Unsupported operator: {operation}")


def validate_dataframe_input(data: list, min_rows: int = 0) -> pl.DataFrame:
    """Validate and convert input data to Polars DataFrame.

    Args:
        data: Input data (list of dictionaries)
        min_rows: Minimum number of rows required (default: 0)

    Returns:
        Polars DataFrame

    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")

    if len(data) < min_rows:
        raise ValueError(f"Data must have at least {min_rows} rows")

    # Create polars dataframe
    try:
        df = pl.DataFrame(data, strict=False)
    except Exception as e:
        raise ValueError(f"Failed to create DataFrame: {str(e)}")

    return df


def build_error_result(message: str) -> dict:
    """Build a standard error result.

    Args:
        message: Error message

    Returns:
        Dictionary with error information
    """
    return {
        "success": False,
        "message": message,
        "data": []
    }


def build_success_result(data: list, **metadata) -> dict:
    """Build a standard success result.

    Args:
        data: Result data
        **metadata: Additional metadata fields

    Returns:
        Dictionary with success information
    """
    result = {
        "success": True,
        "data": data
    }
    result.update(metadata)
    return result