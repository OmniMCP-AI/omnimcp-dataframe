"""Data models for DataFrame operations."""

from typing import Any, Dict, List, Optional
import polars as pl
from pydantic import BaseModel, Field


class JoinKeyAnalysis(BaseModel):
    """Analysis result for identifying join keys between dataframes."""

    can_join: bool = Field(description="Whether the dataframes can be joined")
    left_columns: List[str] = Field(description="Columns from left dataframe suitable for joining")
    right_columns: List[str] = Field(description="Columns from right dataframe suitable for joining")
    confidence: float = Field(description="Confidence score of the join key identification (0.0-1.0)")
    reason: str = Field(description="Explanation of why these columns were chosen")


class DataFrameOperationResult(BaseModel):
    """Standard result format for DataFrame operations."""

    data: List[Dict[str, Any]] = Field(description="Resulting dataframe data")
    success: bool = Field(description="Whether the operation was successful")
    message: Optional[str] = Field(default=None, description="Optional message or error information")

    # Operation-specific metadata
    input_rows: Optional[int] = Field(default=None, description="Number of rows in input dataframe")
    output_rows: Optional[int] = Field(default=None, description="Number of rows in output dataframe")
    left_rows: Optional[int] = Field(default=None, description="Number of rows in left columns")
    right_rows: Optional[int] = Field(default=None, description="Number of rows in right columns")

    shape: Optional[str] = Field(default=None, description="Shape of resulting dataframe '(rows, columns)'")

    # Operation-specific fields
    total_rows: Optional[int] = Field(default=None, description="Total rows after operation")
    duplicates_removed: Optional[int] = Field(default=None, description="Number of duplicates removed")
    columns: Optional[List[str]] = Field(default=None, description="Column names in result")
    group_columns: Optional[List[str]] = Field(default=None, description="Columns used for grouping")
    aggregated_columns: Optional[List[str]] = Field(default=None, description="Aggregated column names")
    left_columns: Optional[List[str]] = Field(default=None, description="Left columns used in join")
    right_columns: Optional[List[str]] = Field(default=None, description="Right columns used in join")
    confidence: Optional[float] = Field(default=None, description="Confidence score for fuzzy operations")

    # Explode-specific fields
    json_conversion_applied: Optional[bool] = Field(default=None, description="Whether JSON string conversion was applied")
    exploded_column: Optional[str] = Field(default=None, description="Column that was exploded")
    extracted_fields: Optional[List[str]] = Field(default=None, description="Fields extracted from struct objects")
    original_column_dropped: Optional[bool] = Field(default=None, description="Whether original column was dropped after field extraction")
    note: Optional[str] = Field(default=None, description="Additional notes about the operation")

    # DataFrame object field (excluded from JSON serialization but available in Python)
    data_df: Optional[pl.DataFrame] = Field(default=pl.DataFrame(), description="Resulting dataframe as Polars DataFrame", exclude=False)

    class Config:
        arbitrary_types_allowed = True  # Allow Polars DataFrame type


class FilterCondition(BaseModel):
    """Individual filter condition for DataFrame filtering."""

    column: str = Field(description="Column name to filter on")
    op: str = Field(description="Filter operator")
    value: Any = Field(default=None, description="Value to compare against")


class AggregationSpec(BaseModel):
    """Aggregation specification for group_by operations."""

    column: str = Field(description="Column name to aggregate")
    function: str = Field(description="Aggregation function")


class DataFrameConfig(BaseModel):
    """Configuration for DataFrame operations."""

    # Performance settings
    max_memory_mb: Optional[int] = Field(default=1024, description="Maximum memory usage in MB")
    chunk_size: Optional[int] = Field(default=10000, description="Chunk size for large operations")

    # LLM settings for intelligent operations
    llm_model: Optional[str] = Field(default="openai/gpt-4o-mini", description="LLM model for intelligent operations")
    llm_temperature: Optional[float] = Field(default=0.0, description="LLM temperature for intelligent operations")
    llm_max_tokens: Optional[int] = Field(default=1000, description="LLM max tokens for intelligent operations")

    # Fallback settings
    enable_fuzzy_matching: bool = Field(default=True, description="Enable fuzzy matching for joins")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence for intelligent operations")