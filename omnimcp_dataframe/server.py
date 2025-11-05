"""
DataFrame Server - Core DataFrame manipulation toolkit.

This module provides a comprehensive set of DataFrame operations built on Polars
for high performance data processing. It's designed to be framework-agnostic
and can be used with MCP servers, REST APIs, or direct Python applications.
"""

import json
import re
from typing import Any, Dict, List, Optional, Callable, Union

import formulas
import polars as pl
import structlog

from .models import (
    DataFrameOperationResult,
    JoinKeyAnalysis,
    FilterCondition,
    AggregationSpec,
    DataFrameConfig,
)
from .prompts import IDENTIFY_JOIN_KEYS_PROMPT
from .utils import (
    parse_human_readable_number,
    safe_column_operation,
    validate_dataframe_input,
    build_error_result,
    build_success_result,
)

logger = structlog.get_logger(__name__)


class DataFrameOperations:
    """Core DataFrame operations class."""

    def __init__(self, config: Optional[DataFrameConfig] = None):
        """Initialize DataFrame operations with optional configuration.

        Args:
            config: Configuration for DataFrame operations
        """
        self.config = config or DataFrameConfig()
        self.logger = structlog.get_logger(__name__)
        self.formula_parser = formulas.Parser()

    async def sort(
        self,
        dataframe: pl.DataFrame,
        by: List[str],
        descending: Optional[List[bool]] = None,
        limit: Optional[int] = None,
    ) -> DataFrameOperationResult:
        """Sort a dataframe by specified columns with optional limit for top-n results.

        Args:
            dataframe: Polars DataFrame
            by: List of column names to sort by
            descending: List of boolean values for sort direction
            limit: Optional limit to return only top N rows

        Returns:
            DataFrameOperationResult with sorted data
        """
        try:
            if dataframe.height == 0:
                return DataFrameOperationResult(data=[], success=True, shape="(0, 0)", data_df=pl.DataFrame())

            df = dataframe

            # Validate columns exist
            missing_columns = [col for col in by if col not in df.columns]
            if missing_columns:
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message=f"Columns not found: {missing_columns}"
                )

            # Convert string columns to numeric for sorting
            for col in by:
                if df[col].dtype in [pl.Utf8, pl.String]:
                    try:
                        df = df.with_columns(
                            pl.col(col)
                            .map_elements(parse_human_readable_number, return_dtype=pl.Float64)
                            .alias(col)
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not convert column '{col}' to numeric: {e}")

            # Process descending parameter
            if descending is None:
                descending = [False] * len(by)
            elif not isinstance(descending, list):
                descending = [descending]
            elif len(descending) != len(by):
                descending = descending + [False] * (len(by) - len(descending))

            # Sort and optionally limit
            sorted_df = df.sort(by=by, descending=descending)
            if limit is not None and limit > 0:
                sorted_df = sorted_df.head(limit)

            result_data = sorted_df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                input_rows=df.height,
                output_rows=sorted_df.height,
                shape=f"({sorted_df.height}, {sorted_df.width})",
                data_df=sorted_df
            )

        except Exception as e:
            self.logger.error(f"Error sorting dataframe: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error sorting dataframe: {str(e)}"
            )

    async def filter(
        self,
        dataframe: pl.DataFrame,
        conditions: List[Dict[str, Any]],
        logic: str = "AND",
    ) -> DataFrameOperationResult:
        """Filter a dataframe based on single or multiple conditions.

        Args:
            dataframe: Polars DataFrame
            conditions: Array of condition objects
            logic: Logic operator for combining conditions (AND/OR)

        Returns:
            DataFrameOperationResult with filtered data
        """
        try:
            if dataframe.height == 0:
                return DataFrameOperationResult(
                    data=[],
                    success=True,
                    input_rows=0,
                    output_rows=0,
                    data_df=pl.DataFrame()
                )

            df = dataframe
            # Build filter expressions
            expressions = []
            for cond in conditions:
                column = cond.get("column")
                op = cond.get("op")
                value = cond.get("value")

                if not column or not op:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message="Each condition must have 'column' and 'op' fields"
                    )

                expr = safe_column_operation(df, column, op, value)
                expressions.append(expr)

            # Combine expressions with logic
            if logic == "AND":
                filter_expr = expressions[0]
                for expr in expressions[1:]:
                    filter_expr = filter_expr & expr
            elif logic == "OR":
                filter_expr = expressions[0]
                for expr in expressions[1:]:
                    filter_expr = filter_expr | expr
            else:
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message=f"Invalid logic operator '{logic}'. Must be 'AND' or 'OR'"
                )

            # Apply filter
            filtered_df = df.filter(filter_expr)
            result_data = filtered_df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                input_rows=df.height,
                output_rows=filtered_df.height,
                data_df=filtered_df
            )

        except Exception as e:
            self.logger.error(f"Error filtering dataframe: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error filtering dataframe: {str(e)}"
            )

    async def concat(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        drop_duplicates: bool = False,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> DataFrameOperationResult:
        """Concatenate two dataframes with optional duplicate removal.

        Args:
            left: Left Polars DataFrame
            right: Right Polars DataFrame
            drop_duplicates: Whether to drop duplicates
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep

        Returns:
            DataFrameOperationResult with concatenated data
        """
        try:
            # Use input DataFrames directly
            pl_dataframes = []
            if left.height > 0:
                pl_dataframes.append(left)
            if right.height > 0:
                pl_dataframes.append(right)

            if not pl_dataframes:
                return DataFrameOperationResult(
                    data=[],
                    success=True,
                    total_rows=0,
                    duplicates_removed=0,
                    data_df=pl.DataFrame()
                )

            # Concatenate
            concatenated_df = pl.concat(pl_dataframes, how="vertical_relaxed")
            rows_before = concatenated_df.height

            # Remove duplicates if requested
            duplicates_removed = 0
            if drop_duplicates:
                keep_option = None if keep == "none" else keep

                if subset:
                    # Validate subset columns exist
                    missing_cols = [col for col in subset if col not in concatenated_df.columns]
                    if missing_cols:
                        return DataFrameOperationResult(
                            data=[],
                            success=False,
                            message=f"Subset columns not found: {missing_cols}"
                        )
                    concatenated_df = concatenated_df.unique(subset=subset, keep=keep_option)
                else:
                    concatenated_df = concatenated_df.unique(keep=keep_option)

                duplicates_removed = rows_before - concatenated_df.height

            result_data = concatenated_df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                total_rows=concatenated_df.height,
                duplicates_removed=duplicates_removed,
                data_df=concatenated_df
            )

        except Exception as e:
            self.logger.error(f"Error concatenating dataframes: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error concatenating dataframes: {str(e)}"
            )

    async def merge(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None,
        how: str = "inner",
    ) -> DataFrameOperationResult:
        """Merge two dataframes on specified columns.

        Args:
            left: Left Polars DataFrame
            right: Right Polars DataFrame
            on: Column(s) to join on (when column names match)
            left_on: Column(s) from left dataframe
            right_on: Column(s) from right dataframe
            how: Join type (inner, left, outer, cross)

        Returns:
            DataFrameOperationResult with merged data
        """
        try:
            if left.height == 0 or right.height == 0:
                return DataFrameOperationResult(
                    data=[],
                    success=True,
                    total_rows=0,
                    data_df=pl.DataFrame()
                )

            # Validate join parameters
            if not on and not (left_on and right_on):
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message="Must specify either 'on' or both 'left_on' and 'right_on'"
                )

            if on and (left_on or right_on):
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message="Cannot specify both 'on' and 'left_on'/'right_on'"
                )

            # Convert lists to strings if needed
            if on and isinstance(on, str):
                on = [on]
            if left_on and isinstance(left_on, str):
                left_on = [left_on]
            if right_on and isinstance(right_on, str):
                right_on = [right_on]

            # Use input DataFrames directly
            left_df = left
            right_df = right

            # Validate join columns exist
            if on:
                missing_left = [col for col in on if col not in left_df.columns]
                missing_right = [col for col in on if col not in right_df.columns]
                if missing_left:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Join columns not found in left dataframe: {missing_left}"
                    )
                if missing_right:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Join columns not found in right dataframe: {missing_right}"
                    )
            else:
                missing_left = [col for col in (left_on or []) if col not in left_df.columns]
                missing_right = [col for col in (right_on or []) if col not in right_df.columns]
                if missing_left or missing_right:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Join columns not found: {missing_left + missing_right}"
                    )

            # Validate join type
            valid_joins = ["inner", "left", "outer", "cross"]
            if how not in valid_joins:
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message=f"Invalid join type '{how}'. Must be one of: {valid_joins}"
                )

            # Perform merge
            if on:
                merged_df = left_df.join(right_df, on=on, how=how)
            else:
                merged_df = left_df.join(right_df, left_on=left_on, right_on=right_on, how=how)

            result_data = merged_df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                total_rows=merged_df.height,
                left_rows=left_df.height,
                right_rows=right_df.height,
                data_df=merged_df
            )

        except Exception as e:
            self.logger.error(f"Error merging dataframes: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error merging dataframes: {str(e)}"
            )

    async def group_by(
        self,
        dataframe: pl.DataFrame,
        by: List[str],
        aggregations: List[Dict[str, Any]],
    ) -> DataFrameOperationResult:
        """Group dataframe by specified columns and apply aggregation functions.

        Args:
            dataframe: Polars DataFrame
            by: List of column names to group by
            aggregations: List of aggregation specifications

        Returns:
            DataFrameOperationResult with grouped and aggregated data
        """
        try:
            if dataframe.height == 0:
                return DataFrameOperationResult(
                    data=[],
                    success=True,
                    input_rows=0,
                    output_rows=0,
                    group_columns=by,
                    aggregated_columns=[],
                    data_df=pl.DataFrame()
                )

            df = dataframe
            # Validate group columns exist
            missing_columns = [col for col in by if col not in df.columns]
            if missing_columns:
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message=f"Group by columns not found: {missing_columns}"
                )

            # Convert string columns to numeric for aggregation
            agg_columns_set = {agg["column"] for agg in aggregations}
            numeric_only_functions = {"sum", "mean", "median", "std", "var"}

            for col in agg_columns_set:
                if col not in df.columns:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Aggregation column '{col}' not found"
                    )

                # Check if any numeric-only functions are applied to this column
                col_functions = [agg["function"] for agg in aggregations if agg["column"] == col]
                needs_numeric = any(func in numeric_only_functions for func in col_functions)

                if df[col].dtype in [pl.Utf8, pl.String] and needs_numeric:
                    try:
                        df = df.with_columns(
                            pl.col(col)
                            .map_elements(parse_human_readable_number, return_dtype=pl.Float64)
                            .alias(col)
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not convert column '{col}' to numeric: {e}")

            # Start grouping
            grouped = df.group_by(by, maintain_order=True)

            # Build aggregation expressions
            agg_exprs = []
            aggregated_column_names = []

            for agg in aggregations:
                column = agg["column"]
                function = agg["function"]
                output_name = f"{column}_{function}"
                aggregated_column_names.append(output_name)

                col_expr = pl.col(column)

                if function == "count":
                    agg_exprs.append(col_expr.count().alias(output_name))
                elif function == "sum":
                    agg_exprs.append(col_expr.sum().alias(output_name))
                elif function == "mean":
                    agg_exprs.append(col_expr.mean().alias(output_name))
                elif function == "min":
                    agg_exprs.append(col_expr.min().alias(output_name))
                elif function == "max":
                    agg_exprs.append(col_expr.max().alias(output_name))
                elif function == "median":
                    agg_exprs.append(col_expr.median().alias(output_name))
                elif function == "first":
                    agg_exprs.append(col_expr.first().alias(output_name))
                elif function == "last":
                    agg_exprs.append(col_expr.last().alias(output_name))
                elif function == "std":
                    agg_exprs.append(col_expr.std().alias(output_name))
                elif function == "var":
                    agg_exprs.append(col_expr.var().alias(output_name))
                elif function == "count_distinct":
                    agg_exprs.append(col_expr.n_unique().alias(output_name))
                else:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Unsupported aggregation function: {function}"
                    )

            # Apply aggregations
            result_df = grouped.agg(agg_exprs)
            result_data = result_df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                input_rows=df.height,
                output_rows=result_df.height,
                group_columns=by,
                aggregated_columns=aggregated_column_names,
                data_df=result_df
            )

        except Exception as e:
            self.logger.error(f"Error in group_by operation: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error in group_by operation: {str(e)}"
            )

    async def apply_formula(
        self,
        dataframe: pl.DataFrame,
        formula: str,
        column_name: str,
        use_excel_refs: bool = False,
        target_columns: Optional[Dict[str, str]] = None,
    ) -> DataFrameOperationResult:
        """Apply Excel-like formulas to dataframe columns.

        Args:
            dataframe: Polars DataFrame
            formula: Formula string
            column_name: Name of the column to store results
            use_excel_refs: Whether to use Excel cell references
            target_columns: Mapping of Excel letters to column names

        Returns:
            DataFrameOperationResult with formula applied
        """
        try:
            if dataframe.height == 0:
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message="Dataframe cannot be empty"
                )

            df = dataframe

            # Simple mode: use column names directly
            if not use_excel_refs:
                clean_formula = formula.lstrip("=")

                try:
                    # Parse the formula
                    ast_tuple = self.formula_parser.ast(f"={clean_formula}")
                    ast_node, builder = ast_tuple
                    compiled = builder.compile()

                    # Apply formula to each row
                    results = []
                    for row in df.to_dicts():
                        try:
                            # Extract variables used in formula
                            formula_vars = set()
                            for col in df.columns:
                                if col.upper() in clean_formula.upper():
                                    formula_vars.add(col.upper())

                            # Build context with only needed variables
                            context_vars = {}
                            for col in df.columns:
                                if col.upper() in formula_vars:
                                    context_vars[col.upper()] = row[col]

                            # Evaluate formula
                            result = compiled(**context_vars)

                            # Convert to Python native type
                            if hasattr(result, "tolist"):
                                value = result.tolist()
                            elif hasattr(result, "item"):
                                value = result.item()
                            else:
                                value = result

                            results.append(value)

                        except Exception as e:
                            results.append(None)
                            self.logger.warning(f"Formula evaluation failed: {e}")

                    # Add new column
                    df = df.with_columns(pl.Series(column_name, results, strict=False))

                except Exception as e:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Invalid formula: {str(e)}"
                    )

            # Excel mode: use A1, B1 cell references
            else:
                # Create mapping from Excel letters to dataframe columns
                excel_columns = {}
                for i, col in enumerate(df.columns):
                    excel_letter = chr(65 + i)  # A, B, C, ...
                    excel_columns[excel_letter] = col

                # Override with user mapping
                if target_columns:
                    excel_columns.update(target_columns)

                # Ensure formula starts with '='
                if not formula.startswith("="):
                    formula = "=" + formula

                # Apply formula row by row
                results = []
                for row_idx, row in enumerate(df.to_dicts()):
                    try:
                        # Create row-specific formula
                        row_formula = formula
                        temp_vars = {}

                        # Replace Excel references with variables
                        for excel_col, data_col in excel_columns.items():
                            if data_col in row:
                                cell_ref = f"{excel_col}1"
                                if cell_ref in formula:
                                    var_name = f"VAR_{excel_col}"
                                    temp_vars[var_name] = row[data_col]
                                    row_formula = row_formula.replace(cell_ref, var_name)

                        # Parse and evaluate
                        ast_tuple = self.formula_parser.ast(row_formula)
                        ast_node, builder = ast_tuple
                        compiled = builder.compile()

                        result = compiled(**temp_vars)

                        # Convert to Python native type
                        if hasattr(result, "tolist"):
                            value = result.tolist()
                        elif hasattr(result, "item"):
                            value = result.item()
                        else:
                            value = result

                        results.append(value)

                    except Exception as e:
                        results.append(None)
                        self.logger.warning(f"Formula evaluation failed for row {row_idx}: {e}")

                # Add new column
                df = df.with_columns(pl.Series(column_name, results, strict=False))

            result_data = df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                shape=f"({df.height}, {df.width})",
                columns=df.columns,
                data_df=df,
                **{
                    "formula_applied": formula,
                    "result_column": column_name
                }
            )

        except Exception as e:
            self.logger.error(f"Error applying Excel formula: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error applying Excel formula: {str(e)}"
            )

    async def drop_duplicates(
        self,
        dataframe: pl.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> DataFrameOperationResult:
        """Remove duplicate rows from a dataframe.

        Args:
            dataframe: Polars DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep

        Returns:
            DataFrameOperationResult with duplicates removed
        """
        try:
            if dataframe.height == 0:
                return DataFrameOperationResult(
                    data=[],
                    success=True,
                    input_rows=0,
                    output_rows=0,
                    duplicates_removed=0,
                    data_df=pl.DataFrame()
                )

            df = dataframe
            input_rows = df.height

            # Validate subset columns exist
            if subset:
                missing_columns = [col for col in subset if col not in df.columns]
                if missing_columns:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Subset columns not found: {missing_columns}"
                    )

            # Apply unique operation
            if subset:
                deduplicated_df = df.unique(subset=subset, keep=keep, maintain_order=True)
            else:
                deduplicated_df = df.unique(keep=keep, maintain_order=True)

            output_rows = deduplicated_df.height
            duplicates_removed = input_rows - output_rows

            result_data = deduplicated_df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                input_rows=input_rows,
                output_rows=output_rows,
                duplicates_removed=duplicates_removed,
                data_df=deduplicated_df
            )

        except Exception as e:
            self.logger.error(f"Error removing duplicates: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error removing duplicates: {str(e)}"
            )

    async def explode(
        self,
        dataframe: pl.DataFrame,
        column: str,
    ) -> DataFrameOperationResult:
        """Explode a list/array column to long format by creating a row for each list element.

        Args:
            dataframe: Polars DataFrame
            column: Column name to explode (must contain List or Array type)

        Returns:
            DataFrameOperationResult with exploded data
        """
        try:
            if dataframe.height == 0:
                return DataFrameOperationResult(
                    data=[],
                    success=True,
                    input_rows=0,
                    output_rows=0,
                    data_df=pl.DataFrame()
                )

            df = dataframe
            input_rows = df.height

            # Validate column exists
            if column not in df.columns:
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message=f"Column not found: {column}"
                )

            # Validate that column is List or Array type
            dtype = df[column].dtype
            if not (isinstance(dtype, pl.List) or isinstance(dtype, pl.Array)):
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message=f"Column must be List or Array type. Column '{column}' is {dtype}"
                )

            # Explode the specified column
            exploded_df = df.explode(column)
            output_rows = exploded_df.height

            result_data = exploded_df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                input_rows=input_rows,
                output_rows=output_rows,
                shape=f"({exploded_df.height}, {exploded_df.width})",
                data_df=exploded_df
            )

        except Exception as e:
            self.logger.error(f"Error exploding dataframe: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error exploding dataframe: {str(e)}"
            )

    async def init_dataframe(self, dataframe: Union[List[Dict[str, Any]], str]) -> DataFrameOperationResult:
        """Initialize a dataframe from a list of dictionaries or JSON string.

        Args:
            dataframe: List of dict objects or JSON string

        Returns:
            DataFrameOperationResult with initialized dataframe
        """
        try:
            # Handle JSON string input
            if isinstance(dataframe, str):
                try:
                    data = json.loads(dataframe)
                    self.logger.info("Successfully parsed dataframe from JSON string")
                except json.JSONDecodeError as e:
                    return DataFrameOperationResult(
                        data=[],
                        success=False,
                        message=f"Invalid JSON string: {str(e)}"
                    )
            else:
                data = dataframe

            if not isinstance(data, list):
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message="Data must be an array of objects or JSON string"
                )

            df = pl.DataFrame(data)
            shape = f"({df.height}, {df.width})"
            result_data = df.to_dicts()

            return DataFrameOperationResult(
                data=result_data,
                success=True,
                shape=shape,
                input_rows=df.height,
                output_rows=df.height,
                data_df=df
            )

        except Exception as e:
            self.logger.error(f"Error initializing dataframe: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error initializing dataframe: {str(e)}"
            )


class DataFrameServer:
    """Main DataFrame server class providing MCP-compatible interface."""

    def __init__(self, config: Optional[DataFrameConfig] = None):
        """Initialize DataFrame server.

        Args:
            config: Configuration for DataFrame operations
        """
        self.config = config or DataFrameConfig()
        self.operations = DataFrameOperations(config)
        self.logger = structlog.get_logger(__name__)

        # Tool mapping for MCP compatibility
        self.tool_mapping = {
            "sort": self.operations.sort,
            "filter": self.operations.filter,
            "concat": self.operations.concat,
            "merge": self.operations.merge,
            "group_by": self.operations.group_by,
            "apply_formula": self.operations.apply_formula,
            "drop_duplicates": self.operations.drop_duplicates,
            "explode": self.operations.explode,
            "init": self.operations.init_dataframe,
        }

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available DataFrame tools.

        Returns:
            List of tool definitions compatible with MCP
        """
        return [
            {
                "name": "sort",
                "description": "Sort a dataframe by specified columns with optional limit for top-n results",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataframe": {"type": "array", "items": {"type": "object"}},
                        "by": {"type": "array", "items": {"type": "string"}},
                        "descending": {"type": "array", "items": {"type": "boolean"}},
                        "limit": {"type": "integer"},
                    },
                    "required": ["dataframe", "by"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "input_rows": {"type": "integer"},
                        "output_rows": {"type": "integer"},
                        "shape": {"type": "string"},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "filter",
                "description": "Filter a dataframe based on single or multiple conditions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataframe": {"type": "array", "items": {"type": "object"}},
                        "conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "op": {"type": "string", "enum": ["eq", "ne", "gt", "lt", "gte", "lte", "contains", "in", "between", "is_null", "not_null", "regex"]},
                                    "value": {},
                                },
                                "required": ["column", "op"],
                            },
                        },
                        "logic": {"type": "string", "enum": ["AND", "OR"]},
                    },
                    "required": ["dataframe", "conditions"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "input_rows": {"type": "integer"},
                        "output_rows": {"type": "integer"},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "concat",
                "description": "Concatenate two dataframes with optional duplicate removal",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "array", "items": {"type": "object"}},
                        "right": {"type": "array", "items": {"type": "object"}},
                        "drop_duplicates": {"type": "boolean"},
                        "subset": {"type": "array", "items": {"type": "string"}},
                        "keep": {"type": "string", "enum": ["first", "last", "none"]},
                    },
                    "required": ["left", "right"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "total_rows": {"type": "integer"},
                        "duplicates_removed": {"type": "integer"},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "merge",
                "description": "Merge two dataframes on specified columns with different join strategies",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "array", "items": {"type": "object"}},
                        "right": {"type": "array", "items": {"type": "object"}},
                        "on": {"type": "array", "items": {"type": "string"}},
                        "left_on": {"type": "array", "items": {"type": "string"}},
                        "right_on": {"type": "array", "items": {"type": "string"}},
                        "how": {"type": "string", "enum": ["inner", "left", "outer", "cross"]},
                    },
                    "required": ["left", "right"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "total_rows": {"type": "integer"},
                        "left_rows": {"type": "integer"},
                        "right_rows": {"type": "integer"},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "group_by",
                "description": "Group dataframe by specified columns and apply aggregation functions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataframe": {"type": "array", "items": {"type": "object"}},
                        "by": {"type": "array", "items": {"type": "string"}},
                        "aggregations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "function": {"type": "string", "enum": ["count", "sum", "mean", "min", "max", "median", "first", "last", "std", "var", "count_distinct"]},
                                },
                                "required": ["column", "function"],
                            },
                        },
                    },
                    "required": ["dataframe", "by", "aggregations"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "input_rows": {"type": "integer"},
                        "output_rows": {"type": "integer"},
                        "group_columns": {"type": "array", "items": {"type": "string"}},
                        "aggregated_columns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "apply_formula",
                "description": "Apply Excel-like formulas to dataframe columns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataframe": {"type": "array", "items": {"type": "object"}},
                        "formula": {"type": "string"},
                        "column_name": {"type": "string"},
                        "use_excel_refs": {"type": "boolean"},
                        "target_columns": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["dataframe", "formula", "column_name"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "shape": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "formula_applied": {"type": "string"},
                        "result_column": {"type": "string"},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "drop_duplicates",
                "description": "Remove duplicate rows from a dataframe based on specified columns",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataframe": {"type": "array", "items": {"type": "object"}},
                        "subset": {"type": "array", "items": {"type": "string"}},
                        "keep": {"type": "string", "enum": ["first", "last", "none"]},
                    },
                    "required": ["dataframe"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "input_rows": {"type": "integer"},
                        "output_rows": {"type": "integer"},
                        "duplicates_removed": {"type": "integer"},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "explode",
                "description": "Explode a list/array column to long format by creating a row for each list element",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataframe": {"type": "array", "items": {"type": "object"}},
                        "column": {"type": "string", "description": "Column name to explode (must contain List or Array type)"},
                    },
                    "required": ["dataframe", "column"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "input_rows": {"type": "integer"},
                        "output_rows": {"type": "integer"},
                        "shape": {"type": "string"},
                    },
                    "required": ["data", "success"],
                },
            },
            {
                "name": "init",
                "description": "Initialize a dataframe from a list of dictionaries or a JSON string",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array", "items": {"type": "object"}},
                                {"type": "string"},
                            ],
                        },
                    },
                    "required": ["dataframe"],
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "object"}},
                        "success": {"type": "boolean"},
                        "shape": {"type": "string"},
                    },
                    "required": ["data", "success"],
                },
            },
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DataFrameOperationResult:
        """Call a specific DataFrame tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            headers: Optional headers (not used in this implementation)

        Returns:
            DataFrameOperationResult with tool execution results
        """
        if arguments is None:
            arguments = {}

        try:
            tool_func = self.tool_mapping.get(tool_name)
            if tool_func:
                # For async compatibility, we just call the sync function
                result = tool_func(**arguments)
                return result
            else:
                return DataFrameOperationResult(
                    data=[],
                    success=False,
                    message=f"Unknown tool: {tool_name}"
                )
        except Exception as e:
            self.logger.error(f"Error executing {tool_name}: {str(e)}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error executing {tool_name}: {str(e)}"
            )


# Convenience class for easier usage
class DataFrameToolkit:
    """High-level toolkit for DataFrame operations."""

    def __init__(self, config: Optional[DataFrameConfig] = None):
        """Initialize toolkit.

        Args:
            config: Optional configuration
        """
        self.server = DataFrameServer(config)

    def _convert_to_dataframe(self, data: Union[List[Dict[str, Any]], pl.DataFrame]) -> pl.DataFrame:
        """Convert list of dictionaries to polars DataFrame if needed.

        Args:
            data: Input data as list of dicts or polars DataFrame

        Returns:
            Polars DataFrame
        """
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, list):
            return validate_dataframe_input(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}. Expected list or polars.DataFrame.")

    async def sort(self, dataframe: Union[List[Dict[str, Any]], pl.DataFrame], by: List[str], **kwargs) -> DataFrameOperationResult:
        """Sort dataframe."""
        df = self._convert_to_dataframe(dataframe)
        return await self.server.operations.sort(df, by, **kwargs)

    async def filter(self, dataframe: Union[List[Dict[str, Any]], pl.DataFrame], conditions: List[Dict[str, Any]], **kwargs) -> DataFrameOperationResult:
        """Filter dataframe."""
        df = self._convert_to_dataframe(dataframe)
        return await self.server.operations.filter(df, conditions, **kwargs)

    async def concat(self, left: Union[List[Dict[str, Any]], pl.DataFrame], right: Union[List[Dict[str, Any]], pl.DataFrame], **kwargs) -> DataFrameOperationResult:
        """Concatenate dataframes."""
        left_df = self._convert_to_dataframe(left)
        right_df = self._convert_to_dataframe(right)
        return await self.server.operations.concat(left_df, right_df, **kwargs)

    async def merge(self, left: Union[List[Dict[str, Any]], pl.DataFrame], right: Union[List[Dict[str, Any]], pl.DataFrame], **kwargs) -> DataFrameOperationResult:
        """Merge dataframes."""
        left_df = self._convert_to_dataframe(left)
        right_df = self._convert_to_dataframe(right)
        return await self.server.operations.merge(left_df, right_df, **kwargs)

    async def group_by(self, dataframe: Union[List[Dict[str, Any]], pl.DataFrame], by: List[str], aggregations: List[Dict[str, Any]], **kwargs) -> DataFrameOperationResult:
        """Group and aggregate dataframe."""
        df = self._convert_to_dataframe(dataframe)
        return await self.server.operations.group_by(df, by, aggregations, **kwargs)

    async def apply_formula(self, dataframe: Union[List[Dict[str, Any]], pl.DataFrame], formula: str, column_name: str, **kwargs) -> DataFrameOperationResult:
        """Apply formula to dataframe."""
        df = self._convert_to_dataframe(dataframe)
        return await self.server.operations.apply_formula(df, formula, column_name, **kwargs)

    async def drop_duplicates(self, dataframe: Union[List[Dict[str, Any]], pl.DataFrame], **kwargs) -> DataFrameOperationResult:
        """Remove duplicates from dataframe."""
        df = self._convert_to_dataframe(dataframe)
        return await self.server.operations.drop_duplicates(df, **kwargs)

    async def explode(self, dataframe: Union[List[Dict[str, Any]], pl.DataFrame], column: str, **kwargs) -> DataFrameOperationResult:
        """Explode list/array column in dataframe."""
        df = self._convert_to_dataframe(dataframe)
        return await self.server.operations.explode(df, column, **kwargs)

    async def init(self, dataframe: Union[List[Dict[str, Any]], str]) -> DataFrameOperationResult:
        """Initialize dataframe."""
        return await self.server.operations.init_dataframe(dataframe)