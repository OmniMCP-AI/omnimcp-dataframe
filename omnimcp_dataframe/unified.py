"""
Unified DataFrame Toolkit Interface

Provides a single entry point for calling DataFrame operations by name and arguments.
Supports both list-of-dicts and polars.DataFrame input types.
"""

from typing import Any, Dict, List, Optional, Union
import polars as pl
from .models import DataFrameConfig, DataFrameOperationResult
from .server import DataFrameToolkit
import structlog

logger = structlog.get_logger(__name__)


class UnifiedDataFrameToolkit:
    """
    Unified interface for DataFrame operations.

    Provides a single `call` method that accepts a tool name and arguments,
    routing to the appropriate DataFrame operation. Supports both list-of-dicts
    and polars.DataFrame input types.
    """

    def __init__(self, config: Optional[DataFrameConfig] = None):
        """Initialize the unified toolkit.

        Args:
            config: Optional configuration for DataFrame operations
        """
        self.config = config or DataFrameConfig()
        self.toolkit = DataFrameToolkit(config)
        self.logger = structlog.get_logger(__name__)

        # Map tool names to their methods
        self._tool_registry = {
            "sort": self._call_sort,
            "filter": self._call_filter,
            "concat": self._call_concat,
            "merge": self._call_merge,
            "group_by": self._call_group_by,
            "apply_formula": self._call_apply_formula,
            "drop_duplicates": self._call_drop_duplicates,
            "explode": self._call_explode,
            "init": self._call_init,
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools with their schemas.

        Returns:
            List of tool dictionaries with name, description, and parameter schema
        """
        tools = [
            {
                "name": "sort",
                "description": "Sort dataframe by specified columns with optional limit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "DataFrame as list of dicts or polars DataFrame"
                        },
                        "by": {"type": "array", "items": {"type": "string"}, "description": "List of column names to sort by"},
                        "descending": {"type": "array", "items": {"type": "boolean"}, "description": "List of sort directions"},
                        "limit": {"type": "integer", "description": "Optional limit to return only top N rows"}
                    },
                    "required": ["dataframe", "by"]
                }
            },
            {
                "name": "filter",
                "description": "Filter dataframe based on conditions with AND/OR logic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "DataFrame as list of dicts or polars DataFrame"
                        },
                        "conditions": {"type": "array", "description": "List of filter conditions"},
                        "logic": {"type": "string", "enum": ["and", "or"], "default": "and", "description": "Logic for combining conditions"}
                    },
                    "required": ["dataframe", "conditions"]
                }
            },
            {
                "name": "concat",
                "description": "Concatenate two dataframes with optional duplicate removal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "First dataframe as list of dicts or polars DataFrame"
                        },
                        "right": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "Second dataframe as list of dicts or polars DataFrame"
                        },
                        "drop_duplicates": {"type": "boolean", "default": False, "description": "Whether to drop duplicates"}
                    },
                    "required": ["left", "right"]
                }
            },
            {
                "name": "merge",
                "description": "Merge two dataframes on specified columns with different join strategies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "Left dataframe as list of dicts or polars DataFrame"
                        },
                        "right": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "Right dataframe as list of dicts or polars DataFrame"
                        },
                        "on": {"type": "array", "items": {"type": "string"}, "description": "Columns to join on"},
                        "how": {"type": "string", "enum": ["inner", "left", "right", "outer"], "default": "inner", "description": "Join strategy"},
                        "fuzzy": {"type": "boolean", "default": False, "description": "Enable fuzzy matching"}
                    },
                    "required": ["left", "right"]
                }
            },
            {
                "name": "group_by",
                "description": "Group dataframe by specified columns and apply aggregation functions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "DataFrame as list of dicts or polars DataFrame"
                        },
                        "by": {"type": "array", "items": {"type": "string"}, "description": "Columns to group by"},
                        "aggregations": {"type": "array", "description": "List of aggregation specifications"}
                    },
                    "required": ["dataframe", "by", "aggregations"]
                }
            },
            {
                "name": "apply_formula",
                "description": "Apply Excel-like formulas to dataframe columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "DataFrame as list of dicts or polars DataFrame"
                        },
                        "formula": {"type": "string", "description": "Excel-like formula to apply"},
                        "column_name": {"type": "string", "description": "Name for the new column"},
                        "use_excel_refs": {"type": "boolean", "default": False, "description": "Use Excel cell references"}
                    },
                    "required": ["dataframe", "formula", "column_name"]
                }
            },
            {
                "name": "drop_duplicates",
                "description": "Remove duplicate rows from dataframe",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "DataFrame as list of dicts or polars DataFrame"
                        },
                        "subset": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of columns to check for duplicates"}
                    },
                    "required": ["dataframe"]
                }
            },
            {
                "name": "explode",
                "description": "Explode a list/array column to long format by creating a row for each list element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array"},
                                {"$ref": "#/definitions/DataFrame"}
                            ],
                            "description": "DataFrame as list of dicts or polars DataFrame"
                        },
                        "column": {"type": "string", "description": "Column name to explode (must contain List or Array type)"}
                    },
                    "required": ["dataframe", "column"]
                }
            },
            {
                "name": "init",
                "description": "Initialize a dataframe from a list of dictionaries or JSON string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "oneOf": [
                                {"type": "array"},
                                {"type": "string"}
                            ],
                            "description": "Data as list of dicts or JSON string"
                        }
                    },
                    "required": ["dataframe"]
                }
            }
        ]

        return tools

    async def call(self, tool_name: str, **kwargs) -> DataFrameOperationResult:
        """
        Call a DataFrame operation by name with arguments.

        Args:
            tool_name: Name of the operation to call
            **kwargs: Arguments to pass to the operation. Can be direct arguments
                     or wrapped in an 'args' dictionary.

        Returns:
            DataFrameOperationResult with the operation result

        Raises:
            ValueError: If tool_name is not supported
        """
        if tool_name not in self._tool_registry:
            available_tools = list(self._tool_registry.keys())
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Unknown tool: {tool_name}. Available tools: {available_tools}"
            )

        try:
            # Check if arguments are wrapped in an 'args' dictionary
            # This handles the case where call is invoked as call(tool_name, args={...})
            if "args" in kwargs and isinstance(kwargs["args"], dict) and len(kwargs) == 1:
                actual_kwargs = kwargs["args"]
            else:
                actual_kwargs = kwargs

            return await self._tool_registry[tool_name](**actual_kwargs)
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return DataFrameOperationResult(
                data=[],
                success=False,
                message=f"Error calling {tool_name}: {str(e)}"
            )

    async def _call_sort(self, **kwargs) -> DataFrameOperationResult:
        """Call sort operation."""
        return await self.toolkit.sort(**kwargs)

    async def _call_filter(self, **kwargs) -> DataFrameOperationResult:
        """Call filter operation."""
        return await self.toolkit.filter(**kwargs)

    async def _call_concat(self, **kwargs) -> DataFrameOperationResult:
        """Call concat operation."""
        return await self.toolkit.concat(**kwargs)

    async def _call_merge(self, **kwargs) -> DataFrameOperationResult:
        """Call merge operation."""
        return await self.toolkit.merge(**kwargs)

    async def _call_group_by(self, **kwargs) -> DataFrameOperationResult:
        """Call group_by operation."""
        return await self.toolkit.group_by(**kwargs)

    async def _call_apply_formula(self, **kwargs) -> DataFrameOperationResult:
        """Call apply_formula operation."""
        return await self.toolkit.apply_formula(**kwargs)

    async def _call_drop_duplicates(self, **kwargs) -> DataFrameOperationResult:
        """Call drop_duplicates operation."""
        return await self.toolkit.drop_duplicates(**kwargs)

    async def _call_explode(self, **kwargs) -> DataFrameOperationResult:
        """Call explode operation."""
        return await self.toolkit.explode(**kwargs)

    async def _call_init(self, **kwargs) -> DataFrameOperationResult:
        """Call init operation."""
        return await self.toolkit.init(**kwargs)


# Convenience function for quick usage
async def call_dataframe_tool(tool_name: str, config: Optional[DataFrameConfig] = None, **kwargs) -> DataFrameOperationResult:
    """
    Convenience function to call a DataFrame tool without explicitly creating a toolkit instance.

    Args:
        tool_name: Name of the operation to call
        config: Optional configuration for DataFrame operations
        **kwargs: Arguments to pass to the operation

    Returns:
        DataFrameOperationResult with the operation result
    """
    toolkit = UnifiedDataFrameToolkit(config)
    return await toolkit.call(tool_name, **kwargs)