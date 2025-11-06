"""
Shared tool definitions registry for DataFrame operations.

This module provides the single source of truth for all DataFrame tool definitions.
The base schemas use MCP format (JSON-only, array types) and can be transformed
for different use cases.
"""

from typing import Any, Dict, List
import copy


# Canonical tool definitions in MCP format (JSON-serializable only)
# This is the single source of truth - all other formats derive from this
TOOL_DEFINITIONS_MCP = [
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
        "description": "Explode a list/array column to long format with optional struct field extraction. Supports basic explode (original behavior) AND struct explode + extract (NEW) for JSON arrays with objects. Automatically handles JSON string parsing and relaxed JSON format.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataframe": {"type": "array", "items": {"type": "object"}},
                "column": {"type": "string", "description": "Column name to explode. Can be List/Array type or JSON string (e.g., '[{\"url\":1},{\"url\":2}]')"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "Optional list of field names to extract from each object. If provided, extracts struct fields to separate columns. If None, does basic explode only."},
                "parse_json": {"type": "boolean", "description": "If true, parse string columns as JSON before exploding (default: true)", "default": True},
                "drop_original": {"type": "boolean", "description": "If true, removes the original column after field extraction (default: true)", "default": True},
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
                "json_conversion_applied": {"type": "boolean"},
                "exploded_column": {"type": "string"},
                "extracted_fields": {"type": "array", "items": {"type": "string"}},
                "original_column_dropped": {"type": "boolean"},
                "note": {"type": "string"},
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


# DataFrame parameter names that should be transformed for Python SDK
DATAFRAME_PARAMS = ["dataframe", "left", "right"]


def get_tools_mcp_format() -> List[Dict[str, Any]]:
    """
    Get tool definitions in MCP format (source of truth).

    Returns MCP-compatible schemas with JSON-only types.
    Use this for MCP server implementations.

    Returns:
        List of tools with 'inputSchema' and 'outputSchema' keys
    """
    return copy.deepcopy(TOOL_DEFINITIONS_MCP)


def get_tools_sdk_format() -> List[Dict[str, Any]]:
    """
    Get tool definitions for Python SDK usage.

    Transforms MCP schemas to support both array and DataFrame types.
    Use this for direct Python package usage where DataFrame objects are supported.

    Returns:
        List of tools with 'parameters' key (transformed to support DataFrame)
    """
    tools = []

    for tool in TOOL_DEFINITIONS_MCP:
        # Deep copy to avoid modifying the source
        tool_copy = copy.deepcopy(tool)

        # Convert to SDK format: use 'parameters' instead of 'inputSchema'
        sdk_tool = {
            "name": tool_copy["name"],
            "description": tool_copy["description"],
            "parameters": tool_copy["inputSchema"]
        }

        # Transform dataframe parameters to support both array and DataFrame
        properties = sdk_tool["parameters"]["properties"]
        for param_name in DATAFRAME_PARAMS:
            if param_name in properties:
                # Skip if already has oneOf (like init's dataframe parameter)
                if "oneOf" in properties[param_name]:
                    continue

                # Transform array-only to array OR DataFrame
                properties[param_name] = {
                    "oneOf": [
                        {"type": "array"},
                        {"$ref": "#/definitions/DataFrame"}
                    ],
                    "description": f"DataFrame as list of dicts or polars DataFrame"
                }

        tools.append(sdk_tool)

    return tools


def get_tool_names() -> List[str]:
    """
    Get list of all available tool names.

    Returns:
        List of tool names
    """
    return [tool["name"] for tool in TOOL_DEFINITIONS_MCP]


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """
    Get a specific tool definition by name (MCP format).

    Args:
        name: Tool name

    Returns:
        Tool definition dictionary in MCP format

    Raises:
        ValueError: If tool name not found
    """
    for tool in TOOL_DEFINITIONS_MCP:
        if tool["name"] == name:
            return copy.deepcopy(tool)
    raise ValueError(f"Tool '{name}' not found. Available tools: {get_tool_names()}")
