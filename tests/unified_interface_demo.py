#!/usr/bin/env python3
"""
Demo script for the unified DataFrame interface.

Shows how to use the UnifiedDataFrameToolkit to call operations by name
with both list-of-dicts and polars.DataFrame input types.
"""

import sys
import os
import asyncio

# Add the parent directory to the path so we can import omnimcp_dataframe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from omnimcp_dataframe import UnifiedDataFrameToolkit, call_dataframe_tool

async def main():
    """Demonstrate the unified interface functionality."""

    # Sample data
    sample_data = [
        {"name": "Alice", "age": 30, "city": "New York", "salary": "75K"},
        {"name": "Bob", "age": 25, "city": "San Francisco", "salary": "85K"},
        {"name": "Charlie", "age": 35, "city": "New York", "salary": "90K"},
        {"name": "Diana", "age": 28, "city": "Chicago", "salary": "70K"},
        {"name": "Eve", "age": 32, "city": "San Francisco", "salary": "95K"},
    ]

    print("=== Unified DataFrame Interface Demo ===\n")

    # Create toolkit instance
    toolkit = UnifiedDataFrameToolkit()

    # 1. Test with list of dictionaries
    print("1. Testing with list of dictionaries:")
    result = await toolkit.call(
        "sort",
        dataframe=sample_data,
        by=["age"],
        descending=[False]
    )

    if result.success:
        print(f"✓ Sort successful: {len(result.data)} rows")
        for row in result.data[:3]:  # Show first 3 rows
            print(f"  {row}")
    else:
        print(f"✗ Sort failed: {result.message}")

    print()

    # 2. Test with polars DataFrame
    print("2. Testing with polars DataFrame:")
    df = pl.DataFrame(sample_data)
    result = await toolkit.call(
        "filter",
        dataframe=df,
        conditions=[
            {"column": "age", "op": "gte", "value": 30}
        ]
    )

    if result.success:
        print(f"✓ Filter successful: {len(result.data)} rows")
        for row in result.data:
            print(f"  {row}")
    else:
        print(f"✗ Filter failed: {result.message}")

    print()

    # 3. Test group_by operation
    print("3. Testing group_by operation:")
    result = await toolkit.call(
        "group_by",
        dataframe=sample_data,
        by=["city"],
        aggregations=[
            {"column": "salary", "function": "mean"},
            {"column": "age", "function": "count"}
        ]
    )

    if result.success:
        print(f"✓ Group by successful: {len(result.data)} groups")
        for row in result.data:
            print(f"  {row}")
    else:
        print(f"✗ Group by failed: {result.message}")

    print()

    # 4. Test concat operation
    print("4. Testing concat operation:")
    more_data = [
        {"name": "Frank", "age": 29, "city": "Boston", "salary": "80K"},
        {"name": "Grace", "age": 31, "city": "Seattle", "salary": "88K"},
    ]

    result = await toolkit.call(
        "concat",
        left=sample_data,
        right=more_data,
        drop_duplicates=False
    )

    if result.success:
        print(f"✓ Concat successful: {result.output_rows} total rows")
        print(f"  Shape: {result.shape}")
    else:
        print(f"✗ Concat failed: {result.message}")

    print()

    # 5. Test apply_formula operation
    print("5. Testing apply_formula operation:")
    result = await toolkit.call(
        "apply_formula",
        dataframe=sample_data,
        formula="age * 12",
        column_name="age_in_months"
    )

    if result.success:
        print(f"✓ Formula applied successfully:")
        for row in result.data[:2]:  # Show first 2 rows
            print(f"  {row}")
    else:
        print(f"✗ Formula application failed: {result.message}")

    print()

    # 6. Test convenience function
    print("6. Testing convenience function:")
    result = await call_dataframe_tool(
        "drop_duplicates",
        dataframe=sample_data + sample_data[:2],  # Add some duplicates
        subset=["name"]
    )

    if result.success:
        print(f"✓ Drop duplicates successful: {result.output_rows} rows (duplicates removed: {result.duplicates_removed})")
    else:
        print(f"✗ Drop duplicates failed: {result.message}")

    print()

    # 7. Test error handling
    print("7. Testing error handling:")
    result = await toolkit.call(
        "unknown_tool",
        dataframe=sample_data
    )

    if not result.success:
        print(f"✓ Error handling works: {result.message}")
    else:
        print("✗ Error handling failed")

    print()

    # 8. Show available tools
    print("8. Available tools:")
    tools = toolkit.get_available_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    print("\n=== Demo completed ===")


if __name__ == "__main__":
    asyncio.run(main())