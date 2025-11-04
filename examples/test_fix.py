#!/usr/bin/env python3
"""
Simple test to verify the async fixes work correctly.
"""

import sys
import os
import asyncio

# Add the parent directory to the path so we can import omnimcp_dataframe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnimcp_dataframe import UnifiedDataFrameToolkit

async def test_fixes():
    """Test the fixes for async methods."""

    # Sample data
    sample_data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "San Francisco"},
        {"name": "Alice", "age": 30, "city": "New York"},  # duplicate
    ]

    toolkit = UnifiedDataFrameToolkit()

    # Test group_by (previously failed)
    print("Testing group_by...")
    result = await toolkit.call(
        "group_by",
        dataframe=sample_data,
        by=["city"],
        aggregations=[
            {"column": "age", "function": "count"}
        ]
    )

    if result.success:
        print(f"✓ group_by successful: {len(result.data)} groups")
        for row in result.data:
            print(f"  {row}")
    else:
        print(f"✗ group_by failed: {result.message}")

    print()

    # Test drop_duplicates (previously failed)
    print("Testing drop_duplicates...")
    result = await toolkit.call(
        "drop_duplicates",
        dataframe=sample_data,
        subset=["name"]
    )

    if result.success:
        print(f"✓ drop_duplicates successful: {result.output_rows} rows")
        for row in result.data:
            print(f"  {row}")
    else:
        print(f"✗ drop_duplicates failed: {result.message}")

    print("\n=== Test completed ===")

if __name__ == "__main__":
    asyncio.run(test_fixes())