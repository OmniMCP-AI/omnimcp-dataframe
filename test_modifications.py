#!/usr/bin/env python3
"""Test script to verify data_df field and args parameter passing."""

import asyncio
import polars as pl
from omnimcp_dataframe.unified import UnifiedDataFrameToolkit
from omnimcp_dataframe.server import DataFrameOperations

async def test_data_df_field():
    """Test that all operations return data_df field correctly."""
    print("=" * 60)
    print("Test 1: Testing data_df field in all operations")
    print("=" * 60)

    ops = DataFrameOperations()

    # Test data
    test_data = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "Bob", "age": 25, "score": 92},
        {"name": "Charlie", "age": 35, "score": 78},
    ]
    df = pl.DataFrame(test_data)

    # Test 1: Sort
    print("\n1. Testing sort...")
    result = await ops.sort(df, by=["age"])
    assert result.success, "Sort failed"
    assert result.data_df is not None, "data_df is None"
    assert isinstance(result.data_df, pl.DataFrame), "data_df is not a DataFrame"
    print(f"   ✓ data_df type: {type(result.data_df).__name__}")
    print(f"   ✓ data_df shape: ({result.data_df.height}, {result.data_df.width})")
    print(f"   ✓ data field has {len(result.data)} items")

    # Test 2: Filter
    print("\n2. Testing filter...")
    result = await ops.filter(df, conditions=[{"column": "age", "op": "gt", "value": 25}])
    assert result.success, "Filter failed"
    assert result.data_df is not None, "data_df is None"
    assert isinstance(result.data_df, pl.DataFrame), "data_df is not a DataFrame"
    print(f"   ✓ data_df shape: ({result.data_df.height}, {result.data_df.width})")

    # Test 3: Group by
    print("\n3. Testing group_by...")
    result = await ops.group_by(df, by=["age"], aggregations=[{"column": "score", "function": "mean"}])
    assert result.success, "Group by failed"
    assert result.data_df is not None, "data_df is None"
    assert isinstance(result.data_df, pl.DataFrame), "data_df is not a DataFrame"
    print(f"   ✓ data_df shape: ({result.data_df.height}, {result.data_df.width})")

    # Test 4: JSON serialization (data_df should be excluded)
    print("\n4. Testing JSON serialization...")
    result_dict = result.dict()
    assert "data_df" not in result_dict, "data_df should be excluded from dict"
    assert "data" in result_dict, "data should be in dict"
    assert "success" in result_dict, "success should be in dict"
    print("   ✓ data_df correctly excluded from serialization")
    print(f"   ✓ Serialized keys: {list(result_dict.keys())}")

    print("\n✅ All data_df tests passed!\n")

async def test_args_parameter():
    """Test args parameter passing in UnifiedDataFrameToolkit."""
    print("=" * 60)
    print("Test 2: Testing args parameter passing")
    print("=" * 60)

    toolkit = UnifiedDataFrameToolkit()

    test_data = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "Bob", "age": 25, "score": 92},
        {"name": "Charlie", "age": 35, "score": 78},
    ]

    # Test 1: Direct parameter passing
    print("\n1. Testing direct parameter passing...")
    result = await toolkit.call(
        "filter",
        dataframe=test_data,
        conditions=[{"column": "age", "op": "gt", "value": 25}]
    )
    assert result.success, f"Direct parameter passing failed: {result.message}"
    assert len(result.data) == 2, f"Expected 2 rows, got {len(result.data)}"
    assert result.data_df is not None, "data_df is None"
    assert isinstance(result.data_df, pl.DataFrame), "data_df is not a DataFrame"
    print(f"   ✓ Direct passing works, filtered {len(result.data)} rows")
    print(f"   ✓ data_df present: {result.data_df is not None}")

    # Test 2: Args dictionary passing
    print("\n2. Testing args dictionary passing...")
    result = await toolkit.call(
        "filter",
        args={
            "dataframe": test_data,
            "conditions": [{"column": "age", "op": "lt", "value": 35}]
        }
    )
    assert result.success, f"Args dictionary passing failed: {result.message}"
    assert len(result.data) == 2, f"Expected 2 rows, got {len(result.data)}"
    assert result.data_df is not None, "data_df is None"
    assert isinstance(result.data_df, pl.DataFrame), "data_df is not a DataFrame"
    print(f"   ✓ Args dict passing works, filtered {len(result.data)} rows")
    print(f"   ✓ data_df present: {result.data_df is not None}")

    # Test 3: Sort with args
    print("\n3. Testing sort with args dictionary...")
    result = await toolkit.call(
        "sort",
        args={
            "dataframe": test_data,
            "by": ["age"],
            "descending": [True]
        }
    )
    assert result.success, f"Sort with args failed: {result.message}"
    assert result.data[0]["age"] == 35, "Sort order incorrect"
    assert result.data_df is not None, "data_df is None"
    print(f"   ✓ Sort with args works")
    print(f"   ✓ First row age: {result.data[0]['age']} (should be 35)")

    # Test 4: Group by with args
    print("\n4. Testing group_by with args dictionary...")
    result = await toolkit.call(
        "group_by",
        args={
            "dataframe": test_data,
            "by": ["age"],
            "aggregations": [{"column": "score", "function": "sum"}]
        }
    )
    assert result.success, f"Group by with args failed: {result.message}"
    assert result.data_df is not None, "data_df is None"
    assert isinstance(result.data_df, pl.DataFrame), "data_df is not a DataFrame"
    print(f"   ✓ Group by with args works, {len(result.data)} groups")
    print(f"   ✓ data_df shape: ({result.data_df.height}, {result.data_df.width})")

    print("\n✅ All args parameter tests passed!\n")

async def test_data_df_content():
    """Test that data and data_df have the same content."""
    print("=" * 60)
    print("Test 3: Testing data and data_df consistency")
    print("=" * 60)

    toolkit = UnifiedDataFrameToolkit()

    test_data = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "Bob", "age": 25, "score": 92},
    ]

    result = await toolkit.call("init", dataframe=test_data)

    assert result.success, "Init failed"
    assert result.data_df is not None, "data_df is None"

    # Compare data and data_df
    print("\n1. Comparing data (list) and data_df (DataFrame)...")
    print(f"   data length: {len(result.data)}")
    print(f"   data_df height: {result.data_df.height}")
    assert len(result.data) == result.data_df.height, "Row count mismatch"

    # Check column names
    data_cols = set(result.data[0].keys())
    df_cols = set(result.data_df.columns)
    print(f"   data columns: {data_cols}")
    print(f"   data_df columns: {df_cols}")
    assert data_cols == df_cols, "Column names mismatch"

    # Check values
    df_as_dicts = result.data_df.to_dicts()
    print(f"\n2. Checking values...")
    for i, (dict_row, df_row) in enumerate(zip(result.data, df_as_dicts)):
        assert dict_row == df_row, f"Row {i} mismatch: {dict_row} != {df_row}"
    print(f"   ✓ All {len(result.data)} rows match")

    print("\n✅ Data consistency test passed!\n")

async def main():
    """Run all tests."""
    try:
        await test_data_df_field()
        await test_args_parameter()
        await test_data_df_content()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nSummary:")
        print("  ✓ data_df field correctly added to all operations")
        print("  ✓ data_df is properly typed as pl.DataFrame")
        print("  ✓ data_df is excluded from JSON serialization")
        print("  ✓ args parameter passing works correctly")
        print("  ✓ Both direct and args-wrapped calls work")
        print("  ✓ data and data_df content is consistent")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
