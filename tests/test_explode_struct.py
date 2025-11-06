#!/usr/bin/env python3
"""
Test suite for the explode_struct DataFrame operation.
"""

import sys
import os
import asyncio
import polars as pl

# Add the parent directory to the path so we can import omnimcp_dataframe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnimcp_dataframe import DataFrameToolkit

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False



@pytest.mark.asyncio
async def test_explode_struct_basic():
    """Test basic explode_struct functionality with url and price fields."""
    data = [
        {'id': 1, 'image_urls': [{'url': 'apple', 'price': 23}, {'url': 'banana', 'price': 23}]},
        {'id': 2, 'image_urls': [{'url': 'orange', 'price': 23}]}
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'image_urls', fields=['url', 'price'])
    print(result.data_df)

    assert result.success is True
    assert result.input_rows == 2
    assert result.output_rows == 3
    assert result.extracted_fields == ['url', 'price']
    assert result.original_column_dropped is True

    # Check the exploded data
    assert len(result.data) == 3
    assert result.data[0]['id'] == 1
    assert result.data[0]['url'] == 'apple'
    assert result.data[0]['price'] == 23
    assert result.data[1]['id'] == 1
    assert result.data[1]['url'] == 'banana'
    assert result.data[1]['price'] == 23
    assert result.data[2]['id'] == 2
    assert result.data[2]['url'] == 'orange'
    assert result.data[2]['price'] == 23

    # Check that original column is dropped
    assert 'image_urls' not in result.data[0]


@pytest.mark.asyncio
async def test_explode_struct_auto_detect_fields():
    """Test explode_struct with automatic field detection."""
    data = [
        {'id': 1, 'items': [{'name': 'apple', 'qty': 5, 'category': 'fruit'}]},
        {'id': 2, 'items': [{'name': 'bread', 'qty': 2, 'category': 'bakery'}]}
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'items', fields=None)
    print(result.data_df)

    assert result.success is True
    assert result.output_rows == 2
    assert set(result.extracted_fields) == {'name', 'qty', 'category'}

    # Check that all fields were extracted
    row = result.data[0]
    assert 'name' in row
    assert 'qty' in row
    assert 'category' in row
    assert 'items' not in row  # Original column should be dropped


@pytest.mark.asyncio
async def test_explode_struct_keep_original_column():
    """Test explode_struct with keep_original=False."""
    data = [
        {'id': 1, 'data': [{'field1': 'a', 'field2': 'b'}]}
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'data', fields=['field1'], drop_original=False)
    print(result.data_df)

    assert result.success is True
    assert result.original_column_dropped is False

    # Check that original column is preserved
    row = result.data[0]
    assert 'data' in row
    assert 'field1' in row
    assert row['field1'] == 'a'
    assert row['data']['field1'] == 'a'


@pytest.mark.asyncio
async def test_explode_struct_json_string_column():
    """Test explode_struct with JSON string columns."""
    data = [
        {'id': 1, 'items': '[{"name": "apple", "price": 5}, {"name": "banana", "price": 3}]'},
        {'id': 2, 'items': '[{"name": "orange", "price": 2}]'}
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'items', fields=['name', 'price'])

    assert result.success is True
    assert result.json_conversion_applied is True
    assert result.output_rows == 3

    # Check the parsed data
    apples = [row for row in result.data if row['name'] == 'apple']
    assert len(apples) == 1
    assert apples[0]['price'] == 5


@pytest.mark.asyncio
async def test_explode_struct_relaxed_json_format():
    """Test explode_struct with relaxed JSON format (unquoted keys)."""
    data = [
        {'id': 1, 'data': '[{url:1, content:2}, {url:2, content:3}]'}
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'data', fields=['url', 'content'])
    print(result.data_df)

    assert result.success is True
    assert result.json_conversion_applied is True
    assert result.output_rows == 2

    # Check that relaxed JSON was parsed
    assert result.data[0]['url'] == 1
    assert result.data[0]['content'] == 2
    assert result.data[1]['url'] == 2
    assert result.data[1]['content'] == 3


@pytest.mark.asyncio
async def test_explode_struct_empty_lists():
    """Test explode_struct with empty lists."""
    data = [
        {'id': 1, 'items': [{'name': 'apple'}]},
        {'id': 2, 'items': []},
        {'id': 3, 'items': [{'name': 'orange'}]}
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'items', fields=['name'])
    print(result.data_df)

    assert result.success is True
    assert result.output_rows == 3  # Polars creates a row with None for empty lists

    # Check that empty list creates a row with None value
    empty_row = [row for row in result.data if row['id'] == 2][0]
    assert empty_row['name'] is None


@pytest.mark.asyncio
async def test_explode_struct_missing_column():
    """Test that explode_struct fails gracefully when column doesn't exist."""
    data = [{'id': 1, 'name': 'Alice'}]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'missing_column')

    assert result.success is False
    assert 'Column not found: missing_column' in result.message


@pytest.mark.asyncio
async def test_explode_struct_non_list_column():
    """Test that explode_struct fails when column is not a list."""
    data = [{'id': 1, 'name': 'Alice'}]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'name', parse_json=False)

    assert result.success is False
    assert 'must be List or Array type' in result.message


@pytest.mark.asyncio
async def test_explode_struct_non_struct_elements():
    """Test explode_struct with non-struct elements (should return exploded without field extraction)."""
    data = [
        {'id': 1, 'tags': ['apple', 'banana']},
        {'id': 2, 'tags': ['orange']}
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'tags', fields=['url'])

    assert result.success is True
    assert result.output_rows == 3
    assert 'non-struct values' in result.note

    # Should return original exploded data without field extraction
    assert 'tags' in result.data[0]
    assert result.data[0]['tags'] == 'apple'


@pytest.mark.asyncio
async def test_explode_struct_no_fields_specified():
    """Test explode_struct when no fields can be auto-detected."""
    data = [
        {'id': 1, 'data': [None, 123]}  # No struct to extract fields from
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'data', fields=None)

    # When no struct fields can be detected, it should still explode the data
    # and return a note about non-struct values
    assert result.success is True
    assert result.output_rows == 2
    assert 'non-struct values' in result.note
    assert 'data' in result.data[0]  # Original column should remain


@pytest.mark.asyncio
async def test_explode_struct_empty_dataframe():
    """Test explode_struct with empty dataframe."""
    data = []

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'items')

    assert result.success is True
    assert result.input_rows == 0
    assert result.output_rows == 0
    assert len(result.data) == 0


@pytest.mark.asyncio
async def test_explode_struct_mixed_objects():
    """Test explode_struct with objects that have different fields."""
    data = [
        {'id': 1, 'items': [{'name': 'apple', 'price': 5}, {'name': 'banana', 'qty': 3}]},
    ]

    toolkit = DataFrameToolkit()
    result = await toolkit.explode_struct(data, 'items', fields=['name', 'price', 'qty'])

    assert result.success is True
    assert result.output_rows == 2

    # Check that missing fields become None
    banana_row = [row for row in result.data if row['name'] == 'banana'][0]
    assert banana_row['price'] is None
    assert banana_row['qty'] == 3


def run_sync_tests():
    """Run tests synchronously for manual testing."""
    print("Running explode_struct tests...\n")

    async def run_all_tests():
        tests = [
            ("Basic explode_struct", test_explode_struct_basic),
            ("Auto detect fields", test_explode_struct_auto_detect_fields),
            ("Keep original column", test_explode_struct_keep_original_column),
            ("JSON string column", test_explode_struct_json_string_column),
            ("Relaxed JSON format", test_explode_struct_relaxed_json_format),
            ("Empty lists", test_explode_struct_empty_lists),
            ("Missing column (should fail)", test_explode_struct_missing_column),
            ("Non-list column (should fail)", test_explode_struct_non_list_column),
            ("Non-struct elements", test_explode_struct_non_struct_elements),
            ("No fields specified (should fail)", test_explode_struct_no_fields_specified),
            ("Empty dataframe", test_explode_struct_empty_dataframe),
            ("Mixed objects", test_explode_struct_mixed_objects),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                await test_func()
                print(f"✓ {test_name}")
                passed += 1
            except AssertionError as e:
                print(f"✗ {test_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"✗ {test_name}: Unexpected error: {e}")
                failed += 1

        print(f"\n{'='*50}")
        print(f"Tests passed: {passed}/{passed + failed}")
        print(f"Tests failed: {failed}/{passed + failed}")
        print(f"{'='*50}")

    asyncio.run(run_all_tests())


if __name__ == "__main__":
    # If pytest is not available, run tests manually
    if PYTEST_AVAILABLE:
        import pytest as real_pytest
        real_pytest.main([__file__, "-v"])
    else:
        print("pytest not found, running tests manually...\n")
        run_sync_tests()