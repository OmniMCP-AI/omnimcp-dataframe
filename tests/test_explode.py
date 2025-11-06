#!/usr/bin/env python3
"""
Test suite for the explode DataFrame operation.
"""

import sys
import os
import asyncio

# Add the parent directory to the path so we can import omnimcp_dataframe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnimcp_dataframe import UnifiedDataFrameToolkit

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy decorator if pytest is not available
    class DummyMarker:
        @staticmethod
        def asyncio(func):
            return func

    class DummyMark:
        mark = DummyMarker()

    pytest = DummyMark()


@pytest.mark.asyncio
async def test_explode_basic():
    """Test basic explode functionality with a single list column."""
    data = [
        {'id': 1, 'name': 'Alice', 'tags': ['python', 'rust', 'go']},
        {'id': 2, 'name': 'Bob', 'tags': ['java', 'scala']},
        {'id': 3, 'name': 'Charlie', 'tags': ['javascript', 'typescript']}
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='tags')

    assert result.success is True
    assert result.input_rows == 3
    assert result.output_rows == 7
    assert result.shape == "(7, 3)"
    assert len(result.data) == 7

    # Check that each tag is properly exploded
    alice_rows = [row for row in result.data if row['name'] == 'Alice']
    assert len(alice_rows) == 3
    assert set(row['tags'] for row in alice_rows) == {'python', 'rust', 'go'}


@pytest.mark.asyncio
async def test_explode_single_element_list():
    """Test explode with lists containing a single element."""
    data = [
        {'id': 1, 'skills': ['python']},
        {'id': 2, 'skills': ['java']},
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='skills')

    assert result.success is True
    assert result.input_rows == 2
    assert result.output_rows == 2
    assert len(result.data) == 2


@pytest.mark.asyncio
async def test_explode_empty_list():
    """Test explode with empty lists."""
    data = [
        {'id': 1, 'items': ['a', 'b']},
        {'id': 2, 'items': []},
        {'id': 3, 'items': ['c']},
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='items')

    assert result.success is True
    assert result.input_rows == 3
    # Empty list creates a row with None value in Polars
    assert result.output_rows == 4
    assert len(result.data) == 4

    # Check that empty list becomes None
    id2_row = [row for row in result.data if row['id'] == 2]
    assert len(id2_row) == 1
    assert id2_row[0]['items'] is None


@pytest.mark.asyncio
async def test_explode_non_list_column():
    """Test that explode fails gracefully when column is not a list."""
    data = [
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'name': 'Bob'}
    ]

    toolkit = UnifiedDataFrameToolkit()
    # Disable JSON parsing to test that non-list columns are rejected
    result = await toolkit.call('explode', dataframe=data, column='name', parse_json=False)

    assert result.success is False
    assert 'must be List or Array type' in result.message
    assert 'String' in result.message


@pytest.mark.asyncio
async def test_explode_missing_column():
    """Test that explode fails gracefully when column doesn't exist."""
    data = [
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'name': 'Bob'}
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='tags')

    assert result.success is False
    assert 'Column not found: tags' in result.message


@pytest.mark.asyncio
async def test_explode_with_null_values():
    """Test explode with null values in the list."""
    data = [
        {'id': 1, 'values': ['a', None, 'b']},
        {'id': 2, 'values': [None, 'c']},
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='values')

    assert result.success is True
    assert result.input_rows == 2
    assert result.output_rows == 5


@pytest.mark.asyncio
async def test_explode_preserves_other_columns():
    """Test that explode preserves other columns correctly."""
    data = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'hobbies': ['reading', 'coding']},
        {'id': 2, 'name': 'Bob', 'age': 25, 'hobbies': ['gaming']},
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='hobbies')

    assert result.success is True
    assert result.output_rows == 3

    # Check that Alice's rows have correct id, name, and age
    alice_rows = [row for row in result.data if row['name'] == 'Alice']
    assert len(alice_rows) == 2
    for row in alice_rows:
        assert row['id'] == 1
        assert row['age'] == 30

    # Check Bob's single row
    bob_rows = [row for row in result.data if row['name'] == 'Bob']
    assert len(bob_rows) == 1
    assert bob_rows[0]['id'] == 2
    assert bob_rows[0]['age'] == 25
    assert bob_rows[0]['hobbies'] == 'gaming'


@pytest.mark.asyncio
async def test_explode_empty_dataframe():
    """Test explode with empty dataframe."""
    data = []

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='tags')

    assert result.success is True
    assert result.input_rows == 0
    assert result.output_rows == 0
    assert len(result.data) == 0


@pytest.mark.asyncio
async def test_explode_json_string_column():
    """Test explode with JSON string columns (new feature)."""
    data = [
        {'id': 1, 'items': '[{"name": "apple", "qty": 5}, {"name": "banana", "qty": 3}]'},
        {'id': 2, 'items': '[{"name": "orange", "qty": 2}]'},
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='items')

    assert result.success is True
    assert result.input_rows == 2
    assert result.output_rows == 3
    assert result.json_conversion_applied is True

    # Verify the exploded data
    id1_rows = [row for row in result.data if row['id'] == 1]
    assert len(id1_rows) == 2
    assert id1_rows[0]['items']['name'] in ['apple', 'banana']
    assert id1_rows[1]['items']['name'] in ['apple', 'banana']


@pytest.mark.asyncio
async def test_explode_json_string_relaxed_format():
    """Test explode with relaxed JSON format (unquoted keys)."""
    data = [
        {'id': 1, 'data': '[{url:1, content:2}, {url:2, content:3}]'},
    ]

    toolkit = UnifiedDataFrameToolkit()
    result = await toolkit.call('explode', dataframe=data, column='data')

    assert result.success is True
    assert result.input_rows == 1
    assert result.output_rows == 2
    assert result.json_conversion_applied is True

    # Verify the exploded data has parsed objects
    assert result.data[0]['data']['url'] == 1
    assert result.data[1]['data']['url'] == 2


def run_sync_tests():
    """Run tests synchronously for manual testing."""
    print("Running explode tests...\n")

    async def run_all_tests():
        tests = [
            ("Basic explode", test_explode_basic),
            ("Single element list", test_explode_single_element_list),
            ("Empty list", test_explode_empty_list),
            ("Non-list column (should fail)", test_explode_non_list_column),
            ("Missing column (should fail)", test_explode_missing_column),
            ("Null values", test_explode_with_null_values),
            ("Preserve other columns", test_explode_preserves_other_columns),
            ("Empty dataframe", test_explode_empty_dataframe),
            ("JSON string column", test_explode_json_string_column),
            ("JSON relaxed format", test_explode_json_string_relaxed_format),
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
