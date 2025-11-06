"""Test script for improved explode functionality with JSON string support."""

import asyncio
import polars as pl
from omnimcp_dataframe.unified import UnifiedDataFrameToolkit


async def test_explode_with_json_strings():
    """Test exploding columns that contain JSON strings."""

    toolkit = UnifiedDataFrameToolkit()

    # Test 1: JSON array of objects (most common case)
    print("=" * 60)
    print("Test 1: JSON array of objects")
    print("=" * 60)

    data1 = [
        {
            "id": 1,
            "name": "Alice",
            "metadata": '[{"url": "https://example.com/1", "content": "Hello"}, {"url": "https://example.com/2", "content": "World"}]'
        },
        {
            "id": 2,
            "name": "Bob",
            "metadata": '[{"url": "https://example.com/3", "content": "Foo"}, {"url": "https://example.com/4", "content": "Bar"}]'
        }
    ]

    result1 = await toolkit.call("explode", dataframe=data1, column="metadata")

    print(f"Success: {result1.success}")
    print(f"Input rows: {result1.input_rows}")
    print(f"Output rows: {result1.output_rows}")
    print(f"JSON conversion applied: {result1.json_conversion_applied}")
    print(f"\nResulting data:")
    for row in result1.data:
        print(row)

    # Test 2: Relaxed JSON format (unquoted keys)
    print("\n" + "=" * 60)
    print("Test 2: Relaxed JSON format (unquoted keys)")
    print("=" * 60)

    data2 = [
        {
            "id": 1,
            "items": '[{url:1, content:2}, {url:2, content:3}]'
        },
        {
            "id": 2,
            "items": '[{url:4, content:5}]'
        }
    ]

    result2 = await toolkit.call("explode", dataframe=data2, column="items")

    print(f"Success: {result2.success}")
    print(f"Input rows: {result2.input_rows}")
    print(f"Output rows: {result2.output_rows}")
    print(f"JSON conversion applied: {result2.json_conversion_applied}")
    print(f"\nResulting data:")
    for row in result2.data:
        print(row)

    # Test 3: Already parsed list (backward compatibility)
    print("\n" + "=" * 60)
    print("Test 3: Already parsed list (backward compatibility)")
    print("=" * 60)

    data3 = pl.DataFrame({
        "id": [1, 2],
        "tags": [["python", "data"], ["sql", "analytics"]]
    })

    result3 = await toolkit.call("explode", dataframe=data3, column="tags")

    print(f"Success: {result3.success}")
    print(f"Input rows: {result3.input_rows}")
    print(f"Output rows: {result3.output_rows}")
    print(f"JSON conversion applied: {result3.json_conversion_applied}")
    print(f"\nResulting data:")
    for row in result3.data:
        print(row)

    # Test 4: Single object wrapped as list
    print("\n" + "=" * 60)
    print("Test 4: Single JSON object (auto-wrapped as list)")
    print("=" * 60)

    data4 = [
        {
            "id": 1,
            "config": '{"url": "example.com", "timeout": 30}'
        }
    ]

    result4 = await toolkit.call("explode", dataframe=data4, column="config")

    print(f"Success: {result4.success}")
    print(f"Input rows: {result4.input_rows}")
    print(f"Output rows: {result4.output_rows}")
    print(f"JSON conversion applied: {result4.json_conversion_applied}")
    print(f"\nResulting data:")
    for row in result4.data:
        print(row)

    # Test 5: Empty and null values
    print("\n" + "=" * 60)
    print("Test 5: Empty and null values")
    print("=" * 60)

    data5 = [
        {
            "id": 1,
            "items": '[{"url": 1}]'
        },
        {
            "id": 2,
            "items": '[]'  # Empty array
        },
        {
            "id": 3,
            "items": None  # Null value
        }
    ]

    result5 = await toolkit.call("explode", dataframe=data5, column="items")

    print(f"Success: {result5.success}")
    print(f"Input rows: {result5.input_rows}")
    print(f"Output rows: {result5.output_rows}")
    print(f"JSON conversion applied: {result5.json_conversion_applied}")
    print(f"\nResulting data:")
    for row in result5.data:
        print(row)


if __name__ == "__main__":
    asyncio.run(test_explode_with_json_strings())
