# OmniMCP DataFrame Toolkit

A comprehensive DataFrame manipulation toolkit for MCP servers and data processing applications. Built on Polars for high performance operations with async support.

## Features

- **High Performance**: Built on Polars for fast data processing with direct DataFrame operations
- **Async Support**: Full async/await support for non-blocking operations
- **Unified Interface**: Single entry point for calling operations by name and arguments
- **Flexible Input**: Support for both list-of-dictionaries and polars DataFrame types
- **Rich Operations**: Sort, filter, merge, group by, formula application, and more
- **Excel Compatibility**: Support for Excel-like formulas with both column names and cell references
- **Intelligent Operations**: Fuzzy merging with LLM-powered join key detection
- **Framework Agnostic**: Works with MCP servers, REST APIs, or direct Python applications
- **Type Safety**: Full Pydantic model validation and type hints

## Installation

```bash
pip install omnimcp-dataframe
```

For development dependencies:
```bash
pip install omnimcp-dataframe[dev]
```

For MCP compatibility:
```bash
pip install omnimcp-dataframe[mcp]
```

## Quick Start

### Unified Interface (Recommended)

The unified interface provides a single entry point for all operations with async support:

```python
import asyncio
from omnimcp_dataframe import UnifiedDataFrameToolkit, call_dataframe_tool

async def main():
    # Initialize the unified toolkit
    toolkit = UnifiedDataFrameToolkit()

    # Sample data
    data = [
        {"name": "Alice", "age": 30, "city": "New York", "salary": "75K"},
        {"name": "Bob", "age": 25, "city": "San Francisco", "salary": "85K"},
        {"name": "Charlie", "age": 35, "city": "New York", "salary": "90K"},
    ]

    # Sort by age using unified interface
    result = await toolkit.call("sort", dataframe=data, by=["age"])
    if result.success:
        print("Sorted data:", result.data)

    # Filter by age using convenience function
    result = await call_dataframe_tool(
        "filter",
        dataframe=data,
        conditions=[{"column": "age", "op": "gte", "value": 30}]
    )
    if result.success:
        print("Filtered data:", result.data)

# Run the async function
asyncio.run(main())
```

### Direct API Usage

For more control, you can use the direct async API:

```python
import asyncio
from omnimcp_dataframe import DataFrameToolkit
import polars as pl

async def main():
    # Initialize the toolkit
    toolkit = DataFrameToolkit()

    # Sample data (can be list of dicts or polars DataFrame)
    data = [
        {"name": "Alice", "age": 30, "city": "New York", "salary": "75K"},
        {"name": "Bob", "age": 25, "city": "San Francisco", "salary": "85K"},
        {"name": "Charlie", "age": 35, "city": "New York", "salary": "90K"},
    ]

    # Or use polars DataFrame directly
    df = pl.DataFrame(data)

    # Sort by age
    result = await toolkit.sort(dataframe=data, by=["age"])
    if result.success:
        print("Sorted data:", result.data)

    # Filter by age
    result = await toolkit.filter(
        dataframe=data,
        conditions=[
            {"column": "age", "op": "gte", "value": 30}
        ]
    )
    if result.success:
        print("Filtered data:", result.data)

    # Group by city and calculate average salary
    result = await toolkit.group_by(
        dataframe=data,
        by=["city"],
        aggregations=[
            {"column": "salary", "function": "mean"},
            {"column": "age", "function": "count"}
        ]
    )
    if result.success:
        print("Grouped data:", result.data)

# Run the async function
asyncio.run(main())
```

### Excel Formula Application

```python
import asyncio
from omnimcp_dataframe import UnifiedDataFrameToolkit

async def main():
    toolkit = UnifiedDataFrameToolkit()

    # Sample data
    data = [
        {"name": "Alice", "age": 30, "salary": "75K"},
        {"name": "Bob", "age": 25, "salary": "85K"},
    ]

    # Apply formulas using column names
    result = await toolkit.call(
        "apply_formula",
        dataframe=data,
        formula="salary * 1.1",  # 10% raise
        column_name="new_salary"
    )
    if result.success:
        print("Formula applied:", result.data)

    # Apply formulas using Excel cell references
    result = await toolkit.call(
        "apply_formula",
        dataframe=data,
        formula="=B1 * 12",  # age * 12 (months)
        column_name="age_in_months",
        use_excel_refs=True
    )
    if result.success:
        print("Excel reference formula applied:", result.data)

asyncio.run(main())
```

### Advanced Operations

```python
import asyncio
from omnimcp_dataframe import UnifiedDataFrameToolkit

async def main():
    toolkit = UnifiedDataFrameToolkit()

    # Merge two dataframes
    left_data = [{"id": 1, "name": "Alice", "dept": "Engineering"}]
    right_data = [{"id": 1, "salary": "75K"}, {"id": 2, "salary": "80K"}]

    result = await toolkit.call(
        "merge",
        left=left_data,
        right=right_data,
        on=["id"],
        how="inner"
    )
    if result.success:
        print("Merged data:", result.data)

    # Concatenate dataframes
    result = await toolkit.call(
        "concat",
        left=left_data,
        right=right_data,
        drop_duplicates=True
    )
    if result.success:
        print("Concatenated data:", result.data)

asyncio.run(main())
```

## MCP Server Integration

```python
import asyncio
from omnimcp_dataframe import DataFrameServer

async def main():
    # Create server
    server = DataFrameServer()

    # Get available tools
    tools = await server.get_tools()
    print(f"Available tools: {len(tools)}")

    # Sample data
    data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "San Francisco"},
    ]

    # Call a tool
    result = await server.call_tool(
        tool_name="sort",
        arguments={
            "dataframe": data,
            "by": ["age"],
            "descending": [True]
        }
    )

    if result.get("success", False):
        print("Sorted data:", result["data"])

asyncio.run(main())
```

## Configuration

```python
import asyncio
from omnimcp_dataframe import UnifiedDataFrameToolkit, DataFrameConfig

async def main():
    # Configure with custom settings
    config = DataFrameConfig(
        max_memory_mb=2048,  # Memory limit
        llm_model="openai/gpt-4",  # LLM model for intelligent operations
        enable_fuzzy_matching=True,  # Enable fuzzy matching
        confidence_threshold=0.7  # Confidence threshold for intelligent ops
    )

    # Initialize toolkit with configuration
    toolkit = UnifiedDataFrameToolkit(config)

    # Use the configured toolkit
    result = await toolkit.call("sort", dataframe=data, by=["age"])

asyncio.run(main())
```

## Available Operations

### Sorting
- **sort**: Sort dataframe by specified columns with optional limit

### Filtering
- **filter**: Filter dataframe based on single or multiple conditions with AND/OR logic

### Combining
- **concat**: Concatenate two dataframes with optional duplicate removal
- **merge**: Merge two dataframes on specified columns with different join strategies

### Aggregation
- **group_by**: Group dataframe by specified columns and apply aggregation functions

### Transformation
- **apply_formula**: Apply Excel-like formulas to dataframe columns
- **drop_duplicates**: Remove duplicate rows from dataframe

### Initialization
- **init**: Initialize a dataframe from a list of dictionaries or JSON string

## Unified Interface Usage

The unified interface provides a consistent way to call all operations:

### Getting Available Tools

```python
import asyncio
from omnimcp_dataframe import UnifiedDataFrameToolkit

async def main():
    toolkit = UnifiedDataFrameToolkit()

    # Get all available tools with their schemas
    tools = toolkit.get_available_tools()

    for tool in tools:
        print(f"Tool: {tool['name']}")
        print(f"Description: {tool['description']}")
        print(f"Required parameters: {tool['parameters'].get('required', [])}")

asyncio.run(main())
```

### Error Handling

All operations return a `DataFrameOperationResult` object:

```python
result = await toolkit.call("sort", dataframe=data, by=["age"])

if result.success:
    print("Operation successful")
    print(f"Result: {result.data}")
    print(f"Output rows: {result.output_rows}")
    print(f"Shape: {result.shape}")
else:
    print(f"Operation failed: {result.message}")
```

### Flexible Input Types

All operations accept both list-of-dictionaries and polars DataFrames:

```python
import polars as pl

# Using list of dictionaries
data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
result1 = await toolkit.call("sort", dataframe=data, by=["age"])

# Using polars DataFrame directly
df = pl.DataFrame(data)
result2 = await toolkit.call("sort", dataframe=df, by=["age"])

# Both results are equivalent
```

## Supported Operators

### Filter Operators
- `eq`: Equals
- `ne`: Not equals
- `gt`: Greater than
- `lt`: Less than
- `gte`: Greater than or equal
- `lte`: Less than or equal
- `contains`: String contains
- `in`: Value in list
- `between`: Value between two numbers
- `is_null`: Value is null
- `not_null`: Value is not null
- `regex`: Regular expression match

### Aggregation Functions
- `count`: Count non-null values
- `sum`: Sum of values
- `mean`: Average of values
- `min`: Minimum value
- `max`: Maximum value
- `median`: Median value
- `first`: First value
- `last`: Last value
- `std`: Standard deviation
- `var`: Variance
- `count_distinct`: Count distinct values

## Data Type Handling

The toolkit automatically handles:

- **Human-readable numbers**: "1.5K", "2.3M", "100B" → numeric values
- **String numbers**: "500", "1,000" → numeric values for comparisons
- **Type conversion**: Automatic casting when needed for operations
- **Missing values**: Graceful handling of null/missing values

## Error Handling

All operations return a `DataFrameOperationResult` object:

```python
result = toolkit.sort(dataframe=data, by=["invalid_column"])

if result.success:
    print("Operation successful:", result.data)
else:
    print("Operation failed:", result.message)
```

## Performance Considerations

- Built on Polars for high performance
- Automatic memory management
- Streaming support for large datasets
- Configurable memory limits
- Efficient data type handling

## Development

```bash
# Clone repository
git clone https://github.com/omnimcp/omnimcp-dataframe.git
cd omnimcp-dataframe

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
black omnimcp_dataframe/
isort omnimcp_dataframe/
ruff check omnimcp_dataframe/
mypy omnimcp_dataframe/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## Support

- Documentation: https://omnimcp-dataframe.readthedocs.io
- Issues: https://github.com/omnimcp/omnimcp-dataframe/issues
- Discussions: https://github.com/omnimcp/omnimcp-dataframe/discussions