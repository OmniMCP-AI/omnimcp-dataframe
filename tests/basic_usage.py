"""
Basic usage examples for omnimcp-dataframe package.
"""

from omnimcp_dataframe import DataFrameToolkit

async def main():
    """Demonstrate basic DataFrame operations."""
    print("=== OmniMCP DataFrame Toolkit - Basic Usage ===\n")

    # Initialize toolkit
    toolkit = DataFrameToolkit()

    # Sample data
    sales_data = [
        {"name": "Alice", "age": 30, "city": "New York", "salary": "75K", "department": "Engineering"},
        {"name": "Bob", "age": 25, "city": "San Francisco", "salary": "85K", "department": "Sales"},
        {"name": "Charlie", "age": 35, "city": "New York", "salary": "90K", "department": "Engineering"},
        {"name": "Diana", "age": 28, "city": "Chicago", "salary": "70K", "department": "Marketing"},
        {"name": "Eve", "age": 32, "city": "San Francisco", "salary": "95K", "department": "Sales"},
    ]

    print("Original data:")
    for row in sales_data:
        print(f"  {row}")
    print()

    # Example 1: Sort by age
    print("1. Sort by age (ascending):")
    result = await toolkit.sort(dataframe=sales_data, by=["age"])
    if result.success:
        for row in result.data:
            print(f"  {row}")
    print()

    # Example 2: Sort by salary (descending)
    print("2. Sort by salary (descending):")
    result = await toolkit.sort(dataframe=sales_data, by=["salary"], descending=[True])
    if result.success:
        for row in result.data:
            print(f"  {row}")
    print()

    # Example 3: Filter by age
    print("3. Filter employees age >= 30:")
    result = await toolkit.filter(
        dataframe=sales_data,
        conditions=[
            {"column": "age", "op": "gte", "value": 30}
        ]
    )
    if result.success:
        print(f"  Found {result.output_rows} employees:")
        for row in result.data:
            print(f"    {row}")
    print()

    # Example 4: Filter by city and department
    print("4. Filter by city = 'New York' AND department = 'Engineering':")
    result = await toolkit.filter(
        dataframe=sales_data,
        conditions=[
            {"column": "city", "op": "eq", "value": "New York"},
            {"column": "department", "op": "eq", "value": "Engineering"}
        ],
        logic="AND"
    )
    if result.success:
        for row in result.data:
            print(f"  {row}")
    print()

    # Example 5: Group by city and count employees
    print("5. Group by city and count employees:")
    result = await toolkit.group_by(
        dataframe=sales_data,
        by=["city"],
        aggregations=[
            {"column": "name", "function": "count"},
            {"column": "salary", "function": "mean"}
        ]
    )
    if result.success:
        for row in result.data:
            print(f"  {row}")
    print()

    # Example 6: Group by department and get salary statistics
    print("6. Group by department - salary statistics:")
    result = await toolkit.group_by(
        dataframe=sales_data,
        by=["department"],
        aggregations=[
            {"column": "salary", "function": "mean"},
            {"column": "salary", "function": "max"},
            {"column": "name", "function": "count"}
        ]
    )
    if result.success:
        for row in result.data:
            print(f"  {row}")
    print()

    # Example 7: Apply formula - calculate annual salary
    print("7. Calculate annual salary (monthly * 12):")
    result = await toolkit.apply_formula(
        dataframe=sales_data,
        formula="salary * 12",
        column_name="annual_salary"
    )
    if result.success:
        print("  Data with annual salary:")
        for row in result.data[:3]:  # Show first 3 rows
            print(f"    {row}")
    print()

    # Example 8: Apply Excel-style formula
    print("8. Excel-style formula - age in months:")
    result = await toolkit.apply_formula(
        dataframe=sales_data,
        formula="=B1 * 12",  # B1 refers to age column
        column_name="age_in_months",
        use_excel_refs=True
    )
    if result.success:
        print("  Data with age in months:")
        for row in result.data[:3]:  # Show first 3 rows
            print(f"    {row}")
    print()

    # Example 9: Merge two dataframes
    print("9. Merge employee data with bonus data:")
    bonus_data = [
        {"name": "Alice", "bonus": "5K"},
        {"name": "Charlie", "bonus": "8K"},
        {"name": "Eve", "bonus": "10K"},
    ]

    result = await toolkit.merge(
        left=sales_data,
        right=bonus_data,
        on=["name"],
        how="left"
    )
    if result.success:
        print("  Merged data:")
        for row in result.data:
            print(f"    {row}")
    print()

    # Example 10: Concatenate dataframes
    print("10. Concatenate with existing new employees:")
    new_employees = [
        {"name": "Frank", "age": 29, "city": "Seattle", "salary": "80K", "department": "Engineering"},
        {"name": "Grace", "age": 26, "city": "Boston", "salary": "72K", "department": "Marketing"},
    ]

    result = await toolkit.concat(
        left=sales_data,
        right=new_employees,
        drop_duplicates=True
    )
    if result.success:
        print(f"  Combined data ({result.total_rows} total employees):")
        for row in result.data:
            print(f"    {row}")
    print()

    # Example 11: Remove duplicates
    print("11. Remove duplicates (if any):")
    data_with_duplicates = sales_data + sales_data[:2]  # Add duplicates
    print(f"  Before: {len(data_with_duplicates)} rows")

    result = await toolkit.drop_duplicates(dataframe=data_with_duplicates)
    if result.success:
        print(f"  After: {result.output_rows} rows")
        print(f"  Removed: {result.duplicates_removed} duplicates")
    print()

    print("=== Examples completed successfully! ===")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())