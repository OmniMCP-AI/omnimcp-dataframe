#!/usr/bin/env python3
"""
Very simple test to check if the fixes work without external dependencies.
"""

import sys
import os
import asyncio

# Add the parent directory to the path so we can import omnimcp_dataframe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_imports():
    """Test that we can import the async modules."""
    try:
        from omnimcp_dataframe.unified import UnifiedDataFrameToolkit
        print("✓ UnifiedDataFrameToolkit imported successfully")

        # Create toolkit instance
        toolkit = UnifiedDataFrameToolkit()
        print("✓ Toolkit instance created successfully")

        # Get available tools
        tools = toolkit.get_available_tools()
        print(f"✓ Available tools: {len(tools)} tools found")

        return True
    except Exception as e:
        print(f"✗ Import/creation failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_imports())
    if success:
        print("\n=== Basic test passed ===")
    else:
        print("\n=== Basic test failed ===")