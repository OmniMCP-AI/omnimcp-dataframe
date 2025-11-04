"""
OmniMCP DataFrame Toolkit

A comprehensive DataFrame manipulation toolkit for MCP servers and data processing applications.
Built on Polars for high performance operations.
"""

from .server import DataFrameServer
from .models import JoinKeyAnalysis
from .utils import parse_human_readable_number
from .version import __version__

__all__ = [
    "DataFrameServer",
    "JoinKeyAnalysis",
    "parse_human_readable_number",
    "__version__",
]

# Convenience imports for easy access
from .server import (
    DataFrameToolkit,
    DataFrameOperations,
)