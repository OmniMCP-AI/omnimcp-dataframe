"""Prompt templates for intelligent DataFrame operations."""

IDENTIFY_JOIN_KEYS_PROMPT = """
You are a data analysis expert. Analyze two dataframes and identify the best columns for joining them.

Left Dataframe:
{df1}

Right Dataframe:
{df2}

Your task:
1. Examine the column names and data types
2. Look at sample data to understand the content
3. Identify columns that could be used as join keys
4. Consider both exact matches and semantic similarities
5. Assess the confidence of your recommendations

Return your analysis in the following format:
- can_join: boolean (whether the dataframes can be joined)
- left_columns: list of column names from left dataframe
- right_columns: list of column names from right dataframe
- confidence: float between 0.0 and 1.0
- reason: explanation of your choice

Focus on finding columns that represent the same entities (IDs, names, codes, etc.).
Consider fuzzy matching for names, partial matches for codes, etc.
If no suitable join keys are found, set can_join to false and explain why.
"""