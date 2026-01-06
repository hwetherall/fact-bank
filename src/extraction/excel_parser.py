"""
Excel document parser using pandas and openpyxl.

Extracts data from Excel spreadsheets and converts to markdown tables.
"""

import logging
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .normalizer import ExtractedDocument, ContentType

logger = logging.getLogger(__name__)


def extract_excel(
    file_path: str | Path | BinaryIO, 
    filename: str | None = None
) -> ExtractedDocument:
    """
    Extract data from an Excel file.
    
    Args:
        file_path: Path to the Excel file or file-like object
        filename: Original filename (required if file_path is BinaryIO)
        
    Returns:
        ExtractedDocument with all sheets as table sections
    """
    # Determine filename and file type
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        filename = filename or file_path.name
        file_type = file_path.suffix.lower().lstrip(".")
    else:
        # File-like object
        if filename is None:
            filename = "uploaded.xlsx"
        file_type = filename.split(".")[-1].lower() if "." in filename else "xlsx"
    
    # Ensure valid file type
    if file_type not in ("xlsx", "xls"):
        file_type = "xlsx"
    
    extracted = ExtractedDocument(
        filename=filename,
        file_type=file_type,
    )
    
    logger.info(f"Processing Excel file: {filename}")
    
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path, engine="openpyxl" if file_type == "xlsx" else "xlrd")
        sheet_names = excel_file.sheet_names
        extracted.total_pages = len(sheet_names)
        
        for sheet_name in sheet_names:
            # Read sheet into DataFrame
            df = pd.read_excel(
                excel_file, 
                sheet_name=sheet_name,
                header=0,  # Use first row as header
            )
            
            # Skip empty sheets
            if df.empty:
                logger.debug(f"Skipping empty sheet: {sheet_name}")
                continue
            
            # Clean the DataFrame
            df = _clean_dataframe(df)
            
            if df.empty:
                continue
            
            # Convert to markdown
            markdown_table = _dataframe_to_markdown(df)
            
            if markdown_table:
                extracted.add_section(
                    content=markdown_table,
                    content_type=ContentType.TABLE,
                    page_or_sheet=sheet_name,
                    title=sheet_name,
                )
        
        excel_file.close()
        
    except Exception as e:
        logger.error(f"Error processing Excel file {filename}: {e}")
        raise
    
    logger.info(
        f"Extracted {len(extracted.sections)} sheets from {filename}"
    )
    
    return extracted


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame for better extraction.
    
    - Remove completely empty rows and columns
    - Clean up column names
    - Handle NaN values
    """
    # Remove completely empty rows
    df = df.dropna(how="all")
    
    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")
    
    if df.empty:
        return df
    
    # Clean column names
    df.columns = [_clean_column_name(col) for col in df.columns]
    
    # Fill NaN with empty string for display
    df = df.fillna("")
    
    # Convert all values to strings and clean
    for col in df.columns:
        df[col] = df[col].apply(_clean_cell_value)
    
    # Remove rows that are all empty strings
    df = df[~(df == "").all(axis=1)]
    
    return df


def _clean_column_name(name) -> str:
    """Clean a column name."""
    if pd.isna(name):
        return "Column"
    
    name_str = str(name).strip()
    
    # Handle unnamed columns
    if name_str.startswith("Unnamed:"):
        return "Column"
    
    # Remove newlines and excessive whitespace
    name_str = " ".join(name_str.split())
    
    # Escape pipe characters for markdown
    name_str = name_str.replace("|", "\\|")
    
    return name_str


def _clean_cell_value(value) -> str:
    """Clean a cell value for display."""
    # Handle Series objects (can occur with merged cells or complex structures)
    if isinstance(value, pd.Series):
        # Convert Series to its first value or string representation
        if len(value) == 1:
            value = value.iloc[0]
        else:
            value = value.to_string(index=False)
    
    # Now safely check for NA - use try/except to handle edge cases
    try:
        is_na = pd.isna(value)
        # If is_na is not a scalar bool, treat as non-NA
        if not isinstance(is_na, bool):
            is_na = False
    except (ValueError, TypeError):
        is_na = False
    
    if is_na or value == "":
        return ""
    
    value_str = str(value).strip()
    
    # Handle floats that are actually integers
    if isinstance(value, float) and value.is_integer():
        value_str = str(int(value))
    
    # Remove newlines
    value_str = " ".join(value_str.split())
    
    # Escape pipe characters for markdown
    value_str = value_str.replace("|", "\\|")
    
    # Truncate very long values
    if len(value_str) > 200:
        value_str = value_str[:197] + "..."
    
    return value_str


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a markdown table.
    
    Args:
        df: The DataFrame to convert
        
    Returns:
        Markdown table string
    """
    if df.empty:
        return ""
    
    lines = []
    
    # Header row
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    
    # Separator
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Data rows (limit to 500 rows to avoid huge outputs)
    max_rows = 500
    if len(df) > max_rows:
        logger.warning(f"DataFrame has {len(df)} rows, truncating to {max_rows}")
    
    for _, row in df.head(max_rows).iterrows():
        row_values = [str(v) for v in row.values]
        lines.append("| " + " | ".join(row_values) + " |")
    
    if len(df) > max_rows:
        lines.append(f"| ... | *{len(df) - max_rows} more rows truncated* |")
    
    return "\n".join(lines)


def extract_excel_metadata(file_path: str | Path) -> dict:
    """
    Extract metadata from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Dictionary of metadata
    """
    file_path = Path(file_path)
    file_type = file_path.suffix.lower().lstrip(".")
    
    excel_file = pd.ExcelFile(
        file_path, 
        engine="openpyxl" if file_type == "xlsx" else "xlrd"
    )
    
    sheet_info = {}
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheet_info[sheet_name] = {
            "rows": len(df),
            "columns": len(df.columns),
        }
    
    excel_file.close()
    
    return {
        "sheet_count": len(excel_file.sheet_names),
        "sheet_names": excel_file.sheet_names,
        "sheets": sheet_info,
    }

