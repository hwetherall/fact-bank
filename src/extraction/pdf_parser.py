"""
PDF document parser using PyMuPDF.

Extracts text and tables from PDF files.
"""

import logging
from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF

from .normalizer import ExtractedDocument, ContentType

logger = logging.getLogger(__name__)


def extract_pdf(file_path: str | Path | BinaryIO, filename: str | None = None) -> ExtractedDocument:
    """
    Extract text and tables from a PDF file.
    
    Args:
        file_path: Path to the PDF file or file-like object
        filename: Original filename (required if file_path is BinaryIO)
        
    Returns:
        ExtractedDocument with all content
    """
    # Handle different input types
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        filename = filename or file_path.name
        doc = fitz.open(str(file_path))
    else:
        # File-like object (e.g., from Streamlit upload)
        if filename is None:
            filename = "uploaded.pdf"
        file_bytes = file_path.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    extracted = ExtractedDocument(
        filename=filename,
        file_type="pdf",
        total_pages=len(doc),
    )
    
    logger.info(f"Processing PDF: {filename} ({len(doc)} pages)")
    
    for page_num, page in enumerate(doc, 1):
        # Extract prose text
        text = page.get_text("text")
        if text.strip():
            # Clean up the text
            cleaned_text = _clean_pdf_text(text)
            if cleaned_text:
                extracted.add_section(
                    content=cleaned_text,
                    content_type=ContentType.PROSE,
                    page_or_sheet=page_num,
                )
        
        # Extract tables
        tables = _extract_tables_from_page(page)
        for i, table_md in enumerate(tables):
            extracted.add_section(
                content=table_md,
                content_type=ContentType.TABLE,
                page_or_sheet=page_num,
                title=f"Table {i + 1}",
            )
    
    doc.close()
    
    logger.info(
        f"Extracted {len(extracted.sections)} sections from {filename} "
        f"({len(extracted.get_prose_only())} prose, {len(extracted.get_tables_only())} tables)"
    )
    
    return extracted


def _clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text.
    
    - Remove excessive whitespace
    - Fix common OCR issues
    - Normalize line breaks
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        # Strip whitespace
        line = line.strip()
        
        # Skip empty lines in sequence
        if not line and cleaned_lines and not cleaned_lines[-1]:
            continue
        
        # Skip page numbers and common noise
        if _is_noise_line(line):
            continue
        
        cleaned_lines.append(line)
    
    # Join and normalize
    result = "\n".join(cleaned_lines)
    
    # Remove excessive blank lines
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    
    return result.strip()


def _is_noise_line(line: str) -> bool:
    """Check if a line is likely noise (page numbers, headers, etc.)."""
    # Empty or very short
    if len(line) < 3:
        return True
    
    # Pure numbers (likely page numbers)
    if line.isdigit():
        return True
    
    # Common patterns
    noise_patterns = [
        "page ",
        "confidential",
        "all rights reserved",
        "©",
    ]
    
    line_lower = line.lower()
    return any(pattern in line_lower for pattern in noise_patterns)


def _extract_tables_from_page(page: fitz.Page) -> list[str]:
    """
    Extract tables from a PDF page and convert to markdown.
    
    Uses PyMuPDF's table detection feature.
    """
    tables = []
    
    try:
        # Find tables on the page
        found_tables = page.find_tables()
        
        for table in found_tables:
            # Extract table data
            data = table.extract()
            
            if not data or len(data) < 2:
                continue
            
            # Convert to markdown table
            markdown = _table_data_to_markdown(data)
            if markdown:
                tables.append(markdown)
                
    except Exception as e:
        logger.warning(f"Error extracting tables from page: {e}")
    
    return tables


def _table_data_to_markdown(data: list[list]) -> str:
    """
    Convert table data to markdown format.
    
    Args:
        data: 2D list of cell values
        
    Returns:
        Markdown table string
    """
    if not data:
        return ""
    
    # Clean cell values
    cleaned_data = []
    for row in data:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                # Clean and truncate long cells
                cell_str = str(cell).strip().replace("|", "\\|")
                if len(cell_str) > 100:
                    cell_str = cell_str[:97] + "..."
                cleaned_row.append(cell_str)
        cleaned_data.append(cleaned_row)
    
    if not cleaned_data:
        return ""
    
    # Ensure consistent column count
    max_cols = max(len(row) for row in cleaned_data)
    for row in cleaned_data:
        while len(row) < max_cols:
            row.append("")
    
    # Build markdown table
    lines = []
    
    # Header row
    header = cleaned_data[0]
    lines.append("| " + " | ".join(header) + " |")
    
    # Separator
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    # Data rows
    for row in cleaned_data[1:]:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def extract_pdf_metadata(file_path: str | Path) -> dict:
    """
    Extract metadata from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary of metadata
    """
    doc = fitz.open(str(file_path))
    
    metadata = {
        "page_count": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "creator": doc.metadata.get("creator", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
    }
    
    doc.close()
    return metadata

