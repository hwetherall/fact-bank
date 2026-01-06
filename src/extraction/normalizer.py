"""
Content normalizer for extracted documents.

Provides unified data structures and formatting for LLM consumption.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ContentType(str, Enum):
    """Type of content within a document section."""
    PROSE = "prose"
    TABLE = "table"


@dataclass
class Section:
    """
    A section of extracted content from a document.
    
    Attributes:
        content: The text or table content
        content_type: Whether this is prose or tabular data
        page_or_sheet: Page number (PDF) or sheet name (Excel)
        title: Optional section title if detected
    """
    content: str
    content_type: ContentType
    page_or_sheet: str | int
    title: str | None = None
    
    def to_markdown(self) -> str:
        """Convert section to markdown format."""
        parts = []
        
        # Add location header
        location = f"Page {self.page_or_sheet}" if isinstance(self.page_or_sheet, int) else f"Sheet: {self.page_or_sheet}"
        parts.append(f"### {location}")
        
        if self.title:
            parts.append(f"**{self.title}**")
        
        # Mark content type for LLM clarity
        if self.content_type == ContentType.TABLE:
            parts.append("[TABLE DATA]")
        
        parts.append(self.content)
        
        return "\n".join(parts)


@dataclass
class ExtractedDocument:
    """
    A document with all extracted content.
    
    Attributes:
        filename: Original filename
        file_type: File extension (pdf, xlsx, xls)
        sections: List of extracted sections
        total_pages: Number of pages (PDF) or sheets (Excel)
        metadata: Additional document metadata
    """
    filename: str
    file_type: Literal["pdf", "xlsx", "xls"]
    sections: list[Section] = field(default_factory=list)
    total_pages: int | None = None
    metadata: dict = field(default_factory=dict)
    
    def add_section(
        self,
        content: str,
        content_type: ContentType,
        page_or_sheet: str | int,
        title: str | None = None,
    ):
        """Add a section to the document."""
        self.sections.append(Section(
            content=content,
            content_type=content_type,
            page_or_sheet=page_or_sheet,
            title=title,
        ))
    
    def get_full_text(self) -> str:
        """Get all content as plain text."""
        return "\n\n".join(s.content for s in self.sections)
    
    def get_tables_only(self) -> list[Section]:
        """Get only table sections."""
        return [s for s in self.sections if s.content_type == ContentType.TABLE]
    
    def get_prose_only(self) -> list[Section]:
        """Get only prose sections."""
        return [s for s in self.sections if s.content_type == ContentType.PROSE]


def normalize_to_prompt(doc: ExtractedDocument) -> str:
    """
    Convert an extracted document to a format suitable for LLM prompts.
    
    The output clearly delineates between prose and table content,
    and includes source information for traceability.
    
    Args:
        doc: The extracted document
        
    Returns:
        Formatted string for LLM consumption
    """
    parts = [
        f"# Document: {doc.filename}",
        f"**Type:** {doc.file_type.upper()}",
    ]
    
    if doc.total_pages:
        parts.append(f"**Total Pages/Sheets:** {doc.total_pages}")
    
    parts.append("")
    parts.append("---")
    parts.append("")
    
    # Group sections by type for clarity
    prose_sections = doc.get_prose_only()
    table_sections = doc.get_tables_only()
    
    if prose_sections:
        parts.append("## Text Content")
        parts.append("")
        for section in prose_sections:
            parts.append(section.to_markdown())
            parts.append("")
    
    if table_sections:
        parts.append("## Tabular Data")
        parts.append("")
        parts.append("*Note: Tables often contain dense factual information. Pay special attention to numerical data, dates, and named entities.*")
        parts.append("")
        for section in table_sections:
            parts.append(section.to_markdown())
            parts.append("")
    
    return "\n".join(parts)


def combine_documents_for_prompt(docs: list[ExtractedDocument]) -> str:
    """
    Combine multiple documents into a single prompt.
    
    Args:
        docs: List of extracted documents
        
    Returns:
        Combined prompt string
    """
    parts = [
        "# Source Documents",
        "",
        f"The following content has been extracted from {len(docs)} document(s).",
        "Extract all relevant facts from each document.",
        "",
        "---",
        "",
    ]
    
    for i, doc in enumerate(docs, 1):
        parts.append(f"## Document {i}: {doc.filename}")
        parts.append("")
        parts.append(normalize_to_prompt(doc))
        parts.append("")
        parts.append("---")
        parts.append("")
    
    return "\n".join(parts)

