"""
Data models for Factor.

Defines the Fact and Document schemas using Pydantic.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# Standard chapters for investment memos
STANDARD_CHAPTERS = [
    "Opportunity Validation",
    "Product & Technology",
    "Market Research",
    "Competitive Analysis",
    "Revenue Model",
    "Go-to-Market",
    "Unit Economics",
    "Finance & Operations",
    "Talent & Team",
    "Legal & IP",
]

# Chapter descriptions for LLM prompts
CHAPTER_DESCRIPTIONS = {
    "Opportunity Validation": "Customer need, demand evidence, market timing, problem-solution fit",
    "Product & Technology": "What's being built, technical feasibility, IP, product roadmap",
    "Market Research": "Market size (TAM/SAM/SOM), structure, dynamics, trends, segments",
    "Competitive Analysis": "Competitive landscape, positioning, differentiation, moats",
    "Revenue Model": "How money is made, pricing strategy, unit economics, monetization",
    "Go-to-Market": "Customer acquisition, sales strategy, channels, partnerships",
    "Unit Economics": "CAC, LTV, margins, payback period, key financial metrics",
    "Finance & Operations": "Financial projections, assumptions, capital needs, operations",
    "Talent & Team": "Team capabilities, leadership, hiring plans, culture, advisors",
    "Legal & IP": "Corporate structure, compliance, intellectual property, contracts",
}


class SourceType(str, Enum):
    """Type of source document."""
    COMPANY_PRESENTATION = "company_presentation"
    MARKET_RESEARCH = "market_research"
    FINANCIALS = "financials"


class RelevanceLevel(str, Enum):
    """
    Relevance level of a fact for investment memo generation.
    
    - CRITICAL: Top tier facts relevant to most/all chapters. Should be <10% of all facts.
    - CHAPTER_SPECIFIC: Very important facts but only to certain chapters (e.g., burn rate for Unit Economics).
    - ADDITIONAL_CONTEXT: Interesting contextual facts. Memo could be written without them.
    - NOISE: Low value or irrelevant facts. Can be hidden and excluded from exports.
    """
    CRITICAL = "critical"
    CHAPTER_SPECIFIC = "chapter_specific"
    ADDITIONAL_CONTEXT = "additional_context"
    NOISE = "noise"


# Legacy alias for backwards compatibility during migration
ImportanceLevel = RelevanceLevel


class Believability(str, Enum):
    """
    How much we should believe this fact without additional verification.
    
    - VERIFIED: Should be believed as stated. No secondary verification needed.
              Example: Team member names/roles from pitch deck.
    - NEEDS_VERIFICATION: Claims that may be exaggerated. Need external validation.
              Example: TAM claims of $1B+ that could be inflated.
    - OPINION: Subjective statements that cannot be validated.
              Example: "Our team is legendary" - this is marketing, not a fact.
    """
    VERIFIED = "verified"
    NEEDS_VERIFICATION = "needs_verification"
    OPINION = "opinion"


# Base believability by source type (used when LLM doesn't provide specific assessment)
BASE_BELIEVABILITY_SCORES = {
    SourceType.FINANCIALS: Believability.VERIFIED,
    SourceType.MARKET_RESEARCH: Believability.NEEDS_VERIFICATION,
    SourceType.COMPANY_PRESENTATION: Believability.NEEDS_VERIFICATION,
}


class Fact(BaseModel):
    """
    A single atomic fact extracted from a document.
    
    Represents a verifiable claim that can be used in investment memo generation.
    """
    
    id: str = Field(
        description="Unique identifier (UUID)"
    )
    content: str = Field(
        description="The atomic fact text (single verifiable claim)"
    )
    source_quote: str | None = Field(
        default=None,
        description="Original text snippet the fact was extracted from"
    )
    source_documents: list[str] = Field(
        default_factory=list,
        description="List of source document filenames"
    )
    source_type: SourceType = Field(
        description="Type of the primary source document"
    )
    believability: Believability = Field(
        default=Believability.NEEDS_VERIFICATION,
        description="How much we should believe this fact without additional verification"
    )
    relevance: RelevanceLevel = Field(
        description="Relevance level for investment memo generation"
    )
    chapter_relevance: dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores for each standard chapter (0.0-1.0)"
    )
    extraction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the fact was extracted"
    )
    usage_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this fact has been used in generation"
    )
    used_in_chapters: list[str] = Field(
        default_factory=list,
        description="List of chapters where this fact was used"
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Embedding vector for deduplication"
    )
    
    # Legacy field aliases for backwards compatibility
    @property
    def confidence(self) -> float:
        """Legacy property - maps believability to a numeric score."""
        mapping = {
            Believability.VERIFIED: 0.95,
            Believability.NEEDS_VERIFICATION: 0.70,
            Believability.OPINION: 0.40,
        }
        return mapping.get(self.believability, 0.70)
    
    @property
    def importance(self) -> str:
        """Legacy property - maps relevance to old importance values."""
        mapping = {
            RelevanceLevel.CRITICAL: "high",
            RelevanceLevel.CHAPTER_SPECIFIC: "medium",
            RelevanceLevel.ADDITIONAL_CONTEXT: "low",
            RelevanceLevel.NOISE: "low",
        }
        return mapping.get(self.relevance, "medium")
    
    class Config:
        use_enum_values = True
    
    def get_top_chapters(self, n: int = 3) -> list[tuple[str, float]]:
        """
        Get the top N most relevant chapters.
        
        Args:
            n: Number of chapters to return
            
        Returns:
            List of (chapter_name, relevance_score) tuples
        """
        sorted_chapters = sorted(
            self.chapter_relevance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_chapters[:n]
    
    def is_relevant_to_chapter(
        self, 
        chapter: str, 
        threshold: float = 0.5
    ) -> bool:
        """Check if fact is relevant to a specific chapter."""
        return self.chapter_relevance.get(chapter, 0.0) >= threshold
    
    def is_noise(self) -> bool:
        """Check if this fact is categorized as noise."""
        return self.relevance == RelevanceLevel.NOISE
    
    def to_display_dict(self) -> dict:
        """Convert to a dictionary suitable for display."""
        top_chapters = self.get_top_chapters(3)
        top_chapters_str = ", ".join(
            f"{ch} ({score:.0%})" for ch, score in top_chapters if score > 0.3
        )
        
        # Format relevance for display
        relevance_display = {
            RelevanceLevel.CRITICAL: "Critical",
            RelevanceLevel.CHAPTER_SPECIFIC: "Chapter Specific",
            RelevanceLevel.ADDITIONAL_CONTEXT: "Additional Context",
            RelevanceLevel.NOISE: "Noise",
        }
        
        # Format believability for display
        believability_display = {
            Believability.VERIFIED: "Verified",
            Believability.NEEDS_VERIFICATION: "Needs Verification",
            Believability.OPINION: "Opinion",
        }
        
        return {
            "ID": self.id[:8],
            "Content": self.content,
            "Source(s)": ", ".join(self.source_documents),
            "Believability": believability_display.get(self.believability, str(self.believability)),
            "Relevance": relevance_display.get(self.relevance, str(self.relevance)),
            "Top Chapters": top_chapters_str or "General",
            "Usage Count": self.usage_count,
            "Source Quote": self.source_quote or self.content,
        }


class Document(BaseModel):
    """
    A source document that has been processed.
    
    Stores metadata and raw content for traceability.
    """
    
    id: str = Field(
        description="Unique identifier (UUID)"
    )
    filename: str = Field(
        description="Original filename"
    )
    file_type: Literal["pdf", "xlsx", "xls"] = Field(
        description="File extension/type"
    )
    source_type: SourceType = Field(
        description="Classification of the document"
    )
    upload_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the document was uploaded"
    )
    raw_content: str = Field(
        description="Extracted text/markdown content"
    )
    page_count: int | None = Field(
        default=None,
        description="Number of pages (for PDFs)"
    )
    sheet_count: int | None = Field(
        default=None,
        description="Number of sheets (for Excel files)"
    )
    fact_count: int = Field(
        default=0,
        description="Number of facts extracted from this document"
    )
    
    class Config:
        use_enum_values = True


class FactCorrection(BaseModel):
    """
    A logged correction made to a fact by the user.
    
    Used to track LLM errors for potential model improvement.
    """
    
    id: str = Field(
        description="Unique identifier (UUID)"
    )
    fact_id: str = Field(
        description="ID of the fact that was corrected"
    )
    field_name: str = Field(
        description="Name of the field that was changed (e.g., 'content', 'believability')"
    )
    old_value: str = Field(
        description="The original value before correction"
    )
    new_value: str = Field(
        description="The new corrected value"
    )
    correction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the correction was made"
    )
    
    class Config:
        use_enum_values = True
    
    
class ExtractedFactRaw(BaseModel):
    """
    Raw fact as returned by the LLM before post-processing.
    
    This is an intermediate representation before UUIDs and 
    final processing is applied.
    """
    
    content: str = Field(
        description="The fact content"
    )
    relevance: RelevanceLevel = Field(
        description="Relevance level for investment memo generation"
    )
    believability: Believability = Field(
        default=Believability.NEEDS_VERIFICATION,
        description="How much we should believe this fact without verification"
    )
    chapter_relevance: dict[str, float] = Field(
        description="Relevance scores for chapters"
    )
    recency_indicator: Literal["current", "dated", "unknown"] | None = Field(
        default=None,
        description='LLM-assessed recency of the information: "current", "dated", or "unknown"',
    )
    
    # Legacy field aliases for backwards compatibility
    @property
    def importance(self) -> str:
        """Legacy property for backwards compatibility."""
        mapping = {
            RelevanceLevel.CRITICAL: "high",
            RelevanceLevel.CHAPTER_SPECIFIC: "medium",
            RelevanceLevel.ADDITIONAL_CONTEXT: "low",
            RelevanceLevel.NOISE: "low",
        }
        return mapping.get(self.relevance, "medium")
    
    class Config:
        use_enum_values = True
    
    
class FactExtractionResponse(BaseModel):
    """Response schema for fact extraction LLM calls."""
    
    facts: list[ExtractedFactRaw] = Field(
        description="List of extracted facts"
    )

