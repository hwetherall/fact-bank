"""
Tests for the extraction module.

Run with: pytest tests/test_extraction.py -v
"""

import pytest
from pathlib import Path

from src.extraction.normalizer import (
    ExtractedDocument,
    Section,
    ContentType,
    normalize_to_prompt,
)
from src.storage.models import (
    Fact,
    Document,
    SourceType,
    RelevanceLevel,
    Believability,
    STANDARD_CHAPTERS,
    BASE_BELIEVABILITY_SCORES,
)
from src.scoring.scorer import FactScorer, get_chapter_summary


class TestExtractedDocument:
    """Tests for ExtractedDocument dataclass."""
    
    def test_create_document(self):
        """Test creating an ExtractedDocument."""
        doc = ExtractedDocument(
            filename="test.pdf",
            file_type="pdf",
        )
        
        assert doc.filename == "test.pdf"
        assert doc.file_type == "pdf"
        assert doc.sections == []
        assert doc.total_pages is None
    
    def test_add_section(self):
        """Test adding sections to a document."""
        doc = ExtractedDocument(
            filename="test.pdf",
            file_type="pdf",
        )
        
        doc.add_section(
            content="Sample text content",
            content_type=ContentType.PROSE,
            page_or_sheet=1,
        )
        
        doc.add_section(
            content="| Col1 | Col2 |\n| --- | --- |\n| A | B |",
            content_type=ContentType.TABLE,
            page_or_sheet=2,
            title="Sample Table",
        )
        
        assert len(doc.sections) == 2
        assert len(doc.get_prose_only()) == 1
        assert len(doc.get_tables_only()) == 1
    
    def test_get_full_text(self):
        """Test getting full text from document."""
        doc = ExtractedDocument(
            filename="test.pdf",
            file_type="pdf",
        )
        
        doc.add_section(
            content="First section",
            content_type=ContentType.PROSE,
            page_or_sheet=1,
        )
        
        doc.add_section(
            content="Second section",
            content_type=ContentType.PROSE,
            page_or_sheet=2,
        )
        
        full_text = doc.get_full_text()
        assert "First section" in full_text
        assert "Second section" in full_text


class TestNormalizeToPrompt:
    """Tests for normalize_to_prompt function."""
    
    def test_normalize_basic_document(self):
        """Test normalizing a basic document."""
        doc = ExtractedDocument(
            filename="report.pdf",
            file_type="pdf",
            total_pages=5,
        )
        
        doc.add_section(
            content="Company overview text.",
            content_type=ContentType.PROSE,
            page_or_sheet=1,
        )
        
        result = normalize_to_prompt(doc)
        
        assert "report.pdf" in result
        assert "PDF" in result
        assert "Company overview text" in result
        assert "Text Content" in result


class TestFactModel:
    """Tests for the Fact Pydantic model."""
    
    def test_create_fact(self):
        """Test creating a Fact."""
        fact = Fact(
            id="test-123",
            content="The company generated $5M in revenue in 2024.",
            source_documents=["presentation.pdf"],
            source_type=SourceType.COMPANY_PRESENTATION,
            believability=Believability.NEEDS_VERIFICATION,
            relevance=RelevanceLevel.CRITICAL,
            chapter_relevance={"Financial": 0.9, "Summary": 0.7},
        )
        
        assert fact.id == "test-123"
        assert fact.believability == Believability.NEEDS_VERIFICATION.value
        assert fact.relevance == RelevanceLevel.CRITICAL.value
        assert fact.usage_count == 0
    
    def test_get_top_chapters(self):
        """Test getting top relevant chapters."""
        fact = Fact(
            id="test-123",
            content="Test fact",
            source_documents=["test.pdf"],
            source_type=SourceType.MARKET_RESEARCH,
            believability=Believability.NEEDS_VERIFICATION,
            relevance=RelevanceLevel.CHAPTER_SPECIFIC,
            chapter_relevance={
                "Financial": 0.9,
                "Market": 0.7,
                "Legal": 0.3,
                "Technical": 0.1,
            },
        )
        
        top = fact.get_top_chapters(2)
        
        assert len(top) == 2
        assert top[0][0] == "Financial"
        assert top[0][1] == 0.9
        assert top[1][0] == "Market"
    
    def test_is_relevant_to_chapter(self):
        """Test chapter relevance check."""
        fact = Fact(
            id="test-123",
            content="Test fact",
            source_documents=["test.pdf"],
            source_type=SourceType.FINANCIALS,
            believability=Believability.VERIFIED,
            relevance=RelevanceLevel.CRITICAL,
            chapter_relevance={
                "Financial": 0.9,
                "Market": 0.4,
            },
        )
        
        assert fact.is_relevant_to_chapter("Financial", threshold=0.5)
        assert not fact.is_relevant_to_chapter("Market", threshold=0.5)
        assert fact.is_relevant_to_chapter("Market", threshold=0.3)
    
    def test_to_display_dict(self):
        """Test converting fact to display dictionary."""
        fact = Fact(
            id="abcd1234-5678-90ef-ghij-klmnopqrstuv",
            content="Revenue was $10M in Q4 2024.",
            source_documents=["report.pdf", "data.xlsx"],
            source_type=SourceType.FINANCIALS,
            believability=Believability.VERIFIED,
            relevance=RelevanceLevel.CRITICAL,
            chapter_relevance={"Financial": 0.95, "Summary": 0.6},
        )
        
        display = fact.to_display_dict()
        
        assert display["ID"] == "abcd1234"
        assert "Revenue" in display["Content"]
        assert "report.pdf" in display["Source(s)"]
        assert display["Believability"] == "Verified"
        assert display["Relevance"] == "Critical"


class TestFactScorer:
    """Tests for the FactScorer class."""
    
    def test_validate_fact(self):
        """Test fact validation."""
        scorer = FactScorer()
        
        # Valid fact
        valid_fact = Fact(
            id="test",
            content="A valid fact with enough content.",
            source_documents=["test.pdf"],
            source_type=SourceType.MARKET_RESEARCH,
            believability=Believability.VERIFIED,
            relevance=RelevanceLevel.CRITICAL,
            chapter_relevance={"Financial": 0.8},
        )
        
        issues = scorer.validate_fact(valid_fact)
        assert len(issues) == 0
        
        # Invalid fact (too short content)
        invalid_fact = Fact(
            id="test",
            content="Hi",
            source_documents=["test.pdf"],
            source_type=SourceType.MARKET_RESEARCH,
            believability=Believability.VERIFIED,
            relevance=RelevanceLevel.CRITICAL,
            chapter_relevance={},
        )
        
        issues = scorer.validate_fact(invalid_fact)
        assert len(issues) > 0


class TestBaseBelievabilityScores:
    """Tests for base believability score constants."""
    
    def test_financials_verified(self):
        """Test that financials default to verified believability."""
        assert BASE_BELIEVABILITY_SCORES[SourceType.FINANCIALS] == Believability.VERIFIED
    
    def test_market_research_needs_verification(self):
        """Test market research needs verification."""
        assert BASE_BELIEVABILITY_SCORES[SourceType.MARKET_RESEARCH] == Believability.NEEDS_VERIFICATION
    
    def test_company_presentation_needs_verification(self):
        """Test company presentation needs verification."""
        assert BASE_BELIEVABILITY_SCORES[SourceType.COMPANY_PRESENTATION] == Believability.NEEDS_VERIFICATION


class TestStandardChapters:
    """Tests for standard chapters constant."""
    
    def test_all_chapters_present(self):
        """Test all expected chapters are defined."""
        expected = [
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
        
        # Ensure we haven't accidentally changed the standard chapter set/order
        assert STANDARD_CHAPTERS == expected


class TestChapterSummary:
    """Tests for chapter summary function."""
    
    def test_empty_facts(self):
        """Test with empty fact list."""
        summary = get_chapter_summary([])
        
        assert len(summary) == 10
        for chapter in STANDARD_CHAPTERS:
            assert summary[chapter]["fact_count"] == 0
    
    def test_with_facts(self):
        """Test summary with actual facts."""
        facts = [
            Fact(
                id="1",
                content="Finance and operations fact",
                source_documents=["test.pdf"],
                source_type=SourceType.FINANCIALS,
                believability=Believability.VERIFIED,
                relevance=RelevanceLevel.CRITICAL,
                chapter_relevance={
                    "Finance & Operations": 0.95,
                    "Unit Economics": 0.6,
                },
            ),
            Fact(
                id="2",
                content="Market and competitive fact",
                source_documents=["test.pdf"],
                source_type=SourceType.MARKET_RESEARCH,
                believability=Believability.NEEDS_VERIFICATION,
                relevance=RelevanceLevel.CHAPTER_SPECIFIC,
                chapter_relevance={
                    "Market Research": 0.85,
                    "Competitive Analysis": 0.6,
                },
            ),
        ]
        
        summary = get_chapter_summary(facts)
        
        assert summary["Finance & Operations"]["fact_count"] == 1
        assert summary["Market Research"]["fact_count"] == 1
        # Second-highest chapter for first fact still counted at threshold
        assert summary["Unit Economics"]["fact_count"] == 1  # 0.6 >= 0.5
        # Chapter with no relevance scores should have zero facts
        assert summary["Legal & IP"]["fact_count"] == 0
        # Check new relevance breakdown
        assert summary["Finance & Operations"]["relevance_breakdown"]["critical"] == 1
        assert summary["Market Research"]["relevance_breakdown"]["chapter_specific"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

