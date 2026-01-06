"""
Tests for the storage repository.

Run with: pytest tests/test_repository.py -v
"""

import os
import pytest
import tempfile
from datetime import datetime

from src.storage.repository import FactRepository
from src.storage.models import Fact, Document, SourceType, RelevanceLevel, Believability


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def repo(temp_db):
    """Create a repository with temporary database."""
    return FactRepository(db_path=temp_db)


@pytest.fixture
def sample_fact():
    """Create a sample fact for testing."""
    return Fact(
        id="test-fact-001",
        content="The company achieved $10M ARR in Q4 2024.",
        source_documents=["presentation.pdf"],
        source_type=SourceType.COMPANY_PRESENTATION,
        believability=Believability.NEEDS_VERIFICATION,
        relevance=RelevanceLevel.CRITICAL,
        chapter_relevance={
            "Financial": 0.95,
            "Summary": 0.7,
            "Strategy": 0.5,
        },
        extraction_timestamp=datetime(2024, 12, 1, 10, 0, 0),
        usage_count=0,
        used_in_chapters=[],
    )


class TestFactRepository:
    """Tests for FactRepository class."""
    
    def test_init_creates_tables(self, repo):
        """Test that initialization creates required tables."""
        # If we can insert and retrieve, tables exist
        fact = Fact(
            id="test",
            content="Test fact",
            source_documents=["test.pdf"],
            source_type=SourceType.FINANCIALS,
            believability=Believability.VERIFIED,
            relevance=RelevanceLevel.CHAPTER_SPECIFIC,
            chapter_relevance={},
        )
        
        repo.insert_fact(fact)
        retrieved = repo.get_fact("test")
        
        assert retrieved is not None
        assert retrieved.content == "Test fact"
    
    def test_insert_and_get_fact(self, repo, sample_fact):
        """Test inserting and retrieving a fact."""
        repo.insert_fact(sample_fact)
        
        retrieved = repo.get_fact(sample_fact.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_fact.id
        assert retrieved.content == sample_fact.content
        assert retrieved.believability == sample_fact.believability
        assert retrieved.relevance == sample_fact.relevance
    
    def test_get_nonexistent_fact(self, repo):
        """Test getting a fact that doesn't exist."""
        result = repo.get_fact("nonexistent-id")
        assert result is None
    
    def test_insert_facts_batch(self, repo):
        """Test batch inserting multiple facts."""
        facts = [
            Fact(
                id=f"fact-{i}",
                content=f"Test fact {i}",
                source_documents=["test.pdf"],
                source_type=SourceType.MARKET_RESEARCH,
                believability=Believability.NEEDS_VERIFICATION,
                relevance=RelevanceLevel.CHAPTER_SPECIFIC,
                chapter_relevance={"Market": 0.7},
            )
            for i in range(5)
        ]
        
        count = repo.insert_facts(facts)
        
        assert count == 5
        
        all_facts = repo.get_all_facts()
        assert len(all_facts) == 5
    
    def test_get_facts_by_relevance(self, repo):
        """Test filtering facts by relevance level."""
        facts = [
            Fact(
                id="critical-1",
                content="Critical fact for all chapters",
                source_documents=["test.pdf"],
                source_type=SourceType.FINANCIALS,
                believability=Believability.VERIFIED,
                relevance=RelevanceLevel.CRITICAL,
                chapter_relevance={},
            ),
            Fact(
                id="chapter-1",
                content="Chapter-specific fact",
                source_documents=["test.pdf"],
                source_type=SourceType.FINANCIALS,
                believability=Believability.NEEDS_VERIFICATION,
                relevance=RelevanceLevel.CHAPTER_SPECIFIC,
                chapter_relevance={},
            ),
            Fact(
                id="context-1",
                content="Additional context fact",
                source_documents=["test.pdf"],
                source_type=SourceType.FINANCIALS,
                believability=Believability.OPINION,
                relevance=RelevanceLevel.ADDITIONAL_CONTEXT,
                chapter_relevance={},
            ),
        ]
        
        repo.insert_facts(facts)
        
        critical_facts = repo.get_facts_by_relevance(RelevanceLevel.CRITICAL)
        assert len(critical_facts) == 1
        assert critical_facts[0].id == "critical-1"
        
        chapter_facts = repo.get_facts_by_relevance("chapter_specific")
        assert len(chapter_facts) == 1
    
    def test_get_facts_by_chapter(self, repo):
        """Test filtering facts by chapter relevance."""
        facts = [
            Fact(
                id="financial-1",
                content="Financial fact",
                source_documents=["test.pdf"],
                source_type=SourceType.FINANCIALS,
                believability=Believability.VERIFIED,
                relevance=RelevanceLevel.CRITICAL,
                chapter_relevance={"Financial": 0.95, "Summary": 0.3},
            ),
            Fact(
                id="market-1",
                content="Market fact",
                source_documents=["test.pdf"],
                source_type=SourceType.MARKET_RESEARCH,
                believability=Believability.NEEDS_VERIFICATION,
                relevance=RelevanceLevel.CHAPTER_SPECIFIC,
                chapter_relevance={"Market": 0.9, "Financial": 0.2},
            ),
        ]
        
        repo.insert_facts(facts)
        
        financial_facts = repo.get_facts_by_chapter("Financial", min_relevance=0.5)
        assert len(financial_facts) == 1
        assert financial_facts[0].id == "financial-1"
        
        market_facts = repo.get_facts_by_chapter("Market", min_relevance=0.5)
        assert len(market_facts) == 1
    
    def test_search_facts(self, repo):
        """Test searching facts by content."""
        facts = [
            Fact(
                id="1",
                content="Revenue grew by 50% year over year.",
                source_documents=["test.pdf"],
                source_type=SourceType.FINANCIALS,
                believability=Believability.VERIFIED,
                relevance=RelevanceLevel.CRITICAL,
                chapter_relevance={},
            ),
            Fact(
                id="2",
                content="The market size is estimated at $5B.",
                source_documents=["test.pdf"],
                source_type=SourceType.MARKET_RESEARCH,
                believability=Believability.NEEDS_VERIFICATION,
                relevance=RelevanceLevel.CHAPTER_SPECIFIC,
                chapter_relevance={},
            ),
        ]
        
        repo.insert_facts(facts)
        
        revenue_facts = repo.search_facts("revenue")
        assert len(revenue_facts) == 1
        assert "Revenue" in revenue_facts[0].content
        
        market_facts = repo.search_facts("market")
        assert len(market_facts) == 1
    
    def test_update_usage(self, repo, sample_fact):
        """Test updating fact usage count."""
        repo.insert_fact(sample_fact)
        
        # Update usage
        result = repo.update_usage(sample_fact.id, chapter="Financial")
        assert result is True
        
        updated = repo.get_fact(sample_fact.id)
        assert updated.usage_count == 1
        assert "Financial" in updated.used_in_chapters
        
        # Update again
        repo.update_usage(sample_fact.id, chapter="Summary")
        updated = repo.get_fact(sample_fact.id)
        assert updated.usage_count == 2
        assert "Summary" in updated.used_in_chapters
    
    def test_delete_fact(self, repo, sample_fact):
        """Test deleting a fact."""
        repo.insert_fact(sample_fact)
        
        # Verify it exists
        assert repo.get_fact(sample_fact.id) is not None
        
        # Delete
        result = repo.delete_fact(sample_fact.id)
        assert result is True
        
        # Verify it's gone
        assert repo.get_fact(sample_fact.id) is None
    
    def test_get_statistics(self, repo):
        """Test getting repository statistics."""
        facts = [
            Fact(
                id="1",
                content="Critical fact",
                source_documents=["doc1.pdf"],
                source_type=SourceType.FINANCIALS,
                believability=Believability.VERIFIED,
                relevance=RelevanceLevel.CRITICAL,
                chapter_relevance={},
            ),
            Fact(
                id="2",
                content="Chapter-specific fact",
                source_documents=["doc2.pdf"],
                source_type=SourceType.MARKET_RESEARCH,
                believability=Believability.NEEDS_VERIFICATION,
                relevance=RelevanceLevel.CHAPTER_SPECIFIC,
                chapter_relevance={},
            ),
        ]
        
        repo.insert_facts(facts)
        
        # Insert documents
        doc1 = Document(
            id="doc1",
            filename="doc1.pdf",
            file_type="pdf",
            source_type=SourceType.FINANCIALS,
            raw_content="Content",
        )
        repo.insert_document(doc1)
        
        stats = repo.get_statistics()
        
        assert stats["total_facts"] == 2
        assert stats["total_documents"] == 1
        assert stats["relevance_breakdown"]["critical"] == 1
        assert stats["relevance_breakdown"]["chapter_specific"] == 1
        assert stats["believability_breakdown"]["verified"] == 1
        assert stats["believability_breakdown"]["needs_verification"] == 1


class TestDocumentRepository:
    """Tests for document storage in repository."""
    
    def test_insert_and_get_document(self, repo):
        """Test inserting and retrieving a document."""
        doc = Document(
            id="doc-001",
            filename="annual_report.pdf",
            file_type="pdf",
            source_type=SourceType.FINANCIALS,
            raw_content="Full document content here...",
            page_count=25,
        )
        
        repo.insert_document(doc)
        
        retrieved = repo.get_document("doc-001")
        
        assert retrieved is not None
        assert retrieved.filename == "annual_report.pdf"
        assert retrieved.page_count == 25
    
    def test_get_document_by_filename(self, repo):
        """Test finding document by filename."""
        doc = Document(
            id="doc-002",
            filename="market_research.pdf",
            file_type="pdf",
            source_type=SourceType.MARKET_RESEARCH,
            raw_content="Research content",
        )
        
        repo.insert_document(doc)
        
        retrieved = repo.get_document_by_filename("market_research.pdf")
        
        assert retrieved is not None
        assert retrieved.id == "doc-002"
    
    def test_get_all_documents(self, repo):
        """Test getting all documents."""
        docs = [
            Document(
                id=f"doc-{i}",
                filename=f"document_{i}.pdf",
                file_type="pdf",
                source_type=SourceType.COMPANY_PRESENTATION,
                raw_content="Content",
            )
            for i in range(3)
        ]
        
        for doc in docs:
            repo.insert_document(doc)
        
        all_docs = repo.get_all_documents()
        assert len(all_docs) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

