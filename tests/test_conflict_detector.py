"""
Tests for fact conflict detection.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.storage.models import Fact, SourceType, RelevanceLevel, Believability, STANDARD_CHAPTERS
from src.analysis.conflict_detector import (
    ConflictDetector,
    SyncConflictDetector,
    FactConflict,
    ConflictSeverity,
    ConflictType,
    ConflictAnalysis,
)


def create_test_fact(
    id: str,
    content: str,
    chapter_relevance: dict[str, float] | None = None,
    source_documents: list[str] | None = None,
) -> Fact:
    """Create a test fact with default values."""
    default_relevance = {ch: 0.0 for ch in STANDARD_CHAPTERS}
    if chapter_relevance:
        default_relevance.update(chapter_relevance)
    
    return Fact(
        id=id,
        content=content,
        source_documents=source_documents or ["test_doc.pdf"],
        source_type=SourceType.COMPANY_PRESENTATION,
        believability=Believability.NEEDS_VERIFICATION,
        relevance=RelevanceLevel.CHAPTER_SPECIFIC,
        chapter_relevance=default_relevance,
        extraction_timestamp=datetime.utcnow(),
        usage_count=0,
        used_in_chapters=[],
        embedding=None,
    )


class TestConflictDetector:
    """Tests for ConflictDetector class."""
    
    def test_get_affected_chapters(self):
        """Test that affected chapters are correctly identified."""
        detector = ConflictDetector.__new__(ConflictDetector)
        
        fact_a = create_test_fact(
            "fact-a",
            "Revenue is $5M",
            {"Finance & Operations": 0.9, "Unit Economics": 0.7, "Market Research": 0.2}
        )
        fact_b = create_test_fact(
            "fact-b",
            "Revenue is $8M",
            {"Finance & Operations": 0.85, "Unit Economics": 0.5, "Revenue Model": 0.6}
        )
        
        affected = detector._get_affected_chapters(fact_a, fact_b)
        
        # Both have >0.3 relevance in these chapters
        assert "Finance & Operations" in affected
        assert "Unit Economics" in affected
        assert "Market Research" not in affected  # Only fact_a has >0.3
    
    def test_chapter_overlap_score(self):
        """Test chapter overlap calculation."""
        detector = ConflictDetector.__new__(ConflictDetector)
        
        fact_a = create_test_fact(
            "fact-a",
            "Test fact A",
            {"Finance & Operations": 0.9, "Unit Economics": 0.7}
        )
        fact_b = create_test_fact(
            "fact-b",
            "Test fact B",
            {"Finance & Operations": 0.5, "Unit Economics": 0.3}
        )
        
        score = detector._chapter_overlap_score(fact_a, fact_b)
        
        # min(0.9, 0.5) + min(0.7, 0.3) = 0.5 + 0.3 = 0.8
        assert score == pytest.approx(0.8, rel=0.01)
    
    def test_deduplicate_conflicts(self):
        """Test conflict deduplication."""
        detector = ConflictDetector.__new__(ConflictDetector)
        
        conflict1 = FactConflict(
            fact_a_id="fact-a",
            fact_b_id="fact-b",
            fact_a_content="Content A",
            fact_b_content="Content B",
            conflict_type=ConflictType.NUMERICAL,
            severity=ConflictSeverity.HIGH,
            explanation="Numbers differ",
            resolution_suggestion="Verify with source",
            affected_chapters=["Finance & Operations"],
        )
        
        # Same pair, different order
        conflict2 = FactConflict(
            fact_a_id="fact-b",
            fact_b_id="fact-a",
            fact_a_content="Content B",
            fact_b_content="Content A",
            conflict_type=ConflictType.NUMERICAL,
            severity=ConflictSeverity.HIGH,
            explanation="Numbers differ",
            resolution_suggestion="Verify with source",
            affected_chapters=["Unit Economics"],
        )
        
        unique = detector._deduplicate_conflicts([conflict1, conflict2])
        
        assert len(unique) == 1
        # Affected chapters should be merged
        assert "Finance & Operations" in unique[0].affected_chapters
        assert "Unit Economics" in unique[0].affected_chapters
    
    def test_get_primary_shared_chapter(self):
        """Test finding the primary shared chapter."""
        detector = ConflictDetector.__new__(ConflictDetector)
        
        fact_a = create_test_fact(
            "fact-a",
            "Test fact A",
            {"Finance & Operations": 0.9, "Market Research": 0.5}
        )
        fact_b = create_test_fact(
            "fact-b",
            "Test fact B",
            {"Finance & Operations": 0.8, "Market Research": 0.6}
        )
        
        chapter = detector._get_primary_shared_chapter(fact_a, fact_b)
        
        # Finance & Operations: 0.9 + 0.8 = 1.7
        # Market Research: 0.5 + 0.6 = 1.1
        assert chapter == "Finance & Operations"


class TestConflictModels:
    """Tests for conflict-related models."""
    
    def test_fact_conflict_creation(self):
        """Test FactConflict model creation."""
        conflict = FactConflict(
            fact_a_id="id-a",
            fact_b_id="id-b",
            fact_a_content="Revenue is $5M ARR",
            fact_b_content="Revenue is $8M ARR",
            conflict_type=ConflictType.NUMERICAL,
            severity=ConflictSeverity.CRITICAL,
            explanation="Different revenue figures reported",
            resolution_suggestion="Verify with latest financial statements",
            affected_chapters=["Finance & Operations", "Unit Economics"],
        )
        
        assert conflict.fact_a_id == "id-a"
        assert conflict.conflict_type == "numerical"
        assert conflict.severity == "critical"
        assert len(conflict.affected_chapters) == 2
    
    def test_conflict_analysis_parsing(self):
        """Test ConflictAnalysis model parsing."""
        data = {
            "has_conflict": True,
            "conflict_type": "numerical",
            "severity": "high",
            "explanation": "Revenue figures are inconsistent",
            "resolution_suggestion": "Check source documents"
        }
        
        analysis = ConflictAnalysis(**data)
        
        assert analysis.has_conflict is True
        assert analysis.conflict_type == "numerical"
        assert analysis.severity == "high"
    
    def test_conflict_analysis_no_conflict(self):
        """Test ConflictAnalysis when no conflict exists."""
        data = {
            "has_conflict": False,
            "explanation": "Facts are complementary, not contradictory"
        }
        
        analysis = ConflictAnalysis(**data)
        
        assert analysis.has_conflict is False
        assert analysis.conflict_type is None
        assert analysis.severity is None


class TestConflictTypeEnum:
    """Tests for ConflictType enum."""
    
    def test_conflict_types(self):
        """Test all conflict types are defined."""
        assert ConflictType.NUMERICAL.value == "numerical"
        assert ConflictType.TEMPORAL.value == "temporal"
        assert ConflictType.FACTUAL.value == "factual"
        assert ConflictType.DEFINITIONAL.value == "definitional"
        assert ConflictType.STRATEGIC.value == "strategic"


class TestConflictSeverityEnum:
    """Tests for ConflictSeverity enum."""
    
    def test_severity_levels(self):
        """Test all severity levels are defined."""
        assert ConflictSeverity.CRITICAL.value == "critical"
        assert ConflictSeverity.HIGH.value == "high"
        assert ConflictSeverity.MEDIUM.value == "medium"
        assert ConflictSeverity.LOW.value == "low"


@pytest.mark.asyncio
class TestConflictDetectorAsync:
    """Async tests for ConflictDetector."""
    
    async def test_detect_conflicts_empty_list(self):
        """Test with empty fact list."""
        detector = ConflictDetector()
        
        conflicts = await detector.detect_conflicts([])
        
        assert conflicts == []
        await detector.close()
    
    async def test_detect_conflicts_single_fact(self):
        """Test with single fact."""
        detector = ConflictDetector()
        fact = create_test_fact("single", "Only one fact")
        
        conflicts = await detector.detect_conflicts([fact])
        
        assert conflicts == []
        await detector.close()
    
    async def test_check_pair_mock(self):
        """Test pair checking with mocked LLM response."""
        detector = ConflictDetector()
        
        # Mock the LLM client
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "has_conflict": True,
            "conflict_type": "numerical",
            "severity": "critical",
            "explanation": "Revenue figures differ significantly",
            "resolution_suggestion": "Verify with Q3 financials"
        })
        
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = mock_response
        detector.llm_client = mock_client
        detector._owns_client = False
        
        fact_a = create_test_fact(
            "fact-a",
            "Annual revenue reached $5M in 2023",
            {"Finance & Operations": 0.9}
        )
        fact_b = create_test_fact(
            "fact-b",
            "2023 revenue was $8.5M",
            {"Finance & Operations": 0.85}
        )
        
        conflict = await detector._check_pair(mock_client, fact_a, fact_b, "Finance & Operations")
        
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.NUMERICAL
        assert conflict.severity == ConflictSeverity.CRITICAL
        assert "Revenue figures" in conflict.explanation

