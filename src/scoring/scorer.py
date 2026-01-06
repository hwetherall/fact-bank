"""
Scoring logic for facts.

Handles confidence, importance, and chapter relevance scoring.
"""

import logging
from typing import TYPE_CHECKING

from src.storage.models import (
    Fact,
    SourceType,
    RelevanceLevel,
    Believability,
    STANDARD_CHAPTERS,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FactScorer:
    """
    Scores and validates facts.
    
    Provides utilities for:
    - Believability assessment
    - Relevance validation and normalization
    - Chapter relevance scoring
    """
    
    def __init__(self):
        """Initialize the scorer."""
        self.standard_chapters = STANDARD_CHAPTERS.copy()
    
    def upgrade_believability(
        self,
        fact: Fact,
        new_believability: Believability,
    ) -> Fact:
        """
        Upgrade a fact's believability level (e.g., after verification).
        
        Args:
            fact: The fact to update
            new_believability: The new believability level
            
        Returns:
            Updated fact with new believability
        """
        fact.believability = new_believability
        return fact
    
    def normalize_chapter_relevance(self, fact: Fact) -> Fact:
        """
        Ensure all standard chapters have relevance scores.
        
        Args:
            fact: The fact to normalize
            
        Returns:
            Updated fact with all chapters
        """
        normalized = {ch: 0.0 for ch in self.standard_chapters}
        normalized.update(fact.chapter_relevance)
        
        # Clamp all values to valid range
        for chapter in normalized:
            normalized[chapter] = max(0.0, min(1.0, normalized[chapter]))
        
        fact.chapter_relevance = normalized
        return fact
    
    def validate_fact(self, fact: Fact) -> list[str]:
        """
        Validate a fact and return any issues.
        
        Args:
            fact: The fact to validate
            
        Returns:
            List of validation issue strings (empty if valid)
        """
        issues = []
        
        # Check content
        if not fact.content or len(fact.content.strip()) < 5:
            issues.append("Fact content is too short or empty")
        
        if len(fact.content) > 1000:
            issues.append("Fact content is unusually long (>1000 chars)")
        
        # Check believability
        if fact.believability not in [e.value for e in Believability]:
            issues.append(f"Invalid believability level: {fact.believability}")
        
        # Check relevance
        if fact.relevance not in [e.value for e in RelevanceLevel]:
            issues.append(f"Invalid relevance level: {fact.relevance}")
        
        # Check chapter relevance
        for chapter, score in fact.chapter_relevance.items():
            if score < 0.0 or score > 1.0:
                issues.append(f"Chapter {chapter} relevance {score} is out of range")
        
        return issues
    
    def score_fact_quality(self, fact: Fact) -> float:
        """
        Calculate an overall quality score for a fact.
        
        Considers:
        - Content specificity (length, numbers, names)
        - Believability level
        - Relevance level
        - Chapter relevance spread
        
        Args:
            fact: The fact to score
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        scores = []
        
        # Content specificity (presence of numbers, specific terms)
        content = fact.content
        specificity = 0.5
        
        # Bonus for numbers
        if any(char.isdigit() for char in content):
            specificity += 0.2
        
        # Bonus for specific length (not too short, not too long)
        if 20 < len(content) < 200:
            specificity += 0.2
        
        # Bonus for currency/percentage symbols
        if any(sym in content for sym in ["$", "€", "£", "%"]):
            specificity += 0.1
        
        scores.append(min(1.0, specificity))
        
        # Believability contributes to quality
        believability_scores = {
            Believability.VERIFIED.value: 1.0,
            Believability.NEEDS_VERIFICATION.value: 0.7,
            Believability.OPINION.value: 0.3,
        }
        scores.append(believability_scores.get(fact.believability, 0.5))
        
        # Relevance rating
        relevance_scores = {
            RelevanceLevel.CRITICAL.value: 1.0,
            RelevanceLevel.CHAPTER_SPECIFIC.value: 0.8,
            RelevanceLevel.ADDITIONAL_CONTEXT.value: 0.5,
            RelevanceLevel.NOISE.value: 0.1,
        }
        scores.append(relevance_scores.get(fact.relevance, 0.5))
        
        # Chapter relevance (prefer focused facts over generic ones)
        relevance_values = list(fact.chapter_relevance.values())
        if relevance_values:
            max_relevance = max(relevance_values)
            avg_relevance = sum(relevance_values) / len(relevance_values)
            # High max with low average = focused (good)
            focus_score = max_relevance * (1 - avg_relevance * 0.5)
            scores.append(focus_score)
        
        return round(sum(scores) / len(scores), 3)


def get_facts_by_quality(
    facts: list[Fact],
    min_quality: float = 0.0,
) -> list[Fact]:
    """
    Filter and sort facts by quality score.
    
    Args:
        facts: List of facts to filter
        min_quality: Minimum quality threshold
        
    Returns:
        Filtered list sorted by quality (descending)
    """
    scorer = FactScorer()
    
    scored = [
        (fact, scorer.score_fact_quality(fact))
        for fact in facts
    ]
    
    filtered = [
        (fact, score) for fact, score in scored
        if score >= min_quality
    ]
    
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    return [fact for fact, _ in filtered]


def categorize_by_relevance(
    facts: list[Fact],
) -> dict[str, list[Fact]]:
    """
    Categorize facts by relevance level.
    
    Args:
        facts: List of facts to categorize
        
    Returns:
        Dictionary mapping relevance level to facts
    """
    result = {
        RelevanceLevel.CRITICAL.value: [],
        RelevanceLevel.CHAPTER_SPECIFIC.value: [],
        RelevanceLevel.ADDITIONAL_CONTEXT.value: [],
        RelevanceLevel.NOISE.value: [],
    }
    
    for fact in facts:
        relevance = fact.relevance
        if relevance in result:
            result[relevance].append(fact)
        else:
            result[RelevanceLevel.ADDITIONAL_CONTEXT.value].append(fact)
    
    return result


def categorize_by_believability(
    facts: list[Fact],
) -> dict[str, list[Fact]]:
    """
    Categorize facts by believability level.
    
    Args:
        facts: List of facts to categorize
        
    Returns:
        Dictionary mapping believability level to facts
    """
    result = {
        Believability.VERIFIED.value: [],
        Believability.NEEDS_VERIFICATION.value: [],
        Believability.OPINION.value: [],
    }
    
    for fact in facts:
        believability = fact.believability
        if believability in result:
            result[believability].append(fact)
        else:
            result[Believability.NEEDS_VERIFICATION.value].append(fact)
    
    return result


def filter_noise(facts: list[Fact], include_noise: bool = False) -> list[Fact]:
    """
    Filter out noise facts unless explicitly requested.
    
    Args:
        facts: List of facts to filter
        include_noise: If True, include noise facts
        
    Returns:
        Filtered list of facts
    """
    if include_noise:
        return facts
    return [f for f in facts if f.relevance != RelevanceLevel.NOISE.value]


def get_chapter_summary(facts: list[Fact]) -> dict[str, dict]:
    """
    Get summary statistics for each chapter.
    
    Args:
        facts: List of facts to analyze
        
    Returns:
        Dictionary with chapter statistics
    """
    chapter_stats = {}
    
    for chapter in STANDARD_CHAPTERS:
        relevant_facts = [
            f for f in facts
            if f.chapter_relevance.get(chapter, 0.0) >= 0.5
        ]
        
        if relevant_facts:
            # Count by believability
            verified_count = sum(1 for f in relevant_facts if f.believability == Believability.VERIFIED.value)
            needs_verification_count = sum(1 for f in relevant_facts if f.believability == Believability.NEEDS_VERIFICATION.value)
            opinion_count = sum(1 for f in relevant_facts if f.believability == Believability.OPINION.value)
            
            # Count by relevance
            relevance_counts = {
                "critical": sum(1 for f in relevant_facts if f.relevance == RelevanceLevel.CRITICAL.value),
                "chapter_specific": sum(1 for f in relevant_facts if f.relevance == RelevanceLevel.CHAPTER_SPECIFIC.value),
                "additional_context": sum(1 for f in relevant_facts if f.relevance == RelevanceLevel.ADDITIONAL_CONTEXT.value),
                "noise": sum(1 for f in relevant_facts if f.relevance == RelevanceLevel.NOISE.value),
            }
            
            # Legacy importance mapping for backwards compatibility
            importance_counts = {
                "high": relevance_counts["critical"],
                "medium": relevance_counts["chapter_specific"],
                "low": relevance_counts["additional_context"] + relevance_counts["noise"],
            }
            
            # Calculate average "confidence" using legacy mapping
            avg_confidence = sum(f.confidence for f in relevant_facts) / len(relevant_facts)
        else:
            verified_count = 0
            needs_verification_count = 0
            opinion_count = 0
            relevance_counts = {"critical": 0, "chapter_specific": 0, "additional_context": 0, "noise": 0}
            importance_counts = {"high": 0, "medium": 0, "low": 0}
            avg_confidence = 0.0
        
        chapter_stats[chapter] = {
            "fact_count": len(relevant_facts),
            "avg_confidence": round(avg_confidence, 3),
            "importance_breakdown": importance_counts,  # Legacy
            "relevance_breakdown": relevance_counts,
            "believability_breakdown": {
                "verified": verified_count,
                "needs_verification": needs_verification_count,
                "opinion": opinion_count,
            },
        }
    
    return chapter_stats

