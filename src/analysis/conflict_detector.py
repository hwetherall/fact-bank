"""
Fact conflict detection using LLMs.

Identifies contradictions and inconsistencies between facts in the fact bank.
"""

import json
import logging
import asyncio
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.storage.models import Fact, STANDARD_CHAPTERS
from src.utils.llm_client import LLMClient, LLMProvider, LLMResponse
from src.config import Config

logger = logging.getLogger(__name__)


class ConflictSeverity(str, Enum):
    """Severity level of a fact conflict."""
    CRITICAL = "critical"      # Direct numerical contradiction (e.g., different revenue figures)
    HIGH = "high"              # Strong logical contradiction 
    MEDIUM = "medium"          # Partial contradiction or inconsistency
    LOW = "low"                # Minor tension or potential discrepancy


class ConflictType(str, Enum):
    """Type of conflict detected."""
    NUMERICAL = "numerical"              # Conflicting numbers/metrics
    TEMPORAL = "temporal"                # Timeline inconsistencies
    FACTUAL = "factual"                  # Contradicting claims
    DEFINITIONAL = "definitional"        # Different definitions/descriptions
    STRATEGIC = "strategic"              # Conflicting strategic statements


class FactConflict(BaseModel):
    """
    Represents a conflict between two facts.
    """
    fact_a_id: str = Field(description="ID of the first fact")
    fact_b_id: str = Field(description="ID of the second fact")
    fact_a_content: str = Field(description="Content of the first fact")
    fact_b_content: str = Field(description="Content of the second fact")
    conflict_type: ConflictType = Field(description="Type of conflict")
    severity: ConflictSeverity = Field(description="Severity of the conflict")
    explanation: str = Field(description="Detailed explanation of the conflict")
    resolution_suggestion: str = Field(description="Suggested way to resolve the conflict")
    affected_chapters: list[str] = Field(
        default_factory=list,
        description="Chapters where this conflict could cause issues"
    )
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the conflict was detected"
    )
    
    class Config:
        use_enum_values = True


class ConflictAnalysis(BaseModel):
    """Response schema for conflict detection LLM calls."""
    
    has_conflict: bool = Field(
        description="Whether there is a conflict between the facts"
    )
    conflict_type: str | None = Field(
        default=None,
        description="Type of conflict if detected"
    )
    severity: str | None = Field(
        default=None,
        description="Severity of conflict if detected"
    )
    explanation: str = Field(
        description="Explanation of the conflict or why there is no conflict"
    )
    resolution_suggestion: str | None = Field(
        default=None,
        description="How to resolve the conflict if detected"
    )


class BatchConflictResponse(BaseModel):
    """Response schema for batch conflict detection."""
    
    conflicts: list[dict] = Field(
        default_factory=list,
        description="List of detected conflicts with fact indices"
    )


# System prompt for conflict detection
CONFLICT_DETECTION_PROMPT = """You are an expert analyst specializing in detecting contradictions and inconsistencies in investment research data.

Your task is to analyze pairs of facts and determine if they conflict with each other.

## Types of Conflicts to Detect

1. **NUMERICAL**: Conflicting numbers, metrics, or quantitative data
   - Example: "Revenue is $5M" vs "Revenue is $8M"
   - Example: "Growth rate is 15% YoY" vs "Growth rate is 25% YoY"

2. **TEMPORAL**: Timeline or date inconsistencies
   - Example: "Founded in 2018" vs "Founded in 2020"
   - Example: "Product launched Q1 2024" vs "Product launching Q3 2024"

3. **FACTUAL**: Directly contradicting claims or statements
   - Example: "Company is profitable" vs "Company is pre-revenue"
   - Example: "No competitors in market" vs "Competing with 5 major players"

4. **DEFINITIONAL**: Conflicting definitions or descriptions
   - Example: "B2B SaaS platform" vs "B2C marketplace"
   - Example: "Target market is SMBs" vs "Focus on enterprise customers"

5. **STRATEGIC**: Conflicting strategic statements
   - Example: "Focus on organic growth" vs "Planning aggressive M&A strategy"
   - Example: "Premium pricing strategy" vs "Competing on price"

## Severity Levels

- **CRITICAL**: Direct contradiction that cannot be true simultaneously, especially numerical conflicts
- **HIGH**: Strong contradiction that significantly impacts analysis
- **MEDIUM**: Partial contradiction or notable inconsistency
- **LOW**: Minor tension or potential discrepancy that might have explanation

## Guidelines

- Consider context and timing: facts from different time periods might both be correct
- Look for implicit contradictions, not just explicit ones
- Be precise about what specifically conflicts
- Provide actionable resolution suggestions
- Only flag as conflict if there's a genuine inconsistency"""


class ConflictDetector:
    """
    Detects conflicts and contradictions between facts.
    
    Usage:
        detector = ConflictDetector()
        conflicts = await detector.detect_conflicts(facts)
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        provider: LLMProvider | str | None = None,
    ):
        """
        Initialize the conflict detector.
        
        Args:
            llm_client: Optional pre-configured LLM client
            model: Model to use for analysis (defaults to DEFAULT_EXTRACTION_MODEL)
            provider: LLM provider if creating new client
        """
        self.llm_client = llm_client
        self.model = model or Config.DEFAULT_EXTRACTION_MODEL
        if provider is None:
            provider = LLMProvider(Config.DEFAULT_LLM_PROVIDER)
        elif isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        self.provider = provider
        self._owns_client = False
    
    async def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self.llm_client is None:
            self.llm_client = LLMClient(provider=self.provider)
            self._owns_client = True
        return self.llm_client
    
    async def detect_conflicts(
        self,
        facts: list[Fact],
        relevance_threshold: float = 0.5,
        max_comparisons: int = 500,
    ) -> list[FactConflict]:
        """
        Find facts that might contradict each other.
        
        Groups facts by chapter and checks for contradictions within each chapter.
        
        Args:
            facts: List of facts to analyze
            relevance_threshold: Minimum chapter relevance to consider (0.0-1.0)
            max_comparisons: Maximum number of fact pairs to compare (for cost control)
            
        Returns:
            List of detected conflicts
        """
        if len(facts) < 2:
            return []
        
        all_conflicts: list[FactConflict] = []
        comparisons_made = 0
        
        # Group facts by chapter
        for chapter in STANDARD_CHAPTERS:
            chapter_facts = [
                f for f in facts 
                if f.chapter_relevance.get(chapter, 0) > relevance_threshold
            ]
            
            if len(chapter_facts) < 2:
                continue
            
            logger.info(f"Checking {len(chapter_facts)} facts in '{chapter}' for conflicts")
            
            # Check for conflicts within this chapter
            conflicts = await self._check_chapter_conflicts(
                chapter_facts, 
                chapter,
                max_comparisons - comparisons_made,
            )
            
            all_conflicts.extend(conflicts)
            comparisons_made += len(chapter_facts) * (len(chapter_facts) - 1) // 2
            
            if comparisons_made >= max_comparisons:
                logger.warning(f"Reached max comparisons limit ({max_comparisons})")
                break
        
        # Deduplicate conflicts (same pair might appear in multiple chapters)
        unique_conflicts = self._deduplicate_conflicts(all_conflicts)
        
        logger.info(f"Detected {len(unique_conflicts)} unique conflicts")
        return unique_conflicts
    
    async def _check_chapter_conflicts(
        self,
        facts: list[Fact],
        chapter: str,
        remaining_budget: int,
    ) -> list[FactConflict]:
        """Check for conflicts among facts in a single chapter."""
        if len(facts) < 2 or remaining_budget <= 0:
            return []
        
        # For efficiency, use batch checking for larger sets
        if len(facts) > 10:
            return await self._batch_check_conflicts(facts, chapter, remaining_budget)
        
        # For smaller sets, do pairwise comparison
        return await self._pairwise_check_conflicts(facts, chapter, remaining_budget)
    
    async def _pairwise_check_conflicts(
        self,
        facts: list[Fact],
        chapter: str,
        remaining_budget: int,
    ) -> list[FactConflict]:
        """Check each pair of facts for conflicts."""
        conflicts: list[FactConflict] = []
        client = await self._get_client()
        comparisons = 0
        
        for i, fact_a in enumerate(facts):
            for fact_b in facts[i + 1:]:
                if comparisons >= remaining_budget:
                    break
                
                conflict = await self._check_pair(client, fact_a, fact_b, chapter)
                if conflict:
                    conflicts.append(conflict)
                
                comparisons += 1
            
            if comparisons >= remaining_budget:
                break
        
        return conflicts
    
    async def _check_pair(
        self,
        client: LLMClient,
        fact_a: Fact,
        fact_b: Fact,
        chapter: str,
    ) -> FactConflict | None:
        """Check if two facts conflict."""
        prompt = f"""Analyze these two facts from an investment research context and determine if they conflict:

**Fact A:**
{fact_a.content}
(Source: {', '.join(fact_a.source_documents)}, Confidence: {fact_a.confidence:.0%})

**Fact B:**
{fact_b.content}
(Source: {', '.join(fact_b.source_documents)}, Confidence: {fact_b.confidence:.0%})

**Context:** These facts are both relevant to the "{chapter}" section of an investment memo.

Respond with a JSON object:
{{
    "has_conflict": true/false,
    "conflict_type": "numerical" | "temporal" | "factual" | "definitional" | "strategic" | null,
    "severity": "critical" | "high" | "medium" | "low" | null,
    "explanation": "Detailed explanation of the conflict or why there is no conflict",
    "resolution_suggestion": "How to resolve if conflict exists, or null"
}}"""

        messages = [
            {"role": "system", "content": CONFLICT_DETECTION_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = await client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.2,  # Low temperature for consistent analysis
                response_format={"type": "json_object"},
            )
            
            analysis = self._parse_conflict_response(response)
            
            if analysis.has_conflict:
                # Determine affected chapters
                affected = self._get_affected_chapters(fact_a, fact_b)
                
                return FactConflict(
                    fact_a_id=fact_a.id,
                    fact_b_id=fact_b.id,
                    fact_a_content=fact_a.content,
                    fact_b_content=fact_b.content,
                    conflict_type=ConflictType(analysis.conflict_type or "factual"),
                    severity=ConflictSeverity(analysis.severity or "medium"),
                    explanation=analysis.explanation,
                    resolution_suggestion=analysis.resolution_suggestion or "Review source documents to verify accuracy",
                    affected_chapters=affected,
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check conflict between facts: {e}")
            return None
    
    async def _batch_check_conflicts(
        self,
        facts: list[Fact],
        chapter: str,
        remaining_budget: int,
    ) -> list[FactConflict]:
        """
        Batch check for conflicts - more efficient for larger fact sets.
        
        Groups facts and asks LLM to identify conflicts in batches.
        """
        conflicts: list[FactConflict] = []
        client = await self._get_client()
        
        # Create fact summaries for the prompt
        fact_list = "\n".join([
            f"[{i}] {f.content} (Source: {', '.join(f.source_documents[:2])})"
            for i, f in enumerate(facts)
        ])
        
        prompt = f"""Analyze these facts from investment research and identify any conflicts or contradictions between them:

**Facts (indexed):**
{fact_list}

**Context:** These facts are all relevant to the "{chapter}" section of an investment memo.

Find ALL pairs of facts that conflict with each other. A conflict is when two facts cannot both be true, or when they present contradictory information.

Respond with a JSON object:
{{
    "conflicts": [
        {{
            "fact_index_a": 0,
            "fact_index_b": 3,
            "conflict_type": "numerical" | "temporal" | "factual" | "definitional" | "strategic",
            "severity": "critical" | "high" | "medium" | "low",
            "explanation": "Why these facts conflict",
            "resolution_suggestion": "How to resolve"
        }}
    ]
}}

Return an empty conflicts array if no conflicts are found."""

        messages = [
            {"role": "system", "content": CONFLICT_DETECTION_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = await client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            
            data = json.loads(response.content)
            batch_response = BatchConflictResponse(**data)
            
            for conflict_data in batch_response.conflicts:
                idx_a = conflict_data.get("fact_index_a", 0)
                idx_b = conflict_data.get("fact_index_b", 1)
                
                if idx_a >= len(facts) or idx_b >= len(facts):
                    continue
                
                fact_a = facts[idx_a]
                fact_b = facts[idx_b]
                affected = self._get_affected_chapters(fact_a, fact_b)
                
                conflicts.append(FactConflict(
                    fact_a_id=fact_a.id,
                    fact_b_id=fact_b.id,
                    fact_a_content=fact_a.content,
                    fact_b_content=fact_b.content,
                    conflict_type=ConflictType(conflict_data.get("conflict_type", "factual")),
                    severity=ConflictSeverity(conflict_data.get("severity", "medium")),
                    explanation=conflict_data.get("explanation", "Conflict detected"),
                    resolution_suggestion=conflict_data.get("resolution_suggestion", "Review source documents"),
                    affected_chapters=affected,
                ))
                
        except Exception as e:
            logger.error(f"Failed to batch check conflicts: {e}")
            # Fall back to pairwise if batch fails
            return await self._pairwise_check_conflicts(facts, chapter, remaining_budget)
        
        return conflicts
    
    def _parse_conflict_response(self, response: LLMResponse) -> ConflictAnalysis:
        """Parse the LLM response for conflict analysis."""
        try:
            data = json.loads(response.content)
            return ConflictAnalysis(**data)
        except Exception as e:
            logger.error(f"Failed to parse conflict response: {e}")
            return ConflictAnalysis(
                has_conflict=False,
                explanation="Failed to analyze due to parsing error"
            )
    
    def _get_affected_chapters(self, fact_a: Fact, fact_b: Fact) -> list[str]:
        """Get chapters where both facts have significant relevance."""
        affected = []
        for chapter in STANDARD_CHAPTERS:
            relevance_a = fact_a.chapter_relevance.get(chapter, 0)
            relevance_b = fact_b.chapter_relevance.get(chapter, 0)
            if relevance_a > 0.3 and relevance_b > 0.3:
                affected.append(chapter)
        return affected
    
    def _deduplicate_conflicts(
        self, 
        conflicts: list[FactConflict]
    ) -> list[FactConflict]:
        """Remove duplicate conflicts (same fact pair found in multiple chapters)."""
        seen_pairs: set[tuple[str, str]] = set()
        unique: list[FactConflict] = []
        
        for conflict in conflicts:
            # Normalize pair order
            pair = tuple(sorted([conflict.fact_a_id, conflict.fact_b_id]))
            
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique.append(conflict)
            else:
                # Merge affected chapters into existing conflict
                for existing in unique:
                    existing_pair = tuple(sorted([existing.fact_a_id, existing.fact_b_id]))
                    if existing_pair == pair:
                        for chapter in conflict.affected_chapters:
                            if chapter not in existing.affected_chapters:
                                existing.affected_chapters.append(chapter)
                        break
        
        return unique
    
    async def check_specific_facts(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> FactConflict | None:
        """
        Check if two specific facts conflict.
        
        Useful for checking new facts against existing ones.
        
        Args:
            fact_a: First fact
            fact_b: Second fact
            
        Returns:
            FactConflict if conflict detected, None otherwise
        """
        client = await self._get_client()
        
        # Determine the most relevant chapter for context
        shared_chapters = self._get_affected_chapters(fact_a, fact_b)
        chapter = shared_chapters[0] if shared_chapters else "General"
        
        return await self._check_pair(client, fact_a, fact_b, chapter)
    
    async def find_conflicts_for_fact(
        self,
        new_fact: Fact,
        existing_facts: list[Fact],
        max_checks: int = 50,
    ) -> list[FactConflict]:
        """
        Find conflicts between a new fact and existing facts.
        
        Useful when adding a new fact to check for contradictions.
        
        Args:
            new_fact: The new fact to check
            existing_facts: List of existing facts to check against
            max_checks: Maximum number of comparisons
            
        Returns:
            List of detected conflicts
        """
        conflicts: list[FactConflict] = []
        client = await self._get_client()
        
        # Prioritize facts with similar chapter relevance
        relevant_facts = sorted(
            existing_facts,
            key=lambda f: self._chapter_overlap_score(new_fact, f),
            reverse=True
        )[:max_checks]
        
        for existing in relevant_facts:
            # Skip if there's no chapter overlap
            if self._chapter_overlap_score(new_fact, existing) < 0.1:
                continue
            
            chapter = self._get_primary_shared_chapter(new_fact, existing)
            conflict = await self._check_pair(client, new_fact, existing, chapter)
            
            if conflict:
                conflicts.append(conflict)
        
        return conflicts
    
    def _chapter_overlap_score(self, fact_a: Fact, fact_b: Fact) -> float:
        """Calculate how much chapter relevance two facts share."""
        total_overlap = 0.0
        for chapter in STANDARD_CHAPTERS:
            rel_a = fact_a.chapter_relevance.get(chapter, 0)
            rel_b = fact_b.chapter_relevance.get(chapter, 0)
            total_overlap += min(rel_a, rel_b)
        return total_overlap
    
    def _get_primary_shared_chapter(self, fact_a: Fact, fact_b: Fact) -> str:
        """Get the chapter where both facts have highest combined relevance."""
        best_chapter = STANDARD_CHAPTERS[0]
        best_score = 0.0
        
        for chapter in STANDARD_CHAPTERS:
            rel_a = fact_a.chapter_relevance.get(chapter, 0)
            rel_b = fact_b.chapter_relevance.get(chapter, 0)
            combined = rel_a + rel_b
            if combined > best_score:
                best_score = combined
                best_chapter = chapter
        
        return best_chapter
    
    async def close(self):
        """Close the LLM client if we own it."""
        if self._owns_client and self.llm_client:
            await self.llm_client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Synchronous wrapper
class SyncConflictDetector:
    """Synchronous wrapper for ConflictDetector."""
    
    def __init__(self, **kwargs):
        self._async_detector = ConflictDetector(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None
    
    def _get_loop(self):
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def detect_conflicts(
        self,
        facts: list[Fact],
        relevance_threshold: float = 0.5,
        max_comparisons: int = 500,
    ) -> list[FactConflict]:
        """Synchronous conflict detection."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_detector.detect_conflicts(
                facts, relevance_threshold, max_comparisons
            )
        )
    
    def check_specific_facts(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> FactConflict | None:
        """Synchronous check for two specific facts."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_detector.check_specific_facts(fact_a, fact_b)
        )
    
    def find_conflicts_for_fact(
        self,
        new_fact: Fact,
        existing_facts: list[Fact],
        max_checks: int = 50,
    ) -> list[FactConflict]:
        """Synchronous conflict search for a new fact."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_detector.find_conflicts_for_fact(
                new_fact, existing_facts, max_checks
            )
        )
    
    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._async_detector.close())
            self._loop.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

