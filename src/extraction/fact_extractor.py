"""
Fact extraction using LLMs.

Extracts atomic facts from documents using carefully crafted prompts.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from src.storage.models import (
    Fact, 
    SourceType, 
    RelevanceLevel,
    Believability,
    ExtractedFactRaw,
    FactExtractionResponse,
    BASE_BELIEVABILITY_SCORES,
    STANDARD_CHAPTERS,
    CHAPTER_DESCRIPTIONS,
)
from src.utils.llm_client import LLMClient, LLMProvider, LLMResponse
from src.config import Config, get_config

if TYPE_CHECKING:
    from src.extraction.normalizer import ExtractedDocument

logger = logging.getLogger(__name__)


# System prompt for fact extraction
FACT_EXTRACTION_SYSTEM_PROMPT = """You are an expert fact extractor for investment analysis. Your task is to extract every distinct, atomic fact from the provided document that could be relevant to evaluating a business opportunity.

## Guidelines

### Atomicity
- Each fact must be a single, verifiable claim
- Do NOT combine multiple facts into one statement
- Break compound statements into separate facts

### Precision
- Preserve exact numbers, dates, percentages, and currency amounts
- Include specific names of people, companies, and products
- Maintain units of measurement (e.g., "$4.2M ARR", "15% YoY growth")
- Use the original terminology from the document

### Neutrality
- Extract facts as stated, without editorializing or interpreting
- Do not add qualifiers or judgments
- Avoid summarizing or paraphrasing that loses specificity

### Completeness
- Extract ALL investment-relevant facts, including:
  - Financial metrics (revenue, margins, growth rates, valuations)
  - Market data (size, growth, trends, segments)
  - Company information (team, products, customers, partnerships)
  - Competitive landscape (competitors, market position, moats)
  - Risks and challenges
  - Technical capabilities and IP
  - Legal and regulatory matters
  - Operational details

### Chapter Relevance
For each fact, assess its relevance (0.0 to 1.0) to these investment memo chapters:
- Opportunity Validation: Customer need, demand evidence, market timing, problem-solution fit
- Product & Technology: What's being built, technical feasibility, IP, product roadmap
- Market Research: Market size (TAM/SAM/SOM), structure, dynamics, trends, segments
- Competitive Analysis: Competitive landscape, positioning, differentiation, moats
- Revenue Model: How money is made, pricing strategy, unit economics, monetization
- Go-to-Market: Customer acquisition, sales strategy, channels, partnerships
- Unit Economics: CAC, LTV, margins, payback period, key financial metrics
- Finance & Operations: Financial projections, assumptions, capital needs, operations
- Talent & Team: Team capabilities, leadership, hiring plans, culture, advisors
- Legal & IP: Corporate structure, compliance, intellectual property, contracts

### Relevance Level
Categorize each fact's relevance to the overall investment memo:

- CRITICAL: Top-tier facts relevant to most or all chapters. These are foundational facts that 
  the entire memo builds upon. Should be no more than 10% of extracted facts.
  Examples: Core product description, total revenue, primary market, key competitive advantage.

- CHAPTER_SPECIFIC: Very important facts, but primarily relevant to specific chapters.
  Examples: Burn rate (Finance), CAC metrics (Unit Economics), team credentials (Talent & Team).

- ADDITIONAL_CONTEXT: Interesting contextual facts. The memo could be written without them,
  but they add depth and supporting detail.
  Examples: Office locations, minor product features, historical company milestones.

- NOISE: Low-value or irrelevant facts that would overload the analysis. These provide
  minimal investment insight or are redundant.
  Examples: Marketing superlatives, vague claims, duplicate information, formatting artifacts.

### Believability
Assess how much we should believe each fact without external verification:

- VERIFIED: The fact should be believed as stated. No secondary verification needed.
  Examples: Team member names/roles from pitch deck, company founding date, product names,
  specific metrics from audited financials.

- NEEDS_VERIFICATION: Claims that could be exaggerated or require validation from other sources.
  Examples: TAM claims ("$1B+ market"), growth projections, competitive positioning claims,
  customer satisfaction rates, market share estimates.

- OPINION: Subjective statements that cannot be objectively validated. These are marketing
  language or value judgments rather than facts.
  Examples: "World-class team", "Revolutionary technology", "Industry-leading solution",
  "Legendary founders", "Best-in-class product".

### Recency
Classify whether the information appears "current", "dated", or if recency is "unknown".

## Output Format

Return a JSON object with this exact structure:
{
  "facts": [
    {
      "content": "The exact fact as a single sentence",
      "relevance": "critical" | "chapter_specific" | "additional_context" | "noise",
      "believability": "verified" | "needs_verification" | "opinion",
      "chapter_relevance": {
        "Opportunity Validation": 0.0-1.0,
        "Product & Technology": 0.0-1.0,
        "Market Research": 0.0-1.0,
        "Competitive Analysis": 0.0-1.0,
        "Revenue Model": 0.0-1.0,
        "Go-to-Market": 0.0-1.0,
        "Unit Economics": 0.0-1.0,
        "Finance & Operations": 0.0-1.0,
        "Talent & Team": 0.0-1.0,
        "Legal & IP": 0.0-1.0
      },
      "recency_indicator": "current" | "dated" | "unknown"
    }
  ]
}

## Extraction Strategy
- Extract EVERY distinct fact, even if they seem minor
- Financial documents should yield 20-50+ facts covering all data points
- Each number, percentage, date, or metric should be its own fact
- Break down complex statements into multiple atomic facts
- Don't summarize - extract the raw data points
- Be conservative with CRITICAL - reserve it for truly foundational facts
- Be honest about believability - most company claims need verification

Extract comprehensively - it's better to extract more facts than to miss important information."""


class FactExtractor:
    """
    Extracts facts from documents using LLMs.
    
    Usage:
        extractor = FactExtractor()
        facts = await extractor.extract_facts(document, source_type)
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        provider: LLMProvider | str | None = None,
    ):
        """
        Initialize the fact extractor.
        
        Args:
            llm_client: Optional pre-configured LLM client
            model: Model to use for extraction (defaults to DEFAULT_EXTRACTION_MODEL)
            provider: LLM provider if creating new client (defaults to DEFAULT_LLM_PROVIDER)
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
    
    # Chunk size for optimal fact extraction (configurable via CHUNK_SIZE_CHARS env var)
    # With large context models (Gemini 3 Flash = 1M tokens), we can use bigger chunks
    # But smaller chunks still help the LLM focus and extract more granular facts
    @property
    def CHUNK_SIZE_CHARS(self) -> int:
        return get_config().CHUNK_SIZE_CHARS
    
    # Minimum document size before chunking kicks in (~75% of chunk size)
    @property
    def MIN_CHUNK_THRESHOLD(self) -> int:
        return int(get_config().CHUNK_SIZE_CHARS * 0.75)
    
    async def extract_facts(
        self,
        document: "ExtractedDocument",
        source_type: SourceType | str,
    ) -> list[Fact]:
        """
        Extract facts from a document.
        
        Args:
            document: The extracted document content
            source_type: Type of source (affects confidence scoring)
            
        Returns:
            List of extracted facts
        """
        from src.extraction.normalizer import normalize_to_prompt
        
        if isinstance(source_type, str):
            source_type = SourceType(source_type)
        
        client = await self._get_client()
        
        # Prepare the document content for the prompt
        document_content = normalize_to_prompt(document)
        
        logger.info(f"Extracting facts from {document.filename} ({len(document_content)} chars)")
        
        # Process in chunks for more thorough extraction
        # Smaller chunks help the LLM focus and extract more granular facts
        if len(document_content) > self.MIN_CHUNK_THRESHOLD:
            logger.info(f"Processing document in chunks for thorough extraction")
            return await self._extract_facts_chunked(
                document=document,
                document_content=document_content,
                source_type=source_type,
                client=client,
            )
        
        # For small documents, process in one go
        messages = [
            {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract ALL facts from this document. Be thorough - extract every data point, number, date, and claim as a separate fact:\n\n{document_content}"},
        ]
        
        response = await client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=0.3,  # Lower temperature for more consistent extraction
            response_format={"type": "json_object"},
        )
        
        # Parse the response
        raw_facts = self._parse_response(response)
        
        # Convert to Fact objects with proper scoring
        facts = self._process_raw_facts(
            raw_facts=raw_facts,
            source_document=document.filename,
            source_type=source_type,
        )
        
        logger.info(f"Extracted {len(facts)} facts from {document.filename}")
        
        return facts
    
    async def _extract_facts_chunked(
        self,
        document: "ExtractedDocument",
        document_content: str,
        source_type: SourceType,
        client: LLMClient,
    ) -> list[Fact]:
        """
        Extract facts from a document by processing it in chunks.
        
        Smaller chunks help the LLM focus and extract more granular facts.
        Splits by sections if available, otherwise by character count.
        """
        all_facts: list[Fact] = []
        
        # Try to split by document sections first
        if document.sections and len(document.sections) > 1:
            chunks = self._chunk_by_sections(document)
        else:
            chunks = self._chunk_by_size(document_content)
        
        logger.info(f"Processing {len(chunks)} chunks for {document.filename}")
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            messages = [
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract ALL facts from this document section (part {i} of {len(chunks)}). Be extremely thorough - every number, date, percentage, name, and claim should be a separate fact:\n\n{chunk}"},
            ]
            
            try:
                response = await client.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                
                raw_facts = self._parse_response(response)
                facts = self._process_raw_facts(
                    raw_facts=raw_facts,
                    source_document=document.filename,
                    source_type=source_type,
                )
                all_facts.extend(facts)
                logger.info(f"Extracted {len(facts)} facts from chunk {i}")
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i}: {e}")
                # Continue with other chunks
                continue
        
        logger.info(f"Extracted {len(all_facts)} total facts from {document.filename}")
        return all_facts
    
    def _chunk_by_sections(self, document: "ExtractedDocument") -> list[str]:
        """Split document by sections, grouping to stay under chunk size limit."""
        from src.extraction.normalizer import ContentType
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in document.sections:
            section_text = f"## {section.title or 'Section'}\n{section.content}\n\n"
            section_size = len(section_text)
            
            if current_size + section_size > self.CHUNK_SIZE_CHARS and current_chunk:
                # Save current chunk and start new one
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # If a single section is too large, split it by size
            if section_size > self.CHUNK_SIZE_CHARS:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                # Split the large section
                chunks.extend(self._chunk_by_size(section_text))
            else:
                current_chunk.append(section_text)
                current_size += section_size
        
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def _chunk_by_size(self, content: str) -> list[str]:
        """Split content by size, trying to break at paragraph boundaries."""
        chunks = []
        
        # Split by double newlines (paragraphs) first
        paragraphs = content.split("\n\n")
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_with_spacing = para + "\n\n"
            para_size = len(para_with_spacing)
            
            if current_size + para_size > self.CHUNK_SIZE_CHARS and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # If a single paragraph is too large, split it roughly
            if para_size > self.CHUNK_SIZE_CHARS:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split at sentence boundaries or just by size
                for j in range(0, len(para), self.CHUNK_SIZE_CHARS):
                    chunks.append(para[j:j + self.CHUNK_SIZE_CHARS])
            else:
                current_chunk.append(para_with_spacing)
                current_size += para_size
        
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def _parse_response(self, response: LLMResponse) -> list[ExtractedFactRaw]:
        """Parse the LLM response into raw facts."""
        try:
            data = json.loads(response.content)
            
            # Validate with Pydantic
            extraction_response = FactExtractionResponse(**data)
            return extraction_response.facts
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response content: {response.content[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            raise ValueError(f"Failed to parse fact extraction response: {e}")
    
    def _process_raw_facts(
        self,
        raw_facts: list[ExtractedFactRaw],
        source_document: str,
        source_type: SourceType,
    ) -> list[Fact]:
        """
        Process raw facts into final Fact objects.
        
        Applies:
        - UUID generation
        - Timestamp
        - Default believability based on source type if not provided
        """
        default_believability = BASE_BELIEVABILITY_SCORES.get(
            source_type, Believability.NEEDS_VERIFICATION
        )
        timestamp = datetime.utcnow()
        
        facts = []
        for raw in raw_facts:
            # Use LLM-provided believability or fall back to source-type default
            believability = getattr(raw, 'believability', None)
            if believability is None:
                believability = default_believability
            elif isinstance(believability, str):
                # Handle string values from JSON parsing
                try:
                    believability = Believability(believability)
                except ValueError:
                    believability = default_believability
            
            # Use LLM-provided relevance or fall back to additional_context
            relevance = getattr(raw, 'relevance', None)
            if relevance is None:
                relevance = RelevanceLevel.ADDITIONAL_CONTEXT
            elif isinstance(relevance, str):
                # Handle string values from JSON parsing
                try:
                    relevance = RelevanceLevel(relevance)
                except ValueError:
                    relevance = RelevanceLevel.ADDITIONAL_CONTEXT
            
            # Ensure all chapters have scores (default to 0.0)
            chapter_relevance = {ch: 0.0 for ch in STANDARD_CHAPTERS}
            chapter_relevance.update(raw.chapter_relevance)
            
            fact = Fact(
                id=str(uuid.uuid4()),
                content=raw.content,
                source_quote=raw.content,
                source_documents=[source_document],
                source_type=source_type,
                believability=believability,
                relevance=relevance,
                chapter_relevance=chapter_relevance,
                extraction_timestamp=timestamp,
                usage_count=0,
                used_in_chapters=[],
                embedding=None,
            )
            facts.append(fact)
        
        return facts
    
    async def extract_facts_batch(
        self,
        documents: list[tuple["ExtractedDocument", SourceType | str]],
    ) -> list[Fact]:
        """
        Extract facts from multiple documents.
        
        Args:
            documents: List of (document, source_type) tuples
            
        Returns:
            Combined list of all extracted facts
        """
        all_facts = []
        
        for doc, source_type in documents:
            try:
                facts = await self.extract_facts(doc, source_type)
                all_facts.extend(facts)
            except Exception as e:
                logger.error(f"Failed to extract facts from {doc.filename}: {e}")
                # Continue with other documents
                continue
        
        return all_facts
    
    async def close(self):
        """Close the LLM client if we own it."""
        if self._owns_client and self.llm_client:
            await self.llm_client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Synchronous wrapper
class SyncFactExtractor:
    """Synchronous wrapper for FactExtractor."""
    
    def __init__(self, **kwargs):
        import asyncio
        self._async_extractor = FactExtractor(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None
    
    def _get_loop(self):
        import asyncio
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def extract_facts(
        self,
        document: "ExtractedDocument",
        source_type: SourceType | str,
    ) -> list[Fact]:
        """Synchronous fact extraction."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_extractor.extract_facts(document, source_type)
        )
    
    def extract_facts_batch(
        self,
        documents: list[tuple["ExtractedDocument", SourceType | str]],
    ) -> list[Fact]:
        """Synchronous batch extraction."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_extractor.extract_facts_batch(documents)
        )
    
    def close(self):
        import asyncio
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._async_extractor.close())
            self._loop.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

