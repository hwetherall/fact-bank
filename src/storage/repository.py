"""
SQLite repository for facts and documents.

Provides CRUD operations with connection pooling and query support.
"""

import json
import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv

from .models import Fact, Document, SourceType, RelevanceLevel, Believability, STANDARD_CHAPTERS, FactCorrection

load_dotenv()

logger = logging.getLogger(__name__)


class FactRepository:
    """
    SQLite repository for storing and querying facts.
    
    Usage:
        repo = FactRepository()
        repo.insert_fact(fact)
        facts = repo.get_all_facts()
    """
    
    def __init__(self, db_path: str | None = None):
        """
        Initialize the repository.
        
        Args:
            db_path: Path to SQLite database file (defaults to env var or data/factor.db)
        """
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "data/factor.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    page_count INTEGER,
                    sheet_count INTEGER,
                    fact_count INTEGER DEFAULT 0
                )
            """)
            
            # Facts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_documents TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    believability TEXT NOT NULL DEFAULT 'needs_verification',
                    relevance TEXT NOT NULL DEFAULT 'additional_context',
                    chapter_relevance TEXT NOT NULL,
                    extraction_timestamp TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    used_in_chapters TEXT DEFAULT '[]',
                    embedding TEXT,
                    source_quote TEXT
                )
            """)

            # Schema migrations for existing databases
            cursor.execute("PRAGMA table_info(facts)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "source_quote" not in columns:
                cursor.execute("ALTER TABLE facts ADD COLUMN source_quote TEXT")
            # Migration: Add new columns for believability/relevance if upgrading from old schema
            if "believability" not in columns:
                cursor.execute("ALTER TABLE facts ADD COLUMN believability TEXT DEFAULT 'needs_verification'")
            if "relevance" not in columns:
                cursor.execute("ALTER TABLE facts ADD COLUMN relevance TEXT DEFAULT 'additional_context'")
            
            # Store schema info for insert compatibility
            self._has_old_schema = "confidence" in columns and "importance" in columns
            
            # Corrections table for tracking user edits
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id TEXT PRIMARY KEY,
                    fact_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT NOT NULL,
                    new_value TEXT NOT NULL,
                    correction_timestamp TEXT NOT NULL,
                    FOREIGN KEY (fact_id) REFERENCES facts(id)
                )
            """)
            
            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_relevance 
                ON facts(relevance)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_usage_count 
                ON facts(usage_count)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_believability 
                ON facts(believability)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_corrections_fact_id 
                ON corrections(fact_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_corrections_timestamp 
                ON corrections(correction_timestamp)
            """)
            
            logger.info(f"Database initialized at {self.db_path}")
    
    # ==================== Fact Operations ====================
    
    def insert_fact(self, fact: Fact) -> str:
        """
        Insert a new fact.
        
        Args:
            fact: The fact to insert
            
        Returns:
            The fact's ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Handle old schema compatibility (databases with confidence/importance columns)
            if getattr(self, '_has_old_schema', False):
                cursor.execute("""
                    INSERT INTO facts (
                        id, content, source_documents, source_type, believability,
                        relevance, chapter_relevance, extraction_timestamp,
                        usage_count, used_in_chapters, embedding, source_quote,
                        confidence, importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fact.id,
                    fact.content,
                    json.dumps(fact.source_documents),
                    fact.source_type,
                    fact.believability,
                    fact.relevance,
                    json.dumps(fact.chapter_relevance),
                    fact.extraction_timestamp.isoformat(),
                    fact.usage_count,
                    json.dumps(fact.used_in_chapters),
                    json.dumps(fact.embedding) if fact.embedding else None,
                    fact.source_quote,
                    fact.confidence,  # Legacy property
                    fact.importance,  # Legacy property
                ))
            else:
                cursor.execute("""
                    INSERT INTO facts (
                        id, content, source_documents, source_type, believability,
                        relevance, chapter_relevance, extraction_timestamp,
                        usage_count, used_in_chapters, embedding, source_quote
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fact.id,
                    fact.content,
                    json.dumps(fact.source_documents),
                    fact.source_type,
                    fact.believability,
                    fact.relevance,
                    json.dumps(fact.chapter_relevance),
                    fact.extraction_timestamp.isoformat(),
                    fact.usage_count,
                    json.dumps(fact.used_in_chapters),
                    json.dumps(fact.embedding) if fact.embedding else None,
                    fact.source_quote,
                ))
        
        logger.debug(f"Inserted fact {fact.id[:8]}")
        return fact.id
    
    def insert_facts(self, facts: list[Fact]) -> int:
        """
        Insert multiple facts in a batch.
        
        Args:
            facts: List of facts to insert
            
        Returns:
            Number of facts inserted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Handle old schema compatibility (databases with confidence/importance columns)
            if getattr(self, '_has_old_schema', False):
                cursor.executemany("""
                    INSERT INTO facts (
                        id, content, source_documents, source_type, believability,
                        relevance, chapter_relevance, extraction_timestamp,
                        usage_count, used_in_chapters, embedding, source_quote,
                        confidence, importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        f.id,
                        f.content,
                        json.dumps(f.source_documents),
                        f.source_type,
                        f.believability,
                        f.relevance,
                        json.dumps(f.chapter_relevance),
                        f.extraction_timestamp.isoformat(),
                        f.usage_count,
                        json.dumps(f.used_in_chapters),
                        json.dumps(f.embedding) if f.embedding else None,
                        f.source_quote,
                        f.confidence,  # Legacy property
                        f.importance,  # Legacy property
                    )
                    for f in facts
                ])
            else:
                cursor.executemany("""
                    INSERT INTO facts (
                        id, content, source_documents, source_type, believability,
                        relevance, chapter_relevance, extraction_timestamp,
                        usage_count, used_in_chapters, embedding, source_quote
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        f.id,
                        f.content,
                        json.dumps(f.source_documents),
                        f.source_type,
                        f.believability,
                        f.relevance,
                        json.dumps(f.chapter_relevance),
                        f.extraction_timestamp.isoformat(),
                        f.usage_count,
                        json.dumps(f.used_in_chapters),
                        json.dumps(f.embedding) if f.embedding else None,
                        f.source_quote,
                    )
                    for f in facts
                ])
        
        logger.info(f"Inserted {len(facts)} facts")
        return len(facts)
    
    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        """Convert a database row to a Fact object."""
        # Get column names from the row
        columns = row.keys()
        
        # Handle migration from old schema: map old confidence/importance to new fields
        if "believability" in columns:
            believability = row["believability"] or "needs_verification"
        else:
            believability = "needs_verification"
        
        if "relevance" in columns:
            relevance = row["relevance"] or "additional_context"
        else:
            relevance = "additional_context"
        
        return Fact(
            id=row["id"],
            content=row["content"],
            source_quote=row["source_quote"] if "source_quote" in columns else None,
            source_documents=json.loads(row["source_documents"]),
            source_type=row["source_type"],
            believability=believability,
            relevance=relevance,
            chapter_relevance=json.loads(row["chapter_relevance"]),
            extraction_timestamp=datetime.fromisoformat(row["extraction_timestamp"]),
            usage_count=row["usage_count"],
            used_in_chapters=json.loads(row["used_in_chapters"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        )
    
    def get_fact(self, fact_id: str) -> Fact | None:
        """Get a fact by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_fact(row)
    
    def get_all_facts(self) -> list[Fact]:
        """Get all facts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM facts ORDER BY extraction_timestamp DESC")
            return [self._row_to_fact(row) for row in cursor.fetchall()]
    
    def get_facts_by_relevance(self, relevance: RelevanceLevel | str) -> list[Fact]:
        """Get facts filtered by relevance level."""
        if isinstance(relevance, RelevanceLevel):
            relevance = relevance.value
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM facts WHERE relevance = ? ORDER BY extraction_timestamp DESC",
                (relevance,)
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]
    
    def get_facts_by_believability(self, believability: Believability | str) -> list[Fact]:
        """Get facts filtered by believability level."""
        if isinstance(believability, Believability):
            believability = believability.value
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM facts WHERE believability = ? ORDER BY extraction_timestamp DESC",
                (believability,)
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]
    
    def get_non_noise_facts(self) -> list[Fact]:
        """Get all facts excluding noise."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM facts WHERE relevance != ? ORDER BY extraction_timestamp DESC",
                (RelevanceLevel.NOISE.value,)
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]
    
    def get_facts_by_chapter(
        self, 
        chapter: str, 
        min_relevance: float = 0.5
    ) -> list[Fact]:
        """
        Get facts relevant to a specific chapter.
        
        Args:
            chapter: Chapter name
            min_relevance: Minimum relevance score threshold
            
        Returns:
            List of relevant facts, sorted by relevance
        """
        all_facts = self.get_all_facts()
        
        relevant = [
            f for f in all_facts
            if f.chapter_relevance.get(chapter, 0.0) >= min_relevance
        ]
        
        # Sort by relevance to this chapter
        relevant.sort(
            key=lambda f: f.chapter_relevance.get(chapter, 0.0),
            reverse=True
        )
        
        return relevant
    
    def get_facts_by_source(self, filename: str) -> list[Fact]:
        """Get facts from a specific source document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM facts WHERE source_documents LIKE ?",
                (f'%"{filename}"%',)
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]
    
    def get_underused_facts(self, max_usage: int = 0) -> list[Fact]:
        """
        Get facts that have been used infrequently.
        
        Args:
            max_usage: Maximum usage count to include
            
        Returns:
            List of underused facts, sorted by usage count ascending
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM facts WHERE usage_count <= ? ORDER BY usage_count ASC",
                (max_usage,)
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]
    
    def search_facts(self, query: str) -> list[Fact]:
        """
        Search facts by content.
        
        Args:
            query: Search string (case-insensitive substring match)
            
        Returns:
            List of matching facts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM facts WHERE content LIKE ? ORDER BY confidence DESC",
                (f"%{query}%",)
            )
            return [self._row_to_fact(row) for row in cursor.fetchall()]
    
    def update_usage(
        self, 
        fact_id: str, 
        chapter: str | None = None
    ) -> bool:
        """
        Increment usage count and optionally record the chapter.
        
        Args:
            fact_id: The fact's ID
            chapter: Optional chapter where the fact was used
            
        Returns:
            True if updated, False if fact not found
        """
        fact = self.get_fact(fact_id)
        if fact is None:
            return False
        
        new_count = fact.usage_count + 1
        used_chapters = fact.used_in_chapters.copy()
        
        if chapter and chapter not in used_chapters:
            used_chapters.append(chapter)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE facts 
                SET usage_count = ?, used_in_chapters = ?
                WHERE id = ?
            """, (new_count, json.dumps(used_chapters), fact_id))
        
        return True
    
    def delete_fact(self, fact_id: str) -> bool:
        """
        Delete a fact by ID.
        
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.debug(f"Deleted fact {fact_id[:8]}")
        
        return deleted
    
    def clear_all_facts(self):
        """Delete all facts from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM facts")
            count = cursor.rowcount
        
        logger.info(f"Cleared {count} facts from database")
    
    def update_fact(self, fact_id: str, updates: dict) -> bool:
        """
        Update a fact's fields.
        
        Args:
            fact_id: The fact's ID
            updates: Dictionary of field names to new values
            
        Returns:
            True if updated, False if fact not found
        """
        fact = self.get_fact(fact_id)
        if fact is None:
            return False
        
        allowed_fields = ["content", "believability", "relevance", "chapter_relevance"]
        update_parts = []
        update_values = []
        
        for field, value in updates.items():
            if field not in allowed_fields:
                continue
            if field == "chapter_relevance" and isinstance(value, dict):
                value = json.dumps(value)
            update_parts.append(f"{field} = ?")
            update_values.append(value)
        
        if not update_parts:
            return False
        
        update_values.append(fact_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE facts SET {', '.join(update_parts)} WHERE id = ?",
                tuple(update_values)
            )
        
        logger.debug(f"Updated fact {fact_id[:8]}: {list(updates.keys())}")
        return True
    
    # ==================== Correction Operations ====================
    
    def insert_correction(self, correction: FactCorrection) -> str:
        """
        Insert a new correction record.
        
        Args:
            correction: The correction to insert
            
        Returns:
            The correction's ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO corrections (
                    id, fact_id, field_name, old_value, new_value, correction_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                correction.id,
                correction.fact_id,
                correction.field_name,
                correction.old_value,
                correction.new_value,
                correction.correction_timestamp.isoformat(),
            ))
        
        logger.debug(f"Inserted correction {correction.id[:8]} for fact {correction.fact_id[:8]}")
        return correction.id
    
    def _row_to_correction(self, row: sqlite3.Row) -> FactCorrection:
        """Convert a database row to a FactCorrection object."""
        return FactCorrection(
            id=row["id"],
            fact_id=row["fact_id"],
            field_name=row["field_name"],
            old_value=row["old_value"],
            new_value=row["new_value"],
            correction_timestamp=datetime.fromisoformat(row["correction_timestamp"]),
        )
    
    def get_all_corrections(self) -> list[FactCorrection]:
        """Get all corrections, ordered by timestamp (most recent first)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM corrections ORDER BY correction_timestamp DESC")
            return [self._row_to_correction(row) for row in cursor.fetchall()]
    
    def get_corrections_for_fact(self, fact_id: str) -> list[FactCorrection]:
        """Get all corrections for a specific fact."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM corrections WHERE fact_id = ? ORDER BY correction_timestamp DESC",
                (fact_id,)
            )
            return [self._row_to_correction(row) for row in cursor.fetchall()]
    
    def get_correction_statistics(self) -> dict:
        """
        Get statistics about corrections for model improvement insights.
        
        Returns:
            Dictionary with correction counts by field name
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total corrections
            cursor.execute("SELECT COUNT(*) FROM corrections")
            total = cursor.fetchone()[0]
            
            # By field name
            cursor.execute("""
                SELECT field_name, COUNT(*) 
                FROM corrections 
                GROUP BY field_name
            """)
            by_field = dict(cursor.fetchall())
            
            # Recent corrections (last 7 days)
            cursor.execute("""
                SELECT COUNT(*) FROM corrections 
                WHERE correction_timestamp > datetime('now', '-7 days')
            """)
            recent = cursor.fetchone()[0]
        
        return {
            "total_corrections": total,
            "by_field": by_field,
            "recent_corrections": recent,
        }
    
    def clear_all_corrections(self):
        """Delete all corrections from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM corrections")
            count = cursor.rowcount
        
        logger.info(f"Cleared {count} corrections from database")
    
    # ==================== Document Operations ====================
    
    def insert_document(self, document: Document) -> str:
        """Insert a new document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (
                    id, filename, file_type, source_type, upload_timestamp,
                    raw_content, page_count, sheet_count, fact_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document.id,
                document.filename,
                document.file_type,
                document.source_type,
                document.upload_timestamp.isoformat(),
                document.raw_content,
                document.page_count,
                document.sheet_count,
                document.fact_count,
            ))
        
        logger.debug(f"Inserted document {document.filename}")
        return document.id
    
    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert a database row to a Document object."""
        return Document(
            id=row["id"],
            filename=row["filename"],
            file_type=row["file_type"],
            source_type=row["source_type"],
            upload_timestamp=datetime.fromisoformat(row["upload_timestamp"]),
            raw_content=row["raw_content"],
            page_count=row["page_count"],
            sheet_count=row["sheet_count"],
            fact_count=row["fact_count"],
        )
    
    def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_document(row)
    
    def get_document_by_filename(self, filename: str) -> Document | None:
        """Get a document by filename."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_document(row)
    
    def get_all_documents(self) -> list[Document]:
        """Get all documents."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY upload_timestamp DESC")
            return [self._row_to_document(row) for row in cursor.fetchall()]
    
    def update_document_fact_count(self, doc_id: str, fact_count: int) -> bool:
        """Update the fact count for a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE documents SET fact_count = ? WHERE id = ?",
                (fact_count, doc_id)
            )
            return cursor.rowcount > 0
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cursor.rowcount > 0
    
    def clear_all_documents(self):
        """Delete all documents from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents")
            count = cursor.rowcount
        
        logger.info(f"Cleared {count} documents from database")
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> dict:
        """
        Get summary statistics about the fact bank.
        
        Returns:
            Dictionary with various statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total facts
            cursor.execute("SELECT COUNT(*) FROM facts")
            total_facts = cursor.fetchone()[0]
            
            # Total documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_documents = cursor.fetchone()[0]
            
            # Facts by relevance
            cursor.execute("""
                SELECT relevance, COUNT(*) 
                FROM facts 
                GROUP BY relevance
            """)
            relevance_breakdown = dict(cursor.fetchall())
            
            # Facts by believability
            cursor.execute("""
                SELECT believability, COUNT(*) 
                FROM facts 
                GROUP BY believability
            """)
            believability_breakdown = dict(cursor.fetchall())
            
            # Source documents
            cursor.execute("SELECT DISTINCT filename FROM documents")
            source_files = [row[0] for row in cursor.fetchall()]
            
            # Legacy importance breakdown (for backwards compatibility)
            importance_breakdown = {
                "high": relevance_breakdown.get("critical", 0),
                "medium": relevance_breakdown.get("chapter_specific", 0),
                "low": relevance_breakdown.get("additional_context", 0) + relevance_breakdown.get("noise", 0),
            }
            
            # Calculate average confidence (using legacy mapping from believability)
            # This is approximate for backwards compatibility
            believability_to_confidence = {
                "verified": 0.95,
                "needs_verification": 0.70,
                "opinion": 0.40,
            }
            total_confidence = 0.0
            for bel, count in believability_breakdown.items():
                total_confidence += believability_to_confidence.get(bel, 0.70) * count
            avg_confidence = total_confidence / total_facts if total_facts > 0 else 0.0
            
        return {
            "total_facts": total_facts,
            "total_documents": total_documents,
            "relevance_breakdown": {
                "critical": relevance_breakdown.get("critical", 0),
                "chapter_specific": relevance_breakdown.get("chapter_specific", 0),
                "additional_context": relevance_breakdown.get("additional_context", 0),
                "noise": relevance_breakdown.get("noise", 0),
            },
            "believability_breakdown": {
                "verified": believability_breakdown.get("verified", 0),
                "needs_verification": believability_breakdown.get("needs_verification", 0),
                "opinion": believability_breakdown.get("opinion", 0),
            },
            # Legacy fields for backwards compatibility
            "importance_breakdown": importance_breakdown,
            "average_confidence": round(avg_confidence, 3),
            "source_documents": source_files,
        }

