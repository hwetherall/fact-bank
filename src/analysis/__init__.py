"""
Analysis module for fact bank operations.

Provides conflict detection, trend analysis, and fact quality assessment.
"""

from .conflict_detector import (
    ConflictDetector,
    SyncConflictDetector,
    FactConflict,
    ConflictSeverity,
)

__all__ = [
    "ConflictDetector",
    "SyncConflictDetector",
    "FactConflict",
    "ConflictSeverity",
]

