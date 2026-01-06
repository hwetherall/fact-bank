"""Factor - Fact Bank System for Investment Memos"""

__version__ = "0.1.0"

from src.analysis import (
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

