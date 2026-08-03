"""SQLite persistence for acoustic fingerprints (Milestone 4B)."""

from neuroacoustic.persistence.config_hash import (
    compute_config_hash,
    scientific_config_payload,
)
from neuroacoustic.persistence.database import (
    assert_db_usable,
    init_db,
    session_scope,
)
from neuroacoustic.persistence.models import DB_SCHEMA_VERSION, Analysis, Track
from neuroacoustic.persistence.repository import (
    PersistDisposition,
    analysis_to_summary_dict,
    compute_stats,
    fingerprint_from_analysis,
    get_completed_analysis,
    list_analyses,
    persist_completed_analysis,
    record_failed_analysis,
    resolve_analysis_or_track_id,
)

__all__ = [
    "Analysis",
    "DB_SCHEMA_VERSION",
    "PersistDisposition",
    "Track",
    "analysis_to_summary_dict",
    "assert_db_usable",
    "compute_config_hash",
    "compute_stats",
    "fingerprint_from_analysis",
    "get_completed_analysis",
    "init_db",
    "list_analyses",
    "persist_completed_analysis",
    "record_failed_analysis",
    "resolve_analysis_or_track_id",
    "scientific_config_payload",
    "session_scope",
]
