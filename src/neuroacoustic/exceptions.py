"""Domain-specific exceptions for NeuroAcoustic Lab."""

from __future__ import annotations


class NeuroAcousticError(Exception):
    """Base error for the NeuroAcoustic Lab application."""


class ConfigError(NeuroAcousticError):
    """Invalid or missing configuration."""


class AudioValidationError(NeuroAcousticError):
    """Audio file failed validation before or during probing."""


class AudioDecodeError(NeuroAcousticError):
    """Audio file could not be decoded."""


class DependencyError(NeuroAcousticError):
    """A required system or Python dependency is missing or unusable."""


class NotImplementedMilestoneError(NeuroAcousticError):
    """Feature deferred to a later milestone."""


class DatabaseError(NeuroAcousticError):
    """SQLite database is missing, unusable, or otherwise unavailable."""


class DatabaseSchemaError(DatabaseError):
    """Existing database schema is incompatible with this software version."""


class AnalysisNotFoundError(DatabaseError):
    """Requested analysis / track ID does not exist."""
