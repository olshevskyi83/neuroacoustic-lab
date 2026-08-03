"""SQLite engine / session helpers and idempotent schema initialization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from neuroacoustic.exceptions import DatabaseError, DatabaseSchemaError
from neuroacoustic.logging import get_logger
from neuroacoustic.persistence.models import (
    DB_SCHEMA_VERSION,
    DB_SCHEMA_VERSION_KEY,
    Base,
    SchemaMeta,
)

logger = get_logger(__name__)

# WAL improves concurrent readers/writers for the local MVP (documented).
_PRAGMA_STATEMENTS = (
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=5000",
    "PRAGMA journal_mode=WAL",
)


def sqlite_url(path: Path | str) -> str:
    """Build a SQLAlchemy SQLite URL from a filesystem path."""
    resolved = Path(path).expanduser().resolve()
    return f"sqlite:///{resolved.as_posix()}"


def create_db_engine(path: Path | str, *, echo: bool = False) -> Engine:
    """Create an engine with foreign keys, busy timeout, and WAL."""
    db_path = Path(path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        sqlite_url(db_path),
        echo=echo,
        future=True,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        for stmt in _PRAGMA_STATEMENTS:
            cursor.execute(stmt)
        cursor.close()

    return engine


def get_schema_version(session: Session) -> int | None:
    """Return stored db schema version, or None if metadata is absent."""
    row = session.get(SchemaMeta, DB_SCHEMA_VERSION_KEY)
    if row is None:
        return None
    try:
        return int(row.value)
    except ValueError as exc:
        raise DatabaseSchemaError(
            f"Invalid {DB_SCHEMA_VERSION_KEY} value: {row.value!r}"
        ) from exc


def init_db(path: Path | str, *, echo: bool = False) -> Engine:
    """Create tables if needed and verify schema version (idempotent).

    Raises
    ------
    DatabaseSchemaError
        If an existing database declares a different ``db_schema_version``.
        Never deletes or silently rebuilds user data.
    """
    engine = create_db_engine(path, echo=echo)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        existing = get_schema_version(session)
        if existing is None:
            session.merge(
                SchemaMeta(key=DB_SCHEMA_VERSION_KEY, value=str(DB_SCHEMA_VERSION))
            )
            session.commit()
            logger.info(
                "Initialized SQLite schema v%s at %s", DB_SCHEMA_VERSION, path
            )
        elif existing != DB_SCHEMA_VERSION:
            raise DatabaseSchemaError(
                f"Database schema version {existing} is incompatible with "
                f"software schema version {DB_SCHEMA_VERSION}. "
                "Refusing to migrate or delete data automatically. "
                "Backup the database and use a new path, or upgrade with a "
                "compatible migration tool when available."
            )
        else:
            logger.debug("SQLite schema v%s already present at %s", existing, path)

    return engine


def assert_db_usable(path: Path | str) -> Engine:
    """Open an existing database and verify schema, or raise DatabaseError."""
    db_path = Path(path).expanduser()
    if not db_path.exists():
        raise DatabaseError(
            f"Database not found: {db_path}. "
            "Run `neuroacoustic db init --database ...`."
        )
    engine = create_db_engine(db_path)
    with Session(engine) as session:
        try:
            session.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Unusable database {db_path}: {exc}") from exc
        version = get_schema_version(session)
        if version is None:
            raise DatabaseSchemaError(
                f"Database {db_path} has no {DB_SCHEMA_VERSION_KEY} metadata. "
                "It may be incomplete or from an unsupported layout."
            )
        if version != DB_SCHEMA_VERSION:
            raise DatabaseSchemaError(
                f"Database schema version {version} is incompatible with "
                f"software schema version {DB_SCHEMA_VERSION}."
            )
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Transactional session scope."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
