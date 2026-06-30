"""Database engine and session factory wiring for prediction logging."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.configs.config import AppConfig
from app.utils.errors import ServiceError


engine = None
SessionLocal = None

if AppConfig.DB_LOGGING_ENABLED:
    # Engine creation is gated by configuration so local demos and CI can run
    # without requiring PostgreSQL or SQL Server credentials.
    engine = create_engine(
        AppConfig.get_database_url(),
        echo=AppConfig.DB_ECHO,
        pool_pre_ping=True,
        future=True,
    )

    SessionLocal = sessionmaker(
        bind=engine,
        # Routes commit explicitly after all child rows are staged.
        autoflush=False,
        autocommit=False,
        future=True,
    )


def get_db() -> Generator:
    """Yield a SQLAlchemy session for FastAPI dependencies.

    Yields:
        Active SQLAlchemy session bound to the configured engine.

    Raises:
        ServiceError: If database logging is disabled and the logs endpoint
        still attempts to request a session.

    Notes:
        FastAPI closes the generator after request handling, which guarantees
        the session is released even when route serialization fails.
    """
    if SessionLocal is None:
        raise ServiceError("DB_DISABLED", "Database logging is disabled.")

    # FastAPI advances this generator once per request and then resumes it for
    # cleanup, which gives route handlers a short-lived unit-of-work session.
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
