from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.configs.config import AppConfig
from app.utils.errors import ServiceError


engine = None
SessionLocal = None

if AppConfig.DB_LOGGING_ENABLED:
    engine = create_engine(
        AppConfig.get_database_url(),
        echo=AppConfig.DB_ECHO,
        pool_pre_ping=True,
        future=True,
    )

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        future=True,
    )


def get_db() -> Generator:
    if SessionLocal is None:
        raise ServiceError("DB_DISABLED", "Database logging is disabled.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
