"""SQLAlchemy declarative base shared by all database models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for ORM models.

    All SQLAlchemy prediction-log models inherit from this class so table
    metadata can be created during startup and shared by tests, direct DB
    logging, and Kafka consumers.
    """

    # SQLAlchemy uses the subclass itself as the metadata anchor; no additional
    # fields are required on the shared base class.
    pass
