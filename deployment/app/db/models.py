from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class PredictionRequest(Base):
    __tablename__ = "prediction_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36), index=True, nullable=False, unique=True
    )
    input_type: Mapped[str] = mapped_column(String(20), nullable=False)
    final_label: Mapped[int] = mapped_column(Integer, nullable=False)
    final_label_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    final_prob: Mapped[float] = mapped_column(Float, nullable=False)
    final_probs_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    binary_model_results: Mapped[list["PredictionBinaryModelResult"]] = relationship(
        back_populates="prediction_request",
        cascade="all, delete-orphan",
    )
    disease_result: Mapped["PredictionDiseaseResult | None"] = relationship(
        back_populates="prediction_request",
        uselist=False,
        cascade="all, delete-orphan",
    )
    image_links: Mapped["PredictionImageLink | None"] = relationship(
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )


class PredictionBinaryModelResult(Base):
    __tablename__ = "prediction_binary_model_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_request_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prediction_requests.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    label_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prob: Mapped[float] = mapped_column(Float, nullable=False)
    probs_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    prediction_request: Mapped["PredictionRequest"] = relationship(
        back_populates="binary_model_results"
    )


class PredictionDiseaseResult(Base):
    __tablename__ = "prediction_disease_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_request_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prediction_requests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    label_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    probs_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    prediction_request: Mapped["PredictionRequest"] = relationship(
        back_populates="disease_result"
    )


class PredictionImageLink(Base):
    __tablename__ = "prediction_image_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_request_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prediction_requests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    source_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    mask_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    roi_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    overlay_url: Mapped[str | None] = mapped_column(String(300), nullable=True)

    prediction: Mapped["PredictionRequest"] = relationship(back_populates="image_links")
