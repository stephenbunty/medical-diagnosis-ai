"""Medical image study and prediction history."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class Study(Base):
    """One uploaded exam (X-ray or MRI) belonging to a user."""

    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    modality: Mapped[str] = mapped_column(String(32), nullable=False)  # chest_xray | brain_mri
    original_filename: Mapped[str] = mapped_column(String(512), default="")
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owner: Mapped["User"] = relationship("User", back_populates="studies")
    predictions: Mapped[list["PredictionRecord"]] = relationship(
        "PredictionRecord", back_populates="study", cascade="all, delete-orphan"
    )


class PredictionRecord(Base):
    """Stored inference output for audit and dashboard history."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id", ondelete="CASCADE"), index=True)
    labels_json: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_json: Mapped[str] = mapped_column(Text, nullable=False)
    heatmap_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    mask_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    report_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity_score: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    study: Mapped["Study"] = relationship("Study", back_populates="predictions")
