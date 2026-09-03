import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import SCHEMA, Base


class Analysis(Base):
    """One uploaded set. Heavy artifacts (annotated video, per-stage JSON) live on
    disk under `data/analyses/<id>/`; this row is the index and the status."""

    __tablename__ = "analyses"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    exercise: Mapped[str] = mapped_column(String, nullable=False)          # slug: bench | squat | rdl
    exercise_id: Mapped[int] = mapped_column(Integer, nullable=False)      # engine id: 1 | 6 | 7
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    compare_dtw: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    stage: Mapped[str | None] = mapped_column(String, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    video_filename: Mapped[str] = mapped_column(String, nullable=False)
    total_reps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # mean per-rep technique score
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
