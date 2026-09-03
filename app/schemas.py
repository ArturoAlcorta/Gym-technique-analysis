import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ExerciseOut(BaseModel):
    slug: str
    id: int
    name: str


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    exercise: str
    exercise_id: int
    weight: float | None
    compare_dtw: bool
    status: str
    stage: str | None
    video_filename: str
    total_reps: int | None
    score: float | None
    error_message: str | None
    created_at: datetime


class MetricOut(BaseModel):
    key: str
    label: str
    value: float | bool | None
    display: str
    score: float | None = None   # 0-100 sub-score, when the metric has one
    status: str                  # ok | warn | bad | info | na
    hint: str = ""


class TimingOut(BaseModel):
    start_s: float | None = None
    end_s: float | None = None
    total_s: float | None = None
    eccentric_s: float | None = None
    concentric_s: float | None = None


class JointScoreOut(BaseModel):
    joint: str
    label: str
    score: float


class ComparisonOut(BaseModel):
    """Present only when the analysis was run with the DTW comparison enabled."""

    best_score: float | None
    mean_score: float | None
    best_reference: str | None
    n_references: int
    by_joint: list[JointScoreOut]


class FaultOut(BaseModel):
    code: str
    severity: str
    cue: str


class RepOut(BaseModel):
    rep_number: int
    timing: TimingOut
    score: float | None = None           # 0.5·pattern + 0.5·metrics, or whichever half exists
    metric_score: float | None = None
    pattern_score: float | None = None   # only with the DTW comparison enabled
    metrics: list[MetricOut]
    comparison: ComparisonOut | None = None
    faults: list[FaultOut] = []


class SummaryOut(BaseModel):
    score: float | None                  # the headline number: mean rep score
    metric_score: float | None
    pattern_score: float | None
    avg_total_s: float | None
    avg_eccentric_s: float | None
    avg_concentric_s: float | None
    n_faults: int


class ReferenceOut(BaseModel):
    files: list[str]
    n: int | None
    angles: list[str] = []


class ReportOut(BaseModel):
    exercise: ExerciseOut
    compare: bool
    total_reps: int
    summary: SummaryOut
    reference: ReferenceOut | None = None
    reps: list[RepOut]
