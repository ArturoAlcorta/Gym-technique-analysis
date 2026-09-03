import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import exercises as catalog
from app.config import settings
from app.db import get_session
from app.models import Analysis
from app.report import build_report
from services.celery.reference_band_service import load_band
from app.schemas import AnalysisOut, ExerciseOut, ReportOut
from app.tasks import analyze_video
from app.templating import templates

router = APIRouter()


def _render_row(request: Request, analysis: Analysis) -> str:
    return templates.get_template("partials/analysis_row.html").render(request=request, analysis=analysis)


def _render_detail(request: Request, analysis: Analysis) -> str:
    return templates.get_template("partials/analysis_detail.html").render(request=request, analysis=analysis)


def _get_or_404(session: Session, analysis_id: uuid.UUID) -> Analysis:
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return analysis


@router.get("/exercises", response_model=list[ExerciseOut])
def list_exercises():
    return [ExerciseOut(slug=e.slug, id=e.id, name=e.name) for e in catalog.EXERCISES.values()]


@router.get("/analyses", response_class=HTMLResponse)
def list_analyses(request: Request, session: Session = Depends(get_session)):
    rows = session.scalars(select(Analysis).order_by(Analysis.created_at.desc())).all()
    return templates.TemplateResponse(request, "partials/analysis_list.html", {"analyses": rows})


@router.get("/api/analyses", response_model=list[AnalysisOut])
def api_list_analyses(session: Session = Depends(get_session)):
    return session.scalars(select(Analysis).order_by(Analysis.created_at.desc())).all()


@router.post("/analyses", response_class=HTMLResponse)
def create_analysis(
    request: Request,
    video: UploadFile,
    name: str = Form(...),
    exercise: str = Form(...),
    weight: float | None = Form(None),
    compare: bool = Form(False),
    session: Session = Depends(get_session),
):
    try:
        ex = catalog.get(exercise)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    analysis_id = uuid.uuid4()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_filename = f"{analysis_id}{suffix}"
    with open(settings.uploads_dir / video_filename, "wb") as fh:
        fh.write(video.file.read())

    analysis = Analysis(
        id=analysis_id, name=name, exercise=ex.slug, exercise_id=ex.id, weight=weight,
        compare_dtw=compare, status="pending", video_filename=video_filename,
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)

    analyze_video.delay(str(analysis_id))

    # The new row goes on top of the list; the detail panel is swapped
    # out-of-band so the upload you just made is the one on screen.
    return (
        _render_row(request, analysis)
        + f'<div id="analysis-detail-panel" hx-swap-oob="innerHTML">{_render_detail(request, analysis)}</div>'
    )


@router.get("/analyses/{analysis_id}/row", response_class=HTMLResponse)
def analysis_row(request: Request, analysis_id: uuid.UUID, session: Session = Depends(get_session)):
    return _render_row(request, _get_or_404(session, analysis_id))


@router.get("/analyses/{analysis_id}", response_class=HTMLResponse)
def analysis_detail(request: Request, analysis_id: uuid.UUID, session: Session = Depends(get_session)):
    return _render_detail(request, _get_or_404(session, analysis_id))


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/analyses/{analysis_id}/results", response_model=ReportOut)
def analysis_results(analysis_id: uuid.UUID, session: Session = Depends(get_session)):
    """Built on read, not cached: the engine artifacts are the durable record, and
    which metrics the UI surfaces is a presentation choice that can change without
    reprocessing anyone's video."""
    analysis = _get_or_404(session, analysis_id)
    job_dir = settings.analyses_dir / str(analysis_id)
    technique = _load(job_dir / "technique.json")
    if technique is None:
        raise HTTPException(status_code=409, detail=f"analysis is {analysis.status}, no results yet")
    return build_report(
        technique,
        _load(job_dir / "dtw_result.json"),
        catalog.get(analysis.exercise),
        load_band(settings.references_dir, analysis.exercise_id),
    )


@router.get("/analyses/{analysis_id}/video")
def analysis_video(analysis_id: uuid.UUID, session: Session = Depends(get_session)):
    _get_or_404(session, analysis_id)
    video_path = settings.analyses_dir / str(analysis_id) / "pose.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="annotated video not available")
    return FileResponse(video_path, media_type="video/mp4")
