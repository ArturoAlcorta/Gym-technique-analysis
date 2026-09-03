from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.exercises import EXERCISES
from app.db import get_session, init_db
from app.models import Analysis
from app.routers import analyses, events
from app.templating import templates

app = FastAPI(title="gym-technique-analyzer")

app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
app.include_router(analyses.router)
app.include_router(events.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def index(request: Request, session: Session = Depends(get_session)):
    rows = session.scalars(select(Analysis).order_by(Analysis.created_at.desc())).all()
    return templates.TemplateResponse(
        request, "index.html", {"analyses": rows, "exercises": list(EXERCISES.values())}
    )
