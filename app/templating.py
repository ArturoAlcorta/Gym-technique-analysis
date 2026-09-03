from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.exercises import EXERCISES

templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

# Rows store the exercise slug; the UI shows the human name.
templates.env.globals["exercise_name"] = lambda slug: (
    EXERCISES[slug].name if slug in EXERCISES else slug
)
