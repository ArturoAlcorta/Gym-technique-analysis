from celery import Celery

from app.config import settings

celery_app = Celery(
    "gym_technique_analyzer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.task_routes = {"app.tasks.analyze_video": {"queue": "technique"}}
celery_app.conf.task_serializer = "json"
celery_app.conf.result_expires = 3600
celery_app.autodiscover_tasks(["app"])
