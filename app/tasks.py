import uuid

from app.celery_app import celery_app
from app.config import settings
from app.db import SessionLocal
from app.exercises import BY_ID
from app.models import Analysis
from app.pubsub import publish_event
from app.report import build_report
from services.celery.dtw_service import analyze_dtw
from services.celery.normalization_service import normalize_keypoints
from services.celery.pose_inference_service import run_pose_inference
from services.celery.reference_band_service import load_band
from services.celery.rep_counting_service import count_repetitions
from services.celery.technique_service import analyze_technique_files
from services.celery.video_services import preprocess_video, transcode_h264


def _update(analysis_id: uuid.UUID, **fields) -> None:
    with SessionLocal() as session:
        analysis = session.get(Analysis, analysis_id)
        if analysis is not None:
            for key, value in fields.items():
                setattr(analysis, key, value)
            session.commit()


def _stage(analysis_id: uuid.UUID, stage: str) -> None:
    _update(analysis_id, stage=stage)
    publish_event(analysis_id, {"stage": stage, "status": "processing"})


def _yolo_weights() -> str:
    """Prefer the checkpoint baked into the image; fall back to the name so
    ultralytics fetches it on first use."""
    return str(settings.yolo_weights) if settings.yolo_weights.exists() else "yolo11m.pt"


@celery_app.task(name="app.tasks.analyze_video", bind=True)
def analyze_video(self, analysis_id: str) -> None:
    aid = uuid.UUID(analysis_id)
    job_dir = settings.analyses_dir / analysis_id
    job_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        analysis = session.get(Analysis, aid)
        if analysis is None:
            return
        analysis.status = "processing"
        analysis.celery_task_id = self.request.id
        session.commit()
        video_path = settings.uploads_dir / analysis.video_filename
        exercise_id = analysis.exercise_id
        compare = analysis.compare_dtw

    exercise = BY_ID[exercise_id]

    try:
        _stage(aid, "preprocess")
        preprocessed = preprocess_video(
            str(video_path),
            output_path=job_dir / "preprocessed.mp4",
            target_fps=settings.target_fps,
            max_side=settings.max_video_side,
        )

        _stage(aid, "pose")
        pose_video, keypoints_json = run_pose_inference(
            str(preprocessed),
            output_path=job_dir / "pose.mp4",
            device=settings.device or None,
            single=True,
            batch_size=settings.yolo_batch_size,
            yolo_model_name=_yolo_weights(),
        )
        # OpenCV writes mp4v, which browsers refuse to decode — see transcode_h264.
        transcode_h264(pose_video)

        _stage(aid, "normalize")
        _, normalized_json = normalize_keypoints(keypoints_json, exercise_id)

        _stage(aid, "reps")
        reps, _, rep_count_json = count_repetitions(normalized_json, exercise_id)

        _stage(aid, "technique")
        technique, _ = analyze_technique_files(
            normalized_json, rep_count_json, exercise_id, output_path=job_dir / "technique.json"
        )

        dtw = None
        if compare:
            _stage(aid, "compare")
            dtw, _ = analyze_dtw(rep_count_json, normalized_json, exercise_id, settings.references_dir)

        # Only for the row's summary columns — the report itself is rebuilt from
        # technique.json / dtw_result.json whenever it is requested.
        report = build_report(technique, dtw, exercise, load_band(settings.references_dir, exercise_id))

        _update(aid, status="done", stage="done", total_reps=report["total_reps"],
                score=report["summary"]["score"], error_message=None)
        publish_event(aid, {"stage": "done", "status": "done", "reps": report["total_reps"]})
    except Exception as exc:
        _update(aid, status="error", stage="error", error_message=str(exc)[:500])
        publish_event(aid, {"stage": "error", "status": "error", "message": str(exc)[:200]})
        raise
    finally:
        _free_gpu()


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch missing or CUDA unavailable — nothing to release
        pass
