from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://gym:gym@localhost:5432/gym"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    redis_url: str = "redis://localhost:6379/2"

    # Torch device for pose inference: "cpu", "cuda", "0"… Empty = auto-detect
    # (CUDA if available, else MPS, else CPU). SynthPose is a ViT-Huge, so CPU
    # works but is slow — a GPU is strongly recommended.
    device: str | None = None

    yolo_models_dir: Path = PROJECT_DIR / "models"
    references_dir: Path = PROJECT_DIR / "references"
    data_dir: Path = PROJECT_DIR / "data"

    # Video preprocessing (before pose inference)
    target_fps: float = 15.0
    max_video_side: int = 1920
    # Frames per YOLO forward pass during person detection.
    yolo_batch_size: int = 8

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def analyses_dir(self) -> Path:
        return self.data_dir / "analyses"

    @property
    def yolo_weights(self) -> Path:
        return self.yolo_models_dir / "yolo11m.pt"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.analyses_dir.mkdir(parents=True, exist_ok=True)
