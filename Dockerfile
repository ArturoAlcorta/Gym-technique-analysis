FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# ffmpeg: video decode/encode for the preprocessing stage.
# libgl1/libglib2.0-0: required by opencv-python-headless at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first, against pyproject.toml alone: editing a source file must
# not invalidate the layer that installs several GB of torch. The stub packages
# exist only so the metadata resolves; the real code lands in the next layer and
# is installed with --no-deps, which is a couple of seconds.
COPY pyproject.toml ./
RUN mkdir -p app src/services \
    && touch app/__init__.py src/services/__init__.py \
    && uv pip install --system --no-cache . \
    && rm -rf app src

COPY app ./app
COPY src ./src
COPY references ./references
COPY scripts ./scripts
RUN uv pip install --system --no-cache --no-deps .

# The torch wheels installed above bundle their own CUDA runtime, so this image
# needs no CUDA base — only the host driver plus the nvidia-container-toolkit,
# wired up by the GPU block in docker-compose.yml.

# Bake the YOLO detector into the image (ultralytics downloads into the CWD).
# `app` is installed into site-packages, so the path defaults derived from the
# source tree no longer apply — point them at the image layout explicitly.
ENV YOLO_CONFIG_DIR=/tmp \
    HF_HOME=/srv/data/hf-cache \
    DATA_DIR=/srv/data \
    REFERENCES_DIR=/srv/references \
    YOLO_MODELS_DIR=/srv/models
RUN mkdir -p /srv/models && cd /srv/models \
    && python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

# Run as a non-root, fixed-UID user so files written to the bind-mounted ./data
# (uploads, annotated videos, JSON artifacts) are owned by a regular host user
# rather than root. SynthPose weights are fetched on first inference into
# HF_HOME, which lives inside that same mount so the download survives restarts.
RUN useradd --uid 1000 --create-home gym && chown -R gym:gym /srv
USER gym

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
