"""
Turn a video of a well-executed set into per-rep reference files.

Runs the SAME front end a user submission goes through, so references and user
clips are measured identically — anything else and the comparison is between two
different measurement procedures rather than between two lifters:

    video → preprocess (long side 1920, 15 fps) → SynthPose 2D pose (single athlete)
          → normalize → 2D rep counter → one JSON per rep

A 7-rep squat produces 7 files. The annotated `<name>_pose.mp4` is kept for
review: watch it, delete the rep files for any rep you would not want a stranger
scored against, then copy the rest into `references/<exercise_id>/`.

The per-rep files are written from the **normalized** keypoints rather than the
raw pose output, purely so they load: the raw JSON nests keypoints as
`frames[].persons[].keypoints`, a list of `{id, name, x, y, score}`, while every
consumer here indexes `frames[].keypoints["L_Knee"]` on a single athlete. It
makes no difference to any measurement — normalization is a translation and a
single isotropic scale, and every metric downstream is an angle or a ratio, so
the numbers come out identical either way (verified: 1e-4° of rounding).

Filming matters more than anything else this script does — see the README:
squat/RDL at 30° off the sagittal plane, bench press at 45° between the frontal
and transverse planes. A reference filmed side-on quietly poisons every
comparison made against it.

Needs the worker's dependencies and the pose model, so run it in the container:

    docker compose exec worker python /srv/scripts/extract_reference_keypoints.py \\
        /srv/data/reference-videos/squat_alex.mp4 6 --out /srv/data/reference-out --name alex

Then move the reps you are keeping into `references/6/` and rebuild the band:

    python -m services.celery.reference_band_service 6 /srv/references
"""

import argparse
import json
import sys
from pathlib import Path

from services.celery.normalization_service import normalize_keypoints
from services.celery.pose_inference_service import run_pose_inference
from services.celery.rep_counting_service import count_repetitions
from services.celery.video_services import preprocess_video


def _write_rep_files(normalized: dict, reps: list[dict], out_dir: Path, stem: str) -> list[Path]:
    """Slice the normalized keypoints into one file per rep, by frame_idx window."""
    frames = normalized["frames"]
    written = []
    for rep in reps:
        start, end = rep["frame_start"], rep["frame_end"]
        rep_frames = [f for f in frames if start <= f["frame_idx"] <= end]
        out = out_dir / f"{stem}_rep_{rep['rep_number']:03d}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({
                "metadata": {
                    **normalized["metadata"],
                    "source_video": stem,
                    "rep_number": rep["rep_number"],
                    "frame_start": start,
                    "frame_end": end,
                },
                "frames": rep_frames,
            }, fh, ensure_ascii=False)
        written.append(out)
    return written


def extract_reference_reps(video_path: Path, exercise_id: int, out_dir: Path, name: str | None = None,
                           keep_intermediates: bool = False) -> list[Path]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or video_path.stem

    preprocessed = out_dir / f"{stem}_preprocessed.mp4"
    print(f"\n▶  Preprocessing {video_path.name}")
    preprocess_video(str(video_path), output_path=preprocessed)

    print("▶  Extracting 2D keypoints (SynthPose, single athlete)…")
    pose_video, pose_json = run_pose_inference(
        str(preprocessed), output_path=out_dir / f"{stem}_pose.mp4", single=True,
    )

    print("▶  Normalizing and segmenting reps…")
    normalized, normalized_json = normalize_keypoints(pose_json, exercise_id)
    reps, _, rep_count_json = count_repetitions(normalized_json, exercise_id)
    if not reps:
        raise RuntimeError("no reps detected — check the exercise id and that the whole lift is in frame")

    rep_files = _write_rep_files(normalized, reps, out_dir, stem)

    if not keep_intermediates:
        for path in (preprocessed, Path(pose_json), Path(normalized_json), Path(rep_count_json)):
            path.unlink(missing_ok=True)

    print(f"\n✅ {len(rep_files)} reference rep file(s) → {out_dir}")
    for path in rep_files:
        print(f"     {path.name}")
    print(f"   review before keeping them → {pose_video}")
    return rep_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("exercise_id", type=int, help="1 = bench press, 6 = squat, 7 = RDL")
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory")
    parser.add_argument("--name", help="stem for the rep files (default: the video's filename)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="keep the preprocessed mp4 and the whole-video JSONs")
    args = parser.parse_args(argv)

    try:
        extract_reference_reps(args.video, args.exercise_id, args.out, args.name,
                               keep_intermediates=args.keep_intermediates)
    except Exception as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
