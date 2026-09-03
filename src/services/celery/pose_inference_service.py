import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO

POSE_MODEL_ID = "stanfordmimi/synthpose-vitpose-huge-hf"
DEFAULT_YOLO  = os.path.join(os.getenv("YOLO_MODELS_DIR", "/app/models"), "yolo11m.pt")

KEYPOINT_NAMES = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
    13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle",
    17: "sternum", 18: "rshoulder", 19: "lshoulder",
    20: "r_lelbow", 21: "l_lelbow", 22: "r_melbow", 23: "l_melbow",
    24: "r_lwrist", 25: "l_lwrist", 26: "r_mwrist", 27: "l_mwrist",
    28: "r_ASIS", 29: "l_ASIS", 30: "r_PSIS", 31: "l_PSIS",
    32: "r_knee", 33: "l_knee", 34: "r_mknee", 35: "l_mknee",
    36: "r_ankle", 37: "l_ankle", 38: "r_mankle", 39: "l_mankle",
    40: "r_5meta", 41: "l_5meta", 42: "r_toe", 43: "l_toe",
    44: "r_big_toe", 45: "l_big_toe", 46: "l_calc", 47: "r_calc",
    48: "C7", 49: "L2", 50: "T11", 51: "T6",
}

PALETTE = np.array([
    [0, 128, 255],   [51, 153, 255],  [102, 178, 255], [0, 230, 230],
    [255, 153, 255], [255, 204, 153], [255, 102, 255], [255, 51, 255],
    [255, 178, 102], [255, 153, 51],  [153, 153, 255], [102, 102, 255],
    [51, 51, 255],   [153, 255, 153], [102, 255, 102], [51, 255, 51],
    [0, 255, 0],     [255, 0, 0],     [0, 0, 255],     [255, 255, 255],
], dtype=np.uint8)

KP_COLORS   = PALETTE[[16,16,16,16,16, 9,9,9,9,9,9, 0,0,0,0,0,0] + [4]*35]
LINK_COLORS = PALETTE[[0,0,0,0, 7,7,7, 9,9,9,9,9, 16,16,16,16,16,16,16]]


# ── Tracking ──────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id:     int
    last_bbox:    np.ndarray
    frame_count:  int  = 0
    missed:       int  = 0
    center_dists: list = field(default_factory=list)
    bbox_areas:   list = field(default_factory=list)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed    = max_missed
        self._next_id      = 0
        self.tracks: dict[int, Track] = {}

    def _iou(self, a: np.ndarray, b: np.ndarray) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter)

    def update(self, detections: np.ndarray, frame_w: int, frame_h: int) -> list[int]:
        cx, cy = frame_w / 2.0, frame_h / 2.0
        for t in self.tracks.values():
            t.missed += 1

        assigned_ids = [-1] * len(detections)

        if len(detections) > 0 and self.tracks:
            track_ids  = list(self.tracks.keys())
            iou_matrix = np.zeros((len(detections), len(track_ids)))
            for di, det in enumerate(detections):
                for ti, tid in enumerate(track_ids):
                    iou_matrix[di, ti] = self._iou(det, self.tracks[tid].last_bbox)

            used_tracks = set()
            order = np.dstack(np.unravel_index(
                np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape
            ))[0]
            for di, ti in order:
                if iou_matrix[di, ti] < self.iou_threshold:
                    break
                tid = track_ids[ti]
                if assigned_ids[di] != -1 or tid in used_tracks:
                    continue
                assigned_ids[di] = tid
                used_tracks.add(tid)

        for di, det in enumerate(detections):
            area = (det[2]-det[0]) * (det[3]-det[1])
            bcx  = (det[0]+det[2]) / 2
            bcy  = (det[1]+det[3]) / 2
            dist = math.hypot(bcx - cx, bcy - cy) / math.hypot(cx, cy)

            if assigned_ids[di] == -1:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(track_id=tid, last_bbox=det.copy())
                assigned_ids[di] = tid
            else:
                tid = assigned_ids[di]
                self.tracks[tid].last_bbox = det.copy()
                self.tracks[tid].missed    = 0

            self.tracks[tid].frame_count += 1
            self.tracks[tid].center_dists.append(dist)
            self.tracks[tid].bbox_areas.append(float(area))

        dead = [tid for tid, t in self.tracks.items() if t.missed > self.max_missed]
        for tid in dead:
            del self.tracks[tid]

        return assigned_ids

    def best_track_id(self, frame_w: int, frame_h: int) -> Optional[int]:
        """Track con mayor mean_area — usado para highlight en vídeo anotado."""
        if not self.tracks:
            return None
        frame_area = frame_w * frame_h
        best_id, best_area = None, -1.0
        for tid, t in self.tracks.items():
            if t.frame_count == 0:
                continue
            mean_area = np.mean(t.bbox_areas) / frame_area
            if mean_area > best_area:
                best_area = mean_area
                best_id   = tid
        return best_id

    def principal_track_ids(self, frame_w: int, frame_h: int, area_ratio: float = 0.6) -> set[int]:
        """
        Devuelve todos los track IDs que pertenecen al sujeto principal.

        El sujeto es la persona más grande en pantalla (más cerca de la cámara).
        Se incluyen todos los tracks cuya área media sea >= area_ratio * max_area,
        capturando así los fragmentos del mismo atleta con IDs distintos.
        """
        if not self.tracks:
            return set()
        frame_area = frame_w * frame_h
        track_areas = {
            tid: np.mean(t.bbox_areas) / frame_area
            for tid, t in self.tracks.items()
            if t.frame_count > 0
        }
        if not track_areas:
            return set()
        max_area = max(track_areas.values())
        return {tid for tid, area in track_areas.items() if area >= max_area * area_ratio}


# ── Dibujo ────────────────────────────────────────────────────────────────────

def _draw_keypoints(image, keypoints, scores, threshold, radius=8):
    for kid, (kpt, score) in enumerate(zip(keypoints, scores)):
        if score < threshold:
            continue
        x, y  = int(kpt[0]), int(kpt[1])
        color = tuple(int(c) for c in KP_COLORS[kid])
        cv2.circle(image, (x, y), radius,     color,     -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius + 2, (0, 0, 0),  2, cv2.LINE_AA)


def _draw_links(image, keypoints, scores, edges, threshold, thickness=5):
    h, w = image.shape[:2]
    for sk_id, (a, b) in enumerate(edges):
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        x1, y1, s1 = int(keypoints[a, 0]), int(keypoints[a, 1]), scores[a]
        x2, y2, s2 = int(keypoints[b, 0]), int(keypoints[b, 1]), scores[b]
        if s1 < threshold or s2 < threshold:
            continue
        if not (0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h):
            continue
        conf   = float((s1 + s2) / 2)
        t      = max(1, int(thickness * conf))
        color  = tuple(int(c) for c in LINK_COLORS[sk_id % len(LINK_COLORS)])
        mean_x = (x1 + x2) / 2
        mean_y = (y1 + y2) / 2
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            continue
        angle  = math.degrees(math.atan2(y1 - y2, x1 - x2))
        poly   = cv2.ellipse2Poly(
            (int(mean_x), int(mean_y)),
            (max(1, int(length / 2)), max(4, t)),
            int(angle), 0, 360, 1,
        )
        cv2.fillConvexPoly(image, poly, color, cv2.LINE_AA)


def _annotate_frame(frame, pose_results, edges, threshold, highlight_idx=None):
    annotated = frame.copy()
    for p_idx, person in enumerate(pose_results):
        if highlight_idx is not None and p_idx != highlight_idx:
            continue
        kpts   = person["keypoints"].cpu().numpy() if hasattr(person["keypoints"], "cpu") else np.array(person["keypoints"])
        scores = person["scores"].cpu().numpy()    if hasattr(person["scores"],    "cpu") else np.array(person["scores"])
        _draw_links(annotated, kpts, scores, edges, threshold)
        _draw_keypoints(annotated, kpts, scores, threshold)
    return annotated


def _pose_to_dict(frame_idx, fps, pose_results, person_boxes_coco, track_ids, single_track_id):
    persons = []
    for p_idx, person in enumerate(pose_results):
        tid = track_ids[p_idx] if p_idx < len(track_ids) else -1
        if single_track_id is not None and tid != single_track_id:
            continue
        kpts   = person["keypoints"].cpu().numpy() if hasattr(person["keypoints"], "cpu") else np.array(person["keypoints"])
        scores = person["scores"].cpu().numpy()    if hasattr(person["scores"],    "cpu") else np.array(person["scores"])
        bbox   = person_boxes_coco[p_idx].tolist() if p_idx < len(person_boxes_coco) else []
        persons.append({
            "person_idx": p_idx,
            "track_id":   tid,
            "bbox":       [round(float(v), 2) for v in bbox],
            "keypoints": [
                {
                    "id":    int(kid),
                    "name":  KEYPOINT_NAMES.get(kid, f"kp_{kid}"),
                    "x":     round(float(kpts[kid, 0]), 2),
                    "y":     round(float(kpts[kid, 1]), 2),
                    "score": round(float(scores[kid]),  4),
                }
                for kid in range(len(kpts))
            ],
        })
    return {
        "frame_idx":   frame_idx,
        "timestamp_s": round(frame_idx / fps, 4),
        "persons":     persons,
    }


def _detect_persons_yolo(yolo_model, frames_bgr: list, det_threshold: float) -> list:
    results = yolo_model(frames_bgr, verbose=False)
    all_boxes = []
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            all_boxes.append(np.zeros((0, 4), dtype=np.float32))
            continue
        mask = (boxes.cls.cpu().numpy() == 0) & (boxes.conf.cpu().numpy() >= det_threshold)
        xyxy = boxes.xyxy.cpu().numpy()[mask]
        all_boxes.append(xyxy if len(xyxy) > 0 else np.zeros((0, 4), dtype=np.float32))
    return all_boxes


def _select_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_output_paths(input_path: str, output_path: Path | None) -> tuple[Path, Path]:
    if output_path:
        video_out = output_path
    else:
        p = Path(input_path)
        video_out = p.parent / f"{p.stem}_pose.mp4"
    return video_out, video_out.with_suffix(".json")


def _principal_ids_from_frames(
    all_frames_data: list[dict],
    frame_area: float,
    area_ratio: float = 0.6,
    min_frames: int = 30,
) -> set[int]:
    """
    Calcula qué track IDs pertenecen al sujeto principal a partir de los frames
    ya recogidos (no desde tracker.tracks, que solo conserva tracks vivos).

    El sujeto es la persona con mayor área media de bbox (más cerca de la cámara)
    entre los tracks con duración >= min_frames (filtra detecciones fugaces de
    personas que pasan cerca de la cámara brevemente).
    Se incluyen todos los tracks que superen el umbral con área >= area_ratio * max_area.
    """
    area_sum: dict[int, float] = {}
    count:    dict[int, int]   = {}
    for frame_data in all_frames_data:
        for person in frame_data["persons"]:
            tid  = person["track_id"]
            bbox = person["bbox"]          # [x, y, w, h] formato COCO
            area = (bbox[2] * bbox[3]) / frame_area
            area_sum[tid] = area_sum.get(tid, 0.0) + area
            count[tid]    = count.get(tid, 0) + 1
    if not count:
        return set()
    # Solo candidatos con duración mínima; si ninguno la supera, usar todos
    candidates = {tid: area_sum[tid] / count[tid] for tid in count if count[tid] >= min_frames}
    if not candidates:
        candidates = {tid: area_sum[tid] / count[tid] for tid in count}
    max_area = max(candidates.values())
    return {tid for tid, area in candidates.items() if area >= max_area * area_ratio}


# ── Servicio principal ────────────────────────────────────────────────────────

def run_pose_inference(
    input_path: str,
    output_path: Path | None = None,
    threshold: float = 0.3,
    det_threshold: float = 0.4,
    device: str | None = None,
    single: bool = False,
    iou_threshold: float = 0.3,
    batch_size: int = 8,
    yolo_model_name: str = DEFAULT_YOLO,
) -> tuple[Path, Path]:
    """
    Ejecuta detección de personas (YOLO) + estimación de pose (SynthPose) sobre un vídeo.
    Devuelve (video_anotado_path, keypoints_json_path).
    """
    device = _select_device(device)
    video_out, json_out = _build_output_paths(input_path, output_path)
    video_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧 Dispositivo: {device.upper()}")
    print(f"📥 Cargando detector YOLO ({yolo_model_name})…")
    yolo = YOLO(yolo_model_name)
    yolo.to(device)

    print("📥 Cargando modelo de pose (SynthPose VitPose-Huge)…")
    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL_ID, backend="torchvision")
    pose_model     = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID, device_map=device)
    pose_model.eval()

    edges = pose_model.config.edges
    print(f"   └─ {len(edges)} enlaces de esqueleto cargados")

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {input_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer  = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        tracker = IoUTracker(iou_threshold=iou_threshold)

        sep = "─" * 57
        print(f"\n{'🎬 INFERENCIA DE POSE':^57}")
        print(sep)
        print(f"  📁 Entrada      : {os.path.basename(input_path)}")
        print(f"  📐 Resolución   : {width} × {height} px  |  {fps:.1f} FPS")
        print(f"  🔢 Frames       : {total_frames:,}")
        print(f"  🎯 Threshold kp : {threshold}  |  det: {det_threshold}")
        print(f"  📦 Batch size   : {batch_size}")
        print(f"  👤 Modo single  : {'SÍ' if single else 'NO'}")
        print(f"  💾 Vídeo        : {video_out}")
        print(f"  📄 JSON         : {json_out}")
        print(sep + "\n")

        all_frames_data    = []
        frames_sin_persona = 0

        def process_batch(batch_bgr: list, batch_idxs: list):
            nonlocal frames_sin_persona, yolo, pose_processor, pose_model

            all_boxes = _detect_persons_yolo(yolo, batch_bgr, det_threshold)

            frame_track_ids  = []
            frame_boxes_coco = []
            fi_to_crops: dict = {}

            for fi, (frame_bgr_i, boxes_voc) in enumerate(zip(batch_bgr, all_boxes)):
                if len(boxes_voc) == 0:
                    frame_track_ids.append([])
                    frame_boxes_coco.append(np.zeros((0, 4), dtype=np.float32))
                    continue

                tids = tracker.update(boxes_voc, width, height)
                frame_track_ids.append(tids)

                boxes_coco = boxes_voc.copy()
                boxes_coco[:, 2] -= boxes_coco[:, 0]
                boxes_coco[:, 3] -= boxes_coco[:, 1]
                frame_boxes_coco.append(boxes_coco)

                fi_to_crops[fi] = (tids, boxes_coco)

            pose_results_by_fi: dict = {}
            for fi, (_, boxes_coco_i) in fi_to_crops.items():
                pil_i   = Image.fromarray(cv2.cvtColor(batch_bgr[fi], cv2.COLOR_BGR2RGB))
                pose_in = pose_processor(pil_i, boxes=[boxes_coco_i], return_tensors="pt").to(device)
                with torch.inference_mode():
                    pose_out = pose_model(**pose_in)
                pose_results_by_fi[fi] = pose_processor.post_process_pose_estimation(
                    pose_out, boxes=[boxes_coco_i], threshold=None
                )[0]

            for fi, frame_bgr_i in enumerate(batch_bgr):
                global_idx = batch_idxs[fi]
                boxes_coco = frame_boxes_coco[fi] if fi < len(frame_boxes_coco) else np.zeros((0, 4))
                tids       = frame_track_ids[fi]  if fi < len(frame_track_ids)  else []

                if len(boxes_coco) == 0:
                    frames_sin_persona += 1
                    writer.write(frame_bgr_i)
                    all_frames_data.append({
                        "frame_idx":   global_idx,
                        "timestamp_s": round(global_idx / fps, 4),
                        "persons":     [],
                    })
                    continue

                frame_pose_results = pose_results_by_fi.get(fi, [])

                highlight_idx = None
                if single:
                    hint_id = tracker.best_track_id(width, height)
                    if hint_id is not None and hint_id in tids:
                        highlight_idx = tids.index(hint_id)

                annotated = _annotate_frame(frame_bgr_i, frame_pose_results, edges, threshold, highlight_idx)
                writer.write(annotated)
                # Collect all tracks; principal filtering happens after the full video
                all_frames_data.append(
                    _pose_to_dict(global_idx, fps, frame_pose_results, boxes_coco, tids, None)
                )

        with tqdm(total=total_frames, unit="frame", desc="Procesando") as pbar:
            frame_idx  = 0
            batch_bgr  = []
            batch_idxs = []

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                batch_bgr.append(frame_bgr)
                batch_idxs.append(frame_idx)

                if len(batch_bgr) == batch_size:
                    process_batch(batch_bgr, batch_idxs)
                    pbar.update(len(batch_bgr))
                    batch_bgr  = []
                    batch_idxs = []

                frame_idx += 1

            if batch_bgr:
                process_batch(batch_bgr, batch_idxs)
                pbar.update(len(batch_bgr))

        cap.release()
        writer.release()

        best_id = tracker.best_track_id(width, height)

        # Compute principal IDs from frame data (tracker.tracks only has alive tracks at end)
        if single:
            principal_ids = _principal_ids_from_frames(all_frames_data, width * height)
        else:
            principal_ids = None

        # Build per-track summary from frame data for display
        track_summary: dict[int, dict] = {}
        for frame_data in all_frames_data:
            for person in frame_data["persons"]:
                tid  = person["track_id"]
                bbox = person["bbox"]
                area = (bbox[2] * bbox[3]) / (width * height)
                if tid not in track_summary:
                    track_summary[tid] = {"frames": 0, "area_sum": 0.0}
                track_summary[tid]["frames"]   += 1
                track_summary[tid]["area_sum"] += area

        print(f"\n📊 Tracks detectados: {tracker._next_id}")
        for tid, info in sorted(track_summary.items(), key=lambda x: -x[1]["area_sum"] / x[1]["frames"]):
            area_pct = 100 * info["area_sum"] / info["frames"]
            if single:
                marker = " ← PRINCIPAL" if tid in principal_ids else ""
            else:
                marker = " ← PRINCIPAL" if tid == best_id else ""
            print(f"   Track {tid:3d} | {info['frames']:4d} frames | área {area_pct:4.1f}%{marker}")

        # Filter by principal tracks if single mode
        if single:
            filtered_data = []
            for fd in all_frames_data:
                filtered_persons = [p for p in fd["persons"] if p["track_id"] in principal_ids]
                if filtered_persons:
                    fd["persons"] = filtered_persons
                    filtered_data.append(fd)
            all_frames_data = filtered_data

        # Save JSON
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "video": os.path.basename(input_path),
                    "fps": fps,
                    "total_frames": total_frames,
                    "main_track_id": next(iter(principal_ids)) if principal_ids else None,
                },
                "frames": all_frames_data,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Guardado: {len(all_frames_data)} frames → {json_out}")

        return video_out, json_out

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise