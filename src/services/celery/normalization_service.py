"""
Normalización de keypoints SynthPose por ejercicio.

Estrategia general (dos pasadas):
  1. Primera pasada: recoger ref_point y scale de todos los frames válidos
     y calcular la MEDIANA como valores fijos para el vídeo completo.
  2. Segunda pasada: normalizar todos los frames con esos valores fijos.

Por qué escala fija (no por frame):
  - En sentadilla/RDL el hombro puede quedar parcialmente oculto en el fondo
    del movimiento, produciendo escalas artificialmente bajas en esos frames.
  - En press banca el ancho de hombros aparente varía ~25% por perspectiva
    al bajar y subir la barra.
  - Con escala y ref_point fijos las coordenadas reflejan solo el movimiento
    real, sin ruido de detección, lo que mejora la señal del DTW y del
    conteo de repeticiones.

Para añadir un nuevo ejercicio:
  1. Subclasifica ExerciseNormalizer e implementa get_reference_and_scale
     y _normalization_meta.
  2. Añade una entrada en EXERCISE_NORMALIZERS con el exercise_id.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

SCORE_THRESHOLD = 0.5


# ── Shared utilities ──────────────────────────────────────────────────────────

def _get_keypoint(keypoints: list[dict], name: str) -> dict | None:
    for kp in keypoints:
        if kp["name"] == name and kp["score"] >= SCORE_THRESHOLD:
            return kp
    return None


def _best_of(keypoints: list[dict], *names: str) -> dict | None:
    candidates = [kp for kp in keypoints if kp["name"] in names and kp["score"] >= SCORE_THRESHOLD]
    if not candidates:
        return None
    return max(candidates, key=lambda kp: kp["score"])


# ── Base class ────────────────────────────────────────────────────────────────

class ExerciseNormalizer(ABC):

    @property
    @abstractmethod
    def exercise_name(self) -> str: ...

    @property
    @abstractmethod
    def plane(self) -> str: ...

    @property
    @abstractmethod
    def relevant_joints(self) -> list[str]: ...

    @abstractmethod
    def get_reference_and_scale(
        self, keypoints: list[dict]
    ) -> tuple[np.ndarray, float, dict] | tuple[None, None, None]:
        """
        Devuelve (ref_point, scale, debug_info) o (None, None, None) si el frame
        no tiene suficientes keypoints para calcular la referencia.
        """
        ...

    @abstractmethod
    def _normalization_meta(self) -> dict: ...

    def normalize_frame(
        self, keypoints: list[dict], ref_point: np.ndarray, scale: float
    ) -> dict:
        normalized = {}
        for name in self.relevant_joints:
            kp = _get_keypoint(keypoints, name)
            if kp is None:
                normalized[name] = None
            else:
                rel = (np.array([kp["x"], kp["y"]]) - ref_point) / scale
                normalized[name] = {
                    "x":     round(float(rel[0]), 6),
                    "y":     round(float(rel[1]), 6),
                    "score": round(kp["score"], 4),
                }
        return normalized

    def normalize_video(self, data: dict) -> dict:
        """
        Normalización en dos pasadas con ref_point y scale fijos (mediana del vídeo).
        """
        main_track_id = data["metadata"].get("main_track_id")

        def get_person(frame):
            persons = frame.get("persons", [])
            if not persons:
                return None
            if main_track_id is not None:
                p = next((p for p in persons if p["track_id"] == main_track_id), None)
                return p if p else persons[0]
            return persons[0]

        # ── Primera pasada: recoger ref y scale de frames válidos ──
        raw_ref_x, raw_ref_y, raw_scales = [], [], []
        for frame in data["frames"]:
            person = get_person(frame)
            if person is None:
                continue
            ref_point, scale, _ = self.get_reference_and_scale(person["keypoints"])
            if ref_point is not None:
                raw_ref_x.append(ref_point[0])
                raw_ref_y.append(ref_point[1])
                raw_scales.append(scale)

        if not raw_scales:
            raise ValueError("No se encontraron frames válidos para calcular la normalización.")

        fixed_ref   = np.array([np.median(raw_ref_x), np.median(raw_ref_y)])
        fixed_scale = float(np.median(raw_scales))
        scale_var   = float(np.max(raw_scales) / np.min(raw_scales)) if len(raw_scales) > 1 else 1.0

        print(f"Escala — mean: {np.mean(raw_scales):.1f}px  "
              f"min: {np.min(raw_scales):.1f}px  max: {np.max(raw_scales):.1f}px  "
              f"(variación {scale_var:.2f}x)")
        if scale_var > 1.15:
            print(f"  ⚠️  Variación de escala {scale_var:.2f}x — usando mediana fija ({fixed_scale:.1f}px)")
        print(f"Reference point fijo: ({fixed_ref[0]:.1f}, {fixed_ref[1]:.1f})px")

        # ── Segunda pasada: normalizar con valores fijos ──
        results = []
        skipped = 0
        for frame in data["frames"]:
            person = get_person(frame)
            if person is None:
                skipped += 1
                continue
            normalized = self.normalize_frame(person["keypoints"], fixed_ref, fixed_scale)
            results.append({
                "frame_idx":       frame["frame_idx"],
                "timestamp_s":     frame["timestamp_s"],
                "reference_point": {
                    "x": round(float(fixed_ref[0]), 2),
                    "y": round(float(fixed_ref[1]), 2),
                },
                "scale":           round(fixed_scale, 4),
                "keypoints":       normalized,
            })

        print(f"Frames procesados: {len(results)} | Saltados: {skipped}")

        return {
            "metadata": {
                "source_video":  data["metadata"]["video"],
                "fps":           data["metadata"]["fps"],
                "exercise":      self.exercise_name,
                "plane":         self.plane,
                "normalization": {
                    **self._normalization_meta(),
                    "fixed_ref_x":     round(float(fixed_ref[0]), 2),
                    "fixed_ref_y":     round(float(fixed_ref[1]), 2),
                    "fixed_scale_px":  round(fixed_scale, 4),
                    "scale_variation": round(scale_var, 3),
                    "score_threshold": SCORE_THRESHOLD,
                },
                "total_frames":  len(results),
            },
            "frames": results,
        }


# ── Sentadilla y RDL (plano sagital) ─────────────────────────────────────────

class SagittalNormalizer(ExerciseNormalizer):
    """
    Referencia: cadera dominante (mayor score entre L_Hip/l_ASIS y R_Hip/r_ASIS).
    Escala:     distancia cadera → rodilla del MISMO lado (longitud de fémur).
    Normalización fija: mediana de ref y scale del vídeo completo.

    Se usa fémur en lugar de torso porque:
    - La rodilla no queda oculta por los discos (a diferencia del hombro).
    - El fémur determina directamente la mecánica de la sentadilla.
    - Reduce la scale_variation cuando el hombro se ocluye por equipamiento.
    """

    exercise_name = "sagittal"
    plane         = "sagital"
    relevant_joints = [
        "L_Shoulder", "R_Shoulder",
        "L_Hip",      "R_Hip",
        "L_Knee",     "R_Knee",
        "L_Ankle",    "R_Ankle",
        "sternum",
        "r_ASIS",     "l_ASIS",
        "r_PSIS",     "l_PSIS",
        "r_knee",     "l_knee",
        "r_ankle",    "l_ankle",
        "r_calc",     "l_calc",
        "r_big_toe",  "l_big_toe",
        "C7", "T6", "T11", "L2",
    ]

    _LEFT_HIP   = ["L_Hip", "l_ASIS"]
    _RIGHT_HIP  = ["R_Hip", "r_ASIS"]
    _LEFT_KNEE  = ["L_Knee", "l_knee"]
    _RIGHT_KNEE = ["R_Knee", "r_knee"]

    def get_reference_and_scale(self, keypoints):
        l_hip = _best_of(keypoints, *self._LEFT_HIP)
        r_hip = _best_of(keypoints, *self._RIGHT_HIP)

        if l_hip is None and r_hip is None:
            return None, None, None
        elif l_hip is None:
            dominant_hip, side = r_hip, "right"
        elif r_hip is None:
            dominant_hip, side = l_hip, "left"
        else:
            if l_hip["score"] >= r_hip["score"]:
                dominant_hip, side = l_hip, "left"
            else:
                dominant_hip, side = r_hip, "right"

        knee = _best_of(keypoints, *(self._LEFT_KNEE if side == "left" else self._RIGHT_KNEE))
        if knee is None:
            return None, None, None

        ref_point    = np.array([dominant_hip["x"], dominant_hip["y"]])
        femur_length = np.linalg.norm(np.array([knee["x"], knee["y"]]) - ref_point)
        if femur_length < 1e-6:
            return None, None, None

        return ref_point, femur_length, {"dominant_side": side}

    def _normalization_meta(self):
        return {
            "reference_point": "dominant_hip_median",
            "scale":           "same_side_femur_length_median",
        }


# ── Press banca (plano diagonal) ─────────────────────────────────────────────

class BenchPressNormalizer(ExerciseNormalizer):
    """
    Referencia: punto medio entre ambos hombros (mediana del vídeo).
    Escala:     ancho de hombros (mediana del vídeo).
    """

    exercise_name = "bench_press"
    plane         = "diagonal"
    relevant_joints = [
        "L_Shoulder", "R_Shoulder",
        "L_Elbow",    "R_Elbow",
        "L_Wrist",    "R_Wrist",
        "lshoulder",  "rshoulder",
        "l_lelbow",   "r_lelbow",
        "l_melbow",   "r_melbow",
        "l_lwrist",   "r_lwrist",
        "l_mwrist",   "r_mwrist",
        "sternum",
        "L_Hip", "R_Hip",
        "C7", "T6", "T11", "L2",
    ]

    _LEFT_SHOULDER  = ["L_Shoulder", "lshoulder"]
    _RIGHT_SHOULDER = ["R_Shoulder", "rshoulder"]

    def get_reference_and_scale(self, keypoints):
        l_sh = _best_of(keypoints, *self._LEFT_SHOULDER)
        r_sh = _best_of(keypoints, *self._RIGHT_SHOULDER)
        if l_sh is None or r_sh is None:
            return None, None, None

        l_pt = np.array([l_sh["x"], l_sh["y"]])
        r_pt = np.array([r_sh["x"], r_sh["y"]])
        ref_point      = (l_pt + r_pt) / 2.0
        shoulder_width = np.linalg.norm(l_pt - r_pt)
        if shoulder_width < 1e-6:
            return None, None, None

        debug_info = {
            "l_shoulder_joint": l_sh["name"], "l_shoulder_score": round(l_sh["score"], 4),
            "r_shoulder_joint": r_sh["name"], "r_shoulder_score": round(r_sh["score"], 4),
        }
        return ref_point, shoulder_width, debug_info

    def _normalization_meta(self):
        return {
            "reference_point": "shoulder_midpoint_median",
            "scale":           "shoulder_width_median",
        }


# ── Registry ──────────────────────────────────────────────────────────────────

_sagittal    = SagittalNormalizer()
_bench_press = BenchPressNormalizer()

EXERCISE_NORMALIZERS: dict[int, ExerciseNormalizer] = {
    6:  _sagittal,     # sentadilla
    7:  _sagittal,     # RDL
    1:  _bench_press,  # press banca
    2:  _bench_press,  # press banca (alias)
    48: _sagittal,     # RDL (alias)
}


# ── Public function ───────────────────────────────────────────────────────────

def normalize_keypoints(json_path: Path, exercise_id: int) -> tuple[dict, Path]:
    """
    Normaliza los keypoints de un JSON de SynthPose según el ejercicio.
    Devuelve (resultado_dict, output_path).
    """
    normalizer = EXERCISE_NORMALIZERS.get(exercise_id)
    if normalizer is None:
        raise ValueError(
            f"exercise_id={exercise_id} no tiene normalizador registrado. "
            f"IDs disponibles: {list(EXERCISE_NORMALIZERS)}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = normalizer.normalize_video(data)

    output_path = json_path.parent / f"{json_path.stem}_normalized.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Normalización guardada → {output_path}")
    return result, output_path