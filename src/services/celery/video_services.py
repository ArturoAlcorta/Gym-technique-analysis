import os
import subprocess
from pathlib import Path

import cv2
from tqdm import tqdm

# service to preprocess technique videos
def build_output_path(input_path: str, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    p = Path(input_path)
    return p.parent / f"{p.stem}_preprocessed.mp4"


def compute_output_size(src_w: int, src_h: int, max_side: int) -> tuple[int, int]:
    """
    Devuelve (out_w, out_h) escalando el lado largo a max_side,
    manteniendo el ratio de aspecto. Si max_side == 0, devuelve el tamaño original sin tocar.
    Si la imagen es más pequeña tambien hace el resize hacia arriba
    """
    if max_side == 0:
        return src_w, src_h

    if src_w >= src_h:                         # landscape o cuadrada
        out_w = max_side
        out_h = int(round(src_h * max_side / src_w))
    else:                                      # portrait
        out_h = max_side
        out_w = int(round(src_w * max_side / src_h))

    # Forzar dimensiones pares (necesario para algunos codecs)
    out_w += out_w % 2
    out_h += out_h % 2

    return out_w, out_h


def preprocess_video(
    input_path: str,
    output_path: Path | None = None,
    target_fps: float = 15,
    max_side: int = 1920,
    save_frames: bool = False,
) -> Path:
    if output_path is None:
        output_path = build_output_path(input_path, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Abrir vídeo ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {input_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = compute_output_size(src_w, src_h, max_side)
    do_resize    = (out_w != src_w or out_h != src_h)
    do_fps       = abs(src_fps - target_fps) >= 0.01

    if not do_resize and not do_fps:
        print(f"\n✅ El vídeo ya cumple las condiciones ({src_w}×{src_h}, {src_fps:.1f} FPS). Nada que hacer.")
        cap.release()
        return input_path

    # ── Selección de frames a conservar (remuestreo temporal) ────────────────
    if do_fps:
        step = src_fps / target_fps
        selected_indices = []
        t = 0.0
        while t < src_total:
            selected_indices.append(round(t))
            t += step
    else:
        selected_indices = list(range(src_total))

    selected_set = set(selected_indices)
    n_out        = len(selected_indices)
    fps_out      = target_fps if do_fps else src_fps

    # ── Escritor de vídeo ─────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el vídeo de salida: {output_path}")

    # ── Carpeta de frames PNG (opcional) ─────────────────────────────────────
    frames_dir: Path | None = None
    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Resumen ───────────────────────────────────────────────────────────────
    resize_str = f"{out_w} × {out_h} px" if do_resize else f"{src_w} × {src_h} px  (sin cambios)"
    sep = "─" * 56
    print(f"\n{'⚙️  PREPROCESADO':^56}")
    print(sep)
    print(f"  📁 Entrada        : {os.path.basename(input_path)}")
    print(f"  📐 Res. original  : {src_w} × {src_h} px")
    print(f"  🎞️  FPS original   : {src_fps:.3f}")
    print(f"  🔢 Frames orig.   : {src_total:,}")
    print(f"  {'─'*50}")
    print(f"  📐 Res. salida    : {resize_str}")
    print(f"  🎞️  FPS salida     : {fps_out:.1f}")
    print(f"  🔢 Frames salida  : {n_out:,}")
    print(f"  💾 Salida         : {output_path}")
    if save_frames:
        print(f"  🖼️  Frames PNG     : {frames_dir}")
    print(sep + "\n")

    # ── Bucle principal ───────────────────────────────────────────────────────
    frame_idx = 0
    out_idx   = 0

    interp = cv2.INTER_AREA if (out_w * out_h < src_w * src_h) else cv2.INTER_CUBIC

    with tqdm(total=n_out, unit="frame", desc="Procesando") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in selected_set:
                if do_resize:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=interp)

                writer.write(frame)

                if frames_dir is not None:
                    cv2.imwrite(str(frames_dir / f"frame_{out_idx:05d}.png"), frame)

                out_idx += 1
                pbar.update(1)

            frame_idx += 1

    cap.release()
    writer.release()

    # ── Estimación del tamaño del archivo ────────────────────────────────────
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Listo. {out_idx} frames escritos → {output_path}  ({size_mb:.1f} MB)\n")

    return output_path

def transcode_h264(path: Path, crf: int = 23, preset: str = "veryfast") -> Path:
    """
    Re-encode a clip to H.264 in place, for playback in a browser.

    OpenCV's `mp4v` fourcc writes MPEG-4 Part 2, which no browser decodes: a
    `<video>` element fetches the whole file and then renders nothing at all.
    H.264 with `yuv420p` is the format every browser accepts, and `+faststart`
    moves the index to the front so playback can begin before the download
    finishes. It also shrinks the annotated output by roughly an order of
    magnitude, since the OpenCV writer is not doing any real rate control.

    Returns the path either way: a transcode failure leaves the original file
    untouched rather than losing the analysis it belongs to.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.stem}_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = exc.stderr.decode(errors="replace")[-500:] if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f"⚠️  H.264 transcode failed, leaving {path.name} as MPEG-4 Part 2: {detail}")
        tmp.unlink(missing_ok=True)
        return path

    before, after = path.stat().st_size, tmp.stat().st_size
    tmp.replace(path)
    print(f"Transcoded {path.name} to H.264 ({before / 1e6:.0f} MB → {after / 1e6:.0f} MB)")
    return path
