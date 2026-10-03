"""Shared feature extraction for ONE video (used by scripts/extract.py and src/pipeline.py).

Everything model-related is wrapped in try/except: a failure sets that modality to N/A
(None / NaN) and appends a warning; it never raises.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import config, media, models
from .faces import blur_score, crop_with_margin


def _detect_scaled(det, frame_bgr, max_side=640):
    h, w = frame_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(frame_bgr, (int(w * s), int(h * s))) if s < 1.0 else frame_bgr
    r = det.detect_largest(small)
    if r is None:
        return None
    x0, y0, x1, y1 = [int(round(v / s)) for v in r["box"]]
    r = dict(r)
    r["box"] = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    return r


def extract_features(path, keep_crops: bool = False, timings: dict | None = None,
                     progress=None, max_seconds: float | None = config.MAX_VIDEO_S) -> dict:
    """Returns dict with keys: meta, windows, crops, audio, warnings (see module docstring)."""
    timings = timings if timings is not None else {}
    warnings: list[str] = []
    prog = progress or (lambda stage, frac: None)

    # ------------------------------------------------------------ container info
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pr = media.probe(path)
    duration = (n_frames / fps) if fps > 0 and n_frames > 0 else (pr["duration"] or 0.0)
    if pr["duration"]:
        duration = min(duration, pr["duration"]) if duration > 0 else pr["duration"]
    if max_seconds and duration > max_seconds:
        warnings.append(f"Video longer than {max_seconds:.0f}s: only the first {max_seconds:.0f}s analysed.")
        duration = max_seconds
    wins = media.make_windows(duration, config.WINDOW_S, config.HOP_S)
    nw = len(wins)
    meta = {"duration": float(duration), "fps": fps, "width": W, "height": H, "n_frames": n_frames,
            "has_audio": bool(pr["has_audio"]), "loudness_db": float("nan"), "vad": media.vad_backend(),
            "face_backend": None}

    # ------------------------------------------------------------ visual
    prog("faces", 0.0)
    t0 = time.time()
    crop_rows: list[dict] = []
    win_nframes = np.zeros(nw, np.int32)
    win_nfaces = np.zeros(nw, np.int32)
    visual_ok = True
    crops_img: list[np.ndarray] = []
    try:
        det = models.get_face_detector()
        meta["face_backend"] = det.backend
        if det.note:
            warnings.append(det.note)
        # target frames: two evenly spaced per window at 1/3 and 2/3 (avoids overlap duplicates)
        targets: dict[int, list[int]] = {}
        for wi, (ws, we) in enumerate(wins):
            for k in range(config.FRAMES_PER_WINDOW):
                t = ws + (we - ws) * (k + 1) / (config.FRAMES_PER_WINDOW + 1)
                fi = int(round(t * fps)) if fps > 0 else 0
                targets.setdefault(fi, []).append(wi)
        last = max(targets) if targets else -1
        cap = cv2.VideoCapture(str(path))
        fi = 0
        while fi <= last:
            ok = cap.grab()
            if not ok:
                break
            if fi in targets:
                ok, frame = cap.retrieve()
                if ok:
                    try:
                        r = _detect_scaled(det, frame)
                    except Exception as e:  # noqa: BLE001
                        r = None
                        if not any("detector error" in w for w in warnings):
                            warnings.append(f"Face detector error ({type(e).__name__}: {e}); affected frames treated as no face.")
                    for wi in targets[fi]:
                        win_nframes[wi] += 1
                        if r is not None:
                            win_nfaces[wi] += 1
                            crop = crop_with_margin(frame, r["box"], config.CROP_MARGIN)
                            if crop.size == 0:
                                win_nfaces[wi] -= 1
                                continue
                            crop_rows.append({
                                "win": wi, "t": fi / fps if fps > 0 else 0.0, "conf": r["conf"],
                                "size": r["size_frac"], "blur": blur_score(crop),
                                "rgb": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                            })
            fi += 1
            if fi % 50 == 0 and last > 0:
                prog("faces", min(1.0, fi / last))
        cap.release()
    except Exception as e:  # noqa: BLE001
        visual_ok = False
        warnings.append(f"Visual stage failed ({type(e).__name__}: {e}); visual analysis N/A.")
    timings["faces"] = time.time() - t0

    t0 = time.time()
    prog("xception", 0.0)
    crop_emb = None
    if visual_ok and crop_rows:
        try:
            crop_emb = models.xception_embed([c["rgb"] for c in crop_rows])
        except Exception as e:  # noqa: BLE001
            visual_ok = False
            warnings.append(f"Xception failed ({type(e).__name__}: {e}); visual analysis N/A.")
    if keep_crops:
        for c in crop_rows:
            crops_img.append(cv2.resize(c["rgb"], (160, 160)))
    timings["xception"] = time.time() - t0

    crops = {
        "ok": bool(visual_ok),
        "emb": crop_emb,
        "win": np.array([c["win"] for c in crop_rows], np.int32),
        "t": np.array([c["t"] for c in crop_rows], np.float32),
        "conf": np.array([c["conf"] for c in crop_rows], np.float32),
        "size": np.array([c["size"] for c in crop_rows], np.float32),
        "blur": np.array([c["blur"] for c in crop_rows], np.float32),
        "img": crops_img,
    }

    # ------------------------------------------------------------ audio
    t0 = time.time()
    prog("audio", 0.0)
    nan = np.full(nw, np.nan, np.float32)
    audio = {"ok": False, "emb": np.full((nw, len(config.W2V_LAYERS), 768), np.nan, np.float32),
             "valid": np.zeros(nw, bool), "speech": nan.copy(), "clip": nan.copy(), "snr": nan.copy(),
             "wave": None}
    if not pr["has_audio"]:
        warnings.append("No audio stream: audio analysis N/A.")
    else:
        try:
            raw = media.decode_audio(path, normalize=False, max_seconds=max_seconds)
            nrm = media.decode_audio(path, normalize=True, max_seconds=max_seconds)
            if raw is None or nrm is None or len(raw) < config.SAMPLE_RATE // 2:
                raise RuntimeError("audio decode returned nothing")
            meta["loudness_db"] = media.loudness_dbfs(raw)
            bounds = media.trim_bounds(nrm)
            if bounds is None:
                warnings.append("Audio is silent: audio analysis N/A.")
            else:
                a0, a1 = bounds
                sr = config.SAMPLE_RATE
                audio["wave"] = nrm
                for wi, (ws, we) in enumerate(wins):
                    s, e = max(ws, a0), min(we, a1)
                    if e - s < 1.0:
                        continue
                    seg_n = nrm[int(s * sr): int(e * sr)]
                    seg_r = raw[int(s * sr): int(e * sr)]
                    audio["speech"][wi] = media.speech_ratio(seg_n)
                    audio["clip"][wi] = media.clipping_fraction(seg_r)
                    audio["snr"][wi] = media.snr_like_db(seg_n)
                    try:
                        audio["emb"][wi] = models.wav2vec2_embed(seg_n)
                        audio["valid"][wi] = True
                    except Exception as ex:  # noqa: BLE001
                        if not any("Wav2Vec2" in w for w in warnings):
                            warnings.append(f"Wav2Vec2 failed ({type(ex).__name__}: {ex}); affected windows N/A.")
                    prog("audio", (wi + 1) / max(nw, 1))
                audio["ok"] = bool(audio["valid"].any())
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Audio stage failed ({type(e).__name__}: {e}); audio analysis N/A.")
    timings["audio"] = time.time() - t0

    windows = {
        "start": np.array([w[0] for w in wins], np.float32),
        "end": np.array([w[1] for w in wins], np.float32),
        "nframes": win_nframes,
        "nfaces": win_nfaces,
    }
    return {"meta": meta, "windows": windows, "crops": crops, "audio": audio, "warnings": warnings}
