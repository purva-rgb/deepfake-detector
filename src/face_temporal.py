"""Facial temporal features from MediaPipe Face Landmarker (478-point mesh) at ~12 fps.

NOTE: the original pipeline only used the BlazeFace *detector* (no landmarks). Landmarks (needed for MAR / EAR /
head pose) come from the SPEC-named MediaPipe Face Landmarker, a small .task asset downloaded once to models/.
The existing detector, crops and visual reliability are untouched. If the landmarker cannot load, every
temporal feature is N/A with a reason and the pipeline continues.

Definitions (MediaPipe mesh indices):
  MAR = (|82-87| + |13-14| + |312-317|) / (3 * |78-308|)     inner-lip vertical gaps / mouth width
  EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|) per eye, mean of both eyes
        right (33,160,158,133,153,144), left (362,385,387,263,373,380)
  head pose = yaw/pitch/roll (deg) from the landmarker's facial transformation matrix
  blink = a transition from open to closed where EAR < BLINK_FRAC * median EAR of the video
Statistics use valid frames only; a window needs MIN_VALID valid frames or it is N/A.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import MODELS_DIR

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
                  "float16/1/face_landmarker.task")
LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
TARGET_FPS = 12.0
MAX_SIDE = 640
MIN_VALID = 8
BLINK_FRAC = 0.75

MOUTH_V = ((82, 87), (13, 14), (312, 317))
MOUTH_H = (78, 308)
EYE_R = (33, 160, 158, 133, 153, 144)
EYE_L = (362, 385, 387, 263, 373, 380)

STAT_NAMES = ["mar_mean", "mar_std", "mar_range", "mar_velocity", "ear_mean", "ear_std", "blink_count",
              "yaw_mean", "yaw_std", "pitch_mean", "pitch_std", "roll_mean", "roll_std"]

_CACHE: dict = {}


# ------------------------------------------------------------------ pure geometry (unit-tested)
def _d(p, a, b):
    return float(np.linalg.norm(p[a, :2] - p[b, :2]))


def mar(pts: np.ndarray) -> float:
    """pts: (N,2+) landmark array in PIXEL units. Returns NaN if the mouth is degenerate."""
    w = _d(pts, *MOUTH_H)
    if w < 1e-6:
        return float("nan")
    return float(sum(_d(pts, a, b) for a, b in MOUTH_V) / (3.0 * w))


def _ear_one(pts, idx):
    p1, p2, p3, p4, p5, p6 = idx
    w = _d(pts, p1, p4)
    if w < 1e-6:
        return float("nan")
    return float((_d(pts, p2, p6) + _d(pts, p3, p5)) / (2.0 * w))


def ear(pts: np.ndarray) -> float:
    v = [_ear_one(pts, EYE_R), _ear_one(pts, EYE_L)]
    return float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")


def euler_from_matrix(M: np.ndarray):
    """(yaw, pitch, roll) in degrees from a 4x4 (or 3x3) transformation matrix."""
    R = np.asarray(M, dtype=np.float64)[:3, :3]
    sy = float(np.hypot(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(yaw), float(pitch), float(roll)


def blink_count(ear_vals: np.ndarray, valid: np.ndarray, thr: float) -> int:
    """Number of open->closed transitions (EAR < thr) over consecutive valid frames."""
    n, closed_prev = 0, False
    for e, v in zip(ear_vals, valid):
        if not v or not np.isfinite(e):
            closed_prev = False
            continue
        closed = e < thr
        if closed and not closed_prev:
            n += 1
        closed_prev = closed
    return n


# ------------------------------------------------------------------ series container + window stats
@dataclass
class FaceSeries:
    t: np.ndarray
    valid: np.ndarray
    mar: np.ndarray
    ear: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    fps: float = TARGET_FPS
    ear_median: float = float("nan")
    n_multi: int = 0                      # frames where more than one face was found
    notes: list = field(default_factory=list)


def series_from_arrays(t, mar_, ear_, yaw, pitch, roll, valid=None, fps=TARGET_FPS) -> FaceSeries:
    t = np.asarray(t, float)
    mar_, ear_ = np.asarray(mar_, float), np.asarray(ear_, float)
    valid = np.isfinite(mar_) & np.isfinite(ear_) if valid is None else np.asarray(valid, bool)
    med = float(np.nanmedian(ear_[valid])) if valid.any() else float("nan")
    return FaceSeries(t, valid, mar_, ear_, np.asarray(yaw, float), np.asarray(pitch, float),
                      np.asarray(roll, float), fps, med)


def _s(a, fn):
    return float(fn(a)) if len(a) else float("nan")


def window_stats(series: FaceSeries | None, start: float, end: float, min_valid: int = MIN_VALID) -> dict:
    """Temporal facial statistics for [start, end). Returns {"features", "available", "reason", "n_valid", "n_frames",
    "valid_frac"}. Features are NaN (never 0) when unavailable."""
    nan = {k: float("nan") for k in STAT_NAMES}
    if series is None:
        return {"features": nan, "available": False, "reason": "no landmark series (face landmarker unavailable)",
                "n_valid": 0, "n_frames": 0, "valid_frac": 0.0}
    m = (series.t >= start) & (series.t < end)
    n_frames = int(m.sum())
    v = m & series.valid
    nv = int(v.sum())
    frac = nv / n_frames if n_frames else 0.0
    if nv < min_valid:
        why = "no face landmarks in this window" if nv == 0 else f"only {nv} valid landmark frames (< {min_valid})"
        return {"features": nan, "available": False, "reason": why, "n_valid": nv, "n_frames": n_frames, "valid_frac": frac}
    mv, ev = series.mar[v], series.ear[v]
    dt = np.diff(series.t[v])
    ok = (dt > 0) & (dt < 2.5 / series.fps)           # consecutive frames only
    vel = np.abs(np.diff(mv))[ok] / dt[ok]
    f = {
        "mar_mean": float(mv.mean()), "mar_std": float(mv.std()), "mar_range": float(mv.max() - mv.min()),
        "mar_velocity": float(vel.mean()) if len(vel) else float("nan"),
        "ear_mean": float(ev.mean()), "ear_std": float(ev.std()),
        "blink_count": float(blink_count(series.ear[m], series.valid[m], BLINK_FRAC * series.ear_median)),
    }
    for name, arr in (("yaw", series.yaw), ("pitch", series.pitch), ("roll", series.roll)):
        a = arr[v]
        a = a[np.isfinite(a)]
        f[f"{name}_mean"], f[f"{name}_std"] = _s(a, np.mean), _s(a, np.std)
    return {"features": f, "available": True, "reason": "", "n_valid": nv, "n_frames": n_frames, "valid_frac": frac}


# ------------------------------------------------------------------ landmarker (model) + video pass
def get_landmarker():
    if "lm" in _CACHE:
        return _CACHE["lm"]
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    if not LANDMARKER_PATH.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LANDMARKER_URL, timeout=120) as r:
            LANDMARKER_PATH.write_bytes(r.read())
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        num_faces=2, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    _CACHE["lm"] = (vision.FaceLandmarker.create_from_options(opts), mp)
    return _CACHE["lm"]


def landmark_series(path, max_seconds: float = 90.0, target_fps: float = TARGET_FPS, progress=None) -> FaceSeries:
    """Run the landmarker at ~target_fps over the video. Raises on landmarker/video failure (caller wraps)."""
    lm, mp = get_landmarker()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        raise RuntimeError("video fps unknown")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / target_fps)))
    last = int(min(n_total, max_seconds * fps)) if n_total > 0 else int(max_seconds * fps)
    T, MAR, EAR, YAW, PIT, ROLL, VALID = [], [], [], [], [], [], []
    n_multi, i = 0, 0
    while i < last:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                s = min(1.0, MAX_SIDE / max(h, w))
                if s < 1.0:
                    frame = cv2.resize(frame, (int(w * s), int(h * s)))
                    h, w = frame.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                rec = (float("nan"),) * 5
                if res.face_landmarks:
                    pts_all = [np.array([[p.x * w, p.y * h] for p in f]) for f in res.face_landmarks]
                    areas = [(p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()) for p in pts_all]
                    k = int(np.argmax(areas))
                    n_multi += int(len(pts_all) > 1)
                    pts = pts_all[k]
                    ang = (float("nan"),) * 3
                    if res.facial_transformation_matrixes:
                        ang = euler_from_matrix(np.array(res.facial_transformation_matrixes[k]))
                    rec = (mar(pts), ear(pts)) + ang
                T.append(i / fps)
                MAR.append(rec[0]); EAR.append(rec[1]); YAW.append(rec[2]); PIT.append(rec[3]); ROLL.append(rec[4])
        i += 1
        if progress and i % 60 == 0:
            progress(min(1.0, i / max(last, 1)))
    cap.release()
    ser = series_from_arrays(T, MAR, EAR, YAW, PIT, ROLL, fps=fps / step)
    ser.n_multi = n_multi
    return ser
