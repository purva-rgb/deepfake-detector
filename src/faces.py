"""Face detection: MediaPipe (tasks API) with cv2 Haar cascade fallback.

detect_largest(bgr) -> dict(box=(x0,y0,x1,y1), conf=float, size_frac=float) or None
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

# Official MediaPipe model asset (BlazeFace short range).
MP_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MP_FACE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"


class FaceDetector:
    def __init__(self, backend: str = "auto", download_timeout: float = 120.0):
        self.backend = None
        self.note = ""
        self._mp = None
        self._haar = None
        if backend in ("auto", "mediapipe"):
            try:
                self._init_mediapipe(download_timeout)
                self.backend = "mediapipe"
            except Exception as e:  # noqa: BLE001
                self.note = f"MediaPipe unavailable ({type(e).__name__}: {e}); using Haar cascade"
        if self.backend is None:
            self._init_haar()
            self.backend = "haar"

    def _init_mediapipe(self, timeout):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        if not MP_FACE_MODEL_PATH.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(MP_FACE_MODEL_URL, timeout=timeout) as r:
                MP_FACE_MODEL_PATH.write_bytes(r.read())
        opts = vision.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(MP_FACE_MODEL_PATH)),
            min_detection_confidence=0.5,
        )
        self._mp_mod = mp
        self._mp = vision.FaceDetector.create_from_options(opts)
        # smoke call so a broken runtime fails here, not mid-extraction
        dummy = np.zeros((64, 64, 3), np.uint8)
        self._mp.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy))

    def _init_haar(self):
        path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        self._haar = cv2.CascadeClassifier(path)
        if self._haar.empty():
            raise RuntimeError(f"Could not load Haar cascade at {path}")

    def detect_largest(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        if self.backend == "mediapipe":
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = self._mp.detect(self._mp_mod.Image(image_format=self._mp_mod.ImageFormat.SRGB, data=rgb))
            best = None
            for d in res.detections:
                bb = d.bounding_box
                area = bb.width * bb.height
                conf = float(d.categories[0].score) if d.categories else 0.0
                if best is None or area > best[0]:
                    best = (area, bb.origin_x, bb.origin_y, bb.width, bb.height, conf)
            if best is None:
                return None
            _, x, y, bw, bh, conf = best
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rects, _, weights = self._haar.detectMultiScale3(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), outputRejectLevels=True
            )
            if len(rects) == 0:
                return None
            i = int(np.argmax([r[2] * r[3] for r in rects]))
            x, y, bw, bh = [int(v) for v in rects[i]]
            # Haar level-weights are not probabilities; squash into (0,1) as a rough confidence proxy
            conf = float(1.0 - np.exp(-max(float(weights[i]), 0.0) / 3.0))
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            return None
        return {
            "box": (int(x0), int(y0), int(x1), int(y1)),
            "conf": conf,
            "size_frac": float((y1 - y0) / h),
        }


def crop_with_margin(bgr: np.ndarray, box, margin: float):
    """Square-ish crop around box with `margin` fraction added on each side."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    mx, my = int(bw * margin), int(bh * margin)
    X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    return bgr[Y0:Y1, X0:X1]


def blur_score(crop_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128, 128))  # normalise scale so score is comparable across face sizes
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
