"""ffmpeg / OpenCV helpers and audio quality measurements (no model code here)."""
from __future__ import annotations

import re
import subprocess

import numpy as np

from .config import SAMPLE_RATE

_FFMPEG = None


def ffmpeg_exe() -> str:
    """System ffmpeg if on PATH, else the static binary shipped by imageio-ffmpeg."""
    global _FFMPEG
    if _FFMPEG is None:
        import shutil

        _FFMPEG = shutil.which("ffmpeg")
        if not _FFMPEG:
            import imageio_ffmpeg

            _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def probe(path) -> dict:
    """Parse `ffmpeg -i` stderr: audio/video stream presence and duration (seconds)."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    err = p.stderr
    dur = None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return {
        "has_audio": bool(re.search(r"Stream #\d+:\d+.*Audio:", err)),
        "has_video": bool(re.search(r"Stream #\d+:\d+.*Video:", err)),
        "duration": dur,
    }


def decode_audio(path, normalize: bool = False, max_seconds: float | None = None) -> np.ndarray | None:
    """Decode to 16 kHz mono float32. normalize=True applies ffmpeg loudnorm (EBU R128, single pass)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    if normalize:
        cmd += ["-af", "loudnorm=I=-23:TP=-2:LRA=11"]
    cmd += ["-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or len(p.stdout) < 4:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


# ---------------------------------------------------------------- audio measurements
FRAME_MS = 30


def _frames(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    return x[: n * frame].reshape(n, frame) if n > 0 else np.zeros((0, frame), np.float32)


def frame_db(x: np.ndarray, frame_ms: int = 20) -> np.ndarray:
    f = _frames(x, int(SAMPLE_RATE * frame_ms / 1000))
    if len(f) == 0:
        return np.zeros(0)
    return 10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + 1e-10)


def trim_bounds(x: np.ndarray, thr_db: float = -45.0, frame_ms: int = 20) -> tuple[float, float] | None:
    """(t0, t1) seconds of the non-silent region of loudness-normalised audio; None if all silent."""
    db = frame_db(x, frame_ms)
    idx = np.where(db > thr_db)[0]
    if len(idx) == 0:
        return None
    return idx[0] * frame_ms / 1000.0, (idx[-1] + 1) * frame_ms / 1000.0


def vad_backend() -> str:
    try:
        import webrtcvad  # noqa: F401

        return "webrtcvad"
    except Exception:  # noqa: BLE001
        return "energy"


def speech_ratio(x: np.ndarray) -> float:
    """Fraction of 30 ms frames flagged as speech. webrtcvad (aggr. 2) or energy fallback."""
    frame = int(SAMPLE_RATE * FRAME_MS / 1000)
    f = _frames(x, frame)
    if len(f) == 0:
        return 0.0
    try:
        import webrtcvad

        vad = webrtcvad.Vad(2)
        pcm = (np.clip(f, -1, 1) * 32767).astype(np.int16)
        voiced = [vad.is_speech(r.tobytes(), SAMPLE_RATE) for r in pcm]
        return float(np.mean(voiced))
    except Exception:  # noqa: BLE001  energy-based fallback
        db = 10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + 1e-10)
        floor = np.percentile(db, 10)
        return float(np.mean(db > max(floor + 10.0, -50.0)))


def clipping_fraction(x: np.ndarray, level: float = 0.99) -> float:
    return float(np.mean(np.abs(x) >= level)) if len(x) else 0.0


def snr_like_db(x: np.ndarray) -> float:
    """SNR-like score: 10*log10(P90 / P10) of 20 ms frame energies (loud frames vs. noise floor).
    A heuristic proxy, not a true SNR. Clipped to [0, 60] dB."""
    f = _frames(x, int(SAMPLE_RATE * 0.02))
    if len(f) < 5:
        return float("nan")
    e = np.mean(f.astype(np.float64) ** 2, axis=1) + 1e-12
    return float(np.clip(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 10)), 0, 60))


def loudness_dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)) if len(x) else float("nan")


def make_windows(duration: float, win: float, hop: float) -> list[tuple[float, float]]:
    """4 s windows with 2 s hop; clips shorter than one window give a single window."""
    if duration <= 0:
        return []
    if duration < win:
        return [(0.0, float(duration))]
    out, s = [], 0.0
    while s + win <= duration + 1e-6:
        out.append((s, s + win))
        s += hop
    return out
