"""Lightweight handcrafted acoustic features for one audio window (numpy only, CPU, no librosa).

Extracted per window (SR = 16 kHz; 32 ms frames, 10 ms hop):
  mfcc_mean_01..13, mfcc_std_01..13   (40 HTK-mel filters, log, DCT-II, first 13 coefficients)
  rms_mean, rms_std                   (frame RMS)
  zcr_mean                            (zero crossings per sample)
  spectral_centroid_mean (Hz), spectral_bandwidth_mean (Hz), spectral_rolloff_mean (Hz, 85% energy)
  pitch_mean, pitch_std (Hz)          (autocorrelation F0 on voiced frames; NaN when < MIN_VOICED frames)
  speech_ratio                        (webrtcvad / energy fallback from src.media)
  silence_ratio                       (fraction of 20 ms frames below SILENCE_DB)

Missing values are NaN (never 0). compute() never raises; it returns available=False + a reason instead.
FEATURE_NAMES gives a deterministic order.
"""
from __future__ import annotations

import numpy as np

from . import media

SR = 16000
N_MFCC = 13
N_MELS = 40
FRAME = 512          # 32 ms
HOP = 160            # 10 ms
SILENCE_DB = -45.0   # 20 ms frame energy (dBFS) below which a frame counts as silence
MIN_SECONDS = 0.5
MIN_VOICED = 5
F0_MIN, F0_MAX = 70.0, 400.0
VOICED_PEAK = 0.45   # min normalised autocorrelation peak for a voiced frame
VOICED_RMS = 0.005   # min frame RMS for a voiced frame

FEATURE_NAMES = (
    [f"mfcc_mean_{i:02d}" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_std_{i:02d}" for i in range(1, N_MFCC + 1)]
    + ["rms_mean", "rms_std", "zcr_mean", "spectral_centroid_mean", "spectral_bandwidth_mean",
       "spectral_rolloff_mean", "pitch_mean", "pitch_std", "speech_ratio", "silence_ratio"]
)

_CACHE: dict = {}


def _mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _inv_mel(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _filterbank():
    if "fb" not in _CACHE:
        pts = _inv_mel(np.linspace(_mel(0.0), _mel(SR / 2), N_MELS + 2))
        bins = np.floor((FRAME + 1) * pts / SR).astype(int)
        fb = np.zeros((N_MELS, FRAME // 2 + 1))
        for m in range(1, N_MELS + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            for k in range(l, c):
                fb[m - 1, k] = (k - l) / max(c - l, 1)
            for k in range(c, r):
                fb[m - 1, k] = (r - k) / max(r - c, 1)
        n = np.arange(N_MELS)
        dct = np.cos(np.pi / N_MELS * (n + 0.5)[None, :] * np.arange(N_MFCC)[:, None]) * np.sqrt(2.0 / N_MELS)
        dct[0] /= np.sqrt(2.0)
        _CACHE["fb"], _CACHE["dct"], _CACHE["win"] = fb, dct, np.hanning(FRAME)
    return _CACHE["fb"], _CACHE["dct"], _CACHE["win"]


def _frames(x, size, hop):
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _pitch(x):
    """Autocorrelation F0 per 40 ms frame (20 ms hop). Returns array of Hz with NaN for unvoiced."""
    size, hop = 640, 320
    if len(x) < size:
        return np.array([])
    fr = _frames(x, size, hop).astype(np.float64)
    fr -= fr.mean(1, keepdims=True)
    energy = np.sqrt((fr ** 2).mean(1))
    spec = np.fft.rfft(fr, 2048, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :size]
    ac0 = ac[:, 0] + 1e-12
    lo, hi = int(SR / F0_MAX), int(SR / F0_MIN)
    seg = ac[:, lo:hi + 1] / ac0[:, None]
    peak = seg.max(1)
    lag = seg.argmax(1) + lo
    f0 = np.where((peak > VOICED_PEAK) & (energy > VOICED_RMS), SR / lag, np.nan)
    return f0


def _nan_features():
    return {k: float("nan") for k in FEATURE_NAMES}


def compute(x: np.ndarray, sr: int = SR) -> dict:
    """Returns {"features": {name: float}, "available": bool, "reason": str}. Never raises."""
    feats = _nan_features()
    try:
        if sr != SR:
            return {"features": feats, "available": False, "reason": f"expected {SR} Hz audio, got {sr}"}
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1 or len(x) < int(MIN_SECONDS * SR):
            return {"features": feats, "available": False, "reason": f"audio shorter than {MIN_SECONDS}s"}
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.max(np.abs(x))) < 1e-4:
            feats["rms_mean"] = 0.0
            feats["speech_ratio"] = 0.0
            feats["silence_ratio"] = 1.0
            return {"features": feats, "available": False, "reason": "silent audio"}
        fb, dct, win = _filterbank()
        fr = _frames(x, FRAME, HOP).astype(np.float64)
        rms = np.sqrt((fr ** 2).mean(1))
        feats["rms_mean"], feats["rms_std"] = float(rms.mean()), float(rms.std())
        feats["zcr_mean"] = float((np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)).sum(1) / FRAME).mean())
        mag = np.abs(np.fft.rfft(fr * win, axis=1))
        pw = mag ** 2
        mel = np.log(pw @ fb.T + 1e-10)
        mf = mel @ dct.T
        for i in range(N_MFCC):
            feats[f"mfcc_mean_{i+1:02d}"] = float(mf[:, i].mean())
            feats[f"mfcc_std_{i+1:02d}"] = float(mf[:, i].std())
        freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
        keep = mag.sum(1) > 1e-8
        if keep.any():
            m = mag[keep]
            tot = m.sum(1)
            cen = (m * freqs).sum(1) / tot
            feats["spectral_centroid_mean"] = float(cen.mean())
            feats["spectral_bandwidth_mean"] = float(np.sqrt((m * (freqs[None, :] - cen[:, None]) ** 2).sum(1) / tot).mean())
            cum = np.cumsum(m ** 2, axis=1)
            roll = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
            feats["spectral_rolloff_mean"] = float(roll.mean())
        f0 = _pitch(x)
        voiced = f0[np.isfinite(f0)]
        if len(voiced) >= MIN_VOICED:
            feats["pitch_mean"], feats["pitch_std"] = float(voiced.mean()), float(voiced.std())
        feats["speech_ratio"] = float(media.speech_ratio(x))
        db = media.frame_db(x, 20)
        feats["silence_ratio"] = float(np.mean(db < SILENCE_DB)) if len(db) else float("nan")
        reason = "" if len(voiced) >= MIN_VOICED else f"pitch N/A (only {len(voiced)} voiced frames)"
        return {"features": feats, "available": True, "reason": reason}
    except Exception as e:  # noqa: BLE001
        return {"features": _nan_features(), "available": False, "reason": f"acoustic features failed ({type(e).__name__}: {e})"}


def to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)
