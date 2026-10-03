"""Reliability formulas (SPEC 5). All outputs in [0,1]. Plain, documented heuristics (not calibrated).

R_v (per window, from the face crops found in that window):
    per crop  q = (conf_term * size_term * blur_term) ** (1/3)          # geometric mean
        conf_term = clip(detector confidence, 0, 1)
        size_term = clip(face_height / frame_height / FACE_SIZE_FULL, 0, 1)
        blur_term = clip(variance_of_Laplacian(128x128 gray crop) / BLUR_FULL, 0, 1)
    R_v = mean(q over crops) * (frames_with_face / frames_sampled)       # visible fraction
    R_v = 0.0 ONLY when no face was found in the window.  If the visual stage itself failed -> None (N/A).

R_a (per window, from audio measurements of the non-silent part of the window):
    R_a = (speech_term * clip_term * snr_term) ** (1/3)
        speech_term = clip(speech_ratio / SPEECH_FULL, 0, 1)
        clip_term   = 1 - clip(clipping_fraction / CLIP_BAD, 0, 1)
        snr_term    = clip(snr_like_dB / SNR_FULL_DB, 0, 1)
    No audio / audio stage failed -> None (N/A).

R_t: mean over transcript segments of exp(avg_logprob) * (1 - no_speech_prob)  (see semantic.py).

Gate: a modality with R < R_GATE (0.15) is dropped (N/A) - never replaced by 0.
"""
from __future__ import annotations

import math

import numpy as np

from . import config


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def crop_quality(conf: float, size_frac: float, blur: float) -> float:
    terms = (
        _clip01(conf),
        _clip01(size_frac / config.FACE_SIZE_FULL),
        _clip01(blur / config.BLUR_FULL),
    )
    return float(np.prod(terms) ** (1.0 / 3.0))


def visual_reliability(confs, sizes, blurs, n_frames: int, n_faces: int) -> float:
    """R_v for one window. 0.0 only if no face; caller passes None itself if the visual stage failed."""
    if n_frames <= 0 or n_faces <= 0 or len(confs) == 0:
        return 0.0
    q = np.mean([crop_quality(c, s, b) for c, s, b in zip(confs, sizes, blurs)])
    return _clip01(q * (n_faces / n_frames))


def audio_reliability(speech_ratio: float, clipping: float, snr_db: float) -> float | None:
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (speech_ratio, clipping, snr_db)):
        return None
    terms = (
        _clip01(speech_ratio / config.SPEECH_FULL),
        1.0 - _clip01(clipping / config.CLIP_BAD),
        _clip01(snr_db / config.SNR_FULL_DB),
    )
    return float(np.prod(terms) ** (1.0 / 3.0))


def gate(r: float | None) -> float | None:
    """Drop a modality (None = N/A) when its reliability is below the gate."""
    if r is None or r < config.R_GATE:
        return None
    return float(r)
