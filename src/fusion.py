"""Reliability-aware fusion, SPEC sections 6-7. Pure functions, no model code.

Missing modality = None (N/A). It is never replaced by 0.

Per window:
    D        = (R_v*S_visual + R_a*S_audio) / (R_v + R_a)   over SURVIVING modalities only
               (survive = score present AND R >= 0.15). A modality's fusion weight is further
               multiplied by `weights[...]` (0.3 for a low-confidence head, SPEC fallback).
    R_media  = 0.6*D + 0.4*M   if M available else D
    cap      : R_sum = R_v + R_a (surviving, unweighted). If R_sum < 0.5: D, R_media <= 0.64
    R        = max(S_scam, D, R_media)  over available values
    synergy  : if D > 0.65 and S_scam > 0.65: R = min(1, R + 0.10)   (applied once)
    insufficient: R_sum < 0.30 and S < 0.35 (S N/A counts as no evidence) -> INSUFFICIENT EVIDENCE
Video level (aggregate_video) follows SPEC 7: D_video = mean of top 3 windows by D (top 25% if > 12 windows).
"""
from __future__ import annotations

import math

from . import config

INSUFFICIENT = "INSUFFICIENT EVIDENCE"


def band(r: float | None) -> str:
    if r is None:
        return INSUFFICIENT
    if r < config.BAND_LOW:
        return "LOW"
    if r < config.BAND_HIGH:
        return "MEDIUM"
    return "HIGH"


def _survives(score, rel):
    return score is not None and rel is not None and rel >= config.R_GATE


def fuse(S_visual=None, R_v=None, S_audio=None, R_a=None, S_scam=None, M=None,
         weights: dict | None = None) -> dict:
    w = {"visual": 1.0, "audio": 1.0}
    w.update(weights or {})
    terms = []          # (weighted reliability, score)
    r_sum = 0.0
    used = {"visual": False, "audio": False}
    if _survives(S_visual, R_v):
        terms.append((R_v * w["visual"], S_visual))
        r_sum += R_v
        used["visual"] = True
    if _survives(S_audio, R_a):
        terms.append((R_a * w["audio"], S_audio))
        r_sum += R_a
        used["audio"] = True

    D = None
    den = sum(t[0] for t in terms)
    if terms and den > 0:
        D = sum(wr * s for wr, s in terms) / den
    R_media = None
    if D is not None:
        R_media = 0.6 * D + 0.4 * M if M is not None else D

    cap_applied = False
    if D is not None and r_sum < config.R_SUM_CAP:
        if D > config.CAP_VALUE or (R_media is not None and R_media > config.CAP_VALUE):
            cap_applied = True
        D = min(D, config.CAP_VALUE)
        R_media = min(R_media, config.CAP_VALUE)

    avail = [v for v in (S_scam, D, R_media) if v is not None]
    R = max(avail) if avail else None
    synergy = False
    if D is not None and S_scam is not None and D > config.SYNERGY_THRESH and S_scam > config.SYNERGY_THRESH:
        R = min(1.0, R + config.SYNERGY_BONUS)
        synergy = True

    s_eff = S_scam if S_scam is not None else 0.0
    insufficient = (r_sum < config.INSUFFICIENT_R_SUM and s_eff < config.INSUFFICIENT_S) or R is None
    return {
        "D": D, "R_media": R_media, "R": R, "S": S_scam, "M": M, "R_sum": r_sum, "used": used,
        "cap_applied": cap_applied, "synergy": synergy, "insufficient": insufficient,
        "band": INSUFFICIENT if insufficient else band(R),
    }


def top_k_mean(values, window_count: int | None = None):
    """Mean of top 3 values; top 25% (rounded up) when more than 12 windows. None if no values."""
    vals = sorted([v for v in values if v is not None], reverse=True)
    if not vals:
        return None
    n = window_count if window_count is not None else len(values)
    k = math.ceil(0.25 * n) if n > 12 else config.TOP_K_WINDOWS
    k = max(1, min(k, len(vals)))
    return sum(vals[:k]) / k


def aggregate_video(window_results: list[dict], R_v_video=None, R_a_video=None, weights=None) -> dict:
    """SPEC 7. window_results are outputs of fuse(). R_*_video are video-level reliabilities
    (None = modality unavailable). A modality below the gate at video level is dropped:
    if both are dropped, D_video = N/A and the band comes from S."""
    n = len(window_results)
    D_video = top_k_mean([r["D"] for r in window_results], n)
    s_vals = [r["S"] for r in window_results if r["S"] is not None]
    S_video = max(s_vals) if s_vals else None
    M_valid = [r["M"] for r in window_results if r["M"] is not None]
    M_video = top_k_mean(M_valid, len(M_valid)) if len(M_valid) >= 2 else None

    surviving = [r for r in (R_v_video, R_a_video) if r is not None and r >= config.R_GATE]
    r_sum = float(sum(surviving))
    if not surviving:
        D_video = None
    R_media_video = None
    if D_video is not None:
        R_media_video = 0.6 * D_video + 0.4 * M_video if M_video is not None else D_video
        if r_sum < config.R_SUM_CAP:
            D_video = min(D_video, config.CAP_VALUE)
            R_media_video = min(R_media_video, config.CAP_VALUE)

    avail = [v for v in (S_video, D_video, R_media_video) if v is not None]
    R_video = max(avail) if avail else None
    synergy = False
    if D_video is not None and S_video is not None and D_video > config.SYNERGY_THRESH and S_video > config.SYNERGY_THRESH:
        R_video = min(1.0, R_video + config.SYNERGY_BONUS)
        synergy = True
    s_eff = S_video if S_video is not None else 0.0
    insufficient = (r_sum < config.INSUFFICIENT_R_SUM and s_eff < config.INSUFFICIENT_S) or R_video is None
    return {
        "D_video": D_video, "S_video": S_video, "M_video": M_video, "R_media_video": R_media_video,
        "R_video": R_video, "R_sum": r_sum, "synergy": synergy, "insufficient": insufficient,
        "band": INSUFFICIENT if insufficient else band(R_video),
    }
