"""Lightweight audio-visual sync (SPEC 4.5): MAR(t) vs short-time audio RMS, normalised cross-correlation, +/-200 ms.

Per window:
  1. preconditions (SPEC 5): >= 70% valid landmark frames, >= MIN_FRAMES valid frames, speech ratio >= 0.4, R_a >= 0.3
  2. MAR (valid frames, linearly interpolated) and audio RMS (40 ms frames) are put on a 50 Hz grid,
     smoothed with a 100 ms moving average and z-scored
  3. Pearson-style normalised cross-correlation at lags -200..+200 ms (20 ms steps)
  sync_score  = max correlation over lags (in [-1, 1]; HIGH = mouth and audio energy move together)
  best_lag_ms = lag at the maximum (positive = audio lags the mouth)
  reliability = valid_frac * min(1, speech_ratio/0.6) * R_a
Otherwise sync_score = None with an explicit reason. Not a mismatch score and NOT fed into fusion (M stays N/A):
dubbed / voice-over content can legitimately be out of sync, so it is shown as supporting evidence only.
"""
from __future__ import annotations

import numpy as np

GRID_HZ = 50.0
MAX_LAG_MS = 200.0
MIN_VALID_FRAC = 0.70
MIN_FRAMES = 12
MIN_SPEECH = 0.40
MIN_RA = 0.30
SMOOTH = 5  # samples at 50 Hz = 100 ms
SR = 16000


def _unavail(reason, **kw):
    return {"sync_score": None, "best_lag_ms": None, "reliability": None, "status": "N/A", "reason": reason, **kw}


def _z(x):
    s = x.std()
    return None if s < 1e-8 else (x - x.mean()) / s


def _smooth(x):
    k = SMOOTH // 2   # edge-padded moving average (zero padding would inject artificial edge dips)
    return np.convolve(np.pad(x, k, mode="edge"), np.ones(SMOOTH) / SMOOTH, mode="valid")


def xcorr_sync(t_mar, mar, wave, t0, sr=SR):
    """Core computation on already-validated inputs. t_mar: times (s) of valid MAR samples; wave: audio starting at
    absolute time t0. Returns (score, lag_ms, prominence) or (None, None, reason)."""
    t_mar, mar = np.asarray(t_mar, float), np.asarray(mar, float)
    lo, hi = float(t_mar[0]), float(t_mar[-1])
    n = int((hi - lo) * GRID_HZ)
    if n < 25:
        return None, None, "overlap too short"
    grid = lo + np.arange(n) / GRID_HZ
    vis = np.interp(grid, t_mar, mar)
    # audio RMS (40 ms frame centred on each grid time)
    half = int(0.02 * sr)
    centres = ((grid - t0) * sr).astype(int)
    if centres[0] - half < 0 or centres[-1] + half > len(wave):
        keep = (centres - half >= 0) & (centres + half <= len(wave))
        if keep.sum() < 25:
            return None, None, "audio does not cover the mouth-landmark interval"
        grid, vis, centres = grid[keep], vis[keep], centres[keep]
    idx = centres[:, None] + np.arange(-half, half)[None, :]
    aud = np.sqrt((wave[idx].astype(np.float64) ** 2).mean(1))
    zv, za = _z(_smooth(vis)), _z(_smooth(aud))
    if zv is None:
        return None, None, "no mouth movement (flat MAR signal)"
    if za is None:
        return None, None, "flat audio energy"
    L = int(MAX_LAG_MS / 1000 * GRID_HZ)
    cors = []
    for lag in range(-L, L + 1):
        a, b = (zv[: len(zv) - lag], za[lag:]) if lag >= 0 else (zv[-lag:], za[: len(za) + lag])
        cors.append(float(np.mean(a * b)) if len(a) >= 20 else float("nan"))
    cors = np.array(cors)
    if np.all(np.isnan(cors)):
        return None, None, "overlap too short"
    k = int(np.nanargmax(cors))
    prominence = float(cors[k] - np.nanmedian(cors))
    return float(cors[k]), float((k - L) * 1000.0 / GRID_HZ), prominence


def sync_window(series, wave, start, end, speech_ratio, r_a, sr=SR, wave_t0=0.0) -> dict:
    """series: face_temporal.FaceSeries or None; wave: normalised mono audio from absolute time wave_t0."""
    try:
        if series is None:
            return _unavail("no facial landmark series")
        if wave is None or len(wave) == 0:
            return _unavail("no audio")
        m = (series.t >= start) & (series.t < end)
        n_frames, v = int(m.sum()), m & series.valid
        nv = int(v.sum())
        frac = nv / n_frames if n_frames else 0.0
        info = {"n_valid_frames": nv, "valid_frac": frac}
        if frac < MIN_VALID_FRAC:
            return _unavail(f"only {frac:.0%} usable face frames (< {MIN_VALID_FRAC:.0%})", **info)
        if nv < MIN_FRAMES:
            return _unavail(f"only {nv} valid frames (< {MIN_FRAMES})", **info)
        if speech_ratio is None or not np.isfinite(speech_ratio) or speech_ratio < MIN_SPEECH:
            return _unavail("speech ratio below 0.4 or unavailable", **info)
        if r_a is None or r_a < MIN_RA:
            return _unavail("audio reliability below 0.3 or unavailable", **info)
        score, lag, extra = xcorr_sync(series.t[v], series.mar[v], wave, wave_t0, sr)
        if score is None:
            return _unavail(extra, **info)
        rel = float(frac * min(1.0, speech_ratio / 0.6) * min(1.0, r_a))
        return {"sync_score": score, "best_lag_ms": lag, "reliability": rel, "status": "ok", "reason": "",
                "peak_prominence": extra, **info}
    except Exception as e:  # noqa: BLE001
        return _unavail(f"sync failed ({type(e).__name__}: {e})")


def summarise(window_syncs: list[dict]) -> dict:
    ok = [w for w in window_syncs if w.get("sync_score") is not None]
    if not ok:
        reasons = sorted({w.get("reason", "") for w in window_syncs if w.get("reason")})
        return _unavail("; ".join(reasons) or "no windows", n_valid_windows=0)
    return {"sync_score": float(np.mean([w["sync_score"] for w in ok])),
            "best_lag_ms": float(np.median([w["best_lag_ms"] for w in ok])),
            "reliability": float(np.mean([w["reliability"] for w in ok])),
            "status": "ok", "reason": "", "n_valid_windows": len(ok)}
