import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import av_sync as sy
from src import face_temporal as ft

SR = 16000


def env(t):
    return 1.0 + np.sin(2 * np.pi * 1.7 * t) + 0.7 * np.sin(2 * np.pi * 3.1 * t + 1.0)


def make(lag_s=0.0, seconds=4.0, fps=12.0, seed=0):
    t_v = np.arange(0, seconds, 1 / fps)
    mar = 0.3 + 0.05 * env(t_v)                             # mouth opening follows the envelope
    ta = np.arange(int(SR * seconds)) / SR
    amp = np.clip(env(ta - lag_s), 0.05, None)              # audio energy follows it, delayed by lag_s
    wave = (np.random.RandomState(seed).randn(len(ta)) * 0.1 * amp).astype(np.float32)
    series = ft.series_from_arrays(t_v, mar, np.full_like(t_v, 0.3), np.zeros_like(t_v), np.zeros_like(t_v),
                                   np.zeros_like(t_v), fps=fps)
    return series, wave


def test_aligned_signals_high_sync_zero_lag():
    series, wave = make(0.0)
    r = sy.sync_window(series, wave, 0.0, 4.0, speech_ratio=0.9, r_a=0.9)
    assert r["status"] == "ok" and r["sync_score"] > 0.85
    assert abs(r["best_lag_ms"]) <= 40
    assert 0 < r["reliability"] <= 1


def test_known_lag_is_recovered():
    series, wave = make(0.1)                                # audio delayed 100 ms
    r = sy.sync_window(series, wave, 0.0, 4.0, 0.9, 0.9)
    assert r["status"] == "ok" and r["sync_score"] > 0.8
    assert 60 <= r["best_lag_ms"] <= 140                    # positive: audio lags mouth


def test_unrelated_signals_score_lower():
    series, wave = make(0.0)
    rng = np.random.RandomState(3)
    series.mar = 0.3 + 0.05 * rng.randn(len(series.mar))
    r = sy.sync_window(series, wave, 0.0, 4.0, 0.9, 0.9)
    assert r["status"] == "ok" and r["sync_score"] < 0.7


@pytest.mark.parametrize("kw,needle", [
    (dict(speech_ratio=0.2, r_a=0.9), "speech"),
    (dict(speech_ratio=0.9, r_a=0.1), "reliability"),
    (dict(speech_ratio=None, r_a=0.9), "speech"),
    (dict(speech_ratio=0.9, r_a=None), "reliability"),
])
def test_insufficient_audio_quality(kw, needle):
    series, wave = make()
    r = sy.sync_window(series, wave, 0.0, 4.0, **kw)
    assert r["sync_score"] is None and r["status"] == "N/A" and needle in r["reason"]


def test_insufficient_face_frames_and_missing_inputs():
    series, wave = make()
    series.valid[::2] = False                               # 50% valid < 70%
    r = sy.sync_window(series, wave, 0.0, 4.0, 0.9, 0.9)
    assert r["sync_score"] is None and "usable face frames" in r["reason"]
    assert sy.sync_window(None, wave, 0.0, 4.0, 0.9, 0.9)["sync_score"] is None
    series, _ = make()
    assert sy.sync_window(series, None, 0.0, 4.0, 0.9, 0.9)["reason"] == "no audio"
    flat, wave = make()
    flat.mar[:] = 0.3
    assert "no mouth movement" in sy.sync_window(flat, wave, 0.0, 4.0, 0.9, 0.9)["reason"]
    assert "flat audio" in sy.sync_window(series, np.full(SR * 4, 0.1, np.float32), 0.0, 4.0, 0.9, 0.9)["reason"]


def test_summary_never_fabricates():
    na = sy.summarise([sy.sync_window(None, None, 0, 4, 0.9, 0.9)])
    assert na["sync_score"] is None and na["status"] == "N/A" and na["reason"]
    series, wave = make()
    ok = sy.sync_window(series, wave, 0.0, 4.0, 0.9, 0.9)
    s = sy.summarise([ok, na])
    assert s["status"] == "ok" and s["n_valid_windows"] == 1 and s["sync_score"] == pytest.approx(ok["sync_score"])
