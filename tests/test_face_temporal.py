import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import face_temporal as ft


def mesh():
    return np.zeros((478, 2))


def test_mar_hand_computed():
    p = mesh()
    p[78], p[308] = (0, 0), (10, 0)                        # width 10
    p[82], p[87] = (2, -1), (2, 1)                         # gap 2
    p[13], p[14] = (5, -1.5), (5, 1.5)                     # gap 3
    p[312], p[317] = (8, -1), (8, 1)                       # gap 2
    assert ft.mar(p) == pytest.approx((2 + 3 + 2) / (3 * 10))
    p[308] = (0, 0)                                        # degenerate mouth width
    assert np.isnan(ft.mar(p))


def test_ear_hand_computed():
    p = mesh()
    for (a, b, c, d, e, f), x0 in ((ft.EYE_R, 0.0), (ft.EYE_L, 50.0)):
        p[a], p[d] = (x0, 0), (x0 + 10, 0)                 # width 10
        p[b], p[f] = (x0 + 3, -2), (x0 + 3, 2)             # gap 4
        p[c], p[e] = (x0 + 7, -2), (x0 + 7, 2)             # gap 4
    assert ft.ear(p) == pytest.approx((4 + 4) / (2 * 10))  # 0.4
    p[ft.EYE_R[1]], p[ft.EYE_R[5]] = (3, 0), (3, 0)        # right eye half closed -> (0+4)/20 = 0.2 ; mean with 0.4
    assert ft.ear(p) == pytest.approx((0.2 + 0.4) / 2)


def test_euler_angles():
    th = np.radians(20)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    yaw, pitch, roll = ft.euler_from_matrix(Ry)
    assert yaw == pytest.approx(20.0) and pitch == pytest.approx(0.0, abs=1e-6) and roll == pytest.approx(0.0, abs=1e-6)
    ph = np.radians(30)
    Rz = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
    assert ft.euler_from_matrix(Rz)[2] == pytest.approx(30.0)


def make_series(seconds=8.0, fps=12.0, drop=None):
    t = np.arange(0, seconds, 1 / fps)
    mar = 0.3 + 0.1 * np.sin(2 * np.pi * 2 * t)
    ear = np.full_like(t, 0.30)
    ear[10:12] = 0.05                                       # blink 1 (frames 10-11, t~0.83 s)
    ear[60:62] = 0.05                                       # blink 2 (t = 5.0 s)
    yaw = np.full_like(t, 5.0)
    pitch = 2.0 + np.sin(t)
    roll = np.zeros_like(t)
    valid = np.ones_like(t, bool)
    if drop:
        valid[(t >= drop[0]) & (t < drop[1])] = False
        mar = np.where(valid, mar, np.nan)
    return ft.series_from_arrays(t, mar, ear, yaw, pitch, roll, valid=valid, fps=fps)


def test_temporal_aggregation_hand_checked():
    s = make_series()
    w = ft.window_stats(s, 0.0, 4.0)
    assert w["available"] and w["n_frames"] == 48 and w["n_valid"] == 48
    f = w["features"]
    assert f["mar_std"] == pytest.approx(0.1 / np.sqrt(2), rel=1e-6)
    # 12 fps sampling of a 2 Hz sine hits phases 0, 60, 120... deg: max |sin| = sin(60 deg) = 0.866 -> range = 2*0.1*0.866
    assert f["mar_range"] == pytest.approx(2 * 0.1 * np.sin(np.pi / 3), rel=1e-6)
    assert f["mar_velocity"] > 0
    assert f["ear_mean"] == pytest.approx(0.30 - 0.25 * 2 / 48, rel=1e-6)
    assert f["blink_count"] == 1                            # only the 0.83 s blink is inside [0, 4)
    assert ft.window_stats(s, 4.0, 8.0)["features"]["blink_count"] == 1
    assert f["yaw_mean"] == pytest.approx(5.0) and f["yaw_std"] == pytest.approx(0.0, abs=1e-9)
    assert f["roll_mean"] == pytest.approx(0.0)
    assert np.isfinite(list(f.values())).all()              # no NaN/inf for an available window


def test_missing_face_window_is_na_not_zero():
    s = make_series(drop=(4.0, 8.0))
    w = ft.window_stats(s, 4.0, 8.0)
    assert not w["available"] and w["n_valid"] == 0 and "no face" in w["reason"]
    assert all(np.isnan(v) for v in w["features"].values())
    assert ft.window_stats(s, 0.0, 4.0)["available"]        # the other window is unaffected


def test_insufficient_frames_and_no_series():
    s = make_series(drop=(0.0, 3.5))                        # only 6 valid frames left in [0,4)
    w = ft.window_stats(s, 0.0, 4.0)
    assert not w["available"] and "valid landmark frames" in w["reason"]
    w = ft.window_stats(None, 0.0, 4.0)
    assert not w["available"] and "landmarker unavailable" in w["reason"]


def test_blink_counter():
    e = np.array([.3, .3, .1, .1, .3, .3, .1, .3])
    v = np.ones(8, bool)
    assert ft.blink_count(e, v, 0.2) == 2
    v[2] = False                                            # invalid frames never start or extend a blink
    assert ft.blink_count(e, v, 0.2) == 2
