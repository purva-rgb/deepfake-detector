import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import acoustic_features as af

SR = 16000


def voiced(f0=150.0, seconds=4.0, seed=0):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))       # harmonic "vowel"
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))                         # syllable-like envelope
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def test_feature_names_deterministic_and_complete():
    r = af.compute(voiced())
    assert list(r["features"].keys()) == af.FEATURE_NAMES
    assert len(af.FEATURE_NAMES) == 2 * af.N_MFCC + 10
    assert len(set(af.FEATURE_NAMES)) == len(af.FEATURE_NAMES)


def test_voiced_audio_features_valid_and_finite():
    r = af.compute(voiced(150.0))
    assert r["available"]
    f = r["features"]
    assert abs(f["pitch_mean"] - 150.0) < 8.0, f["pitch_mean"]
    assert f["pitch_std"] < 15.0
    for k in af.FEATURE_NAMES:
        assert np.isfinite(f[k]), k            # no NaN/inf where a valid value is expected
    assert f["rms_mean"] > 0 and f["spectral_centroid_mean"] > 0 and f["spectral_rolloff_mean"] >= f["spectral_centroid_mean"] * 0.5
    assert 0.0 <= f["speech_ratio"] <= 1.0 and 0.0 <= f["silence_ratio"] <= 1.0
    # deterministic
    assert np.array_equal(af.to_vector(r["features"]), af.to_vector(af.compute(voiced(150.0))["features"]))


def test_pitch_tracks_frequency():
    lo, hi = af.compute(voiced(110.0))["features"]["pitch_mean"], af.compute(voiced(220.0))["features"]["pitch_mean"]
    assert abs(lo - 110) < 8 and abs(hi - 220) < 12


def test_silent_audio_is_unavailable_not_crash():
    r = af.compute(np.zeros(SR * 4, np.float32))
    assert not r["available"] and "silent" in r["reason"]
    assert r["features"]["speech_ratio"] == 0.0 and r["features"]["silence_ratio"] == 1.0
    assert np.isnan(r["features"]["pitch_mean"]) and np.isnan(r["features"]["mfcc_mean_01"])   # N/A, never 0


def test_too_short_or_bad_input():
    assert not af.compute(np.zeros(100, np.float32))["available"]
    assert not af.compute(voiced(), sr=8000)["available"]
    x = voiced()
    x[10] = np.nan
    assert af.compute(x)["available"]            # NaN samples are sanitised, no crash


def test_noise_has_no_fabricated_pitch():
    rng = np.random.RandomState(0)
    r = af.compute((0.1 * rng.randn(SR * 4)).astype(np.float32))
    assert r["available"]
    assert np.isnan(r["features"]["pitch_mean"]) and "pitch N/A" in r["reason"]
    assert np.isfinite(r["features"]["mfcc_std_03"])
