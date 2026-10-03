"""Fusion tests with hand-computed numbers (SPEC 6-7, 13)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion import INSUFFICIENT, aggregate_video, band, fuse, top_k_mean


def test_D_weighted_by_reliability():
    # D = (0.8*0.9 + 0.4*0.3) / (0.8 + 0.4) = (0.72 + 0.12) / 1.2 = 0.70
    r = fuse(S_visual=0.9, R_v=0.8, S_audio=0.3, R_a=0.4, S_scam=0.2)
    assert r["D"] == pytest.approx(0.70)
    assert r["R"] == pytest.approx(0.70) and r["band"] == "HIGH"
    assert not r["cap_applied"] and not r["synergy"]


def test_missing_modality_removes_its_term_and_weight():
    # audio absent: D = S_visual exactly (0.6), NOT (0.5*0.6 + 0.1*0)/0.6 = 0.5
    assert fuse(S_visual=0.6, R_v=0.5, S_audio=None, R_a=None)["D"] == pytest.approx(0.6)
    # audio present but below the 0.15 gate (R_a = 0.10): dropped too, even with a high score
    r = fuse(S_visual=0.6, R_v=0.5, S_audio=0.99, R_a=0.10)
    assert r["D"] == pytest.approx(0.6) and r["used"] == {"visual": True, "audio": False}
    assert r["R_sum"] == pytest.approx(0.5)      # 0.5 is not < 0.5, so no cap
    # visual absent instead
    assert fuse(S_visual=None, R_v=None, S_audio=0.3, R_a=0.9)["D"] == pytest.approx(0.3)


def test_low_reliability_cap():
    # R_sum = 0.4 < 0.5 (audio gated out): D = 0.95 capped to 0.64 -> MEDIUM, never HIGH
    r = fuse(S_visual=0.95, R_v=0.4, S_audio=0.9, R_a=0.09, S_scam=0.2)
    assert r["D"] == pytest.approx(0.64) and r["R"] == pytest.approx(0.64)
    assert r["cap_applied"] and r["band"] == "MEDIUM"
    # a strong scam score can still be HIGH; D=0.64 is not > 0.65 so no synergy: R = 0.9
    r2 = fuse(S_visual=0.95, R_v=0.4, S_scam=0.9)
    assert r2["R"] == pytest.approx(0.9) and not r2["synergy"] and r2["band"] == "HIGH"
    # cap also limits R_media when M is present: D capped 0.64, R_media = 0.6*0.64 + 0.4*0.9 = 0.744 -> capped 0.64
    r3 = fuse(S_visual=0.95, R_v=0.4, M=0.9)
    assert r3["R_media"] == pytest.approx(0.64)


def test_synergy_applied_once():
    # D = 0.8, M = 0.6 -> R_media = 0.6*0.8 + 0.4*0.6 = 0.72; S = 0.7
    # R = max(0.7, 0.8, 0.72) = 0.8; synergy once -> 0.9 (not 1.0)
    r = fuse(S_visual=0.8, R_v=1.0, S_audio=0.8, R_a=1.0, S_scam=0.7, M=0.6)
    assert r["R_media"] == pytest.approx(0.72)
    assert r["synergy"] and r["R"] == pytest.approx(0.9)


def test_synergy_clamped_and_strict_threshold():
    assert fuse(S_visual=0.95, R_v=1.0, S_scam=0.95)["R"] == pytest.approx(1.0)  # 0.95 + 0.10 -> min(1, 1.05)
    # D exactly 0.65 is NOT > 0.65: no synergy, R = max(0.9, 0.65) = 0.9
    r = fuse(S_visual=0.65, R_v=1.0, S_scam=0.9)
    assert not r["synergy"] and r["R"] == pytest.approx(0.9)
    # S exactly 0.65 also no synergy
    r = fuse(S_visual=0.9, R_v=1.0, S_scam=0.65)
    assert not r["synergy"] and r["R"] == pytest.approx(0.9)


def test_insufficient_evidence_never_low():
    # both media dropped (R < 0.15), S = 0.2 -> INSUFFICIENT, D is N/A
    r = fuse(S_visual=0.9, R_v=0.10, S_audio=0.9, R_a=0.10, S_scam=0.2)
    assert r["D"] is None and r["band"] == INSUFFICIENT
    # same media but strong scam: D N/A and the band comes from S
    r = fuse(S_visual=0.9, R_v=0.10, S_audio=0.9, R_a=0.10, S_scam=0.8)
    assert r["D"] is None and r["band"] == "HIGH" and r["R"] == pytest.approx(0.8)
    # R_sum = 0.2 < 0.30 and S = 0.1 < 0.35 -> INSUFFICIENT even though one modality survived
    r = fuse(S_visual=0.9, R_v=0.2, S_scam=0.1)
    assert r["band"] == INSUFFICIENT
    # no face (R_v = 0, no score), no audio, no text -> INSUFFICIENT
    assert fuse(S_visual=None, R_v=0.0)["band"] == INSUFFICIENT
    # S exactly at 0.35 is enough evidence to leave INSUFFICIENT: MEDIUM, 0.34 is not
    assert fuse(R_v=0.0, S_scam=0.35)["band"] == "MEDIUM"
    assert fuse(R_v=0.0, S_scam=0.34)["band"] == INSUFFICIENT
    # R_sum exactly 0.30 is not < 0.30
    assert fuse(S_visual=0.1, R_v=0.30, S_scam=0.0)["band"] == "LOW"


def test_low_confidence_head_weight():
    # visual weight x0.3: D = (0.8*0.3*1.0 + 0.8*0.0) / (0.8*0.3 + 0.8) = 0.24 / 1.04
    r = fuse(S_visual=1.0, R_v=0.8, S_audio=0.0, R_a=0.8, weights={"visual": 0.3})
    assert r["D"] == pytest.approx(0.24 / 1.04)


def test_bands_boundaries():
    assert band(0.3499) == "LOW" and band(0.35) == "MEDIUM"
    assert band(0.6499) == "MEDIUM" and band(0.65) == "HIGH"
    assert band(None) == INSUFFICIENT


def test_top3_window_aggregation():
    assert top_k_mean([0.9, 0.8, 0.7, 0.2, 0.1]) == pytest.approx(0.8)            # (0.9+0.8+0.7)/3
    # 16 windows -> top 25% = 4 windows: (0.9+0.9+0.7+0.7)/4 = 0.8   (top-3 would be 0.8333)
    vals = [0.9, 0.9, 0.7, 0.7] + [0.1] * 12
    assert top_k_mean(vals) == pytest.approx(0.8)
    assert top_k_mean([None, None]) is None
    assert top_k_mean([0.5, None]) == pytest.approx(0.5)                           # N/A windows are skipped, not zero


def test_video_aggregation():
    wins = [fuse(S_visual=d, R_v=1.0) for d in (0.9, 0.8, 0.7, 0.2, 0.1)]
    v = aggregate_video(wins, R_v_video=1.0, R_a_video=None)
    assert v["D_video"] == pytest.approx(0.8) and v["S_video"] is None and v["M_video"] is None
    assert v["R_video"] == pytest.approx(0.8) and v["band"] == "HIGH"


def test_video_both_media_dropped_band_from_S():
    wins = [{"D": None, "S": 0.2, "M": None}, {"D": None, "S": 0.5, "M": None}]
    v = aggregate_video(wins, R_v_video=0.10, R_a_video=None)
    assert v["D_video"] is None and v["S_video"] == pytest.approx(0.5) and v["band"] == "MEDIUM"
    low_s = [{"D": None, "S": 0.2, "M": None}]
    assert aggregate_video(low_s, R_v_video=0.0, R_a_video=None)["band"] == INSUFFICIENT


def test_video_cap_and_synergy():
    wins = [{"D": 0.9, "S": 0.2, "M": None}] * 3
    v = aggregate_video(wins, R_v_video=0.4, R_a_video=None)          # R_sum 0.4 < 0.5 -> D capped
    assert v["D_video"] == pytest.approx(0.64) and v["band"] == "MEDIUM"
    wins = [{"D": 0.8, "S": 0.7, "M": None}]
    v = aggregate_video(wins, R_v_video=1.0)                          # max(0.7, 0.8) + 0.10
    assert v["synergy"] and v["R_video"] == pytest.approx(0.9)
