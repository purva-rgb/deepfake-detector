"""The API serializer is a view of the existing report: no invented scores, None stays N/A."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.serializer import build_response
from src import config, fusion

DEMO = Path("outputs/reports/demo_report.json")


def _tiny_report():
    """Minimal report with several N/A branches (no audio, no visual, no transcript)."""
    win = {
        "D": None, "R_media": None, "R": None, "S": None, "M": None, "R_sum": 0.0,
        "used": {"visual": False, "audio": False}, "cap_applied": False, "synergy": False,
        "insufficient": True, "band": fusion.INSUFFICIENT, "start": 0.0, "end": 4.0,
        "S_visual": None, "S_audio": None, "R_v": None, "R_a": None, "R_t": None,
        "n_faces": 0, "n_frames": 2, "speech_ratio": None, "categories": [],
        "face_temporal": {"available": False, "reason": "no usable landmarks", "features": {}},
        "acoustic": {"available": False, "reason": "no usable audio in this window", "features": {}},
        "sync": {"sync_score": None, "best_lag_ms": None, "reliability": None, "status": "N/A",
                 "reason": "no usable landmarks"},
    }
    return {
        "path": "x.mp4", "sha1": "abc123",
        "meta": {"duration": 4.0, "has_audio": False, "fps": 25, "width": 224, "height": 224},
        "video": {
            "D_video": None, "S_video": None, "M_video": None, "R_media_video": None, "R_video": None,
            "R_sum": 0.0, "synergy": False, "insufficient": True, "band": fusion.INSUFFICIENT,
            "R_v": None, "R_a": None, "R_t": None, "M": None, "D": None, "S": None, "R": None,
            "media_band": "N/A", "scam_band": "N/A",
            "sync": {"sync_score": None, "best_lag_ms": None, "reliability": None, "status": "N/A",
                     "reason": "no usable landmarks", "n_valid_windows": 0},
            "coverage": {"n_windows": 1, "n_windows_with_D": 0, "n_windows_D_ge_0.5": 0,
                         "D_median": None, "D_max": None, "n_windows_sync_ok": 0,
                         "n_windows_face_temporal_ok": 0, "n_windows_acoustic_ok": 0},
        },
        "windows": [win], "evidence": [], "checklist": [
            {"category": "urgency", "label": "Urgency / pressure", "found": False, "matches": []},
        ],
        "suppressed_matches": [], "transcript": [], "transcript_ok": False,
        "warnings": ["English only: other languages are not analysed reliably.",
                     "AV mismatch (M) is not fused: the lightweight sync score below is supporting evidence only."],
        "timings": {"total": 1.0}, "low_confidence": {"visual": False, "audio": False},
        "head_meta": {}, "weights": {}, "crop_thumbs_available": 0, "face_backend": "none",
    }


def test_na_never_becomes_zero():
    r = build_response(_tiny_report())
    assert r["media_risk"]["score"] is None
    assert r["media_risk"]["band"] == "N/A"
    assert r["scam_risk"]["score"] is None
    assert r["visual"]["available"] is False and r["visual"]["score"] is None
    assert r["audio"]["available"] is False and r["audio"]["score"] is None
    assert r["semantic"]["available"] is False and r["semantic"]["score"] is None
    assert r["av_sync"]["available"] is False and r["av_sync"]["score"] is None
    assert r["av_sync"]["supporting_only"] is True
    assert r["acoustic"]["available"] is False
    assert r["facial_temporal"]["available"] is False
    assert r["transcript"] == ""
    assert r["evidence"] == []
    assert r["warnings"] == []                          # restated-elsewhere notes are dropped
    assert "path" not in r and "timings" not in r and "head_meta" not in r


@pytest.mark.skipif(not DEMO.exists(), reason="no saved real report")
def test_demo_report_maps_existing_scores():
    raw = json.loads(DEMO.read_text(encoding="utf-8"))
    r = build_response(raw)
    v = raw["video"]
    assert r["analysis_id"] == raw["sha1"]
    assert r["media_risk"]["score"] == v["D"]
    assert r["scam_risk"]["score"] == v["S"]
    assert r["media_risk"]["band"] == fusion.band(v["D"])
    assert r["scam_risk"]["band"] == fusion.band(v["S"])
    assert r["reliability"]["visual"] == v["R_v"]
    assert r["reliability"]["audio"] == v["R_a"]
    assert r["reliability"]["transcript"] == v["R_t"]
    assert r["av_sync"]["score"] == v["sync"]["sync_score"]
    assert r["av_sync"]["best_lag_ms"] == v["sync"]["best_lag_ms"]
    assert r["av_sync"]["supporting_only"] is True
    assert len(r["windows"]) == len(raw["windows"])
    assert len(r["evidence"]) == len(raw["evidence"])
    assert r["windows"][0]["start"] == raw["windows"][0]["start"]
    assert r["windows"][0]["visual"]["score"] == raw["windows"][0]["S_visual"]
    assert r["windows"][0]["audio"]["score"] == raw["windows"][0]["S_audio"]
    assert r["semantic"]["categories"][0]["found"] == raw["checklist"][0]["found"]
    assert r["video"]["window_s"] == config.WINDOW_S
    assert r["video"]["hop_s"] == config.HOP_S
    # video-level visual score is the existing top-k mean, not a new formula
    gated = [w["S_visual"] for w in raw["windows"]
             if w["S_visual"] is not None and w["R_v"] is not None and w["R_v"] >= config.R_GATE]
    assert r["visual"]["score"] == fusion.top_k_mean(gated, len(raw["windows"]))
    assert "SPEC" not in json.dumps(r)
    assert "R_sum" not in json.dumps(r)
    assert "xception" not in json.dumps(r).lower()
    assert "wav2vec" not in json.dumps(r).lower()
