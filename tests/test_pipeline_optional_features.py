"""Pipeline robustness: the new supporting features are optional. When they fail / are N/A the pipeline still
completes, reports N/A with reasons (never 0), and the SPEC fusion outputs (D, S, band) are unchanged."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import acoustic_features, config, face_temporal, pipeline

pytestmark = pytest.mark.skipif(not config.SUBSET_CSV.exists(), reason="needs data/subset.csv")


def _clip():
    sub = pd.read_csv(config.SUBSET_CSV)
    return config.DATASET_ROOT / sub[sub.split == "val"].iloc[1].rel_path


def test_supporting_features_present_and_fusion_unchanged(monkeypatch):
    base = pipeline.run_pipeline(_clip())
    w = base["windows"][0]
    assert set(w) >= {"face_temporal", "acoustic", "sync", "S_visual", "S_audio", "S", "start", "end", "R_v", "R_a", "R_t"}
    assert base["video"]["M"] is None                       # sync is supporting evidence, never fused as M
    assert base["video"]["coverage"]["n_windows"] == len(base["windows"])

    def boom(*a, **k):
        raise RuntimeError("landmarker offline")

    monkeypatch.setattr(face_temporal, "landmark_series", boom)
    monkeypatch.setattr(acoustic_features, "compute", boom)
    degraded = pipeline.run_pipeline(_clip())
    for w in degraded["windows"]:
        assert not w["face_temporal"]["available"] and w["face_temporal"]["reason"]
        assert w["sync"]["sync_score"] is None and w["sync"]["status"] == "N/A" and w["sync"]["reason"]
        assert not w["acoustic"]["available"] and "failed" in w["acoustic"]["reason"]
        assert all(v is None for v in w["acoustic"]["features"].values())      # N/A, not 0
    assert degraded["video"]["sync"]["sync_score"] is None
    assert any("Facial temporal analysis N/A" in x for x in degraded["warnings"])
    # core SPEC outputs identical with or without the optional features
    for k in ("D_video", "S_video", "R_video", "band"):
        assert degraded["video"][k] == base["video"][k]
