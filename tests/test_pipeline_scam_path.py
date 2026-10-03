"""Integration test of the text->fusion->UI-data path using a MOCKED transcript (not a real ASR result).

The media/models run for real on one validation clip; only semantic.transcribe is replaced with a scripted
scam transcript so the scam-score, fusion and evidence code can be exercised end to end.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, pipeline, semantic


def _clip():
    sub = pd.read_csv(config.SUBSET_CSV)
    r = sub[sub.split == "val"].iloc[0]
    return config.DATASET_ROOT / r.rel_path


@pytest.mark.skipif(not config.SUBSET_CSV.exists(), reason="needs data/subset.csv")
def test_mock_scam_transcript(monkeypatch):
    fake = {"ok": True, "warning": "", "r_t": 0.9, "segments": [
        {"start": 0.0, "end": 1.5, "text": "Guaranteed returns, act now.", "avg_logprob": -0.1, "no_speech_prob": 0.0},
        {"start": 1.5, "end": 3.9, "text": "Never share your OTP with anyone.", "avg_logprob": -0.1, "no_speech_prob": 0.0},
    ]}
    monkeypatch.setattr(semantic, "transcribe", lambda wave: fake)
    rep = pipeline.run_pipeline(_clip())
    # guaranteed returns 0.85 + 0.10 (2 categories: financial_promise, urgency) = 0.95
    assert rep["video"]["S"] == pytest.approx(0.95)
    found = {c["category"] for c in rep["checklist"] if c["found"]}
    assert found == {"financial_promise", "urgency"}          # OTP warning suppressed
    assert any(m["text"].lower().endswith("share your otp") for m in rep["suppressed_matches"])
    assert any(e["modality"] == "text" for e in rep["evidence"])
    assert rep["video"]["band"] in ("MEDIUM", "HIGH")          # S = 0.95 can never be LOW
