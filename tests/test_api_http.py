"""HTTP surface of the API wrapper. Does not run the ML pipeline."""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pytest

os.environ["WARM_MODELS"] = "0"

from fastapi.testclient import TestClient

from api.main import app

DEMO = Path("outputs/reports/demo_report.json")
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models_loaded" in body
    assert body["analysing"] is False


def test_api_health_alias():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_progress_idle():
    r = client.get("/api/progress")
    assert r.status_code == 200
    assert r.json()["running"] is False


def test_analyze_rejects_wrong_type():
    r = client.post("/api/analyze", files={"video": ("notes.txt", BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 415
    assert r.json()["detail"]["code"] == "unsupported_type"


def test_analyze_rejects_empty():
    r = client.post("/api/analyze", files={"video": ("empty.mp4", BytesIO(b""), "video/mp4")})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "empty_file"


@pytest.mark.skipif(not DEMO.exists(), reason="no saved real report")
def test_demo_returns_real_saved_report():
    r = client.get("/api/demo")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["analysis_id"]
    assert "media_risk" in body and "scam_risk" in body
    assert body["media_risk"]["score"] is None or 0.0 <= body["media_risk"]["score"] <= 1.0
