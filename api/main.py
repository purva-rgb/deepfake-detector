"""FastAPI wrapper around the existing pipeline.

Run (from the project root, in the project's Python environment):
    python -m uvicorn api.main:app --reload --port 8000

Endpoints
    GET  /health         liveness + whether the frozen models are loaded
    GET  /api/progress   real stage name of the analysis currently running (from the pipeline's progress callback)
    POST /api/analyze    multipart/form-data, field `video` (mp4 / mov / webm) -> analysis JSON
    GET  /api/demo       a previously saved REAL report (outputs/reports/demo_report.json), if present
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.pipeline import load_all, run_pipeline

from .serializer import build_response

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ALLOWED_EXT = {".mp4", ".mov", ".webm"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
DEMO_REPORT = config.OUT_DIR / "reports" / "demo_report.json"

_model_lock = threading.Lock()      # the pipeline's models are shared, process-wide objects: one use at a time
_state_lock = threading.Lock()
_state = {"models_loaded": False, "analysing": False, "stage": None, "fraction": 0.0, "started": None}


def _warm_models():
    try:
        with _model_lock:
            times = load_all()
        _state["models_loaded"] = True
        log.info("models ready: %s", {k: (round(v, 1) if not isinstance(v, str) else v) for k, v in times.items()})
    except Exception:  # noqa: BLE001
        log.exception("model warm-up failed; they will be retried on the first request")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("WARM_MODELS", "1") == "1":
        threading.Thread(target=_warm_models, daemon=True, name="warm-models").start()
    yield


app = FastAPI(title="Is That Really Them? - analysis API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


def _err(status: int, code: str, message: str):
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": _state["models_loaded"], "analysing": _state["analysing"]}


@app.get("/api/progress")
def progress():
    return {"running": _state["analysing"], "stage": _state["stage"], "fraction": _state["fraction"]}


@app.get("/api/demo")
def demo():
    if not DEMO_REPORT.exists():
        raise _err(404, "no_demo", "No saved report is available.")
    try:
        return build_response(json.loads(DEMO_REPORT.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        log.exception("could not read demo report")
        raise _err(500, "demo_unreadable", "The saved report could not be read.")


def _save_upload(video: UploadFile, dest: Path) -> int:
    size = 0
    with open(dest, "wb") as f:
        while chunk := video.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise _err(413, "too_large", f"Video is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            f.write(chunk)
    return size


@app.post("/api/analyze")
def analyze(video: UploadFile = File(...)):
    ext = Path(video.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise _err(415, "unsupported_type", "Please upload an MP4, MOV or WebM video.")
    with _state_lock:
        if _state["analysing"]:
            raise _err(429, "busy", "Another video is being analysed. Please try again in a moment.")
        _state.update(analysing=True, stage="Uploading", fraction=0.0, started=time.time())
    tmpdir = Path(tempfile.mkdtemp(prefix="verisight_"))
    try:
        path = tmpdir / f"upload{ext}"
        if _save_upload(video, path) == 0:
            raise _err(400, "empty_file", "The uploaded file is empty.")

        def cb(stage, frac=0.0):
            _state["stage"], _state["fraction"] = stage, float(frac)

        try:
            with _model_lock:                               # waits if start-up model loading is still running
                rep = run_pipeline(path, cb)
            _state["models_loaded"] = True
        except HTTPException:
            raise
        except RuntimeError as e:                           # e.g. "Video has no analysable duration."
            log.warning("analysis rejected: %s", e)
            raise _err(422, "unreadable_video", "This video could not be analysed. It may be corrupt, empty or in an unsupported format.")
        except Exception:  # noqa: BLE001
            log.exception("analysis failed")
            raise _err(500, "analysis_failed", "Something went wrong while analysing this video. Please try another file.")
        log.info("timings (s): %s", {k: round(v, 2) for k, v in rep["timings"].items()})   # console only
        return build_response(rep)
    finally:
        _state.update(analysing=False, stage=None, fraction=0.0)
        video.file.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
