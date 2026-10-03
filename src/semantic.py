"""ASR (faster-Whisper, English only) -> timestamped segments + text reliability R_t."""
from __future__ import annotations

import math

import numpy as np

from . import config, models


def transcribe(wave_16k: np.ndarray | None) -> dict:
    """Returns {ok, segments:[{start,end,text,avg_logprob,no_speech_prob}], r_t, warning}.
    N/A (ok=False, r_t=None) on any failure or when there is no speech."""
    if wave_16k is None or len(wave_16k) < config.SAMPLE_RATE // 2:
        return {"ok": False, "segments": [], "r_t": None, "warning": "No audio: transcript N/A."}
    try:
        model = models.get_whisper()
        segs, _info = model.transcribe(
            wave_16k.astype(np.float32), language="en", beam_size=1, condition_on_previous_text=False,
            temperature=0.0,
        )
        out = []
        for s in segs:
            text = s.text.strip()
            # Whisper's own silence heuristics: drop likely hallucinations on non-speech
            if not text or (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0):
                continue
            out.append({"start": float(s.start), "end": float(s.end), "text": text,
                        "avg_logprob": float(s.avg_logprob), "no_speech_prob": float(s.no_speech_prob)})
        if not out:
            return {"ok": False, "segments": [], "r_t": None, "warning": "No speech detected: transcript N/A."}
        return {"ok": True, "segments": out, "r_t": text_reliability(out), "warning": ""}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "segments": [], "r_t": None,
                "warning": f"Whisper failed ({type(e).__name__}: {e}); transcript N/A."}


def segment_confidence(seg: dict) -> float:
    """exp(avg_logprob) = geometric-mean token probability in [0,1], damped by no_speech_prob."""
    return float(min(1.0, max(0.0, math.exp(seg["avg_logprob"]) * (1.0 - seg["no_speech_prob"]))))


def text_reliability(segments: list[dict]) -> float:
    return float(np.mean([segment_confidence(s) for s in segments])) if segments else 0.0
