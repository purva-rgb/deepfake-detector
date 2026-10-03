"""Map the raw report from `src.pipeline.run_pipeline` to the JSON the React frontend consumes.

Presentation only: no scores are computed here. Every number is copied from the report. Two derived values
reuse existing project code instead of new logic:
  * `visual.score` / `audio.score`: `fusion.top_k_mean` (the SPEC section 7 aggregation already used for the
    deepfake score) over the per-window scores of windows that pass the reliability gate.
  * band names: `fusion.band` with the configured thresholds.
Unavailable values are `null` (N/A), never 0. Nothing here exposes file paths, model names or timings.
"""
from __future__ import annotations

import base64
import math
from pathlib import Path

from src import config, fusion

# categories whose per-window features are summarised for the "Detailed analysis" section (mfcc* omitted: not user-facing)
_ACOUSTIC_KEYS = ("pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean", "spectral_centroid_mean",
                  "speech_ratio", "silence_ratio")
_MODALITY_OUT = {"visual": "visual", "audio": "audio", "text": "scam"}


def clean(obj):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> python scalars."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:  # noqa: BLE001
            return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _band(x):
    return "N/A" if x is None else fusion.band(x)


def _gated(windows, score_key, rel_key):
    return [w[score_key] for w in windows
            if w.get(score_key) is not None and w.get(rel_key) is not None and w[rel_key] >= config.R_GATE]


def _modality(rep, name):
    """visual / audio block: score (video-level, None = N/A), reliability, availability, reason."""
    skey, rkey, vkey = ("S_visual", "R_v", "R_v") if name == "visual" else ("S_audio", "R_a", "R_a")
    v, ws = rep["video"], rep["windows"]
    rel = v.get(vkey)
    vals = _gated(ws, skey, rkey)
    score = fusion.top_k_mean(vals, len(ws)) if vals and rel is not None and rel >= config.R_GATE else None
    reason = None
    if score is None:
        if name == "audio" and not rep["meta"].get("has_audio", True):
            reason = "No audio track"
        elif all(w.get(skey) is None for w in ws):
            reason = "No clear face" if name == "visual" else "No clear speech"
        elif rel is None or rel < config.R_GATE:
            reason = "Face quality too low" if name == "visual" else "Audio quality too low"
        else:
            reason = "Not enough reliable segments"
        if any(("head failed" in w or "head could not be loaded" in w) and name in w.lower() for w in rep["warnings"]):
            reason = "Analysis unavailable"
    return {"available": score is not None, "score": score, "reliability": rel, "reason": reason,
            "low_confidence": bool(rep["low_confidence"].get(name))}


def _semantic(rep):
    v = rep["video"]
    cats = [{"id": c["category"], "label": c["label"], "found": bool(c["found"]),
             "matches": [{"text": m["text"], "at": m["t"]} for m in c["matches"]]} for c in rep["checklist"]]
    score = v.get("S") if rep["transcript_ok"] else None
    reason = None
    if score is None:
        if not rep["meta"].get("has_audio", True):
            reason = "No audio track"
        elif not rep["transcript"]:
            reason = "No speech detected"
        elif not rep["transcript_ok"]:
            reason = "Speech too unclear to analyse"
        else:
            reason = "No speech in analysed segments"
    return {"available": score is not None, "score": score, "reliability": v.get("R_t"), "reason": reason,
            "categories": cats if rep["transcript_ok"] else [{**c, "found": False, "matches": []} for c in cats],
            "categories_available": bool(rep["transcript_ok"])}


def _mean_features(per_window, keys=None):
    """Mean of each numeric feature over windows where the feature group was available (None values skipped)."""
    ok = [w for w in per_window if w.get("available")]
    if not ok:
        return {}
    names = keys or list(ok[0]["features"].keys())
    out = {}
    for k in names:
        vals = [w["features"].get(k) for w in ok if isinstance(w["features"].get(k), (int, float))]
        out[k] = (sum(vals) / len(vals)) if vals else None
    return out


def _support_block(rep, key, keys=None):
    per = [w[key] for w in rep["windows"]]
    n_ok = sum(1 for p in per if p.get("available"))
    reasons = sorted({p.get("reason", "") for p in per if not p.get("available") and p.get("reason")})
    return {"available": n_ok > 0, "windows_available": n_ok, "windows_total": len(per),
            "aggregation": "mean over available windows", "features": _mean_features(per, keys),
            "reason": None if n_ok else ("; ".join(reasons) or "Not available"), "supporting_only": True}


def _sync(rep):
    sy = rep["video"]["sync"]
    ok = sy.get("sync_score") is not None
    return {"available": ok, "score": sy.get("sync_score"), "best_lag_ms": sy.get("best_lag_ms"),
            "reliability": sy.get("reliability"), "status": sy.get("status"),
            "reason": None if ok else (sy.get("reason") or "Not analysed"),
            "windows_used": sy.get("n_valid_windows", 0), "supporting_only": True}


def _thumb_data_uri(path):
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _evidence(rep):
    out = []
    for i, e in enumerate(rep["evidence"], 1):
        out.append({"rank": i, "start": e["t_start"], "end": e["t_end"], "modality": _MODALITY_OUT[e["modality"]],
                    "score": e["score"], "reliability": e["reliability"], "reason": e["reason"],
                    "text": e.get("text") or None, "thumbnail": _thumb_data_uri(e.get("thumbnail_path"))})
    return out


def _transcript(rep):
    labels = {c["category"]: c["label"] for c in rep["checklist"]}
    segs = []
    for s in rep["transcript"]:
        matches = [{"start": h["start"], "end": h["end"], "text": h["text"], "category": h["category"],
                    "label": labels.get(h["category"], h["category"]), "suppressed": bool(h["suppressed"]),
                    "note": h["reason"] or None} for h in s["hits"]]
        segs.append({"start": s["start"], "end": s["end"], "text": s["text"], "confidence": s["conf"],
                     "matches": matches})
    return {"transcript": " ".join(s["text"].strip() for s in rep["transcript"]) if rep["transcript"] else "",
            "transcript_segments": segs}


def _windows(rep):
    labels = {c["category"]: c["label"] for c in rep["checklist"]}
    out = []
    for w in rep["windows"]:
        ac, ft, sy = w["acoustic"], w["face_temporal"], w["sync"]
        out.append({
            "start": w["start"], "end": w["end"],
            "media": {"score": w["D"], "band": _band(w["D"])},
            "scam": {"score": w["S"], "band": _band(w["S"])},
            "visual": {"score": w["S_visual"], "reliability": w["R_v"]},
            "audio": {"score": w["S_audio"], "reliability": w["R_a"]},
            "semantic": {"score": w["S"], "reliability": w["R_t"],
                         "categories": [labels.get(c, c) for c in w["categories"]]},
            "faces": {"frames_with_face": w["n_faces"], "frames": w["n_frames"]},
            "acoustic": {"available": bool(ac["available"]), "reason": ac.get("reason") or None,
                         "features": {k: ac["features"].get(k) for k in _ACOUSTIC_KEYS}},
            "facial_temporal": {"available": bool(ft["available"]), "reason": ft.get("reason") or None,
                                "features": ft.get("features", {}) if ft["available"] else {}},
            "av_sync": {"available": sy.get("sync_score") is not None, "score": sy.get("sync_score"),
                        "best_lag_ms": sy.get("best_lag_ms"), "reliability": sy.get("reliability"),
                        "reason": None if sy.get("sync_score") is not None else (sy.get("reason") or None)},
        })
    return out


# --- warnings: user-facing wording for known pipeline notes; anything unrecognised goes to technical notes ---------
def _split_warnings(rep):
    friendly, technical = [], []
    n_w = len(rep["windows"])
    for w in rep["warnings"]:
        if w.startswith("AV mismatch (M) is not fused") or w.startswith("English only") or w.startswith("AV sync N/A"):
            continue                                    # restated elsewhere in the UI (AV sync card / footer)
        if w.startswith("No face detected in"):
            friendly.append("No face was detected in " + w.split("No face detected in ")[1].split(" windows")[0]
                            + " segments, so visual analysis is unavailable there.")
        elif w.startswith("Face too small"):
            friendly.append("The face was small, blurred or low-confidence in "
                            + w.split(" in ")[-1].split(" windows")[0] + " segments; those were not used for the visual score.")
        elif w.startswith("Audio quality is low"):
            friendly.append("Audio quality is low, so the voice analysis is less trustworthy.")
        elif w.startswith("Face quality is low overall"):
            friendly.append("Face quality is low overall, so the visual analysis is less trustworthy.")
        elif w.startswith("Low-confidence visual"):
            friendly.append("Visual analysis is low-confidence and counts for less in the result.")
        elif w.startswith("Low-confidence voice"):
            friendly.append("Voice analysis is low-confidence and counts for less in the result.")
        elif w.startswith("Total media reliability"):
            friendly.append("Media evidence was limited, so the deepfake score is capped and cannot reach HIGH on its own.")
        elif w.startswith("Transcript reliability"):
            friendly.append("Speech was too unclear to check for scam language.")
        elif w.startswith("More than one face"):
            friendly.append(w)
        elif w.startswith("Facial temporal features N/A in every window"):
            technical.append("Facial temporal features were unavailable in every segment (no usable landmarks).")
        else:
            technical.append(w)
    if technical and any(t for t in technical if "failed" in t.lower() or "could not be loaded" in t.lower()):
        friendly.append("Part of the analysis could not be completed; affected results are shown as N/A.")
    return list(dict.fromkeys(friendly)), list(dict.fromkeys(technical))


def build_response(rep: dict) -> dict:
    """Raw report -> API JSON. Pure function; safe to unit test with a saved report."""
    v, meta = rep["video"], rep["meta"]
    visual, audio, semantic = _modality(rep, "visual"), _modality(rep, "audio"), _semantic(rep)
    media_score = v.get("D")
    media_reason = None
    if media_score is None:
        media_reason = ("Face and voice analysis were both unavailable or unreliable"
                        if not visual["available"] and not audio["available"] else "Not enough reliable evidence")
    scam_score = v.get("S")
    friendly, technical = _split_warnings(rep)
    n_dur = max((w["end"] for w in rep["windows"]), default=0.0)
    cov = v["coverage"]
    resp = {
        "status": "ok",
        "analysis_id": rep["sha1"],
        "video": {"duration_s": meta.get("duration"), "analysed_s": n_dur, "has_audio": bool(meta.get("has_audio")),
                  "truncated": bool(meta.get("duration") and meta["duration"] > config.MAX_VIDEO_S + 0.5),
                  "windows": len(rep["windows"]), "window_s": config.WINDOW_S, "hop_s": config.HOP_S},
        "insufficient_evidence": bool(v["insufficient"]),
        "media_risk": {"score": media_score, "band": _band(media_score), "reason": media_reason,
                       "windows_flagged": cov["n_windows_D_ge_0.5"], "windows_scored": cov["n_windows_with_D"]},
        "scam_risk": {"score": scam_score, "band": _band(scam_score),
                      "reason": None if scam_score is not None else semantic["reason"]},
        "visual": visual,
        "audio": audio,
        "semantic": semantic,
        "acoustic": _support_block(rep, "acoustic", list(_ACOUSTIC_KEYS)),
        "facial_temporal": _support_block(rep, "face_temporal"),
        "av_sync": _sync(rep),
        "reliability": {"visual": v.get("R_v"), "audio": v.get("R_a"), "transcript": v.get("R_t")},
        "reliability_gate": config.R_GATE,
        "warnings": friendly,
        "technical_notes": technical,
        "evidence": _evidence(rep),
        "timeline": [{"start": w["start"], "end": w["end"], "media": w["media"], "scam": w["scam"]}
                     for w in _windows(rep)],
        "windows": _windows(rep),
        **_transcript(rep),
        "transcript_available": bool(rep["transcript"]),
        "scam_analysis_available": bool(rep["transcript_ok"]),
    }
    return clean(resp)
