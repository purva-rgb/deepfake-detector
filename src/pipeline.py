"""End-to-end analysis of ONE video: run_pipeline(path, progress_callback) -> report dict.

Stages: ingest -> faces -> xception -> audio (wav2vec2) -> heads -> whisper -> scam rules -> fusion ->
aggregation -> evidence. Every model call is wrapped: failure => N/A + warning, pipeline continues.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import cv2
import joblib
import numpy as np

from . import acoustic_features, av_sync, config, face_temporal, fusion, models, reliability, scam, semantic
from .features import extract_features

THUMB_DIR = config.OUT_DIR / "thumbs"
_STATE: dict = {}


def mmss(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def load_heads() -> dict:
    """Load LR heads + scalers once. Missing/corrupt head => that modality is N/A (warning)."""
    if "heads" in _STATE:
        return _STATE["heads"]
    heads = {"visual": None, "audio": None, "meta": {}, "warnings": []}
    try:
        heads["meta"] = json.loads((config.MODELS_DIR / "head_meta.json").read_text())
    except Exception as e:  # noqa: BLE001
        heads["warnings"].append(f"head_meta.json unreadable ({e}); default fusion weights used.")
    for k in ("visual", "audio"):
        try:
            heads[k] = (joblib.load(config.MODELS_DIR / f"{k}_scaler.pkl"), joblib.load(config.MODELS_DIR / f"{k}_lr.pkl"))
        except Exception as e:  # noqa: BLE001
            heads["warnings"].append(f"{k} head could not be loaded ({type(e).__name__}); {k} analysis N/A.")
    _STATE["heads"] = heads
    return heads


def load_all(progress_callback=None) -> dict:
    """Warm every model once. Returns load times (s); failures recorded, never raised."""
    if "load_times" in _STATE:
        return _STATE["load_times"]
    cb = progress_callback or (lambda *a, **k: None)
    times = {}
    for name, fn in (("xception", models.get_xception), ("wav2vec2", models.get_wav2vec2),
                     ("whisper", models.get_whisper), ("face_detector", models.get_face_detector)):
        cb(f"Loading {name}", 0.0)
        t = time.time()
        try:
            fn()
            times[name] = time.time() - t
        except Exception as e:  # noqa: BLE001
            times[name] = f"FAILED: {type(e).__name__}: {e}"
    t = time.time()
    load_heads()
    _STATE["rules"] = scam.load_rules()
    times["heads+rules"] = time.time() - t
    _STATE["load_times"] = times
    return times


def _rules():
    if "rules" not in _STATE:
        _STATE["rules"] = scam.load_rules()
    return _STATE["rules"]


def _file_hash(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def run_pipeline(path, progress_callback=None) -> dict:
    cb = progress_callback or (lambda stage, frac=0.0: None)
    T0 = time.time()
    timings: dict = {}
    warnings: list[str] = []

    cb("Loading models", 0.0)
    t = time.time()
    load_times = load_all()
    timings["model_load (once per process)"] = time.time() - t
    heads = load_heads()
    warnings += heads["warnings"]
    for k, v in load_times.items():
        if isinstance(v, str):
            warnings.append(f"Model '{k}' failed to load: {v}")
    rules = _rules()
    hmeta = heads["meta"]
    weights = {k: float(hmeta.get(k, {}).get("fusion_weight", 1.0)) for k in ("visual", "audio")}
    low_conf = {k: bool(hmeta.get(k, {}).get("low_confidence", False)) for k in ("visual", "audio")}
    audio_layer = hmeta.get("audio", {}).get("layer", config.W2V_LAYERS[1])
    layer_idx = list(config.W2V_LAYERS).index(audio_layer) if audio_layer in config.W2V_LAYERS else 0

    # ------------------------------------------------ ingest + feature extraction
    t = time.time()
    cb("Ingest: reading video, extracting faces / frames", 0.02)
    feats = extract_features(
        path, keep_crops=True, timings=timings,
        progress=lambda stage, frac: cb({"faces": "Detecting faces", "xception": "Xception embeddings",
                                         "audio": "Wav2Vec2 audio embeddings"}.get(stage, stage), frac),
    )
    timings["extract_total"] = time.time() - t
    warnings += feats["warnings"]
    meta, W, C, A = feats["meta"], feats["windows"], feats["crops"], feats["audio"]
    nw = len(W["start"])
    if nw == 0:
        raise RuntimeError("Video has no analysable duration.")

    # ------------------------------------------------ facial landmarks (supporting temporal features + sync)
    cb("Facial landmarks (temporal features)", 0.0)
    t = time.time()
    fseries = None
    try:
        fseries = face_temporal.landmark_series(
            path, max_seconds=config.MAX_VIDEO_S,
            progress=lambda f: cb("Facial landmarks (temporal features)", f))
        if fseries.n_multi:
            warnings.append(f"More than one face in {fseries.n_multi} sampled frames: the largest face was used.")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Facial temporal analysis N/A ({type(e).__name__}: {e}); temporal face features and AV sync unavailable.")
    timings["landmarks"] = time.time() - t

    # ------------------------------------------------ heads
    cb("Scoring visual and audio heads", 0.0)
    t = time.time()
    crop_p = np.zeros(len(C["win"]), np.float32)
    visual_available = False
    if heads["visual"] is not None and C["ok"] and C["emb"] is not None and len(C["emb"]) > 0:
        try:
            sc, lr = heads["visual"]
            crop_p = lr.predict_proba(sc.transform(C["emb"]))[:, 1].astype(np.float32)
            visual_available = True
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Visual head failed ({type(e).__name__}: {e}); visual analysis N/A.")
    # stage usable if extraction + head worked (or there were simply no faces at all -> R_v = 0 per window)
    visual_stage_ok = bool(C["ok"]) and heads["visual"] is not None and (visual_available or len(C["win"]) == 0)
    S_visual = [None] * nw
    R_v = [None] * nw
    best_crop = [None] * nw
    for wi in range(nw):
        if not visual_stage_ok:
            continue
        idx = np.where(C["win"] == wi)[0]
        R_v[wi] = reliability.visual_reliability(C["conf"][idx], C["size"][idx], C["blur"][idx],
                                                 int(W["nframes"][wi]), int(W["nfaces"][wi]))
        if len(idx) and visual_available:
            p = crop_p[idx]
            S_visual[wi] = float(0.5 * p.mean() + 0.5 * p.max())          # SPEC section 4.3
            best_crop[wi] = int(idx[int(np.argmax(p))])

    S_audio = [None] * nw
    R_a = [None] * nw
    if heads["audio"] is not None and A["ok"]:
        try:
            sc, lr = heads["audio"]
            ok = np.where(A["valid"])[0]
            if len(ok):
                p = lr.predict_proba(sc.transform(A["emb"][ok, layer_idx]))[:, 1]
                for wi, pi in zip(ok, p):
                    S_audio[wi] = float(pi)
                    R_a[wi] = reliability.audio_reliability(float(A["speech"][wi]), float(A["clip"][wi]),
                                                            float(A["snr"][wi]))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Audio head failed ({type(e).__name__}: {e}); audio analysis N/A.")
            S_audio = [None] * nw
            R_a = [None] * nw
    timings["heads"] = time.time() - t

    # ------------------------------------------------ text
    cb("Transcribing speech (Whisper, English only)", 0.0)
    t = time.time()
    asr = semantic.transcribe(A["wave"])
    timings["whisper"] = time.time() - t
    if asr["warning"]:
        warnings.append(asr["warning"])
    t = time.time()
    segments = asr["segments"]
    r_t_video = asr["r_t"]
    text_ok = asr["ok"] and r_t_video is not None and r_t_video >= config.R_GATE
    if asr["ok"] and not text_ok:
        warnings.append(f"Transcript reliability {r_t_video:.2f} < {config.R_GATE}: text analysis dropped (N/A).")
    seg_hits = []   # per segment: list of Hit (active + suppressed, for the UI)
    for s in segments:
        try:
            s["hits"] = scam.scan(s["text"], rules, include_suppressed=True) if text_ok else []
        except Exception as e:  # noqa: BLE001
            s["hits"] = []
            warnings.append(f"Scam rule engine failed on a segment ({type(e).__name__}); text N/A for it.")
        s["conf"] = semantic.segment_confidence(s)
        seg_hits.append(s["hits"])
    S_scam = [None] * nw
    R_t = [None] * nw
    win_cats = [[] for _ in range(nw)]
    if text_ok:
        for wi in range(nw):
            ws, we = float(W["start"][wi]), float(W["end"][wi])
            segs = [s for s in segments if _overlap(ws, we, s["start"], s["end"]) > 0]
            if not segs:
                continue                               # no speech in this window -> text N/A (not 0)
            R_t[wi] = float(np.mean([s["conf"] for s in segs]))
            if R_t[wi] < config.R_GATE:
                R_t[wi] = None
                continue
            active = [h for s in segs for h in s["hits"] if not h.suppressed]
            sc_ = scam.score(active)
            S_scam[wi] = sc_["S"]
            win_cats[wi] = sc_["categories"]
    timings["scam_rules"] = time.time() - t

    # ------------------------------------------------ fusion
    cb("Fusing evidence", 0.5)
    t = time.time()
    win_res = []
    for wi in range(nw):
        r = fusion.fuse(S_visual[wi], R_v[wi], S_audio[wi], R_a[wi], S_scam[wi], M=None, weights=weights)
        r.update({"start": float(W["start"][wi]), "end": float(W["end"][wi]), "S_visual": S_visual[wi],
                  "S_audio": S_audio[wi], "R_v": R_v[wi], "R_a": R_a[wi], "R_t": R_t[wi],
                  "n_faces": int(W["nfaces"][wi]), "n_frames": int(W["nframes"][wi]),
                  "speech_ratio": None if np.isnan(A["speech"][wi]) else float(A["speech"][wi]),
                  "categories": win_cats[wi]})
        r.update(_supporting_features(fseries, A, wi, r))
        win_res.append(r)
    rv_vals = [x for x in R_v if x is not None]
    ra_vals = [x for x in R_a if x is not None]
    rt_vals = [x for x in R_t if x is not None]
    R_v_video = float(np.mean(rv_vals)) if rv_vals else None
    R_a_video = float(np.mean(ra_vals)) if ra_vals else None
    R_t_video = float(np.mean(rt_vals)) if rt_vals else None
    vid = fusion.aggregate_video(win_res, R_v_video, R_a_video, weights)
    sync_video = av_sync.summarise([w["sync"] for w in win_res])
    d_vals = [w["D"] for w in win_res if w["D"] is not None]
    coverage = {"n_windows": nw, "n_windows_with_D": len(d_vals),
                "n_windows_D_ge_0.5": int(sum(1 for d in d_vals if d >= 0.5)),
                "D_median": float(np.median(d_vals)) if d_vals else None,
                "D_max": float(max(d_vals)) if d_vals else None,
                "n_windows_sync_ok": int(sync_video.get("n_valid_windows", 0)),
                "n_windows_face_temporal_ok": int(sum(1 for w in win_res if w["face_temporal"]["available"])),
                "n_windows_acoustic_ok": int(sum(1 for w in win_res if w["acoustic"]["available"]))}
    timings["fusion"] = time.time() - t

    # ------------------------------------------------ evidence
    cb("Building evidence", 0.8)
    t = time.time()
    evidence = _build_evidence(path, W, C, crop_p, best_crop, win_res, segments, rules, low_conf, R_t_video)
    timings["evidence"] = time.time() - t

    # ------------------------------------------------ checklist + warnings panel
    cat_hits = {k: [] for k in rules.categories}
    suppressed = []
    for s in segments:
        for h in s.get("hits", []):
            row = {"text": h.text, "segment_text": s["text"], "t": s["start"], "rule": h.rule_id,
                   "strength": h.strength, "reason": h.reason}
            (suppressed if h.suppressed else cat_hits[h.category]).append(row)
    checklist = [{"category": k, "label": v, "found": bool(cat_hits[k]), "matches": cat_hits[k]}
                 for k, v in rules.categories.items()]

    panel_warnings = list(dict.fromkeys(warnings))
    n_noface = sum(1 for w in win_res if w["R_v"] == 0.0)
    if visual_stage_ok and n_noface:
        panel_warnings.append(f"No face detected in {n_noface}/{nw} windows (R_v = 0 there; visual score N/A).")
    n_lowv = sum(1 for w in win_res if w["R_v"] is not None and 0 < w["R_v"] < config.R_GATE)
    if n_lowv:
        panel_warnings.append(f"Face too small, blurred or low-confidence in {n_lowv}/{nw} windows (visual dropped there).")
    if R_a_video is not None and R_a_video < 0.4:
        panel_warnings.append("Audio quality is low (little speech, clipping or noisy): audio score is less trustworthy.")
    if R_v_video is not None and R_v_video < 0.4:
        panel_warnings.append("Face quality is low overall: visual score is less trustworthy.")
    for k, lab in (("visual", "visual"), ("audio", "voice")):
        if low_conf[k]:
            panel_warnings.append(f"Low-confidence {lab} analysis: head val AUC {hmeta[k]['val_auc']:.2f} < "
                                  f"{config.LOW_CONF_AUC}; its fusion weight is x{config.LOW_CONF_WEIGHT}.")
    if vid["R_sum"] < config.R_SUM_CAP and vid["D_video"] is not None:
        panel_warnings.append(f"Total media reliability {vid['R_sum']:.2f} < {config.R_SUM_CAP}: deepfake score capped at "
                              f"{config.CAP_VALUE}, so media evidence alone cannot produce HIGH.")
    panel_warnings.append("AV mismatch (M) is not fused: the lightweight sync score below is supporting evidence only "
                          "(dubbed / voice-over content can legitimately be out of sync).")
    if sync_video["status"] != "ok":
        panel_warnings.append(f"AV sync N/A: {sync_video['reason']}")
    if coverage["n_windows_face_temporal_ok"] == 0:
        panel_warnings.append("Facial temporal features N/A in every window (no usable landmarks).")
    panel_warnings.append("English only: other languages are not analysed reliably.")

    timings["total (excl. model load)"] = time.time() - T0 - timings["model_load (once per process)"]
    cb("Done", 1.0)
    return {
        "path": str(path), "sha1": _file_hash(path), "meta": meta,
        "video": {**vid, "R_v": R_v_video, "R_a": R_a_video, "R_t": R_t_video,
                  "M": None, "D": vid["D_video"], "S": vid["S_video"], "R": vid["R_video"],
                  "media_band": fusion.band(vid["D_video"]) if vid["D_video"] is not None else "N/A",
                  "scam_band": fusion.band(vid["S_video"]) if vid["S_video"] is not None else "N/A",
                  "sync": sync_video, "coverage": coverage},
        "windows": win_res, "evidence": evidence, "checklist": checklist,
        "suppressed_matches": suppressed,
        "transcript": [{"start": s["start"], "end": s["end"], "text": s["text"], "conf": s["conf"],
                        "hits": [{"start": h.start, "end": h.end, "category": h.category, "suppressed": h.suppressed,
                                  "reason": h.reason, "text": h.text} for h in s.get("hits", [])]} for s in segments],
        "transcript_ok": bool(text_ok), "warnings": panel_warnings, "timings": timings,
        "low_confidence": low_conf, "head_meta": hmeta, "weights": weights,
        "crop_thumbs_available": len(C["img"]), "face_backend": meta.get("face_backend"),
    }


def _build_evidence(path, W, C, crop_p, best_crop, win_res, segments, rules, low_conf, r_t_video) -> list[dict]:
    """Evidence records {t_start,t_end,modality,detector,score,reliability,text,thumbnail_path,reason},
    ranked by score*reliability, adjacent same-modality intervals merged, deduped, top 5."""
    recs = []
    for wi, w in enumerate(win_res):
        if (w["S_visual"] is not None and w["S_visual"] >= config.EVIDENCE_MIN_SCORE
                and w["R_v"] is not None and w["R_v"] >= config.R_GATE):
            crop_i = best_crop[wi]
            recs.append({"t_start": w["start"], "t_end": w["end"], "modality": "visual", "detector": "xception+LR",
                         "score": w["S_visual"], "reliability": w["R_v"], "text": "", "crop": crop_i,
                         "low_conf": low_conf["visual"]})
        if (w["S_audio"] is not None and w["S_audio"] >= config.EVIDENCE_MIN_SCORE
                and w["R_a"] is not None and w["R_a"] >= config.R_GATE):
            recs.append({"t_start": w["start"], "t_end": w["end"], "modality": "audio", "detector": "wav2vec2+LR",
                         "score": w["S_audio"], "reliability": w["R_a"], "text": "", "crop": None,
                         "low_conf": low_conf["audio"]})
    for s in segments:
        for h in s.get("hits", []):
            if h.suppressed:
                continue
            recs.append({"t_start": s["start"], "t_end": s["end"], "modality": "text", "detector": f"rule:{h.rule_id}",
                         "score": h.strength, "reliability": s["conf"], "text": s["text"], "crop": None,
                         "category": h.category, "match": h.text, "low_conf": False})
    # merge adjacent / overlapping intervals of the same modality (+ same rule for text)
    recs.sort(key=lambda r: (r["modality"], r.get("detector", "") if r["modality"] == "text" else "", r["t_start"]))
    merged = []
    for r in recs:
        m = merged[-1] if merged else None
        same = m and m["modality"] == r["modality"] and (r["modality"] != "text" or m["detector"] == r["detector"])
        if same and r["t_start"] <= m["t_end"] + 1e-6 and (r["modality"] != "text" or r["match"] == m["match"]):
            m["t_end"] = max(m["t_end"], r["t_end"])
            if r["score"] * r["reliability"] > m["score"] * m["reliability"]:
                for k in ("score", "reliability", "crop", "text"):
                    m[k] = r[k]
        else:
            merged.append(dict(r))
    seen, uniq = set(), []
    for r in merged:
        k = (r["modality"], round(r["t_start"], 2), round(r["t_end"], 2), r["text"], r.get("match"))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    uniq.sort(key=lambda r: r["score"] * r["reliability"], reverse=True)
    top = uniq[:5]

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    sha = _file_hash(path)
    out = []
    for i, r in enumerate(top):
        # thumbnail: crop with the highest visual score inside the interval (any face crop as fallback)
        idx = [j for j in range(len(C["win"])) if C["t"][j] >= r["t_start"] - 1e-6 and C["t"][j] <= r["t_end"] + 1e-6]
        thumb = None
        if idx and len(C["img"]) == len(C["win"]):
            j = max(idx, key=lambda j: crop_p[j]) if len(crop_p) == len(C["win"]) else idx[0]
            thumb = C["img"][j]
        thumb_path = None
        if thumb is not None:
            thumb_path = str(THUMB_DIR / f"{sha}_{i}.jpg")
            cv2.imwrite(thumb_path, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        span = f"{mmss(r['t_start'])}-{mmss(r['t_end'])}"
        lc = " (low-confidence head)" if r.get("low_conf") else ""
        if r["modality"] == "visual":
            reason = (f"Face appearance at {span} looks more synthetic-like than real to the visual detector "
                      f"(score {round(100 * r['score'])}/100*, face reliability {round(100 * r['reliability'])}/100){lc}.")
        elif r["modality"] == "audio":
            reason = (f"Voice at {span} has characteristics the audio detector associates with cloned or synthetic speech "
                      f"(score {round(100 * r['score'])}/100*, audio reliability {round(100 * r['reliability'])}/100){lc}.")
        else:
            reason = (f"Scam-style language at {span}: \"{r['match']}\" ({rules.categories[r['category']]}); "
                      f"transcript confidence {round(100 * r['reliability'])}/100.")
        out.append({"t_start": r["t_start"], "t_end": r["t_end"], "modality": r["modality"], "detector": r["detector"],
                    "score": float(r["score"]), "reliability": float(r["reliability"]), "text": r["text"],
                    "thumbnail_path": thumb_path, "reason": reason, "rank_value": float(r["score"] * r["reliability"])})
    return out



def _supporting_features(fseries, A, wi, r) -> dict:
    """Per-window supporting evidence (not fused): facial temporal stats, handcrafted acoustic stats, AV sync.
    Each is N/A with a reason when unavailable; never raises."""
    ws, we = r["start"], r["end"]
    try:
        ft_w = face_temporal.window_stats(fseries, ws, we)
    except Exception as e:  # noqa: BLE001
        ft_w = {"features": {}, "available": False, "reason": f"facial temporal failed ({type(e).__name__})",
                "n_valid": 0, "n_frames": 0, "valid_frac": 0.0}
    wave = A["wave"]
    sr = config.SAMPLE_RATE
    ac = None
    if wave is not None and bool(A["valid"][wi]):
        try:
            ac = acoustic_features.compute(wave[int(ws * sr): int(we * sr)])
        except Exception as e:  # noqa: BLE001
            ac = {"features": {k: float("nan") for k in acoustic_features.FEATURE_NAMES}, "available": False,
                  "reason": f"acoustic features failed ({type(e).__name__})"}
    if ac is None:
        ac = {"features": {k: float("nan") for k in acoustic_features.FEATURE_NAMES}, "available": False,
              "reason": "no usable audio in this window"}
    sync_w = av_sync.sync_window(fseries, wave, ws, we, r["speech_ratio"], r["R_a"])
    return {"face_temporal": _nan_to_none(ft_w), "acoustic": _nan_to_none(ac), "sync": sync_w}


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
