"""Evaluate arbitrary local videos WITHOUT training on them (inference only through the existing pipeline).

Usage:  python scripts/eval_real_world.py                       # every mp4/mov/webm in data/real_world_eval/
        python scripts/eval_real_world.py my.mp4 other.mov      # explicit files and/or folders
Writes outputs/real_world_eval/results.csv and results.json (git-ignored). Nothing is added to any manifest.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config, fusion
from src.pipeline import load_all, mmss, run_pipeline

EVAL_DIR = config.ROOT / "data" / "real_world_eval"
OUT = config.OUT_DIR / "real_world_eval"
EXTS = {".mp4", ".mov", ".webm"}


def pct(x):
    return "N/A" if x is None else f"{round(100 * x)}/100*"


def collect(args):
    files = []
    for a in args or [str(EVAL_DIR)]:
        p = Path(a)
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix.lower() in EXTS)
        elif p.exists():
            files.append(p)
        else:
            print(f"not found: {p}")
    return files


def summarise(path, rep):
    v = rep["video"]
    W = rep["windows"]
    cats = [c["label"] for c in rep["checklist"] if c["found"]]
    return {
        "file": Path(path).name,
        "duration_s": round(rep["meta"]["duration"], 1),
        "windows": len(W),
        "final_band": v["band"],
        "final_score": v["R_video"],
        "media_deepfake_D": v["D_video"],
        "media_band": v["media_band"],
        "visual_score": fusion.top_k_mean([w["S_visual"] for w in W], len(W)),
        "audio_score": fusion.top_k_mean([w["S_audio"] for w in W], len(W)),
        "scam_S": v["S_video"],
        "scam_band": v["scam_band"],
        "scam_categories": "; ".join(cats),
        "sync_score": v["sync"]["sync_score"],
        "sync_lag_ms": v["sync"]["best_lag_ms"],
        "sync_status": "ok" if v["sync"]["sync_score"] is not None else f"N/A: {v['sync']['reason']}",
        "R_v": v["R_v"], "R_a": v["R_a"], "R_t": v["R_t"],
        "D_windows>=0.5": f"{v['coverage']['n_windows_D_ge_0.5']}/{v['coverage']['n_windows_with_D']}",
        "evidence": " | ".join(f"[{e['modality']}] {mmss(e['t_start'])}-{mmss(e['t_end'])} {e['reason']}" for e in rep["evidence"]),
        "n_warnings": len(rep["warnings"]),
        "seconds": round(rep["timings"]["total (excl. model load)"], 1),
    }


def main():
    files = collect(sys.argv[1:])
    if not files:
        print(f"No videos found. Put .mp4/.mov/.webm files in {EVAL_DIR} or pass paths. (Files there are git-ignored.)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    load_all()
    rows, full = [], {}
    for f in files:
        t = time.time()
        try:
            rep = run_pipeline(f)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {f.name}: {type(e).__name__}: {e}")
            rows.append({"file": f.name, "final_band": f"ERROR: {type(e).__name__}: {e}"})
            continue
        row = summarise(f, rep)
        rows.append(row)
        full[f.name] = {"summary": row, "warnings": rep["warnings"], "checklist": [
            {"label": c["label"], "found": c["found"], "matches": [m["text"] for m in c["matches"]]} for c in rep["checklist"]],
            "transcript": [{"t": f"{mmss(s['start'])}-{mmss(s['end'])}", "text": s["text"], "conf": s["conf"]} for s in rep["transcript"]],
            "suppressed_by_guard": [{"text": m["text"], "reason": m["reason"]} for m in rep["suppressed_matches"]]}
        print("=" * 100)
        print(f"{f.name}  ({row['duration_s']}s, {row['windows']} windows, {time.time()-t:.1f}s)")
        print(f"  FINAL: {row['final_band']} ({pct(row['final_score'])})")
        print(f"  media/deepfake D={pct(row['media_deepfake_D'])} [{row['media_band']}]  visual={pct(row['visual_score'])}  audio={pct(row['audio_score'])}")
        print(f"  scam S={pct(row['scam_S'])} [{row['scam_band']}]  categories: {row['scam_categories'] or 'none'}")
        sy = row["sync_score"]
        print(f"  AV sync: {'N/A - ' + row['sync_status'][5:] if sy is None else f'{sy:.2f} @ {row['sync_lag_ms']:.0f} ms'}"
              f"   reliability R_v={row['R_v'] if row['R_v'] is None else round(row['R_v'], 2)} "
              f"R_a={row['R_a'] if row['R_a'] is None else round(row['R_a'], 2)} R_t={row['R_t'] if row['R_t'] is None else round(row['R_t'], 2)}")
        print(f"  windows with D>=0.5: {row['D_windows>=0.5']}")
        for s in rep["transcript"]:
            print(f"  transcript [{mmss(s['start'])}-{mmss(s['end'])}] {s['text']}")
        for m in rep["suppressed_matches"]:
            print(f"  guard suppressed: '{m['text']}' ({m['reason']})")
        for e in rep["evidence"]:
            print(f"  evidence [{e['modality']}] {mmss(e['t_start'])}-{mmss(e['t_end'])}: {e['reason']}")
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'results.csv'} and results.json")


if __name__ == "__main__":
    main()
