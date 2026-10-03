"""Run the full pipeline on clips from the CLI and print the report + per-stage timings.

Usage: python scripts/run_pipeline_cli.py <clip_id|path> [<clip_id|path> ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.pipeline import load_all, mmss, run_pipeline


def fmt(x, pct=True):
    return "N/A" if x is None else (f"{round(100 * x)}/100*" if pct else f"{x:.2f}")


def print_report(rep, truth=""):
    v = rep["video"]
    print("=" * 100)
    print(f"FILE: {rep['path']}  {truth}")
    print(f"duration={rep['meta']['duration']:.1f}s fps={rep['meta']['fps']:.1f} {rep['meta']['width']}x{rep['meta']['height']} "
          f"audio={rep['meta']['has_audio']} face_backend={rep['face_backend']} vad={rep['meta']['vad']}")
    print(f"\nRISK BAND: {v['band']}   (R_video={fmt(v['R_video'])})")
    print(f"  Deepfake D = {fmt(v['D_video'])}   Scam S = {fmt(v['S_video'])}   AV mismatch M = {fmt(v['M_video'])} (not fused)")
    print(f"  Reliability: R_v={fmt(v['R_v'], False)}  R_a={fmt(v['R_a'], False)}  R_t={fmt(v['R_t'], False)}  "
          f"(R_sum over surviving media={v['R_sum']:.2f}, synergy={v['synergy']})")
    print(f"  Media/deepfake band: {v['media_band']}   Scam band: {v['scam_band']}   (reported separately; overall band above = SPEC fusion)")
    cov = v["coverage"]
    print(f"  Coverage: {cov['n_windows']} windows | D available in {cov['n_windows_with_D']} | D>=0.5 in {cov['n_windows_D_ge_0.5']} | "
          f"D median={fmt(cov['D_median'])} max={fmt(cov['D_max'])} | face-temporal ok {cov['n_windows_face_temporal_ok']} | "
          f"acoustic ok {cov['n_windows_acoustic_ok']} | sync ok {cov['n_windows_sync_ok']}")
    sy = v["sync"]
    print("  AV sync (supporting, not fused): " + (f"score={sy['sync_score']:.2f} lag={sy['best_lag_ms']:.0f} ms reliability={sy['reliability']:.2f} "
          f"({sy['n_valid_windows']} windows)" if sy["sync_score"] is not None else f"N/A - {sy['reason']}"))
    print("\nWINDOWS (t, S_visual, R_v, S_audio, R_a, S_scam, R_t, D, R, band):")
    for w in rep["windows"]:
        print(f"  {mmss(w['start'])}-{mmss(w['end'])}  Sv={fmt(w['S_visual'])} Rv={fmt(w['R_v'], False)} "
              f"Sa={fmt(w['S_audio'])} Ra={fmt(w['R_a'], False)} Ss={fmt(w['S'])} Rt={fmt(w['R_t'], False)} "
              f"D={fmt(w['D'])} R={fmt(w['R'])} -> {w['band']}")
    print("\nSUPPORTING FEATURES per window (not fused):")
    for w in rep["windows"]:
        ft, ac, sy_w = w["face_temporal"], w["acoustic"], w["sync"]
        fts = (f"MAR mean/std={ft['features']['mar_mean']:.2f}/{ft['features']['mar_std']:.2f} blinks={ft['features']['blink_count']:.0f} "
               f"yaw std={ft['features']['yaw_std']:.1f}" if ft["available"] else f"face-temporal N/A ({ft['reason']})")
        pm = ac["features"].get("pitch_mean")
        acs = (f"pitch={'N/A' if pm is None else f'{pm:.0f}Hz'} rms={ac['features']['rms_mean']:.3f} speech={ac['features']['speech_ratio']:.2f} "
               f"silence={ac['features']['silence_ratio']:.2f}" if ac["available"] else f"acoustic N/A ({ac['reason']})")
        sys_ = (f"sync={sy_w['sync_score']:.2f}@{sy_w['best_lag_ms']:.0f}ms" if sy_w["sync_score"] is not None else f"sync N/A ({sy_w['reason']})")
        print(f"  {mmss(w['start'])}-{mmss(w['end'])}  {fts} | {acs} | {sys_}")
    print("\nEVIDENCE (top 5 by score*reliability):")
    for i, e in enumerate(rep["evidence"], 1):
        print(f"  #{i} [{e['modality']}] {mmss(e['t_start'])}-{mmss(e['t_end'])} score*rel={e['rank_value']:.2f} "
              f"thumb={'yes' if e['thumbnail_path'] else 'no'}\n      {e['reason']}")
    if not rep["evidence"]:
        print("  (none)")
    print("\nSCAM CHECKLIST:")
    for c in rep["checklist"]:
        print(f"  [{'X' if c['found'] else ' '}] {c['label']}" + (f"  <- {[m['text'] for m in c['matches']]}" if c["found"] else ""))
    print(f"\nTRANSCRIPT (ok={rep['transcript_ok']}):")
    for s in rep["transcript"]:
        print(f"  [{mmss(s['start'])}-{mmss(s['end'])}] {s['text']}  (conf {s['conf']:.2f})")
    if rep["suppressed_matches"]:
        print("  suppressed by context guard:", [(m["text"], m["reason"]) for m in rep["suppressed_matches"]])
    print("\nWARNINGS:")
    for w in rep["warnings"]:
        print("  -", w)
    print("\nPER-STAGE TIMINGS (s):")
    for k, t in rep["timings"].items():
        print(f"  {k:32s} {t:7.2f}")


def main():
    sub = pd.read_csv(config.SUBSET_CSV).set_index("clip_id")
    t = time.time()
    print("loading models once ...")
    lt = load_all()
    print("load times:", {k: (round(v, 1) if not isinstance(v, str) else v) for k, v in lt.items()}, f"total {time.time()-t:.1f}s")
    for arg in sys.argv[1:]:
        truth = ""
        if arg in sub.index:
            r = sub.loc[arg]
            path = config.DATASET_ROOT / r.rel_path
            truth = f"[ground truth: {r.category}, split={r.split}, video_label={r.video_label}, audio_label={r.audio_label}]"
        else:
            path = Path(arg)
        stages = []
        rep = run_pipeline(path, progress_callback=lambda s, f=0.0: stages.append(s) if not stages or stages[-1] != s else None)
        print_report(rep, truth)
        print("stage sequence:", " > ".join(stages))


if __name__ == "__main__":
    main()
