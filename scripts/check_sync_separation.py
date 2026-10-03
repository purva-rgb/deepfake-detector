"""Diagnostic (SPEC 12.7): does the lightweight AV-sync score separate real from fake clips on the dev manifest?
Samples N clips per category (seed 0) from outputs/manifests/dev_800.csv, reuses cached per-window speech/clipping/SNR
(outputs/embeddings) for the sync preconditions. Inference only: nothing is trained or tuned on these numbers.

Usage: python scripts/check_sync_separation.py [N_per_category]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import av_sync, config, face_temporal, media, reliability


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    m = pd.read_csv(config.ROOT / "outputs" / "manifests" / "dev_800.csv")
    sub = pd.concat([m[m.category == c].sample(min(n, (m.category == c).sum()), random_state=0) for c in sorted(m.category.unique())])
    rows = []
    t0 = time.time()
    for r in sub.itertuples():
        d = np.load(config.EMB_DIR / f"{r.clip_id}.npz")
        path = config.DATASET_ROOT / r.rel_path
        try:
            ser = face_temporal.landmark_series(path)
            wave = media.decode_audio(path, normalize=True)
            ws = []
            for wi in range(len(d["win_start"])):
                ra = reliability.audio_reliability(float(d["aud_speech"][wi]), float(d["aud_clip"][wi]), float(d["aud_snr"][wi]))
                ws.append(av_sync.sync_window(ser, wave, float(d["win_start"][wi]), float(d["win_end"][wi]),
                                              float(d["aud_speech"][wi]), ra))
            s = av_sync.summarise(ws)
        except Exception as e:  # noqa: BLE001
            s = {"sync_score": None, "reason": f"{type(e).__name__}: {e}", "n_valid_windows": 0}
        rows.append({"clip_id": r.clip_id, "category": r.category, "video_label": r.video_label, "audio_label": r.audio_label,
                     "sync": s["sync_score"], "lag_ms": s.get("best_lag_ms"), "reason": s.get("reason", "")})
    df = pd.DataFrame(rows)
    print(f"{len(df)} clips in {time.time()-t0:.0f}s; sync available for {df.sync.notna().sum()}")
    print("\nper category: n, available, mean / median sync_score")
    print(df.groupby("category").agg(n=("sync", "size"), avail=("sync", lambda x: x.notna().sum()),
                                     mean=("sync", "mean"), median=("sync", "median")).round(3).to_string())
    ok = df.dropna(subset=["sync"])
    print("\nunavailable reasons:", df[df.sync.isna()].reason.value_counts().to_dict())
    rv = ok[ok.category == "RealVideo-RealAudio"]
    for cat in ["RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
        o = ok[ok.category == cat]
        if len(rv) > 3 and len(o) > 3:
            auc = roc_auc_score(np.r_[np.zeros(len(rv)), np.ones(len(o))], -np.r_[rv.sync, o.sync])
            print(f"AUC (LOW sync => fake) RVRA vs {cat}: {auc:.3f}  (n={len(rv)} vs {len(o)})")
    anyfake = ok[ok.category != "RealVideo-RealAudio"]
    if len(rv) > 3:
        print(f"AUC RVRA vs all other categories: {roc_auc_score(np.r_[np.zeros(len(rv)), np.ones(len(anyfake))], -np.r_[rv.sync, anyfake.sync]):.3f}")
    print("0.5 = no separation. Diagnostic only; sync is NOT fused into the score.")


if __name__ == "__main__":
    main()
