"""Ablation step 1: extract clip-level handcrafted features (src/acoustic_features.py, src/face_temporal.py) for
every clip in outputs/manifests/dev_800.csv. Clip feature = nan-mean over the clip's 4 s windows (2 s hop) of the
per-window features that were available; NaN (never 0) if no window had them. Labels/splits are NOT read here.
Resumable; writes outputs/ablation/handcrafted.csv.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import acoustic_features, config, face_temporal, media

OUT = config.OUT_DIR / "ablation"
CSV = OUT / "handcrafted.csv"
MANIFEST = config.OUT_DIR / "manifests" / "dev_800.csv"


def clip_features(path: Path) -> dict:
    row: dict = {}
    info = media.probe(path)
    dur = float(info.get("duration") or 0.0)
    wins = media.make_windows(dur, config.WINDOW_S, config.HOP_S)
    # ---- acoustic
    ac_rows = []
    try:
        wave = media.decode_audio(path)
        if wave is not None and len(wave):
            sr = config.SAMPLE_RATE
            for ws, we in wins:
                r = acoustic_features.compute(wave[int(ws * sr): int(we * sr)], sr)
                if r["available"]:
                    ac_rows.append(r["features"])
    except Exception as e:  # noqa: BLE001
        row["acoustic_err"] = f"{type(e).__name__}: {e}"
    for k in acoustic_features.FEATURE_NAMES:
        v = [r[k] for r in ac_rows if np.isfinite(r[k])]
        row["ac_" + k] = float(np.mean(v)) if v else float("nan")
    row["acoustic_windows_ok"] = len(ac_rows)
    # ---- facial temporal
    ft_rows = []
    try:
        series = face_temporal.landmark_series(path, max_seconds=config.MAX_VIDEO_S)
        for ws, we in wins:
            r = face_temporal.window_stats(series, ws, we)
            if r["available"]:
                ft_rows.append(r["features"])
    except Exception as e:  # noqa: BLE001
        row["face_err"] = f"{type(e).__name__}: {e}"
    for k in face_temporal.STAT_NAMES:
        v = [r[k] for r in ft_rows if np.isfinite(r[k])]
        row["ft_" + k] = float(np.mean(v)) if v else float("nan")
    row["face_windows_ok"] = len(ft_rows)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Get-Date start:", time.strftime("%F %T"), flush=True)
    m = pd.read_csv(MANIFEST)
    done = pd.read_csv(CSV).set_index("clip_id") if CSV.exists() else pd.DataFrame()
    rows = [r.to_dict() | {"clip_id": cid} for cid, r in done.iterrows()] if len(done) else []
    have = set(done.index) if len(done) else set()
    t0 = time.time()
    for i, r in enumerate(m.itertuples()):
        if r.clip_id in have:
            continue
        row = {"clip_id": r.clip_id}
        try:
            row.update(clip_features(config.DATASET_ROOT / r.rel_path))
        except Exception as e:  # noqa: BLE001
            row["fatal_err"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(f"{i+1}/{len(m)}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print("done", len(rows), "Get-Date end:", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
