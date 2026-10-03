"""M2: resumable feature extraction. Reads only data/subset.csv; one .npz per clip in outputs/embeddings/.

Usage:  python scripts/extract.py --limit 3 --workers 1
        python scripts/extract.py --workers 2 > logs/extract.log 2>&1      (background)
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config


def _init_worker(threads):
    import torch

    torch.set_num_threads(threads)


def process_clip(row: dict) -> dict:
    """Runs in a worker. Never raises; returns a status dict."""
    from src.features import extract_features

    out = config.EMB_DIR / f"{row['clip_id']}.npz"
    if out.exists():
        return {"clip_id": row["clip_id"], "status": "skipped"}
    t0 = time.time()
    try:
        timings = {}
        f = extract_features(config.DATASET_ROOT / row["rel_path"], timings=timings)
        c, a, w, m = f["crops"], f["audio"], f["windows"], f["meta"]
        tmp = out.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            clip_id=row["clip_id"],
            video_label=int(row["video_label"]),
            audio_label=int(row["audio_label"]),
            meta_json=json.dumps({**m, "warnings": f["warnings"], "timings": timings}),
            # windows
            win_start=w["start"], win_end=w["end"], win_nframes=w["nframes"], win_nfaces=w["nfaces"],
            # per-crop visual
            crop_emb=(c["emb"] if c["emb"] is not None else np.zeros((0, 2048), np.float32)),
            crop_win=c["win"], crop_t=c["t"], crop_conf=c["conf"], crop_size=c["size"], crop_blur=c["blur"],
            visual_ok=bool(c["ok"]),
            # per-window audio (NaN / False where N/A)
            aud_emb=a["emb"], aud_valid=a["valid"], aud_speech=a["speech"], aud_clip=a["clip"], aud_snr=a["snr"],
            audio_ok=bool(a["ok"]),
        )
        tmp.replace(out)
        return {"clip_id": row["clip_id"], "status": "ok", "sec": time.time() - t0, "timings": timings,
                "n_crops": int(len(c["win"])), "n_win": int(len(w["start"])),
                "aud_valid": int(a["valid"].sum()), "emb_shape": tuple(c["emb"].shape) if c["emb"] is not None else None,
                "aud_shape": tuple(a["emb"].shape), "warnings": f["warnings"], "face_backend": m["face_backend"],
                "vad": m["vad"]}
    except Exception as e:  # noqa: BLE001
        return {"clip_id": row["clip_id"], "status": "failed", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3), "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4, help="torch threads per worker")
    ap.add_argument("--manifest", default=str(config.SUBSET_CSV), help="manifest CSV (default data/subset.csv)")
    ap.add_argument("--fail-log", default=str(config.LOG_DIR / "extract_failures.log"))
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df.sample(frac=1.0, random_state=0)  # mixed order so a partial run stays balanced
    if args.limit:
        df = df.head(args.limit)
    config.EMB_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    rows = df.to_dict("records")
    todo = [r for r in rows if not (config.EMB_DIR / f"{r['clip_id']}.npz").exists()]
    print(f"{len(rows)} clips in manifest, {len(todo)} to do, {args.workers} worker(s)", flush=True)
    fail_log = Path(args.fail_log)
    t_start, done, n_ok, n_fail = time.time(), 0, 0, 0

    def handle(res):
        nonlocal done, n_ok, n_fail
        done += 1
        if res["status"] == "ok":
            n_ok += 1
            tm = res["timings"]
            print(f"[{done}/{len(todo)}] {res['clip_id']} ok {res['sec']:.1f}s "
                  f"(faces {tm.get('faces',0):.1f}s xcep {tm.get('xception',0):.1f}s audio {tm.get('audio',0):.1f}s) "
                  f"crops={res['n_crops']} windows={res['n_win']} aud_valid={res['aud_valid']} "
                  f"crop_emb{res['emb_shape']} aud_emb{res['aud_shape']} face={res['face_backend']} vad={res['vad']}"
                  + (f" WARN={res['warnings']}" if res["warnings"] else ""), flush=True)
        elif res["status"] == "failed":
            n_fail += 1
            print(f"[{done}/{len(todo)}] {res['clip_id']} FAILED {res['error']}", flush=True)
            with open(fail_log, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%F %T')} {res['clip_id']} {res['error']}\n{res.get('trace','')}\n")

    if args.workers <= 1:
        _init_worker(args.threads)
        for r in todo:
            handle(process_clip(r))
    else:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(args.threads,)) as pool:
            for res in pool.imap_unordered(process_clip, todo):
                handle(res)
    el = time.time() - t_start
    print(f"FINISHED: ok={n_ok} failed={n_fail} elapsed={el:.0f}s ({el/max(done,1):.1f}s/clip wall)", flush=True)


if __name__ == "__main__":
    main()
