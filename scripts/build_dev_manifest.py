"""Expand the development subset to ~800 clips -> outputs/manifests/dev_800.csv  (manifest only, no copying).

Same discovery / CSV matching / source-speaker grouping / group split as scripts/prepare_data.py (seed 42).
The existing 160 clips (data/subset.csv) are kept verbatim (same clip_id, split, group) so their cached
embeddings are reused; only new clips are added. Deterministic (fixed seed).
Population = all category-folder clips of the existing train/val group split. No other held-out set is
defined anywhere in this project, so none can be excluded by identity (reported in the output).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from src import config, media

SEED = 42
VAL_FRACTION = 0.20
TARGET_PER_CATEGORY = 200      # -> 800 total
CAP_PER_GROUP = 2              # per (category, group), relaxed only if the pool is too small
CATEGORIES = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
OUT = config.OUT_DIR / "manifests" / "dev_800.csv"


def load_population():
    root = config.DATASET_ROOT
    disk = {}
    for p in root.rglob("*.mp4"):
        rel = p.relative_to(root).as_posix()
        if rel.split("/")[0] in CATEGORIES:
            disk[rel] = p
    meta = pd.read_csv(root / "meta_data.csv", dtype=str, keep_default_na=False)
    file_col = max(meta.columns, key=lambda c: meta[c].str.lower().str.endswith(".mp4").mean())
    dir_col = max(meta.columns, key=lambda c: (meta[c].str.contains("/").mean() if c != file_col else -1))
    src_col = next(c for c in meta.columns if c.lower().startswith("source"))

    def to_key(d, f):
        d = d.replace("\\", "/").strip("/")
        for c in CATEGORIES:
            i = d.find(c)
            if i >= 0:
                return d[i:] + "/" + f
        return None

    meta["_key"] = [to_key(d, f) for d, f in zip(meta[dir_col], meta[file_col])]
    meta = meta.drop_duplicates("_key")
    m = meta[meta["_key"].isin(disk)].copy()
    m["category"] = m["_key"].str.split("/").str[0]
    m["video_label"] = m["category"].str.startswith("FakeVideo").astype(int)
    m["audio_label"] = m["category"].str.endswith("FakeAudio").astype(int)
    m["group"] = "src_" + m[src_col].astype(str)       # method that passed the <40% assertion in M1
    return m, disk


def split_groups(m):
    sizes = m["group"].value_counts()
    rng = np.random.RandomState(SEED)
    groups = sorted(m["group"].unique())
    rng.shuffle(groups)
    val_groups, n_val = set(), 0
    for g in groups:
        if n_val >= VAL_FRACTION * len(m):
            break
        val_groups.add(g)
        n_val += int(sizes[g])
    m["split"] = np.where(m["group"].isin(val_groups), "val", "train")
    return m


def main():
    m, disk = load_population()
    m = split_groups(m)
    print(f"population: {len(m)} clips, {m.group.nunique()} source-speaker groups "
          f"(largest {m.group.value_counts().iloc[0]} = {m.group.value_counts().iloc[0]/len(m):.1%})")

    old = pd.read_csv(config.SUBSET_CSV)
    key2row = {r["_key"]: r for _, r in m.iterrows()}
    for r in old.itertuples():       # the split/grouping must reproduce the existing 160-clip manifest exactly
        mr = key2row[r.rel_path]
        assert mr["split"] == r.split and mr["group"] == r.group, f"split mismatch for {r.rel_path}"
    print(f"REPRODUCIBILITY ASSERT OK: all {len(old)} existing clips have identical group and split under the recomputed split")

    rows = [dict(clip_id=r.clip_id, rel_path=r.rel_path, category=r.category, video_label=r.video_label,
                 audio_label=r.audio_label, group=r.group, split=r.split, origin="existing_160") for r in old.itertuples()]
    have = {r["rel_path"] for r in rows}
    n_val_cat = int(round(TARGET_PER_CATEGORY * VAL_FRACTION))
    quotas = {"train": TARGET_PER_CATEGORY - n_val_cat, "val": n_val_cat}
    rejected, next_id = Counter(), len(old)
    for cat in CATEGORIES:
        for split, q in quotas.items():
            cur = [r for r in rows if r["category"] == cat and r["split"] == split]
            per_group = defaultdict(int)
            for r in cur:
                per_group[r["group"]] += 1
            got = len(cur)
            pool = m[(m.category == cat) & (m.split == split)].sample(frac=1.0, random_state=SEED)
            for cap_g in (CAP_PER_GROUP, 10**9):
                for _, r in pool.iterrows():
                    if got >= q:
                        break
                    if r["_key"] in have or per_group[r.group] >= cap_g:
                        continue
                    p = disk[r["_key"]]
                    cap = cv2.VideoCapture(str(p))
                    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 and cap.read()[0]
                    cap.release()
                    has_audio = media.probe(p)["has_audio"] if ok else False
                    if not ok or not has_audio:
                        rejected[(cat, "no-video" if not ok else "no-audio")] += 1
                        continue
                    per_group[r.group] += 1
                    got += 1
                    have.add(r["_key"])
                    rows.append(dict(clip_id=f"c{next_id:04d}" if next_id >= 1000 else f"c{next_id:03d}",
                                     rel_path=r["_key"], category=cat, video_label=int(r.video_label),
                                     audio_label=int(r.audio_label), group=r.group, split=split, origin="new"))
                    next_id += 1
            if got < q:
                print(f"NOTE: {cat}/{split}: only {got}/{q} available")
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("rejected during verification:", dict(rejected) or "none")
    print(f"wrote {OUT}")

    # ------------------------------------------------------------ verification
    print("\n=== MANIFEST VERIFICATION ===")
    print("total samples:", len(out), "| unique rel_path:", out.rel_path.nunique(), "| unique clip_id:", out.clip_id.nunique())
    assert out.rel_path.is_unique and out.clip_id.is_unique
    print("origin:", out.origin.value_counts().to_dict())
    print("\nsamples per class x split:")
    print(pd.crosstab(out.category, out.split, margins=True).to_string())
    tr, va = set(out[out.split == "train"].group), set(out[out.split == "val"].group)
    print(f"\ngroups: total {out.group.nunique()} | train {len(tr)} | val {len(va)} | train/val overlap = {len(tr & va)}")
    assert not (tr & va), "GROUP LEAKAGE"
    # same source-speaker group may hold RVRA+RVFA clips: they must be on one side
    sides = out.groupby("group")["split"].nunique()
    assert (sides == 1).all()
    print("each group on exactly one side: OK")
    print("max clips per group:", out.group.value_counts().max())
    print("video_label counts:", out.video_label.value_counts().to_dict(), "| audio_label counts:", out.audio_label.value_counts().to_dict())
    print("TEST SET: no test split/frozen set exists in this project (cut in the brief) and no test manifest was found or opened; "
          "nothing to intersect against. Population is limited to the existing train/val group split.")


if __name__ == "__main__":
    main()
