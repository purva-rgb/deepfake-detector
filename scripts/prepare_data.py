"""M1: discover videos, parse meta_data.csv, build identity groups, group-split, write data/subset.csv.

Manifests only: no video is copied. Labels come from the category folder name.
Folder names / demographics / paths are NEVER written as features (only rel_path for loading).
"""
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from src import config, media

SEED = 42
PER_CATEGORY = 40
VAL_FRACTION = 0.20
CAP_PER_GROUP = 2
CATEGORIES = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    root = config.DATASET_ROOT
    print("DATASET_ROOT =", root)

    # ---- 1. find videos recursively
    t = time.time()
    all_files = [p for p in root.rglob("*") if p.is_file()]
    ext_counts = Counter(p.suffix.lower() for p in all_files)
    print("file extensions:", dict(ext_counts), f"({time.time()-t:.1f}s)")
    top_counts = Counter()
    disk = {}
    for p in all_files:
        if p.suffix.lower() != ".mp4":
            continue
        rel = p.relative_to(root).as_posix()
        top = rel.split("/")[0]
        top_counts[top] += 1
        if top in CATEGORIES:
            disk[rel] = p
    print("mp4 per top-level folder:", dict(top_counts))
    ignored = {k: v for k, v in top_counts.items() if k not in CATEGORIES}
    if ignored:
        print("IGNORED folders (not a category; not assumed to hold usable videos):", ignored)
    print("category videos on disk:", len(disk))

    # ---- 2. meta_data.csv (schema not assumed)
    meta = pd.read_csv(root / "meta_data.csv", dtype=str, keep_default_na=False)
    print("meta_data.csv columns:", meta.columns.tolist())
    print("meta_data.csv rows:", len(meta))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(meta.head(4).to_string())
    file_col = max(meta.columns, key=lambda c: meta[c].str.lower().str.endswith(".mp4").mean())
    dir_col = max(meta.columns, key=lambda c: (meta[c].str.contains("/").mean() if c != file_col else -1))
    print(f"detected filename column={file_col!r} ({meta[file_col].str.endswith('.mp4').mean():.3f} end in .mp4); "
          f"directory column={dir_col!r}")
    id_cols = [c for c in meta.columns if c.lower().startswith(("source", "target"))]
    print("identity columns:", id_cols)

    def to_key(d, f):
        d = d.replace("\\", "/").strip("/")
        for c in CATEGORIES:
            i = d.find(c)
            if i >= 0:
                return d[i:] + "/" + f
        return None

    meta["_key"] = [to_key(d, f) for d, f in zip(meta[dir_col], meta[file_col])]
    print("csv rows with unparseable path:", int(meta["_key"].isna().sum()))
    dup = int(meta["_key"].duplicated().sum())
    print("csv duplicate keys:", dup)
    meta = meta.drop_duplicates("_key")
    m = meta[meta["_key"].isin(disk)].copy()
    print(f"csv rows matched to a file on disk: {len(m)}; disk files without csv row: "
          f"{len(set(disk) - set(meta['_key']))}; csv rows without file: {len(meta) - len(m)}")
    m["category"] = m["_key"].str.split("/").str[0]

    # ---- 3. labels (from category folder only)
    m["video_label"] = m["category"].str.startswith("FakeVideo").astype(int)
    m["audio_label"] = m["category"].str.endswith("FakeAudio").astype(int)

    # ---- 4. identity groups
    def clean(v):
        v = str(v).strip()
        return None if v in ("", "-", "nan", "None") else v

    def components():
        uf = UF()
        for _, r in m[id_cols].iterrows():
            ids = [clean(r[c]) for c in id_cols]
            ids = [i for i in ids if i]
            for i in ids[1:]:
                uf.union(ids[0], i)
            if ids:
                uf.find(ids[0])
        return [("cc_" + uf.find(clean(r[id_cols[0]]))) if clean(r[id_cols[0]]) else "none"
                for _, r in m[id_cols].iterrows()]

    src_col = next(c for c in id_cols if c.lower().startswith("source"))
    method = "connected components over " + "/".join(id_cols)
    m["group"] = components()
    sizes = m["group"].value_counts()
    frac = sizes.iloc[0] / len(m)
    print(f"[group method 1] {method}: {len(sizes)} groups, largest={sizes.iloc[0]} clips = {frac:.1%}")
    if not frac < 0.40:
        print("ASSERT FAILED (largest group >= 40%). Regrouping by source-speaker column alone.")
        method = f"source-speaker column '{src_col}' alone"
        m["group"] = "src_" + m[src_col].astype(str)
        sizes = m["group"].value_counts()
        frac = sizes.iloc[0] / len(m)
        print(f"[group method 2] {method}: {len(sizes)} groups, largest={sizes.iloc[0]} clips = {frac:.1%}")
        if not frac < 0.40:
            print("ASSERT FAILED again. STOP. Top groups:\n", sizes.head(10))
            print("Per-category counts:\n", m["category"].value_counts())
            sys.exit(3)
    print("GROUPING METHOD USED:", method)
    print("group sizes: min/median/max =", sizes.min(), sizes.median(), sizes.max())

    # ---- 5. split whole groups 80/20
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
    tr_g, va_g = set(m[m.split == "train"].group), set(m[m.split == "val"].group)
    assert not (tr_g & va_g), "group leakage between train and val"
    print(f"split over ALL matched clips: train {int((m.split=='train').sum())} / val {int((m.split=='val').sum())}; "
          f"groups train {len(tr_g)} / val {len(va_g)}; overlap = {len(tr_g & va_g)}  [ASSERT OK]")

    # ---- 6. pick 40/category (32 train + 8 val), cap per group, verify decodable with audio
    n_val_pick = int(round(PER_CATEGORY * VAL_FRACTION))
    quotas = {"train": PER_CATEGORY - n_val_pick, "val": n_val_pick}
    rows, rejected = [], Counter()
    for cat in CATEGORIES:
        for split, q in quotas.items():
            pool = m[(m.category == cat) & (m.split == split)].sample(frac=1.0, random_state=SEED)
            per_group, got = defaultdict(int), 0
            # first pass honours cap; second pass relaxes it only if the pool is too small
            for cap_g in (CAP_PER_GROUP, 10**9):
                for _, r in pool.iterrows():
                    if got >= q:
                        break
                    if per_group[r.group] >= cap_g or any(x["_key"] == r["_key"] for x in rows):
                        continue
                    p = disk[r["_key"]]
                    cap = cv2.VideoCapture(str(p))
                    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 and cap.read()[0]
                    cap.release()
                    pr = media.probe(p) if ok else {"has_audio": False}
                    if not ok or not pr["has_audio"]:
                        rejected[(cat, "no-video" if not ok else "no-audio")] += 1
                        continue
                    per_group[r.group] += 1
                    got += 1
                    rows.append(r)
            if got < q:
                print(f"NOTE: {cat}/{split}: only {got}/{q} clips available")
    sub = pd.DataFrame(rows)
    out = pd.DataFrame({
        "clip_id": [f"c{i:03d}" for i in range(len(sub))],
        "rel_path": sub["_key"].values,
        "category": sub["category"].values,
        "video_label": sub["video_label"].values,
        "audio_label": sub["audio_label"].values,
        "group": sub["group"].values,
        "split": sub["split"].values,
    })
    assert not (set(out[out.split == "train"].group) & set(out[out.split == "val"].group)), "overlap in subset"
    assert not out.rel_path.duplicated().any()
    config.DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(config.SUBSET_CSV, index=False)
    print("rejected during verification:", dict(rejected) or "none")
    print(f"wrote {config.SUBSET_CSV} ({len(out)} clips; all verified OpenCV-readable .mp4 with an ffmpeg audio stream)")
    print("\ncounts per split x category:")
    print(pd.crosstab(out.category, out.split, margins=True).to_string())
    print("\ngroups in subset: train", out[out.split == "train"].group.nunique(),
          "val", out[out.split == "val"].group.nunique(), "| zero train/val group overlap [ASSERT OK]")
    print("grouping method used:", method)


if __name__ == "__main__":
    main()
