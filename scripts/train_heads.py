"""M4: train logistic-regression heads on cached frozen embeddings. Fit on split=train, report split=val.

Visual rows = face crops (label video_label). Audio rows = windows (label audio_label).
StandardScaler + LogisticRegression(class_weight="balanced"); rows weighted 1/(rows in clip).
C in {0.03, 0.3, 1} (and Wav2Vec2 layer in {6, 9, 12}) chosen by clip-level val AUC.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src import config

CS = [0.03, 0.3, 1.0]
LAYER_IDX = {l: i for i, l in enumerate(config.W2V_LAYERS)}
NOTE = "val only, small subset, indicative"


def load_clips(manifest=None):
    sub = pd.read_csv(manifest or config.SUBSET_CSV)
    clips = []
    for r in sub.itertuples():
        f = config.EMB_DIR / f"{r.clip_id}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta_json"]))
        clips.append({"row": r, "d": {k: d[k] for k in d.files if k != "meta_json"}, "meta": meta})
    return clips


def visual_rows(clips, split):
    X, y, w, cid, win = [], [], [], [], []
    for c in clips:
        if c["row"].split != split or not bool(c["d"]["visual_ok"]) or len(c["d"]["crop_win"]) == 0:
            continue
        e = c["d"]["crop_emb"]
        n = len(e)
        X.append(e)
        y += [c["row"].video_label] * n
        w += [1.0 / n] * n
        cid += [c["row"].clip_id] * n
        win += list(c["d"]["crop_win"])
    return np.vstack(X), np.array(y), np.array(w), np.array(cid), np.array(win)


def audio_rows(clips, split, layer):
    X, y, w, cid, win = [], [], [], [], []
    for c in clips:
        if c["row"].split != split or not bool(c["d"]["audio_ok"]):
            continue
        ok = c["d"]["aud_valid"]
        idx = np.where(ok)[0]
        if len(idx) == 0:
            continue
        X.append(c["d"]["aud_emb"][idx, LAYER_IDX[layer]])
        y += [c["row"].audio_label] * len(idx)
        w += [1.0 / len(idx)] * len(idx)
        cid += [c["row"].clip_id] * len(idx)
        win += list(idx)
    return np.vstack(X), np.array(y), np.array(w), np.array(cid), np.array(win)


def fit(X, y, w, C):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    lr.fit(sc.transform(X), y, sample_weight=w)
    return sc, lr


def clip_scores(p, cid, win, visual):
    """Window score (visual: 0.5*mean(p)+0.5*max(p) over the window's crops; audio: p), then clip score =
    mean of the top-3 windows."""
    df = pd.DataFrame({"p": p, "cid": cid, "win": win})
    if visual:
        wdf = df.groupby(["cid", "win"])["p"].agg(lambda s: 0.5 * s.mean() + 0.5 * s.max()).reset_index()
    else:
        wdf = df
    return wdf.groupby("cid")["p"].apply(lambda s: float(np.mean(sorted(s, reverse=True)[:3])))


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def evaluate(label, scores, labels_by_clip, cats_by_clip):
    ids = list(scores.index)
    y = np.array([labels_by_clip[i] for i in ids])
    s = scores.values
    pred = (s >= 0.5).astype(int)
    out = {
        "n_clips": len(ids), "n_pos": int(y.sum()), "auc": safe_auc(y, s),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix_[[TN,FP],[FN,TP]]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    cat = np.array([cats_by_clip[i] for i in ids])
    out["per_category_auc"] = {}
    for c in sorted(set(cat)):
        # category c versus all val clips of the opposite label
        lab_c = int(y[cat == c][0])
        m = (cat == c) | (y != lab_c)
        out["per_category_auc"][c] = safe_auc(y[m], s[m])
    return out


def band_counts(scores_real):
    b = pd.cut(scores_real, [-1, config.BAND_LOW, config.BAND_HIGH, 2], right=False, labels=["LOW", "MEDIUM", "HIGH"])
    return {k: int(v) for k, v in b.value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).items()}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.SUBSET_CSV))
    args = ap.parse_args()
    print("Get-Date start:", time.strftime("%F %T"), "| manifest:", args.manifest)
    n_manifest = len(pd.read_csv(args.manifest))
    clips = load_clips(args.manifest)
    print(f"loaded {len(clips)} / {n_manifest} manifest clips with embeddings (missing = extraction failures: {n_manifest - len(clips)})")
    lab_v = {c["row"].clip_id: int(c["row"].video_label) for c in clips}
    lab_a = {c["row"].clip_id: int(c["row"].audio_label) for c in clips}
    cats = {c["row"].clip_id: c["row"].category for c in clips}
    groups = {c["row"].clip_id: c["row"].group for c in clips}
    config.MODELS_DIR.mkdir(exist_ok=True)
    dist = pd.DataFrame([{"category": c["row"].category, "split": c["row"].split,
                          "visual_usable": bool(c["d"]["visual_ok"]) and len(c["d"]["crop_win"]) > 0,
                          "audio_usable": bool(c["d"]["audio_ok"])} for c in clips])
    print("class distribution of usable clips (split x category):")
    print(pd.crosstab(dist.category, dist.split, margins=True).to_string())
    print("usable for visual head:", int(dist.visual_usable.sum()), "| usable for audio head:", int(dist.audio_usable.sum()))
    results, val_pred = {}, pd.DataFrame({"clip_id": list(lab_v)}).set_index("clip_id")

    # ------------------------------------------------------------------ visual
    print("\n=== VISUAL head (Xception embedding of face crops) ===")
    Xtr, ytr, wtr, ctr, wtr_win = visual_rows(clips, "train")
    Xva, yva, wva, cva, wiva = visual_rows(clips, "val")
    print(f"train rows {Xtr.shape} (clips {len(set(ctr))}), val rows {Xva.shape} (clips {len(set(cva))})")
    best = None
    for C in CS:
        sc, lr = fit(Xtr, ytr, wtr, C)
        cs_ = clip_scores(lr.predict_proba(sc.transform(Xva))[:, 1], cva, wiva, True)
        auc = safe_auc([lab_v[i] for i in cs_.index], cs_.values)
        print(f"  C={C:<5} val clip AUC={auc:.3f}")
        if best is None or auc > best[0] + 1e-9:
            best = (auc, C, sc, lr, cs_)
    auc, C, sc, lr, cs_ = best
    print(f"  -> chosen C={C} (val clip AUC {auc:.3f})")
    res = evaluate("visual", cs_, lab_v, cats)
    res.update({"C": C, "note": NOTE, "real_val_clips_by_band": band_counts(cs_[[i for i in cs_.index if lab_v[i] == 0]]),
                "low_confidence": bool(res["auc"] < config.LOW_CONF_AUC)})
    # grouped out-of-fold scores on TRAIN real clips (extra, uses train groups only)
    gtr = np.array([groups[i] for i in ctr])
    oof = np.full(len(ytr), np.nan)
    for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, gtr):
        s2, l2 = fit(Xtr[a], ytr[a], wtr[a], C)
        oof[b] = l2.predict_proba(s2.transform(Xtr[b]))[:, 1]
    oof_clip = clip_scores(oof, ctr, wtr_win, True)
    real_oof = oof_clip[[i for i in oof_clip.index if lab_v[i] == 0]]
    res["train_grouped_oof_clip_auc"] = safe_auc([lab_v[i] for i in oof_clip.index], oof_clip.values)
    res["real_train_oof_clips_by_band"] = band_counts(real_oof)
    results["visual"] = res
    joblib.dump(lr, config.MODELS_DIR / "visual_lr.pkl")
    joblib.dump(sc, config.MODELS_DIR / "visual_scaler.pkl")
    val_pred["visual_clip_score"] = cs_

    # ------------------------------------------------------------------ audio
    print("\n=== AUDIO head (Wav2Vec2 mean-pooled layer) ===")
    best = None
    for layer in config.W2V_LAYERS:
        Xtr, ytr, wtr, ctr, _ = audio_rows(clips, "train", layer)
        Xva, yva, wva, cva, wiva = audio_rows(clips, "val", layer)
        for C in CS:
            sc, lr = fit(Xtr, ytr, wtr, C)
            cs_ = clip_scores(lr.predict_proba(sc.transform(Xva))[:, 1], cva, wiva, False)
            auc = safe_auc([lab_a[i] for i in cs_.index], cs_.values)
            print(f"  layer={layer:<2} C={C:<5} val clip AUC={auc:.3f}   (train rows {Xtr.shape[0]}, val rows {Xva.shape[0]})")
            if best is None or auc > best[0] + 1e-9:
                best = (auc, layer, C, sc, lr, cs_)
    auc, layer, C, sc, lr, cs_ = best
    print(f"  -> chosen layer={layer} C={C} (val clip AUC {auc:.3f})")
    res = evaluate("audio", cs_, lab_a, cats)
    res.update({"layer": layer, "C": C, "note": NOTE,
                "real_val_clips_by_band": band_counts(cs_[[i for i in cs_.index if lab_a[i] == 0]]),
                "low_confidence": bool(res["auc"] < config.LOW_CONF_AUC)})
    Xtr, ytr, wtr, ctr, wtr_win = audio_rows(clips, "train", layer)
    gtr = np.array([groups[i] for i in ctr])
    oof = np.full(len(ytr), np.nan)
    for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, gtr):
        s2, l2 = fit(Xtr[a], ytr[a], wtr[a], C)
        oof[b] = l2.predict_proba(s2.transform(Xtr[b]))[:, 1]
    oof_clip = clip_scores(oof, ctr, wtr_win, False)
    real_oof = oof_clip[[i for i in oof_clip.index if lab_a[i] == 0]]
    res["train_grouped_oof_clip_auc"] = safe_auc([lab_a[i] for i in oof_clip.index], oof_clip.values)
    res["real_train_oof_clips_by_band"] = band_counts(real_oof)
    results["audio"] = res
    joblib.dump(lr, config.MODELS_DIR / "audio_lr.pkl")
    joblib.dump(sc, config.MODELS_DIR / "audio_scaler.pkl")
    val_pred["audio_clip_score"] = cs_

    # ------------------------------------------------------------------ shortcut check
    print("\n=== SHORTCUT CHECK: LR on duration, loudness, fps, width, height ONLY ===")
    def meta_X(split):
        rows = [c for c in clips if c["row"].split == split]
        X = np.array([[c["meta"]["duration"], c["meta"]["loudness_db"], c["meta"]["fps"], c["meta"]["width"],
                       c["meta"]["height"]] for c in rows], float)
        return rows, X
    rtr, Xm_tr = meta_X("train")
    rva, Xm_va = meta_X("val")
    shortcut = {}
    for name, key in (("video_label", "video_label"), ("audio_label", "audio_label")):
        ytr_m = np.array([getattr(c["row"], key) for c in rtr])
        yva_m = np.array([getattr(c["row"], key) for c in rva])
        ok_tr, ok_va = ~np.isnan(Xm_tr).any(1), ~np.isnan(Xm_va).any(1)
        sc_m = StandardScaler().fit(Xm_tr[ok_tr])
        lr_m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc_m.transform(Xm_tr[ok_tr]), ytr_m[ok_tr])
        a = safe_auc(yva_m[ok_va], lr_m.predict_proba(sc_m.transform(Xm_va[ok_va]))[:, 1])
        shortcut[name] = a
        print(f"  target={name}: val AUC = {a:.3f}   (0.5 = no shortcut; ~1.0 = metadata alone separates classes)")
    results["shortcut_check_val_auc"] = shortcut

    # ------------------------------------------------------------------ fusion weights / flags
    meta = {
        "visual": {"val_auc": results["visual"]["auc"], "low_confidence": results["visual"]["low_confidence"],
                   "fusion_weight": config.LOW_CONF_WEIGHT if results["visual"]["low_confidence"] else 1.0},
        "audio": {"val_auc": results["audio"]["auc"], "low_confidence": results["audio"]["low_confidence"],
                  "layer": results["audio"]["layer"],
                  "fusion_weight": config.LOW_CONF_WEIGHT if results["audio"]["low_confidence"] else 1.0},
        "note": NOTE,
    }
    (config.MODELS_DIR / "head_meta.json").write_text(json.dumps(meta, indent=2))
    config.OUT_DIR.mkdir(exist_ok=True)
    (config.OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    sub = pd.read_csv(args.manifest).set_index("clip_id")
    val_pred = val_pred.join(sub[["rel_path", "category", "split", "video_label", "audio_label"]])
    val_pred[val_pred.split == "val"].to_csv(config.OUT_DIR / "val_predictions.csv")

    print("\n================ RESULTS (" + NOTE + ") ================")
    for k in ("visual", "audio"):
        r = results[k]
        print(f"\n[{k.upper()}] {NOTE}")
        print(f"  confusion matrix @0.5 [[TN,FP],[FN,TP]] = {r['confusion_matrix_[[TN,FP],[FN,TP]]']}")
        print(f"  val clips={r['n_clips']} (positives {r['n_pos']})  clip-level ROC-AUC={r['auc']:.3f}  "
              f"precision={r['precision']:.3f} recall={r['recall']:.3f} F1={r['f1']:.3f}  @0.5")
        print("  per-category AUC (category vs all opposite-label val clips):",
              {c.replace('Video', 'V').replace('Audio', 'A'): round(v, 3) for c, v in r["per_category_auc"].items()})
        print(f"  real val clips by band (held-out scores): {r['real_val_clips_by_band']}")
        print(f"  grouped 5-fold OOF on TRAIN: clip AUC={r['train_grouped_oof_clip_auc']:.3f}; real train clips by band: "
              f"{r['real_train_oof_clips_by_band']}")
        if r["low_confidence"]:
            print(f"  !! val AUC {r['auc']:.3f} < {config.LOW_CONF_AUC}: fusion weight x{config.LOW_CONF_WEIGHT}; "
                  f"UI shows 'low-confidence {'visual' if k=='visual' else 'voice'} analysis'")
    print("\nsaved models/visual_lr.pkl, visual_scaler.pkl, audio_lr.pkl, audio_scaler.pkl, head_meta.json; "
          "outputs/metrics.json, outputs/val_predictions.csv")
    print("Get-Date end:", time.strftime("%F %T"))


if __name__ == "__main__":
    main()
