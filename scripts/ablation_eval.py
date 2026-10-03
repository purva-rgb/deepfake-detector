"""Ablation: does adding handcrafted acoustic / facial-temporal features improve the existing VISUAL (video_label)
clip-level detector on the exact same 160 validation clips?

Protocol (no validation leakage, nothing overwritten):
  * Frozen Xception + the existing visual head (models/visual_*.pkl, read only) give the baseline clip score.
  * Stage 2 = small logistic regression on [logit(visual clip score), handcrafted features].
      - TRAIN visual scores are grouped 5-fold out-of-fold (same recipe/C as train_heads.py) so the stacker never
        sees in-sample head scores; VAL visual scores come from the final existing head.
      - Handcrafted features: median-imputed with TRAIN medians (+1 availability indicator per group), standardised
        on TRAIN. Stage-2 C is chosen by GroupKFold on TRAIN only. Validation labels are used only for reporting.
  * Same 160 val clips, same metrics (ROC-AUC, precision/recall/F1 @ 0.5) as outputs/metrics.json.
Writes only to outputs/ablation/.
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.train_heads import clip_scores, fit, load_clips, visual_rows
from src import acoustic_features, config, face_temporal

OUT = config.OUT_DIR / "ablation"
MANIFEST = config.OUT_DIR / "manifests" / "dev_800.csv"
AC_COLS = ["ac_" + k for k in acoustic_features.FEATURE_NAMES]
FT_COLS = ["ft_" + k for k in face_temporal.STAT_NAMES]
CS2 = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def metrics(y, s):
    pred = (s >= 0.5).astype(int)
    return {"auc": float(roc_auc_score(y, s)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0))}


def design(df, cols, med, ind_cols):
    X = df[cols].copy()
    for c in cols:
        X[c] = X[c].fillna(med[c])
    parts = [X.values]
    for ic in ind_cols:
        parts.append(df[[ic]].values.astype(float))
    return np.hstack(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Get-Date start:", time.strftime("%F %T"), flush=True)
    man = pd.read_csv(MANIFEST).set_index("clip_id")
    hc = pd.read_csv(OUT / "handcrafted.csv").set_index("clip_id")
    assert set(man.index) == set(hc.index), "handcrafted features missing for some clips"
    clips = load_clips(str(MANIFEST))
    print(f"clips with cached Xception embeddings: {len(clips)} / {len(man)}")

    # ---------------- baseline visual clip scores (existing head, read only) + train OOF
    sc0, lr0 = joblib.load(config.MODELS_DIR / "visual_scaler.pkl"), joblib.load(config.MODELS_DIR / "visual_lr.pkl")
    C0 = float(json.loads((config.OUT_DIR / "metrics.json").read_text())["visual"]["C"])
    Xva, yva_rows, wva, cva, wiva = visual_rows(clips, "val")
    val_s = clip_scores(lr0.predict_proba(sc0.transform(Xva))[:, 1], cva, wiva, True)
    Xtr, ytr, wtr, ctr, wintr = visual_rows(clips, "train")
    groups_row = np.array([man.loc[c, "group"] for c in ctr])
    oof = np.full(len(ytr), np.nan)
    for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups_row):
        s2, l2 = fit(Xtr[a], ytr[a], wtr[a], C0)
        oof[b] = l2.predict_proba(s2.transform(Xtr[b]))[:, 1]
    tr_s = clip_scores(oof, ctr, wintr, True)

    tr_ids, va_ids = list(tr_s.index), list(val_s.index)
    assert set(va_ids) == set(man.index[man.split == "val"]), "val set differs from the 160 validation clips"
    ytr_c = man.loc[tr_ids, "video_label"].values
    yva_c = man.loc[va_ids, "video_label"].values
    base = metrics(yva_c, val_s.values)
    print(f"reproduced baseline on {len(va_ids)} val clips: AUC={base['auc']:.4f} P={base['precision']:.3f} "
          f"R={base['recall']:.3f} F1={base['f1']:.3f}  (recorded: 0.8995 / 0.767 / 0.863 / 0.812)")
    print(f"train clips for stage 2: {len(tr_ids)}; train OOF visual AUC={roc_auc_score(ytr_c, tr_s.values):.3f}")

    htr, hva = hc.loc[tr_ids], hc.loc[va_ids]
    for name, df in (("train", htr), ("val", hva)):
        print(f"  {name}: acoustic available {int((df.acoustic_windows_ok > 0).sum())}/{len(df)}, "
              f"facial-temporal available {int((df.face_windows_ok > 0).sum())}/{len(df)}")
    htr = htr.assign(ac_ok=(htr.acoustic_windows_ok > 0).astype(float), ft_ok=(htr.face_windows_ok > 0).astype(float))
    hva = hva.assign(ac_ok=(hva.acoustic_windows_ok > 0).astype(float), ft_ok=(hva.face_windows_ok > 0).astype(float))
    gtr_c = man.loc[tr_ids, "group"].values

    configs = {
        "control (visual score only, re-calibrated)": ([], []),
        "+ Acoustic": (AC_COLS, ["ac_ok"]),
        "+ Facial temporal": (FT_COLS, ["ft_ok"]),
        "+ Both": (AC_COLS + FT_COLS, ["ac_ok", "ft_ok"]),
    }
    results = {"Existing baseline": base}
    val_pred = pd.DataFrame({"clip_id": va_ids, "video_label": yva_c, "baseline": val_s.values}).set_index("clip_id")
    detail = {}
    for name, (cols, inds) in configs.items():
        med = htr[cols].median() if cols else None
        keep = [c for c in cols if np.isfinite(med[c])]  # drop columns that are entirely NaN on train
        inds_used = inds
        def X_of(df, s):
            F = design(df, keep, med, inds_used) if keep else df[[]].values.reshape(len(df), 0)
            return np.hstack([logit(s).reshape(-1, 1), F])
        Xa, Xb = X_of(htr, tr_s.values), X_of(hva, val_s.values)
        # choose C by grouped CV on TRAIN only
        best = None
        for C in CS2:
            o = np.zeros(len(ytr_c))
            for a, b in GroupKFold(n_splits=5).split(Xa, ytr_c, gtr_c):
                sc_ = StandardScaler().fit(Xa[a])
                m_ = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc_.transform(Xa[a]), ytr_c[a])
                o[b] = m_.predict_proba(sc_.transform(Xa[b]))[:, 1]
            auc = roc_auc_score(ytr_c, o)
            if best is None or auc > best[0] + 1e-9:
                best = (auc, C)
        sc_ = StandardScaler().fit(Xa)
        m_ = LogisticRegression(C=best[1], class_weight="balanced", max_iter=5000).fit(sc_.transform(Xa), ytr_c)
        s = m_.predict_proba(sc_.transform(Xb))[:, 1]
        assert np.isfinite(s).all()
        results[name] = metrics(yva_c, s)
        val_pred[name] = s
        detail[name] = {"C": best[1], "train_grouped_cv_auc": float(best[0]), "n_features": int(Xa.shape[1] - 1),
                        "dropped_all_nan_cols": [c for c in cols if c not in keep]}
        print(f"  {name}: stage-2 C={best[1]} train-grouped-CV AUC={best[0]:.3f} features={Xa.shape[1]-1}", flush=True)

    # paired bootstrap (clips resampled) of val AUC difference vs baseline, for context only
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(yva_c), len(yva_c)) for _ in range(2000)]
    boot = {}
    for name in configs:
        d = []
        for i in idx:
            if len(set(yva_c[i])) < 2:
                continue
            d.append(roc_auc_score(yva_c[i], val_pred[name].values[i]) - roc_auc_score(yva_c[i], val_pred["baseline"].values[i]))
        boot[name] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    (OUT / "ablation_results.json").write_text(json.dumps(
        {"n_val": len(va_ids), "n_train": len(tr_ids), "metrics": results, "stage2": detail,
         "delta_auc_vs_baseline_bootstrap95": boot,
         "note": "val-only; threshold 0.5; stage-2 stacker on frozen Xception visual-head score + handcrafted features"},
        indent=2))
    val_pred.to_csv(OUT / "ablation_val_predictions.csv")

    print("\n| Configuration | AUC | Precision | Recall | F1 | dAUC vs baseline (95% bootstrap CI) |")
    print("|---|---:|---:|---:|---:|---|")
    print(f"| Existing baseline | {base['auc']:.3f} | {base['precision']:.3f} | {base['recall']:.3f} | {base['f1']:.3f} | - |")
    for name in configs:
        r = results[name]
        lo, hi = boot[name]
        print(f"| {name} | {r['auc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
              f"{r['auc']-base['auc']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    print("Get-Date end:", time.strftime("%F %T"))


if __name__ == "__main__":
    main()
