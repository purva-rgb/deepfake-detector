"""Compare the 160-clip heads (outputs/metrics_160.json) with the 800-clip heads (outputs/metrics.json).
Also scores the NEW heads on the ORIGINAL 32 val clips only (like-for-like), using outputs/val_predictions.csv."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config

old = json.loads((config.OUT_DIR / "metrics_160.json").read_text())
new = json.loads((config.OUT_DIR / "metrics.json").read_text())
cm = "confusion_matrix_[[TN,FP],[FN,TP]]"
print(f"{'':28s}{'160-clip model':>16s}{'800-clip model':>16s}{'delta':>9s}")
for head in ("visual", "audio"):
    print(f"[{head}]  (each on ITS OWN val split: {old[head]['n_clips']} vs {new[head]['n_clips']} clips)")
    for k in ("auc", "precision", "recall", "f1"):
        print(f"  {k:26s}{old[head][k]:16.3f}{new[head][k]:16.3f}{new[head][k]-old[head][k]:+9.3f}")
    print(f"  {'confusion [[TN,FP],[FN,TP]]':26s}{str(old[head].get(cm, 'n/a')):>16s}{str(new[head][cm]):>16s}")
    print(f"  {'train grouped OOF AUC':26s}{old[head]['train_grouped_oof_clip_auc']:16.3f}{new[head]['train_grouped_oof_clip_auc']:16.3f}"
          f"{new[head]['train_grouped_oof_clip_auc']-old[head]['train_grouped_oof_clip_auc']:+9.3f}")
    print(f"  chosen: old {({k: old[head][k] for k in ('C', 'layer') if k in old[head]})}  new {({k: new[head][k] for k in ('C', 'layer') if k in new[head]})}")
print("\nshortcut check val AUC (160 -> 800):", old["shortcut_check_val_auc"], "->", new["shortcut_check_val_auc"])

vp = pd.read_csv(config.OUT_DIR / "val_predictions.csv")
orig = pd.read_csv(config.SUBSET_CSV)
sub = vp[vp.clip_id.isin(orig.clip_id)]
print(f"\nNEW heads on the ORIGINAL {len(sub)} val clips only (these clips were never in training either time):")
print(f"  visual AUC {roc_auc_score(sub.video_label, sub.visual_clip_score):.3f}   (old model: {old['visual']['auc']:.3f})")
print(f"  audio  AUC {roc_auc_score(sub.audio_label, sub.audio_clip_score):.3f}   (old model: {old['audio']['auc']:.3f})")
print("NOTE: with 32 clips one flipped pair moves AUC by ~0.016 for each class - treat differences as indicative only.")
